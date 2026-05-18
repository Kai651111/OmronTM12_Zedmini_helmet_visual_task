"""
vision_zed.py

ZED Mini vision module.
ZED Mini 视觉模块。

Main features / 主要功能:
    1. Open ZED camera and enable positional tracking.
       打开 ZED 相机并启用 positional tracking。

    2. Show real-time preview before capture.
       正式采集前实时显示画面，方便调整相机位置。

    3. Detect green defect / sticker.
       检测绿色 defect / sticker。

    4. Estimate defect position in current camera frame: P_camera.
       估计 defect 在当前相机坐标系下的位置：P_camera。

    5. Record camera pose relative to P1 reference pose.
       记录相机相对于 P1 参考位姿的位姿。

    6. Transform defect point into P1 reference frame: P_P1.
       将 defect 点转换到 P1 参考坐标系：P_P1。

    7. Record optional IMU acceleration and angular velocity.
       记录可选 IMU 加速度和角速度。

    8. Use multi-frame robust statistics.
       使用多帧鲁棒统计方法。
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np
import pyzed.sl as sl
import config as cfg


# ============================================================
# 1. Configuration / 参数配置
# ============================================================

ZED_RESOLUTION = cfg.ZED_RESOLUTION
ZED_FPS = cfg.ZED_FPS
ZED_DEPTH_MODE = cfg.ZED_DEPTH_MODE
ZED_UNIT = cfg.ZED_UNIT

CAPTURE_SECONDS_PER_VIEW = cfg.CAPTURE_SECONDS_PER_VIEW

LOWER_GREEN = cfg.LOWER_GREEN
UPPER_GREEN = cfg.UPPER_GREEN

MIN_GREEN_AREA = cfg.MIN_GREEN_AREA
MAX_GREEN_AREA = cfg.MAX_GREEN_AREA
MIN_CIRCULARITY = cfg.MIN_CIRCULARITY

ROI_REL = cfg.ROI_REL

MAX_POINT_SAMPLES = cfg.MAX_POINT_SAMPLES
Z_MIN_M = cfg.Z_MIN_M
Z_MAX_M = cfg.Z_MAX_M

MIN_VALID_FRAMES_PER_VIEW = cfg.MIN_VALID_FRAMES_PER_VIEW
MAX_CENTROID_STD_PX = cfg.MAX_CENTROID_STD_PX
MAX_AREA_REL_STD = cfg.MAX_AREA_REL_STD

USE_DEPTH_QUALITY_CHECK = cfg.USE_DEPTH_QUALITY_CHECK
MAX_DEPTH_MAD_MM = cfg.MAX_DEPTH_MAD_MM

# ============================================================
# 2. Data structures / 数据结构
# ============================================================

@dataclass
class DefectDetectionResult:
    """
    Single-frame green defect detection result.
    单帧绿色 defect 检测结果。
    """
    found: bool
    mask: np.ndarray
    contour: Optional[np.ndarray]
    centroid_px: Optional[Tuple[int, int]]
    point_camera_m: Optional[np.ndarray]
    area: float = 0.0
    circularity: float = 0.0
    valid_3d_points: int = 0


@dataclass
class FrameRecord:
    """
    Per-frame record.
    每一帧的数据记录。
    """
    point_id: int
    frame_id: int
    timestamp: float

    found: bool
    cx: Optional[int]
    cy: Optional[int]
    area: float
    circularity: float

    # Defect in current camera frame.
    # defect 在当前相机坐标系下的位置。
    Xc: Optional[float]
    Yc: Optional[float]
    Zc: Optional[float]

    # Defect in P1 reference frame.
    # defect 在 P1 参考坐标系下的位置。
    Xr: Optional[float]
    Yr: Optional[float]
    Zr: Optional[float]

    # Current camera position relative to P1.
    # 当前相机相对于 P1 的位置。
    camera_ref_x: Optional[float]
    camera_ref_y: Optional[float]
    camera_ref_z: Optional[float]

    # Tracking state.
    # 跟踪状态。
    tracking_ok: bool
    tracking_state: str

    # Optional IMU data.
    # 可选 IMU 数据。
    imu_acc_x: Optional[float]
    imu_acc_y: Optional[float]
    imu_acc_z: Optional[float]
    imu_gyro_x: Optional[float]
    imu_gyro_y: Optional[float]
    imu_gyro_z: Optional[float]

    valid_3d_points: int


@dataclass
class CameraEstimate:
    """
    Multi-frame camera estimate.
    多帧统计后的相机估计结果。
    """
    success: bool
    reason: str

    valid_frame_count: int
    total_frame_count: int

    # Final robust estimate in current camera frame.
    # 在相机坐标系下的最终鲁棒估计。
    point_camera_m: Optional[np.ndarray]

    # Final robust estimate in P1 reference frame.
    # 在 P1 参考坐标系下的最终鲁棒估计。
    point_reference_m: Optional[np.ndarray]

    centroid_std_px: Optional[float]
    depth_mad_mm: Optional[float]
    area_median: Optional[float]
    area_rel_std: Optional[float]
    circularity_median: Optional[float]
    sigma_camera_mm: Optional[float]
    sigma_reference_mm: Optional[float]
    camera_motion_std_mm: Optional[float]

    records: List[FrameRecord]


# ============================================================
# 3. Math helpers / 数学工具
# ============================================================

def quaternion_xyzw_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [x, y, z, w] to rotation matrix.
    将四元数 [x, y, z, w] 转换为旋转矩阵。
    """
    x, y, z, w = q
    n = np.linalg.norm(q)

    if n < 1e-12:
        return np.eye(3, dtype=np.float64)

    x, y, z, w = q / n

    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),         2 * (x * z + y * w)],
        [2 * (x * y + z * w),         1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [2 * (x * z - y * w),         2 * (y * z + x * w),         1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def make_transform_from_t_q(t_m: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    """
    Build 4x4 transform from translation and quaternion.
    由平移向量和四元数构造 4x4 齐次变换矩阵。
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_xyzw_to_rotation_matrix(q_xyzw)
    T[:3, 3] = t_m
    return T


def transform_point(T: np.ndarray, point_m: np.ndarray) -> np.ndarray:
    """
    Transform 3D point using 4x4 matrix.
    使用 4x4 变换矩阵变换 3D 点。
    """
    p_h = np.array([point_m[0], point_m[1], point_m[2], 1.0], dtype=np.float64)
    out_h = T @ p_h
    return out_h[:3]


def median_absolute_deviation(values: np.ndarray) -> float:
    """
    Median absolute deviation.
    中位数绝对偏差。
    """
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def robust_point_estimate(points: np.ndarray):
    """
    Robustly estimate one point from many points.
    从多个点中鲁棒估计一个点。

    Method / 方法:
        median -> remove farthest 20% -> mean of inliers
        中位数 -> 去掉最远的 20% -> 对内点求平均
    """
    median = np.median(points, axis=0)
    dist = np.linalg.norm(points - median, axis=1)

    threshold = np.percentile(dist, 80)
    inliers = points[dist <= threshold]

    if len(inliers) == 0:
        return median, points

    return np.mean(inliers, axis=0), inliers


# ============================================================
# 4. ZED camera / ZED 相机
# ============================================================

class ZedCamera:
    """
    ZED camera wrapper with positional tracking.
    带 positional tracking 的 ZED 相机封装。
    """

    def __init__(self):
        self.zed = sl.Camera()
        self.runtime = sl.RuntimeParameters()
        self.image_zed = sl.Mat()
        self.point_cloud = sl.Mat()

        self.pose = sl.Pose()
        self.translation = sl.Translation()
        self.orientation = sl.Orientation()
        self.sensors_data = sl.SensorsData()

        self.tracking_enabled = False

        # T_reference_world = inverse(T_world_camera_at_P1)
        # 用于把当前相机位姿转到 P1 参考坐标系。
        self.T_reference_world = None

    def open(self):
        """
        Open ZED and enable positional tracking.
        打开 ZED 并启用 positional tracking。
        """
        init = sl.InitParameters()
        init.camera_resolution = ZED_RESOLUTION
        init.camera_fps = ZED_FPS
        init.depth_mode = ZED_DEPTH_MODE
        init.coordinate_units = ZED_UNIT

        status = self.zed.open(init)

        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"[ZED] Opening error / 打开失败: {status}")

        print("[ZED] Camera opened. / 相机已打开。")

        try:
            tracking_params = sl.PositionalTrackingParameters()

            # Some SDK versions support this attribute.
            # 部分 SDK 版本支持这个属性。
            if hasattr(tracking_params, "enable_imu_fusion"):
                tracking_params.enable_imu_fusion = True

            tracking_status = self.zed.enable_positional_tracking(tracking_params)

            if tracking_status == sl.ERROR_CODE.SUCCESS:
                self.tracking_enabled = True
                print("[ZED] Positional tracking enabled. / 已启用位置跟踪。")
            else:
                self.tracking_enabled = False
                print(f"[ZED] Positional tracking failed. / 位置跟踪启用失败: {tracking_status}")

        except Exception as e:
            self.tracking_enabled = False
            print(f"[ZED] Positional tracking exception. / 位置跟踪异常: {e}")

    def grab(self) -> Optional[Tuple[np.ndarray, sl.Mat]]:
        """
        Grab one frame.
        获取一帧图像和点云。
        """
        status = self.zed.grab(self.runtime)

        if status != sl.ERROR_CODE.SUCCESS:
            return None

        self.zed.retrieve_image(self.image_zed, sl.VIEW.LEFT)
        self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)

        image = self.image_zed.get_data()

        if image.shape[2] == 4:
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            bgr = image

        return bgr, self.point_cloud

    def get_camera_pose_world(self):
        """
        Get current camera pose in ZED world frame.
        获取当前相机在 ZED world frame 下的位姿。

        Returns / 返回:
            ok, tracking_state, T_world_camera, camera_position_world_m
        """
        if not self.tracking_enabled:
            return False, "TRACKING_DISABLED / 跟踪未启用", None, None

        try:
            tracking_state = self.zed.get_position(self.pose, sl.REFERENCE_FRAME.WORLD)
            tracking_state_str = str(tracking_state)

            if tracking_state != sl.POSITIONAL_TRACKING_STATE.OK:
                return False, tracking_state_str, None, None

            self.pose.get_translation(self.translation)
            self.pose.get_orientation(self.orientation)

            t = np.array(self.translation.get(), dtype=np.float64)
            q = np.array(self.orientation.get(), dtype=np.float64)

            T_world_camera = make_transform_from_t_q(t, q)

            return True, tracking_state_str, T_world_camera, t

        except Exception as e:
            return False, f"TRACKING_EXCEPTION / 跟踪异常: {e}", None, None

    def set_reference_from_current_pose(self, max_attempts: int = 60) -> bool:
        """
        Set P1 reference from current camera pose.
        用当前相机位姿设置 P1 参考坐标系。
        """
        print("[ZED] Setting P1 reference pose... / 正在设置 P1 参考位姿...")

        for _ in range(max_attempts):
            frame = self.grab()

            if frame is None:
                continue

            ok, state, T_world_camera, _ = self.get_camera_pose_world()

            if ok and T_world_camera is not None:
                self.T_reference_world = np.linalg.inv(T_world_camera)
                print("[ZED] P1 reference pose set. / P1 参考位姿已设置。")
                return True

            time.sleep(0.03)

        print("[ZED] Failed to set P1 reference pose. / P1 参考位姿设置失败。")
        return False

    def get_camera_pose_reference(self):
        """
        Get current camera pose relative to P1 reference frame.
        获取当前相机相对于 P1 参考坐标系的位姿。

        Returns / 返回:
            ok, tracking_state, T_reference_camera, camera_position_reference_m
        """
        if self.T_reference_world is None:
            return False, "REFERENCE_NOT_SET / P1参考未设置", None, None

        ok, state, T_world_camera, _ = self.get_camera_pose_world()

        if not ok or T_world_camera is None:
            return False, state, None, None

        T_reference_camera = self.T_reference_world @ T_world_camera
        camera_position_reference_m = T_reference_camera[:3, 3]

        return True, state, T_reference_camera, camera_position_reference_m

    def get_imu_data(self):
        """
        Get IMU data if available.
        如果可用，读取 IMU 数据。

        Note / 注意:
            IMU is recorded for analysis.
            IMU 只用于记录分析，不直接用于位置积分。
        """
        try:
            status = self.zed.get_sensors_data(self.sensors_data, sl.TIME_REFERENCE.IMAGE)

            if status != sl.ERROR_CODE.SUCCESS:
                return None, None

            imu = self.sensors_data.get_imu_data()

            acc = np.array(imu.get_linear_acceleration(), dtype=np.float64)
            gyro = np.array(imu.get_angular_velocity(), dtype=np.float64)

            return acc, gyro

        except Exception:
            return None, None

    def close(self):
        """
        Close ZED.
        关闭 ZED。
        """
        try:
            if self.tracking_enabled:
                self.zed.disable_positional_tracking()
        except Exception:
            pass

        self.zed.close()
        print("[ZED] Camera closed. / 相机已关闭。")


# ============================================================
# 5. Green defect detector / 绿色 defect 检测器
# ============================================================

class GreenDefectDetector:
    """
    Green defect / sticker detector.
    绿色 defect / sticker 检测器。
    """

    @staticmethod
    def apply_roi(mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape[:2]

        x_min_rel, y_min_rel, x_max_rel, y_max_rel = ROI_REL

        x_min = int(x_min_rel * w)
        x_max = int(x_max_rel * w)
        y_min = int(y_min_rel * h)
        y_max = int(y_max_rel * h)

        roi_mask = np.zeros_like(mask)
        roi_mask[y_min:y_max, x_min:x_max] = 255

        return cv2.bitwise_and(mask, roi_mask)

    @staticmethod
    def contour_circularity(contour: np.ndarray) -> float:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 1e-6:
            return 0.0

        return float(4.0 * np.pi * area / (perimeter * perimeter))

    @staticmethod
    def contour_centroid(contour: np.ndarray) -> Optional[Tuple[int, int]]:
        M = cv2.moments(contour)

        if abs(M["m00"]) < 1e-6:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return cx, cy

    @staticmethod
    def extract_3d_points_from_mask(target_mask: np.ndarray, point_cloud: sl.Mat):
        ys, xs = np.where(target_mask > 0)

        if len(xs) == 0:
            return np.empty((0, 3), dtype=np.float32), 0

        if len(xs) > MAX_POINT_SAMPLES:
            idx = np.linspace(0, len(xs) - 1, MAX_POINT_SAMPLES).astype(np.int32)
            xs = xs[idx]
            ys = ys[idx]

        points = []

        for x, y in zip(xs, ys):
            err, point = point_cloud.get_value(int(x), int(y))

            if err != sl.ERROR_CODE.SUCCESS:
                continue

            X, Y, Z, _ = point

            if not (np.isfinite(X) and np.isfinite(Y) and np.isfinite(Z)):
                continue

            if Z < Z_MIN_M or Z > Z_MAX_M:
                continue

            points.append([X, Y, Z])

        if len(points) == 0:
            return np.empty((0, 3), dtype=np.float32), 0

        return np.array(points, dtype=np.float32), len(points)

    @staticmethod
    def robust_3d_estimate(points_3d: np.ndarray):
        if points_3d.shape[0] == 0:
            return None

        median = np.median(points_3d, axis=0)
        dist = np.linalg.norm(points_3d - median, axis=1)

        threshold = np.percentile(dist, 80)
        inliers = points_3d[dist <= threshold]

        if len(inliers) == 0:
            return median.astype(np.float64)

        return np.mean(inliers, axis=0).astype(np.float64)

    def detect(self, bgr_image: np.ndarray, point_cloud: sl.Mat) -> DefectDetectionResult:
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
        mask = self.apply_roi(mask)

        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return DefectDetectionResult(
                found=False,
                mask=mask,
                contour=None,
                centroid_px=None,
                point_camera_m=None,
            )

        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < MIN_GREEN_AREA or area > MAX_GREEN_AREA:
                continue

            circularity = self.contour_circularity(contour)

            if circularity < MIN_CIRCULARITY:
                continue

            candidates.append((contour, area, circularity))

        if not candidates:
            return DefectDetectionResult(
                found=False,
                mask=mask,
                contour=None,
                centroid_px=None,
                point_camera_m=None,
            )

        contour, area, circularity = max(candidates, key=lambda item: item[1])
        centroid = self.contour_centroid(contour)

        if centroid is None:
            return DefectDetectionResult(
                found=False,
                mask=mask,
                contour=None,
                centroid_px=None,
                point_camera_m=None,
            )

        target_mask = np.zeros_like(mask)
        cv2.drawContours(target_mask, [contour], -1, 255, thickness=-1)

        points_3d, valid_count = self.extract_3d_points_from_mask(target_mask, point_cloud)
        point_camera_m = self.robust_3d_estimate(points_3d)

        if point_camera_m is None:
            return DefectDetectionResult(
                found=False,
                mask=target_mask,
                contour=contour,
                centroid_px=centroid,
                point_camera_m=None,
                area=area,
                circularity=circularity,
                valid_3d_points=valid_count,
            )

        return DefectDetectionResult(
            found=True,
            mask=target_mask,
            contour=contour,
            centroid_px=centroid,
            point_camera_m=point_camera_m,
            area=area,
            circularity=circularity,
            valid_3d_points=valid_count,
        )


# ============================================================
# 6. Preview function / 实时预览函数
# ============================================================

# 安全关闭窗口函数
def safe_destroy_window(window_name: str):
    """
    Safely close one OpenCV window.
    安全关闭一个 OpenCV 窗口。
    """
    try:
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)
    except cv2.error:
        pass

def preview_camera_until_confirm(
    zed: ZedCamera,
    detector: Optional[GreenDefectDetector] = None,
    point_id: int = 0,
    title: str = "Live Preview / 实时预览",
    show_mask: bool = True,
):
    """
    Show live ZED preview until user confirms.
    实时显示 ZED 画面，直到用户确认继续。
    """
    print("\n[PREVIEW] Live preview started. / 实时预览已开始。")
    print("[PREVIEW] Adjust camera position while watching the window. / 请看着窗口调整相机位置。")
    print("[PREVIEW] Press Enter / c / Space to continue. / 按 Enter / c / 空格继续。")
    print("[PREVIEW] Press q / ESC to abort. / 按 q / ESC 中断。")

    try:
        while True:
            frame = zed.grab()

            if frame is None:
                continue

            bgr_image, point_cloud = frame
            preview = bgr_image.copy()

            result = None

            if detector is not None:
                result = detector.detect(bgr_image, point_cloud)

                if result.contour is not None:
                    cv2.drawContours(preview, [result.contour], -1, (0, 255, 0), 2)

                if result.centroid_px is not None:
                    cv2.circle(preview, result.centroid_px, 6, (0, 0, 255), -1)

                if result.found and result.point_camera_m is not None:
                    Xc, Yc, Zc = result.point_camera_m

                    cv2.putText(
                        preview,
                        f"P_camera / Camera coord = ({Xc:.3f}, {Yc:.3f}, {Zc:.3f}) m",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        preview,
                        f"area={result.area:.0f}, circ={result.circularity:.2f}, valid3D={result.valid_3d_points}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            if zed.T_reference_world is not None:
                tracking_ok, tracking_state, _, camera_pos = zed.get_camera_pose_reference()
                camera_pose_label = "Camera wrt P1 / 相机相对P1"
            else:
                tracking_ok, tracking_state, _, camera_pos = zed.get_camera_pose_world()
                camera_pose_label = "Camera world / 相机world位置"

            cv2.putText(
                preview,
                f"Point {point_id} | Live preview / 实时预览",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                preview,
                f"Tracking / 跟踪: {tracking_ok} | {tracking_state}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if camera_pos is not None:
                cv2.putText(
                    preview,
                    (
                        f"{camera_pose_label} = "
                        f"({camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}) m"
                    ),
                    (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                preview,
                "Enter / c / Space: continue | q / ESC: abort",
                (20, preview.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(title, preview)

            if show_mask and result is not None:
                cv2.imshow("Green Mask Preview / 绿色掩膜预览", result.mask)

            key = cv2.waitKey(1) & 0xFF

            if key in [13, 10, ord("c"), ord(" ")]:
                print("[PREVIEW] Confirmed. / 已确认，继续下一步。")
                break

            if key == 27 or key == ord("q"):
                raise KeyboardInterrupt

    finally:
        safe_destroy_window(title)

        if show_mask:
            safe_destroy_window("Green Mask Preview / 绿色掩膜预览")


# ============================================================
# 7. Multi-frame analysis / 多帧统计分析
# ============================================================

def analyze_frame_records(records: List[FrameRecord]) -> CameraEstimate:
    """
    Analyze per-frame records and return robust estimate.
    分析每帧记录，并返回鲁棒估计。
    """
    valid_camera_records = [
        r for r in records
        if r.found and r.Xc is not None and r.Yc is not None and r.Zc is not None
    ]

    valid_reference_records = [
        r for r in records
        if r.found and r.Xr is not None and r.Yr is not None and r.Zr is not None
    ]

    valid_count = len(valid_camera_records)
    total_count = len(records)

    if valid_count < MIN_VALID_FRAMES_PER_VIEW:
        return CameraEstimate(
            success=False,
            reason=f"Not enough valid frames / 有效帧不足: {valid_count}/{total_count}",
            valid_frame_count=valid_count,
            total_frame_count=total_count,
            point_camera_m=None,
            point_reference_m=None,
            centroid_std_px=None,
            depth_mad_mm=None,
            area_median=None,
            area_rel_std=None,
            circularity_median=None,
            sigma_camera_mm=None,
            sigma_reference_mm=None,
            camera_motion_std_mm=None,
            records=records,
        )

    points_camera = np.array(
        [[r.Xc, r.Yc, r.Zc] for r in valid_camera_records],
        dtype=np.float64,
    )

    centroids = np.array(
        [[r.cx, r.cy] for r in valid_camera_records],
        dtype=np.float64,
    )

    areas = np.array([r.area for r in valid_camera_records], dtype=np.float64)
    circularities = np.array([r.circularity for r in valid_camera_records], dtype=np.float64)

    point_camera_m, inliers_camera = robust_point_estimate(points_camera)

    centroid_std_px = float(np.mean(np.std(centroids, axis=0)))
    depth_mad_mm = median_absolute_deviation(points_camera[:, 2]) * 1000.0

    area_median = float(np.median(areas))
    area_rel_std = float(np.std(areas) / max(area_median, 1.0))

    circularity_median = float(np.median(circularities))
    sigma_camera_mm = float(np.mean(np.std(inliers_camera, axis=0)) * 1000.0)

    point_reference_m = None
    sigma_reference_mm = None

    if len(valid_reference_records) >= MIN_VALID_FRAMES_PER_VIEW:
        points_reference = np.array(
            [[r.Xr, r.Yr, r.Zr] for r in valid_reference_records],
            dtype=np.float64,
        )

        point_reference_m, inliers_reference = robust_point_estimate(points_reference)
        sigma_reference_mm = float(np.mean(np.std(inliers_reference, axis=0)) * 1000.0)

    camera_positions = np.array(
        [
            [r.camera_ref_x, r.camera_ref_y, r.camera_ref_z]
            for r in records
            if r.camera_ref_x is not None and r.camera_ref_y is not None and r.camera_ref_z is not None
        ],
        dtype=np.float64,
    )

    camera_motion_std_mm = None

    if len(camera_positions) >= 2:
        camera_motion_std_mm = float(np.mean(np.std(camera_positions, axis=0)) * 1000.0)

    if centroid_std_px > MAX_CENTROID_STD_PX:
        success = False
        reason = f"Centroid std too high / 2D 中心点抖动过大: {centroid_std_px:.2f}px"
    elif USE_DEPTH_QUALITY_CHECK and depth_mad_mm > MAX_DEPTH_MAD_MM:
        success = False
        reason = f"Depth MAD too high / 深度 MAD 过大: {depth_mad_mm:.1f}mm"
    elif area_rel_std > MAX_AREA_REL_STD:
        success = False
        reason = f"Area relative std too high / 面积相对波动过大: {area_rel_std:.2f}"
    else:
        success = True
        reason = "OK / 通过"

    return CameraEstimate(
        success=success,
        reason=reason,
        valid_frame_count=valid_count,
        total_frame_count=total_count,
        point_camera_m=point_camera_m,
        point_reference_m=point_reference_m,
        centroid_std_px=centroid_std_px,
        depth_mad_mm=depth_mad_mm,
        area_median=area_median,
        area_rel_std=area_rel_std,
        circularity_median=circularity_median,
        sigma_camera_mm=sigma_camera_mm,
        sigma_reference_mm=sigma_reference_mm,
        camera_motion_std_mm=camera_motion_std_mm,
        records=records,
    )


def capture_defect_position_for_seconds(
    zed: ZedCamera,
    detector: GreenDefectDetector,
    capture_seconds: float = CAPTURE_SECONDS_PER_VIEW,
    show_debug: bool = True,
    point_id: int = 0,
) -> CameraEstimate:
    """
    Capture ZED frames for several seconds and estimate defect position.
    采集若干秒 ZED 数据，并估计 defect 位置。

    Returns / 返回:
        CameraEstimate.point_camera_m:
            defect in current camera frame.
            defect 在相机坐标系下的位置。

        CameraEstimate.point_reference_m:
            defect in P1 reference frame.
            defect 在 P1 参考坐标系下的位置。
    """
    records = []
    start_time = time.time()
    frame_id = 0

    print(f"[VISION] Start capture / 开始采集: point {point_id}, {capture_seconds:.1f}s")

    while True:
        now = time.time()
        elapsed = now - start_time

        if elapsed >= capture_seconds:
            break

        frame = zed.grab()

        if frame is None:
            continue

        bgr_image, point_cloud = frame

        tracking_ok, tracking_state, T_ref_camera, camera_pos_ref = zed.get_camera_pose_reference()
        acc, gyro = zed.get_imu_data()

        result = detector.detect(bgr_image, point_cloud)

        Xc = Yc = Zc = None
        Xr = Yr = Zr = None
        cx = cy = None

        if result.found and result.point_camera_m is not None and result.centroid_px is not None:
            cx, cy = result.centroid_px
            Xc, Yc, Zc = [float(v) for v in result.point_camera_m]

            if tracking_ok and T_ref_camera is not None:
                p_ref = transform_point(T_ref_camera, result.point_camera_m)
                Xr, Yr, Zr = [float(v) for v in p_ref]

        record = FrameRecord(
            point_id=point_id,
            frame_id=frame_id,
            timestamp=now,

            found=bool(result.found),
            cx=cx,
            cy=cy,
            area=float(result.area),
            circularity=float(result.circularity),

            Xc=Xc,
            Yc=Yc,
            Zc=Zc,

            Xr=Xr,
            Yr=Yr,
            Zr=Zr,

            camera_ref_x=float(camera_pos_ref[0]) if camera_pos_ref is not None else None,
            camera_ref_y=float(camera_pos_ref[1]) if camera_pos_ref is not None else None,
            camera_ref_z=float(camera_pos_ref[2]) if camera_pos_ref is not None else None,

            tracking_ok=bool(tracking_ok),
            tracking_state=tracking_state,

            imu_acc_x=float(acc[0]) if acc is not None else None,
            imu_acc_y=float(acc[1]) if acc is not None else None,
            imu_acc_z=float(acc[2]) if acc is not None else None,
            imu_gyro_x=float(gyro[0]) if gyro is not None else None,
            imu_gyro_y=float(gyro[1]) if gyro is not None else None,
            imu_gyro_z=float(gyro[2]) if gyro is not None else None,

            valid_3d_points=int(result.valid_3d_points),
        )

        records.append(record)

        if show_debug:
            debug = bgr_image.copy()

            if result.contour is not None:
                cv2.drawContours(debug, [result.contour], -1, (0, 255, 0), 2)

            if result.centroid_px is not None:
                cv2.circle(debug, result.centroid_px, 6, (0, 0, 255), -1)

            lines = [
                f"Point {point_id} | Capture / 采集: {elapsed:.1f}/{capture_seconds:.1f}s",
                f"Tracking / 跟踪: {tracking_ok} | {tracking_state}",
            ]

            if result.found and result.point_camera_m is not None:
                lines.append(f"P_camera / 相机坐标 = ({Xc:.3f}, {Yc:.3f}, {Zc:.3f}) m")

            if Xr is not None:
                lines.append(f"P_P1 / P1参考坐标 = ({Xr:.3f}, {Yr:.3f}, {Zr:.3f}) m")

            if camera_pos_ref is not None:
                lines.append(
                    f"Camera wrt P1 / 相机相对P1 = "
                    f"({camera_pos_ref[0]:.3f}, {camera_pos_ref[1]:.3f}, {camera_pos_ref[2]:.3f}) m"
                )

            y = 30

            for line in lines:
                cv2.putText(
                    debug,
                    line,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y += 30

            cv2.imshow("ZED Defect Capture / ZED Defect 采集", debug)
          #  cv2.imshow("Green Mask / 绿色掩膜", result.mask) # 不想显示musk就给他注释掉

            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord("q"):
                raise KeyboardInterrupt

        frame_id += 1

    estimate = analyze_frame_records(records)

    print("[VISION] Capture estimate / 采集统计结果:")
    print(f"  success / 是否成功: {estimate.success}")
    print(f"  reason / 原因: {estimate.reason}")
    print(f"  valid frames / 有效帧: {estimate.valid_frame_count}/{estimate.total_frame_count}")

    if estimate.point_camera_m is not None:
        p = estimate.point_camera_m
        print(f"  P_camera / 相机坐标: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) m")

    if estimate.point_reference_m is not None:
        p = estimate.point_reference_m
        print(f"  P_P1 / P1参考坐标: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) m")

    print(f"  centroid_std_px / 中心点标准差: {estimate.centroid_std_px}")
    print(f"  depth_mad_mm / 深度MAD: {estimate.depth_mad_mm}")
    print(f"  area_rel_std / 面积相对标准差: {estimate.area_rel_std}")
    print(f"  sigma_camera_mm / 相机坐标sigma: {estimate.sigma_camera_mm}")
    print(f"  sigma_reference_mm / P1参考坐标sigma: {estimate.sigma_reference_mm}")
    print(f"  camera_motion_std_mm / 采集期间相机自身运动std: {estimate.camera_motion_std_mm}")

    return estimate