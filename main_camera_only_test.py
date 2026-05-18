"""
main_camera_only_test.py

Camera-only test without P1 reference compensation.
纯相机测试：不使用 P1 参考坐标，不使用 positional tracking 补偿。

Workflow / 流程:
    1. Open ZED.
       打开 ZED。

    2. For each test point:
       对每个测试点：

        a. Show live preview.
           实时预览画面，方便调整相机位置。

        b. Confirm with Enter / c / Space.
           按 Enter / c / 空格确认。

        c. Capture for several seconds.
           采集若干秒。

        d. Estimate P_camera only.
           只估计当前相机坐标系下的 P_camera。

    3. Summarize camera-frame points.
       汇总相机坐标系下的结果。

    4. Save CSV log.
       保存 CSV 日志。
"""

import csv
import os
from datetime import datetime

import cv2
import numpy as np

from vision_zed import (
    ZedCamera,
    GreenDefectDetector,
    CAPTURE_SECONDS_PER_VIEW,
    capture_defect_position_for_seconds,
    preview_camera_until_confirm,
)
import techman as tm



# ============================================================
# User settings / 用户设置
# ============================================================

NUM_TEST_POINTS = 5

# If True, capture_defect_position_for_seconds may show debug windows.
# 如果 True，采集阶段可能显示 debug 窗口。
#
# 如果你还没有删除 vision_zed.py 里的 Green Mask imshow，
# 建议这里先用 False，避免显示 mask 界面。
SHOW_DEBUG = True

SAVE_CSV_LOG = True
LOG_FOLDER = "camera_only_logs"


# ============================================================
# Helper functions / 辅助函数
# ============================================================

def bilingual(en: str, zh: str) -> str:
    """
    Build bilingual message.
    构造中英文双语消息。
    """
    return f"{en} / {zh}"


def create_log_folder():
    """
    Create log folder if it does not exist.
    如果日志文件夹不存在，则创建。
    """
    if not os.path.exists(LOG_FOLDER):
        os.makedirs(LOG_FOLDER)


def save_all_records(estimates, csv_path: str):
    """
    Save useful camera-only frame records to CSV.
    保存纯相机测试的逐帧记录到 CSV。
    """
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow([
            "point_id_测试点编号",
            "frame_id_帧编号",
            "timestamp_时间戳",

            "found_是否检测到",
            "cx_中心x",
            "cy_中心y",
            "area_面积",
            "circularity_圆度",

            "Xc_m_相机X",
            "Yc_m_相机Y",
            "Zc_m_相机Z",

            "valid_3d_points_有效3D点数量",
        ])

        for estimate in estimates:
            for r in estimate.records:
                writer.writerow([
                    r.point_id,
                    r.frame_id,
                    r.timestamp,

                    r.found,
                    r.cx,
                    r.cy,
                    r.area,
                    r.circularity,

                    r.Xc,
                    r.Yc,
                    r.Zc,

                    r.valid_3d_points,
                ])

    print(bilingual(f"CSV saved to: {csv_path}", f"CSV 已保存到: {csv_path}"))


def print_estimate(point_id: int, estimate):
    """
    Print estimate result for one test point.
    打印单个测试点统计结果。
    """
    print("\n" + "-" * 80)
    print(bilingual(f"Point {point_id} estimate", f"测试点 {point_id} 统计结果"))
    print("-" * 80)

    print(bilingual(f"success: {estimate.success}", f"是否成功: {estimate.success}"))
    print(bilingual(f"reason: {estimate.reason}", f"原因: {estimate.reason}"))

    print(
        bilingual(
            f"valid frames: {estimate.valid_frame_count}/{estimate.total_frame_count}",
            f"有效帧: {estimate.valid_frame_count}/{estimate.total_frame_count}",
        )
    )

    if estimate.point_camera_m is not None:
        p = estimate.point_camera_m
        d = float(np.linalg.norm(p))

        print(
            bilingual(
                f"P_camera = ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) m, distance={d:.4f} m",
                f"相机坐标 P_camera = ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) m，距离={d:.4f} m",
            )
        )

    print(
        bilingual(
            f"centroid_std_px: {estimate.centroid_std_px}",
            f"中心点标准差 px: {estimate.centroid_std_px}",
        )
    )

    print(
        bilingual(
            f"depth_mad_mm: {estimate.depth_mad_mm}",
            f"深度 MAD mm: {estimate.depth_mad_mm}",
        )
    )

    print(
        bilingual(
            f"area_rel_std: {estimate.area_rel_std}",
            f"面积相对标准差: {estimate.area_rel_std}",
        )
    )

    print(
        bilingual(
            f"sigma_camera_mm: {estimate.sigma_camera_mm}",
            f"相机坐标 sigma mm: {estimate.sigma_camera_mm}",
        )
    )


def summarize_estimates(estimates):
    """
    Summarize all camera-frame point estimates.
    汇总所有相机坐标系下的测试点结果。
    """
    valid_camera = []

    for idx, est in enumerate(estimates, start=1):
        if est.success and est.point_camera_m is not None:
            valid_camera.append((idx, est.point_camera_m))

    print("\n" + "=" * 80)
    print(bilingual("Final summary", "最终汇总"))
    print("=" * 80)

    print(
        bilingual(
            f"valid camera-frame points: {len(valid_camera)}/{len(estimates)}",
            f"有效相机坐标点: {len(valid_camera)}/{len(estimates)}",
        )
    )

    if len(valid_camera) == 0:
        print(bilingual(
            "No valid camera-frame points.",
            "没有有效的相机坐标点。",
        ))
        return

    points = np.array([p for _, p in valid_camera], dtype=np.float64)

    mean_p = np.mean(points, axis=0)
    median_p = np.median(points, axis=0)
    std_p = np.std(points, axis=0)

    print("\n" + bilingual("Camera-frame coordinates", "相机坐标系结果"))

    for point_id, p in valid_camera:
        print(
            bilingual(
                f"Point {point_id}: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) m",
                f"测试点 {point_id}: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) m",
            )
        )

    print(
        bilingual(
            f"Mean P_camera = ({mean_p[0]:.4f}, {mean_p[1]:.4f}, {mean_p[2]:.4f}) m",
            f"相机坐标平均值 = ({mean_p[0]:.4f}, {mean_p[1]:.4f}, {mean_p[2]:.4f}) m",
        )
    )

    print(
        bilingual(
            f"Median P_camera = ({median_p[0]:.4f}, {median_p[1]:.4f}, {median_p[2]:.4f}) m",
            f"相机坐标中位数 = ({median_p[0]:.4f}, {median_p[1]:.4f}, {median_p[2]:.4f}) m",
        )
    )

    print(
        bilingual(
            f"Std P_camera = ({std_p[0] * 1000:.1f}, {std_p[1] * 1000:.1f}, {std_p[2] * 1000:.1f}) mm",
            f"相机坐标标准差 = ({std_p[0] * 1000:.1f}, {std_p[1] * 1000:.1f}, {std_p[2] * 1000:.1f}) mm",
        )
    )

    if len(points) >= 2:
        max_pairwise = 0.0

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                d_mm = float(np.linalg.norm(points[i] - points[j]) * 1000.0)
                max_pairwise = max(max_pairwise, d_mm)

        print(
            bilingual(
                f"Max pairwise distance in camera frame = {max_pairwise:.1f} mm",
                f"相机坐标下最大两两距离 = {max_pairwise:.1f} mm",
            )
        )

    print("\n" + bilingual(
        "Note: P_camera is measured in the current camera frame. If the camera moves, P_camera will naturally change.",
        "注意：P_camera 是当前相机坐标系下的测量结果。如果相机移动，P_camera 本来就会变化。",
    ))


# ============================================================
# Main / 主程序
# ============================================================

def main():
    zed = None
    estimates = []


    try:
        print(bilingual("Opening ZED camera...", "正在打开 ZED 相机..."))

        zed = ZedCamera()
        zed.open()

        detector = GreenDefectDetector()

        print("\n" + bilingual(
            "Camera-only test started.",
            "纯相机测试开始。",
        ))

        print(bilingual(
            "This script only estimates P_camera. P1 reference compensation is not used.",
            "本脚本只估计 P_camera，不使用 P1 参考坐标补偿。",
        ))

        print(bilingual(
            "If the camera moves between test points, P_camera will naturally change.",
            "如果相机在测试点之间移动，P_camera 本来就会变化。",
        ))

        for point_id in range(1, NUM_TEST_POINTS + 1):
            print("\n" + "=" * 80)
            print(bilingual(
                f"Prepare test point {point_id}/{NUM_TEST_POINTS}",
                f"准备测试点 {point_id}/{NUM_TEST_POINTS}",
            ))
            print("=" * 80)

            print(bilingual(
                f"Move camera to point {point_id}. A live preview window is shown now.",
                f"请将相机移动到测试点 {point_id}。现在会显示实时预览窗口。",
            ))

            preview_camera_until_confirm(
                zed=zed,
                detector=detector,
                point_id=point_id,
                title=f"Point {point_id} Preview / 测试点 {point_id} 预览",
                show_mask=False,
            )

            estimate = capture_defect_position_for_seconds(
                zed=zed,
                detector=detector,
                capture_seconds=CAPTURE_SECONDS_PER_VIEW,
                show_debug=SHOW_DEBUG,
                point_id=point_id,
            )

            estimates.append(estimate)
            print_estimate(point_id, estimate)

        summarize_estimates(estimates)

        if SAVE_CSV_LOG:
            create_log_folder()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(LOG_FOLDER, f"camera_only_{ts}.csv")
            save_all_records(estimates, csv_path)

    except KeyboardInterrupt:
        print("\n" + bilingual("Interrupted by user.", "用户中断。"))

    finally:
        print("\n" + bilingual("Closing resources...", "正在关闭资源..."))

        if zed is not None:
            try:
                zed.close()
            except Exception as e:
                print(bilingual(f"Failed to close ZED: {e}", f"关闭 ZED 失败: {e}"))

        cv2.destroyAllWindows()
        print(bilingual("Done.", "完成。"))


if __name__ == "__main__":
    main()