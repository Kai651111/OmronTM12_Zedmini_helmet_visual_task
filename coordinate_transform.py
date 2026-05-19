"""
coordinate_transform.py

坐标变换模块。
Coordinate transform module.
"""

from typing import Tuple
import numpy as np


def rot_x(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ], dtype=np.float64)


def rot_y(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float64)


def rot_z(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
    ], dtype=np.float64)


def pose6d_to_transform(
    pose_mm_deg: list,
    use_rotation: bool = True,
) -> np.ndarray:
    """
    将 6D pose 转换成 4x4 齐次变换矩阵。
    Convert 6D pose to 4x4 homogeneous transform.

    Input:
        pose_mm_deg = [x, y, z, rx, ry, rz]

    Unit:
        x/y/z: mm
        rx/ry/rz: degree

    Output:
        T: 4x4 matrix, translation in meter.

    注意：
        这里暂时使用 ZYX Euler 假设：
        R = Rz(rz) @ Ry(ry) @ Rx(rx)

        实机前必须验证 TM 的姿态角定义。
    """
    x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = pose_mm_deg

    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    if use_rotation:
        R = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    else:
        R = np.eye(3, dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0

    return T


def transform_point(T: np.ndarray, point_m: np.ndarray) -> np.ndarray:
    """
    使用 4x4 变换矩阵变换 3D 点。
    Transform 3D point using 4x4 matrix.

    Input:
        point_m: [x, y, z], unit meter.

    Output:
        transformed point, unit meter.
    """
    p_h = np.array([point_m[0], point_m[1], point_m[2], 1.0], dtype=np.float64)
    out_h = T @ p_h
    return out_h[:3]


def read_tm12_base_pose(robot, fallback_pose=None) -> list:
    """
    读取 TM12 当前 TCP 在 robot base 下的 pose。

    对真实机器人：
        如果 robot 支持 tcp_coord，则读取 robot.tcp_coord。

    对 simulator_listen_only：
        TM_ListenNodeRobot 不能读 Modbus，所以没有 tcp_coord。
        这时使用 fallback_pose。

    Output:
        [x, y, z, rx, ry, rz]
    """
    if hasattr(robot, "tcp_coord"):
        return robot.tcp_coord

    if fallback_pose is not None:
        return fallback_pose

    raise RuntimeError(
        "Cannot read robot base pose. "
        "This robot interface has no tcp_coord. "
        "Provide fallback_pose."
    )


def build_camera_to_base_matrix(
    camera_relative_pose_tcp: list,
    robot_tcp_pose_base: list,
    use_base_aligned_tcp_rotation: bool = False,
    camera_rotation_matrix_tcp=None,
) -> np.ndarray:
    """
    计算 T_base_camera。

    Input:
        camera_relative_pose_tcp:
            摄像头相对于机械臂末端/TCP 的 6D pose。
            [x, y, z, rx, ry, rz], mm + degree.

        robot_tcp_pose_base:
            当前 TCP 相对于 robot base 的 6D pose。
            [x, y, z, rx, ry, rz], mm + degree.

    Output:
        T_base_camera:
            camera frame 到 robot base/global frame 的 4x4 变换矩阵。

    Formula:
        T_base_camera = T_base_tcp @ T_tcp_camera
    """
    T_base_tcp = pose6d_to_transform(
        robot_tcp_pose_base,
        use_rotation=not use_base_aligned_tcp_rotation,
    )
    T_tcp_camera = pose6d_to_transform(camera_relative_pose_tcp)

    if camera_rotation_matrix_tcp is not None:
        R_tcp_camera = np.array(camera_rotation_matrix_tcp, dtype=np.float64)

        if R_tcp_camera.shape != (3, 3):
            raise ValueError(
                "camera_rotation_matrix_tcp must be a 3x3 matrix. "
                f"Got shape {R_tcp_camera.shape}."
            )

        T_tcp_camera[:3, :3] = R_tcp_camera

    T_base_camera = T_base_tcp @ T_tcp_camera

    return T_base_camera


def camera_point_to_global(
    point_camera_m: np.ndarray,
    T_base_camera: np.ndarray,
) -> np.ndarray:
    """
    将 defect 坐标从 camera frame 转换到 global/base frame。

    Input:
        point_camera_m:
            defect 在 camera frame 下的坐标，单位 m。

        T_base_camera:
            camera frame 到 base frame 的矩阵。

    Output:
        point_global_m:
            defect 在 global/base frame 下的坐标，单位 m。
    """
    return transform_point(T_base_camera, point_camera_m)
