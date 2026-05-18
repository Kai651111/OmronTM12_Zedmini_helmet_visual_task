"""
robot_motion.py

机器人运动与轨迹生成。
Robot motion and trajectory generation.
"""

import math
import time
from typing import List

from config import (
    ENABLE_ROBOT_MOVE,
    SAFE_X_RANGE,
    SAFE_Y_RANGE,
    SAFE_Z_RANGE,
    PTP_SPEED_PERCENT,
    LINE_SPEED_MM_S,
    MARK_DWELL_SECONDS,
)


def is_pose_safe(pose: list) -> bool:
    """
    检查机器人 pose 是否在安全工作空间内。
    Check if robot pose is inside safe workspace.
    """
    x, y, z, rx, ry, rz = pose

    return (
        SAFE_X_RANGE[0] <= x <= SAFE_X_RANGE[1]
        and SAFE_Y_RANGE[0] <= y <= SAFE_Y_RANGE[1]
        and SAFE_Z_RANGE[0] <= z <= SAFE_Z_RANGE[1]
    )


def move_ptp(robot, pose: list, speed_percent: float = PTP_SPEED_PERCENT):
    """
    PTP 移动。
    PTP motion.
    """
    if not is_pose_safe(pose):
        raise ValueError(f"Unsafe PTP pose: {pose}")

    print("[ROBOT] PTP:", pose)

    if robot is not None and ENABLE_ROBOT_MOVE:
        robot.ptp(
            pose,
            speed_percent,
            data_format="CPP",
            blending=0,
            precision_positioning="false",
        )


def move_line(robot, pose: list, speed_mm_s: float = LINE_SPEED_MM_S):
    """
    直线移动。
    Linear motion.
    """
    if not is_pose_safe(pose):
        raise ValueError(f"Unsafe Line pose: {pose}")

    print("[ROBOT] Line:", pose)

    if robot is not None and ENABLE_ROBOT_MOVE:
        robot.line(
            pose,
            speed_mm_s,
            data_format="CAP",
            blending=0,
            precision_positioning="false",
        )


def generate_circle_scan_poses(
    center_pose: list,
    radius_mm: float,
    num_points: int,
    start_angle_deg: float = 0.0,
) -> List[list]:
    """
    以 center_pose 为圆心，在 global XY 平面生成圆形扫描点。

    The circular path plane is parallel to global XY plane.

    Input:
        center_pose:
            [x, y, z, rx, ry, rz]

        radius_mm:
            radius in mm.

        num_points:
            number of points.

    Output:
        list of poses.
    """
    cx, cy, cz, rx, ry, rz = center_pose

    poses = []

    for i in range(num_points):
        theta = math.radians(start_angle_deg + i * 360.0 / num_points)

        x = cx + radius_mm * math.cos(theta)
        y = cy + radius_mm * math.sin(theta)
        z = cz

        pose = [
            float(x),
            float(y),
            float(z),
            float(rx),
            float(ry),
            float(rz),
        ]

        poses.append(pose)

    return poses


def execute_vertical_marking_motion(
    robot,
    target_pose: list,
    approach_height_mm: float,
    speed_mm_s: float = LINE_SPEED_MM_S,
):
    """
    最终画笔动作：

    1. 移动到目标点正上方 approach_height_mm。
    2. 下移 approach_height_mm。
    3. 停留 1 秒。
    4. 上移 approach_height_mm。

    注意：
        这里假设“垂直于 xy 平面”就是 global Z 方向。
    """
    x, y, z, rx, ry, rz = target_pose

    above_pose = [
        x,
        y,
        z + approach_height_mm,
        rx,
        ry,
        rz,
    ]

    touch_pose = [
        x,
        y,
        z,
        rx,
        ry,
        rz,
    ]

    print("[MARK] Move above target.")
    move_line(robot, above_pose, speed_mm_s=speed_mm_s)

    print("[MARK] Move down to target.")
    move_line(robot, touch_pose, speed_mm_s=speed_mm_s)

    print(f"[MARK] Dwell {MARK_DWELL_SECONDS:.1f}s.")
    time.sleep(MARK_DWELL_SECONDS)

    print("[MARK] Move back up.")
    move_line(robot, above_pose, speed_mm_s=speed_mm_s)