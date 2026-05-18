"""
marker_compensation.py

画笔补偿模块。
Marker compensation module.
"""

import numpy as np

from coordinate_transform import pose6d_to_transform, transform_point


def compensate_marker_point_global(
    defect_point_global_m: np.ndarray,
    marker_relative_pose_tcp: list,
    current_tcp_pose_base: list,
) -> np.ndarray:
    """
    根据画笔相对 TCP 的位置，对 defect target 做补偿。

    Input:
        defect_point_global_m:
            defect 在 global/base frame 下的位置，单位 m。

        marker_relative_pose_tcp:
            画笔相对 TCP 的 6D pose。
            [x, y, z, rx, ry, rz], mm + degree.

        current_tcp_pose_base:
            当前 TCP 相对 base 的 6D pose。
            [x, y, z, rx, ry, rz], mm + degree.

    Output:
        compensated_point_global_m:
            补偿后的 global target point，单位 m。

    说明：
        这里先采用最简单的补偿逻辑：
        计算 marker tip 相对于 TCP 的 global offset，
        然后让 TCP 运动目标反向补偿这个 offset。

        target_tcp = defect_point - marker_offset_global
    """
    T_base_tcp = pose6d_to_transform(current_tcp_pose_base)
    T_tcp_marker = pose6d_to_transform(marker_relative_pose_tcp)

    T_base_marker = T_base_tcp @ T_tcp_marker

    tcp_origin_global_m = T_base_tcp[:3, 3]
    marker_origin_global_m = T_base_marker[:3, 3]

    marker_offset_global_m = marker_origin_global_m - tcp_origin_global_m

    compensated_point_global_m = defect_point_global_m - marker_offset_global_m

    return compensated_point_global_m