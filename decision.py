"""
decision.py

多点稳定性判断与融合。
Multi-point stability decision and fusion.
"""

from typing import List, Tuple, Optional
import numpy as np

from config import (
    CONSISTENCY_TOLERANCE_MM,
    MIN_VALID_GLOBAL_POINTS,
)


def pairwise_max_distance_mm(points_m: np.ndarray) -> float:
    """
    计算点集中的最大两两距离，单位 mm。
    Compute max pairwise distance, unit mm.
    """
    max_dist = 0.0

    for i in range(len(points_m)):
        for j in range(i + 1, len(points_m)):
            d_mm = float(np.linalg.norm(points_m[i] - points_m[j]) * 1000.0)
            max_dist = max(max_dist, d_mm)

    return max_dist


def robust_fuse_global_points(points_global_m: List[np.ndarray]) -> np.ndarray:
    """
    融合多个 global defect points。

    方法：
        1. 求 median。
        2. 去掉最远的 20%。
        3. 对剩余点求 mean。

    Input:
        list of points in meter.

    Output:
        fused point in meter.
    """
    points = np.array(points_global_m, dtype=np.float64)

    median = np.median(points, axis=0)
    dist = np.linalg.norm(points - median, axis=1)

    threshold = np.percentile(dist, 80)
    inliers = points[dist <= threshold]

    if len(inliers) == 0:
        return median

    return np.mean(inliers, axis=0)


def judge_global_points_stability(
    points_global_m: List[np.ndarray],
) -> Tuple[bool, Optional[np.ndarray], str]:
    """
    判断多个 global coordinates 下的 defect points 是否稳定。

    Input:
        points_global_m:
            list[np.ndarray], 每个元素为 [Xb, Yb, Zb]，单位 m。
            列表大小不固定。

    Output:
        stable:
            是否稳定。

        fused_point_m:
            置信度最高/融合后的 global point，单位 m。

        reason:
            原因说明。
    """
    if len(points_global_m) < MIN_VALID_GLOBAL_POINTS:
        return (
            False,
            None,
            f"Not enough valid global points: {len(points_global_m)}"
        )

    points = np.array(points_global_m, dtype=np.float64)

    max_dist_mm = pairwise_max_distance_mm(points)

    if max_dist_mm > CONSISTENCY_TOLERANCE_MM:
        return (
            False,
            None,
            f"Global points inconsistent. Max distance = {max_dist_mm:.1f} mm"
        )

    fused_point_m = robust_fuse_global_points(points_global_m)

    return (
        True,
        fused_point_m,
        f"Stable. Max distance = {max_dist_mm:.1f} mm"
    )