"""
Global ellipsoid boundary helpers.

The ellipsoid is aligned with the robot base/global axes:
    length -> global X
    width  -> global Y
    height -> global Z
"""

from typing import Optional, Tuple

import cv2
import numpy as np

import config as cfg


def is_ellipsoid_boundary_enabled() -> bool:
    return bool(getattr(cfg, "ENABLE_ELLIPSOID_BOUNDARY", False))


def get_ellipsoid_center_and_axes_m() -> Tuple[np.ndarray, np.ndarray]:
    top_mm = np.array(
        getattr(cfg, "ELLIPSOID_TOP_TANGENT_POINT_GLOBAL_MM"),
        dtype=np.float64,
    )

    length_x_mm = float(getattr(cfg, "ELLIPSOID_LENGTH_X_MM"))
    width_y_mm = float(getattr(cfg, "ELLIPSOID_WIDTH_Y_MM"))
    height_z_mm = float(getattr(cfg, "ELLIPSOID_HEIGHT_Z_MM"))
    margin_mm = float(getattr(cfg, "ELLIPSOID_BOUNDARY_MARGIN_MM", 0.0))

    center_mm = top_mm.copy()
    center_mm[2] = top_mm[2] - height_z_mm / 2.0

    axes_mm = np.array(
        [
            length_x_mm / 2.0 + margin_mm,
            width_y_mm / 2.0 + margin_mm,
            height_z_mm / 2.0 + margin_mm,
        ],
        dtype=np.float64,
    )

    if np.any(axes_mm <= 1e-6):
        raise ValueError(
            "Ellipsoid semi-axes must be positive after applying margin. "
            f"Got axes_mm={axes_mm}."
        )

    return center_mm / 1000.0, axes_mm / 1000.0


def ellipsoid_metric(point_global_m: np.ndarray) -> float:
    center_m, axes_m = get_ellipsoid_center_and_axes_m()
    q = (np.array(point_global_m, dtype=np.float64) - center_m) / axes_m
    return float(np.dot(q, q))


def is_point_inside_global_ellipsoid(point_global_m: np.ndarray) -> Tuple[bool, float]:
    if not is_ellipsoid_boundary_enabled():
        return True, 0.0

    metric = ellipsoid_metric(point_global_m)
    return metric <= 1.0, metric


def make_ellipsoid_surface_points_m(
    latitude_steps: int = 24,
    longitude_steps: int = 48,
) -> np.ndarray:
    center_m, axes_m = get_ellipsoid_center_and_axes_m()

    points = []
    for i in range(latitude_steps + 1):
        theta = np.pi * i / latitude_steps
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for j in range(longitude_steps):
            phi = 2.0 * np.pi * j / longitude_steps
            local = np.array(
                [
                    axes_m[0] * sin_theta * np.cos(phi),
                    axes_m[1] * sin_theta * np.sin(phi),
                    axes_m[2] * cos_theta,
                ],
                dtype=np.float64,
            )
            points.append(center_m + local)

    return np.array(points, dtype=np.float64)


def _project_global_points_to_image(
    points_global_m: np.ndarray,
    T_base_camera: np.ndarray,
    intrinsics: dict,
) -> np.ndarray:
    T_camera_base = np.linalg.inv(T_base_camera)

    points_h = np.column_stack(
        [
            points_global_m,
            np.ones(len(points_global_m), dtype=np.float64),
        ]
    )
    points_camera = (T_camera_base @ points_h.T).T[:, :3]

    z = points_camera[:, 2]
    visible = z > 0.02

    if not np.any(visible):
        return np.empty((0, 2), dtype=np.int32)

    points_camera = points_camera[visible]

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    u = fx * points_camera[:, 0] / points_camera[:, 2] + cx
    v = fy * points_camera[:, 1] / points_camera[:, 2] + cy

    projected = np.column_stack([u, v])
    finite = np.isfinite(projected).all(axis=1)
    return np.round(projected[finite]).astype(np.int32)


def draw_ellipsoid_projection_on_image(
    image_bgr: np.ndarray,
    T_base_camera: Optional[np.ndarray],
    intrinsics: Optional[dict],
) -> np.ndarray:
    if not is_ellipsoid_boundary_enabled():
        return image_bgr

    if not bool(getattr(cfg, "SHOW_ELLIPSOID_PROJECTION", True)):
        return image_bgr

    if T_base_camera is None or intrinsics is None:
        return image_bgr

    surface_points = make_ellipsoid_surface_points_m()
    projected = _project_global_points_to_image(
        surface_points,
        T_base_camera=T_base_camera,
        intrinsics=intrinsics,
    )

    if len(projected) < 6:
        return image_bgr

    h, w = image_bgr.shape[:2]
    in_frame = (
        (projected[:, 0] >= -w)
        & (projected[:, 0] <= 2 * w)
        & (projected[:, 1] >= -h)
        & (projected[:, 1] <= 2 * h)
    )
    projected = projected[in_frame]

    if len(projected) < 6:
        return image_bgr

    color = tuple(
        int(v) for v in getattr(cfg, "ELLIPSOID_PROJECTION_COLOR_BGR", [255, 0, 255])
    )
    alpha = float(getattr(cfg, "ELLIPSOID_PROJECTION_ALPHA", 0.20))
    alpha = min(max(alpha, 0.0), 1.0)

    hull = cv2.convexHull(projected.reshape(-1, 1, 2))

    overlay = image_bgr.copy()
    cv2.fillConvexPoly(overlay, hull, color)
    cv2.addWeighted(overlay, alpha, image_bgr, 1.0 - alpha, 0.0, image_bgr)

    cv2.polylines(image_bgr, [hull], isClosed=True, color=color, thickness=2)

    cv2.putText(
        image_bgr,
        "Ellipsoid boundary",
        (20, max(25, h - 55)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )

    return image_bgr
