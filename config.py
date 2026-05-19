"""
config.py

项目参数配置。
Project configuration.
"""

import numpy as np
import pyzed.sl as sl


# ============================================================
# Robot connection
# ============================================================

ROBOT_IP = "192.168.19.22"

# simulator_listen_only:
#   使用 TMFlow Simulator + Listen Node。
#
# real_robot_full:
#   后面实机时可以扩展。
ROBOT_MODE = "real_robot_full"

ENABLE_ROBOT_MOVE = True # 是否真的让机器人执行运动指令

# 是否执行“下笔”Marking动作
RUN_FINAL_MARKING = False


# ============================================================
# Reference point
# ============================================================

# P_circle_center:
#   头盔正上方 30 cm 的参考点。
#   你之后手动填写。
#
# Format:
#   [x, y, z, rx, ry, rz]
#
# Unit:
#   x/y/z: mm
#   rx/ry/rz: degree
P_CIRCLE_CENTER = [
    -206.0, 900.0, 730.0, -90.0, 45, 0
]

CIRCLE_RADIUS_MM = 100.0
NUM_SCAN_POINTS = 7

# 圆形轨迹所在平面与 global XY 平面平行。
# Therefore only x/y change; z and orientation are copied from P_CIRCLE_CENTER.
CIRCLE_START_ANGLE_DEG = 0.0 # 第一个扫描点从圆上的哪个角度开始。


# ============================================================
# Pre tasks before circle scan
# ============================================================

PRE_TASKS = [
    # Placeholder pre-task poses. Motion supports "ptp" and "line".
    {
        "name": "pre_ptp_above_circle_center",
        "motion": "ptp",
        "pose": [-360.0, 630.0, 760.0, -90.0, 45.0, 0.0],
    },
    {
        "name": "pre_line_small_offset",
        "motion": "line",
        "pose": [-330.0, 630.0, 760.0, -90.0, 45.0, 0.0],
    },
]


# ============================================================
# ZED settings
# ============================================================

ZED_RESOLUTION = sl.RESOLUTION.HD720
ZED_FPS = 15
ZED_DEPTH_MODE = sl.DEPTH_MODE.NEURAL #有PERFORMANCE、NEURAL和ULTRA。PERFORMANCE是性能，但越往后（尤其是ULTRA）越稳。
ZED_UNIT = sl.UNIT.METER

# 每个扫描点采集多少秒。
CAPTURE_SECONDS_PER_VIEW = 3.0

# HSV 绿色阈值
LOWER_GREEN = np.array([30, 50, 50], dtype=np.uint8)   #HSV阈值下界
UPPER_GREEN = np.array([95, 255, 255], dtype=np.uint8) #HSV阈值上界

# 过滤太小或太大的绿色区域。
MIN_GREEN_AREA = 200   #最小绿色面积
MAX_GREEN_AREA = 50000 #最大绿色面积
MIN_CIRCULARITY = 0.40 #圆度

ROI_REL = (0.05, 0.05, 0.95, 0.95)

MAX_POINT_SAMPLES = 1500
Z_MIN_M = 0.10
Z_MAX_M = 2.00

MIN_VALID_FRAMES_PER_VIEW = 20
MAX_CENTROID_STD_PX = 10.0
MAX_AREA_REL_STD = 0.40

# ZED depth 在低纹理区域可能抖动，调试阶段可以先 False。
USE_DEPTH_QUALITY_CHECK = True
MAX_DEPTH_MAD_MM = 20 #深度抖动的毫米mm


# 是否显示 ZED 原始实时图像窗口。
# Whether to show raw ZED live image window.
SHOW_RAW_ZED_WINDOW = True
ENABLE_PREVIEW_BEFORE_CAPTURE = True

# ============================================================
# Coordinate transform
# ============================================================

# 摄像头相对于机械臂末端/TCP 的 6D pose。
# 表示ZED 相机坐标系相对于机器人 TCP 坐标系的位置和姿态。
#
# Format:
#   [x, y, z, rx, ry, rz]
#
# Unit:
#   x/y/z: mm
#   rx/ry/rz: degree
#
# 你之后需要根据机械安装或 hand-eye calibration 填写。
CAMERA_RELATIVE_POSE_TCP = [                # 非常重要！！！要写。
    -21.15, 4.7, 178.85, 0.0, 0.0, 180.0
]

# Direct ZED camera-axis mapping in TCP coordinates.
# Rows compute TCP coordinates from camera coordinates:
#   [x_T, y_T, z_T]^T = CAMERA_ROTATION_MATRIX_TCP @ [x_C, y_C, z_C]^T
# This mapping is inferred from the latest P4/P5/P6 scan data:
#   x_T ~= y_C, y_T ~= x_C, z_T ~= z_C
CAMERA_ROTATION_MATRIX_TCP = [
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
]

# Coordinate transform debug mode.
# Your Tool jog test showed Tool +X/+Y/+Z align with Base +X/+Y/+Z.
# When this is True, main_script treats the TCP frame rotation as base-aligned:
#   R_base_tcp = I
# It still uses the TCP position [x, y, z] and still uses CAMERA_RELATIVE_POSE_TCP.
USE_BASE_ALIGNED_TCP_ROTATION = True

# 画笔相对于机械臂末端/TCP 的 6D pose。
#
# 先空着；你之后手动填写。
MARKER_RELATIVE_POSE_TCP = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]


# ============================================================
# Decision 稳定性判断参数设定
# ============================================================

# 多个 global defect points 之间的一致性阈值。（这些点允许的最大误差 in mm）
CONSISTENCY_TOLERANCE_MM = 80.0 # 系统如果准了降到20mm

# 至少需要多少个扫描的有效点（假设一共扫描了5个点）
MIN_VALID_GLOBAL_POINTS = 2


# ============================================================
# Final marking motion
# ============================================================

# 最终标记动作：
#   移动到目标点正上方 30 cm
#   下移 30 cm
#   停留 1s
#   上移 30 cm
APPROACH_HEIGHT_MM = 300.0
MARK_DWELL_SECONDS = 1.0

LINE_SPEED_MM_S = 50.0
PTP_SPEED_PERCENT = 10



# 安全工作空间范围：限制机器人的动作（defect）在这个位置之内，一共是xyz三个参数。
SAFE_X_RANGE = (-600.0, 900.0)
SAFE_Y_RANGE = (-500.0, 1000.0)
SAFE_Z_RANGE = (50.0, 1200.0)



# ============================================================
# Logs
# ============================================================

SAVE_CSV_LOG = True
LOG_FOLDER = "scan_logs"
