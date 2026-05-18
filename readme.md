调用坐标配置矩阵函数
    → build_camera_transform_for_current_pose()
    → build_camera_to_base_matrix()

传入摄像头相对位置向量
    → CAMERA_RELATIVE_POSE_TCP

读取 tm12 base 坐标
    → read_tm12_base_pose()

计算坐标转换矩阵
    → T_base_camera

执行前置任务
    → run_pre_tasks()

回到 P_circle_center
    → move_ptp(robot, P_CIRCLE_CENTER)

以 P_circle_center 为中心，半径 20 cm，生成五个点
    → generate_circle_scan_poses()

机器人移动到第 i 个点 Pi
    → move_ptp(robot, Pi)

ZED 录制一段时间并统计相对位置
    → capture_defect_position_for_seconds()

转换 defect 到 global coordinates
    → camera_point_to_global()

判断 global points 是否稳定
    → judge_global_points_stability()

画笔补偿
    → compensate_marker_point_global()

垂直于 XY 平面执行标记动作
    → execute_vertical_marking_motion()

***

# 手动填写：

`config.py`里面的：

```
P_CIRCLE_CENTER = [
    370.0, 300.0, 300.0, 150.0, 0.0, 90.0
]

CAMERA_RELATIVE_POSE_TCP = [
    0.0, 0.0, 100.0, 0.0, 0.0, 0.0
]

MARKER_RELATIVE_POSE_TCP = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]
```

