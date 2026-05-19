"""
main.py

TM12 + ZED Mini 主流程脚本。
TM12 + ZED Mini main workflow script.

核心原则：
    main.py 只负责流程编排，不重复实现底层函数。

依赖模块：
    config.py               参数配置
    techman.py              TM12 通信
    vision_zed.py           ZED 相机和绿色 defect 检测
    coordinate_transform.py 坐标变换
    robot_motion.py         机器人运动与扫描轨迹
    decision.py             多点稳定性判断
    marker_compensation.py  画笔补偿
"""

import time
import traceback

import cv2
import numpy as np

import config as cfg
import techman as tm

from vision_zed import (
    ZedCamera,
    GreenDefectDetector,
    preview_camera_until_confirm,
    capture_defect_position_for_seconds,
)

from coordinate_transform import (
    read_tm12_base_pose,
    build_camera_to_base_matrix,
    camera_point_to_global,
)

from robot_motion import (
    move_ptp,
    generate_circle_scan_poses,
    execute_vertical_marking_motion,
)

from decision import judge_global_points_stability

from marker_compensation import compensate_marker_point_global


# ============================================================
# Optional config values / 可选配置项
# ============================================================

ETHERNET_TABLE_NAME = getattr(cfg, "ETHERNET_TABLE_NAME", "Default")

SETTLE_SECONDS_AFTER_MOVE = getattr(cfg, "SETTLE_SECONDS_AFTER_MOVE", 1.0)

ENABLE_PREVIEW_BEFORE_CAPTURE = getattr(
    cfg,
    "ENABLE_PREVIEW_BEFORE_CAPTURE",
    False,
)

SET_ZED_REFERENCE_AT_CIRCLE_CENTER = getattr(
    cfg,
    "SET_ZED_REFERENCE_AT_CIRCLE_CENTER",
    True,
)

BLOCK_IF_NOT_STABLE = getattr(
    cfg,
    "BLOCK_IF_NOT_STABLE",
    True,
)

# 安全起见，默认不执行最终下笔动作。
# 等你确认 P_global 稳定之后，再在 config.py 里加：
# RUN_FINAL_MARKING = True
RUN_FINAL_MARKING = getattr(
    cfg,
    "RUN_FINAL_MARKING",
    False,
)


# 等待函数
def wait_until_pose_reached(
    robot,
    target_pose,
    position_tolerance_mm=5.0,
    angle_tolerance_deg=5.0,
    timeout_s=120.0,
    check_interval_s=0.5,
):
    """
    Wait until robot TCP is close to target pose.
    等待机器人 TCP 接近目标 pose。

    target_pose:
        [x, y, z, rx, ry, rz], mm + degree
    """

    start_time = time.time()

    target_pos = np.array(target_pose[:3], dtype=np.float64)
    target_rot = np.array(target_pose[3:6], dtype=np.float64)

    while True:
        current_pose = robot.tcp_coord
        current_pose = [float(v) for v in current_pose[:6]]

        current_pos = np.array(current_pose[:3], dtype=np.float64)
        current_rot = np.array(current_pose[3:6], dtype=np.float64)

        pos_error_mm = float(np.linalg.norm(current_pos - target_pos))
        rot_error_deg = float(np.linalg.norm(current_rot - target_rot))

        print(
            f"[WAIT] pos_error = {pos_error_mm:.2f} mm, "
            f"rot_error = {rot_error_deg:.2f} deg, "
            f"tcp = {np.round(current_pose, 3)}"
        )

        if pos_error_mm <= position_tolerance_mm and rot_error_deg <= angle_tolerance_deg:
            print("[WAIT] Target reached.")
            print("[WAIT] 已到达目标点。")
            return current_pose

        elapsed = time.time() - start_time

        if elapsed > timeout_s:
            raise TimeoutError(
                f"Robot did not reach target within {timeout_s:.1f}s. "
                f"Last pos_error={pos_error_mm:.2f} mm, "
                f"rot_error={rot_error_deg:.2f} deg"
            )

        time.sleep(check_interval_s)


# ============================================================
# Main workflow / 主流程
# ============================================================

def main():
    robot = None
    zed = None

    global_points_m = []
    tcp_poses_base = []

    try:

        # Debug Viability Checkt
        print("检查")



        # ----------------------------------------------------
        # 0. Safety checks
        # 0. 安全检查
        # ----------------------------------------------------

        print("\n" + "=" * 80)
        print("[MAIN] TM12 + ZED Mini scanning workflow started.")
        print("[MAIN] TM12 + ZED Mini 扫描主流程开始。")
        print("=" * 80)


        # step 0 安全检查：
        #
        # ROBOT_MODE 是否是 real_robot_full
        # ROBOT_IP 是否还是 127.0.0.1 （这是本机Simulator的地址）
        # RUN_FINAL_MARKING 是否为 False
        # ENABLE_ROBOT_MOVE 当前是什么状态
        robot_mode = getattr(cfg, "ROBOT_MODE", "real_robot_full")

        if robot_mode != "real_robot_full":
            raise RuntimeError(
                "请先在 config.py 中设置 ROBOT_MODE = 'real_robot_full'。"
            )

        if cfg.ROBOT_IP in ("127.0.0.1", "localhost"):
            raise RuntimeError(
                "当前 ROBOT_IP 还是本机地址。请先在 config.py 中设置真实 TM12 IP。"
            )

        print(f"[CONFIG] ROBOT_IP = {cfg.ROBOT_IP}")
        print(f"[CONFIG] ROBOT_MODE = {robot_mode}")
        print(f"[CONFIG] ENABLE_ROBOT_MOVE = {cfg.ENABLE_ROBOT_MOVE}")
        print(f"[CONFIG] RUN_FINAL_MARKING = {RUN_FINAL_MARKING}")

        if not RUN_FINAL_MARKING:
            print(
                "[SAFETY] RUN_FINAL_MARKING = False, "
                "本次只扫描和计算，不执行最终下笔动作。"
            )

        # ----------------------------------------------------
        # 1. Connect real TM12
        # 1. 连接真实 TM12
        # ----------------------------------------------------

        print("\n" + "-" * 80)
        print("[1/7] Connecting real TM12...")
        print("[1/7] 正在连接真实 TM12...")


        # 链接真实tm12
        robot = tm.TM_Robot(
            cfg.ROBOT_IP,
            table_name=ETHERNET_TABLE_NAME,
        )

        robot.connect_listen_node()

        print("[ROBOT] Connected.")
        print("[ROBOT] 已连接。")

        # ----------------------------------------------------
        # 2. Open ZED camera
        # 2. 打开 ZED 相机
        # ----------------------------------------------------

        print("\n" + "-" * 80)
        print("[2/7] Opening ZED camera...")
        print("[2/7] 正在打开 ZED 相机...")

        zed = ZedCamera()
        zed.open()

        detector = GreenDefectDetector() #创建绿色的sticker检测器

        print("[ZED] Ready.")
        print("[ZED] 已准备。")

        # # ----------------------------------------------------
        # # 3. Move to circle center
        # # 3. 移动到圆心参考点
        # # ----------------------------------------------------
        #
        # print("\n" + "-" * 80)
        # print("[3/7] Moving to P_CIRCLE_CENTER...")
        # print("[3/7] 正在移动到 P_CIRCLE_CENTER...")
        #
        # print(f"[POSE] P_CIRCLE_CENTER = {cfg.P_CIRCLE_CENTER}")
        #
        #
        # input("Press Enter to continue... / 按回车继续...")
        # # 移动圆心到参考点
        # # 如果不动：看看config.py里面的ENABLE_ROBOT_MOVE是不是False
        # target_pose = cfg.P_CIRCLE_CENTER
        #
        # # ----------------------------------------------------
        # # Direct Techman PTP move to P_CIRCLE_CENTER
        # # 直接使用 techman.py 的 robot.ptp() 移动到 P_CIRCLE_CENTER
        # # ----------------------------------------------------
        #
        # target_pose = cfg.P_CIRCLE_CENTER
        #
        # print("[ROBOT] Direct techman robot.ptp() to P_CIRCLE_CENTER:")
        # print("[ROBOT] 直接使用 techman.py 的 robot.ptp() 移动到 P_CIRCLE_CENTER:")
        # print(f"[ROBOT] target_pose = {target_pose}")
        # print(f"[ROBOT] speed_percent = {cfg.PTP_SPEED_PERCENT}")
        # print(f"[ROBOT] ENABLE_ROBOT_MOVE = {cfg.ENABLE_ROBOT_MOVE}")
        #
        # # 手动安全检查，不再通过 robot_motion.move_ptp()
        # # Manual safety check, because we bypass robot_motion.move_ptp()
        # x, y, z, rx, ry, rz = target_pose
        #
        # if not (
        #         cfg.SAFE_X_RANGE[0] <= x <= cfg.SAFE_X_RANGE[1]
        #         and cfg.SAFE_Y_RANGE[0] <= y <= cfg.SAFE_Y_RANGE[1]
        #         and cfg.SAFE_Z_RANGE[0] <= z <= cfg.SAFE_Z_RANGE[1]
        # ):
        #     raise ValueError(f"Unsafe P_CIRCLE_CENTER pose: {target_pose}")
        #
        # # 调试中断点：按回车才真正发送运动指令
        # # Debug pause: press Enter before sending motion command
        # input("[PAUSE] Press Enter to send robot.ptp()... / 按回车发送 robot.ptp()...")
        #
        # if cfg.ENABLE_ROBOT_MOVE:
        #     print("[ROBOT] Sending robot.ptp() command now...")
        #     print("[ROBOT] 正在发送 robot.ptp() 指令...")
        #
        #     robot.ptp(
        #         target_pose,
        #         cfg.PTP_SPEED_PERCENT,
        #         data_format="CPP",
        #         blending=0,
        #         precision_positioning="false",
        #     )
        #
        #     print("[ROBOT] robot.ptp() command sent.")
        #     print("[ROBOT] robot.ptp() 指令已发送。")
        #
        #     tcp_pose_base = wait_until_pose_reached(
        #         robot,
        #         target_pose,
        #         position_tolerance_mm=10.0,
        #         angle_tolerance_deg=10.0,
        #         timeout_s=180.0,
        #     )
        #
        # else:
        #     print("[ROBOT] ENABLE_ROBOT_MOVE = False, robot.ptp() skipped.")
        #     print("[ROBOT] ENABLE_ROBOT_MOVE = False，跳过真实运动。")
        #
        #     tcp_pose_base = robot.tcp_coord
        #     tcp_pose_base = [float(v) for v in tcp_pose_base[:6]]
        #
        # time.sleep(SETTLE_SECONDS_AFTER_MOVE) # 这个在本script里面设置，在59行附近。
        #
        # # 记录当前的zed的pose作为参考 （依据zed相机的Positional Tracking，不准确。）
        # # 所以这个不是主要坐标转换的核心
        # if SET_ZED_REFERENCE_AT_CIRCLE_CENTER:
        #     print("[ZED] Setting P1 reference at P_CIRCLE_CENTER...")
        #     print("[ZED] 在 P_CIRCLE_CENTER 设置 ZED P1 参考位姿...")
        #
        #     ok = zed.set_reference_from_current_pose()
        #
        #     if not ok:
        #         print("[WARN] Failed to set ZED P1 reference.")
        #         print("[WARN] ZED P1 参考位姿设置失败，但主流程继续。")



        # ----------------------------------------------------
        # 3. Skip P_CIRCLE_CENTER move
        # 3. 调试阶段：跳过圆心移动
        # ----------------------------------------------------

        print("\n" + "-" * 80)
        print("[3/7] Skip moving to P_CIRCLE_CENTER.")
        print("[3/7] 跳过移动到 P_CIRCLE_CENTER。")

        tcp_pose_base = robot.tcp_coord
        tcp_pose_base = [float(v) for v in tcp_pose_base[:6]]

        print("[ROBOT] Current TCP pose:")
        print(tcp_pose_base)

        input("[PAUSE] Check robot current position, press Enter to continue... / 检查机器人当前位置，按回车继续...")

        time.sleep(SETTLE_SECONDS_AFTER_MOVE)

        if SET_ZED_REFERENCE_AT_CIRCLE_CENTER:
            print("[ZED] Setting P1 reference at current pose...")
            print("[ZED] 在当前位置设置 ZED P1 参考位姿...")

            ok = zed.set_reference_from_current_pose()

            if not ok:
                print("[WARN] Failed to set ZED P1 reference.")
                print("[WARN] ZED P1 参考位姿设置失败，但主流程继续。")






        # ----------------------------------------------------
        # # 4. Generate scan poses
        # # 4. 生成圆形扫描点
        # # ----------------------------------------------------
        #
        # print("\n" + "-" * 80)
        # print("[4/7] Generating circular scan poses...")
        # print("[4/7] 正在生成圆形扫描点...")
        #
        # # 生成一组扫描点：根据confiure.py里面的参数写。
        # ----------------------------------------------------
        # 4. Use a tiny Z-only motion for connection debugging
        # 4. 调试阶段：只沿 Z 方向小范围移动
        # ----------------------------------------------------

        print("\n" + "-" * 80)
        print("[4/7] Using small Z-only debug scan poses...")
        print("[4/7] 使用只增加 Z 的小范围调试扫描点...")

        z_step_mm = 10.0
        scan_poses = [
            tcp_pose_base.copy(),
            [
                tcp_pose_base[0],
                tcp_pose_base[1],
                tcp_pose_base[2] + z_step_mm,
                tcp_pose_base[3],
                tcp_pose_base[4],
                tcp_pose_base[5],
            ],
        ]

        for i, pose in enumerate(scan_poses, start=1):
            print(f"[SCAN POSE] P{i}: {pose}")

            x, y, z, _, _, _ = pose
            if not (
                cfg.SAFE_X_RANGE[0] <= x <= cfg.SAFE_X_RANGE[1]
                and cfg.SAFE_Y_RANGE[0] <= y <= cfg.SAFE_Y_RANGE[1]
                and cfg.SAFE_Z_RANGE[0] <= z <= cfg.SAFE_Z_RANGE[1]
            ):
                raise ValueError(f"Unsafe debug scan pose P{i}: {pose}")

        print(
            "[SAFETY] Debug scan poses are inside configured safe workspace. "
            f"Motion is Z-only, max +{z_step_mm:.1f} mm from current TCP."
        )
        print(
            "[SAFETY] 调试扫描点均在配置的安全工作空间内。"
            f"运动只增加 Z，最大相对当前 TCP +{z_step_mm:.1f} mm。"
        )
        # ----------------------------------------------------
        # 5. Scan each point
        # 5. 逐点扫描：进入扫描循环
        # ----------------------------------------------------

        print("\n" + "-" * 80)
        print("[5/7] Starting scan loop...")
        print("[5/7] 开始逐点扫描...")

        for point_id, scan_pose in enumerate(scan_poses, start=1):
            print("\n" + "=" * 80)
            print(f"[SCAN] Point {point_id}/{len(scan_poses)}")
            print(f"[SCAN] 扫描点 {point_id}/{len(scan_poses)}")
            print("=" * 80)

            # ------------------------------------------------
            # 5.1 Move robot to scan pose
            # 5.1 机器人移动到当前扫描点
            # ------------------------------------------------

            target_pose = scan_pose

            print(f"[ROBOT] Direct robot.ptp() to scan point P{point_id}:")
            print(f"[ROBOT] target_pose = {target_pose}")
            print(f"[ROBOT] speed_percent = {cfg.PTP_SPEED_PERCENT}")
            print(f"[ROBOT] ENABLE_ROBOT_MOVE = {cfg.ENABLE_ROBOT_MOVE}")

            input(
                f"[PAUSE] Press Enter to send robot.ptp() to P{point_id}... "
                f"/ 按回车发送 robot.ptp() 到 P{point_id}..."
            )

            if cfg.ENABLE_ROBOT_MOVE:
                current_before_move = robot.tcp_coord
                current_before_move = [float(v) for v in current_before_move[:6]]

                current_pos = np.array(current_before_move[:3], dtype=np.float64)
                target_pos = np.array(target_pose[:3], dtype=np.float64)
                pre_move_error_mm = float(np.linalg.norm(current_pos - target_pos))

                if pre_move_error_mm <= 2.0:
                    print(
                        "[ROBOT] Already at target within 2.0 mm; "
                        "skip sending zero-distance PTP."
                    )
                    print("[ROBOT] 当前已在目标点 2.0 mm 内，跳过零距离 PTP。")
                    tcp_pose_base = current_before_move
                else:
                    print(f"[ROBOT] Sending robot.ptp() command to P{point_id}...")

                    robot.ptp(
                        target_pose,
                        cfg.PTP_SPEED_PERCENT,
                        data_format="CPP",
                        blending=0,
                        precision_positioning="false",
                    )

                    print(f"[ROBOT] robot.ptp() command to P{point_id} sent.")

                    tcp_pose_base = wait_until_pose_reached(
                        robot,
                        target_pose,
                        position_tolerance_mm=2.0,
                        angle_tolerance_deg=5.0,
                        timeout_s=20.0,
                    )

            else:
                print("[ROBOT] ENABLE_ROBOT_MOVE = False, skip real movement.")
                tcp_pose_base = robot.tcp_coord
                tcp_pose_base = [float(v) for v in tcp_pose_base[:6]]

            time.sleep(SETTLE_SECONDS_AFTER_MOVE)

            # ------------------------------------------------
            # 5.2 Read current real TCP pose
            # 5.2 读取当前真实 TCP pose
            # ------------------------------------------------

            tcp_pose_base = robot.tcp_coord
            tcp_pose_base = [float(v) for v in tcp_pose_base[:6]]

            tcp_poses_base.append(tcp_pose_base)

            print(f"[ROBOT] TCP pose base = {np.round(tcp_pose_base, 3)}")

            # ------------------------------------------------
            # 5.3 Build T_base_camera
            # 5.3 构造 T_base_camera
            # ------------------------------------------------

            T_base_camera = build_camera_to_base_matrix(
                camera_relative_pose_tcp=cfg.CAMERA_RELATIVE_POSE_TCP,
                robot_tcp_pose_base=tcp_pose_base,
            )

            print("[TRANSFORM] T_base_camera ready.")
            print("[TRANSFORM] T_base_camera 已计算。")

            # ------------------------------------------------
            # 5.4 Preview before formal capture
            # 5.4 正式采集前实时预览
            # ------------------------------------------------

            if ENABLE_PREVIEW_BEFORE_CAPTURE:
                preview_camera_until_confirm(
                    zed=zed,
                    detector=detector,
                    point_id=point_id,
                    title=f"Scan point {point_id}",
                    show_mask=False,
                )

            # ------------------------------------------------
            # 5.5 Capture P_camera
            # 5.5 采集 P_camera
            # ------------------------------------------------

            estimate = capture_defect_position_for_seconds(
                zed=zed,
                detector=detector,
                capture_seconds=cfg.CAPTURE_SECONDS_PER_VIEW,
                show_debug=False,
                point_id=point_id,
            )

            print(f"[VISION] success = {estimate.success}")
            print(f"[VISION] reason = {estimate.reason}")
            print(
                f"[VISION] valid frames = "
                f"{estimate.valid_frame_count}/{estimate.total_frame_count}"
            )

            if not estimate.success or estimate.point_camera_m is None:
                print(f"[SKIP] No valid P_camera at scan point {point_id}.")
                print(f"[SKIP] 扫描点 {point_id} 没有有效 P_camera，跳过。")
                continue

            p_camera_m = estimate.point_camera_m.astype(np.float64)

            print(
                "[RESULT] P_camera = "
                f"{np.round(p_camera_m * 1000.0, 2)} mm"
            )

            # ------------------------------------------------
            # 5.6 Convert P_camera to P_global/base
            # 5.6 将 P_camera 转换到机器人 base/global 坐标系
            # ------------------------------------------------

            p_global_m = camera_point_to_global(
                point_camera_m=p_camera_m,
                T_base_camera=T_base_camera,
            )

            global_points_m.append(p_global_m)

            print(
                "[RESULT] P_global/base = "
                f"{np.round(p_global_m * 1000.0, 2)} mm"
            )

        # ----------------------------------------------------
        # 6. Judge global point stability
        # 6. 判断多个 global 点是否稳定
        # ----------------------------------------------------

        print("\n" + "-" * 80)
        print("[6/7] Judging global point stability...")
        print("[6/7] 正在判断 global 点稳定性...")

        print(f"[SUMMARY] Valid global points = {len(global_points_m)}")

        for i, p in enumerate(global_points_m, start=1):
            print(
                f"[SUMMARY] P_global {i} = "
                f"{np.round(p * 1000.0, 2)} mm"
            )

        stable, fused_point_m, reason = judge_global_points_stability(
            global_points_m
        )

        print(f"[DECISION] stable = {stable}")
        print(f"[DECISION] reason = {reason}")

        if fused_point_m is not None:
            print(
                "[DECISION] fused defect point = "
                f"{np.round(fused_point_m * 1000.0, 2)} mm"
            )

        if not stable:
            if BLOCK_IF_NOT_STABLE:
                print("[STOP] Points are not stable. Final marking is blocked.")
                print("[STOP] 点不稳定，阻止最终下笔动作。")
                return
            else:
                print("[WARN] Points are not stable, but BLOCK_IF_NOT_STABLE = False.")
                print("[WARN] 点不稳定，但程序配置允许继续。")

        if fused_point_m is None:
            print("[STOP] No fused point. Cannot continue.")
            print("[STOP] 没有融合点，无法继续。")
            return

        # ----------------------------------------------------
        # 7. Marker compensation and final marking
        # 7. 画笔补偿与最终标记
        # ----------------------------------------------------

        print("\n" + "-" * 80)
        print("[7/7] Marker compensation and final marking...")
        print("[7/7] 画笔补偿与最终标记...")

        final_orientation_deg = scan_poses[-1][3:6]

        # 这里构造一个虚拟 TCP pose，用于计算画笔偏移方向。
        # 位置用 fused point，姿态用最终下笔姿态。
        # marker_compensation 里真正影响 offset 方向的是姿态。
        virtual_tcp_pose_base = [
            float(fused_point_m[0] * 1000.0),
            float(fused_point_m[1] * 1000.0),
            float(fused_point_m[2] * 1000.0),
            float(final_orientation_deg[0]),
            float(final_orientation_deg[1]),
            float(final_orientation_deg[2]),
        ]

        target_tcp_m = compensate_marker_point_global(
            defect_point_global_m=fused_point_m,
            marker_relative_pose_tcp=cfg.MARKER_RELATIVE_POSE_TCP,
            current_tcp_pose_base=virtual_tcp_pose_base,
        )

        target_pose = [
            float(target_tcp_m[0] * 1000.0),
            float(target_tcp_m[1] * 1000.0),
            float(target_tcp_m[2] * 1000.0),
            float(final_orientation_deg[0]),
            float(final_orientation_deg[1]),
            float(final_orientation_deg[2]),
        ]

        print(
            "[TARGET] compensated TCP target pose = "
            f"{np.round(target_pose, 3)}"
        )

        if not RUN_FINAL_MARKING:
            print("[SAFETY] RUN_FINAL_MARKING = False.")
            print("[SAFETY] 本次不执行最终下笔动作。")
            print("[SAFETY] 如果确认坐标正确，在 config.py 中加入：")
            print("         RUN_FINAL_MARKING = True")
            return

        print("[MARK] Executing final vertical marking motion...")
        print("[MARK] 正在执行最终垂直标记动作...")

        execute_vertical_marking_motion(
            robot=robot,
            target_pose=target_pose,
            approach_height_mm=cfg.APPROACH_HEIGHT_MM,
            speed_mm_s=cfg.LINE_SPEED_MM_S,
        )

        print("\n" + "=" * 80)
        print("[MAIN] Workflow finished.")
        print("[MAIN] 主流程完成。")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] KeyboardInterrupt received.")
        print("[INTERRUPT] 用户中断程序。")

    except Exception as e:
        print("\n[ERROR] Main workflow failed.")
        print("[ERROR] 主流程发生错误。")
        print(f"[ERROR] {e}")
        traceback.print_exc()

    finally:
        print("\n[CLEANUP] Closing resources...")
        print("[CLEANUP] 正在关闭资源...")

        if zed is not None:
            try:
                zed.close()
            except Exception as e:
                print(f"[CLEANUP WARN] Failed to close ZED: {e}")

        if robot is not None:
            try:
                robot.close_connection()
            except Exception as e:
                print(f"[CLEANUP WARN] Failed to close robot connection: {e}")

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("[CLEANUP] Done.")
        print("[CLEANUP] 完成。")


if __name__ == "__main__":
    main()
