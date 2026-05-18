import time
import techman as tm

ROBOT_IP = "192.168.19.22"
ETHERNET_TABLE_NAME = "Default"


def main():
    robot = tm.TM_Robot(
        ROBOT_IP,
        table_name=ETHERNET_TABLE_NAME,
    )

    robot.connect_listen_node()

    current = robot.tcp_coord
    current = [float(v) for v in current[:6]]

    print("[TEST] Current TCP:")
    print(current)

    target = current.copy()

    # 只沿 X 方向移动 20 mm。
    # 如果你不确定 X 方向是否安全，可以改成 +10。
    target[0] += 20.0

    print("[TEST] Target TCP:")
    print(target)

    input("Press Enter to send small PTP move... / 按回车发送小范围 PTP 移动...")

    robot.ptp(
        target,
        5,
        data_format="CPP",
        blending=0,
        precision_positioning="false",
    )

    print("[TEST] Command sent.")

    for i in range(30):
        tcp = robot.tcp_coord
        print(f"[TCP {i + 1}]", tcp)
        time.sleep(0.5)

    robot.close_connection()


if __name__ == "__main__":
    main()