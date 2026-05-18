import time
import techman as tm


ROBOT_IP = "192.168.19.22"
ETHERNET_TABLE_NAME = "Default"


def main():
    print("[TEST] Connecting to real TM12...")
    robot = tm.TM_Robot(
        ROBOT_IP,
        table_name=ETHERNET_TABLE_NAME,
    )

    print("[TEST] Connected.")

    for i in range(10):
        tcp_pose = robot.tcp_coord
        print(f"[TEST] tcp_coord {i + 1}: {tcp_pose}")
        time.sleep(0.5)

    robot.close_connection()
    print("[TEST] Done.")


if __name__ == "__main__":
    main()