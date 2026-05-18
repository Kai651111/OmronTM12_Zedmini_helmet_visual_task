import time
import techman as tm

ROBOT_IP = "192.168.19.22"
ETHERNET_TABLE_NAME = "Default"

def main():
    robot = tm.TM_Robot(ROBOT_IP, table_name=ETHERNET_TABLE_NAME)
    robot.connect_listen_node()

    print("[TEST] Current TCP before move:")
    print(robot.tcp_coord)

    target = [-300.0, 750.0, 700.0, -90.0, 0.0, 90.0]

    print("[TEST] Sending PTP target:")
    print(target)

    robot.ptp(
        target,
        5,
        data_format="CPP",
        blending=0,
        precision_positioning="false",
    )

    for i in range(20):
        print(f"[TEST] tcp_coord {i+1}:", robot.tcp_coord)
        time.sleep(0.5)

    robot.close_connection()

if __name__ == "__main__":
    main()