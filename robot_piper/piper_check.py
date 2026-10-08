#!/usr/bin/env python3
"""
Read-only PiPER connection check: no enable, no motion.
Prints firmware, arm status, joint angles, end-effector (flange) pose and gripper state for a few seconds.

Prerequisite: CAN is up (sudo bash piper_sdk/piper_sdk/can_activate.sh can0 1000000).
Usage:
    ../perception-main/ai_bubble/.venv/bin/python piper_check.py [--can can0] [--seconds 3]
"""
import argparse
import time

from piper_sdk import C_PiperInterface_V2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--can", default="can0")
    ap.add_argument("--seconds", type=float, default=3.0)
    args = ap.parse_args()

    p = C_PiperInterface_V2(args.can)
    p.ConnectPort()
    time.sleep(0.5)                                   # let the reader thread receive a few frames
    print("firmware:", p.GetPiperFirmwareVersion())
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        j = p.GetArmJointMsgs().joint_state
        e = p.GetArmEndPoseMsgs().end_pose
        g = p.GetArmGripperMsgs().gripper_state
        s = p.GetArmStatus().arm_status
        joints = [round(getattr(j, f"joint_{i}") / 1000, 1) for i in range(1, 7)]
        pose = [round(e.X_axis / 1000, 1), round(e.Y_axis / 1000, 1), round(e.Z_axis / 1000, 1),
                round(e.RX_axis / 1000, 1), round(e.RY_axis / 1000, 1), round(e.RZ_axis / 1000, 1)]
        print(f"joints(deg) {joints} | flange(mm,deg) {pose} | gripper {g.grippers_angle / 1000:.1f} mm "
              f"| ctrl_mode {s.ctrl_mode} arm_status {s.arm_status}")
        time.sleep(0.5)
    print("OK - read only, nothing was sent to move the arm")


if __name__ == "__main__":
    main()
