"""Turn PiPER drag-teach mode on/off from the PC (instead of the arm's teach button).

  perception-main/ai_bubble/.venv/bin/python robot_piper/teach_mode.py status
  perception-main/ai_bubble/.venv/bin/python robot_piper/teach_mode.py on        # arm can be moved by hand
  perception-main/ai_bubble/.venv/bin/python robot_piper/teach_mode.py off       # back to standby (position control)

"off" = end teach record (0x02) + reset (MotionCtrl_1 0x02), as the SDK requires when leaving teach
mode. The reset cuts motor power for a moment: the arm DROPS. So "off" only runs when the arm is folded
in its rest pose (or with --force while someone holds the arm). Re-enable happens in the motion script.
Never send 0x03 (replay the recorded teach trajectory).
"""
import argparse
import time

import numpy as np
from piper_sdk import C_PiperInterface_V2

REST_JOINTS = np.array([0.0, 0.0, 0.0, 0.0, 25.0, 0.0])     # folded rest pose (deg), as measured on site
REST_TOL_DEG = 12.0


def state(p):
    s = p.GetArmStatus().arm_status
    j = p.GetArmJointMsgs().joint_state
    js = np.array([j.joint_1, j.joint_2, j.joint_3, j.joint_4, j.joint_5, j.joint_6]) / 1000.0
    js[1] = (js[1] + 180) % 360 - 180          # joint 2 sometimes reads a full turn off in teach mode
    return s, js


def show(p, tag=""):
    s, js = state(p)
    rest = np.abs(js - REST_JOINTS).max() < REST_TOL_DEG
    print(f"{tag}ctrl_mode {s.ctrl_mode} | teach_status {s.teach_status} | arm_status {s.arm_status} | "
          f"joints {np.round(js, 1).tolist()} | {'at REST pose' if rest else 'NOT at rest pose'}")
    return s, js, rest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "on", "off"])
    ap.add_argument("--can", default="can0")
    ap.add_argument("--force", action="store_true", help="off: allow the reset away from the rest pose (hold the arm!)")
    args = ap.parse_args()
    p = C_PiperInterface_V2(args.can)
    p.ConnectPort()
    time.sleep(0.4)
    s, js, rest = show(p, "now:   ")
    if args.cmd == "status":
        return
    if args.cmd == "on":
        for _ in range(5):
            p.MotionCtrl_1(0x00, 0x00, 0x01)          # start teach record = drag-teach mode
            time.sleep(0.05)
    else:
        if not rest and not args.force:
            raise SystemExit("not at the rest pose - the reset would drop the arm. Move it to rest (teach mode) "
                             "first, or hold the arm and use --force.")
        for _ in range(5):
            p.MotionCtrl_1(0x00, 0x00, 0x02)          # end teach record
            time.sleep(0.05)
        time.sleep(0.3)
        for _ in range(3):
            p.MotionCtrl_1(0x02, 0x00, 0x00)          # reset (required to leave teach mode); arm power drops
            time.sleep(0.05)
    time.sleep(1.0)
    show(p, "after: ")


if __name__ == "__main__":
    main()
