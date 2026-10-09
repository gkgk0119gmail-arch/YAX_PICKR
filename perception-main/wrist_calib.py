"""Wrist camera (D405) -> flange calibration with the ChArUco board still taped where touch_calib.py
measured it. A person moves the arm by hand (teach mode) so the D405 sees the board from different
angles; for each pose: flange pose over CAN (read only) + board pose from the D405 (corner pixels
averaged over frames) + board pose from the D435i (to catch a board that moved).

  conda activate piper_sim && env -u PYTHONPATH python perception-main/wrist_calib.py
  keys: c capture pose   u undo last   w write + quit   q quit without saving
Then solve:  perception-main/ai_bubble/.venv/bin/python perception-main/wrist_solve.py <file>
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from charuco_view import DETECTOR, Cam
from touch_calib import Arm, measure_board

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "pac2026" / "calib"
WIN = "wrist calib (D405 + D435i)"


def main():
    wrist, top = Cam("wrist"), Cam("top")
    arm = Arm()
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1600, 640)
    poses = []
    msg = "move the arm so the D405 sees the whole board, hold still, press c"
    while True:
        wrist.poll()
        top.poll()
        if wrist.img is None or top.img is None:
            continue
        f, j, mode = arm.read()
        tiles = []
        for cam in (wrist, top):
            img = cam.img.copy()
            cc, ci, mc, mi = DETECTOR.detectBoard(img)
            n = 0 if ci is None else len(ci)
            if mi is not None:
                cv2.aruco.drawDetectedMarkers(img, mc, mi)
            if n:
                for (u, v) in np.asarray(cc).reshape(-1, 2):
                    cv2.circle(img, (int(u), int(v)), 5, (0, 255, 0), 2)
            img = cv2.resize(img, (800, 450))
            cv2.putText(img, f"{'D405 wrist' if cam is wrist else 'D435i top'}: {n}/24 corners", (10, 30), 0, 0.9,
                        (0, 255, 0) if n >= 12 else (0, 0, 255), 2)
            tiles.append(img)
        lines = [msg,
                 f"captured {len(poses)} (5-8, change the viewing angle 20-40 deg between poses)   arm {mode}",
                 f"flange x {f[0]:.1f} y {f[1]:.1f} z {f[2]:.1f} mm  rx {f[3]:.1f} ry {f[4]:.1f} rz {f[5]:.1f}",
                 "c capture   u undo   w save+quit   q quit"]
        panel = np.zeros((30 * len(lines) + 14, 1600, 3), np.uint8)
        for k, s in enumerate(lines):
            cv2.putText(panel, s, (12, 28 + 30 * k), 0, 0.8, (0, 255, 255) if k == 0 else (255, 255, 255), 2)
        cv2.imshow(WIN, np.vstack([np.hstack(tiles), panel]))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("u") and poses:
            poses.pop()
            msg = "removed last pose"
        elif key == ord("c"):
            f0, _, spread0 = arm.read_still(0.3)
            bw = measure_board(wrist, frames=15, timeout=4.0)
            bt = measure_board(top, frames=10, timeout=3.0)
            f1, j1, spread1 = arm.read_still(0.3)
            moved = float(np.linalg.norm(np.array(f0[:3]) - np.array(f1[:3])))
            if max(spread0, spread1) > 1.0 or moved > 1.0:
                msg = f"arm moved during capture ({moved:.1f} mm) - hold still, press c again"
            elif bw is None:
                msg = "D405 did not see enough of the board (need 12+ corners) - adjust and retry"
            else:
                poses.append(dict(flange=f1, joints=j1, wrist_board=bw, top_board=bt, t=time.time()))
                t = np.array(bw["T_cam_board"])[:3, 3] * 100
                msg = (f"captured pose {len(poses)}: D405 sees board at {np.linalg.norm(t):.0f} cm, "
                       f"{bw['corners']} corners, reproj {bw['reproj_px']:.2f} px")
        elif key == ord("w"):
            OUT.mkdir(parents=True, exist_ok=True)
            path = OUT / f"wrist_{time.strftime('%Y%m%d_%H%M%S')}.json"
            path.write_text(json.dumps(dict(K=wrist.K.tolist(), dist=wrist.D.tolist(), poses=poses), indent=1))
            print("saved", path, flush=True)
            break
    wrist.pipe.stop()
    top.pipe.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
