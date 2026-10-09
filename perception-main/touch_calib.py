"""Top camera (D435i) -> robot base calibration by touching ChArUco corners with the gripper tip.

The board is taped to the table. The D435i measures the board pose once (hand out of view); then a
person moves the arm by hand (teach mode) and touches the highlighted corners with the closed
gripper tip. For each touch the PiPER flange pose is read over CAN (read only, nothing is sent to
the arm). Touch with different gripper tilts: the solver then also finds the flange-to-tip offset.

  conda activate piper_sim && env -u PYTHONPATH python perception-main/touch_calib.py
  keys: c capture touch   n skip to next corner   u undo last   b re-measure board (hand away)
        w write + quit    q quit without saving
Then solve:  perception-main/ai_bubble/.venv/bin/python perception-main/touch_solve.py <file>
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from charuco_view import BOARD, DETECTOR, SQUARE, SQUARES, Cam

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "pac2026" / "calib"
NX = SQUARES[0] - 1                                   # inner corners per row (6)
TARGETS = [0, 5, 23, 18, 8, 15, 2, 21, 12, 11]        # spread over the board
WIN = "touch calib (D435i)"


def corner_board_xyz(k):
    return np.array([(k % NX + 1) * SQUARE, (k // NX + 1) * SQUARE, 0.0])


def measure_board(cam, frames=30, timeout=10.0):
    """Average the detected corner pixels over many frames (board is static), then one PnP."""
    acc = {}
    t0 = time.time()
    n = 0
    while n < frames and time.time() - t0 < timeout:
        if not cam.poll() or cam.img is None:
            time.sleep(0.005)
            continue
        cc, ci, _, _ = DETECTOR.detectBoard(cam.img)
        if ci is None or len(ci) < 12:
            continue
        for (u, v), k in zip(np.asarray(cc).reshape(-1, 2), np.asarray(ci).reshape(-1)):
            acc.setdefault(int(k), []).append((u, v))
        n += 1
    ids = sorted(k for k, v in acc.items() if len(v) >= max(3, n // 2))
    if n < 5 or len(ids) < 12:
        return None
    img_pts = np.array([np.mean(acc[k], 0) for k in ids], np.float64)
    obj = np.array([corner_board_xyz(k) for k in ids], np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img_pts, cam.K, cam.D)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, cam.K, cam.D)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img_pts) ** 2, -1))))
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, tvec.ravel()
    return dict(T_cam_board=T.tolist(), corners=len(ids), frames=n, reproj_px=err)


class Arm:
    """Read-only PiPER feedback."""

    def __init__(self, can="can0"):
        from piper_sdk import C_PiperInterface_V2
        self.p = C_PiperInterface_V2(can)
        self.p.ConnectPort()
        time.sleep(0.3)

    def read(self):
        e = self.p.GetArmEndPoseMsgs().end_pose
        j = self.p.GetArmJointMsgs().joint_state
        s = self.p.GetArmStatus().arm_status
        flange = [e.X_axis / 1000, e.Y_axis / 1000, e.Z_axis / 1000, e.RX_axis / 1000, e.RY_axis / 1000,
                  e.RZ_axis / 1000]                     # mm, deg (R = Rz*Ry*Rx)
        joints = [j.joint_1 / 1000, j.joint_2 / 1000, j.joint_3 / 1000, j.joint_4 / 1000, j.joint_5 / 1000,
                  j.joint_6 / 1000]
        return flange, joints, str(s.ctrl_mode)

    def read_still(self, secs=0.5):
        """Average over secs; also return the spread so a moving arm is not recorded."""
        F, J = [], []
        t0 = time.time()
        while time.time() - t0 < secs:
            f, j, _ = self.read()
            F.append(f)
            J.append(j)
            time.sleep(0.02)
        F = np.array(F)
        return F.mean(0).tolist(), np.array(J).mean(0).tolist(), float(np.ptp(F[:, :3], 0).max())


def main():
    cam = Cam("top")
    arm = Arm()
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 820)
    board = None
    touches = []
    ti = 0
    msg = "measuring board — keep hands out of view"
    need_board = True
    while True:
        if need_board:
            frame = np.zeros((720, 1280, 3), np.uint8)
            cv2.putText(frame, "measuring board pose (hands out of view)...", (40, 360), 0, 1.2, (0, 255, 255), 2)
            cv2.imshow(WIN, frame)
            cv2.waitKey(1)
            b = measure_board(cam)
            if b:
                board = b
                msg = f"board: {b['corners']} corners, {b['frames']} frames, reproj {b['reproj_px']:.2f} px"
            else:
                msg = "board NOT measured - clear the view and press b"
            need_board = False
        cam.poll()
        if cam.img is None:
            continue
        img = cam.img.copy()
        f, j, mode = arm.read()
        target = TARGETS[ti % len(TARGETS)]
        if board:
            T = np.array(board["T_cam_board"])
            for k in range(NX * (SQUARES[1] - 1)):
                p = T[:3, :3] @ corner_board_xyz(k) + T[:3, 3]
                uv, _ = cv2.projectPoints(p.reshape(1, 3), np.zeros(3), np.zeros(3), cam.K, cam.D)
                u, v = uv.ravel().astype(int)
                done = any(t["corner_id"] == k for t in touches)
                if k == target:
                    cv2.circle(img, (u, v), 16, (0, 0, 255), 3)
                    cv2.circle(img, (u, v), 3, (0, 0, 255), -1)
                cv2.circle(img, (u, v), 4, (0, 200, 0) if done else (255, 255, 0), -1)
                cv2.putText(img, str(k), (u + 6, v - 6), 0, 0.5, (0, 200, 0) if done else (255, 255, 0), 1)
        lines = [f"touch corner {target} (red circle) with the closed gripper tip, vary the tilt, then press c",
                 f"captured {len(touches)}  (need >= 6, 8-10 better)   arm mode {mode}",
                 f"flange  x {f[0]:.1f} y {f[1]:.1f} z {f[2]:.1f} mm  rx {f[3]:.1f} ry {f[4]:.1f} rz {f[5]:.1f}",
                 msg,
                 "c capture  n next corner  u undo  b re-measure board  w save+quit  q quit"]
        panel = np.zeros((32 * len(lines) + 16, img.shape[1], 3), np.uint8)
        for k, s in enumerate(lines):
            cv2.putText(panel, s, (12, 30 + 32 * k), 0, 0.8, (0, 255, 255) if k == 0 else (255, 255, 255), 2)
        cv2.imshow(WIN, np.vstack([img, panel]))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b"):
            need_board = True
        elif key == ord("n"):
            ti += 1
        elif key == ord("u") and touches:
            t = touches.pop()
            msg = f"removed touch on corner {t['corner_id']}"
        elif key == ord("c"):
            if not board:
                msg = "measure the board first (b)"
                continue
            f, j, spread = arm.read_still()
            if spread > 1.0:
                msg = f"arm moving ({spread:.1f} mm in 0.5 s) - hold still and press c again"
                continue
            touches.append(dict(corner_id=target, flange=f, joints=j, t=time.time()))
            msg = f"captured corner {target}: flange ({f[0]:.1f}, {f[1]:.1f}, {f[2]:.1f}) mm, tilt rx {f[3]:.0f} ry {f[4]:.0f}"
            ti += 1
        elif key == ord("w"):
            OUT.mkdir(parents=True, exist_ok=True)
            path = OUT / f"touch_{time.strftime('%Y%m%d_%H%M%S')}.json"
            path.write_text(json.dumps(dict(board=board, K=cam.K.tolist(), dist=cam.D.tolist(),
                                            square_m=SQUARE, inner_cols=NX, touches=touches), indent=1))
            print("saved", path, flush=True)
            break
    cam.pipe.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
