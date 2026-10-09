"""Solve the wrist (D405) -> flange calibration recorded by wrist_calib.py.

Two estimates, compared:
  A) via the top camera: board in base = T_base_cam(top) @ T_top_board, so for every pose
     X = inv(T_base_flange) @ T_base_board @ inv(T_wrist_board); averaged over poses.
  B) hand-eye on its own (AX = ZB): unknowns X (flange -> D405) and Z (base -> board), least squares
     over the board corners of all poses; does not use the top camera at all.
If A and B agree, the top-camera calibration is confirmed too. B is saved (A if B is ill-posed).

  perception-main/ai_bubble/.venv/bin/python perception-main/wrist_solve.py pac2026/calib/wrist_XXX.json
Writes pac2026/calib/wrist_cam_to_flange.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent
CALIB = HERE.parent / "pac2026" / "calib"
SQ, NX, NY = 0.04, 6, 4
P_BOARD = np.array([[(k % NX + 1) * SQ, (k // NX + 1) * SQ, 0.0] for k in range(NX * NY)])


def T_from_flange(f):
    T = np.eye(4)
    T[:3, :3] = Rot.from_euler("xyz", f[3:], degrees=True).as_matrix()    # PiPER: R = Rz*Ry*Rx
    T[:3, 3] = np.array(f[:3]) / 1000
    return T


def to_x(T):
    return np.r_[Rot.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def from_x(x):
    T = np.eye(4)
    T[:3, :3] = Rot.from_rotvec(x[:3]).as_matrix()
    T[:3, 3] = x[3:6]
    return T


def mean_T(Ts):
    T = np.eye(4)
    T[:3, :3] = Rot.from_matrix([t[:3, :3] for t in Ts]).mean().as_matrix()
    T[:3, 3] = np.mean([t[:3, 3] for t in Ts], 0)
    return T


def describe(X):
    return (f"D405 at {np.round(X[:3, 3] * 1000, 1).tolist()} mm in the flange frame, "
            f"looking along {np.round(X[:3, 2], 2).tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=Path)
    args = ap.parse_args()
    d = json.loads(args.file.read_text())
    top = json.loads((CALIB / "top_cam_to_base.json").read_text())
    T_base_top = np.array(top["T_base_cam"])
    poses = d["poses"]
    n = len(poses)
    A = [T_from_flange(p["flange"]) for p in poses]                       # base <- flange
    B = [np.array(p["wrist_board"]["T_cam_board"]) for p in poses]        # wrist cam <- board

    # board still where it was? (top camera re-measured at each capture vs the touch session)
    touch = json.loads(Path(top["source"]).read_text())
    T_top_board0 = np.array(touch["board"]["T_cam_board"])
    shifts = [np.linalg.norm(np.array(p["top_board"]["T_cam_board"])[:3, 3] - T_top_board0[:3, 3]) * 1000
              for p in poses if p.get("top_board")]
    if shifts:
        print(f"board vs touch session (top camera): max shift {max(shifts):.1f} mm")
    Z_top = T_base_top @ T_top_board0                                     # base <- board, from the top camera

    # A) through the top-camera calibration
    XA = mean_T([np.linalg.inv(a) @ Z_top @ np.linalg.inv(b) for a, b in zip(A, B)])

    # B) AX = ZB alone, least squares on board corners
    def resid(x, fixZ=None):
        X = from_x(x[:6])
        Z = fixZ if fixZ is not None else from_x(x[6:12])
        r = []
        for a, b in zip(A, B):
            p_cam = (b[:3, :3] @ P_BOARD.T).T + b[:3, 3]
            via_robot = ((a @ X)[:3, :3] @ p_cam.T).T + (a @ X)[:3, 3]
            via_board = (Z[:3, :3] @ P_BOARD.T).T + Z[:3, 3]
            r.append(via_robot - via_board)
        return np.concatenate(r).ravel()

    rot_spread = max(np.degrees(np.linalg.norm(Rot.from_matrix(A[i][:3, :3].T @ A[j][:3, :3]).as_rotvec()))
                     for i in range(n) for j in range(n))
    print(f"{n} poses, flange rotation spread {rot_spread:.0f} deg")
    errA = np.linalg.norm(resid(to_x(XA), fixZ=Z_top).reshape(-1, 3), axis=1) * 1000
    print(f"A (via top camera): {describe(XA)}; board corner error RMS {np.sqrt(np.mean(errA ** 2)):.1f} mm")

    XB, okB = XA, False
    if n >= 3 and rot_spread > 10:
        sol = least_squares(resid, np.r_[to_x(XA), to_x(Z_top)])
        XB, ZB = from_x(sol.x[:6]), from_x(sol.x[6:12])
        errB = np.linalg.norm(resid(sol.x).reshape(-1, 3), axis=1) * 1000
        okB = True
        print(f"B (hand-eye only):  {describe(XB)}; board corner error RMS {np.sqrt(np.mean(errB ** 2)):.1f} mm")
        dX = np.linalg.norm(XA[:3, 3] - XB[:3, 3]) * 1000
        dR = np.degrees(np.linalg.norm(Rot.from_matrix(XA[:3, :3].T @ XB[:3, :3]).as_rotvec()))
        dZ = np.linalg.norm(Z_top[:3, 3] - ZB[:3, 3]) * 1000
        print(f"A vs B: camera {dX:.1f} mm / {dR:.1f} deg apart; board position from top camera vs hand-eye "
              f"{dZ:.1f} mm apart (= check of the top-camera calibration)")
    else:
        print("B skipped: need >= 3 poses with > 10 deg rotation between them")
    X = XB if okB else XA
    out = CALIB / "wrist_cam_to_flange.json"
    out.write_text(json.dumps(dict(T_flange_cam=X.tolist(), method="hand-eye (B)" if okB else "via top camera (A)",
                                   T_flange_cam_via_top=XA.tolist(), poses=n, source=str(args.file)), indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
