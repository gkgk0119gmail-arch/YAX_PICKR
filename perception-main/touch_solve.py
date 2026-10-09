"""Solve the touch calibration recorded by touch_calib.py.

Unknowns: T_base_cam (top D435i -> PiPER base) and the gripper tip offset in the flange frame.
Each touch says: flange_R @ tip + flange_p == T_base_cam @ (board corner in the camera frame).
With touches at different gripper tilts the tip offset is observable; otherwise it is fixed to
--tip (default: config tcp_offset_m along the flange z axis).

  perception-main/ai_bubble/.venv/bin/python perception-main/touch_solve.py pac2026/calib/touch_XXX.json
Writes pac2026/calib/top_cam_to_base.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

HERE = Path(__file__).resolve().parent


def kabsch(A, B):
    """R, t minimising |R A + t - B| (rows are points)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    S = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ S @ U.T
    return R, cb - R @ ca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", type=Path)
    ap.add_argument("--tip", type=float, nargs=3, help="fix the tip offset (m, flange frame) instead of solving it")
    args = ap.parse_args()
    d = json.loads(args.file.read_text())
    T_cb = np.array(d["board"]["T_cam_board"])
    n_in = d["inner_cols"]
    sq = d["square_m"]
    q, Rf, pf, ids = [], [], [], []
    for t in d["touches"]:
        k = t["corner_id"]
        pb = np.array([(k % n_in + 1) * sq, (k // n_in + 1) * sq, 0.0])
        q.append(T_cb[:3, :3] @ pb + T_cb[:3, 3])                       # corner in camera frame
        f = t["flange"]
        Rf.append(Rot.from_euler("xyz", f[3:], degrees=True).as_matrix())  # PiPER: R = Rz*Ry*Rx
        pf.append(np.array(f[:3]) / 1000)
        ids.append(k)
    q, Rf, pf = np.array(q), np.array(Rf), np.array(pf)
    n = len(q)
    if n < 4:
        raise SystemExit(f"only {n} touches, need >= 4 (6+ recommended)")

    tilt_spread = max(np.degrees(np.arccos(np.clip((np.trace(Rf[i].T @ Rf[j]) - 1) / 2, -1, 1)))
                      for i in range(n) for j in range(n))
    solve_tip = args.tip is None and tilt_spread > 15 and n >= 6
    tip0 = np.array(args.tip) if args.tip else np.array([0, 0, 0.14])

    def tips(tip):
        return np.einsum("nij,j->ni", Rf, tip) + pf

    R0, t0 = kabsch(q, tips(tip0))

    def resid(x):
        R = Rot.from_rotvec(x[:3]).as_matrix()
        tip = x[6:9] if solve_tip else tip0
        return ((q @ R.T + x[3:6]) - tips(tip)).ravel()

    x0 = np.r_[Rot.from_matrix(R0).as_rotvec(), t0, tip0 if solve_tip else []]
    sol = least_squares(resid, x0)
    R = Rot.from_rotvec(sol.x[:3]).as_matrix()
    t = sol.x[3:6]
    tip = sol.x[6:9] if solve_tip else tip0
    err = np.linalg.norm(resid(sol.x).reshape(-1, 3), axis=1) * 1000
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t

    print(f"{n} touches, gripper tilt spread {tilt_spread:.0f} deg -> tip offset "
          f"{'SOLVED' if solve_tip else 'fixed'}: {np.round(tip * 1000, 1).tolist()} mm (flange frame)")
    for k, e in zip(ids, err):
        print(f"  corner {k:2d}: {e:5.1f} mm")
    print(f"RMS {np.sqrt(np.mean(err ** 2)):.1f} mm, max {err.max():.1f} mm")
    cam_in_base = t
    print(f"camera position in base frame: {np.round(cam_in_base * 100, 1).tolist()} cm, "
          f"looking along {np.round(R[:, 2], 2).tolist()}")
    out = HERE.parent / "pac2026" / "calib" / "top_cam_to_base.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(T_base_cam=T.tolist(), tip_offset_m=tip.tolist(), tip_solved=bool(solve_tip),
                                   rms_mm=float(np.sqrt(np.mean(err ** 2))), max_mm=float(err.max()),
                                   per_touch_mm=dict(zip(map(str, ids), err.round(2).tolist())),
                                   source=str(args.file)), indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
