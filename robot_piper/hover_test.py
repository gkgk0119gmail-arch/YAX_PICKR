"""Calibration check: move the closed gripper tip to a hover point above a ChArUco corner.

Target = board corner (top-camera calibration + the board pose from the touch session) + hover_mm up,
gripper pointing straight down. Two steps: approach 100 mm above (MOVE P), then descend to hover_mm
(MOVE L), both at a low speed. The arm stays enabled and holds the last pose (never disabled here,
it would drop). Ctrl-C sends a damped stop.

  perception-main/ai_bubble/.venv/bin/python robot_piper/hover_test.py --corner 8            # dry run
  perception-main/ai_bubble/.venv/bin/python robot_piper/hover_test.py --corner 8 --go       # moves
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[1]
CALIB = ROOT / "pac2026" / "calib"
DOWN_RPY = (180.0, 0.0, 178.0)          # tool z down; yaw as in the touch session (no wrist flip)


def corner_in_base(k):
    top = json.loads((CALIB / "top_cam_to_base.json").read_text())
    touch = json.loads(Path(top["source"]).read_text())
    n, sq = touch["inner_cols"], touch["square_m"]
    pb = np.array([(k % n + 1) * sq, (k // n + 1) * sq, 0.0, 1.0])
    return (np.array(top["T_base_cam"]) @ np.array(touch["board"]["T_cam_board"]) @ pb)[:3]


def flange_for_tip(tip_xyz, tcp):
    R = Rot.from_euler("xyz", DOWN_RPY, degrees=True)
    return tip_xyz - R.apply([0, 0, tcp])


JOINT_LIM = np.array([[-148, 148], [0, 178], [-168, 0], [-98, 98], [-68, 68], [-165, 165]], float)   # arm limits read 2026-10-09: 150, 0..180, -170..0, 100, 70, 170 (2 deg margin)


# PiPER DH table of the SDK (C_PiperForwardKinematics(0x01), the firmware's offset version)
_DH_A = (0.0, 0.0, 285.03, -21.98, 0.0, 0.0)
_DH_ALPHA = (0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 2, np.pi / 2)
_DH_THETA = (0.0, -np.pi * 172.22 / 180, -102.78 / 180 * np.pi, 0.0, 0.0, 0.0)
_DH_D = (123.0, 0.0, 0.0, 250.75, 0.0, 91.0)
_CA, _SA = np.cos(_DH_ALPHA), np.sin(_DH_ALPHA)


def fk(q_deg):
    """PiPER forward kinematics -> flange (mm) and rotation matrix. Same DH chain as the SDK's CalFK
    (checked equal to 1e-9), about 10x faster: called thousands of times by every IK solve."""
    th = np.radians(np.asarray(q_deg, float)) + _DH_THETA
    ct, st = np.cos(th), np.sin(th)
    T = np.eye(4)
    for i in range(6):
        ca, sa, a, d = _CA[i], _SA[i], _DH_A[i], _DH_D[i]
        L = np.array([[ct[i], -st[i], 0.0, a],
                      [st[i] * ca, ct[i] * ca, -sa, -sa * d],
                      [st[i] * sa, ct[i] * sa, ca, ca * d],
                      [0.0, 0.0, 0.0, 1.0]])
        T = T @ L
    return T[:3, 3].copy(), T[:3, :3].copy()


def fk_sdk(q_deg):
    """The SDK's own forward kinematics (reference for fk)."""
    from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics
    f = C_PiperForwardKinematics(0x01).CalFK([np.radians(v) for v in q_deg])[-1]
    return np.array(f[:3]), Rot.from_euler("xyz", f[3:], degrees=True).as_matrix()


def ik(flange_m, q0):
    """Joint angles (deg) for the flange position + DOWN_RPY, starting from q0, within joint limits."""
    from scipy.optimize import least_squares
    Rt = Rot.from_euler("xyz", DOWN_RPY, degrees=True).as_matrix()
    pt = np.asarray(flange_m) * 1000

    def r(q):
        pos, R = fk(q)
        return np.r_[pos - pt, 200 * Rot.from_matrix(Rt.T @ R).as_rotvec()]   # 1 rad ~ 200 mm

    lo, hi = JOINT_LIM[:, 0], JOINT_LIM[:, 1]
    sol = least_squares(r, np.clip(q0, lo + 1e-3, hi - 1e-3), bounds=(lo, hi))
    res = r(sol.x)
    return sol.x, float(np.linalg.norm(res[:3])), float(np.degrees(np.linalg.norm(res[3:]) / 200))


def path_min_tip_z(q_from, q_to, tcp, n=60):
    """Lowest tip height (mm, base frame) along a straight joint-space interpolation."""
    zs = []
    for a in np.linspace(0, 1, n):
        pos, R = fk((1 - a) * np.asarray(q_from) + a * np.asarray(q_to))
        zs.append((pos + R @ [0, 0, tcp * 1000])[2])
    return float(min(zs))


def move_j(p, q_deg, speed, tol_deg=0.5, timeout=40.0, name=None):
    q = (np.asarray(q_deg) * 1000).round().astype(int)
    t0 = time.time()
    hist = []
    ok = False
    try:
        while time.time() - t0 < timeout:
            p.MotionCtrl_2(0x01, 0x01, speed, 0x00)                    # CAN control, MOVE J
            p.JointCtrl(*[int(v) for v in q])
            _, js, st = read(p)
            js[1] = (js[1] + 180) % 360 - 180
            err = float(np.abs(js - q_deg).max())
            hist.append(js)
            if err < tol_deg and len(hist) > 8 and np.ptp(np.array(hist[-8:]), 0).max() < 0.3:
                ok = True
                return True, err, st
            time.sleep(0.02)
        return False, err, st
    finally:                                                           # also on Ctrl-C: what the arm did
        _log_move(name, q_deg, speed, hist, ok, time.time() - t0)


def _log_move(name, q_deg, speed, hist, ok, secs):
    import runlog
    if runlog.folder() is None or not hist:
        return
    H = np.array(hist)
    rest = np.array([0, 0, 0, 0, 25, 0], float)
    d_rest = np.abs(H - rest).max(1)
    i = int(np.argmin(d_rest))
    runlog.event("move_j", name=name, speed=speed, ok=ok, secs=round(secs, 2), q_start=H[0], q_target=np.asarray(q_deg),
                 q_end=H[-1], max_joint_change=np.ptp(H, 0), traj_10hz=H[::5],
                 closest_to_rest_deg=round(float(d_rest[i]), 1), closest_to_rest_q=H[i],
                 through_rest=bool(d_rest[i] < 15 and d_rest[0] >= 15 and np.abs(np.asarray(q_deg) - rest).max() >= 15))


def read(p):
    e = p.GetArmEndPoseMsgs().end_pose
    j = p.GetArmJointMsgs().joint_state
    st = p.GetArmStatus().arm_status
    f = np.array([e.X_axis, e.Y_axis, e.Z_axis, e.RX_axis, e.RY_axis, e.RZ_axis]) / 1000.0
    js = np.array([j.joint_1, j.joint_2, j.joint_3, j.joint_4, j.joint_5, j.joint_6]) / 1000.0
    return f, js, st


def move(p, flange_m, mode, speed, tol_mm=2.0, timeout=25.0):
    """Resend the target until the reported flange is within tol_mm and has settled."""
    x, y, z = (np.asarray(flange_m) * 1e6).round().astype(int)            # m -> 0.001 mm
    rx, ry, rz = (np.asarray(DOWN_RPY) * 1000).round().astype(int)
    t0 = time.time()
    hist = []
    while time.time() - t0 < timeout:
        p.MotionCtrl_2(0x01, mode, speed, 0x00)
        p.EndPoseCtrl(int(x), int(y), int(z), int(rx), int(ry), int(rz))
        f, _, st = read(p)
        err = float(np.linalg.norm(f[:3] - np.asarray(flange_m) * 1000))
        hist.append(f[:3])
        if err < tol_mm and len(hist) > 25 and np.ptp(np.array(hist[-25:]), 0).max() < 0.5:
            return True, err
        time.sleep(0.02)
    return False, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corner", type=int, default=8)
    ap.add_argument("--hover-mm", type=float, default=30.0)
    ap.add_argument("--approach-mm", type=float, default=0.0,
                    help="optional first stop above the corner; 0 = straight to the hover point (j5 hits its 70 deg "
                         "limit above ~60 mm with the tool vertical)")
    ap.add_argument("--speed", type=int, default=10, help="percent")
    ap.add_argument("--go", action="store_true", help="actually move (default: dry run)")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "pac2026" / "config.yaml").read_text())["robot"]
    tcp = cfg["tcp_offset_m"]

    c = corner_in_base(args.corner)
    hover = c + [0, 0, args.hover_mm / 1000]
    above = c + [0, 0, args.approach_mm / 1000]
    f_above, f_hover = flange_for_tip(above, tcp), flange_for_tip(hover, tcp)
    print(f"corner {args.corner} in base: {np.round(c * 1000, 1).tolist()} mm   (tcp {tcp * 1000:.1f} mm, tool down "
          f"rpy {DOWN_RPY})")
    print(f"  1) approach  tip {np.round(above * 1000, 1).tolist()}  -> flange {np.round(f_above * 1000, 1).tolist()} "
          f"(MOVE P, {args.speed}%)")
    print(f"  2) descend   tip {np.round(hover * 1000, 1).tolist()}  -> flange {np.round(f_hover * 1000, 1).tolist()} "
          f"(MOVE L, {args.speed}%)  = {args.hover_mm:.0f} mm above the corner")
    print(f"  reach check: flange at {np.linalg.norm(f_hover[:2]) * 1000:.0f} mm from the base axis (arm reach 626 mm)")

    from piper_sdk import C_PiperInterface_V2
    p = C_PiperInterface_V2(cfg["can_port"])
    p.ConnectPort()
    time.sleep(0.3)
    f, js, st = read(p)
    print(f"now: flange {np.round(f[:3], 1).tolist()} mm rpy {np.round(f[3:], 1).tolist()}, joints {np.round(js, 1).tolist()}, "
          f"ctrl_mode {st.ctrl_mode}, arm_status {st.arm_status}")
    # inverse kinematics on the PC, seeded with the joints of a real touch of this corner (or any touch)
    touch = json.loads(Path(json.loads((CALIB / "top_cam_to_base.json").read_text())["source"]).read_text())
    seeds = [t["joints"] for t in touch["touches"] if t["corner_id"] == args.corner] or \
            [t["joints"] for t in touch["touches"]]
    q_seed = np.array(seeds[0], float)
    q_seed[1] = (q_seed[1] + 180) % 360 - 180
    q_hover, e2, a2 = ik(f_hover, q_seed)
    if args.approach_mm > 0:
        q_above, e1, a1 = ik(f_above, q_hover)
    else:                                                   # go straight to the hover point
        q_above, e1, a1 = q_hover, e2, a2
    q_now = js.copy()
    q_now[1] = (q_now[1] + 180) % 360 - 180
    z1 = path_min_tip_z(q_now, q_above, tcp)
    z2 = path_min_tip_z(q_above, q_hover, tcp)
    print(f"IK approach joints {np.round(q_above, 1).tolist()} (err {e1:.1f} mm / {a1:.1f} deg)")
    print(f"IK hover    joints {np.round(q_hover, 1).tolist()} (err {e2:.1f} mm / {a2:.1f} deg)")
    print(f"lowest tip height on the way: rest->approach {z1:.0f} mm, approach->hover {z2:.0f} mm "
          f"(board at {c[2] * 1000:.0f} mm)")
    floor1 = c[2] * 1000 + (40 if args.approach_mm > 0 else args.hover_mm - 5)   # lowest allowed tip height
    safe = max(e1, e2) < 2 and max(a1, a2) < 1 and z1 > floor1 and z2 > c[2] * 1000 + args.hover_mm - 5
    if not args.go or not safe:
        print("DRY RUN - nothing sent. Add --go to move." if safe else "UNSAFE plan - not moving.")
        return
    if "TEACH" in str(st.ctrl_mode).upper():
        raise SystemExit("arm is in teaching mode - leave teach mode first")
    if abs(js[1]) > 200:
        raise SystemExit(f"joint 2 reads {js[1]:.1f} deg (wrapped) - not moving")
    try:
        t0 = time.time()
        while not p.EnablePiper():
            if time.time() - t0 > 5:
                raise SystemExit("enable timed out")
            time.sleep(0.01)
        print("enabled")
        ok, err, st = move_j(p, q_above, args.speed)
        f, _, _ = read(p)
        print(f"1) approach {'reached' if ok else 'NOT reached'} (joint error {err:.2f} deg, status {st.arm_status}); "
              f"flange {np.round(f[:3], 1).tolist()}")
        if not ok:
            return
        if args.approach_mm > 0:
            ok, err, st = move_j(p, q_hover, args.speed)
        f, _, _ = read(p)
        err = float(np.linalg.norm(f[:3] - f_hover * 1000))
        print(f"2) hover {'reached' if ok else 'NOT reached'} (flange error {err:.1f} mm). Flange now "
              f"{np.round(f[:3], 1).tolist()} mm. Measure the tip offset from corner {args.corner} with a ruler.")
    except KeyboardInterrupt:
        p.MotionCtrl_1(0x01, 0, 0)
        print("stopped (damped)")


if __name__ == "__main__":
    main()
