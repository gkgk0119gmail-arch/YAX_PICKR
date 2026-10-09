"""Live calibration check: detect the ChArUco board with the top D435i now, then hover the closed gripper
tip above random corners (rest -> corner -> rest ...), pausing at each so a person can check the offset.

  perception-main/ai_bubble/.venv/bin/python robot_piper/board_hover.py rest            # back to the rest pose
  perception-main/ai_bubble/.venv/bin/python robot_piper/board_hover.py tour            # plan only (dry run)
  perception-main/ai_bubble/.venv/bin/python robot_piper/board_hover.py tour --go       # move
Every move is MOVE J to joints solved on the PC (hover_test.ik); the tip height along each joint path is
checked against the board before anything is sent. The arm stays enabled at the end.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "robot_piper"), str(ROOT / "perception-main")]
import hover_test as ht  # noqa: E402

REST_Q = np.array([0.0, 0.0, 0.0, 0.0, 25.0, 0.0])        # folded rest (deg), inside the joint limits
NX, NY, SQ = 6, 4, 0.04


def board_now():
    """Board corners in the robot base frame, measured live with the top camera (pixels averaged)."""
    from charuco_view import Cam
    from touch_calib import measure_board
    cam = Cam("top")
    try:
        time.sleep(0.5)
        b = measure_board(cam, frames=30, timeout=10.0)
        img = None
        for _ in range(50):
            if cam.poll() and cam.img is not None:
                img = cam.img.copy()
                break
            time.sleep(0.02)
        K, D = cam.K, cam.D
    finally:
        cam.pipe.stop()
    if b is None:
        raise SystemExit("board not found by the top camera (need 12+ corners, keep the arm/hands out of view)")
    T_base_cam = np.array(json.loads((ht.CALIB / "top_cam_to_base.json").read_text())["T_base_cam"])
    T_cb = np.array(b["T_cam_board"])
    pb = np.array([[(k % NX + 1) * SQ, (k // NX + 1) * SQ, 0.0, 1.0] for k in range(NX * NY)])
    corners = (T_base_cam @ T_cb @ pb.T).T[:, :3]
    return corners, b, img, T_cb, K, D


def plan_corner(k, corners, q_from, cfg, hover_mm):
    tcp = cfg["tcp_offset_m"]
    c = corners[k]
    f_hover = ht.flange_for_tip(c + [0, 0, hover_mm / 1000], tcp)
    q, e, a = ht.ik(f_hover, q_from if q_from[1] > 30 else np.array([0, 125, -94, 0, 64, 0], float))
    z_in = ht.path_min_tip_z(REST_Q, q, tcp)
    z_out = ht.path_min_tip_z(q, REST_Q, tcp)
    ok = e < 2 and a < 1 and min(z_in, z_out) > c[2] * 1000 + hover_mm - 5
    return dict(corner=k, tip_target_mm=np.round((c + [0, 0, hover_mm / 1000]) * 1000, 1).tolist(),
                q=q, ik_err_mm=e, ik_err_deg=a, path_min_z_mm=min(z_in, z_out), ok=ok)


def connect():
    from piper_sdk import C_PiperInterface_V2
    p = C_PiperInterface_V2("can0")
    p.ConnectPort()
    time.sleep(0.3)
    return p


def enable(p):
    _, js, st = ht.read(p)
    if "TEACH" in str(st.ctrl_mode).upper():
        raise SystemExit("arm is in teaching mode - robot_piper/teach_mode.py off first")
    t0 = time.time()
    while not p.EnablePiper():
        if time.time() - t0 > 5:
            raise SystemExit("enable timed out")
        time.sleep(0.01)


def go_rest(p, speed):
    _, js, _ = ht.read(p)
    js[1] = (js[1] + 180) % 360 - 180
    cfg = yaml.safe_load((ROOT / "pac2026" / "config.yaml").read_text())["robot"]
    z = ht.path_min_tip_z(js, REST_Q, cfg["tcp_offset_m"])
    pos, R = ht.fk(js)
    z_now = (pos + R @ [0, 0, cfg["tcp_offset_m"] * 1000])[2]
    if z < min(z_now, 100) - 5:
        raise SystemExit(f"path to rest dips to tip z {z:.0f} mm (now {z_now:.0f}) - not moving")
    ok, err, st = ht.move_j(p, REST_Q, speed)
    print(f"rest {'reached' if ok else 'NOT reached'} (joint error {err:.2f} deg, {st.arm_status})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rest", "tour"])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--hover-mm", type=float, default=30.0)
    ap.add_argument("--pause", type=float, default=8.0)
    ap.add_argument("--speed", type=int, default=10)
    ap.add_argument("--go", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "pac2026" / "config.yaml").read_text())["robot"]
    p = connect()
    if args.cmd == "rest":
        enable(p)
        go_rest(p, args.speed)
        return

    _, js, st = ht.read(p)
    js[1] = (js[1] + 180) % 360 - 180
    if np.abs(js - REST_Q).max() > 12:
        raise SystemExit(f"arm is not at rest ({np.round(js, 1).tolist()}) - run 'rest' first (it hides the board)")
    corners, b, img, T_cb, K, D = board_now()
    print(f"board (live): {b['corners']} corners, reproj {b['reproj_px']:.2f} px; centre at "
          f"{np.round(corners.mean(0) * 1000, 1).tolist()} mm in base")
    rng = np.random.default_rng(args.seed)
    plans = []
    for k in rng.permutation(NX * NY):
        pl = plan_corner(int(k), corners, js, cfg, args.hover_mm)
        if pl["ok"]:
            plans.append(pl)
        if len(plans) == args.n:
            break
    for i, pl in enumerate(plans, 1):
        print(f"  {i}) corner {pl['corner']:2d}: tip -> {pl['tip_target_mm']} mm, joints {np.round(pl['q'], 1).tolist()}, "
              f"lowest tip on the way {pl['path_min_z_mm']:.0f} mm")
    out = ROOT / "pac2026" / "calib" / f"board_hover_{time.strftime('%H%M%S')}.png"
    if img is not None:
        vis = img.copy()
        for i, pl in enumerate(plans, 1):
            pc = T_cb[:3, :3] @ np.array([(pl["corner"] % NX + 1) * SQ, (pl["corner"] // NX + 1) * SQ, 0]) + T_cb[:3, 3]
            uv, _ = cv2.projectPoints(pc.reshape(1, 3), np.zeros(3), np.zeros(3), K, D)
            u, v = uv.ravel().astype(int)
            cv2.circle(vis, (u, v), 14, (0, 0, 255), 3)
            cv2.putText(vis, f"{i}: corner {pl['corner']}", (u + 16, v - 10), 0, 0.8, (0, 0, 255), 2)
        cv2.imwrite(str(out), vis)
        print("targets drawn:", out)
    if len(plans) < args.n:
        print(f"only {len(plans)} reachable corners")
    if not args.go:
        print("DRY RUN - nothing sent. Add --go to move.")
        return
    enable(p)
    try:
        for i, pl in enumerate(plans, 1):
            ok, err, st = ht.move_j(p, pl["q"], args.speed)
            f, _, _ = ht.read(p)
            print(f"{i}) corner {pl['corner']}: {'reached' if ok else 'NOT reached'} ({st.arm_status}), flange "
                  f"{np.round(f[:3], 1).tolist()} - holding {args.pause:.0f} s", flush=True)
            if not ok:
                break
            time.sleep(args.pause)
            if not go_rest(p, args.speed):
                break
    except KeyboardInterrupt:
        p.MotionCtrl_1(0x01, 0, 0)
        print("stopped (damped)")


if __name__ == "__main__":
    main()
