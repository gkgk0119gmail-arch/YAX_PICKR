"""Pick one block (top D435i: Grounding DINO + SAM2 + top-face rectangle) and put it down at a random
free spot on the table. Everything is MOVE J to joints solved on the PC (arm_ik.ik_tip, tool tilted only
as much as the wrist limits need), and every joint path is checked before it is sent:
the tool tip (and the carried block) must stay above the other blocks and the table.

  perception-main/ai_bubble/.venv/bin/python robot_piper/pick_place.py              # plan only
  perception-main/ai_bubble/.venv/bin/python robot_piper/pick_place.py --go         # move
The arm must start at the rest pose (it would hide the blocks). Ends at rest, enabled.
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
import arm_ik  # noqa: E402
import hover_test as ht  # noqa: E402
from board_hover import REST_Q, connect, enable  # noqa: E402

RELEASE_MM = 100.0              # open fully to let go (stroke 100 mm)
PLACE_DROP = 0.020              # let go with the block bottom this far above the table: going lower presses the
                                # bubble wrap into the open fingers and it stays pinched (seen 2026-10-09)
GRASP_ABOVE_TABLE = 0.045       # tip height above the table when closing on a block
CARRY_CLEAR = 0.020             # carried block bottom above other block tops
TIP_TO_BLOCK_BOTTOM = GRASP_ABOVE_TABLE
BLOCK_HALF_DIAG = 0.06


def yaw_for_jaw(jaw_deg):
    """Tool yaw whose jaws close along jaw_deg (tool y = (sin yaw, -cos yaw)); cube: 90 deg symmetric,
    pick the one closest to 180 (wrist mid-range)."""
    base = jaw_deg + 90.0
    cands = [base + k * 90.0 for k in range(-4, 5)]
    return min(cands, key=lambda y: abs(((y - 180) + 180) % 360 - 180)) % 360


def ik_any(xy, z, yaws, tcp, seed=None):
    """Best IK over several tool yaws (a square block looks the same every 90 deg)."""
    best = None
    for yaw in yaws:
        q, beta, pe, re = arm_ik.ik_tip([xy[0], xy[1], z], yaw, tcp, q_seed=seed)
        key = pe + 5 * re + 0.02 * abs(beta)
        if best is None or key < best[0]:
            best = (key, q, beta, pe, re, yaw)
    return best[1:]


def path_ok(q_a, q_b, tcp, table_z, others, carrying, top_z, n=40, end_relax=False):
    """Check tip / carried block along a joint path. Returns (ok, worst_clearance_m, reason)."""
    worst, why = 1.0, ""
    for i, a in enumerate(np.linspace(0, 1, n)):
        tip, _ = arm_ik.tip_of((1 - a) * np.asarray(q_a) + a * np.asarray(q_b), tcp)
        low = tip[2] - (TIP_TO_BLOCK_BOTTOM if carrying else 0.0)
        relax = end_relax and a > 0.999
        c_table = low - table_z
        if c_table < worst:
            worst, why = c_table, "table"
        for o in others:
            near = np.hypot(*(tip[:2] - o[:2])) < (2 * BLOCK_HALF_DIAG if carrying else BLOCK_HALF_DIAG)
            if near and not relax:
                c = low - (o[2] + (CARRY_CLEAR if carrying else 0.010))
                if c < worst:
                    worst, why = c, "block"
    # -3 mm: a held block starts / ends resting on the table (lift, place), numerically ~0 clearance
    return worst > (-0.003 if not end_relax else -0.06), worst, why


class WristSeen:
    """D405 (wrist) view: a held block sits ~6-10 cm in front of it (calibrated mount), the table is
    further. holding() = most of the centre of the view is closer than NEAR_M."""
    NEAR_M = 0.12

    def __init__(self):
        from charuco_view import Cam
        self.cam = Cam("wrist")
        self.last = None

    def holding(self, frames=8):
        fr = []
        t0 = time.time()
        while len(fr) < frames and time.time() - t0 < 3:
            fs = self.cam.pipe.poll_for_frames()
            if fs:
                d = np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32) * self.cam.scale
                fr.append(d)
            time.sleep(0.01)
        if not fr:
            return None, None
        d = np.median(np.array(fr), 0)
        h, w = d.shape
        c = d[int(0.25 * h):int(0.75 * h), int(0.3 * w):int(0.7 * w)]
        valid = c > 0.02
        near = float(((c < self.NEAR_M) & valid).sum() / max(valid.sum(), 1))
        self.last = float(np.median(c[valid])) if valid.any() else None
        return near > 0.4, near

    def close(self):
        self.cam.pipe.stop()


def gripper(p, width_mm, effort, secs=1.2):
    t0 = time.time()
    while time.time() - t0 < secs:
        p.GripperCtrl(int(round(width_mm * 1000)), effort, 0x01, 0)
        time.sleep(0.02)
    return p.GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--speed", type=int, default=10)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--place-dy", type=float, help="put it down this far along base +y (robot's left, m) "
                                                    "instead of a random spot; tries +-3 cm around it")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "pac2026" / "config.yaml").read_text())["robot"]
    tcp, effort, open_mm = cfg["tcp_offset_m"], int(cfg["gripper_effort"]), float(cfg["gripper_open_mm"])
    p = connect()
    _, js, _ = ht.read(p)
    js[1] = (js[1] + 180) % 360 - 180
    if np.abs(js - REST_Q).max() > 12:
        raise SystemExit("arm is not at rest - run robot_piper/board_hover.py rest first")

    # ------------------------------------------------------------ perception
    from block_grasp import BlockFinder, draw
    sys.path.insert(0, str(ROOT / "perception-main" / "research" / "box_cells"))
    from gdino_vggt_lines import grab_top
    color, depth, K = grab_top()
    finder = BlockFinder()
    objs = finder(color, depth, K)
    run_dir = ROOT / "pac2026" / "runs" / f"pick_place_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(run_dir / "color.jpg"), color)                  # keep the frame + per-stage pictures
    np.save(run_dir / "rs_depth_m.npy", depth.astype(np.float32))
    (run_dir / "intrinsics.json").write_text(json.dumps(K))
    from block_stages import stage_images
    for name, im in zip(("1_gdino", "2_sam2", "3_ransac"), stage_images(color, objs, K, finder.T)):
        cv2.imwrite(str(run_dir / f"{name}.png"), im)
    blocks = [o for o in objs if o["label"] == "block" and o.get("robot") and not o.get("in_container")]
    if not blocks:
        raise SystemExit("no block found")
    cand = [b for b in blocks if 0.18 < b["robot"]["reach_m"] < 0.45 and b["robot"]["fits_gripper"]]
    if not cand:
        raise SystemExit("no block within 18-45 cm that fits the gripper")
    tgt = max(cand, key=lambda b: b["score"]["block"])
    r = tgt["robot"]
    table_z = float(r["table_z_m"])        # table right around the target block (RealSense ring); other blocks
                                           # may stand next to bags / hands and give a wrong table height
    H = r["height_m"]
    others = [np.array(b["robot"]["top_xyz_m"]) for b in blocks if b is not tgt]
    pick_xy = np.array(r["top_xyz_m"][:2])
    yaw_p = yaw_for_jaw(r["jaw_yaw_deg"])
    z_grasp = table_z + GRASP_ABOVE_TABLE
    z_carry = table_z + H + 0.025          # tip 2.5 cm above the block tops; path checks catch crossings
    yaws_p = sorted({(yaw_p + k * 90) % 360 for k in range(4)} & set(
        y for y in ((yaw_p + k * 90) % 360 for k in range(4)) if 60 <= y <= 300), key=lambda y: abs(y - 180))

    # ------------------------------------------------------------ plan
    rng = np.random.default_rng(args.seed)
    plan = None
    qs_pick = {}
    tgt_top = np.array(r["top_xyz_m"])
    plan_yaw_p = yaw_p
    tries = ([pick_xy + [0, args.place_dy + d] for d in (0, -0.03, 0.03, -0.06, 0.06)] if args.place_dy is not None
             else [None] * 300)
    for fixed in tries:
        if fixed is None:
            rad, ang = rng.uniform(0.24, 0.40), np.radians(rng.uniform(-40, 40))
            xy = np.array([rad * np.cos(ang), rad * np.sin(ang)])
        else:
            xy = np.asarray(fixed)
        if min([np.hypot(*(xy - o[:2])) for o in others] + [1]) < 0.13 or np.hypot(*(xy - pick_xy)) < 0.10:
            continue
        yaw_l = rng.uniform(150, 210) if fixed is None else plan_yaw_p
        qs = {}
        seed = REST_Q
        ok = True
        if "above_pick" not in qs_pick:                   # the pick side does not depend on the place spot
            q1, b1, pe1, re1, yaw1 = ik_any(pick_xy, z_carry, yaws_p, tcp)
            q2, b2, pe2, re2, _ = ik_any(pick_xy, z_grasp, [yaw1], tcp, seed=q1)
            if max(pe1, pe2) > 2.0 or max(re1, re2) > 1.0:
                raise SystemExit(f"block not reachable with the gripper down (IK error {max(pe1, pe2):.1f} mm)")
            qs_pick.update(above_pick=(q1, b1), grasp=(q2, b2))
        qs.update(qs_pick)
        for name, xyv, z in (("above_place", xy, z_carry), ("place", xy, z_grasp + PLACE_DROP)):
            q, beta, pe, re, _ = ik_any(xyv, z, [yaw_l], tcp, seed=seed if name == "place" else None)
            if pe > 2.0 or re > 1.0:
                ok = False
                break
            qs[name] = (q, beta)
            seed = q
        if not ok:
            continue
        legs = [("rest -> above pick", REST_Q, qs["above_pick"][0], False, False),
                ("descend to grasp", qs["above_pick"][0], qs["grasp"][0], False, False),
                ("lift", qs["grasp"][0], qs["above_pick"][0], True, False),
                ("carry", qs["above_pick"][0], qs["above_place"][0], True, False),
                ("lower to place", qs["above_place"][0], qs["place"][0], True, True),
                ("rise", qs["place"][0], qs["above_place"][0], False, False),
                ("back to rest", qs["above_place"][0], REST_Q, False, False)]
        checks = []
        for name, a, b, carrying, relax in legs:
            # the target block is an obstacle too, except while going down to it / lifting it
            others_here = others + ([tgt_top] if name in ("rest -> above pick",) else [])
            okp, worst, why = path_ok(a, b, tcp, table_z, others_here, carrying, None, end_relax=relax)
            checks.append((name, okp, worst, why))
        if all(c[1] for c in checks):
            plan = dict(place_xy=xy, yaw_l=yaw_l, qs=qs, legs=legs, checks=checks)
            break
    print(f"target block: score {tgt['score']['block']:.2f}, top {np.round(np.array(r['top_xyz_m']) * 1000).tolist()} mm, "
          f"{r['size_m'][0] * 1000:.0f}x{r['size_m'][1] * 1000:.0f} mm, h {H * 1000:.0f} mm, jaw {r['jaw_yaw_deg']:.0f} deg "
          f"-> tool yaw {yaw_p:.0f}; table z {table_z * 1000:.0f} mm; {len(others)} other blocks")
    if plan is None:
        raise SystemExit("no collision-free place found")
    xy = plan["place_xy"]
    print(f"place at ({xy[0] * 1000:.0f}, {xy[1] * 1000:.0f}) mm, yaw {plan['yaw_l']:.0f}; grasp tip z {z_grasp * 1000:.0f}, "
          f"carry tip z {z_carry * 1000:.0f} mm")
    for k, (q, beta) in plan["qs"].items():
        print(f"  {k:12s} joints {np.round(q, 1).tolist()}  tool tilt {beta:+.0f} deg")
    for name, okp, worst, why in plan["checks"]:
        print(f"  path {name:20s} {'ok' if okp else 'FAIL'}  min clearance {worst * 1000:5.0f} mm ({why})")

    # picture of the plan
    T = np.array(json.loads((ht.CALIB / "top_cam_to_base.json").read_text())["T_base_cam"])
    Ti = np.linalg.inv(T)

    def px(P):
        pc = Ti[:3, :3] @ np.asarray(P) + Ti[:3, 3]
        return int(pc[0] / pc[2] * K["fx"] + K["cx"]), int(pc[1] / pc[2] * K["fy"] + K["cy"])
    vis = draw(color, objs)
    a, b = px([*pick_xy, table_z + H]), px([*xy, table_z])
    cv2.arrowedLine(vis, a, b, (255, 0, 0), 3, tipLength=0.08)
    cv2.circle(vis, b, 40, (255, 0, 0), 3)
    cv2.putText(vis, "PLACE", (b[0] - 30, b[1] - 48), 0, 0.8, (255, 0, 0), 2)
    out = run_dir / "4_plan.png"
    cv2.imwrite(str(out), vis)
    print("plan picture:", out)
    if not args.go:
        print("DRY RUN - nothing sent. Add --go to move.")
        return

    # ------------------------------------------------------------ execute
    qs = plan["qs"]
    try:
        wrist = WristSeen()
        h0, n0 = wrist.holding()
        print(f"wrist camera check before pick: near fraction {n0:.2f} (should be low)")
    except Exception as e:                                   # the check is a help, not required
        print(f"wrist camera unavailable ({e}) - release is not verified")
        wrist = None
    enable(p)
    try:
        def mv(name, q):
            ok, err, st = ht.move_j(p, q, args.speed)
            print(f"{name}: {'ok' if ok else 'FAILED'} ({st.arm_status})", flush=True)
            if not ok:
                raise RuntimeError(name)

        mv("above pick", qs["above_pick"][0])
        w = gripper(p, open_mm, effort)
        print(f"gripper open {w:.0f} mm")
        mv("grasp", qs["grasp"][0])
        w = gripper(p, 0, effort, secs=2.0)
        print(f"gripper closed at {w:.0f} mm")
        if w < 60:
            print("no block between the fingers - aborting")
            gripper(p, open_mm, effort)
            mv("lift", qs["above_pick"][0])
            mv("rest", REST_Q)
            return
        mv("lift", qs["above_pick"][0])
        w = p.GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0
        seen = wrist.holding() if wrist else (None, None)
        print(f"holding check after lift: gripper {w:.0f} mm, wrist camera near fraction "
              f"{seen[1] if seen[1] is not None else float('nan'):.2f} -> {'HOLDING' if seen[0] else 'not seen'}")
        mv("above place", qs["above_place"][0])
        mv("place", qs["place"][0])
        w = gripper(p, RELEASE_MM, effort, secs=1.5)           # fully open, wait, go straight up - no check here:
        time.sleep(1.0)                                        # right above the block every view looks "held"
        print(f"released ({w:.0f} mm)")
        mv("rise", qs["above_place"][0])
        mv("rest", REST_Q)
        # checks from the rest pose, away from the block
        if wrist:
            seen = wrist.holding()
            print(f"wrist camera at rest: near fraction {seen[1]:.2f} -> "
                  f"{'BLOCK STILL ON THE GRIPPER' if seen[0] else 'gripper empty'}")
        if wrist:                                   # grab_top opens both cameras: free the wrist one first
            wrist.close()
            wrist = None
        try:
            objs2 = BlockFinder()(*grab_top())
            got = [np.array(o["robot"]["top_xyz_m"][:2]) for o in objs2 if o["label"] == "block" and o.get("robot")]
            if got:
                dmin = min(np.hypot(*(g - xy)) for g in got)
                print(f"top camera: nearest block to the place target is {dmin * 1000:.0f} mm away "
                      f"({'PLACED' if dmin < 0.03 else 'not at the target'})")
        except Exception as e:
            print(f"top camera check failed: {e}")
        gripper(p, 0, effort)
        print("done")
        if wrist:
            wrist.close()
    except KeyboardInterrupt:
        p.MotionCtrl_1(0x01, 0, 0)
        print("stopped (damped)")
    except RuntimeError as e:
        print(f"stopped at '{e}' - arm holds its pose")


if __name__ == "__main__":
    main()
