"""Partition mission: take the top partition of the stack, carry it over the box wall and lower it straight down
into the box; then the next one on top of it.

Why the gripper is TILTED: the PiPER can hold the tool vertical only up to ~5 cm tip height; over the box
(walls ~20 cm) the tip has to be at ~23 cm. So the divider is gripped with the tool tilted IN THE PLANE of the
divider plate (rotation about the jaw axis) and that tilt is kept unchanged until the release: the partition
stays level and goes down vertically.

Grasp choice: every divider plate (the plate, not an intersection), positions every 2 cm along it, a few grip
depths, both tool yaws across the plate and tilts -45..45 deg. A candidate is kept only if the whole motion is
reachable with ONE fixed orientation per side (approach line, grasp, lift line, place line) and the gripper does
not touch another divider while going down. Among those: nearest to the robot first (short reach moves better),
a little preference for the partition centre (balance) and for less tilt.

  ./partition.sh --dry-run        scan + plan, picture in pac2026/runs/partition_<time>/plan.png
  ./partition.sh                  asks once, then moves (both partitions; --count 1 for one)
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
sys.path[:0] = [str(ROOT / "robot_piper"), str(ROOT / "perception-main"),
                str(ROOT / "perception-main" / "research" / "box_cells"), str(ROOT)]   # ROOT: pickr.py (preflight)
import arm_ik  # noqa: E402
import collide  # noqa: E402
import hover_test as ht  # noqa: E402
import live_view  # noqa: E402
import partition_scene as ps  # noqa: E402

PART_SIZE = None               # partition outline (long, short), m: measured; give it when the far end is cut off
CLEAR = 0.04                   # partition bottom above the box rim while it travels (4 cm: team asked higher)
DROP = 0.008                   # release with the partition bottom this far above the floor / the layer below
HOVER = 0.05                   # approach line starts this far above the divider tops
GRIP_DEPTHS = (0.03, 0.025, 0.02, 0.015)   # finger tips below the divider top, deepest that works first: with the
                               # tool tilted 35-45 deg the finger-base corner is only 18-30 mm above the tips
BETAS = list(range(-45, 46, 5))
APPROACH_MM = 70.0             # gripper opening going down (35 mm each side of the plate)
RELEASE_MM = 60.0              # opening to let go inside the box (fingers stay in the two cells)
HOLD_MM = (1.0, 25.0)          # closed on a cardboard plate (fully closed = nothing in the gripper)
EFFORT = 5000                  # gripper effort (0..5000 = 0..5 N m, max): only 2-3 cm of the plate is in the fingers
WALL_GAP = 0.003               # half thickness of a divider / box wall for the collision check
MARGIN = 0.004
STEP = 0.02                    # grasp positions along a divider
CENTRE_ONLY = True             # grip only the lengthwise divider nearest the partition centre (balanced lift)


class PlanFail(Exception):
    pass


# ------------------------------------------------------------------ geometry
def gripper_parts(tcp, opening):
    """Flange-frame points split into (body + camera, fingers)."""
    P = collide.gripper_points(tcp, opening)
    nb = len(collide._box_pts([0, 0, 0.053], [0.035, 0.065, 0.0145]))
    nf = len(collide._box_pts([0, 0, 0], [0.02, 0.006, (tcp - 0.068) / 2]))
    return np.vstack([P[:nb], P[nb + 2 * nf:]]), P[nb:nb + 2 * nf]


def world(q, P_flange):
    pos, R = arm_ik.fk(q)
    return P_flange @ R.T + pos / 1000


def flange_T(q):
    pos, R = arm_ik.fk(q)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, pos / 1000
    return T


def slab_clearance(P, segs):
    """Smallest clearance of points P to vertical slabs [(p0, p1, z_top)] (negative = inside)."""
    best = np.inf
    for p0, p1, zt in segs:
        d = p1 - p0
        t = np.clip((P[:, :2] - p0) @ d / (d @ d), 0, 1)
        dist = np.linalg.norm(P[:, :2] - (p0 + t[:, None] * d), axis=1) - WALL_GAP
        best = min(best, float(np.min(np.maximum(dist, P[:, 2] - zt))))
    return best


def divider_slabs(part, extend=True):
    """Divider plates as slabs over the full footprint (a divider runs from edge to edge)."""
    out = []
    a = part["axis"]
    n = np.array([-a[1], a[0]])
    L, W = part["size"]
    for d in part["dividers"]:
        if "point" in d:                                 # fitted plate line (RANSAC): through its own point
            u = (part["centre"] - d["point"]) @ d["dir"]
            c = d["point"] + d["dir"] * u               # its point level with the partition centre
        else:
            c = part["centre"] + d["normal"] * d["offset"]
        half = (L if d["along"] == "long" else W) / 2
        out.append((c - d["dir"] * half, c + d["dir"] * half, part["top_z"]))
    return out


def box_walls(box):
    C = box["corners"]
    return [(C[i], C[(i + 1) % 4], box["rim_z"]) for i in range(4)]


def effective_box(box, part, side):
    """The camera-facing wall of the box hides the floor next to it, so the measured inner rectangle is too
    narrow there: widen it on that side (box side +y -> its -y wall) to fit the partition + 1 cm."""
    L, W = box["size"]
    need_w = part["size"][1] + 0.01
    need_l = part["size"][0] + 0.01
    a = box["axis"]
    n = np.array([-a[1], a[0]])
    c = np.array(box["centre"], float)
    if W < need_w:                                       # grow toward the camera (towards y = 0)
        toward = -np.sign(side) * np.sign(n[1]) if abs(n[1]) > 0.5 else 1.0
        c = c + n * toward * (need_w - W) / 2
        W = need_w
    if L < need_l and box.get("cut"):
        c = c + a * (need_l - L) / 2                     # far end outside the view
        L = need_l
    return dict(box, centre=c, size=(L, W), corners=ps.rect_corners(c, a, (L, W)))


def rot2(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def place_transform(part, box):
    """Partition footprint -> box: rotation (deg; the least turn that fits - a square box does not force a quarter
    turn) and the map xy -> xy."""
    ap, ab = part["axis"], box["axis"]
    base = np.degrees(np.arctan2(ab[1], ab[0]) - np.arctan2(ap[1], ap[0]))
    (L, W), (Lb, Wb) = part["size"], box["size"]
    cands = []
    for k in range(4):                                   # partition long side along the box long (k even) or short side
        d = ((base + 90 * k) + 180) % 360 - 180
        over = max(L - Lb, W - Wb, 0) if k % 2 == 0 else max(L - Wb, W - Lb, 0)
        cands.append((round(over, 3) > 0.015, abs(d), d))   # fits (within 1.5 cm of measuring error) first, then least turn
    d = min(cands)[2]
    R = rot2(d)
    return d, (lambda xy: box["centre"] + (np.asarray(xy) - part["centre"]) @ R.T)


# ------------------------------------------------------------------ planning
def carry_line(xy0, xy1, z, yaw0, yaw1, beta, tcp, q_seed, step=0.02):
    """Horizontal straight line of the tip at height z, tilt fixed, yaw turning evenly: the partition stays level."""
    n = max(2, int(np.linalg.norm(np.subtract(xy1, xy0)) / step) + 1)
    qs, seed = [], q_seed
    for t in np.linspace(0, 1, n):
        xy = np.asarray(xy0) + t * (np.asarray(xy1) - np.asarray(xy0))
        q, pe, re = arm_ik.ik_pose([xy[0], xy[1], z], yaw0 + t * (yaw1 - yaw0), beta, tcp, seed, quick=bool(qs))
        if pe > 1.5 or re > 0.5 or (qs and np.abs(q - qs[-1]).max() > 15):
            return None
        qs.append(q)
        seed = q
    return qs


def candidates(part):
    """(divider index, grasp xy) along every divider, away from crossings and the ends."""
    out = []
    a = part["axis"]
    slabs = divider_slabs(part)
    longs = [i for i, d in enumerate(part["dividers"]) if d["along"] == "long"]
    centre_long = min(longs, key=lambda i: abs(part["dividers"][i]["offset"])) if longs else None
    for i, (d, (p0, p1, _)) in enumerate(zip(part["dividers"], slabs)):
        if CENTRE_ONLY and centre_long is not None and i != centre_long:
            continue                                     # the middle lengthwise divider only (team rule)
        length = np.linalg.norm(p1 - p0)
        cross = []
        for j, (e, (q0, q1, _)) in enumerate(zip(part["dividers"], slabs)):
            if e["along"] != d["along"]:
                u = ((q0 + q1) / 2 - p0) @ d["dir"]
                cross.append(u)
        for s in np.arange(0.03, length - 0.03 + 1e-9, STEP):
            if all(abs(s - u) > 0.025 for u in cross):
                out.append((i, p0 + d["dir"] * s))
    return out


def plan(sc, tcp, layer_top_z, side=1.0, verbose=True):
    """Best grasp + all joint paths, or PlanFail with the reasons."""
    part, box = sc["part"], effective_box(sc["box"], sc["part"], side)
    dyaw, to_box = place_transform(part, box)
    slabs = divider_slabs(part)
    walls = box_walls(box)
    body0, fing0 = gripper_parts(tcp, APPROACH_MM / 1000)
    bodyc, fingc = gripper_parts(tcp, 0.008)
    bottom = part["bottom_z"]
    lift_bottom = box["rim_z"] + CLEAR                   # partition bottom while it travels
    rel_bottom = max(layer_top_z, box["floor_z"]) + DROP
    reach0 = lambda xy: float(np.hypot(*xy))
    cands = candidates(part)
    pre = lambda c: float(np.linalg.norm(c[1] - part["centre"]))   # the middle of the centre divider first
    cands.sort(key=pre)
    why = {}
    found = []
    t0 = time.time()
    for di, g in cands:
        d = part["dividers"][di]
        jaw_deg = np.degrees(np.arctan2(d["normal"][1], d["normal"][0]))
        yaws = [((jaw_deg + 90 + k * 180 + 180) % 360) - 180 for k in (0, 1)]
        gp = to_box(g)
        best_here = None
        # quickest rejection: over the box with the DEEPEST grip (lowest lift) - if no tilt reaches it there,
        # no shallower grip will
        tl_min = lift_bottom + part["top_z"] - max(GRIP_DEPTHS) - bottom
        ok_b = {(yaw, b) for yaw in yaws for b in BETAS
                if (lambda r: r[1] <= 1.5 and r[2] <= 0.5)(arm_ik.ik_pose([gp[0], gp[1], tl_min], yaw + dyaw, b, tcp))}
        if not ok_b:
            why["over the box unreachable"] = why.get("over the box unreachable", 0) + 1
            continue
        for gd in GRIP_DEPTHS:
            tip_g = part["top_z"] - gd
            below = tip_g - bottom                       # tip height above the partition bottom (fixed in hand)
            tip_lift = lift_bottom + below
            tip_rel = rel_bottom + below
            for yaw in yaws:
                yaw_p = yaw + dyaw
                for beta in sorted(BETAS, key=abs):
                    if (yaw, beta) not in ok_b:
                        continue
                    q_g, pe, re = arm_ik.ik_pose([g[0], g[1], tip_g], yaw, beta, tcp)
                    if pe > 1.5 or re > 0.5:
                        why["grasp unreachable"] = why.get("grasp unreachable", 0) + 1
                        continue
                    q_pt, pe, re = arm_ik.ik_pose([gp[0], gp[1], tip_lift], yaw_p, beta, tcp)
                    if pe > 1.5 or re > 0.5:
                        why["over the box unreachable"] = why.get("over the box unreachable", 0) + 1
                        continue
                    # fingers / body vs the dividers at the grasp (fingers straddle their own plate)
                    others = [s for k, s in enumerate(slabs) if k != di]
                    c_f = slab_clearance(world(q_g, fing0), others) if others else np.inf
                    c_b = slab_clearance(world(q_g, body0), slabs)
                    if min(c_f, c_b) < MARGIN:
                        why["gripper hits a divider"] = why.get("gripper hits a divider", 0) + 1
                        continue
                    L_app = arm_ik.pose_line(g, part["top_z"] + HOVER, tip_g, yaw, beta, tcp, q_seed=q_g)
                    L_up = arm_ik.pose_line(g, tip_g, tip_lift, yaw, beta, tcp, q_seed=q_g)
                    L_dn = arm_ik.pose_line(gp, tip_lift, tip_rel, yaw_p, beta, tcp, q_seed=q_pt)
                    if L_app is None or L_up is None or L_dn is None:
                        why["a straight leg breaks"] = why.get("a straight leg breaks", 0) + 1
                        continue
                    c_app = min(min(slab_clearance(world(q, fing0), others) if others else np.inf,
                                    slab_clearance(world(q, body0), slabs)) for q in L_app)
                    if c_app < MARGIN:
                        why["gripper hits a divider going down"] = why.get("gripper hits a divider going down", 0) + 1
                        continue
                    c_box = min(slab_clearance(world(q, np.vstack([bodyc, fingc])), walls) for q in L_dn)
                    if c_box < MARGIN:
                        why["gripper hits the box wall"] = why.get("gripper hits the box wall", 0) + 1
                        continue
                    cost = float(np.linalg.norm(g - part["centre"])) + 0.0005 * abs(beta)
                    best_here = dict(cost=cost, div=di, along=d["along"], xy=g, xy_place=gp, yaw=yaw, yaw_place=yaw_p,
                                     beta=beta, grip_depth=gd, tip_z=tip_g, tip_lift=tip_lift, tip_release=tip_rel,
                                     L_app=L_app, L_up=L_up, L_dn=L_dn, clear_div=min(c_app, c_f, c_b), clear_box=c_box)
                    break                                # smallest working tilt for this yaw / depth
                if best_here:
                    break
            if best_here:
                break
        if best_here:
            found.append(best_here)
            if len(found) >= 3:
                break
    if not found:
        raise PlanFail("no grasp works: " + ", ".join(f"{k} x{v}" for k, v in why.items()))
    best = min(found, key=lambda f: f["cost"])
    # the partition rigidly in the hand: its outline (bottom / top) in the flange frame at the grasp
    Tg = flange_T(best["L_up"][0])
    C = part["corners"]
    corners = np.vstack([np.c_[C, np.full(4, bottom)], np.c_[C, np.full(4, part["top_z"])]])
    best["part_in_hand"] = (np.c_[corners, np.ones(8)] @ np.linalg.inv(Tg).T)[:, :3]
    R = rot2(dyaw)
    best["part_place_corners"] = box["centre"] + (C - part["centre"]) @ R.T
    best["box_eff"] = box
    best["fits"] = (part["size"][0] <= box["size"][0] - 0.002, part["size"][1] <= box["size"][1] - 0.002)
    # carry: horizontal line from the top of the lift to the top of the place line, orientation kept (only the
    # yaw turns by the box angle); the partition corners must stay above the rim the whole way
    for cand in sorted(found, key=lambda f: f["cost"]):
        Lc = carry_line(cand["xy"], cand["xy_place"], cand["tip_lift"], cand["yaw"], cand["yaw_place"], cand["beta"],
                        tcp, cand["L_up"][-1])
        if Lc is not None:
            best = dict(best, **cand) if cand is best else cand
            break
    else:
        raise PlanFail(f"{len(found)} grasp(s) work, but none can carry the partition level over the box wall")
    if best is not cand:
        best = cand
    if "part_in_hand" not in best:
        Tg = flange_T(best["L_up"][0])
        best["part_in_hand"] = (np.c_[corners, np.ones(8)] @ np.linalg.inv(Tg).T)[:, :3]
        best["part_place_corners"] = box["centre"] + (C - part["centre"]) @ R.T
        best["box_eff"] = box
        best["fits"] = (part["size"][0] <= box["size"][0] - 0.002, part["size"][1] <= box["size"][1] - 0.002)
    best["L_carry"] = Lc
    zmin = min(float(world(q, best["part_in_hand"])[:4, 2].min()) for q in Lc)
    best["carry_min_bottom"] = zmin
    if zmin < box["rim_z"] + 0.005:
        raise PlanFail(f"carry dips: partition bottom {zmin * 1000:.0f} mm, rim {box['rim_z'] * 1000:.0f} mm")
    if verbose:
        print(f"[plan] {len(found)} working grasp(s) in {time.time() - t0:.1f} s; rejected: "
              + (", ".join(f"{k} x{v}" for k, v in why.items()) or "none"))
    return best


# ------------------------------------------------------------------ motion
def run(p, sc, g, args, tcp):
    from pick_place import gripper

    def mv(name, q, speed=None):
        live_view.status(f"partition: {name}")
        ok, err, st = ht.move_j(p, q, speed or args.speed, name=name)
        print(f"   {name}: {'ok' if ok else 'FAILED'} ({st.arm_status})", flush=True)
        if not ok:
            raise SystemExit(f"move '{name}' failed ({st.arm_status}) - the arm holds its pose")

    def line(name, qs, smooth=False):
        dense = [qs[0]]
        for a, b in zip(qs[:-1], qs[1:]):
            dense += [a + (b - a) * f for f in (0.25, 0.5, 0.75, 1.0)]
        live_view.status(f"partition: {name}")
        n = len(dense) - 1
        T = args.line_s * n / 4
        if smooth:                                      # minimum-jerk timing: no jolt at the start / end (the
            tt = np.linspace(0, 1, 2001)                # divider is held only 2-3 cm deep)
            stamps = T * np.interp(np.arange(1, n + 1) / n, 10 * tt ** 3 - 15 * tt ** 4 + 6 * tt ** 5, tt)
        else:
            stamps = T * np.arange(1, n + 1) / n
        t0 = time.time()
        for q, ts in zip(dense[1:], stamps):
            qi = (np.asarray(q) * 1000).round().astype(int)
            t_end = t0 + ts
            while time.time() < t_end:
                p.MotionCtrl_2(0x01, 0x01, args.line_speed, 0x00)
                p.JointCtrl(*[int(v) for v in qi])
                time.sleep(0.01)
        mv(name, qs[-1])

    from board_hover import enable, go_rest
    enable(p)
    mv("above the divider", g["L_app"][0])
    gripper(p, APPROACH_MM, args.effort, secs=0.6)
    line("down to the divider", g["L_app"])
    w = gripper(p, 0.0, args.effort, secs=1.5)
    print(f"   gripper closed at {w:.1f} mm", flush=True)
    if not HOLD_MM[0] <= w <= HOLD_MM[1]:
        gripper(p, APPROACH_MM, args.effort, secs=0.6)
        line("back up (missed)", g["L_app"][::-1])
        raise SystemExit(f"gripper at {w:.1f} mm - no divider between the fingers; stopped above the stack")
    line("lift straight up", g["L_up"], smooth=True)
    line("carry level over the box wall", g["L_carry"], smooth=True)
    line("down into the box", g["L_dn"])
    gripper(p, RELEASE_MM, args.effort, secs=0.8)
    line("up out of the box", g["L_dn"][::-1])
    go_rest(p, args.speed)


def capture():
    import gdino_vggt_lines  # noqa: F401  (pac2026 on the path)
    from pac.cameras import CameraManager
    cfg = yaml.safe_load((ROOT / "pac2026" / "config.yaml").read_text())
    cams = CameraManager(cfg)
    live_view.set_source("top", cams._top.latest, cams._top.intrinsics)
    try:
        time.sleep(1.0)
        f = cams.top(with_depth=True)
    finally:
        live_view.set_source("top", None)
        cams.close()
    return f.color, f.depth_m, f.intrinsics


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="scan + plan only, the arm does not move")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--count", type=int, default=1, help="partitions to move (each one is scanned again first)")
    ap.add_argument("--speed", type=int, default=50, help="joint moves between poses, percent")
    ap.add_argument("--effort", type=int, default=EFFORT, help="gripper effort 0..5000 (5 N m)")
    ap.add_argument("--line-speed", type=int, default=60, help="controller speed %% on the straight legs")
    ap.add_argument("--line-s", type=float, default=0.12, help="seconds per 1 cm on the straight legs (0.12 = 8 cm/s)")
    ap.add_argument("--part-size", type=float, nargs=2, default=PART_SIZE, metavar=("LONG", "SHORT"),
                    help="partition outline in m (default: measured); needed only when it runs out of the view")
    ap.add_argument("--part-height", type=float, default=0.10,
                    help="partition height incl. its bottom board, m: used when the thin divider tops read lower")
    ap.add_argument("--box-side", choices=("left", "right"), default="left", help="robot's left = +y")
    ap.add_argument("--image", type=Path, help="offline: folder with color.jpg, rs_depth_m.npy, intrinsics.json")
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--no-qwen", action="store_true", help="skip the Qwen3-VL cross-check of box / partition")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    side = 1.0 if args.box_side == "left" else -1.0
    cfg = yaml.safe_load((ROOT / "pac2026" / "config.yaml").read_text())["robot"]
    tcp = cfg["tcp_offset_m"]
    out = ROOT / "pac2026" / "runs" / f"partition_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    p = None
    if not args.image:
        import runlog
        runlog.start(sys.argv)                            # console + every move -> pac2026/runs/logs/
    if not args.no_view and not args.image:               # live window (own process), as in pickr.sh
        import subprocess
        try:
            subprocess.Popen([sys.executable, str(ROOT / "robot_piper" / "live_viewer.py")],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError as e:
            print(f"[view] live window not started: {e}")
    if not args.dry_run and not args.image:
        import pickr
        if not (pickr.check_can() and pickr.check_cameras() and pickr.check_arm(False)):
            sys.exit("preflight failed - nothing was started")
        from board_hover import connect
        p = connect()
        live_view.set_arm(p)
    box, layer_top, placed = None, None, []             # placed: (bottom z, height) of the layers in the box
    for k in range(1, args.count + 1):
        print(f"== partition {k}/{args.count} ==", flush=True)
        live_view.status(f"partition {k}: scanning")
        if args.image:
            color = cv2.imread(str(args.image / "color.jpg"))
            depth = np.load(args.image / "rs_depth_m.npy")
            K = json.loads((args.image / "intrinsics.json").read_text())
        else:
            color, depth, K = capture()
        cv2.imwrite(str(out / f"color_{k}.jpg"), color)
        np.save(out / f"rs_depth_m_{k}.npy", depth.astype(np.float32))
        (out / "intrinsics.json").write_text(json.dumps(K))
        qb = None
        if not args.no_qwen:                             # step 1: Qwen alone on the GPU (VGGT / GDINO unloaded)
            import gc
            import torch
            import gdino_vggt_lines as gvl
            import qwen_check
            gvl._VGGT.clear()
            ps._GD.clear()
            gc.collect()
            torch.cuda.empty_cache()
            live_view.status(f"partition {k}: Qwen check")
            qb = qwen_check.qwen_boxes(color)
            print(f"[qwen] box {np.round(qb['box']).tolist() if 'box' in qb else '-'} px, partition "
                  f"{np.round(qb['partition']).tolist() if 'partition' in qb else '-'} px", flush=True)
        live_view.status(f"partition {k}: scanning (VGGT)")
        sc = ps.scan(color, depth, K, box=box, part_size=tuple(args.part_size) if args.part_size else None, side=side,
                     part_h=args.part_height, box_px=(qb or {}).get("box"))
        b, pt = sc["box"], sc["part"]
        box = box or b                                   # the box does not move: keep the first measurement
        layer_top = b["floor_z"]
        if placed:                                       # thin divider tops read low: at least bottom + height
            layer_top = max(b["content_top_z"], placed[-1][0] + placed[-1][1])
        print(f"[scan] box inner {b['size'][0] * 1000:.0f} x {b['size'][1] * 1000:.0f} mm{' (cut)' if b['cut'] else ''}, "
              f"rim z {b['rim_z'] * 1000:.0f}, floor z {b['floor_z'] * 1000:.0f}, release on z {layer_top * 1000:.0f}")
        print(f"[scan] partition {pt['size'][0] * 1000:.0f} x {pt['size'][1] * 1000:.0f} mm{' (cut)' if pt['cut'] else ''} "
              f"at ({pt['centre'][0] * 1000:.0f}, {pt['centre'][1] * 1000:.0f}), board z {pt['board_z'] * 1000:.0f}, "
              f"top z {pt['top_z'] * 1000:.0f}{' (known height; measured ' + format(pt['top_measured'] * 1000, '.0f') + ')' if pt['top_nominal'] else ''}, dividers: "
              + ", ".join(f"{d['along']} {d['offset'] * 1000:+.0f}" for d in pt["dividers"]))
        try:
            g = plan(sc, tcp, layer_top, side)
        except PlanFail as e:
            cv2.imwrite(str(out / f"plan_{k}.png"), ps.draw(color, sc))
            sys.exit(f"[plan] {e}  (picture: {out / f'plan_{k}.png'})")
        img = ps.draw(color, dict(sc, box=g["box_eff"]), g)
        if qb is not None:                               # step 3: Qwen boxes vs the measured outlines
            import qwen_check
            qok, qrep, _ = qwen_check.compare(qb, sc)
            qwen_check.draw(img, qb)
            print(f"[qwen] overlap with the measurement (IoU, need {qwen_check.MIN_IOU}): {qrep}", flush=True)
            if not qok:
                cv2.imwrite(str(out / f"plan_{k}.png"), img)
                sys.exit(f"[qwen] Qwen does not agree with the scan - not moving (picture: {out / f'plan_{k}.png'}); "
                         "--no-qwen to skip this check")
        cv2.imwrite(str(out / f"plan_{k}.png"), img)
        print(f"[plan] grip the {g['along']} divider at ({g['xy'][0] * 1000:.0f}, {g['xy'][1] * 1000:.0f}) mm, "
              f"{g['grip_depth'] * 1000:.0f} mm below its top, tool yaw {g['yaw']:.0f}, tilt {g['beta']} deg; "
              f"lift tip to z {g['tip_lift'] * 1000:.0f}, release tip z {g['tip_release'] * 1000:.0f} at "
              f"({g['xy_place'][0] * 1000:.0f}, {g['xy_place'][1] * 1000:.0f}); clearance divider "
              f"{g['clear_div'] * 1000:.0f} / box wall {g['clear_box'] * 1000:.0f} mm; carry bottom min "
              f"{g['carry_min_bottom'] * 1000:.0f} mm; fits the box: {g['fits']}")
        print(f"[plan] picture: {out / f'plan_{k}.png'}")
        if args.dry_run or args.image:
            break                                        # the next layer only exists after a real move
        try:
            run(p, sc, g, args, tcp)
        except KeyboardInterrupt:
            from fill_cells import safe_rest
            safe_rest(p, args)
            sys.exit("[stop] Ctrl+C - if the partition is still in the gripper, take it out by hand")
        placed.append((g["tip_release"] - (g["tip_z"] - pt["bottom_z"]), pt["height"]))
    live_view.status("partition: finished")


if __name__ == "__main__":
    try:
        if "--no-view" not in sys.argv and "--image" not in sys.argv:
            live_view.start()
        main()
    except KeyboardInterrupt:
        print("\n[stop] Ctrl+C", flush=True)
