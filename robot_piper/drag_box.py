"""Drag the box across the table: grip the middle of one side wall (fingers on both sides of the wall, one
inside the box) and pull the box toward that side, the tool tip moving as a straight horizontal line.

  ./drag_box.sh --dry-run                 scan + plan + picture (pac2026/runs/drag_<time>/plan.png)
  ./drag_box.sh [--side right] [--distance 0.30]

  side      which wall, seen from the robot: right (-y, default), left (+y), far (+x), near (-x)
  distance  how far to pull (m); shortened automatically to stop 2 cm before anything raised in the way

Perception: VGGT (metric from the table plane, no depth camera), the box = the largest raised ring whose walls
stand > 7 cm (a partition inside is allowed); wall centre lines = its outline moved in by half a wall band.
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
                str(ROOT / "perception-main" / "research" / "box_cells"), str(ROOT)]
import arm_ik  # noqa: E402
import hover_test as ht  # noqa: E402
import live_view  # noqa: E402
import partition_scene as ps  # noqa: E402

GRIP_BELOW_RIM = 0.03          # finger tips below the wall top
HOVER = 0.06                   # approach line starts this far above the wall top
APPROACH_MM = 60.0             # opening going down (the wall is ~5 mm; 27 mm each side)
HOLD_MM = (1.0, 25.0)          # closed on the wall
LIFT_DRAG = 0.003              # lift the gripped wall a hair before pulling (less friction)
CLEAR_STOP = 0.02              # stop this far before anything raised in the way
WALL_IN = 0.010                # outline -> wall centre line (half of the wall band in the height map)
BETAS = list(range(-45, 46, 5))
SIDES = {"right": np.array([0.0, -1.0]), "left": np.array([0.0, 1.0]),
         "far": np.array([1.0, 0.0]), "near": np.array([-1.0, 0.0])}


def find_box(Z):
    """Largest raised ring with walls > 7 cm, a partition inside allowed. Returns dict(outline rect, rim_z, table_z, mask)."""
    seen = np.isfinite(Z)
    zt = ps.table_z(Z)
    h = np.nan_to_num(Z - zt, nan=0.0)
    hi = cv2.dilate((h > 0.085).astype(np.uint8), np.ones((3, 3), np.uint8))   # VGGT lifts the table 5-7 cm at the
                                                                                 # picture border: keep it out
    hi[: int((0.06 - ps.X0) / ps.RES)] = 0
    hi[: int((0.115 - ps.X0) / ps.RES), int((-0.10 - ps.Y0) / ps.RES): int((0.10 - ps.Y0) / ps.RES)] = 0   # gripper at rest
    n, lab, st, _ = cv2.connectedComponentsWithStats(hi)
    best = None
    for i in range(1, n):
        if st[i, 4] < 80:
            continue
        m = lab == i
        rr = ps.rect_of(m)
        area = rr["size"][0] * rr["size"][1]
        if not 0.05 < area < 0.5 or rr["size"][1] < 0.15:
            continue
        ring = m & ~ps.rect_mask(Z.shape, rr["corners"], grow=-0.03)    # the outer band = the walls
        wall_h = float(np.percentile(h[ring], 75)) if ring.any() else 0.0
        if wall_h > 0.07 and (best is None or area > best[2]):
            best = (m, rr, area, ring)
    if best is None:
        raise RuntimeError("box: no raised ring with walls > 7 cm")
    m, rr, _, ring = best
    walls = m & (h > 0.09)                              # surely wall: the outline from these, outliers trimmed
    rr = ps.rect_of(walls)                              # min-area rectangle (a full box made grid_rect turn 45 deg)
    ring = walls & ~ps.rect_mask(Z.shape, rr["corners"], grow=-0.015)   # outer 1.5 cm: the wall, not the blocks
    rim = float(np.percentile(Z[ring & seen], 40))     # standing against it inside (taller than the wall)
    return dict(rr, mask=m, rim_z=rim, table_z=zt)


def wall_of(box, side):
    """The wall whose outward normal points most toward `side`: midpoint of its centre line, its direction,
    its outward normal."""
    C, c0 = box["corners"], box["centre"]
    best = None
    for i in range(4):
        a, b = C[i], C[(i + 1) % 4]
        mid = (a + b) / 2
        out = mid - c0
        out = out / np.linalg.norm(out)
        score = out @ SIDES[side]
        if best is None or score > best[0]:
            best = (score, a, b, mid, out)
    _, a, b, mid, out = best
    d = (b - a) / np.linalg.norm(b - a)
    return dict(mid=mid - out * WALL_IN, dir=d, out=out, length=float(np.linalg.norm(b - a)))


def free_distance(Z, box, direction, want, table_z):
    """How far the box can slide along `direction` (unit xy) before its outline meets something raised."""
    h = np.nan_to_num(Z - table_z, nan=0.0)
    raised = (h > 0.03) & ~ps.rect_mask(Z.shape, box["corners"], grow=0.01)
    raised[: int((0.10 - ps.X0) / ps.RES)] = False      # the robot
    for t in np.arange(0.0, want + 1e-9, 0.005):
        moved = box["corners"] + direction * t
        if (raised & ps.rect_mask(Z.shape, moved, grow=0.005)).any():
            return max(0.0, t - CLEAR_STOP), t
    return want, None


def line_xy(xy0, xy1, z, yaw, beta, tcp, q_seed, step=0.01):
    n = max(2, int(np.linalg.norm(np.subtract(xy1, xy0)) / step) + 1)
    qs, seed = [], q_seed
    for t in np.linspace(0, 1, n):
        xy = np.asarray(xy0) + t * (np.asarray(xy1) - np.asarray(xy0))
        q, pe, re = arm_ik.ik_pose([xy[0], xy[1], z], yaw, beta, tcp, seed, quick=bool(qs))
        if pe > 1.5 or re > 0.5 or (qs and np.abs(q - qs[-1]).max() > 15):
            return None
        qs.append(q)
        seed = q
    return qs


def plan(box, wall, dist, tcp):
    """Smallest tilt that reaches: approach line down onto the wall, then the drag line."""
    jaw = np.degrees(np.arctan2(wall["out"][1], wall["out"][0]))      # jaws close across the wall
    tip = box["rim_z"] - GRIP_BELOW_RIM
    g = wall["mid"]
    end = g + wall["out"] * dist
    for beta in sorted(BETAS, key=abs):
        for yaw in (((jaw + 90 + 180) % 360) - 180, ((jaw - 90 + 180) % 360) - 180):
            q, pe, re = arm_ik.ik_pose([g[0], g[1], tip], yaw, beta, tcp)
            if pe > 1.5 or re > 0.5:
                continue
            L_app = arm_ik.pose_line(g, box["rim_z"] + HOVER, tip, yaw, beta, tcp, q_seed=q)
            if L_app is None:
                continue
            L_lift = arm_ik.pose_line(g, tip, tip + LIFT_DRAG, yaw, beta, tcp, q_seed=L_app[-1])
            if L_lift is None:
                continue
            L_drag = line_xy(g, end, tip + LIFT_DRAG, yaw, beta, tcp, L_lift[-1])
            if L_drag is None:
                continue
            L_up = arm_ik.pose_line(end, tip + LIFT_DRAG, box["rim_z"] + HOVER, yaw, beta, tcp, q_seed=L_drag[-1])
            if L_up is None:
                continue
            return dict(yaw=yaw, beta=beta, tip=tip, grip=g, end=end, L_app=L_app, L_lift=L_lift, L_drag=L_drag,
                        L_up=L_up)
    raise SystemExit("no tool orientation reaches the wall and the whole drag line")


def draw(color, K, box, wall, pl, stop_at):
    img = color.copy()
    poly = lambda C, z: ps.project(np.c_[C, np.full(len(C), z)], K).astype(np.int32)
    cv2.polylines(img, [poly(box["corners"], box["rim_z"])], True, (255, 200, 80), 2, cv2.LINE_AA)
    moved = box["corners"] + wall["out"] * np.linalg.norm(pl["end"] - pl["grip"])
    cv2.polylines(img, [poly(moved, box["table_z"])], True, (0, 0, 255), 2, cv2.LINE_AA)
    g = ps.project(np.array([[*pl["grip"], pl["tip"]]]), K)[0].astype(int)
    e = ps.project(np.array([[*pl["end"], pl["tip"]]]), K)[0].astype(int)
    cv2.circle(img, tuple(g), 12, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.arrowedLine(img, tuple(g), tuple(e), (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.3)
    t = (f"grip {('' if stop_at is None else 'SHORTENED ')}wall middle -> pull {np.linalg.norm(pl['end'] - pl['grip']) * 1000:.0f} mm, "
         f"tilt {pl['beta']} deg, rim z {box['rim_z'] * 1000:.0f}")
    cv2.rectangle(img, (0, 0), (img.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(img, t, (12, 25), 0, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=list(SIDES), default="right")
    ap.add_argument("--distance", type=float, default=0.30)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--speed", type=int, default=25)
    ap.add_argument("--line-speed", type=int, default=30)
    ap.add_argument("--line-s", type=float, default=0.2, help="seconds per 1 cm on the vertical legs (5 cm/s)")
    ap.add_argument("--drag-s", type=float, default=0.10, help="seconds per 1 cm while pulling, on average (0.10 -> 30 cm "
                    "in ~3 s); the pull starts and stops smoothly (minimum jerk) so the fingers do not slip off the wall")
    ap.add_argument("--drag-speed", type=int, default=80, help="controller speed %% while pulling")
    ap.add_argument("--effort", type=int, default=5000)
    ap.add_argument("--image", type=Path, help="offline: folder with color.jpg + intrinsics.json")
    return ap


def main(argv=None, p=None, grab=None):
    """p given (demo.py): preflight and the debug log are the caller's; only scan, plan and the pull here.
    grab: () -> (colour, depth, K) from cameras the caller keeps open (live view); default: open, take, close."""
    args = parser().parse_args(argv)
    tcp = yaml.safe_load((ROOT / "pac2026" / "config.yaml").read_text())["robot"]["tcp_offset_m"]
    out = ROOT / "pac2026" / "runs" / f"drag_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    if not args.image and p is None:
        import runlog
        runlog.start(sys.argv)
    if not args.dry_run and not args.image and p is None:
        import pickr
        if not (pickr.check_can() and pickr.check_cameras() and pickr.check_arm(False)):
            sys.exit("preflight failed - nothing was started")
        from board_hover import connect
        p = connect()
    if args.image:
        color = cv2.imread(str(args.image / "color.jpg"))
        K = json.loads((args.image / "intrinsics.json").read_text())
    else:
        import move_partition
        color, _, K = (grab or move_partition.capture)()
    cv2.imwrite(str(out / "color.jpg"), color)
    (out / "intrinsics.json").write_text(json.dumps(K))
    P, s = ps.vggt_points(color, K)
    Z = ps.raster(P)
    box = find_box(Z)
    wall = wall_of(box, args.side)
    dist, hit = free_distance(Z, box, wall["out"], args.distance, box["table_z"])
    print(f"[scan] box outline {box['size'][0] * 1000:.0f} x {box['size'][1] * 1000:.0f} mm at "
          f"({box['centre'][0] * 1000:.0f}, {box['centre'][1] * 1000:.0f}), rim z {box['rim_z'] * 1000:.0f}; "
          f"{args.side} wall middle ({wall['mid'][0] * 1000:.0f}, {wall['mid'][1] * 1000:.0f}) mm")
    if hit is not None:
        print(f"[scan] something raised {hit * 1000:.0f} mm along the pull: pulling {dist * 1000:.0f} mm instead of "
              f"{args.distance * 1000:.0f}")
    if dist < 0.01:
        sys.exit("[plan] nothing to pull: the way is blocked")
    pl = plan(box, wall, dist, tcp)
    cv2.imwrite(str(out / "plan.png"), draw(color, K, box, wall, pl, hit))
    print(f"[plan] grip at ({pl['grip'][0] * 1000:.0f}, {pl['grip'][1] * 1000:.0f}) mm, tip z {pl['tip'] * 1000:.0f}, "
          f"yaw {pl['yaw']:.0f}, tilt {pl['beta']} deg -> pull to ({pl['end'][0] * 1000:.0f}, {pl['end'][1] * 1000:.0f})")
    print(f"[plan] picture: {out / 'plan.png'}")
    if args.dry_run or args.image:
        return
    from board_hover import enable, go_rest
    from pick_place import gripper

    def mv(name, q):
        live_view.status(f"drag box: {name}")
        ok, err, st = ht.move_j(p, q, args.speed, name=name)
        print(f"   {name}: {'ok' if ok else 'FAILED'} ({st.arm_status})", flush=True)
        if not ok:
            raise SystemExit(f"move '{name}' failed - the arm holds its pose")

    def line(name, qs, secs=None, speed=None, smooth=False):
        dense = [qs[0]]
        for a, b in zip(qs[:-1], qs[1:]):
            dense += [a + (b - a) * f for f in (0.25, 0.5, 0.75, 1.0)]
        live_view.status(f"drag box: {name}")
        n = len(dense) - 1
        T = (secs or args.line_s) * n / 4
        if smooth:                                      # waypoint k (evenly spaced) reached at the minimum-jerk time
            tt = np.linspace(0, 1, 2001)
            s_of_t = 10 * tt ** 3 - 15 * tt ** 4 + 6 * tt ** 5
            stamps = T * np.interp(np.arange(1, n + 1) / n, s_of_t, tt)
        else:
            stamps = T * np.arange(1, n + 1) / n
        t0 = time.time()
        for q, ts in zip(dense[1:], stamps):
            qi = (np.asarray(q) * 1000).round().astype(int)
            t_end = t0 + ts
            while time.time() < t_end:
                p.MotionCtrl_2(0x01, 0x01, speed or args.line_speed, 0x00)
                p.JointCtrl(*[int(v) for v in qi])
                time.sleep(0.01)
        print(f"   {name}: streamed in {time.time() - t0:.1f} s", flush=True)
        mv(name, qs[-1])

    enable(p)
    mv("above the wall", pl["L_app"][0])
    gripper(p, APPROACH_MM, args.effort, secs=0.6)
    line("down onto the wall", pl["L_app"])
    w = gripper(p, 0.0, args.effort, secs=1.5)
    print(f"   gripper closed at {w:.1f} mm", flush=True)
    if not HOLD_MM[0] <= w <= HOLD_MM[1]:
        gripper(p, APPROACH_MM, args.effort, secs=0.6)
        line("back up (missed)", pl["L_app"][::-1])
        go_rest(p, args.speed)
        sys.exit(f"gripper at {w:.1f} mm - the wall is not between the fingers")
    line("lift a hair", pl["L_lift"])
    line("pull the box", pl["L_drag"], secs=args.drag_s, speed=args.drag_speed, smooth=True)
    gripper(p, APPROACH_MM, args.effort, secs=0.8)
    line("up off the wall", pl["L_up"])
    go_rest(p, args.speed)
    live_view.status("drag box: finished")


if __name__ == "__main__":
    main()
