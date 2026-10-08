"""Edge case — a tall prism lying on its side has to go into a tall, narrow cell.

Lying, the prism is longer than a cell is wide, so it must go in upright. PiPER can't stand
it up in one rotation: over the table the tool only leans *outward* (fingertips farther from
the base than the wrist), up to ~60° from vertical, never horizontal (IK + workspace
sampling within the joint5 ±65° calibration limit). Strategy — "lean, half-insert, drop":

  1. top-down grasp, grip point offset toward the end that will be on top
  2. lift above the walls, carry top-down over the cell, yawing so the prism's long axis
     lies along a box axis (faces parallel to the cell walls — a diagonal 3 cm square
     jams by its 4.2 cm diagonal) that points away from the robot
  3. rise and lean the tool as far as IK allows, in one move (top-down is out of reach that
     high). This crosses the wrist singularity (joint5 = 0), which a straight Cartesian line
     can't (joint4/6 spin into their limits), so it may be a joint-space move — accepted only
     if all 8 corners of the held prism stay above the walls the whole way
  4. slide down the prism's own axis until its lower end is in the cell, as deep as the
     tilted prism fits, fingers kept above the walls
  5. release and let the cell walls stand it up; back out, un-lean, report the final tilt

Run:
  ./sim_tools/run.sh pac_lying_demo.py                 # viewer
  ./sim_tools/run.sh pac_lying_demo.py --headless
  ./sim_tools/run.sh pac_lying_demo.py --obj 0.03 0.08 --cell 0.045 --height 0.07
"""
import argparse
import math

import mujoco
import numpy as np

from pac_demo import FINGER_REACH, PartitionWatch, grip_for_width
from pac_scene import Blocks, PartitionBox, build
from pick_place_demo import Sim, arm_model

Z = np.array([0.0, 0.0, 1.0])
TIP_ABOVE_BOTTOM = 0.005   # fingertips above the table when grasping a lying prism
CLEAR = 0.015              # lying prism's underside above the wall tops while carried
WALL_CLEAR = 0.004         # held prism's corners above the wall tops while leaning
GRIP_OFFSET = 0.02         # grip this far from the centre toward the end that ends up on top
FIT_MARGIN = 0.004         # tilted prism must leave this much room in the cell
LEAN_DEG = (65, 60, 55, 52, 50, 48, 46, 44, 40)   # tool lean from vertical, most upright prism first
LEAN_RAISE = (0.0, 0.02, 0.04, 0.06)  # extra height to lean at, lowest first
CARRY_RAISE = (0.0, 0.015)            # extra carry height: low reaches the far row, high helps leaning near the base

#: placeholder geometry (real sizes unknown until on-site)
DEFAULT_BOX = dict(rows=4, cols=4, cell=0.045, height=0.07, center=(0.28, 0.0))  # straight ahead: box axes ≈ radial
DEFAULT_OBJ = (0.03, 0.08)            # (width, length) — square cross-section prism
SPOTS = [(0.20, 0.17), (0.30, 0.17), (0.20, -0.17), (0.30, -0.17)]
YAWS = [0, 90, 0, 45]                 # which way each one lies (deg)


# ── rotation helpers ───────────────────────────────────────────────────────────
def frame(x, z):
    """Rotation with tool x and z given (y = z × x)."""
    x, z = x / np.linalg.norm(x), z / np.linalg.norm(z)
    return np.stack([x, np.cross(z, x), z], 1)


def slerp(R0, R1, t):
    q0, q1 = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q0, R0.flatten())
    mujoco.mju_mat2Quat(q1, R1.flatten())
    if q0 @ q1 < 0:
        q1 = -q1
    ang = math.acos(min(1.0, abs(q0 @ q1)))
    q = q0 if ang < 1e-6 else (math.sin((1 - t) * ang) * q0 + math.sin(t * ang) * q1) / math.sin(ang)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q / np.linalg.norm(q))
    return R.reshape(3, 3)


def rot_angle(R0, R1):
    return math.degrees(math.acos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1)))


# ── object geometry from the sim (perception's job on the real robot) ──────────
def prism(sim, bid):
    """Centre, unit long axis, width, length of a block."""
    g = sim.model.geom(f"{bid}_geom")
    half = sim.model.geom_size[g.id]
    Rb = sim.data.body(bid).xmat.reshape(3, 3)
    k = int(np.argmax(half))
    w = 2 * float(np.max(np.delete(half, k)))
    return sim.data.body(bid).xpos.copy(), Rb[:, k].copy(), w, 2 * float(half[k])


# ── geometry of the lean ───────────────────────────────────────────────────────
def lean_dirs(box, cell_xy):
    """Box axes pointing away from the robot, most radial first (the tool can't lean back)."""
    yaw = math.radians(box.yaw_deg)
    ax = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    ay = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    radial = np.array([cell_xy[0], cell_xy[1], 0.0]) / math.hypot(*cell_xy)
    return sorted((d for d in (ax, -ax, ay, -ay) if d @ radial > 0.2), key=lambda d: -(d @ radial))


def insert_depth(box, w, theta):
    """Deepest the outer bottom corner can go below the wall top, and where (along the lean
    direction, from the cell centre) that corner must be so the in-cell part is centred.

    2-D in the lean plane: the prism leans θ outward going up. Below the wall top its outer
    extent is the outer face at the top (corner + δ·tanθ); its inner extent is the inner bottom
    corner (if already below the top) or where the bottom face crosses the top (δ·cotθ).
    """
    room = box.cell - FIT_MARGIN
    t, ct = math.tan(theta), 1 / max(math.tan(theta), 1e-6)

    def span(delta):
        inner = w * math.cos(theta) if w * math.sin(theta) <= delta else delta * ct
        return inner, delta * t

    lo, hi = 0.0, box.height - box.floor_t - 0.005
    if sum(span(hi)) <= room:
        lo = hi
    else:
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if sum(span(mid)) <= room else (lo, mid)
    inner, outer = span(lo)
    return lo, (inner - outer) / 2


class Held:
    """The prism in the gripper, in tool coordinates (origin at the fingertip centre)."""

    def __init__(self, w, L, roll):
        centre = np.array([-roll * GRIP_OFFSET, 0.0, -(w / 2 - TIP_ABOVE_BOTTOM)])
        self.low_face = centre + np.array([-roll * L / 2, 0.0, 0.0])
        self.corners = np.array([centre + np.array([sx * L / 2, sy * w / 2, sz * w / 2])
                                 for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])

    def lowest(self, ik, q):
        T = ik.fk(q)
        tip = T[:3, 3] + T[:3, 2] * FINGER_REACH
        return float(np.min((tip + self.corners @ T[:3, :3].T)[:, 2]))


def plan_lying(box, c, u, w, L, cell_xy, lean_deg, d, top=1, roll=1, raise_=0.0,
               obstacle_top=None, carry_raise=0.0):
    """Keyframes [(name, tip, R, grip, secs)] for one lying prism.

    `lean_deg`: tool lean from vertical; `d`: horizontal lean direction (`lean_dirs`).
    `top` (±1): which end ends up on top. `roll` (±1): which way the gripper faces about the
    prism axis — same line held, but joint6's ±120° makes only some reachable.
    `raise_`: extra height above the pre-insert point to do the lean at.
    `obstacle_top`: highest thing to carry over (walls, or prisms already standing in cells).
    """
    H = box.height
    t_dir = top * u - (u @ Z) * Z
    t_dir /= np.linalg.norm(t_dir)               # toward the end that ends up on top
    phi = math.radians(lean_deg)
    theta = math.pi / 2 - phi                    # prism tilt from vertical
    x_t = math.cos(phi) * d + math.sin(phi) * Z  # prism axis toward its top end, leaned
    R_pick = frame(roll * t_dir, -Z)
    R_down = frame(roll * d, -Z)                 # top-down, prism axis along d
    R_t = frame(roll * x_t, math.sin(phi) * d - math.cos(phi) * Z)   # R_down leaned about the finger axis
    held = Held(w, L, roll)

    depth, corner = insert_depth(box, w, theta)
    cell = np.array([cell_xy[0], cell_xy[1], 0.0])
    # outer bottom corner → bottom-face centre: half a face back along the face (inward, up)
    low_ins = cell + d * (corner - (w / 2) * math.cos(theta)) + Z * (H - depth + (w / 2) * math.sin(theta))
    tip_ins = low_ins - R_t @ held.low_face
    finger_low = 0.01                            # finger corners below the tip point
    if tip_ins[2] - finger_low < H + 0.003:      # fingers would enter the cell: insert shallower
        lift = H + 0.003 + finger_low - tip_ins[2]
        tip_ins = tip_ins + Z * lift
        depth -= lift
    if depth < 0.003:
        raise ValueError(f"a {w * 100:.1f} cm prism tilted {math.degrees(theta):.0f}° can't get into "
                         f"a {box.cell * 100:.1f} cm cell")
    # back up the axis until the lowest corner is just above the walls
    tip_pre = tip_ins + x_t * ((depth + 0.005) / math.sin(phi))
    # lying prism's underside clears the walls and anything already standing in a cell
    z_travel = (obstacle_top or H) + CLEAR + TIP_ABOVE_BOTTOM + carry_raise
    z_lean = max(z_travel, tip_pre[2] + raise_)
    above = np.r_[tip_pre[:2], z_lean]
    g_open, g_rel = grip_for_width(w + 0.016), grip_for_width(w + 0.012)
    keys = [
        ("approach", np.r_[c[:2] + GRIP_OFFSET * t_dir[:2], z_travel], R_pick, g_open, 1.5),
        ("descend", np.r_[c[:2] + GRIP_OFFSET * t_dir[:2], TIP_ABOVE_BOTTOM], R_pick, g_open, 1.2),
        ("grasp", np.r_[c[:2] + GRIP_OFFSET * t_dir[:2], TIP_ABOVE_BOTTOM], R_pick, 0.0, 1.5),
        ("lift", np.r_[c[:2] + GRIP_OFFSET * t_dir[:2], z_travel], R_pick, 0.0, 1.2),
        ("carry", np.r_[tip_pre[:2], z_travel], R_down, 0.0, 2.0),
        # rise and lean in one move — top-down is out of reach this high
        ("lean", above, R_t, 0.0, 2.5),
        ("pre-insert", tip_pre, R_t, 0.0, 0.8),
        ("half-insert", tip_ins, R_t, 0.0, 1.5),
        ("release", tip_ins, R_t, g_rel, 0.8),
        ("back out", tip_pre, R_t, g_rel, 1.0),
        ("unlean", np.r_[tip_pre[:2], z_travel], R_down, g_rel, 2.0),
    ]
    return depth, math.degrees(theta), held, keys


# ── IK along the keyframes ─────────────────────────────────────────────────────
#: in-air moves: in-between waypoints only need to be roughly on the line, and if the line
#: can't be followed (wrist singularity) a joint-space leg is allowed within these bounds:
#: (fingertip off the straight line, dip below the lower end, held prism must clear walls)
JOINT_LEG = {"approach": (0.03, 0.0, False), "lift": (0.02, 0.0, False),
             "carry": (0.02, 0.0, True),
             # over the box only the held prism's clearance matters (checked per waypoint)
             "lean": (0.08, 0.10, True), "unlean": (0.08, 0.03, False),
             "pre-insert": (0.01, 0.0, True)}
LOOSE_MM, LOOSE_DEG = 4.0, 6.0
HOLDING = ("lift", "carry", "lean", "pre-insert")   # prism is in the gripper


def tip_of(ik, q):
    T = ik.fk(q)
    return T[:3, 3] + T[:3, 2] * FINGER_REACH


def solve_pose(ik, tip, R, s, rng=np.random.default_rng(2), tries=30):
    """IK from `s`; if that seed is on the wrong branch, the restart nearest to `s`."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, tip - R[:, 2] * FINGER_REACH
    sol = ik.ik(T, s)
    if sol.ok:
        return sol
    sols = [x for x in (ik.ik(T, rng.uniform(ik.limits[:, 0], ik.limits[:, 1])) for _ in range(tries)) if x.ok]
    return min(sols, key=lambda x: np.linalg.norm(x.q - s)) if sols else sol


def joint_leg(ik, tip0, tip1, R1, s, max_dev, max_dip, clears, rng=np.random.default_rng(1)):
    """Joint-space interpolation to the end pose, accepted only within the bounds."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R1, tip1 - R1[:, 2] * FINGER_REACH
    ends = [ik.ik(T, s)] + [ik.ik(T, rng.uniform(ik.limits[:, 0], ik.limits[:, 1])) for _ in range(20)]
    for e in sorted((e for e in ends if e.ok), key=lambda e: np.linalg.norm(e.q - s))[:3]:
        n = max(2, int(np.max(np.abs(e.q - s)) / math.radians(3)))
        qs = [s + (e.q - s) * k / n for k in range(1, n + 1)]
        if all(np.linalg.norm(tip_of(ik, q) - (tip0 + (tip1 - tip0) * k / n)) <= max_dev
               and tip_of(ik, q)[2] >= min(tip0[2], tip1[2]) - 0.005 - max_dip
               and clears(q) for k, q in enumerate(qs, 1)):
            return qs
    return None


def solve(ik, keys, seed, held, wall_top, start=None, seg_m=0.01, seg_deg=5.0):
    """Straight fingertip lines + slerped orientation between keyframes, IK every ~1 cm/5°;
    joint-space legs where allowed. Returns ([[q...] per keyframe], None) or (None, why)."""
    paths, s, prev = [], seed, start
    for name, tip, R, _g, _t in keys:
        clears = (lambda q: held.lowest(ik, q) >= wall_top + WALL_CLEAR) if name in HOLDING and \
            name != "lift" else (lambda q: True)
        n = 1 if prev is None else max(1, math.ceil(np.linalg.norm(tip - prev[0]) / seg_m),
                                       math.ceil(rot_angle(prev[1], R) / seg_deg))
        path, failed = [], None
        for k in range(1, n + 1):
            p = tip if prev is None else prev[0] + (tip - prev[0]) * k / n
            Rk = R if prev is None else slerp(prev[1], R, k / n)
            if prev is None:
                sol = solve_pose(ik, p, Rk, s)
            else:
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = Rk, p - Rk[:, 2] * FINGER_REACH
                sol = ik.ik(T, s)
            loose = name in JOINT_LEG and k < n and sol.pos_mm < LOOSE_MM and sol.rot_deg < LOOSE_DEG
            if not (sol.ok or loose) or not clears(sol.q):
                failed = (name, p, sol.pos_mm, sol.rot_deg)
                break
            path.append(sol.q)
            s = sol.q
        if failed:
            if name not in JOINT_LEG or prev is None:
                return None, failed
            dev, dip, check = JOINT_LEG[name]
            path = joint_leg(ik, prev[0], tip, R, paths[-1][-1] if paths else seed, dev, dip,
                             clears if check else (lambda q: True))
            if path is None:
                return None, failed
            s = path[-1]
        paths.append(path)
        prev = (tip, R)
    return paths, None


def plan_block(ik, box, c, u, w, L, cell_xy, q, last, obstacle_top=None):
    """Most upright feasible lean over directions / ends / rolls / carry and lean heights."""
    why = "no box axis points away from the robot"
    for lean in LEAN_DEG:
        for d in lean_dirs(box, cell_xy):
            for roll in (1, -1):
                for top in (1, -1):
                    for carry_raise, raise_ in [(cr, r) for cr in CARRY_RAISE for r in LEAN_RAISE]:
                        try:
                            depth, theta, held, keys = plan_lying(box, c, u, w, L, cell_xy, lean, d,
                                                                  top, roll, raise_, obstacle_top,
                                                                  carry_raise)
                        except ValueError as exc:
                            why = str(exc)
                            break
                        # the leaned insert pose must exist at all before chaining anything
                        _, tip_ins, R_ins, _, _ = next(k for k in keys if k[0] == "half-insert")
                        if raise_ == 0.0 and carry_raise == 0.0 and \
                                not solve_pose(ik, tip_ins, R_ins, q).ok:
                            why = f"lean {lean}° unreachable at the cell"
                            break
                        paths, f = solve(ik, keys, q, held, box.height, last)
                        if paths is None and last is not None:   # joint-space transit to the pick
                            paths, f = solve(ik, keys, q, held, box.height, None)
                        if paths:
                            return lean, depth, theta, keys, paths, None
                        why = f"{f[0]} at ({f[1][0]:.3f}, {f[1][1]:.3f}, {f[1][2]:.3f})"
    return None, None, None, None, None, why


# ── recovery: push a wedged prism upright ──────────────────────────────────────
NUDGE_TILT = 15.0          # deg from vertical above which a seated prism gets nudged
FINGER_HALF = 0.016        # closed fingers' half-width along the opening axis


def plan_nudge(box, c, u, w, L):
    """Closed gripper against the outer face of the prism's top end, pushed back over its
    bottom end. A wedged prism leans away from the robot (it went in leaning outward)."""
    up = u if u[2] > 0 else -u
    top, bottom = c + up * L / 2, c - up * L / 2
    h = np.r_[(top - bottom)[:2], 0.0]
    lean_h = np.linalg.norm(h)
    h /= lean_h
    z = max(top[2] - 0.03, box.height + 0.008)             # push ~3 cm below the top (measured best); fingertips above the walls
    start = np.r_[top[:2] + h[:2] * (w / 2 + FINGER_HALF + 0.006), z]
    end = np.r_[bottom[:2] + h[:2] * (w / 2 + FINGER_HALF), z]
    out = []
    for sgn in (1, -1):                                    # finger faces are symmetric: try both
        R = frame(np.cross(sgn * h, -Z), -Z)               # opening axis (tool y) along the push
        out.append([("nudge: above", start + Z * 0.03, R, 0.0, 1.0),
                    ("nudge: down", start, R, 0.0, 0.6),
                    ("nudge: push", end, R, 0.0, 1.5),
                    ("nudge: up", end + Z * 0.04, R, 0.0, 0.6)])
    return out


def result(sim, box, bid):
    c, u, w, L = prism(sim, bid)
    tilt = math.degrees(math.acos(abs(u @ Z)))
    bottom = c[2] - (L / 2) * abs(u @ Z) - (w / 2) * math.sin(math.radians(tilt))
    return box.cell_of(*c[:2]), tilt, bottom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--obj", type=float, nargs=2, default=list(DEFAULT_OBJ), metavar=("W", "L"),
                    help="prism width and length (m)")
    ap.add_argument("--rows", type=int, default=DEFAULT_BOX["rows"])
    ap.add_argument("--cols", type=int, default=DEFAULT_BOX["cols"])
    ap.add_argument("--cell", type=float, default=DEFAULT_BOX["cell"])
    ap.add_argument("--height", type=float, default=DEFAULT_BOX["height"])
    ap.add_argument("--center", type=float, nargs=2, default=list(DEFAULT_BOX["center"]))
    ap.add_argument("--count", type=int, default=len(SPOTS))
    ap.add_argument("--cells", nargs="+", metavar="R,C")
    ap.add_argument("--nudge", action="store_true",
                    help="push a wedged prism upright (experimental: outcome is contact-sensitive)")
    a = ap.parse_args()

    w, L = a.obj
    box = PartitionBox(rows=a.rows, cols=a.cols, cell=a.cell, height=a.height, center=tuple(a.center))
    blocks = Blocks(count=a.count, dims=(L, w, w), spots=SPOTS, yaws=YAWS, mass=0.03,
                    rgba=(0.2, 0.55, 0.85, 1))
    # default: inner cells — at the placeholder layout the corner cells only allow a 44° lean,
    # and below ~50° the prism wedges (see the module docstring)
    targets = ([tuple(int(v) for v in rc.split(",")) for rc in a.cells] if a.cells
               else [(1, 1), (1, 2), (2, 1), (2, 2), (1, 0), (1, 3), (2, 0), (2, 3)][:blocks.count])
    if bad := [rc for rc in targets if rc not in box.cells()]:
        raise SystemExit(f"no such cell {bad} in a {box.rows}x{box.cols} box")
    # farthest cells first: a leaned gripper hangs over the next cell toward the robot, so
    # that one must still be empty (a standing prism sticks out above the walls)
    targets.sort(key=lambda rc: -math.hypot(*box.cell_xy(*rc)))
    print(f"prism {w * 100:.1f}x{w * 100:.1f}x{L * 100:.1f} cm lying; cells {box.cell * 100:.1f} cm, "
          f"walls {box.height * 100:.1f} cm")

    sim = Sim(a.headless, build(box, blocks))
    watch = PartitionWatch(sim.model)
    sim.on_step = watch
    ik = arm_model()
    realtime = not a.headless
    sim.run(0.5, realtime)
    q, grip, last = sim.jm.q_rad(sim.data), 0.0, None
    done = []
    for bid, rc in zip(blocks.ids(), targets):
        c, u, bw, bl = prism(sim, bid)
        # carry over the walls and over whatever already stands in a cell (on the real robot:
        # the heights perception sees inside the box)
        tops = [prism(sim, b)[0][2] + prism(sim, b)[3] / 2 * abs(prism(sim, b)[1][2]) for b in done]
        obstacle_top = max([box.height] + tops)
        lean, depth, theta, keys, paths, why = plan_block(ik, box, c, u, bw, bl, box.cell_xy(*rc), q, last,
                                                          obstacle_top)
        done.append(bid)
        if paths is None:
            print(f"{bid} -> cell {rc}: no feasible plan ({why})")
            continue
        print(f"{bid} -> cell {rc}: lean {lean}° (prism {theta:.0f}° off vertical), "
              f"half-insert {depth * 100:.1f} cm")
        for (name, tip, R, want, secs), path in zip(keys, paths):
            g0 = grip
            for k, q_to in enumerate(path, 1):
                g1 = g0 + (want - g0) * k / len(path)
                sim.ramp(q, q_to, grip, g1, secs / len(path), realtime)
                q, grip = q_to, g1
            last = (tip, R)
        sim.run(0.6, realtime)
        for attempt in range(1 if a.nudge else 0):         # wedged prism → nudge once (a second push tends to knock it over)
            got, tilt, _ = result(sim, box, bid)
            if got != rc or tilt <= NUDGE_TILT:
                break
            c, u, bw, bl = prism(sim, bid)
            for keys in plan_nudge(box, c, u, bw, bl):
                paths, _ = solve(ik, keys, q, None, box.height, last)
                if paths is None:
                    paths, _ = solve(ik, keys, q, None, box.height, None)
                if paths:
                    break
            if not paths:
                print(f"  {bid} wedged at {tilt:.0f}° — nudge out of reach")
                break
            print(f"  {bid} wedged at {tilt:.0f}° — nudging it upright")
            for (name, tip, R, want, secs), path in zip(keys, paths):
                for q_to in path:
                    sim.ramp(q, q_to, grip, 0.0, secs / len(path), realtime)
                    q, grip = q_to, 0.0
                last = (tip, R)
            sim.run(0.6, realtime)
    sim.run(1.0, realtime)

    print("\n=== result ===")
    ok = 0
    for bid, rc in zip(blocks.ids(), targets):
        got, tilt, bottom = result(sim, box, bid)
        seated = got == rc and bottom < box.floor_t + 0.01 and tilt < 20
        ok += seated
        print(f"  {bid}: want {rc}, in {got}, tilt {tilt:4.1f}° from vertical, bottom z {bottom * 100:.1f} cm"
              f" -> {'SEATED ✅' if seated else 'FAIL ❌'}")
    print(f"seated {ok}/{len(targets)}  (seated = right cell, on the floor, < 20° from vertical)")
    if watch.hits:
        top = sorted(watch.hits.items(), key=lambda kv: -kv[1])[:5]
        print("⚠ arm touched the partition (contact steps):", ", ".join(f"{k}×{v}" for k, v in top))
    else:
        print("arm never touched the partition ✅")
    if sim.viewer:
        print("close the viewer window to exit")
        while sim.viewer.is_running():
            sim.run(0.02, True)


if __name__ == "__main__":
    import mujoco.viewer  # noqa: F401
    main()
