"""PAC partition-box pick & place — put each block into its own cell.

Per block: approach → descend → grasp → lift above the walls → carry over the cell →
lower *into* the cell → open just enough to release (fingers must not hit dividers) →
back up. Afterwards: which cell each block sits in, and whether the gripper/arm
touched the partition (the real mission fails if the partition collapses).

Run:
  ./sim_tools/run.sh pac_demo.py                          # 4x4, 5 blocks, viewer
  ./sim_tools/run.sh pac_demo.py --headless
  ./sim_tools/run.sh pac_demo.py --rows 3 --cols 3 --cell 0.08
  ./sim_tools/run.sh pac_demo.py --cells 0,0 1,2 3,3      # choose target cells
"""
import argparse

import mujoco
import numpy as np

from pac_scene import FINGER_GAP0, FINGER_LEN, FINGER_REACH, FINGER_T, Blocks, add_args, box_from_args, build
from pick_place_demo import GRASP_SEED_DEG, Sim, arm_model
from piper_sim.scene import FINGER_STROKE_M

TIP_ABOVE_BOTTOM = 0.005  # fingertips this far above the block bottom while held
CLEAR = 0.012            # block-bottom clearance above the walls while carrying (m)
SEAT_GAP = 0.008         # block-bottom above the cell floor when released (m) …
WRIST_CLEAR = 0.01       # … unless that would bring the gripper housing (above the pads) within this of the wall top
ARM_BODIES = ("link4", "link5", "link6", "gripper_base", "finger_l", "finger_r")


def grip_for_width(width):
    """Gripper fraction whose inner faces are `width` apart."""
    return float(np.clip((width / 2 - FINGER_GAP0) / FINGER_STROKE_M, 0.0, 1.0))


def plan(box, blocks, block_xy, cell_xy):
    """[(name, fingertip-centre (x, y, z), grip, secs)] for one block."""
    s = blocks.size
    z_grasp = TIP_ABOVE_BOTTOM
    z_carry = box.height + CLEAR + TIP_ABOVE_BOTTOM        # block bottom clears the walls
    # release low in the cell, but keep the gripper housing above the walls: only the 6 cm pads
    # may enter the cell, so with 12 cm walls the block drops the last few cm
    z_seat = max(box.floor_t + SEAT_GAP, box.height + WRIST_CLEAR - FINGER_LEN) + TIP_ABOVE_BOTTOM
    # release opening: wide enough to let go, narrow enough that fingers stay inside the cell
    lo, hi = s + 0.004, box.cell - 2 * FINGER_T - 0.006
    if lo > hi:
        raise SystemExit(f"block {s * 100:.1f} cm + two 8 mm fingers does not fit a "
                         f"{box.cell * 100:.1f} cm cell — gripper can't go into the cell")
    g_open, g_release = grip_for_width(s + 0.016), grip_for_width(lo)  # tightest release: room for a tilted finger
    bx, by = block_xy
    cx, cy = cell_xy
    return [
        ("approach", (bx, by, z_carry), g_open, 1.5),
        ("descend", (bx, by, z_grasp), g_open, 1.2),
        ("grasp", (bx, by, z_grasp), 0.0, 1.5),
        ("lift", (bx, by, z_carry), 0.0, 1.2),
        ("carry", (cx, cy, z_carry), 0.0, 1.8),
        ("insert", (cx, cy, z_seat), 0.0, 1.5),
        ("release", (cx, cy, z_seat), g_release, 0.8),
        ("retreat", (cx, cy, z_carry), g_release, 1.2),
    ]


def tool_rotations(ik, tilts_deg=tuple(range(0, 31, 3))):
    """Candidate grasp orientations, least tilted first. Straight down can't reach the far
    cells at carry height, so the tool may lean (top toward the robot, like the repo's seed
    pose) — up to ~30° with the real gripper; yaw is free because square blocks/cells are 90°-symmetric."""
    out = []
    for tilt in tilts_deg:
        t = np.radians(tilt)
        lean = np.array([[np.cos(t), 0, -np.sin(t)], [0, 1, 0], [np.sin(t), 0, np.cos(t)]])
        # yaw 0/180 first: the gripper base's long side (6 cm, tool x) then lies across the
        # lean, so a leaning base stays off the near divider
        for yaw in (0, 180, 90, -90):
            a = np.radians(yaw)
            y = np.array([np.cos(a), np.sin(a), 0.0])     # finger-opening axis
            z = np.array([0.0, 0.0, -1.0])
            out.append((f"tilt {tilt}° yaw {yaw}", lean @ np.stack([np.cross(y, z), y, z], 1)))
    out.append(("seed (tilt ~7°)", ik.fk(np.radians(GRASP_SEED_DEG))[:3, :3]))
    return out


CELL_PHASES = ("carry", "insert", "release", "retreat")


def solve_block(ik, rotations, steps, seed, start_tip=None, seg=0.01, lean_mid=0.0):
    """Pick one orientation that solves every phase of this block.

    Each phase is a straight fingertip line from the previous target, solved by IK every
    `seg` metres — interpolating joints instead swings the fingers sideways inside a cell.
    Over the cell the fingertip aim moves forward by the lean at `lean_mid` above the tips,
    so the part of the gripper inside the cell is centred rather than the tips alone.
    Returns (label, [[q, ...] per phase]) or (None, failure info).
    """
    worst = None
    for label, R in rotations:
        paths, s, prev = [], seed, start_tip
        lean = np.array([R[0, 2], R[1, 2], 0.0]) * lean_mid    # tool z leans forward; its top leans back
        for name, tip, _g, _t in steps:
            tip = np.asarray(tip, float) + (lean if name in CELL_PHASES else 0.0)
            n = 1 if prev is None else max(1, int(np.ceil(np.linalg.norm(tip - prev) / seg)))
            path = []
            for k in range(1, n + 1):
                p = tip if prev is None else prev + (tip - prev) * k / n
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = p - R[:, 2] * FINGER_REACH   # fingertip → link6
                sol = ik.ik(T, s)
                if not sol.ok:
                    worst = (label, name, p, sol.pos_mm)
                    break
                path.append(sol.q)
                s = sol.q
            if len(path) < n:
                break
            paths.append(path)
            prev = tip
        else:
            return label, paths
    return None, worst


class PartitionWatch:
    """Counts arm↔partition contacts (sim stand-in for 'the partition must not collapse')."""

    def __init__(self, model):
        self.arm = {model.body(n).id for n in ARM_BODIES}
        self.box = model.body("pbox").id
        self.hits = {}

    def __call__(self, sim):
        m, d = sim.model, sim.data
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 in self.arm and b2 == self.box) or (b2 in self.arm and b1 == self.box):
                g, arm = (c.geom2, b1) if b2 == self.box else (c.geom1, b2)
                name = f"{m.body(arm).name}↔{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)}"
                self.hits[name] = self.hits.get(name, 0) + 1


def main():
    ap = argparse.ArgumentParser()
    add_args(ap)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--cells", nargs="+", metavar="R,C", help="target cells, in block order")
    a = ap.parse_args()

    box, blocks = box_from_args(a), Blocks(count=a.blocks, size=a.block_size)
    targets = ([tuple(int(v) for v in rc.split(",")) for rc in a.cells] if a.cells
               else box.cells()[:blocks.count])
    if bad := [rc for rc in targets if rc not in box.cells()]:
        raise SystemExit(f"no such cell {bad} in a {box.rows}x{box.cols} box")
    if len(targets) < blocks.count:
        raise SystemExit(f"{blocks.count} blocks but only {len(targets)} target cells")

    sim = Sim(a.headless, build(box, blocks))
    watch = PartitionWatch(sim.model)
    sim.on_step = watch
    ik = arm_model()
    rotations = tool_rotations(ik)
    realtime = not a.headless

    sim.run(0.5, realtime)
    q, grip, seed = sim.jm.q_rad(sim.data), 0.0, np.radians(GRASP_SEED_DEG)
    last_tip = None
    for bid, rc in zip(blocks.ids(), targets):
        steps = plan(box, blocks, sim.body_xy(bid), box.cell_xy(*rc))
        label, qs = solve_block(ik, rotations, steps, seed, last_tip,
                                lean_mid=(box.height - steps[5][1][2]) / 2)  # middle of the in-cell span
        if label is None:
            _, name, tip, mm = qs
            print(f"{bid} -> cell {rc}: unreachable at [{name}] tip ({tip[0]:.3f}, {tip[1]:.3f}, "
                  f"{tip[2]:.3f}), {mm:.0f} mm short — move the box (--center) or lower it (--height)")
            continue
        print(f"{bid} -> cell {rc}  ({label})")
        for (name, tip, want_grip, secs), path in zip(steps, qs):
            g0 = grip
            for k, q_to in enumerate(path, 1):
                g1 = g0 + (want_grip - g0) * k / len(path)
                sim.ramp(q, q_to, grip, g1, secs / len(path), realtime)
                q, grip = q_to, g1
            seed, last_tip = q, np.asarray(tip, float)
    sim.run(1.0, realtime)

    print("\n=== result ===")
    ok = 0
    for bid, rc in zip(blocks.ids(), targets):
        x, y, z = sim.data.body(bid).xpos
        got = box.cell_of(x, y)
        seated = got == rc and z < box.height / 2
        ok += seated
        print(f"  {bid}: want {rc}, in {got}, z={z * 100:.1f} cm -> {'SEATED ✅' if seated else 'FAIL ❌'}")
    print(f"seated {ok}/{blocks.count}")
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
