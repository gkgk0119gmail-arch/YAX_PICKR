"""PAC 2026 scene — PiPER + N×M partition box + blocks, on top of wego-piper-web's base MJCF.

The repo's `preset:bin` can't express a cell count (its params are validated to 1e-4..1 m),
so the partition box is added straight onto the MjSpec after `scene_spec.compose`.
Every piece is a box primitive: MuJoCo collides meshes as convex hulls, so a concave
box would be solid in physics (see scene_spec._bin_geoms).

Box frame: origin = centre of the box bottom on the table, rows along +x (away from
the robot), cols along +y. Cell (r, c) = row r, col c, both from 0.

Run:  ./sim_tools/run.sh pac_scene.py [--rows 3 --cols 3 --cell 0.08]   # just look
"""
import argparse
import math
from dataclasses import dataclass, field

import mujoco

from piper_sim import scene_spec


@dataclass
class PartitionBox:
    rows: int = 4
    cols: int = 4
    cell: float = 0.07          # inner cell size (m), square
    height: float = 0.10        # wall & divider height (m) — perception's BOX_HEIGHT assumption
    wall_t: float = 0.004       # outer wall thickness (m)
    div_t: float = 0.003        # divider thickness (m)
    floor_t: float = 0.003      # floor thickness (m)
    center: tuple = (0.23, -0.12)  # box centre (m) — on the −y side, close in: far cells are at the reach limit
    yaw_deg: float = 0.0

    @property
    def inner(self):
        """Inner size (x, y) in metres."""
        return (self.rows * self.cell + (self.rows - 1) * self.div_t,
                self.cols * self.cell + (self.cols - 1) * self.div_t)

    def _local_cell(self, r, c):
        ix, iy = self.inner
        pitch = self.cell + self.div_t
        return (-ix / 2 + self.cell / 2 + r * pitch, -iy / 2 + self.cell / 2 + c * pitch)

    def cell_xy(self, r, c):
        """World (x, y) of cell (r, c)'s centre."""
        lx, ly = self._local_cell(r, c)
        a = math.radians(self.yaw_deg)
        return (self.center[0] + lx * math.cos(a) - ly * math.sin(a),
                self.center[1] + lx * math.sin(a) + ly * math.cos(a))

    def cells(self):
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]

    def cell_of(self, x, y):
        """(r, c) whose inner square contains world (x, y), else None."""
        a = math.radians(self.yaw_deg)
        dx, dy = x - self.center[0], y - self.center[1]
        lx, ly = dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a)
        for r, c in self.cells():
            cx, cy = self._local_cell(r, c)
            if abs(lx - cx) <= self.cell / 2 and abs(ly - cy) <= self.cell / 2:
                return r, c
        return None

    def geoms(self):
        """(name, half-size, local pos) for every piece."""
        ix, iy = self.inner
        H, t, d, f = self.height, self.wall_t, self.div_t, self.floor_t
        out = [("floor", [ix / 2 + t, iy / 2 + t, f / 2], [0, 0, f / 2]),
               ("wall_xp", [t / 2, iy / 2 + t, H / 2], [ix / 2 + t / 2, 0, H / 2]),
               ("wall_xn", [t / 2, iy / 2 + t, H / 2], [-ix / 2 - t / 2, 0, H / 2]),
               ("wall_yp", [ix / 2, t / 2, H / 2], [0, iy / 2 + t / 2, H / 2]),
               ("wall_yn", [ix / 2, t / 2, H / 2], [0, -iy / 2 - t / 2, H / 2])]
        pitch = self.cell + d
        for k in range(1, self.rows):     # dividers between rows (planes of constant x)
            out.append((f"div_r{k}", [d / 2, iy / 2, H / 2], [-ix / 2 + k * pitch - d / 2, 0, H / 2]))
        for k in range(1, self.cols):     # dividers between cols (planes of constant y)
            out.append((f"div_c{k}", [ix / 2, d / 2, H / 2], [0, -iy / 2 + k * pitch - d / 2, H / 2]))
        return out


@dataclass
class Blocks:
    """Blocks to pick, laid out on the +y side of the table (reachable at carry height)."""
    count: int = 5
    size: float = 0.04          # edge length (m) — fingers are 8 mm thick, so size + 16 mm must fit a cell
    mass: float = 0.03
    rgba: tuple = (0.85, 0.2, 0.15, 1.0)
    #: pick spots — where a straight-down grasp still reaches at carry height (x ≲ 0.32)
    spots: list = field(default_factory=lambda: [
        (0.18, 0.08), (0.25, 0.08), (0.18, 0.15), (0.25, 0.15), (0.32, 0.08),
        (0.18, 0.22), (0.25, 0.22), (0.32, 0.15)])
    #: full edge lengths (x, y, z) in the block's own frame; None → a `size` cube.
    #: e.g. (0.08, 0.03, 0.03) is a 3×3×8 cm prism lying with its long axis along local x
    dims: tuple | None = None
    #: per-spot yaw (deg) — which way a lying prism points
    yaws: list | None = None

    def ids(self):
        return [f"block{i}" for i in range(self.count)]

    def half(self):
        return [d / 2 for d in self.dims] if self.dims else [self.size / 2] * 3


def build(box: PartitionBox | None = None, blocks: Blocks | None = None):
    """Compile the MjModel. Blocks go through scene_spec (free joints `<id>_free`, the
    physics defaults the repo tuned for grasping); the partition box is static."""
    box, blocks = box or PartitionBox(), blocks or Blocks()
    if blocks.count > len(blocks.spots):
        raise ValueError(f"only {len(blocks.spots)} block spots defined")
    spec = {"version": 1, "id": "pac", "name": "PAC partition box", "objects": [
        {"id": bid, "label": bid, "shape": "box", "size": blocks.half(),
         "pos": [x, y, blocks.half()[2]], "euler_deg": [0, 0, yaw],
         "rgba": list(blocks.rgba), "mass": blocks.mass, "movable": True}
        for bid, (x, y), yaw in zip(blocks.ids(), blocks.spots, blocks.yaws or [0] * len(blocks.spots))]}
    s = scene_spec.compose(spec)

    body = s.worldbody.add_body(name="pbox", pos=[*box.center, 0],
                                quat=scene_spec.euler_quat([0, 0, box.yaw_deg]))
    for name, size, pos in box.geoms():
        body.add_geom(name=f"pbox_{name}", type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos,
                      rgba=[0.76, 0.6, 0.42, 1] if name.startswith(("floor", "wall")) else [0.86, 0.72, 0.52, 1],
                      **scene_spec.STATIC_PHYSICS)
    for r, c in box.cells():             # visual-only markers on each cell floor
        lx, ly = box._local_cell(r, c)
        body.add_site(name=f"cell_{r}_{c}", pos=[lx, ly, box.floor_t + 0.0005],
                      size=[0.006, 0.006, 0.0005], type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      rgba=[0.2, 0.6, 0.2, 0.6])
    return s.compile()


def box_from_args(a) -> PartitionBox:
    return PartitionBox(rows=a.rows, cols=a.cols, cell=a.cell, height=a.height,
                        center=tuple(a.center), yaw_deg=a.yaw)


def add_args(ap: argparse.ArgumentParser):
    d = PartitionBox()
    ap.add_argument("--rows", type=int, default=d.rows)
    ap.add_argument("--cols", type=int, default=d.cols)
    ap.add_argument("--cell", type=float, default=d.cell, help="inner cell size (m)")
    ap.add_argument("--height", type=float, default=d.height, help="wall height (m)")
    ap.add_argument("--center", type=float, nargs=2, default=list(d.center), metavar=("X", "Y"))
    ap.add_argument("--yaw", type=float, default=d.yaw_deg, help="box yaw (deg)")
    ap.add_argument("--blocks", type=int, default=Blocks().count)
    ap.add_argument("--block-size", type=float, default=Blocks().size)


if __name__ == "__main__":
    import mujoco.viewer

    ap = argparse.ArgumentParser()
    add_args(ap)
    a = ap.parse_args()
    box = box_from_args(a)
    model = build(box, Blocks(count=a.blocks, size=a.block_size))
    ix, iy = box.inner
    print(f"{box.rows}x{box.cols} box, inner {ix * 100:.1f} x {iy * 100:.1f} cm, "
          f"outer {(ix + 2 * box.wall_t) * 100:.1f} x {(iy + 2 * box.wall_t) * 100:.1f} cm, "
          f"h {box.height * 100:.0f} cm at {box.center}")
    mujoco.viewer.launch(model, mujoco.MjData(model))
