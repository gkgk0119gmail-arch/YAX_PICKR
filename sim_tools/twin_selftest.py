"""Run the real pac2026 Pipeline against the MuJoCo twin with FAKE perception (no cameras, no models).

Checks the twin link end to end where the perception venv is not installed: the fake
perception returns the same dataclasses as pac/perception.py, in colour pixels of the mock
calibration, for a box / blocks layout given in robot metres.

  ./sim_tools/run.sh twin_server.py --headless &      # or with the viewer
  ./sim_tools/run.sh twin_selftest.py [--rows 4 --cols 4 --cell 0.10 --height 0.12 --center 0.34 -0.12]
"""
import argparse
import math
import sys
import types
from pathlib import Path

import numpy as np

PAC = Path(__file__).resolve().parents[1] / "pac2026"
sys.path.insert(0, str(PAC))
from pac.types import BoxInfo, CellInfo, Frame, Grasp, Partition  # noqa: E402
from pac_scene import Blocks, PartitionBox  # noqa: E402

IMG = (720, 1280)
BLOCK_SPOTS = Blocks().spots
BOX = PartitionBox()


def fake_perception(a, calib):
    Hinv = np.linalg.inv(calib.H)

    def px(x, y):
        p = Hinv @ [x, y, 1.0]
        return float(p[0] / p[2]), float(p[1] / p[2])

    s = 1 / 0.0006 * 0.001                                   # px per mm of the mock calibration
    pitch, yaw = a.cell + 0.003, math.radians(a.yaw)
    cells = []
    for r in range(a.rows):
        for c in range(a.cols):
            lx, ly = (r - (a.rows - 1) / 2) * pitch, (c - (a.cols - 1) / 2) * pitch
            x = a.center[0] + lx * math.cos(yaw) - ly * math.sin(yaw)
            y = a.center[1] + lx * math.sin(yaw) + ly * math.cos(yaw)
            h = a.cell / 2
            quad = [px(x + dx, y + dy) for dx, dy in ((-h, -h), (-h, h), (h, h), (h, -h))]
            cells.append(CellInfo(len(cells) + 1, r, c, quad, px(x, y), state="empty", conf=0.99))
    grasps = [Grasp(px(x, y), 0.0, a.block * 1000 * s, a.block * 1000 * s, 0.9, [0, 0, 0, 0])
              for x, y in BLOCK_SPOTS[:a.blocks]]

    P = types.ModuleType("pac.perception")
    P.scan_box = lambda f, hub, cfg, log: BoxInfo("partition_in_box", height_m=a.height, height_vggt_m=a.height)
    P.split_partition = lambda f, box, hub, cfg, log: Partition(a.rows, a.cols, cells)
    P.find_blocks = lambda f, box, hub, cfg, log: list(grasps)
    P.cell_state = lambda *x: ("seated", 1.0)
    P.cells_overlay = lambda cells: []
    return P


class FakeCams:
    static = False

    def top(self, with_depth=True):
        return Frame(np.zeros((*IMG, 3), np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--cell", type=float, default=BOX.cell)
    ap.add_argument("--height", type=float, default=BOX.height)
    ap.add_argument("--center", type=float, nargs=2, default=list(BOX.center))
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--block", type=float, default=0.04, help="block edge (m)")
    ap.add_argument("--safe-margin", type=float, help="override motion.safe_margin_m (m)")
    ap.add_argument("--sim-host", default="localhost")
    a = ap.parse_args()

    import yaml
    from pac.calib import PixelToRobot
    from pac.runlog import RunLog

    cfg = yaml.safe_load((PAC / "config.yaml").read_text())
    cfg["robot"].update(type="sim", sim_host=a.sim_host)
    cfg["motion"]["block_height_m"] = a.block
    if a.safe_margin is not None:
        cfg["motion"]["safe_margin_m"] = a.safe_margin
    calib = PixelToRobot.load_or_mock(Path("/nonexistent"), 0.0, IMG)
    sys.modules["pac.perception"] = fake_perception(a, calib)
    from pac.pipeline import Pipeline
    from pac.robot import make_robot

    log = RunLog(Path(__file__).resolve().parent / "twin_runs")
    pipe = Pipeline(cfg, FakeCams(), None, make_robot(cfg, log), calib, log)
    print(pipe.run(a.blocks))


if __name__ == "__main__":
    main()
