"""Data passed between pipeline steps."""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Frame:
    color: np.ndarray                      # BGR, colour-camera pixels
    depth_m: Optional[np.ndarray] = None   # RealSense depth (its own pixels, not registered to colour)


@dataclass
class BoxInfo:
    state: str                             # partition_in_box | partition_only | empty_box | nothing
    box_xyxy: Optional[list] = None        # colour px
    partition_xyxy: Optional[list] = None
    height_m: Optional[float] = None       # rim height above the table (RealSense p99, VGGT if no depth)
    height_vggt_m: Optional[float] = None
    vggt_depth_m: Optional[np.ndarray] = field(default=None, repr=False)   # metric VGGT depth (VGGT grid)
    roi_vggt: Optional[list] = None        # box ROI on the VGGT grid


@dataclass
class CellInfo:
    id: int
    row: int
    col: int
    quad_px: list                          # 4 corners in colour px (tl, tr, br, bl)
    center_px: tuple
    yaw_deg: float = 0.0                   # cell orientation in the image
    state: str = "unknown"                 # empty | seated | straddle | uncertain | out_of_view
    conf: float = 0.0


@dataclass
class Partition:
    rows: int
    cols: int
    cells: list

    def empty_cells(self):
        return [c for c in self.cells if c.state == "empty"]


@dataclass
class Grasp:
    center_px: tuple
    jaw_deg: float                         # gripper closing axis in the image
    opening_px: float
    length_px: float
    score: float
    box_xyxy: list
