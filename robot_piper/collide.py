"""3D collision check for the gripper (+ wrist D405, + a carried block) against the table, blocks and
the basket walls / dividers. Everything is sampled as points; obstacles are boxes / wall slabs.

Gripper model in the flange frame (z = tool axis, tip at TCP): a body box, two fingers whose gap is the
current opening, the D405 housing (calibrated position). Dimensions are measured roughly on site and
padded by MARGIN, so the check errs on the safe side.
"""
import json
from pathlib import Path

import numpy as np

import arm_ik

ROOT = Path(__file__).resolve().parents[1]
MARGIN = 0.008                                       # (real grasps worked with ~1 cm between blocks)
BLOCK_HALF = 0.045                                   # 90 x 90 x 120 mm blocks (bubble wrap: +-) when no size is known
BLOCK_PAD = 0.006                                    # the top-face fit reads ~5-10 mm small on bubble wrap
SQUARE_TOL_M = 0.015                                 # top sides differing less than this: a square top (fit noise
                                                     # on square tops is <= 8 mm; a block lying on its side: 23-30 mm)
WALL_HALF = 0.006                                    # basket wall / divider half thickness
T_FLANGE_D405 = np.array(json.loads((ROOT / "pac2026" / "calib" / "wrist_cam_to_flange.json").read_text())["T_flange_cam"])


def _box_pts(c, half, step=0.012):
    """Points on the surface of an axis-aligned box (centre c, half sizes)."""
    axes = [np.arange(-h, h + 1e-9, step) if h > 0 else np.array([0.0]) for h in half]
    axes = [np.unique(np.r_[a, -h, h]) for a, h in zip(axes, half)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    on = np.any(np.isclose(np.abs(g), half, atol=1e-6), axis=1)
    return g[on] + c


def gripper_points(tcp, opening):
    """Flange-frame points: body, both fingers (gap = opening), D405 housing."""
    body = _box_pts([0, 0, 0.053], [0.035, 0.065, 0.0145])          # mount face (z 0.0386) .. finger base
    fz0, fz1 = 0.068, tcp                                            # fingers ~60 mm
    fingers = [_box_pts([0, s * (opening / 2 + 0.006), (fz0 + fz1) / 2], [0.02, 0.006, (fz1 - fz0) / 2])
               for s in (-1, 1)]
    cam_c = T_FLANGE_D405[:3, 3]
    cam = (T_FLANGE_D405[:3, :3] @ _box_pts([0, 0, -0.01], [0.022, 0.022, 0.014]).T).T + cam_c
    return np.vstack([body, *fingers, cam])


def is_square(size_m):
    """Top face (short, long side, m) about square? Then either side is the short one."""
    return size_m is None or size_m[1] - size_m[0] <= SQUARE_TOL_M


def grip_step(size_m):
    """Tool yaws that grip the block across its SHORT top side repeat every 180 deg; on a square top every 90."""
    return 90.0 if is_square(size_m) else 180.0


def block_half(size_m):
    """Footprint half sizes (along the jaw = across the short side, along the long side). Square tops keep the
    on-site 90 x 90 mm; a rectangular top (e.g. a block lying on its side) uses its measured sides, padded."""
    if is_square(size_m):
        return (BLOCK_HALF, BLOCK_HALF)
    return (size_m[0] / 2 + BLOCK_PAD, size_m[1] / 2 + BLOCK_PAD)


def block_points(xy, yaw_deg, table_z, height, half=None):
    """yaw_deg: the jaw direction (across the short side), half: block_half() (default: 90 x 90 mm)."""
    a = np.radians(yaw_deg)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    hx, hy = half or (BLOCK_HALF, BLOCK_HALF)
    p = _box_pts([0, 0, height / 2], [hx, hy, height / 2])
    return (R @ p.T).T + [xy[0], xy[1], table_z]


def basket_walls(cells_xy, axis_deg, pitch):
    """Wall / divider segments (2x2 xy) of a grid basket from its cell centres."""
    a = np.radians(axis_deg)
    eu, ev = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    C = np.asarray(cells_xy)
    o = C.mean(0)
    u, v = (C - o) @ eu, (C - o) @ ev

    def bounds(x):
        xs = np.sort(x)
        groups = [[xs[0]]]
        for t in xs[1:]:
            (groups[-1].append(t) if t - groups[-1][-1] < pitch / 2 else groups.append([t]))
        m = [np.mean(g) for g in groups]
        return [m[0] - pitch / 2] + [(p + q) / 2 for p, q in zip(m[:-1], m[1:])] + [m[-1] + pitch / 2]
    ub, vb = bounds(u), bounds(v)
    segs = [(o + ub_ * eu + vb[0] * ev, o + ub_ * eu + vb[-1] * ev) for ub_ in ub]
    segs += [(o + ub[0] * eu + vb_ * ev, o + ub[-1] * eu + vb_ * ev) for vb_ in vb]
    return segs


class Scene:
    def __init__(self, table_z, blocks, walls, rim_z):
        """blocks: list of dict(xy, yaw, top[, half]) (upright boxes from the table to top; yaw = jaw direction,
        half = block_half(), default 90 x 90 mm)."""
        self.table_z, self.blocks, self.walls, self.rim_z = table_z, blocks, walls, rim_z

    def clearance(self, P, skip_block=None, table_tol=0.0, walls=True):
        """Smallest clearance (m, negative = inside) of world points P to every obstacle, and its name.
        table_tol: extra allowance against the table (a carried block may rest on it at the grasp)."""
        worst, why = np.inf, ""
        c = float(np.min(P[:, 2] - self.table_z)) + table_tol
        if c < worst:
            worst, why = c, "table"
        for i, b in enumerate(self.blocks):
            if i == skip_block:
                continue
            a = np.radians(b["yaw"])
            hx, hy = b.get("half") or (BLOCK_HALF, BLOCK_HALF)
            d = P[:, :2] - b["xy"]
            u = np.abs(d @ [np.cos(a), np.sin(a)]) - hx
            v = np.abs(d @ [-np.sin(a), np.cos(a)]) - hy
            h = P[:, 2] - b["top"]
            c = float(np.min(np.max(np.stack([u, v, h]), 0)))        # >0 outside the box (by the largest gap)
            if c < worst:
                worst, why = c, f"block {i}"
        for p0, p1 in (self.walls if walls else []):
            d = p1 - p0
            t = np.clip((P[:, :2] - p0) @ d / (d @ d), 0, 1)
            dist = np.linalg.norm(P[:, :2] - (p0 + t[:, None] * d), axis=1) - WALL_HALF
            c = float(np.min(np.maximum(dist, P[:, 2] - self.rim_z)))
            if c < worst:
                worst, why = c, "basket wall"
        return worst, why


def body_world(q, tcp, opening, carried=None):
    """World points of the gripper (+ carried block given in the flange frame) for joints q."""
    pos, R = arm_ik.fk(q)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, pos / 1000
    P = gripper_points(tcp, opening)
    if carried is not None:
        P = np.vstack([P, carried])
    return P @ T[:3, :3].T + T[:3, 3], T


ENABLED = False               # 3D collision checks while moving: off (pickr.sh --collide turns them on)


def check_path(qs, tcp, opening, scene, carried=None, skip_block=None, n_between=6):
    """Worst clearance along a sequence of joint configurations (joint-interpolated between them)."""
    if not ENABLED:
        return np.inf, "not checked"
    worst, why = np.inf, ""
    seq = [qs[0]]
    for a, b in zip(qs[:-1], qs[1:]):
        seq += [a + (b - a) * f for f in np.linspace(0, 1, n_between + 1)[1:]]
    for q in seq:
        P, T = body_world(q, tcp, opening)
        c, w = scene.clearance(P, skip_block, walls=False)   # the fingers end above the 5 cm basket walls
        if carried is not None:                          # the held block may stand on the table at the grasp
            Pc = carried @ T[:3, :3].T + T[:3, 3]
            c2, w2 = scene.clearance(Pc, skip_block, table_tol=MARGIN + 0.003)
            if c2 < c:
                c, w = c2, "carried block vs " + w2
        if c < worst:
            worst, why = c, w
    return worst - MARGIN, why
