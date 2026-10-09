"""Perception result (colour px) -> MuJoCo twin scene (robot base frame, metres) for --robot sim.

The sim objects do not look like the real ones, so the twin is fed what perception found
instead of sim images: the partition grid from the cell centres, the blocks from the grasps.
Sim box frame (sim_tools/pac_scene.py): rows along local +x, cols along local +y, square cells.
"""
import numpy as np

DIV_T = 0.003            # sim divider thickness (pac_scene.PartitionBox.div_t)


def _mean_step(pts, keys, d):
    steps = [pts[k2] - pts[k] for k in keys if (k2 := (k[0] + d[0], k[1] + d[1])) in pts]
    return np.mean(steps, axis=0) if steps else None


def box_msg(box, part, calib):
    pts = {(c.row, c.col): np.array(calib.to_robot(*c.center_px)) for c in part.cells}
    keys = list(pts)
    center = np.mean(list(pts.values()), axis=0)
    dr, dc = _mean_step(pts, keys, (1, 0)), _mean_step(pts, keys, (0, 1))
    if dr is not None:
        yaw = np.arctan2(dr[1], dr[0])
    elif dc is not None:
        yaw = np.arctan2(dc[1], dc[0]) - np.pi / 2
    else:
        yaw = 0.0
    pitches = [np.linalg.norm(v) for v in (dr, dc) if v is not None]
    if pitches:
        pitch = float(np.mean(pitches))
    else:                                                    # 1x1: cell size from its quad
        q = np.array([calib.to_robot(*p) for p in part.cells[0].quad_px])
        pitch = float(np.mean(np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1))) + DIV_T
    return {"center": [float(center[0]), float(center[1])], "yaw_deg": float(np.degrees(yaw)),
            "rows": part.rows, "cols": part.cols, "cell": pitch - DIV_T, "height": float(box.height_m or 0.12)}       # 4x4 rim ~12 cm (RealSense)


def m_per_px(calib, u, v):
    x0, y0 = calib.to_robot(u, v)
    x1, y1 = calib.to_robot(u + 1, v)
    x2, y2 = calib.to_robot(u, v + 1)
    return (np.hypot(x1 - x0, y1 - y0) + np.hypot(x2 - x0, y2 - y0)) / 2


def blocks_msg(grasps, calib, block_h):
    out = []
    for g in grasps:
        u, v = g.center_px
        s = m_per_px(calib, u, v)
        jaw = calib.yaw_to_robot(u, v, g.jaw_deg)
        # the jaw closes across the short side -> the block's local y (short) lies along the jaw
        out.append({"xy": list(calib.to_robot(u, v)), "yaw_deg": float(np.degrees(jaw) - 90.0),
                    "dims": [float(g.length_px * s), float(g.opening_px * s), float(block_h)]})
    return out


def scene_msg(box, part, grasps, calib, block_h):
    return {"op": "scene", "box": box_msg(box, part, calib), "blocks": blocks_msg(grasps, calib, block_h)}
