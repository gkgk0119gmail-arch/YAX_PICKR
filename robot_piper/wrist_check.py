"""Hover check with the wrist D405, 5 cm above a block's top face, before going down to grasp it.

From the D405 depth (in the base frame through wrist_cam_to_flange.json and the arm's current pose):
  - plane RANSAC on the block top  -> angle between the gripper axis and the top-face normal (vertical?)
  - rectangle RANSAC on the top    -> long-side centre line, its middle, the jaw direction
  - compared with the plan          -> centre offset (mm), jaw angle difference (deg)
check() returns these numbers plus an annotated D405 picture.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "robot_piper"), str(ROOT / "perception-main"), str(ROOT / "perception-main" / "research" / "box_cells")]
import arm_ik  # noqa: E402
from block_grasp import long_side_grasp, plane_ransac  # noqa: E402
from rect_fit import ransac_rect, refine_rect  # noqa: E402

T_FLANGE_CAM = np.array(json.loads((ROOT / "pac2026" / "calib" / "wrist_cam_to_flange.json").read_text())["T_flange_cam"])


def grab_wrist(cam, frames=6):
    """Median of a few aligned D405 depth frames (m) + the last colour frame."""
    if hasattr(cam, "collect"):                               # WristSeen reads the camera in the background
        ds, col = cam.collect(frames, aligned=True)
        if not ds:
            raise RuntimeError("D405: no depth frames")
        d = np.median(np.array(ds), 0)
        d[d <= 0] = np.nan
        return d, col
    ds, col = [], None
    import time
    t0 = time.time()
    while len(ds) < frames and time.time() - t0 < 4:
        fs = cam.pipe.poll_for_frames()
        if fs:
            fs = cam.align.process(fs)
            d, c = fs.get_depth_frame(), fs.get_color_frame()
            if d and c:
                ds.append(np.asanyarray(d.get_data()).astype(np.float32) * cam.scale)
                col = np.asanyarray(c.get_data()).copy()
        time.sleep(0.005)
    d = np.median(np.array(ds), 0)
    d[d <= 0] = np.nan
    return d, col


def truncation_fix(top_xy, top_px, depth, centre, jaw, l_short, l_long, nominal):
    """The top face may run off the D405 picture or under a finger (both hide one end of it): then the
    rectangle is too short on that axis and its centre is pulled toward the visible part. Per rectangle axis:
    if only one end touches the picture border / a finger and the measured side is short, the centre is put
    half the nominal side length in from the OTHER (complete) end. Returns (centre, info dict)."""
    H, W = depth.shape
    zt = float(np.median(depth[top_px[:, 1], top_px[:, 0]]))
    blocked = np.zeros((H, W), np.uint8)
    blocked[:8, :], blocked[-8:, :], blocked[:, :8], blocked[:, -8:] = 1, 1, 1, 1      # picture border
    near = np.isfinite(depth) & (depth > 0.02) & (depth < 0.7 * zt)                    # fingers (closer)
    blocked |= near.astype(np.uint8)
    blocked = cv2.dilate(blocked, np.ones((13, 13), np.uint8)).astype(bool)
    touch = blocked[top_px[:, 1], top_px[:, 0]]
    info = dict(touching_points=int(touch.sum()))
    if touch.sum() < 15:
        return centre, info
    perp = np.array([-jaw[1], jaw[0]])
    new = np.asarray(centre, float).copy()
    for name, a, L_meas, L_nom in (("across", jaw, l_short, nominal[0]), ("along", perp, l_long, nominal[1])):
        s_all = (top_xy - centre) @ a
        s_t = s_all[touch]
        plus, minus = (s_t > 0.3 * L_meas).sum() > 10, (s_t < -0.3 * L_meas).sum() > 10
        if L_meas > L_nom - 0.008 or plus == minus:                   # full length seen, or both / no ends cut
            info[name] = "both ends cut" if plus and minus else None
            continue
        if plus:                                                      # + end hidden: measure from the - end
            shift = (s_all.min() + L_nom / 2)
            info[name] = "+ end cut"
        else:
            shift = (s_all.max() - L_nom / 2)
            info[name] = "- end cut"
        new = new + shift * a
    info["fix_mm"] = np.round((new - centre) * 1000, 1).tolist()
    return new, info


def check(cam, q_now, tcp, plan_xy, plan_top_z, plan_jaw_deg, nominal=None):
    pos, R = arm_ik.fk(q_now)
    T_base_flange = np.eye(4)
    T_base_flange[:3, :3], T_base_flange[:3, 3] = R, pos / 1000
    T = T_base_flange @ T_FLANGE_CAM                                  # base <- D405
    depth, col = grab_wrist(cam)
    H, W = depth.shape
    vv, uu = np.mgrid[0:H:2, 0:W:2]
    d = depth[::2, ::2]
    ok = np.isfinite(d) & (d > 0.03) & (d < 0.5)
    K = cam.K
    Pc = np.stack([(uu[ok] - K[0, 2]) / K[0, 0] * d[ok], (vv[ok] - K[1, 2]) / K[1, 1] * d[ok], d[ok]], 1)
    Pb = Pc @ T[:3, :3].T + T[:3, 3]
    pix = np.stack([uu[ok], vv[ok]], 1)
    # the top may be lower than planned (block lying on its side: 90 mm instead of 120) -> 4.5 cm down
    rxy = np.hypot(Pb[:, 0] - plan_xy[0], Pb[:, 1] - plan_xy[1])
    dzp = Pb[:, 2] - plan_top_z
    sel = (rxy < 0.08) & (dzp > -0.045) & (dzp < 0.03)
    Q = Pb[sel]
    rq = rxy[sel]
    pq = pix[sel]
    out = dict(points=int(sel.sum()))
    if len(Q) < 200:
        out["ok"] = False
        out["why"] = f"D405 sees only {len(Q)} top-face points"
        return out, col
    # plane from the middle of the planned spot first (a neighbour of another height cannot win), then
    # every point of the window on that plane
    mid = Q[rq < 0.035]
    n, c, inl = plane_ransac(mid if len(mid) >= 150 else Q, tol=0.004, iters=1000, max_tilt_deg=30)
    inl = np.abs((Q - c) @ n) < 0.006
    top = Q[inl]
    ptop = pq[inl]
    tool_axis = R[:, 2]                                               # flange z = gripper axis (points down)
    vert_err = float(np.degrees(np.arccos(np.clip(abs(tool_axis @ n), -1, 1))))
    tool_tilt = float(np.degrees(np.arccos(np.clip(-tool_axis[2], -1, 1))))
    top_tilt = float(np.degrees(np.arccos(np.clip(n[2], -1, 1))))
    xy = top[:, :2] * 1000
    o = xy.min(0) - 10
    g = np.zeros((int(np.ptp(xy[:, 1])) + 21, int(np.ptp(xy[:, 0])) + 21), np.uint8)
    u = (xy - o).astype(int)
    g[u[:, 1], u[:, 0]] = 1
    g = cv2.morphologyEx(cv2.dilate(g, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    # a neighbouring block of the same height can fall inside the search window: keep only the piece that
    # holds (or is nearest to) the planned centre, not the largest / merged one
    n_lab, lab, stats, cents = cv2.connectedComponentsWithStats(g)
    pc = (np.asarray(plan_xy) * 1000 - o).astype(int)
    inside_lab = lab[pc[1], pc[0]] if 0 <= pc[1] < g.shape[0] and 0 <= pc[0] < g.shape[1] else 0
    if inside_lab > 0:
        keep = inside_lab
    else:
        dist = np.hypot(*(cents[1:] - pc).T)
        keep = 1 + int(np.argmin(dist))
        if dist.min() > 45:                                           # only a neighbour is seen, not the target
            out.update(ok=False, pieces=int(n_lab - 1),
                       why=f"target top not seen (nearest flat piece {dist.min():.0f} mm away)")
            return out, col
    out["pieces"] = int(n_lab - 1)
    g = (lab == keep).astype(np.uint8)
    keep_pts = g[np.clip(u[:, 1], 0, g.shape[0] - 1), np.clip(u[:, 0], 0, g.shape[1] - 1)] > 0
    top, ptop = top[keep_pts], ptop[keep_pts]
    cnt = max(cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)[:, 0, :]
    _, _, _, ratio, corners = ransac_rect(cnt, iters=1000, tol=2.0)
    ref = refine_rect(cnt, corners)
    if ref is not None:
        corners = ref[4]
    corners = (corners + o) / 1000
    centre, jaw, l_short, l_long, cline = long_side_grasp(corners)
    centre_rect = centre
    if nominal is not None:                                           # one end hidden -> measure from the other
        centre, trunc = truncation_fix(top[:, :2], ptop, depth, centre, jaw, l_short, l_long, nominal)
        out["truncation"] = trunc
        if np.hypot(*(centre - centre_rect)) > 0:
            cline = cline + (centre - centre_rect)
    jaw_deg = float(np.degrees(np.arctan2(jaw[1], jaw[0]))) % 180
    dj = abs((jaw_deg - plan_jaw_deg + 90) % 180 - 90)
    if l_long - l_short <= 0.015:      # square top (collide.SQUARE_TOL_M): a 90 deg turn is the same block;
        dj = min(dj, 90 - dj)          # a rectangular top turned 90 deg would be gripped across the long side
    off = (centre - np.asarray(plan_xy)) * 1000
    out.update(ok=True, vertical_err_deg=round(vert_err, 1), tool_tilt_deg=round(tool_tilt, 1),
               top_tilt_deg=round(top_tilt, 1), top_z_m=round(float(c[2]), 4),
               centre_xy_m=centre.round(4).tolist(), offset_mm=off.round(1).tolist(), jaw_deg=round(jaw_deg, 1),
               jaw_diff_deg=round(float(dj), 1), size_mm=[round(l_short * 1000), round(l_long * 1000)],
               rect_fit=round(float(ratio), 2), inliers=int(inl.sum()),
               top_normal_base=np.round(n, 4).tolist(), top_normal_cam=np.round(T[:3, :3].T @ n, 4).tolist())

    # picture: project the found top face, centre line and the gripper axis point into the D405 colour image
    Ti = np.linalg.inv(T)

    def px(P):
        pc = Ti[:3, :3] @ np.asarray(P) + Ti[:3, 3]
        return int(pc[0] / pc[2] * K[0, 0] + K[0, 2]), int(pc[1] / pc[2] * K[1, 1] + K[1, 2])
    vis = col.copy()
    z = c[2]
    cv2.polylines(vis, [np.array([px([*q, z]) for q in corners], np.int32)], True, (255, 0, 0), 2)
    cv2.line(vis, px([*cline[0], z]), px([*cline[1], z]), (0, 255, 255), 2)
    cv2.drawMarker(vis, px([*centre, z]), (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
    if np.hypot(*(centre - centre_rect)) > 1e-4:                      # before the hidden-end fix (magenta)
        cv2.drawMarker(vis, px([*centre_rect, z]), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
    tip = (pos + R @ [0, 0, tcp * 1000]) / 1000
    axis_hit = tip + tool_axis * max((tip[2] - z) / max(-tool_axis[2], 1e-6), 0)   # gripper axis meets the top
    cv2.drawMarker(vis, px(axis_hit), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 24, 3)
    lines = [f"gripper vs top-face normal: {vert_err:.1f} deg  (tool tilt {tool_tilt:.1f}, top tilt {top_tilt:.1f})",
             f"centre offset {off[0]:+.0f}, {off[1]:+.0f} mm   jaw diff {dj:.0f} deg   top {l_short * 1000:.0f}x{l_long * 1000:.0f} mm",
             "blue top face / yellow centre line / red centre / green gripper axis"]
    tr = out.get("truncation") or {}
    if tr.get("fix_mm"):
        lines.append(f"top face partly hidden ({tr.get('across')}, {tr.get('along')}): centre moved {tr['fix_mm']} mm "
                     f"from the cut rectangle (magenta)")
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 26 * len(lines) + 8), (0, 0, 0), -1)
    for k, s in enumerate(lines):
        cv2.putText(vis, s, (8, 22 + 26 * k), 0, 0.55, (255, 255, 255), 1)
    return out, vis


def check_cell(cam, q_now, tcp, cell_xy, cell_floor_z, rim_z, pitch, axis_deg, half_uv=None):
    """Above the target cell with a block in the gripper: find the cell's walls / dividers in the D405
    depth and return the measured cell centre (and the held block's visible offset, for the log).

    Wall-top points (between the cell floor + 2 cm and the rim + 2 cm, so the held block above is out)
    near the planned cell are put in the cell frame (u along the basket axis, v across). Each side of
    the cell (u = +-pitch/2, v = +-pitch/2) is a robust median of its points; the centre follows from
    the opposite sides (or one side and the pitch)."""
    pos, R = arm_ik.fk(q_now)
    T_bf = np.eye(4)
    T_bf[:3, :3], T_bf[:3, 3] = R, pos / 1000
    T = T_bf @ T_FLANGE_CAM
    depth, col = grab_wrist(cam)
    H, W = depth.shape
    vv, uu = np.mgrid[0:H:2, 0:W:2]
    d = depth[::2, ::2]
    ok = np.isfinite(d) & (d > 0.03) & (d < 0.6)
    K = cam.K
    Pc = np.stack([(uu[ok] - K[0, 2]) / K[0, 0] * d[ok], (vv[ok] - K[1, 2]) / K[1, 1] * d[ok], d[ok]], 1)
    Pb = Pc @ T[:3, :3].T + T[:3, 3]
    a = np.radians(axis_deg)
    eu, ev = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    rel = Pb[:, :2] - np.asarray(cell_xy)
    u, v = rel @ eu, rel @ ev
    zsel = (Pb[:, 2] > cell_floor_z + 0.02) & (Pb[:, 2] < rim_z + 0.02)
    hu, hv = half_uv if half_uv is not None else (pitch / 2, pitch / 2)     # rectangular cells: two half sizes
    half = pitch / 2
    out = dict(ok=False)
    sides = {}
    for name, coord, other, sign, h_this, h_other in (("u+", u, v, 1, hu, hv), ("u-", u, v, -1, hu, hv),
                                                      ("v+", v, u, 1, hv, hu), ("v-", v, u, -1, hv, hu)):
        sel = zsel & (np.abs(coord - sign * h_this) < 0.03) & (np.abs(other) < h_other - 0.01)
        if sel.sum() >= 30:
            sides[name] = float(np.median(coord[sel]))
    du = dv = None
    if "u+" in sides and "u-" in sides:
        du = (sides["u+"] + sides["u-"]) / 2
    elif sides.get("u+") is not None:
        du = sides["u+"] - hu
    elif sides.get("u-") is not None:
        du = sides["u-"] + hu
    if "v+" in sides and "v-" in sides:
        dv = (sides["v+"] + sides["v-"]) / 2
    elif sides.get("v+") is not None:
        dv = sides["v+"] - hv
    elif sides.get("v-") is not None:
        dv = sides["v-"] + hv
    out["sides_found"] = sorted(sides)
    if du is None and dv is None:
        out["why"] = "no cell wall seen"
        return out, col
    shift = (du or 0.0) * eu + (dv or 0.0) * ev
    out.update(ok=True, shift_mm=np.round(shift * 1000, 1).tolist(),
               centre_xy_m=(np.asarray(cell_xy) + shift).round(4).tolist())
    # held block, for the log: its visible points (above the rim) around the fingertips
    tip = (pos + R @ [0, 0, tcp * 1000]) / 1000
    hb = (Pb[:, 2] > rim_z + 0.02) & (np.hypot(*(Pb[:, :2] - tip[:2]).T) < 0.07)
    if hb.sum() > 50:
        out["held_block_offset_mm"] = np.round((np.median(Pb[hb, :2], 0) - tip[:2]) * 1000, 1).tolist()
    # picture
    Ti = np.linalg.inv(T)

    def px(P):
        pc = Ti[:3, :3] @ np.asarray(P) + Ti[:3, 3]
        return int(pc[0] / pc[2] * K[0, 0] + K[0, 2]), int(pc[1] / pc[2] * K[1, 1] + K[1, 2])
    vis = col.copy()
    for c, colr in ((np.asarray(cell_xy), (0, 255, 255)), (np.asarray(cell_xy) + shift, (0, 0, 255))):
        corners = [c + s1 * hu * eu + s2 * hv * ev for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        cv2.polylines(vis, [np.array([px([*q, rim_z]) for q in corners], np.int32)], True, colr, 2)
        cv2.drawMarker(vis, px([*c, rim_z]), colr, cv2.MARKER_CROSS, 20, 2)
    cv2.putText(vis, f"cell shift {out['shift_mm']} mm  sides {out['sides_found']}  (yellow plan, red D405)",
                (8, 22), 0, 0.5, (255, 255, 255), 1)
    return out, vis
