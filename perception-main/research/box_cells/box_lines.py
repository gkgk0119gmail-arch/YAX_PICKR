"""Split a divided box into compartments: Canny edges on depth + sequential RANSAC lines.

Pipeline
  1. RANSAC table plane -> height above the table (removes camera tilt)
  2. box outline (largest raised blob) -> ROI and the box's two axis directions
  3. Canny on a height slice around the expected box height (BOX_HEIGHT)
  4. sequential RANSAC: fit a line, remove its inliers, repeat (only lines within
     ANGLE_TOL of a box axis are accepted)
  5. lines are grouped per axis by their offset across the box; a wall or divider
     gives several parallel edges (both top edges, base of a visible inner face),
     so each group is one physical wall/divider. Outer groups = walls.
  6. each cell is bounded by the cell-facing line of its four neighbouring groups;
     the number of rows/cols follows from the number of groups.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from box_grid import find_box_outline, fit_table_plane

ANGLE_TOL = np.deg2rad(10)   # accepted deviation from a box axis
INLIER_PX = 2.5              # RANSAC inlier distance (2.5 px also follows slightly bent dividers)
MIN_SUPPORT = 0.35           # a line needs inliers >= this fraction of the box side
GROUP_GAP = 0.10             # edges closer than this (fraction of side) belong to one wall/divider
MAX_LINES = 40
BOX_HEIGHT = 0.10            # m, expected wall/divider height above the table


@dataclass
class Line:
    p: np.ndarray        # point on the line
    d: np.ndarray        # unit direction
    n_inliers: int
    axis: int            # 0: runs along box x-axis (horizontal), 1: along box y-axis (vertical)
    offset: float = 0.0  # position across the box, px from the tl corner


@dataclass
class Cell:
    row: int
    col: int
    corners_px: np.ndarray   # 4x2 quad in the image
    center_px: tuple
    floor_m: float


def height_image(depth_m):
    d = np.where(depth_m > 0, depth_m, np.nan).astype(np.float32)
    height = fit_table_plane(d) - d
    return d, height


def canny_edges(height, roi, box_height=BOX_HEIGHT, lo=20, hi=60):
    """Canny on a height slice: <30% of box height -> 0, >70% -> 255.

    Only wall/divider tops survive the slice, so each side of a divider gives one
    clean edge at mid-height and low things on the cell floors are ignored.
    """
    s0, s1 = 0.3 * box_height, 0.7 * box_height
    g = (255 * np.clip((np.nan_to_num(height, nan=0) - s0) / (s1 - s0), 0, 1)).astype(np.uint8)
    holes = (np.isnan(height) & roi).astype(np.uint8)
    g = cv2.inpaint(g, holes, 3, cv2.INPAINT_TELEA)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    edges = cv2.Canny(g, lo, hi)
    edges[~roi] = 0
    return edges


def ransac_lines(pts, axes, min_inliers, seed=0, trials=1500):
    """Sequential RANSAC: repeatedly take the best-supported line, then remove its inliers."""
    rng = np.random.default_rng(seed)
    lines = []
    pts = pts.astype(np.float32)
    cos_tol = np.cos(ANGLE_TOL)
    while len(pts) >= min_inliers and len(lines) < MAX_LINES:
        i, j = rng.integers(0, len(pts), (2, trials))
        d = pts[j] - pts[i]
        L = np.linalg.norm(d, axis=1)
        good = L > 5
        d = d[good] / L[good, None]
        p = pts[i[good]]
        along = np.abs(d @ axes.T)                      # |cos| with each box axis
        axis_ok = along.max(1) >= cos_tol
        if not axis_ok.any():
            break
        d, p = d[axis_ok], p[axis_ok]
        nrm = np.stack([-d[:, 1], d[:, 0]], 1)
        dist = np.abs((pts[None] - p[:, None]) @ nrm[..., None])[..., 0]   # trials x N
        counts = (dist < INLIER_PX).sum(1)
        b = int(np.argmax(counts))
        if counts[b] < min_inliers:
            break
        inl = dist[b] < INLIER_PX
        # least-squares refit (PCA) on the inliers
        q = pts[inl]
        c = q.mean(0)
        _, _, vt = np.linalg.svd(q - c)
        dd = vt[0]
        axis = int(np.argmax(np.abs(axes @ dd)))
        if np.abs(axes[axis] @ dd) >= cos_tol:
            lines.append(Line(c, dd, int(inl.sum()), axis))
        # remove inliers of the refit line (a bit wider so the same edge is not picked twice)
        nn = np.array([-dd[1], dd[0]])
        pts = pts[np.abs((pts - c) @ nn) >= INLIER_PX + 0.5]
    return lines


def group_lines(lines, axis, origin, axes, side):
    """Cluster lines of one axis by their offset across the box. Returns list of groups (sorted)."""
    across = axes[1 - axis]                          # horizontal lines are spaced along the y-axis
    ls = [l for l in lines if l.axis == axis]
    for l in ls:
        l.offset = float((l.p - origin) @ across)
    ls.sort(key=lambda l: l.offset)
    groups = []
    for l in ls:
        if groups and l.offset - groups[-1][-1].offset < GROUP_GAP * side:
            groups[-1].append(l)
        else:
            groups.append([l])
    return groups


def intersect(a, b):
    A = np.array([a.d, -b.d]).T
    t = np.linalg.solve(A, b.p - a.p)
    return a.p + t[0] * a.d


def _strip_ends(axis, corners):
    """The two box edges a strip of this axis runs between, pulled 15% inward off the side walls."""
    tl, tr, br, bl = corners
    e1, e2 = ((tl, bl), (tr, br)) if axis == 0 else ((tl, tr), (bl, br))
    lerp = lambda a, b, t: a + t * (b - a)
    edge = lambda a, b: Line(a, (b - a) / np.linalg.norm(b - a), 0, -1)
    return (edge(lerp(e1[0], e2[0], 0.15), lerp(e1[1], e2[1], 0.15)),
            edge(lerp(e1[0], e2[0], 0.85), lerp(e1[1], e2[1], 0.85)))


def strip_height(a, b, axis, height, corners):
    """Median height above the table of the strip between parallel lines a and b."""
    E1, E2 = _strip_ends(axis, corners)
    quad = np.array([intersect(a, E1), intersect(a, E2), intersect(b, E2), intersect(b, E1)], np.int32)
    m = np.zeros(height.shape, np.uint8)
    cv2.fillConvexPoly(m, quad, 1)
    vals = height[(m > 0) & ~np.isnan(height)]
    return float(np.median(vals)) if vals.size > 10 else float("nan")


def merge_raised_gaps(groups, axis, height, corners, box_height=BOX_HEIGHT):
    """Merge neighbouring groups whose in-between strip is raised (wall face / divider, not a floor)."""
    merged = [groups[0]]
    for g in groups[1:]:
        a = max(merged[-1], key=lambda l: l.offset)
        b = min(g, key=lambda l: l.offset)
        sh = strip_height(a, b, axis, height, corners)
        if np.isnan(sh) or sh > 0.3 * box_height:
            merged[-1] = merged[-1] + g
        else:
            merged.append(g)
    return merged


def add_missing_walls(groups, axis, height, corners, axes, side, box_height=BOX_HEIGHT):
    """If floor (not wall) lies between the outermost group and the box outline, the wall is out of
    view or undetected: use the outline edge itself as that wall."""
    tl, tr, br, bl = corners
    across = axes[1 - axis]
    first_edge, last_edge = ((tl, tr), (bl, br)) if axis == 0 else ((tl, bl), (tr, br))
    out = list(groups)
    for (a, b), at_start in ((first_edge, True), (last_edge, False)):
        e = Line(a, (b - a) / np.linalg.norm(b - a), 0, axis, float((a - tl) @ across))
        if not out:
            out = [[e]]
            continue
        inner = min(out[0], key=lambda l: l.offset) if at_start else max(out[-1], key=lambda l: l.offset)
        sh = strip_height(e, inner, axis, height, corners)
        roomy = abs(inner.offset - e.offset) > 0.08 * side   # room for a real cell, not a sliver
        if roomy and not np.isnan(sh) and sh < 0.3 * box_height:
            out = [[e]] + out if at_start else out + [[e]]
    return out


def detect(depth_m, roi=None):
    """roi: optional (x1, y1, x2, y2) in depth pixels where the box is (e.g. from an
    open-vocabulary detector). Raised things outside it (robot arm, other boxes) are
    ignored when finding the box outline."""
    d, height = height_image(depth_m)
    h_box = height
    if roi is not None:
        x1, y1, x2, y2 = (int(round(v)) for v in roi)
        h_box = np.full_like(height, np.nan)
        h_box[max(y1, 0):y2 + 1, max(x1, 0):x2 + 1] = height[max(y1, 0):y2 + 1, max(x1, 0):x2 + 1]
    corners = find_box_outline(h_box, 0.5 * BOX_HEIGHT)     # tl, tr, br, bl
    tl, tr, br, bl = corners
    ax = (tr - tl) / np.linalg.norm(tr - tl)
    ay = (bl - tl) / np.linalg.norm(bl - tl)
    axes = np.stack([ax, ay])
    side = min(np.linalg.norm(tr - tl), np.linalg.norm(bl - tl))

    roi = np.zeros(d.shape, np.uint8)
    cv2.fillConvexPoly(roi, corners.astype(np.int32), 1)
    roi = cv2.dilate(roi, np.ones((7, 7), np.uint8)).astype(bool)
    edges = canny_edges(height, roi)

    ys, xs = np.nonzero(edges)
    lines = ransac_lines(np.stack([xs, ys], 1), axes, int(MIN_SUPPORT * side))
    hg = group_lines(lines, 0, tl, axes, side)               # horizontal: walls top/bottom + dividers
    vg = group_lines(lines, 1, tl, axes, side)
    hg = merge_raised_gaps(hg, 0, height, corners) if hg else hg
    vg = merge_raised_gaps(vg, 1, height, corners) if vg else vg
    hg = add_missing_walls(hg, 0, height, corners, axes, side)
    vg = add_missing_walls(vg, 1, height, corners, axes, side)
    if len(hg) < 2 or len(vg) < 2:
        raise RuntimeError(f"walls not found (h groups={len(hg)}, v groups={len(vg)})")

    rows, cols = len(hg) - 1, len(vg) - 1
    cells = []
    for r in range(rows):
        top = max(hg[r], key=lambda l: l.offset)            # cell-facing edge of the group above
        bot = min(hg[r + 1], key=lambda l: l.offset)
        for c in range(cols):
            left = max(vg[c], key=lambda l: l.offset)
            right = min(vg[c + 1], key=lambda l: l.offset)
            quad = np.array([intersect(top, left), intersect(top, right),
                             intersect(bot, right), intersect(bot, left)], np.float32)
            m = np.zeros(d.shape, np.uint8)
            cv2.fillConvexPoly(m, quad.astype(np.int32), 1)
            core = cv2.erode(m, np.ones((5, 5), np.uint8)).astype(bool) & ~np.isnan(d)
            cells.append(Cell(r, c, quad, tuple(quad.mean(0)),
                              float(np.median(d[core])) if core.any() else float("nan")))
    return dict(rows=rows, cols=cols, corners=corners, edges=edges, lines=lines,
                h_groups=hg, v_groups=vg, cells=cells, depth=d)


def render(res):
    d = res["depth"]
    lo, hi = np.nanpercentile(d, 2), np.nanpercentile(d, 98)
    vis = cv2.applyColorMap((255 * np.clip((np.nan_to_num(d, nan=hi) - lo) / (hi - lo), 0, 1)).astype(np.uint8),
                            cv2.COLORMAP_JET)
    vis[np.isnan(d)] = 0
    base = (vis * 0.35).astype(np.uint8)

    # panel A: Canny edges
    a = base.copy()
    a[res["edges"] > 0] = (255, 255, 255)

    # panel B: RANSAC lines, one colour per wall/divider group
    b = base.copy()
    groups = res["h_groups"] + res["v_groups"]
    for gi, grp in enumerate(groups):
        col = tuple(int(v) for v in cv2.applyColorMap(np.uint8([[int(gi * 255 / len(groups))]]),
                                                       cv2.COLORMAP_HSV)[0, 0])
        for l in grp:
            p0, p1 = l.p - 400 * l.d, l.p + 400 * l.d
            cv2.line(b, tuple(p0.astype(int)), tuple(p1.astype(int)), col, 1, cv2.LINE_AA)

    # panel C: cells
    c = base.copy()
    n = len(res["cells"])
    for i, cell in enumerate(res["cells"]):
        col = cv2.applyColorMap(np.uint8([[int(i * 255 / n)]]), cv2.COLORMAP_HSV)[0, 0]
        m = np.zeros(d.shape, np.uint8)
        cv2.fillConvexPoly(m, cell.corners_px.astype(np.int32), 1)
        m = m.astype(bool)
        c[m] = (0.3 * c[m] + 0.7 * col).astype(np.uint8)
        cv2.polylines(c, [cell.corners_px.astype(np.int32)], True, (255, 255, 255), 1)
        cx, cy = map(int, cell.center_px)
        cv2.putText(c, str(i + 1), (cx - 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    x0, y0 = res["corners"].min(0).astype(int) - 20
    x1, y1 = res["corners"].max(0).astype(int) + 20
    panels = [cv2.resize(p[max(y0, 0):y1, max(x0, 0):x1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
              for p in (a, b, c)]
    for p, t in zip(panels, ["Canny", f"RANSAC lines ({len(res['lines'])})",
                             f"cells {res['rows']}x{res['cols']}"]):
        cv2.putText(p, t, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return np.hstack(panels)


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "depth.raw"
    raw = np.fromfile(src, np.uint16).reshape(-1, 480, 848)[-10:].astype(np.float32) / 1000.0
    raw[raw == 0] = np.nan
    with np.errstate(all="ignore"):
        depth = np.nan_to_num(np.nanmedian(raw, 0), nan=0)
    res = detect(depth)
    print(f"lines {len(res['lines'])}, h groups {len(res['h_groups'])}, v groups {len(res['v_groups'])}"
          f" -> grid {res['rows']}x{res['cols']}")
    for i, c in enumerate(res["cells"]):
        print(f"cell {i + 1} (r{c.row} c{c.col}) center=({c.center_px[0]:.0f},{c.center_px[1]:.0f}) "
              f"floor={c.floor_m:.3f} m")
    cv2.imwrite("lines_result.png", render(res))
