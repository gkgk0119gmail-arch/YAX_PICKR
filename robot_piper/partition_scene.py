"""Partition mission perception (top D435i, RealSense depth): the open box and the stack of partitions.

Everything is done on a top-down height map in the robot base frame (max z per 4 mm cell):
  box        the largest low floor region on the box side (+y = robot's left by default), bounded by raised
             cells / unseen cells; a big opening cuts thin leaks around the wall ends. Inner rectangle =
             minAreaRect of that floor, rim z = high cells just outside it, floor z = its median height.
             The wall nearest the camera hides the floor next to it, so the rectangle errs on the small side.
  partition  the raised cells on the other side (box excluded), closed so the divider grid becomes solid,
             opened to drop thin divider tips. Footprint = minAreaRect; board z (its bottom board, the base the
             next layer stands on) = low percentile inside; top z = divider tops (high percentile).
             Dividers = peaks of the raised-cell histogram across each footprint axis.
A side that runs out of the camera view is flagged "cut"; the far side of the table is outside the D435i view,
so a known size can be given (sizes in m) and the rectangle is then grown away from the robot (+x).
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CALIB_TOP = ROOT / "pac2026" / "calib" / "top_cam_to_base.json"
RES = 0.004
X0, X1, Y0, Y1 = 0.0, 0.80, -0.60, 0.60


def base_points(depth_m, K):
    T = np.array(json.loads(CALIB_TOP.read_text())["T_base_cam"])
    H, W = depth_m.shape
    vv, uu = np.mgrid[0:H, 0:W]
    d = np.where(depth_m > 0.1, depth_m, np.nan)
    P = np.stack([(uu - K["cx"]) / K["fx"] * d, (vv - K["cy"]) / K["fy"] * d, d], -1)
    return P @ T[:3, :3].T + T[:3, 3]


def raster(P):
    """Top-down map: max z per cell (NaN = not seen). Rows = base x, columns = base y."""
    P = P.reshape(-1, 3)
    P = P[np.isfinite(P).all(1)]
    nx, ny = int(round((X1 - X0) / RES)), int(round((Y1 - Y0) / RES))
    i = ((P[:, 0] - X0) / RES).astype(int)
    j = ((P[:, 1] - Y0) / RES).astype(int)
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    Z = np.full((nx, ny), -np.inf, np.float32)
    np.maximum.at(Z, (i[ok], j[ok]), P[ok, 2])
    Z[~np.isfinite(Z)] = np.nan
    return Z


def cell_xy(rc):
    """(row, col) array -> base xy (cell centres)."""
    rc = np.asarray(rc, float)
    return np.stack([X0 + (rc[..., 0] + 0.5) * RES, Y0 + (rc[..., 1] + 0.5) * RES], -1)


def xy_cell(xy):
    xy = np.asarray(xy, float)
    return np.stack([(xy[..., 0] - X0) / RES - 0.5, (xy[..., 1] - Y0) / RES - 0.5], -1)


def rect_of(mask):
    """minAreaRect of a cell mask in base xy: dict(corners 4x2, centre, axis_long (unit), size (long, short))."""
    rs, cs = np.nonzero(mask)
    pts = cell_xy(np.stack([rs, cs], 1)).astype(np.float32)
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    C = cv2.boxPoints(((cx, cy), (w + RES, h + RES), ang))
    e1, e2 = C[1] - C[0], C[2] - C[1]
    long_e = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    a = long_e / np.linalg.norm(long_e)
    if a[0] < 0:                                       # long axis points away from the robot (+x) by convention
        a = -a
    return dict(corners=C.astype(float), centre=np.array([cx, cy]), axis=a,
                size=(float(max(np.linalg.norm(e1), np.linalg.norm(e2))), float(min(np.linalg.norm(e1), np.linalg.norm(e2)))))


def grid_rect(mask):
    """Rectangle of a divider grid: the angle at which the cells line up best (sharpest projection histograms
    along both axes), then the extent in that frame. minAreaRect turns a plus-shaped grid into a diamond."""
    rs, cs = np.nonzero(mask)
    P = cell_xy(np.stack([rs, cs], 1))
    c0 = P.mean(0)
    best = None
    for deg in np.arange(0, 90, 1.0):
        t = np.radians(deg)
        a, n = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
        sc = 0.0
        for d in (a, n):
            hst, _ = np.histogram((P - c0) @ d, bins=np.arange(-0.4, 0.4, RES))
            sc += float((hst.astype(float) ** 2).sum())
        if best is None or sc > best[0]:
            best = (sc, a, n)
    _, a, n = best
    u, v = (P - c0) @ a, (P - c0) @ n
    u0, u1, v0, v1 = np.percentile(u, 0.5), np.percentile(u, 99.5), np.percentile(v, 0.5), np.percentile(v, 99.5)
    centre = c0 + a * (u0 + u1) / 2 + n * (v0 + v1) / 2
    L1, L2 = u1 - u0 + RES, v1 - v0 + RES
    ax, size = (a, (L1, L2)) if L1 >= L2 else (n, (L2, L1))
    if ax[0] < 0:
        ax = -ax
    return dict(centre=centre, axis=ax, size=(float(size[0]), float(size[1])),
                corners=rect_corners(centre, ax, size))


def rect_corners(centre, axis, size):
    a = np.asarray(axis, float)
    n = np.array([-a[1], a[0]])
    L, W = size
    c = np.asarray(centre, float)
    return np.array([c - a * L / 2 - n * W / 2, c + a * L / 2 - n * W / 2, c + a * L / 2 + n * W / 2,
                     c - a * L / 2 + n * W / 2])


def rect_mask(shape, corners, grow=0.0):
    m = np.zeros(shape, np.uint8)
    c = np.asarray(corners, float)
    c = c + (c - c.mean(0)) / np.linalg.norm(c - c.mean(0), axis=1, keepdims=True) * grow * np.sqrt(2)
    rc = xy_cell(c)
    cv2.fillConvexPoly(m, np.round(rc[:, ::-1]).astype(np.int32), 1)    # (col, row) for OpenCV
    return m.astype(bool)


def touches_unseen(mask, seen, grow=3):
    """Does the region run into the edge of the camera view (unseen cells beyond it)?"""
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)).astype(bool) & ~mask
    edge = ~seen
    edge[:, :] &= True
    return float((ring & edge).mean() / max(ring.mean(), 1e-9)) > 0.25


def apply_size(rect, size, seen_mask):
    """Known (long, short) size: keep the rectangle's near side (small x) and grow it away from the robot."""
    if size is None:
        return rect
    a, (L, W) = rect["axis"], size
    near = rect["centre"] - a * rect["size"][0] / 2
    r = dict(rect, centre=near + a * L / 2, size=(float(L), float(W)), sized=True)
    r["corners"] = rect_corners(r["centre"], a, r["size"])
    return r


_GD = {}


def gdino_boxes(color, prompts="cardboard partition. cardboard dividers. open cardboard box.", thr=0.25):
    """Grounding DINO object boxes (pixels): [(score, [x1, y1, x2, y2], label)]. Labels mix partition / box up,
    so they are only used as object regions; which one is the box is decided by its walls (depth)."""
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    if not _GD:
        mid = "IDEA-Research/grounding-dino-base"
        _GD["proc"] = AutoProcessor.from_pretrained(mid)
        _GD["model"] = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to("cuda").eval()
    proc, model = _GD["proc"], _GD["model"]
    img = Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    inp = proc(images=img, text=prompts, return_tensors="pt").to("cuda")
    with torch.no_grad():
        r = proc.post_process_grounded_object_detection(model(**inp), inp.input_ids, threshold=thr, text_threshold=thr,
                                                        target_sizes=[img.size[::-1]])[0]
    W, H = img.size
    return [(float(sc), [float(v) for v in b], l) for b, sc, l in zip(r["boxes"].tolist(), r["scores"].tolist(),
                                                                       r["text_labels"])
            if (b[2] - b[0]) * (b[3] - b[1]) < 0.6 * W * H]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def partition_region(color, P, box, K):
    """Pixel box of the partition: the Grounding DINO object that is not the box (by overlap with the box seen
    from the camera). None if nothing else was found."""
    hits = gdino_boxes(color)
    C = box["corners"]
    q = project(np.vstack([np.c_[C, np.full(4, box["rim_z"])], np.c_[C, np.full(4, box["floor_z"])]]), K)
    bb = [q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()]
    others = [h for h in hits if iou(h[1], bb) < 0.4]
    if not others:
        return None
    pref = lambda h: (("divider" in h[2] or "partition" in h[2]), h[0])
    return max(others, key=pref)[1]


def board_mask(color, P, region, Zshape, exclude=None):
    """Cardboard (darker than the white table, Otsu on brightness inside the region) as raster cells."""
    x1, y1, x2, y2 = (int(round(v)) for v in region)
    H, W = color.shape[:2]
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
    V = cv2.cvtColor(color[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)[..., 2]
    t, _ = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = V < t
    pts = P[y1:y2, x1:x2][dark]
    pts = pts[np.isfinite(pts).all(1)]
    m = np.zeros(Zshape, np.uint8)
    i = ((pts[:, 0] - X0) / RES).astype(int)
    j = ((pts[:, 1] - Y0) / RES).astype(int)
    ok = (i >= 0) & (i < Zshape[0]) & (j >= 0) & (j < Zshape[1])
    m[i[ok], j[ok]] = 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))       # divider tips sticking out: away
    if exclude is not None:
        m[exclude] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, 4]))
    return lab == k if st[k, 4] * RES * RES > 0.03 else None


def table_z(Z):
    z = Z[np.isfinite(Z)]
    hist, edges = np.histogram(z, bins=np.arange(-0.3, 0.5, 0.004))
    return float(edges[np.argmax(hist)] + 0.002)


def region_cells(P, region, shape, grow=0.03):
    """Raster cells seen through a pixel box (x1, y1, x2, y2), grown by `grow` m."""
    x1, y1, x2, y2 = (int(round(v)) for v in region)
    H, W = P.shape[:2]
    pts = P[max(y1, 0):min(y2, H), max(x1, 0):min(x2, W)].reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(1)]
    m = np.zeros(shape, np.uint8)
    i = ((pts[:, 0] - X0) / RES).astype(int)
    j = ((pts[:, 1] - Y0) / RES).astype(int)
    ok = (i >= 0) & (i < shape[0]) & (j >= 0) & (j < shape[1])
    m[i[ok], j[ok]] = 1
    k = 2 * int(grow / RES) + 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.dilate(m, np.ones((k, k), np.uint8)).astype(bool)


def ring_box(Z, h, seen, only=None):
    """The box by its shape: raised cells forming a ring (walls) around an empty inside. A partition is raised
    too, but its dividers cross its inside. Returns (wall mask, rect) or None."""
    hi = cv2.dilate((h > 0.05).astype(np.uint8), np.ones((3, 3), np.uint8))
    hi[: int((0.06 - X0) / RES)] = 0                     # the robot itself
    if only is not None:                                # the box region from Qwen / Grounding DINO: things standing
        hi[~only] = 0                                   # against the box (partition, blocks) are cut off
    n, lab, st, _ = cv2.connectedComponentsWithStats(hi)
    best = None
    for i in range(1, n):
        if st[i, 4] < 80:
            continue
        m = lab == i
        rr = rect_of(m)
        area = rr["size"][0] * rr["size"][1]
        if not 0.04 < area < 0.5 or rr["size"][1] < 0.15:
            continue
        fill = st[i, 4] * RES * RES / area
        inside = rect_mask(Z.shape, rr["corners"], grow=-0.06)       # wall bands read 3-4 cm wide
        if fill > 0.35 or inside.sum() == 0 or seen[inside].mean() < 0.5:
            continue
        busy = float((h[inside & seen] > 0.04).mean())   # dividers crossing the inside -> a partition
        wall_h = float(np.percentile(h[m], 75))          # box walls ~11 cm; thin dividers read 3-7 cm
        if busy < 0.2 and wall_h > 0.07 and (best is None or area > best[2]):
            best = (m, rr, area)
    return None if best is None else best[:2]


def find_box(Z, side=1.0, size=None, nominal_rim=0.20, only=None):
    """Open box: a ring of walls around an empty inside (anywhere); if none is seen, the largest low floor
    region on the side `side` (+1: +y, robot's left). Returns dict or raises RuntimeError."""
    seen = np.isfinite(Z)
    zt = table_z(Z)
    h = np.nan_to_num(Z - zt, nan=0.0)
    rb = ring_box(Z, h, seen, only)
    if rb is not None:
        wall, r = rb
        for k in (5, 9, 15, 21):                        # close the gaps in the ring (robot, shadows) until it
            ring = cv2.dilate(wall.astype(np.uint8), np.ones((k, k), np.uint8))   # encloses a floor
            outside = ring.copy()
            ff = np.zeros((ring.shape[0] + 2, ring.shape[1] + 2), np.uint8)
            cv2.floodFill(outside, ff, (ring.shape[1] - 1, ring.shape[0] - 1), 2)    # from a far corner
            enclosed = (outside == 0)                   # not wall, not reachable from outside: the inside
            enclosed = cv2.dilate(enclosed.astype(np.uint8), np.ones((k // 2 * 2 + 1,) * 2, np.uint8)).astype(bool) \
                & ~cv2.dilate(wall.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)   # undo the closing
            if enclosed.sum() * RES * RES > 0.04:
                break
        inside = enclosed & seen & (h < 0.03)
        n3, lab3, st3, _ = cv2.connectedComponentsWithStats(inside.astype(np.uint8))
        if n3 >= 2:                                     # inner faces = the floor between the wall bands
            fl = lab3 == 1 + int(np.argmax(st3[1:, 4]))
            fr = rect_of(fl)
            r = dict(r, centre=fr["centre"], axis=fr["axis"], size=(fr["size"][0] + 0.01, fr["size"][1] + 0.01))
        else:
            r = dict(r, size=(r["size"][0] - 0.012, r["size"][1] - 0.012))
        r = dict(r, corners=rect_corners(r["centre"], r["axis"], r["size"]), from_walls=True)
        r["cut"] = touches_unseen(wall, seen)
        r = apply_size(r, size, seen)
        floor = rect_mask(Z.shape, r["corners"], grow=-0.02) & seen & (h < 0.04)
        walls = Z[wall & seen & (h > 0.04)]
        rim = float(np.percentile(walls, 75)) if walls.size > 30 else zt + nominal_rim
        return dict(r, floor_z=float(np.median(Z[floor])) if floor.any() else zt, rim_z=rim, table_z=zt,
                    rim_from_walls=walls.size > 30)
    bar = cv2.dilate(((h > 0.04) | ~seen).astype(np.uint8), np.ones((3, 3), np.uint8))
    low = (1 - bar).astype(np.uint8)
    low = cv2.morphologyEx(low, cv2.MORPH_OPEN, np.ones((31, 31), np.uint8))   # cut leaks around wall ends
    n, lab, st, cen = cv2.connectedComponentsWithStats(low)
    best, best_a = None, 0
    for i in range(1, n):
        y = Y0 + (cen[i][0] + 0.5) * RES
        if np.sign(y) == np.sign(side) and st[i, 4] > best_a:
            best, best_a = i, st[i, 4]
    if best is None or best_a * RES * RES < 0.03:
        raise RuntimeError("box: no floor region on that side")
    floor = lab == best
    # walls: high thin cells forming a ring (open on a side the camera cannot see well); the partition stack is
    # high too but solid (its bottom board), so the ring is the component with a low fill of its rectangle
    hi = cv2.dilate((h > 0.10).astype(np.uint8), np.ones((3, 3), np.uint8))
    hi[: int((0.06 - X0) / RES)] = 0
    near_floor = cv2.dilate(floor.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    n2, lab2, st2, cen2 = cv2.connectedComponentsWithStats(hi)
    wall, wall_fill = None, 1.0
    for i in range(1, n2):
        m = lab2 == i
        if st2[i, 4] < 60 or not (m & near_floor).any():
            continue
        rr = rect_of(m)
        fill = st2[i, 4] * RES * RES / max(rr["size"][0] * rr["size"][1], 1e-6)
        if rr["size"][0] * rr["size"][1] > 0.06 and fill < 0.35 and np.sign(rr["centre"][1]) == np.sign(side) \
                and fill < wall_fill:
            wall, wall_fill = m, fill
    if wall is not None:
        r = rect_of(wall)
        L, W = r["size"][0] - 0.012, r["size"][1] - 0.012          # wall centre line -> inner face
        r = dict(r, size=(L, W), corners=rect_corners(r["centre"], r["axis"], (L, W)), from_walls=True)
        r["cut"] = touches_unseen(wall, seen)
        floor = rect_mask(Z.shape, r["corners"], grow=-0.02) & seen & (h < 0.04)
    else:
        r = rect_of(floor)
        r["cut"] = touches_unseen(floor, seen)
    r = apply_size(r, size, seen)
    if wall is not None:                                # the wall cells themselves (thin tops read a bit low)
        walls = Z[wall & seen & (h > 0.08)]
    else:
        ring = rect_mask(Z.shape, r["corners"], grow=0.03) & ~rect_mask(Z.shape, r["corners"], grow=0.0)
        walls = Z[ring & seen & (h > 0.08)]
    rim = float(np.percentile(walls, 75)) if walls.size > 30 else zt + nominal_rim
    return dict(r, floor_z=float(np.median(Z[floor])) if floor.any() else zt, rim_z=rim, table_z=zt,
                rim_from_walls=walls.size > 30)


def find_partition(Z, box=None, size=None, min_raise=0.025, nominal_h=0.10, board=None):
    """The top partition of the stack (the raised object that is not the box). board: cardboard cells of the
    partition from the colour image (Grounding DINO region) - its outline wins when given."""
    seen = np.isfinite(Z)
    zt = table_z(Z) if box is None else box["table_z"]
    h = np.nan_to_num(Z - zt, nan=0.0)
    tall = (h > min_raise).astype(np.uint8)
    tall[: int((0.10 - X0) / RES)] = 0                  # the robot itself (gripper over x < 10 cm at rest)
    if box is not None:
        tall[rect_mask(Z.shape, box["corners"], grow=0.02)] = 0       # inner face + wall (the partition may
                                                                       # stand 1 cm from the box)
    foot = None
    for close in (0, 31):                               # raised bottom board: solid as it is; a partition
        solid = tall                                    # standing on the table: close the divider grid
        if close:
            solid = cv2.morphologyEx(tall, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
        solid = cv2.morphologyEx(solid, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))   # thin tips / walls away
        n, lab, st, _ = cv2.connectedComponentsWithStats(solid)
        if n >= 2 and st[1:, 4].max() * RES * RES > 0.03:
            foot = lab == 1 + int(np.argmax(st[1:, 4]))
            break
    grown = 0.0
    if board is not None:                               # the bottom board seen in colour: exact outline
        foot = board
    if foot is None:                                    # board flat on the table and wide cells: the dividers alone,
        g = cv2.dilate(tall, np.ones((7, 7), np.uint8))  # joined where they cross; outline = their extent
        n, lab, st, _ = cv2.connectedComponentsWithStats(g)
        if n >= 2:
            i = 1 + int(np.argmax(st[1:, 4]))
            rr = rect_of(lab == i)
            if rr["size"][0] * rr["size"][1] > 0.03 and rr["size"][1] > 0.10:
                foot, grown = lab == i, 0.012
    if foot is None:
        raise RuntimeError("partition: nothing raised outside the box")
    r = rect_of(foot)
    if grown:
        r = grid_rect(foot & (tall > 0))                # the divider cells themselves, grid-aligned
    r["from_board"] = board is not None
    r["cut"] = touches_unseen(foot, seen)
    r = apply_size(r, size, seen)
    for _ in range(2):                                  # 2nd pass: centred between the dividers
        inner = rect_mask(Z.shape, r["corners"], grow=-0.02) & seen
        zin = Z[inner]
        board = float(np.percentile(zin, 30))           # the bottom board between the dividers
        top = float(np.percentile(zin, 99))             # divider tops (RealSense reads thin tops a bit low)
        ridge = inner & (Z > board + 0.35 * (top - board))
        divs = dividers(ridge, r)
        for along in (() if r["from_board"] else ("long", "short")):   # dividers sit symmetric: their mean is
            ds = [d for d in divs if d["along"] == along]   # its centre across them (divider tips sticking out
            if ds and not (along == "short" and r["cut"] and not r.get("sized")):   # of the board fool the outline)
                r["centre"] = r["centre"] + ds[0]["normal"] * float(np.mean([d["offset"] for d in ds]))
        r["corners"] = rect_corners(r["centre"], r["axis"], r["size"])
    top_meas = top
    if nominal_h and top - board < nominal_h - 0.004:   # thin divider tops always read low (VGGT, RealSense):
        top = board + nominal_h - 0.004                  # the known height (higher of the two)
    return dict(r, board_z=board, top_z=top, top_measured=top_meas, top_nominal=top != top_meas,
                bottom_z=board - 0.004, height=top - board + 0.004, dividers=divs)


def dividers(ridge, r, min_cover=0.25, merge=0.03, edge=0.015):
    """Divider plates from the ridge cells: per footprint axis, peaks of the histogram across that axis.
    Returns [dict(p0, p1 (base xy ends), dir (unit), normal, offset (m from the centre, along normal), along)]."""
    rs, cs = np.nonzero(ridge)
    if len(rs) == 0:
        return []
    P = cell_xy(np.stack([rs, cs], 1))
    out = []
    a = r["axis"]
    for along, d in (("long", a), ("short", np.array([-a[1], a[0]]))):
        nrm = np.array([-d[1], d[0]])
        L = r["size"][0] if along == "long" else r["size"][1]          # plate length along d
        W = r["size"][1] if along == "long" else r["size"][0]          # range of offsets
        off = (P - r["centre"]) @ nrm
        bins = np.arange(-W / 2, W / 2 + RES, RES)
        hist, edges = np.histogram(off, bins=bins)
        hist = np.convolve(hist, [1, 2, 3, 2, 1], "same") / 9.0
        need = min_cover * L / RES
        peaks = [k for k in range(1, len(hist) - 1) if hist[k] >= need and hist[k] >= hist[k - 1] and hist[k] >= hist[k + 1]]
        offs = []
        for k in sorted(peaks, key=lambda k: -hist[k]):
            o = (edges[k] + edges[k + 1]) / 2
            if abs(abs(o) - W / 2) < edge:
                continue                                 # the outline itself, not an inner divider
            if all(abs(o - p) > merge for p in offs):
                offs.append(o)
        for o in sorted(offs):
            sel = np.abs(off - o) < 0.008
            s = (P[sel] - r["centre"]) @ d
            s0, s1 = (np.percentile(s, 1), np.percentile(s, 99)) if sel.sum() > 10 else (-L / 2, L / 2)
            c = r["centre"] + nrm * o
            out.append(dict(along=along, dir=d, normal=nrm, offset=float(o), p0=c + d * s0, p1=c + d * s1,
                            length=float(s1 - s0)))
    return out


def dividers_ransac(Z, r, board_z, top_z):
    """Divider plates from Canny + sequential RANSAC on the height map around the footprint: a plate shows as
    two parallel edges (its two faces) -> segments grouped by their offset across the footprint axis; the plate is
    the middle of the group. Same format as dividers()."""
    import vggt_lines as vl
    region = rect_mask(Z.shape, r["corners"], grow=0.03)
    h = np.clip(np.nan_to_num(Z - board_z, nan=0.0), 0, 0.15)
    g = cv2.GaussianBlur((h / 0.15 * 255).astype(np.uint8), (5, 5), 0)
    edges = cv2.Canny(g, 25, 70)
    edges[~region] = 0
    rs, cs = np.nonzero(edges)
    segs = vl.ransac_segments(np.stack([rs, cs], 1), min_len=0.06)
    a = r["axis"]
    out = []
    for along, d in (("long", a), ("short", np.array([-a[1], a[0]]))):
        nrm = np.array([-d[1], d[0]])
        L = r["size"][0] if along == "long" else r["size"][1]
        W = r["size"][1] if along == "long" else r["size"][0]
        items = []
        for sg in segs:
            P = cell_xy(np.array([sg["p0"], sg["p1"]]))
            v = P[1] - P[0]
            if abs(v @ d) / np.linalg.norm(v) < np.cos(np.radians(12)):
                continue                                 # not along this axis
            off = float(((P.mean(0)) - r["centre"]) @ nrm)
            if abs(off) > W / 2 - 0.01:
                continue                                 # the outline / outside
            s0, s1 = sorted(float((q - r["centre"]) @ d) for q in P)
            items.append((off, s0, s1, P))
        items.sort()
        groups = []
        for it in items:                                 # the two faces of one plate lie 0.5-2.5 cm apart
            if groups and it[0] - groups[-1][-1][0] < 0.025:
                groups[-1].append(it)
            else:
                groups.append([it])
        for gr in groups:
            s0, s1 = min(g[1] for g in gr), max(g[2] for g in gr)
            if s1 - s0 < 0.3 * L:
                continue
            E = np.vstack([g[3] for g in gr])           # the plate's own line: both faces' segment ends, PCA
            c = E.mean(0)
            _, _, vt = np.linalg.svd(E - c, full_matrices=False)
            dd = vt[0] if vt[0] @ d >= 0 else -vt[0]
            nn = np.array([-dd[1], dd[0]])
            t = (E - c) @ dd
            o = float((c - r["centre"]) @ nrm)
            out.append(dict(along=along, dir=dd, normal=nn, offset=o, p0=c + dd * t.min(), p1=c + dd * t.max(),
                            length=float(t.max() - t.min()), faces=len(gr), point=c))
    return out


def vggt_points(color, K):
    """VGGT base-frame points at full image resolution (metric from the table plane; no depth camera)."""
    import tempfile
    import vggt_lines as vl
    with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
        cv2.imwrite(f.name, color)
        Pv, s, _ = vl.vggt_base(f.name, K)
    H, W = color.shape[:2]
    return cv2.resize(Pv.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR), s


def scan(color, depth_m, K, box=None, box_size=None, part_size=None, side=1.0, part_h=0.10, use_gdino=True,
         use_vggt=True, box_px=None):
    if use_vggt:
        P, vscale = vggt_points(color, K)
    else:
        P, vscale = base_points(depth_m, K), None
    Z = raster(P)
    if box is None:
        if box_px is None and use_gdino:                # no Qwen box: Grounding DINO's "box" hit, if any
            hits = [h for h in gdino_boxes(color) if "box" in h[2] and "partition" not in h[2]]
            box_px = max(hits)[1] if hits else None
        only = region_cells(P, box_px, Z.shape) if box_px is not None else None
        box = find_box(Z, side, box_size, only=only)
    board, region = None, None
    if use_gdino:
        try:
            region = partition_region(color, P, box, K)
            if region is not None:
                board = board_mask(color, P, region, Z.shape, exclude=rect_mask(Z.shape, box["corners"], grow=0.02))
        except Exception as e:                          # no GPU / model: depth only
            print(f"[scan] Grounding DINO skipped: {e}")
    part = find_partition(Z, box, part_size, nominal_h=part_h, board=board)
    part["region_px"] = region
    if use_vggt:                                        # dividers: Canny + RANSAC on the VGGT height map
        dv = dividers_ransac(Z, part, part["board_z"], part["top_z"])
        if any(d["along"] == "long" for d in dv):
            part["dividers"] = dv
    part["vggt_scale"] = vscale
    inside = rect_mask(Z.shape, box["corners"], grow=-0.03) & np.isfinite(Z)
    box = dict(box, content_top_z=float(np.percentile(Z[inside], 95)) if inside.sum() > 50 else box["floor_z"])
    return dict(Z=Z, box=box, part=part, K=K)


def project(P, K):
    T = np.linalg.inv(np.array(json.loads(CALIB_TOP.read_text())["T_base_cam"]))
    P = np.asarray(P, float)
    pc = P @ T[:3, :3].T + T[:3, 3]
    return np.stack([pc[:, 0] / pc[:, 2] * K["fx"] + K["cx"], pc[:, 1] / pc[:, 2] * K["fy"] + K["cy"]], 1)


def draw(color, sc, grasp=None):
    img = color.copy()
    K, box, part = sc["K"], sc["box"], sc["part"]
    poly = lambda C, z: project(np.c_[C, np.full(len(C), z)], K).astype(np.int32)
    cv2.polylines(img, [poly(box["corners"], box["floor_z"])], True, (255, 160, 60), 2, cv2.LINE_AA)
    cv2.polylines(img, [poly(box["corners"], box["rim_z"])], True, (255, 220, 120), 2, cv2.LINE_AA)
    cv2.polylines(img, [poly(part["corners"], part["board_z"])], True, (60, 200, 255), 2, cv2.LINE_AA)
    if part.get("region_px") is not None:
        x1, y1, x2, y2 = (int(v) for v in part["region_px"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)
    for d in part["dividers"]:
        q = poly(np.array([d["p0"], d["p1"]]), part["top_z"])
        cv2.line(img, tuple(q[0]), tuple(q[1]), (0, 255, 0) if d["along"] == "long" else (0, 180, 0), 3, cv2.LINE_AA)
    if grasp is not None:
        g = project(np.array([[*grasp["xy"], grasp["tip_z"]]]), K)[0].astype(int)
        cv2.circle(img, tuple(g), 12, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.polylines(img, [poly(grasp["part_place_corners"], box["floor_z"])], True, (0, 0, 255), 2, cv2.LINE_AA)
        gp = project(np.array([[*grasp["xy_place"], grasp["tip_release"]]]), K)[0].astype(int)
        cv2.circle(img, tuple(gp), 12, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.arrowedLine(img, tuple(g), tuple(gp), (0, 0, 255), 2, cv2.LINE_AA)
    t = [f"box inner {box['size'][0] * 1000:.0f} x {box['size'][1] * 1000:.0f} mm{' (CUT)' if box['cut'] else ''}"
         f"  rim z {box['rim_z'] * 1000:.0f}  floor z {box['floor_z'] * 1000:.0f}  inside top {box['content_top_z'] * 1000:.0f}",
         f"partition {part['size'][0] * 1000:.0f} x {part['size'][1] * 1000:.0f} mm{' (CUT)' if part['cut'] else ''}"
         f"  board z {part['board_z'] * 1000:.0f}  top z {part['top_z'] * 1000:.0f}  dividers "
         f"{sum(d['along'] == 'long' for d in part['dividers'])} long / {sum(d['along'] == 'short' for d in part['dividers'])} short"]
    cv2.rectangle(img, (0, 0), (img.shape[1], 12 + 26 * len(t)), (0, 0, 0), -1)
    for i, s in enumerate(t):
        cv2.putText(img, s, (12, 28 + 26 * i), 0, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img
