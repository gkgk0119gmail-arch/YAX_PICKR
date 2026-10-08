"""Find a divided box in a top-down depth frame and split it into its compartments.

Nothing about the box is hard-coded: the table plane, the box outline and
rotation, and the number of dividers in each direction are all estimated from
the depth. Dividers are assumed to be straight and to run wall-to-wall (a grid).

Pipeline
  1. RANSAC table plane  -> height above table for every pixel
  2. largest raised blob not touching the image border -> box outline (minAreaRect)
  3. warp the box to an axis-aligned rectangle
  4. dividers = thin ridges; count them from row/column ridge profiles
  5. each divider gets a robust straight-line fit; cells = grid between the lines
  6. labels are warped back to the original image
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

MIN_BOX_HEIGHT = 0.03      # m above the table to count as box wall
MIN_RIDGE = 0.015          # m a divider must stick up above its surroundings
MIN_DIVIDER_COVER = 0.35   # fraction of the box span a divider must be seen along


@dataclass
class Cell:
    row: int
    col: int
    center_px: tuple        # (x, y) in the input depth image
    floor_m: float          # camera distance to the cell floor
    area_px: int


@dataclass
class BoxGrid:
    rows: int
    cols: int
    corners_px: np.ndarray  # 4x2, tl/tr/br/bl in the input image
    labels: np.ndarray      # HxW, -1 outside, else row*cols+col
    cells: list
    v_lines: list           # rectified-frame x(y) per vertical divider
    h_lines: list           # rectified-frame y(x) per horizontal divider


# --------------------------------------------------------------------------- geometry

def fit_table_plane(d, iters=300, tol=0.006, seed=0):
    """RANSAC plane z = a*x + b*y + c over the central part of the image (where the table is)."""
    rng = np.random.default_rng(seed)
    h, w = d.shape
    ys, xs = np.mgrid[0:h, 0:w]
    central = (np.abs(xs - w / 2) < 0.35 * w) & (np.abs(ys - h / 2) < 0.35 * h) & ~np.isnan(d)
    P = np.stack([xs[central], ys[central], d[central]], 1)
    if len(P) > 20000:
        P = P[rng.choice(len(P), 20000, replace=False)]
    A = np.c_[P[:, :2], np.ones(len(P))]
    best, best_n = None, -1
    for _ in range(iters):
        i = rng.choice(len(P), 3, replace=False)
        try:
            coef = np.linalg.solve(A[i], P[i, 2])
        except np.linalg.LinAlgError:
            continue
        n = (np.abs(A @ coef - P[:, 2]) < tol).sum()
        if n > best_n:
            best, best_n = coef, n
    inl = np.abs(A @ best - P[:, 2]) < tol
    coef, *_ = np.linalg.lstsq(A[inl], P[inl, 2], rcond=None)
    return coef[0] * xs + coef[1] * ys + coef[2]


def find_box_outline(height, min_height=MIN_BOX_HEIGHT):
    """Corners (tl, tr, br, bl) of the box: the raised blob whose hull contains the image centre
    (so a box cut off by the frame is still found); otherwise the largest blob off the border."""
    h, w = height.shape
    high = (np.nan_to_num(height, nan=0) > min_height).astype(np.uint8)
    high = cv2.morphologyEx(high, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(high)
    centre = (w / 2, h / 2)
    best, best_area, best_central = None, 0, False
    for i in range(1, n):
        x, y, bw, bh, _ = st[i]
        pts = cv2.findNonZero((lab == i).astype(np.uint8))
        hull = cv2.convexHull(pts)
        central = cv2.pointPolygonTest(hull, centre, False) >= 0
        on_border = x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2
        if not central and on_border:
            continue
        area = cv2.contourArea(hull)  # an open box is a ring, so use its hull area
        if (central, area) > (best_central, best_area):
            best, best_area, best_central = hull, area, central
    if best is None:
        raise RuntimeError("no box found above the table")
    pts = cv2.boxPoints(cv2.minAreaRect(best))
    s, diff = pts.sum(1), pts[:, 1] - pts[:, 0]
    return np.array([pts[s.argmin()], pts[diff.argmin()], pts[s.argmax()], pts[diff.argmax()]], np.float32)


def rectify(corners):
    """Affine warp mapping the box to [0,W]x[0,H] and its size."""
    tl, tr, br, bl = corners
    W = int(round(np.linalg.norm(tr - tl)))
    H = int(round(np.linalg.norm(bl - tl)))
    M = cv2.getAffineTransform(np.float32([tl, tr, bl]), np.float32([[0, 0], [W, 0], [0, H]]))
    return M, W, H


# --------------------------------------------------------------------------- dividers

def wall_extent(cover, frac=0.3):
    """Inner edges of the two walls on a ridge-coverage profile: walk inward from each wall's peak."""
    n = len(cover)
    q = max(n // 4, 1)
    lo = int(np.argmax(cover[:q]))
    while lo < n - 1 and cover[lo] >= frac:
        lo += 1
    hi = n - q + int(np.argmax(cover[n - q:]))
    while hi > 0 and cover[hi] >= frac:
        hi -= 1
    return lo, hi


def divider_positions(cover, lo, hi):
    """Ridge-coverage peaks strictly between the walls."""
    min_gap = max(int(0.08 * len(cover)), 5)
    peaks, _ = find_peaks(cover, height=MIN_DIVIDER_COVER, distance=min_gap, prominence=0.15)
    return [int(p) for p in peaks if lo + 2 < p < hi - 2]


def fit_divider(strength, x0, min_strength, win=6, tol=2.0):
    """Straight line x(y) along a vertical ridge starting near column x0 (robust to gaps/outliers)."""
    h, w = strength.shape
    xs = np.full(h, np.nan)
    for y in range(h):
        a, b = max(x0 - win, 0), min(x0 + win + 1, w)
        seg = strength[y, a:b]
        if seg.size and seg.max() >= min_strength:
            xs[y] = a + int(np.argmax(seg))
    t = np.arange(h)
    ok = ~np.isnan(xs)
    if ok.sum() < 2:
        return np.full(h, float(x0))
    for _ in range(4):
        p = np.polyfit(t[ok], xs[ok], 1)
        res = np.abs(np.polyval(p, t) - xs)
        new_ok = ~np.isnan(xs) & (res < max(tol, 2.5 * np.nanmedian(res[ok])))
        if new_ok.sum() < 2 or (new_ok == ok).all():
            break
        ok = new_ok
    return np.polyval(np.polyfit(t[ok], xs[ok], 1), t)


# --------------------------------------------------------------------------- main entry

def detect_box_grid(depth_m):
    """depth_m: HxW float32 metres, NaN/0 for invalid. Returns BoxGrid."""
    d = np.where(depth_m > 0, depth_m, np.nan).astype(np.float32)
    table = fit_table_plane(d)
    height = table - d                                  # m above table, NaN where invalid

    corners = find_box_outline(height)
    M, W, H = rectify(corners)
    hr = cv2.warpAffine(np.nan_to_num(height, nan=np.nanmax(height)).astype(np.float32), M, (W, H),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    k = max(9, int(0.12 * min(W, H))) | 1               # wider than a divider, narrower than a cell
    strength = cv2.morphologyEx(cv2.GaussianBlur(hr, (3, 3), 0), cv2.MORPH_TOPHAT,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    ridge = strength > MIN_RIDGE
    # walls first (full-span profiles), then dividers using only the box interior
    x_lo, x_hi = wall_extent(gaussian_filter1d(ridge.mean(0), 1.5))
    y_lo, y_hi = wall_extent(gaussian_filter1d(ridge.mean(1), 1.5))
    vx = divider_positions(gaussian_filter1d(ridge[y_lo:y_hi].mean(0), 1.5), x_lo, x_hi)
    hy = divider_positions(gaussian_filter1d(ridge[:, x_lo:x_hi].mean(1), 1.5), y_lo, y_hi)
    v_lines = [fit_divider(strength, x, MIN_RIDGE) for x in vx]
    h_lines = [fit_divider(strength.T, y, MIN_RIDGE) for y in hy]
    rows, cols = len(h_lines) + 1, len(v_lines) + 1

    yy, xx = np.mgrid[0:H, 0:W]
    col = sum((xx > v[yy]).astype(np.int32) for v in v_lines) if v_lines else np.zeros((H, W), np.int32)
    row = sum((yy > hl[xx]).astype(np.int32) for hl in h_lines) if h_lines else np.zeros((H, W), np.int32)
    lab_r = (row * cols + col).astype(np.float32) + 1
    interior = np.zeros((H, W), bool)
    interior[y_lo:y_hi + 1, x_lo:x_hi + 1] = True
    lab_r[~interior] = 0                                # walls belong to no cell

    Minv = cv2.invertAffineTransform(M)
    labels = cv2.warpAffine(lab_r, Minv, d.shape[::-1], flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.int32) - 1
    # mark walls/dividers (raised pixels) so cell stats use only the floor
    raised = cv2.warpAffine(ridge.astype(np.uint8), Minv, d.shape[::-1], flags=cv2.INTER_NEAREST) > 0

    cells = []
    for c in range(rows * cols):
        m = labels == c
        floor = m & ~raised & ~np.isnan(d)
        ys, xs = np.nonzero(m)
        cells.append(Cell(row=c // cols, col=c % cols,
                          center_px=(float(xs.mean()), float(ys.mean())),
                          floor_m=float(np.median(d[floor])) if floor.any() else float("nan"),
                          area_px=int(m.sum())))
    return BoxGrid(rows, cols, corners, labels, cells, v_lines, h_lines)


def render(depth_m, g, lo=None, hi=None):
    d = np.where(depth_m > 0, depth_m, np.nan)
    lo = np.nanpercentile(d, 2) if lo is None else lo
    hi = np.nanpercentile(d, 98) if hi is None else hi
    vis = cv2.applyColorMap((255 * np.clip((np.nan_to_num(d, nan=hi) - lo) / (hi - lo), 0, 1)).astype(np.uint8),
                            cv2.COLORMAP_JET)
    vis[np.isnan(d)] = 0
    out = (vis * 0.4).astype(np.uint8)
    n = g.rows * g.cols
    for i, c in enumerate(g.cells):
        m = g.labels == i
        colr = cv2.applyColorMap(np.uint8([[int(i * 255 / max(n, 1))]]), cv2.COLORMAP_HSV)[0, 0]
        out[m] = (0.35 * out[m] + 0.65 * colr).astype(np.uint8)
        cx, cy = map(int, c.center_px)
        cv2.putText(out, str(i + 1), (cx - 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.polylines(out, [g.corners_px.astype(np.int32)], True, (255, 255, 255), 1)
    cv2.putText(out, f"{g.rows}x{g.cols}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "depth.raw"
    raw = np.fromfile(src, np.uint16).reshape(-1, 480, 848)[-10:].astype(np.float32) / 1000.0
    raw[raw == 0] = np.nan
    with np.errstate(all="ignore"):
        depth = np.nanmedian(raw, 0)
    g = detect_box_grid(np.nan_to_num(depth, nan=0))
    print(f"grid {g.rows}x{g.cols}")
    for i, c in enumerate(g.cells):
        print(f"cell {i + 1} (r{c.row} c{c.col}) center=({c.center_px[0]:.0f},{c.center_px[1]:.0f}) "
              f"floor={c.floor_m:.3f} m area={c.area_px}")
    cv2.imwrite("grid_result.png", render(depth, g))
