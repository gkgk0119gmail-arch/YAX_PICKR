"""Robust rotated-rectangle fit for irregular outlines (bubble-wrap blocks) and the grasp axis."""
import numpy as np


def ransac_rect(pts, iters=300, tol=2.0, rng=None):
    """Fit a rotated rectangle to contour points with RANSAC on orientation.

    Each trial takes the direction between two nearby contour points as one side's
    direction (mod 90 deg), puts the four sides at robust extents in that frame, and
    counts contour points within tol px of a side. Bumps on the outline (bubble wrap)
    only lose a few inliers, so they do not tilt or inflate the rectangle.
    Returns (center_xy, (w, h), angle_deg, inlier_ratio, corners 4x2).
    """
    rng = rng or np.random.default_rng(0)
    pts = pts.astype(np.float64)
    n = len(pts)

    def score(theta):
        c, s_ = np.cos(theta), np.sin(theta)
        R = np.array([[c, s_], [-s_, c]])
        q = pts @ R.T
        lo, hi = np.percentile(q, 5, axis=0), np.percentile(q, 95, axis=0)
        d = np.minimum(np.minimum(np.abs(q[:, 0] - lo[0]), np.abs(q[:, 0] - hi[0])),
                       np.minimum(np.abs(q[:, 1] - lo[1]), np.abs(q[:, 1] - hi[1])))
        inl = d < tol
        return inl.sum(), inl, q, R

    best = (-1, 0.0)
    for _ in range(iters):
        i = rng.integers(n)
        j = (i + rng.integers(3, 9)) % n                 # nearby point on the contour -> local edge direction
        v = pts[j] - pts[i]
        if np.hypot(*v) < 2:
            continue
        theta = np.arctan2(v[1], v[0]) % (np.pi / 2)
        k = score(theta)[0]
        if k > best[0]:
            best = (k, theta)
    k, inl, q, R = score(best[1])
    # refine each side from its own inliers (median offset)
    lo, hi = np.percentile(q, 5, axis=0), np.percentile(q, 95, axis=0)
    side = []
    for ax, v in ((0, lo[0]), (0, hi[0]), (1, lo[1]), (1, hi[1])):
        near = inl & (np.abs(q[:, ax] - v) < tol)
        side.append(np.median(q[near, ax]) if near.any() else v)
    x0, x1, y0, y1 = side
    corners_r = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    corners = corners_r @ R                              # back to image frame (R is orthonormal)
    center = corners.mean(0)
    return center, (x1 - x0, y1 - y0), np.degrees(best[1]), k / n, corners


def grasp_from_rect(center, size, angle_deg):
    """Parallel-gripper grasp: close across the short side.
    Returns (center_xy, jaw_axis_deg in [0,180), opening = short side, long side)."""
    w, h = size
    jaw = angle_deg if w <= h else angle_deg + 90
    return center, jaw % 180, min(w, h), max(w, h)
