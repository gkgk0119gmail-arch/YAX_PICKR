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


def refine_rect(pts, corners, band=6.0, corner_trim=0.15):
    """Least-squares refinement of a RANSAC rectangle with the four sides kept at right angles.

    Contour points within `band` of a side (and not within `corner_trim` of a side's ends, where bubble
    wrap rounds the corners) are assigned to that side. Rotating the points of the 2nd/4th side by 90 deg
    gives all four groups one common normal n: minimise sum_k sum_i (n.(p_i - mean_k))^2 -> n is the
    smallest-eigenvalue eigenvector of the summed scatter matrices. Each side's offset is then its
    points' mean along its normal. Returns (center, (w, h), angle_deg, n_used, corners 4x2).
    """
    pts = np.asarray(pts, np.float64)
    c = np.asarray(corners, np.float64)
    groups = []
    for k in range(4):
        a, b = c[k], c[(k + 1) % 4]
        d = b - a
        L = np.linalg.norm(d)
        t = (pts - a) @ d / L ** 2                              # position along the side, 0..1
        nrm = np.array([-d[1], d[0]]) / L
        dist = np.abs((pts - a) @ nrm)
        groups.append(pts[(dist < band) & (t > corner_trim) & (t < 1 - corner_trim)])
    if min(len(g) for g in groups) < 5:
        return None
    rot = np.array([[0.0, -1.0], [1.0, 0.0]])                  # +90 deg
    S = np.zeros((2, 2))
    for k, g in enumerate(groups):
        q = g @ rot.T if k % 2 else g                            # sides 1 and 3 turned onto sides 0 and 2
        q = q - q.mean(0)
        S += q.T @ q
    w, v = np.linalg.eigh(S)
    n = v[:, 0]                                                  # normal of sides 0 and 2
    m = rot @ n                                                  # normal of sides 1 and 3
    off = [g @ (n if k % 2 == 0 else m) for k, g in enumerate(groups)]
    d0, d2 = np.mean(off[0]), np.mean(off[2])
    d1, d3 = np.mean(off[1]), np.mean(off[3])
    A = np.array([n, m])

    def corner(dn, dm):
        return np.linalg.solve(A, [dn, dm])
    # side k lies on (normal . p = d_k); corners: side 3 & 0, 0 & 1, 1 & 2, 2 & 3
    out = np.array([corner(d0, d3), corner(d0, d1), corner(d2, d1), corner(d2, d3)])
    center = out.mean(0)
    size = (abs(d1 - d3), abs(d2 - d0))
    angle = np.degrees(np.arctan2(m[1], m[0])) % 90             # direction along sides 0 / 2
    return center, size, angle, int(sum(len(g) for g in groups)), out
