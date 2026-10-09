"""VGGT only (no depth camera): colour image -> VGGT depth -> metric with the top-camera calibration (the table
is the plane z = TABLE_Z in the robot base frame, so each pixel ray's distance to it fixes the scale) ->
top-down height map -> Canny -> sequential RANSAC line segments.

  perception-main/ai_bubble/.venv/bin/python robot_piper/vggt_lines.py IMAGE.jpg [--out DIR]
Pictures: 1_height.png 2_canny.png 3_lines.png 4_lines_on_image.png (+ vggt_base.ply)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "robot_piper"), str(ROOT / "perception-main"),
                str(ROOT / "perception-main" / "research" / "box_cells")]
import partition_scene as ps  # noqa: E402

TABLE_Z = -0.055               # table plane in the base frame (measured 2026-10-10 after the camera move)
H_MAX = 0.15                   # height-map colour / Canny scale (m above the table)


def calib():
    return np.array(json.loads(ps.CALIB_TOP.read_text())["T_base_cam"])


def vggt_base(img_path, K):
    """VGGT depth on its grid -> metric (table plane) -> base-frame points (vh, vw, 3) and the scale."""
    import gdino_vggt_lines as gvl
    color = cv2.imread(str(img_path))
    H, W = color.shape[:2]
    d_v, _, _ = gvl.vggt(img_path)
    vh, vw = d_v.shape
    T = calib()
    u = (np.arange(vw) + 0.5) * W / vw - 0.5
    v = (np.arange(vh) + 0.5) * H / vh - 0.5
    uu, vv = np.meshgrid(u, v)
    ray = np.stack([(uu - K["cx"]) / K["fx"], (vv - K["cy"]) / K["fy"], np.ones_like(uu)], -1)   # z = 1 in camera
    rb = ray @ T[:3, :3].T                              # ray directions in the base frame
    d_table = (TABLE_Z - T[2, 3]) / rb[..., 2]          # camera depth at which each ray meets the table
    ratio = d_table / d_v
    s = float(np.median(ratio))
    for _ in range(3):                                  # the table is most of the picture: refine on it
        s = float(np.median(ratio[np.abs(ratio / s - 1) < 0.03]))
    d_m = d_v * s
    Pc = ray * d_m[..., None]
    Pb = Pc @ T[:3, :3].T + T[:3, 3]
    return Pb, s, color


def ransac_segments(pts, min_len=0.08, tol=1.5, gap=0.06, max_lines=40, trials=3000, seed=0):
    """Sequential RANSAC on edge cells (row, col): fit a line, split its inliers where they leave a gap > `gap`
    (bridging a crossing divider), keep every piece >= min_len as a segment, remove the inliers, repeat.
    Returns [dict(p0, p1 (cells), n, dir)]."""
    rng = np.random.default_rng(seed)
    pts = pts.astype(np.float32)
    out = []
    min_n = int(min_len / ps.RES * 0.6)
    while len(pts) >= min_n and len(out) < max_lines:
        i, j = rng.integers(0, len(pts), (2, trials))
        d = pts[j] - pts[i]
        L = np.linalg.norm(d, axis=1)
        ok = L > 8
        if not ok.any():
            break
        d, p = d[ok] / L[ok, None], pts[i[ok]]
        nrm = np.stack([-d[:, 1], d[:, 0]], 1)
        dist = np.abs((pts[None] - p[:, None]) @ nrm[..., None])[..., 0]
        cnt = (dist < tol).sum(1)
        b = int(np.argmax(cnt))
        if cnt[b] < min_n:
            break
        inl = dist[b] < tol
        q = pts[inl]
        c = q.mean(0)
        _, _, vt = np.linalg.svd(q - c, full_matrices=False)
        dd = vt[0]
        t = np.sort((q - c) @ dd)
        breaks = np.nonzero(np.diff(t) > gap / ps.RES)[0]   # a crossing plate leaves a few cm without edges on
        for run in np.split(t, breaks + 1):                 # this line: bridged by `gap`; farther pieces of the
            if (run[-1] - run[0]) * ps.RES >= min_len:      # same line are kept as their own segments (all
                out.append(dict(p0=c + dd * run[0], p1=c + dd * run[-1], n=int(len(run)), dir=dd))   # inliers go)
        nn = np.array([-dd[1], dd[0]])
        pts = pts[np.abs((pts - c) @ nn) >= tol + 0.5]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--intrinsics", type=Path, help="default: intrinsics.json next to the image")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    K = json.loads((args.intrinsics or args.image.parent / "intrinsics.json").read_text())
    out = args.out or args.image.parent / "vggt_lines"
    out.mkdir(parents=True, exist_ok=True)
    Pb, s, color = vggt_base(args.image, K)
    print(f"VGGT scale {s:.4f} (table plane z {TABLE_Z * 1000:.0f} mm)")
    vh, vw = Pb.shape[:2]
    from label_points import write_ply
    col = cv2.resize(color, (vw, vh))[..., ::-1].reshape(-1, 3)
    write_ply(out / "vggt_base.ply", Pb.reshape(-1, 3), col, np.zeros(vh * vw, np.uint8), np.full(vh * vw, -1, np.int16))
    Z = ps.raster(Pb)
    seen = np.isfinite(Z)
    h = np.clip(np.nan_to_num(Z - TABLE_Z, nan=0.0), 0, H_MAX)
    g = (h / H_MAX * 255).astype(np.uint8)
    g = cv2.inpaint(g, (~seen).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    edges = cv2.Canny(g, 25, 70)
    edges[: int((0.06 - ps.X0) / ps.RES)] = 0           # the robot itself
    rs, cs = np.nonzero(edges)
    segs = ransac_segments(np.stack([rs, cs], 1))
    print(f"{len(rs)} edge cells -> {len(segs)} line segments")
    vis_h = cv2.applyColorMap(g, cv2.COLORMAP_JET)
    vis_h[~seen] = 0
    big = lambda im: cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out / "1_height.png"), big(vis_h))
    cv2.imwrite(str(out / "2_canny.png"), big(edges))
    vis_l = (cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) * 0.6).astype(np.uint8)
    vis_l = big(vis_l)
    img = color.copy()
    rng = np.random.default_rng(1)
    for k, sgm in enumerate(segs):
        colr = tuple(int(v) for v in rng.integers(80, 256, 3))
        a, b = sgm["p0"], sgm["p1"]
        cv2.line(vis_l, (int(a[1] * 2), int(a[0] * 2)), (int(b[1] * 2), int(b[0] * 2)), colr, 2, cv2.LINE_AA)
        cv2.putText(vis_l, str(k), (int((a[1] + b[1])), int((a[0] + b[0]))), 0, 0.45, colr, 1, cv2.LINE_AA)
        # on the colour image at the height of the line's cells
        xy = ps.cell_xy(np.array([a, b]))
        rr = np.clip(np.round(np.linspace(a, b, 20)).astype(int), 0, np.array(Z.shape) - 1)
        zl = float(np.nanmedian(Z[rr[:, 0], rr[:, 1]]))
        q = ps.project(np.c_[xy, np.full(2, zl)], K).astype(int)
        cv2.line(img, tuple(q[0]), tuple(q[1]), colr, 2, cv2.LINE_AA)
        L = np.linalg.norm(xy[1] - xy[0])
        ang = np.degrees(np.arctan2(*(xy[1] - xy[0])[::-1]))
        print(f"  {k:2d}: ({xy[0][0] * 1000:.0f}, {xy[0][1] * 1000:.0f}) -> ({xy[1][0] * 1000:.0f}, {xy[1][1] * 1000:.0f}) mm, "
              f"{L * 1000:.0f} mm, angle {ang:.1f} deg (0 = +x), z {zl * 1000:.0f} mm, {sgm['n']} cells")
    cv2.imwrite(str(out / "3_lines.png"), vis_l)
    cv2.imwrite(str(out / "4_lines_on_image.png"), img)
    print(f"pictures -> {out}")


if __name__ == "__main__":
    main()
