#!/usr/bin/env python3
"""
Per-point semantic labels for the VGGT point cloud of a partition-box scene.

Labels:  0 other (sill, floor, robot...)  1 table  2 box wall/rim  3 divider  4 cell floor
Each cell-floor point also gets its cell id (1..N, row-major, row 1 = top of the image).

Pipeline: Grounding DINO box ROI -> VGGT points -> table plane on a thin ring around the box
(box_height.py) -> metric scale from RealSense -> partition grid (box_lines.detect on VGGT depth)
-> label by height + position (inside outline / near the outline / inside a cell quad).

Outputs in --out: points_labeled.ply (label colours + label/cell properties),
points_labeled.npz (xyz_m, rgb, label, cell, height_m), labels.png (2-D preview).

Usage (ai_bubble/.venv):
    HF_HUB_OFFLINE=1 ../ai_bubble/.venv/bin/python label_points.py --image img.jpg --depth depth.raw --out dir
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import box_height as bh
from capture_cells import find_box_roi, to_vggt_px

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "research" / "box_cells"))
import box_grid  # noqa: E402
import box_lines as bl  # noqa: E402

NAMES = ["other", "table", "box_wall", "divider", "cell_floor"]
COLORS = np.array([[128, 128, 128], [70, 130, 255], [255, 140, 0], [255, 220, 0], [40, 200, 80]], np.uint8)  # RGB
TABLE_TOL = 0.015       # m around the table plane
RAISED = 0.5            # fraction of the wall height above which a point is wall/divider
WALL_BAND = 0.06        # fraction of the box size from the outline that counts as outer wall


OPENING_H = 0.05     # m above the table where cell openings are measured (centre of the 3-7 cm Canny slice)


def px_to_plane(px, K, n, d, height_v):
    """Intersect camera rays through pixels with the plane n.x + d = height_v (VGGT units)."""
    uv1 = np.c_[np.asarray(px, np.float64), np.ones(len(px))]
    rays = uv1 @ np.linalg.inv(K).T
    t = (height_v - d) / (rays @ n)
    return rays * t[:, None]


def quad_size(P):
    """Side lengths of a 3-D quad (tl, tr, br, bl): (width, length) as the mean of opposite sides."""
    e = lambda a, b: np.linalg.norm(P[a] - P[b])
    return (e(0, 1) + e(3, 2)) / 2, (e(0, 3) + e(1, 2)) / 2


def write_ply(path, xyz, rgb, label, cell):
    keep = np.isfinite(xyz).all(1)
    xyz, rgb, label, cell = xyz[keep], rgb[keep], label[keep], cell[keep]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "property uchar label\nproperty short cell\nend_header\n")
    rec = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"),
                                    ("b", "u1"), ("label", "u1"), ("cell", "<i2")])
    rec["x"], rec["y"], rec["z"] = xyz.T
    rec["r"], rec["g"], rec["b"] = rgb.T
    rec["label"], rec["cell"] = label, cell
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(rec.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--depth", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(args.image))
    roi = find_box_roi(img)
    pts, K = bh.vggt_points(args.image)
    vh, vw = pts.shape[:2]
    roi_v = to_vggt_px(roi, (vh, vw), img.shape[:2])

    # table plane + metric scale (same as box_height.py)
    ring, _ = bh.roi_masks((vh, vw), roi_v)
    n, d, _ = bh.ransac_plane(pts[ring].reshape(-1, 3))
    rs = bh.load_depth_m(args.depth)
    z_rs = float(box_grid.fit_table_plane(rs)[bh.DH // 2, bh.DW // 2])
    scale = z_rs / abs(d / n[2])
    xyz = pts * scale
    h = (pts @ n + d) * scale                                         # m above the table

    # partition grid on the VGGT depth (depth = z of the camera-frame points)
    res = bl.detect(xyz[..., 2], roi=roi_v)
    outline = res["corners"].astype(np.float32)
    inside = np.zeros((vh, vw), np.uint8)
    cv2.fillConvexPoly(inside, outline.astype(np.int32), 1)
    inside = inside.astype(bool)
    dist = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 5)  # px to the outline
    box_size = np.sqrt(cv2.contourArea(outline))
    wall_h = float(np.nanpercentile(h[inside], 99))

    label = np.zeros((vh, vw), np.uint8)
    label[~inside & (np.abs(h) < TABLE_TOL)] = 1
    raised = inside & (h > RAISED * wall_h)
    label[raised & (dist < WALL_BAND * box_size)] = 2
    label[raised & (dist >= WALL_BAND * box_size)] = 3
    low = inside & ~raised
    cell = np.full((vh, vw), -1, np.int16)
    for k, c in enumerate(res["cells"]):
        m = np.zeros((vh, vw), np.uint8)
        cv2.fillConvexPoly(m, c.corners_px.astype(np.int32), 1)
        cell[(m > 0) & low] = k + 1
    label[low & (cell > 0)] = 4
    label[low & (cell < 0)] = 2        # inside the outline but in no cell: outer face / foot of the wall

    # colours of the original image at VGGT resolution (same resize as load_and_preprocess_images)
    rgb = cv2.cvtColor(cv2.resize(img, (vw, vh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

    flat = lambda a: a.reshape(-1, *a.shape[2:])
    write_ply(args.out / "points_labeled.ply", flat(xyz), flat(COLORS[label]), flat(label), flat(cell))
    write_ply(args.out / "points_rgb.ply", flat(xyz), flat(rgb), flat(label), flat(cell))
    np.savez_compressed(args.out / "points_labeled.npz", xyz_m=xyz.astype(np.float32), rgb=rgb, label=label,
                        cell=cell, height_m=h.astype(np.float32), names=np.array(NAMES),
                        rows=res["rows"], cols=res["cols"])

    vis = COLORS[label][..., ::-1].copy()
    for k, c in enumerate(res["cells"]):
        cx, cy = c.corners_px.mean(0).astype(int)
        cv2.putText(vis, str(k + 1), (cx - 6, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    vis = cv2.resize(vis, (vw * 2, vh * 2), interpolation=cv2.INTER_NEAREST)
    for i, nm in enumerate(NAMES):
        cv2.rectangle(vis, (10, 10 + 22 * i), (28, 28 + 22 * i), COLORS[i][::-1].tolist(), -1)
        cv2.putText(vis, nm, (34, 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.imwrite(str(args.out / "labels.png"), np.hstack([cv2.resize(np.ascontiguousarray(rgb[..., ::-1]),
                                                                    (vw * 2, vh * 2)), vis]))

    # metric sizes: box outline at table level, cell openings at OPENING_H
    bw, bl_ = (v * scale for v in quad_size(px_to_plane(outline, K, n, d, 0.0)))
    sizes = []
    for k, c in enumerate(res["cells"]):
        w, l = quad_size(px_to_plane(c.corners_px, K, n, d, OPENING_H / scale))
        sizes.append({"cell": k + 1, "row": c.row + 1, "col": c.col + 1,
                      "width_cm": round(w * scale * 100, 1), "length_cm": round(l * scale * 100, 1),
                      "area_cm2": round(w * l * scale ** 2 * 1e4, 1)})
    summary = {"rows": res["rows"], "cols": res["cols"], "box_outer_cm": [round(bw * 100, 1), round(bl_ * 100, 1)],
               "wall_height_vggt_cm": round(wall_h * 100, 1), "cells": sizes}
    (args.out / "box_dimensions.json").write_text(json.dumps(summary, indent=1))
    ws = [c["width_cm"] for c in sizes]
    ls = [c["length_cm"] for c in sizes]
    print(f"box outer {bw*100:.1f} x {bl_*100:.1f} cm | cell opening width {np.mean(ws):.1f} cm "
          f"({min(ws):.1f}-{max(ws):.1f}), length {np.mean(ls):.1f} cm ({min(ls):.1f}-{max(ls):.1f})")
    for r in range(res["rows"]):
        print("   " + "  ".join(f"{c['cell']:2d}: {c['width_cm']:4.1f}x{c['length_cm']:4.1f}"
                                for c in sizes[r * res["cols"]:(r + 1) * res["cols"]]))

    counts = {NAMES[i]: int((label == i).sum()) for i in range(len(NAMES))}
    print(f"{res['rows']}x{res['cols']} cells, wall height {wall_h*100:.1f} cm, scale {scale:.3f}, "
          f"{vh*vw} points: {counts}")
    print("cell floor points per cell:", {k: int((cell == k).sum()) for k in range(1, res['rows'] * res['cols'] + 1)})


if __name__ == "__main__":
    main()
