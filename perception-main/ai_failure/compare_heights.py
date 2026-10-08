#!/usr/bin/env python3
"""
Compare RealSense-depth heights with VGGT heights, pixel by pixel, per label and per cell.

The RealSense depth and the color/VGGT image are not registered (no librealsense here), so the
RealSense height map is warped onto the VGGT grid with the homography between the two box
outlines (4 corners). Both height maps are "height above the table plane" in metres, each from
its own table-plane fit.

Usage:
    python compare_heights.py --labeled data/vggt/box2/labeled/points_labeled.npz \
                              --depth data/vggt/box2/depth_b2.raw --out data/vggt/box2/labeled
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "research" / "box_cells"))
import box_grid  # noqa: E402

DW, DH = 848, 480
NAMES = ["other", "table", "box_wall", "divider", "cell_floor"]


def order(pts):
    s, d = pts.sum(1), pts[:, 1] - pts[:, 0]
    return np.float32([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


def rs_height(depth_path, n=10):
    d = np.fromfile(depth_path, np.uint16).reshape(-1, DH, DW)[-n:].astype(np.float32) / 1000.0
    d[d == 0] = np.nan
    with np.errstate(all="ignore"):
        d = np.nanmedian(d, 0)
    return box_grid.fit_table_plane(d) - d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True, type=Path)
    ap.add_argument("--depth", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    L = np.load(args.labeled)
    label, cell, h_v = L["label"], L["cell"], L["height_m"]
    vh, vw = label.shape

    # box outline in VGGT pixels (wall + divider + floor labels)
    box = np.isin(label, [2, 3, 4]).astype(np.uint8)
    cnts, _ = cv2.findContours(box, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c_v = order(cv2.boxPoints(cv2.minAreaRect(max(cnts, key=cv2.contourArea))))

    # box outline in RealSense pixels
    h_r = rs_height(args.depth)
    c_r = box_grid.find_box_outline(h_r, 0.05)

    Hm = cv2.getPerspectiveTransform(c_r, c_v)
    valid = np.isfinite(h_r).astype(np.float32)
    h_rw = cv2.warpPerspective(np.nan_to_num(h_r), Hm, (vw, vh), flags=cv2.INTER_LINEAR)
    v_w = cv2.warpPerspective(valid, Hm, (vw, vh), flags=cv2.INTER_LINEAR)
    h_rw[v_w < 0.99] = np.nan

    both = np.isfinite(h_rw) & np.isfinite(h_v)
    cm = lambda a: a * 100
    lines = [f"{'region':12s} {'n':>6s} | {'VGGT med':>8s} {'RS med':>7s} | {'VGGT p99':>8s} {'RS p99':>7s} | "
             f"{'|diff| med':>10s} {'diff med (V-R)':>14s}  (cm)"]
    for k in (1, 2, 3, 4):
        m = both & (label == k)
        if m.sum() < 20:
            continue
        v, r = h_v[m], h_rw[m]
        lines.append(f"{NAMES[k]:12s} {m.sum():6d} | {cm(np.median(v)):8.1f} {cm(np.median(r)):7.1f} | "
                     f"{cm(np.percentile(v, 99)):8.1f} {cm(np.percentile(r, 99)):7.1f} | "
                     f"{cm(np.median(np.abs(v - r))):10.1f} {cm(np.median(v - r)):14.1f}")
    lines.append("")
    lines.append("cell floors (median height, cm):  cell: VGGT / RS")
    ids = [i for i in np.unique(cell) if i > 0]
    row = []
    for i in ids:
        m = both & (cell == i)
        row.append(f"{i:2d}: {cm(np.median(h_v[m])):5.1f} / {cm(np.median(h_rw[m])):5.1f}" if m.sum() > 20 else f"{i:2d}: n/a")
    cols = int(L["cols"])
    for r0 in range(0, len(row), cols):
        lines.append("   " + "   ".join(row[r0:r0 + cols]))
    text = "\n".join(lines)
    print(text)
    (args.out / "height_compare.txt").write_text(text)

    # visual: VGGT | RealSense (warped) | difference
    def cmap(a, lo, hi):
        im = cv2.applyColorMap((255 * np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0, 1)).astype(np.uint8),
                               cv2.COLORMAP_JET)
        im[~np.isfinite(a)] = 0
        return im
    diff = np.where(both, h_v - h_rw, np.nan)
    tiles = [cmap(h_v, 0, 0.16), cmap(h_rw, 0, 0.16), cmap(diff, -0.05, 0.05)]
    for t, s in zip(tiles, ["VGGT height 0-16cm", "RealSense height 0-16cm", "VGGT - RS  -5..+5cm"]):
        cv2.putText(t, s, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.polylines(t, [c_v.astype(np.int32)], True, (255, 255, 255), 1)
    cv2.imwrite(str(args.out / "height_compare.png"),
                cv2.resize(np.hstack(tiles), None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST))


if __name__ == "__main__":
    main()
