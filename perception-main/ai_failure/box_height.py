#!/usr/bin/env python3
"""
Box height from one top-camera frame: Grounding DINO box ROI + VGGT point map + RealSense scale.

  1. Grounding DINO "cardboard box." -> box ROI in the color image
  2. VGGT -> camera-frame point map for every color pixel (shape right, scale unknown)
  3. RANSAC table plane on VGGT points OUTSIDE the ROI; heights = distance to that plane
     along its normal (camera tilt does not matter)
  4. metric scale = RealSense table distance on the optical axis / VGGT table distance on the axis
  5. also: RealSense-only height (its own table plane + its own box blob) for comparison

Usage (ai_bubble/.venv):
    HF_HUB_OFFLINE=1 ../ai_bubble/.venv/bin/python box_height.py [--image img.jpg --depth depth.raw]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "research" / "box_cells"))
import box_grid  # noqa: E402
from capture_cells import find_box_roi, to_vggt_px  # noqa: E402
from camera import DEPTH_DEV, DH, DW, grab_color  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


def grab_depth_raw(path, n=10):
    subprocess.run(["v4l2-ctl", "-d", DEPTH_DEV, f"--set-fmt-video=width={DW},height={DH},pixelformat=0"],
                   check=True, capture_output=True)
    subprocess.run(["v4l2-ctl", "-d", DEPTH_DEV, "--stream-mmap", f"--stream-count={n}", f"--stream-to={path}"],
                   check=True, capture_output=True)


def load_depth_m(path, n=10):
    d = np.fromfile(path, np.uint16).reshape(-1, DH, DW)[-n:].astype(np.float32) / 1000.0
    d[d == 0] = np.nan
    with np.errstate(all="ignore"):
        return np.nanmedian(d, 0)


def ransac_plane(P, iters=800, tol=0.004, max_tilt_deg=35, rng=None):
    """Plane n.x + d = 0 (unit n, pointing toward the camera) fitted to Nx3 points. Only planes
    within max_tilt_deg of facing the camera are considered, so a wall can never win."""
    rng = np.random.default_rng(0) if rng is None else rng    # fresh per call: same samples every scan
    best, best_n = None, -1
    min_nz = np.cos(np.radians(max_tilt_deg))
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n[2]) < min_nz:
            continue
        d = -n @ a
        k = (np.abs(P @ n + d) < tol).sum()
        if k > best_n:
            best, best_n = (n, d), k
    n, d = best
    inl = np.abs(P @ n + d) < tol
    c = P[inl].mean(0)
    _, _, vt = np.linalg.svd(P[inl] - c, full_matrices=False)   # full U would be N x N (seconds)
    n = vt[2]
    d = -n @ c
    if d < 0:            # camera (origin) on the positive side: heights above the table are positive
        n, d = -n, -d
    return n, d, inl.mean()


def vggt_points(img_path):
    model = VGGT.from_pretrained("facebook/VGGT-1B-Commercial").to("cuda").eval()
    images = load_and_preprocess_images([str(img_path)]).to("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(images[None])
    extri, intri = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    depth = pred["depth"][0].float().cpu().numpy()                       # 1xHxWx1
    pts = unproject_depth_map_to_point_map(depth, extri[0].float().cpu().numpy(),
                                           intri[0].float().cpu().numpy())[0]
    return pts, intri[0, 0].float().cpu().numpy()                        # HxWx3 (world == first camera)


def roi_masks(shape, roi_v, gap=0.02, ring=0.12, shrink=0.05):
    """table = a thin ring hugging the box ROI (gap..ring of the box size): the table the box
    stands on, not the floor or a window sill further away. box = the ROI shrunk a little."""
    h, w = shape
    x1, y1, x2, y2 = roi_v
    bw, bh = x2 - x1, y2 - y1
    yy, xx = np.mgrid[0:h, 0:w]

    def grown(f):
        return (xx >= x1 - f * bw) & (xx <= x2 + f * bw) & (yy >= y1 - f * bh) & (yy <= y2 + f * bh)

    table = grown(ring) & ~grown(gap)
    inside = grown(-shrink)
    return table, inside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path)
    ap.add_argument("--depth", type=Path)
    ap.add_argument("--out", default=HERE / "data" / "height", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    img_path = args.image or args.out / "frame.jpg"
    depth_path = args.depth or args.out / "depth.raw"
    if not args.image:
        cv2.imwrite(str(img_path), grab_color(), [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not args.depth:
        grab_depth_raw(depth_path)
    img = cv2.imread(str(img_path))

    roi = find_box_roi(img)
    if roi is None:
        sys.exit("no box found by Grounding DINO")

    # --- VGGT geometry
    pts, K = vggt_points(img_path)
    vh, vw = pts.shape[:2]
    roi_v = to_vggt_px(roi, (vh, vw), img.shape[:2])
    table_m, box_m = roi_masks((vh, vw), roi_v)
    n, d, frac = ransac_plane(pts[table_m].reshape(-1, 3))
    h_v = pts @ n + d                                                   # VGGT units above the table
    z_axis_v = d / n[2] if abs(n[2]) > 1e-6 else np.nan                 # table distance along the optical axis
    tilt = np.degrees(np.arccos(abs(n[2])))

    # --- RealSense: table distance on the optical axis (its own pixel-plane fit) + its own box height
    rs = load_depth_m(depth_path)
    plane_rs = box_grid.fit_table_plane(rs)
    z_axis_rs = float(plane_rs[DH // 2, DW // 2])
    scale = z_axis_rs / abs(z_axis_v)
    h_m = h_v * scale

    q = lambda a, p: float(np.nanpercentile(a, p)) * 100
    box_h = h_m[box_m]
    print(f"table plane: {frac*100:.0f}% inliers, camera tilt {tilt:.1f} deg, table {z_axis_rs:.3f} m on the axis")
    print(f"VGGT box heights (cm): median {q(box_h,50):.1f}  p90 {q(box_h,90):.1f}  p95 {q(box_h,95):.1f}  "
          f"p99 {q(box_h,99):.1f}")
    print(f"VGGT table spread (cm): p5 {q(h_m[table_m],5):.1f}  p95 {q(h_m[table_m],95):.1f}")

    h_rs = plane_rs - rs
    try:
        corners = box_grid.find_box_outline(h_rs, 0.05)
        mask = np.zeros(rs.shape, np.uint8)
        cv2.fillConvexPoly(mask, corners.astype(np.int32), 1)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8)).astype(bool)
        r = h_rs[mask]
        print(f"RealSense-only box heights (cm): median {q(r,50):.1f}  p90 {q(r,90):.1f}  p95 {q(r,95):.1f}  "
              f"p99 {q(r,99):.1f}")
    except RuntimeError as e:
        print("RealSense-only: box not found:", e)

    # height map preview (color frame size)
    full = cv2.resize(np.nan_to_num(h_m).astype(np.float32), (img.shape[1], img.shape[0]))
    vis = cv2.applyColorMap((255 * np.clip(full / 0.20, 0, 1)).astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(img, 0.45, vis, 0.55, 0)
    cv2.rectangle(out, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])), (255, 255, 255), 2)
    cv2.putText(out, f"box top ~{q(box_h,95):.1f} cm (p95)  table {z_axis_rs:.2f} m  tilt {tilt:.1f} deg",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imwrite(str(args.out / "height_map.png"), out)
    np.save(args.out / "height_m.npy", h_m)


if __name__ == "__main__":
    main()
