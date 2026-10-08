#!/usr/bin/env python3
"""
Compare VGGT-1B (non-commercial) and VGGT-1B-Commercial on our partition scenes:
grid found (rows x cols), number of RANSAC lines, wall / divider / cell-floor heights,
table flatness, inference time. Both depth maps use the same metric scale reference.

Usage (ai_bubble/.venv):  HF_HUB_OFFLINE=1 ../ai_bubble/.venv/bin/python compare_vggt_ckpt.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "research" / "box_cells"))
import box_lines as bl  # noqa: E402
import box_height as bh  # noqa: E402
from capture_cells import find_box_roi, to_vggt_px  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402

D = HERE.parent / "data" / "vggt"
SCENES = [("4x4 (cut off)", D / "box2/cam4_b2.jpg"), ("3x3 old", D / "cam4.jpg"),
          ("4x4 full", HERE / "data/box4_001.jpg"), ("3x3 current", HERE / "data/frame_000.jpg"),
          ("partition only", HERE / "data/scene/scene_000.jpg")]
CKPTS = ["facebook/VGGT-1B", "facebook/VGGT-1B-Commercial"]
REF_M = 0.85   # common metric reference for the table distance (only the relative comparison matters)


def run(model, img_path):
    images = load_and_preprocess_images([str(img_path)]).to("cuda")
    torch.cuda.synchronize()
    t = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(images[None])
    torch.cuda.synchronize()
    dt = time.time() - t
    extri, intri = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    depth = pred["depth"][0].float().cpu().numpy()
    pts = unproject_depth_map_to_point_map(depth, extri[0].float().cpu().numpy(), intri[0].float().cpu().numpy())[0]
    return pts, dt


def main():
    rois = {}
    for name, p in SCENES:
        img = cv2.imread(str(p))
        rois[name] = (img.shape[:2], find_box_roi(img))
    rows = []
    vis = {}
    for ck in CKPTS:
        model = VGGT.from_pretrained(ck).to("cuda").eval()
        run(model, SCENES[0][1])                                   # warm-up
        for name, p in SCENES:
            pts, dt = run(model, p)
            vh, vw = pts.shape[:2]
            shape, roi = rois[name]
            roi_v = to_vggt_px(roi, (vh, vw), shape)
            ring, inside = bh.roi_masks((vh, vw), roi_v)
            n, d, frac = bh.ransac_plane(pts[ring].reshape(-1, 3))
            scale = REF_M / abs(d / n[2])
            h = (pts @ n + d) * scale
            depth_m = pts[..., 2] * scale
            try:
                res = bl.detect(depth_m, roi=roi_v)
                grid, nl = f"{res['rows']}x{res['cols']}", len(res["lines"])
                vis[(ck, name)] = bl.render(res)
            except Exception as e:  # noqa: BLE001
                grid, nl = f"fail ({type(e).__name__})", 0
            hb = h[inside]
            rows.append((name, ck.split("/")[1], grid, nl, np.percentile(hb, 99) * 100, np.percentile(hb, 20) * 100,
                         np.std(h[ring][np.abs(h[ring]) < 0.02]) * 1000, frac * 100, dt * 1000))
        del model
        torch.cuda.empty_cache()

    print(f"{'scene':15s} {'checkpoint':20s} {'grid':6s} {'lines':>5s} {'wall p99':>8s} {'floor p20':>9s} "
          f"{'table std':>9s} {'plane inl':>9s} {'time':>6s}")
    for r in sorted(rows, key=lambda r: (r[0], r[1])):
        print(f"{r[0]:15s} {r[1]:20s} {r[2]:6s} {r[3]:5d} {r[4]:7.1f}cm {r[5]:8.1f}cm {r[6]:7.1f}mm {r[7]:8.0f}% "
              f"{r[8]:5.0f}ms")
    out = HERE / "data" / "vggt_ckpt_compare"
    out.mkdir(parents=True, exist_ok=True)
    for name, _ in SCENES:
        a, b = vis.get((CKPTS[0], name)), vis.get((CKPTS[1], name))
        if a is not None and b is not None:
            w = min(a.shape[1], b.shape[1])
            cv2.imwrite(str(out / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.png"),
                        np.vstack([a[:, :w], b[:, :w]]))
    print("images ->", out, "(top = VGGT-1B, bottom = VGGT-1B-Commercial)")


if __name__ == "__main__":
    main()
