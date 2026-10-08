#!/usr/bin/env python3
"""
Capture one top-camera frame and split the partition box into numbered cells.

  color (V4L2 /dev/video4) + RealSense depth (/dev/video0, only for metric scale)
  -> Grounding DINO "cardboard box." on the color image (box ROI, keeps the robot arm out)
  -> VGGT depth from the color image -> box_lines.detect (Canny + RANSAC grid) inside the ROI
  -> cell quads in color-image pixels, saved next to the frame.

Outputs in <out>/:
  frame_XXX.jpg          raw color frame
  frame_XXX_cells.json   {"rows", "cols", "cells": [{"id", "row", "col", "quad": 4x2 px}]}
  frame_XXX_cells.png    preview with numbered cells (1..N, row-major)

Run in ai_bubble/.venv (has VGGT + transformers):
    HF_HUB_OFFLINE=1 ../ai_bubble/.venv/bin/python capture_cells.py [--out data] [--name frame_000]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "research" / "box_cells"))
import box_lines as bl  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402

from camera import grab_color, grab_depth_median_m  # noqa: E402


class VggtDepth:
    def __init__(self):
        self.model = VGGT.from_pretrained("facebook/VGGT-1B-Commercial").to("cuda").eval()

    def __call__(self, img_path, metric_ref_m):
        images = load_and_preprocess_images([str(img_path)]).to("cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = self.model(images[None])
        depth = pred["depth"][0, 0, ..., 0].float().cpu().numpy()
        h, w = depth.shape
        centre = depth[int(0.29 * h):int(0.71 * h), int(0.29 * w):int(0.71 * w)]
        return depth * (metric_ref_m / np.median(centre))


def find_box_roi(img_bgr, prompt="cardboard box.", grow=0.03):
    """Grounding DINO box around the partition box (color pixels, x1 y1 x2 y2) or None.
    Prefers the detection containing the image centre, then the highest score."""
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    mid = "IDEA-Research/grounding-dino-base"
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to("cuda").eval()
    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    W, H = img.size
    inp = proc(images=img, text=prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        r = proc.post_process_grounded_object_detection(model(**inp), inp.input_ids, threshold=0.3,
                                                        text_threshold=0.25, target_sizes=[(H, W)])[0]
    cands = []
    for b, s in zip(r["boxes"].tolist(), r["scores"].tolist()):
        if (b[2] - b[0]) * (b[3] - b[1]) > 0.85 * W * H:          # whole table / scene
            continue
        central = b[0] <= W / 2 <= b[2] and b[1] <= H / 2 <= b[3]
        cands.append((central, s, b))
    if not cands:
        return None
    # a box on a table often yields two central hits: the box and the table around it.
    # Among central hits scoring close to the best, take the smallest one (the box).
    central = [c for c in cands if c[0]] or cands
    best = max(s for _, s, _ in central)
    close = [c for c in central if c[1] >= 0.75 * best]
    _, _, (x1, y1, x2, y2) = min(close, key=lambda c: (c[2][2] - c[2][0]) * (c[2][3] - c[2][1]))
    gx, gy = grow * (x2 - x1), grow * (y2 - y1)
    return [x1 - gx, y1 - gy, x2 + gx, y2 + gy]


def to_vggt_px(xyxy, vggt_shape, color_shape):
    """Color pixels -> VGGT pixels. load_and_preprocess_images ('crop' mode) resizes a landscape
    frame to width 518 and height rounded to a multiple of 14 (no cropping for 16:9)."""
    vh, vw = vggt_shape
    ch, cw = color_shape
    x1, y1, x2, y2 = xyxy
    return [x1 * vw / cw, y1 * vh / ch, x2 * vw / cw, y2 * vh / ch]


def to_color_px(quad, vggt_shape, color_shape):
    """VGGT pixels -> color pixels (inverse of to_vggt_px)."""
    vh, vw = vggt_shape
    ch, cw = color_shape
    q = np.asarray(quad, np.float64)
    return np.stack([q[:, 0] * cw / vw, q[:, 1] * ch / vh], 1)


def preview(img, cells, rows, cols):
    out = img.copy()
    for c in cells:
        q = np.round(c["quad"]).astype(np.int32)
        cv2.polylines(out, [q], True, (0, 255, 255), 2)
        cx, cy = q.mean(0).astype(int)
        cv2.putText(out, str(c["id"]), (cx - 12, cy + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5)
        cv2.putText(out, str(c["id"]), (cx - 12, cy + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
    cv2.putText(out, f"{rows}x{cols} cells", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=HERE / "data", type=Path)
    ap.add_argument("--name", default=None)
    ap.add_argument("--image", type=Path, default=None, help="use an existing color image instead of capturing")
    ap.add_argument("--ref-m", type=float, default=None, help="table distance in m (skip depth capture)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    name = args.name or f"frame_{len(list(args.out.glob('frame_*_cells.json'))):03d}"

    img = cv2.imread(str(args.image)) if args.image else grab_color()
    ref = args.ref_m if args.ref_m else grab_depth_median_m()
    img_path = args.out / f"{name}.jpg"
    cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    roi = find_box_roi(img)
    depth = VggtDepth()(img_path, ref)
    res = bl.detect(depth, roi=to_vggt_px(roi, depth.shape, img.shape[:2]) if roi else None)
    cells = [{"id": i + 1, "row": c.row, "col": c.col,
              "quad": to_color_px(c.corners_px, depth.shape, img.shape[:2]).round(1).tolist()}
             for i, c in enumerate(res["cells"])]
    (args.out / f"{name}_cells.json").write_text(json.dumps(
        {"image": img_path.name, "rows": res["rows"], "cols": res["cols"], "ref_m": ref, "cells": cells}, indent=1))
    pv = preview(img, cells, res["rows"], res["cols"])
    if roi:
        cv2.rectangle(pv, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])), (255, 0, 255), 2)
    cv2.imwrite(str(args.out / f"{name}_cells.png"), pv)
    print(f"{name}: {res['rows']}x{res['cols']} cells, table ref {ref:.3f} m -> {args.out / (name + '_cells.png')}")


if __name__ == "__main__":
    main()
