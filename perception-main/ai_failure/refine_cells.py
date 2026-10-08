#!/usr/bin/env python3
"""
Refine the partition grid with SAM2 + RANSAC rectangles.

  box:   Grounding DINO box -> SAM2 (box prompt) -> RANSAC rectangle  = box outline + rotation
  cells: each grid cell (capture_cells.py) -> SAM2 (point at the cell centre + the cell quad,
         grown a little, as box prompt) -> mask of that cell's opening -> RANSAC rectangle
         = per-cell centre, yaw and size that follow bent / tilted dividers.

Usage (ai_bubble/.venv):
    HF_HUB_OFFLINE=1 ../ai_bubble/.venv/bin/python refine_cells.py --image data/box4_001.jpg \
        --cells data/box4_001_cells.json
Writes <name>_refined.json and <name>_refined.png next to the cells file.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "research" / "box_cells"))
from rect_fit import ransac_rect  # noqa: E402
from capture_cells import find_box_roi  # noqa: E402

SAM_ID = "facebook/sam2.1-hiera-large"
GROW = 0.15          # box prompt = cell quad bbox grown by this fraction (cell walls must be inside)
CLIP = 0.10          # SAM mask is clipped to the grid cell grown by this fraction


def biggest_contour(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cnts, key=cv2.contourArea)[:, 0, :] if cnts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--cells", required=True, type=Path)
    args = ap.parse_args()

    img_bgr = cv2.imread(str(args.image))
    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    H, W = img_bgr.shape[:2]
    layout = json.loads(args.cells.read_text())
    proc = Sam2Processor.from_pretrained(SAM_ID)
    sam = Sam2Model.from_pretrained(SAM_ID).to("cuda").eval()

    def segment(boxes, points=None, multi=False):
        kw = {"input_boxes": [boxes]}
        if points is not None:
            kw["input_points"] = [[[p] for p in points]]
            kw["input_labels"] = [[[1] for _ in points]]
        inp = proc(images=img, return_tensors="pt", **kw).to("cuda")
        with torch.no_grad():
            out = sam(**inp, multimask_output=multi)
        m = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0].numpy()  # N x K x H x W
        return m if multi else m[:, 0]

    def clean_cell_mask(cands, quad):
        """Pick the candidate that best overlaps the grid cell, keep it inside the grown cell
        (no leaking into wall faces / neighbours) and fill holes (a block in the cell)."""
        g = np.asarray(quad, np.float32)
        c = g.mean(0)
        grid = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(grid, np.round(g).astype(np.int32), 1)
        grown = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(grown, np.round(c + (g - c) * (1 + CLIP)).astype(np.int32), 1)
        grid, grown = grid.astype(bool), grown.astype(bool)
        best, best_iou = None, -1
        for m in cands:
            m = m.astype(bool) & grown
            iou = (m & grid).sum() / max((m | grid).sum(), 1)
            if iou > best_iou:
                best, best_iou = m, iou
        filled = np.zeros((H, W), np.uint8)
        cnts, _ = cv2.findContours(best.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 1, -1)
        return filled.astype(bool), best_iou

    # --- box outline
    roi = find_box_roi(img_bgr)
    box_mask = segment([[float(v) for v in roi]])[0]
    bc = biggest_contour(box_mask)
    b_center, b_size, b_ang, b_fit, b_corners = ransac_rect(bc)

    # --- cells
    quads = [np.asarray(c["quad"], np.float32) for c in layout["cells"]]
    prompts, centers = [], []
    for q in quads:
        x1, y1 = q.min(0)
        x2, y2 = q.max(0)
        gx, gy = GROW * (x2 - x1), GROW * (y2 - y1)
        prompts.append([float(max(x1 - gx, 0)), float(max(y1 - gy, 0)),
                        float(min(x2 + gx, W - 1)), float(min(y2 + gy, H - 1))])
        centers.append([float(v) for v in q.mean(0)])
    cand = segment(prompts, centers, multi=True)
    cleaned = [clean_cell_mask(cand[i], quads[i]) for i in range(len(quads))]
    masks = [m for m, _ in cleaned]

    refined, out = [], (img_bgr * 0.55).astype(np.uint8)
    for c, q, m in zip(layout["cells"], quads, masks):
        cnt = biggest_contour(m)
        r = {"id": c["id"], "row": c["row"], "col": c["col"], "grid_quad": c["quad"]}
        if cnt is not None and len(cnt) > 20:
            center, (w, h), ang, fit, corners = ransac_rect(cnt)
            grid_area = cv2.contourArea(q)
            r.update(center_px=[float(center[0]), float(center[1])], size_px=[float(min(w, h)), float(max(w, h))],
                     angle_deg=float(ang % 90), rect_fit=float(fit), quad=corners.round(1).tolist(),
                     mask_area_vs_grid=float(m.sum() / grid_area) if grid_area > 0 else None)
            col = (0, 220, 0) if fit > 0.6 else (0, 160, 255)
            out[m.astype(bool)] = (out[m.astype(bool)] * 0.5 + np.array(col) * 0.5).astype(np.uint8)
            cv2.polylines(out, [np.round(corners).astype(np.int32)], True, (255, 255, 255), 2)
            cv2.circle(out, (int(center[0]), int(center[1])), 4, (0, 0, 255), -1)
            cv2.putText(out, f"{c['id']}", (int(center[0]) - 8, int(center[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.polylines(out, [np.round(q).astype(np.int32)], True, (0, 255, 255), 1)   # original grid quad
        refined.append(r)
    cv2.polylines(out, [np.round(b_corners).astype(np.int32)], True, (255, 0, 255), 2)
    cv2.putText(out, f"box: SAM2+RANSAC rect, fit {b_fit*100:.0f}%, angle {b_ang % 90:.1f} deg  "
                     f"| cells: white = SAM2+RANSAC, yellow = line grid", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    stem = args.cells.name.replace("_cells.json", "")
    (args.cells.parent / f"{stem}_refined.json").write_text(json.dumps(
        {"image": args.image.name, "rows": layout["rows"], "cols": layout["cols"],
         "box": {"center_px": [float(v) for v in b_center], "size_px": [float(v) for v in b_size],
                 "angle_deg": float(b_ang % 90), "rect_fit": float(b_fit), "quad": b_corners.round(1).tolist()},
         "cells": refined}, indent=1))
    cv2.imwrite(str(args.cells.parent / f"{stem}_refined.png"), out)
    fits = [r["rect_fit"] for r in refined if "rect_fit" in r]
    print(f"{stem}: box fit {b_fit*100:.0f}% angle {b_ang % 90:.1f} deg | {len(fits)}/{len(refined)} cells refined, "
          f"rect fit {np.mean(fits)*100:.0f}% (min {min(fits)*100:.0f}%)")
    for r in refined:
        if "rect_fit" in r:
            print(f"  cell {r['id']:2d}: {r['size_px'][0]:5.1f} x {r['size_px'][1]:5.1f} px  angle {r['angle_deg']:4.1f}  "
                  f"fit {r['rect_fit']*100:3.0f}%  mask/grid area {r['mask_area_vs_grid']:.2f}")


if __name__ == "__main__":
    main()
