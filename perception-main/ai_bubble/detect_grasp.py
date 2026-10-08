#!/usr/bin/env python3
"""
Block grasp detection: RF-DETR boxes -> SAM2 masks (inside each box) -> RANSAC rectangle
-> parallel-gripper grasp (center, jaw axis across the short side, opening width).

All outputs are in image pixels; converting to robot coordinates needs camera
intrinsics + depth + hand-eye calibration (not done here).

Usage:
    HF_HUB_OFFLINE=1 .venv/bin/python detect_grasp.py img1.jpg [img2.jpg ...]
        [--weights output/checkpoint_best_total.pth] [--threshold 0.5] [--out grasp]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from rfdetr import RFDETRMedium
from transformers import Sam2Model, Sam2Processor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "research" / "box_cells"))
from rect_fit import grasp_from_rect, ransac_rect  # noqa: E402

SAM_ID = "facebook/sam2.1-hiera-large"


class BlockGraspDetector:
    def __init__(self, weights, resolution=768, threshold=0.5, device="cuda"):
        self.det = RFDETRMedium(pretrain_weights=str(weights), resolution=resolution)
        self.threshold = threshold
        self.device = device
        self.sam_proc = Sam2Processor.from_pretrained(SAM_ID)
        self.sam = Sam2Model.from_pretrained(SAM_ID).to(device).eval()

    def __call__(self, img_rgb: Image.Image):
        t0 = time.time()
        d = self.det.predict(img_rgb, threshold=self.threshold)
        t_det = time.time() - t0
        boxes = [list(map(float, b)) for b in d.xyxy]
        results = []
        if boxes:
            t0 = time.time()
            sin = self.sam_proc(images=img_rgb, input_boxes=[boxes], return_tensors="pt").to(self.device)
            with torch.no_grad():
                so = self.sam(**sin, multimask_output=False)
            masks = self.sam_proc.post_process_masks(so.pred_masks.cpu(), sin["original_sizes"])[0][:, 0].numpy()
            t_seg = time.time() - t0
            for b, s, m in zip(boxes, d.confidence, masks):
                x1, y1, x2, y2 = map(int, b)
                inside = np.zeros(m.shape, bool)
                inside[y1:y2 + 1, x1:x2 + 1] = True
                m = m.astype(bool) & inside                      # only what is inside the detection box
                cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                r = {"box": [x1, y1, x2, y2], "score": float(s), "mask_area": int(m.sum())}
                if cnts:
                    cnt = max(cnts, key=cv2.contourArea)[:, 0, :]
                    center, size, ang, ratio, corners = ransac_rect(cnt)
                    c, jaw, opening, length = grasp_from_rect(center, size, ang)
                    r.update(center=[float(c[0]), float(c[1])], jaw_axis_deg=float(jaw),
                             opening_px=float(opening), length_px=float(length),
                             rect_fit=float(ratio), corners=corners.tolist(), contour=cnt)
                results.append(r)
        else:
            t_seg = 0.0
        return results, (t_det, t_seg)


def draw(img_bgr, results):
    out = img_bgr.copy()
    for i, r in enumerate(results):
        if "corners" not in r:
            continue
        cv2.drawContours(out, [r["contour"].reshape(-1, 1, 2)], -1, (0, 200, 255), 1)
        cv2.polylines(out, [np.round(r["corners"]).astype(np.int32)], True, (255, 255, 255), 2)
        cx, cy = r["center"]
        a = np.radians(r["jaw_axis_deg"])
        hw = r["opening_px"] / 2 + 8
        p1 = (int(cx - hw * np.cos(a)), int(cy - hw * np.sin(a)))
        p2 = (int(cx + hw * np.cos(a)), int(cy + hw * np.sin(a)))
        cv2.line(out, p1, p2, (0, 0, 255), 2)
        cv2.circle(out, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        x1, y1 = r["box"][:2]
        cv2.putText(out, f"#{i+1} {r['score']:.2f} {r['opening_px']:.0f}x{r['length_px']:.0f}px "
                         f"{r['jaw_axis_deg']:.0f}deg fit{r['rect_fit']*100:.0f}%",
                    (x1 - 40, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--weights", default=HERE / "output" / "checkpoint_best_total.pth", type=Path)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=HERE / "grasp", type=Path)
    args = ap.parse_args()

    det = BlockGraspDetector(args.weights, threshold=args.threshold)
    args.out.mkdir(parents=True, exist_ok=True)
    for p in args.images:
        results, (t_det, t_seg) = det(Image.open(p).convert("RGB"))
        print(f"{p.name}: {len(results)} blocks  (RF-DETR {t_det*1000:.0f} ms, SAM2 {t_seg*1000:.0f} ms)")
        for i, r in enumerate(results):
            if "center" in r:
                print(f"  #{i+1} score {r['score']:.2f}  center ({r['center'][0]:.0f},{r['center'][1]:.0f})  "
                      f"opening {r['opening_px']:.0f}px  length {r['length_px']:.0f}px  "
                      f"jaw {r['jaw_axis_deg']:.0f}deg  rect fit {r['rect_fit']*100:.0f}%")
        cv2.imwrite(str(args.out / f"{p.stem}_grasp.jpg"), draw(cv2.imread(str(p)), results))
        clean = [{k: v for k, v in r.items() if k != "contour"} for r in results]
        (args.out / f"{p.stem}_grasp.json").write_text(json.dumps(clean, indent=1))


if __name__ == "__main__":
    main()
