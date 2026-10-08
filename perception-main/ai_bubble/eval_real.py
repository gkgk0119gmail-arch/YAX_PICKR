#!/usr/bin/env python3
"""
Run the trained block detector on real photos and draw the result.

Usage:
    .venv/bin/python eval_real.py [images ...] [--weights output/checkpoint_best_total.pth]
                                  [--threshold 0.5] [--out eval]
With no images, uses the real test split (dataset/test).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rfdetr import RFDETRMedium

HERE = Path(__file__).resolve().parent


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", type=Path)
    ap.add_argument("--weights", default=HERE / "output" / "checkpoint_best_total.pth", type=Path)
    ap.add_argument("--resolution", type=int, default=768)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=HERE / "eval", type=Path)
    args = ap.parse_args()

    gt = {}
    if not args.images:
        test = HERE / "dataset" / "test"
        c = json.loads((test / "_annotations.coco.json").read_text())
        names = {im["id"]: im["file_name"] for im in c["images"]}
        for a in c["annotations"]:
            x, y, w, h = a["bbox"]
            gt.setdefault(names[a["image_id"]], []).append([x, y, x + w, y + h])
        args.images = [test / n for n in names.values()]

    model = RFDETRMedium(pretrain_weights=str(args.weights), resolution=args.resolution)
    args.out.mkdir(parents=True, exist_ok=True)
    for p in args.images:
        det = model.predict(Image.open(p).convert("RGB"), threshold=args.threshold)
        img = cv2.imread(str(p))
        for b in gt.get(p.name, []):
            cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (255, 255, 255), 1)
        hits = 0
        for b, s in zip(det.xyxy, det.confidence):
            best = max((iou(b, g) for g in gt.get(p.name, [])), default=0)
            hits += best >= 0.5
            cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 200, 0), 2)
            cv2.putText(img, f"{s:.2f}", (int(b[0]), int(b[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 160, 0), 2)
        msg = f"{p.name}: {len(det.xyxy)} detections"
        if p.name in gt:
            msg += f", {hits}/{len(gt[p.name])} ground-truth blocks matched (IoU>=0.5)"
        print(msg, [round(float(s), 2) for s in det.confidence])
        cv2.imwrite(str(args.out / (p.stem + "_det.jpg")), img)


if __name__ == "__main__":
    main()
