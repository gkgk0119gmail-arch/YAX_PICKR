#!/usr/bin/env python3
"""
Capture a frame (or read one) and classify every partition cell: empty / seated / straddle.

Usage (ai_bubble/.venv):
    HF_HUB_OFFLINE=1 ../ai_bubble/.venv/bin/python predict_cells.py --cells data/frame_000_cells.json
        [--image some.jpg] [--out data/tests]
"""
import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image

from camera import grab_color
from label_round import crop_cell
from train_classifier import Dinov3Features

HERE = Path(__file__).resolve().parent
COLORS = {"empty": (180, 180, 180), "seated": (0, 200, 0), "straddle": (0, 0, 255), "collapsed": (255, 0, 255)}
KOR = {"empty": "빈 칸", "seated": "정상", "straddle": "걸침", "collapsed": "무너짐", "uncertain": "불확실"}
COLORS["uncertain"] = (0, 200, 255)
MIN_CONF = 0.7   # below this the cell is reported "uncertain" -> re-check with depth / another shot
MIN_VISIBLE = 0.9  # a cell (without the crop margin) partly outside the frame is "out_of_view"
KOR["out_of_view"] = "안 보임"
COLORS["out_of_view"] = (255, 0, 255)


def visible_fraction(quad, shape):
    """Fraction of the cell polygon itself that lies inside the image (the crop margin may
    run off the frame; only the cell matters for whether a block in it can be seen)."""
    h, w = shape[:2]
    q = np.asarray(quad, np.float32)
    full = cv2.contourArea(q)
    clipped = cv2.intersectConvexConvex(q, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))[0]
    return clipped / full if full > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, type=Path)
    ap.add_argument("--image", type=Path, default=None)
    ap.add_argument("--model", default=HERE / "model" / "cell_classifier.joblib", type=Path)
    ap.add_argument("--out", default=HERE / "data" / "tests", type=Path)
    args = ap.parse_args()

    layout = json.loads(args.cells.read_text())
    bundle = joblib.load(args.model)
    clf, classes = bundle["clf"], bundle["classes"]
    feat = Dinov3Features()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.image:
        img, stem = cv2.imread(str(args.image)), args.image.stem
    else:
        img = grab_color()
        stem = f"test_{len(list(args.out.glob('test_*.jpg'))):03d}"
        cv2.imwrite(str(args.out / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    crops = [crop_cell(img, c["quad"]) for c in layout["cells"]]
    X = feat([Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops])
    proba = clf.predict_proba(X)
    labels = [classes[k] for k in clf.classes_]

    out = img.copy()
    rows = []
    for c, p in zip(layout["cells"], proba):
        k = int(np.argmax(p))
        name, conf = labels[k], float(p[k])
        if visible_fraction(c["quad"], img.shape) < MIN_VISIBLE:
            name = "out_of_view"
        elif conf < MIN_CONF:
            name = "uncertain"
        rows.append((c["id"], name, conf))
        q = np.round(c["quad"]).astype(np.int32)
        cv2.polylines(out, [q], True, COLORS[name], 3)
        cx, cy = q.mean(0).astype(int)
        cv2.putText(out, f"{c['id']} {name}", (cx - 45, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[name], 2)
        cv2.putText(out, f"{conf:.2f}", (cx - 20, cy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS[name], 2)
    cv2.imwrite(str(args.out / f"{stem}_pred.png"), out)

    r, cc = layout["rows"], layout["cols"]
    print(f"{stem}:")
    for i in range(r):
        print("   " + "  ".join(f"{cid}:{KOR[n]}({cf:.2f})" for cid, n, cf in rows[i * cc:(i + 1) * cc]))
    (args.out / f"{stem}_pred.json").write_text(json.dumps(
        [{"cell": cid, "state": n, "confidence": cf} for cid, n, cf in rows], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
