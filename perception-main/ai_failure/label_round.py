#!/usr/bin/env python3
"""
Capture a frame and save one labelled crop per partition cell, reusing a cell layout
from capture_cells.py (valid as long as the box has not moved).

Labels (folder names):
    0 empty      no block in the cell
    1 seated     block fully inside the cell
    2 straddle   block resting on / across a divider or the rim (failed insertion)
    3 collapsed  divider pushed over / partition deformed

Usage (cells not listed are labelled 0 = empty):
    python label_round.py --cells data/frame_000_cells.json --labels "1:1,5:1,3:2"
    python label_round.py --cells data/frame_000_cells.json --labels "" --image data/frame_000.jpg
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from camera import grab_color

HERE = Path(__file__).resolve().parent
NAMES = {0: "empty", 1: "seated", 2: "straddle", 3: "collapsed"}
CROP = 224
MARGIN = 0.25        # include this fraction of the cell size around it, so dividers/rim are visible


def crop_cell(img, quad):
    """Perspective-crop the cell quad (tl, tr, br, bl) grown by MARGIN into a CROP x CROP image."""
    q = np.asarray(quad, np.float32)
    c = q.mean(0)
    q = c + (q - c) * (1 + 2 * MARGIN)
    dst = np.float32([[0, 0], [CROP - 1, 0], [CROP - 1, CROP - 1], [0, CROP - 1]])
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, M, (CROP, CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def parse_labels(s):
    out = {}
    for tok in filter(None, (t.strip() for t in s.split(","))):
        cell, lab = tok.split(":")
        out[int(cell)] = int(lab)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True, type=Path)
    ap.add_argument("--labels", default="", help='e.g. "1:1,5:1,3:2" (cell:label); others = 0')
    ap.add_argument("--image", type=Path, default=None, help="label an existing frame instead of capturing")
    ap.add_argument("--out", default=HERE / "dataset", type=Path)
    args = ap.parse_args()

    layout = json.loads(args.cells.read_text())
    labels = parse_labels(args.labels)
    frames = args.cells.parent
    if args.image:
        img, stem = cv2.imread(str(args.image)), args.image.stem
    else:
        img = grab_color()
        stem = f"shot_{len(list(frames.glob('shot_*.jpg'))):03d}"
        cv2.imwrite(str(frames / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    tiles, log = [], {}
    for c in layout["cells"]:
        lab = labels.get(c["id"], 0)
        crop = crop_cell(img, c["quad"])
        d = args.out / NAMES[lab]
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{stem}_cell{c['id']}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        log[c["id"]] = NAMES[lab]
        t = crop.copy()
        cv2.putText(t, f"{c['id']}:{NAMES[lab]}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        tiles.append(t)
    rows, cols = layout["rows"], layout["cols"]
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
    cv2.imwrite(str(frames / f"{stem}_crops.png"), grid)
    counts = {n: sum(1 for v in log.values() if v == n) for n in NAMES.values()}
    print(f"{stem}: {counts}  -> {frames / (stem + '_crops.png')}")


if __name__ == "__main__":
    main()
