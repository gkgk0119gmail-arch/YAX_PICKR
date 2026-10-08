#!/usr/bin/env python3
"""
Build a COCO dataset for RF-DETR bubble-wrap block detection from copy-paste synthesis.

Each source in source/ is a real top-camera photo plus its SAM2 block masks
(<name>.jpg + <name>_masks.npz, made by research/box_cells/gdino_sam2.py).
LaMa removes the blocks to get a clean background, then synth_blocks.py pastes
blocks back at random position / rotation / scale / zoom with exact labels.

Splits:
  train, valid  synthetic
  test          the real source photos with their mask boxes. These are also where
                the cutouts came from, so test scores are optimistic until real photos
                of other block arrangements are added.

Usage:
    python make_dataset.py [--train 600] [--valid 100] [--out dataset]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "research" / "box_cells"))
import synth_blocks as sb  # noqa: E402

CATEGORY = "bubble_wrap"


def coco(images, annots):
    return {
        "info": {"description": "synthetic bubble-wrap blocks"},
        "images": images,
        "annotations": annots,
        "categories": [{"id": 0, "name": CATEGORY, "supercategory": "none"}],
    }


def write_split(split_dir, samples):
    """samples: list of (file_name, image_bgr, [[x1,y1,x2,y2], ...])."""
    split_dir.mkdir(parents=True, exist_ok=True)
    images, annots = [], []
    for img_id, (name, img, boxes) in enumerate(samples):
        cv2.imwrite(str(split_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        h, w = img.shape[:2]
        images.append({"id": img_id, "file_name": name, "width": w, "height": h})
        for x1, y1, x2, y2 in boxes:
            bw, bh = x2 - x1, y2 - y1
            annots.append({"id": len(annots), "image_id": img_id, "category_id": 0,
                           "bbox": [x1, y1, bw, bh], "area": bw * bh, "iscrowd": 0})
    (split_dir / "_annotations.coco.json").write_text(json.dumps(coco(images, annots)))
    return len(images), len(annots)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--source", default=HERE / "source", type=Path)
    ap.add_argument("--out", default=HERE / "dataset", type=Path)
    ap.add_argument("--train", type=int, default=600)
    ap.add_argument("--valid", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sources = []
    for npz in sorted(args.source.glob("*_masks.npz")):
        img_path = npz.with_name(npz.name.replace("_masks.npz", ".jpg"))
        img, masks = cv2.imread(str(img_path)), np.load(npz)["masks"]
        plate, _ = sb.make_plate(img, masks, args.out / "_lama" / img_path.stem)
        sources.append((img_path, img, masks, plate, sb.cutouts(img, masks)))
    if not sources:
        sys.exit(f"no *_masks.npz in {args.source}")
    # cutouts from every source can land on every background
    all_cuts = [c for s in sources for c in s[4]]
    print(f"{len(sources)} source photo(s), {len(all_cuts)} block cutouts")

    if args.out.exists():
        for split in ("train", "valid", "test"):
            shutil.rmtree(args.out / split, ignore_errors=True)
    rng = np.random.default_rng(args.seed)

    def synth(n, prefix):
        out = []
        for i in range(n):
            plate = sources[rng.integers(len(sources))][3]
            img, boxes = sb.synthesize(plate, all_cuts, rng)
            img, boxes = sb.zoom(img, boxes, float(rng.uniform(*sb.ZOOM)), rng)
            out.append((f"{prefix}_{i:04d}.jpg", img, boxes))
        return out

    for split, n in (("train", args.train), ("valid", args.valid)):
        k, a = write_split(args.out / split, synth(n, split))
        print(f"{split:5s}: {k} images, {a} boxes (synthetic)")

    real = []
    for img_path, img, masks, _, _ in sources:
        boxes = []
        for m in masks:
            ys, xs = np.nonzero(m)
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
        real.append((img_path.name, img, boxes))
    k, a = write_split(args.out / "test", real)
    print(f"test : {k} images, {a} boxes (real photos)")


if __name__ == "__main__":
    main()
