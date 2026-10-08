#!/usr/bin/env python3
"""
Build a COCO-format dataset from annotated captures for RF-DETR fine-tuning.

Reads annotations.json (from annotate.py), copies images into train/valid/test
splits, and generates _annotations.coco.json for each split.

Usage:
    python build_dataset.py [--annotations FILE] [--captures-dir DIR]
                            [--output-dir DIR] [--train-ratio 0.7]
                            [--valid-ratio 0.2]
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path


def build_coco_json(
    image_entries: list[dict],
    box_entries: list[dict],
    categories: list[dict],
) -> dict:
    return {
        "info": {"description": "Box detection dataset", "version": "1.0"},
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "categories": categories,
        "images": image_entries,
        "annotations": box_entries,
    }


def main():
    parser = argparse.ArgumentParser(description="Build COCO dataset from annotations")
    parser.add_argument("--annotations",
                        default=os.path.join(os.path.dirname(__file__),
                                             "annotations.json"))
    parser.add_argument("--captures-dir",
                        default=os.path.join(os.path.dirname(__file__),
                                             "..", "data", "captures"))
    parser.add_argument("--output-dir",
                        default=os.path.join(os.path.dirname(__file__), "dataset"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    annotations_path = Path(args.annotations)
    captures_dir = Path(args.captures_dir)
    output_dir = Path(args.output_dir)

    if not annotations_path.exists():
        raise FileNotFoundError(
            f"Annotations not found: {annotations_path}\n"
            f"Run annotate.py first to create bounding box annotations."
        )

    with open(annotations_path) as f:
        annotations: dict[str, list[list[int]]] = json.load(f)

    annotated = {k: v for k, v in annotations.items() if v}
    if not annotated:
        raise ValueError("No annotated images found (all box lists are empty).")

    print(f"Found {len(annotated)} annotated images "
          f"(out of {len(annotations)} total).")

    image_names = sorted(annotated.keys())
    random.seed(args.seed)
    random.shuffle(image_names)

    n = len(image_names)
    n_train = max(1, round(n * args.train_ratio))
    n_valid = max(1, round(n * args.valid_ratio))
    n_test = max(0, n - n_train - n_valid)

    if n_test == 0 and n > 2:
        n_train -= 1
        n_test = 1

    splits = {
        "train": image_names[:n_train],
        "valid": image_names[n_train:n_train + n_valid],
        "test": image_names[n_train + n_valid:],
    }

    categories = [{"id": 0, "name": "box", "supercategory": "none"}]

    import cv2

    for split_name, names in splits.items():
        if not names:
            print(f"  Skipping empty split: {split_name}")
            continue

        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        image_entries = []
        box_entries = []
        ann_id = 0

        for img_id, name in enumerate(names):
            src = captures_dir / name
            if not src.exists():
                print(f"  WARNING: {src} not found, skipping")
                continue

            shutil.copy2(src, split_dir / name)

            img = cv2.imread(str(src))
            h, w = img.shape[:2]

            image_entries.append({
                "id": img_id,
                "file_name": name,
                "width": w,
                "height": h,
            })

            for box_xyxy in annotated[name]:
                x1, y1, x2, y2 = box_xyxy
                bw = x2 - x1
                bh = y2 - y1
                box_entries.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 0,
                    "bbox": [x1, y1, bw, bh],
                    "area": bw * bh,
                    "segmentation": [],
                    "iscrowd": 0,
                })
                ann_id += 1

        coco = build_coco_json(image_entries, box_entries, categories)
        coco_path = split_dir / "_annotations.coco.json"
        with open(coco_path, "w") as f:
            json.dump(coco, f, indent=2)

        print(f"  {split_name}: {len(image_entries)} images, "
              f"{len(box_entries)} annotations -> {coco_path}")

    print(f"\nDataset written to {output_dir}")
    print(f"  Split sizes: train={len(splits['train'])}, "
          f"valid={len(splits['valid'])}, test={len(splits['test'])}")


if __name__ == "__main__":
    main()
