import argparse
import json
from pathlib import Path

import numpy as np
import supervision as sv
from PIL import Image

from rfdetr import RFDETRMedium

PERCEPTION_ROOT = Path(__file__).resolve().parents[1]
CAPTURES_DIR = PERCEPTION_ROOT / "data" / "captures"
DETECTIONS_DIR = PERCEPTION_ROOT / "data" / "detections"

PRETRAINED_WEIGHTS = PERCEPTION_ROOT / "models" / "rf-detr-medium.pth"
FINETUNED_WEIGHTS = Path(__file__).resolve().parent / "output" / "checkpoint_best_total.pth"

THRESHOLD = 0.5
RESOLUTION = 576

COLOR = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00",
])


def build_annotators(image_size):
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=image_size)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=image_size)
    bbox = sv.BoxAnnotator(color=COLOR, thickness=thickness)
    label = sv.LabelAnnotator(
        color=COLOR,
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        smart_position=True,
    )
    return bbox, label


def detections_to_dict(detections, class_names):
    return {
        "xyxy": detections.xyxy.tolist(),
        "confidence": detections.confidence.tolist(),
        "class_id": detections.class_id.tolist(),
        "labels": [
            class_names[cid] if cid < len(class_names) else f"class_{cid}"
            for cid in detections.class_id
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Run RF-DETR inference on captured images")
    parser.add_argument(
        "--pretrained", action="store_true",
        help="Use the base pretrained COCO model instead of the fine-tuned checkpoint",
    )
    args = parser.parse_args()

    weights = PRETRAINED_WEIGHTS if args.pretrained else FINETUNED_WEIGHTS
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    rgb_paths = sorted(CAPTURES_DIR.glob("*_rgb.png"))
    if not rgb_paths:
        print("No RGB images found in", CAPTURES_DIR)
        return

    print(f"Loading RFDETRMedium (resolution={RESOLUTION}, weights={weights.name})…")
    model = RFDETRMedium(pretrain_weights=str(weights), resolution=RESOLUTION)
    model.optimize_for_inference()

    if args.pretrained:
        from rfdetr.util.coco_classes import COCO_CLASSES
        class_names = COCO_CLASSES
    else:
        class_names = ["box"]

    for img_path in rgb_paths:
        stem = img_path.stem.replace("_rgb", "")
        print(f"  {img_path.name}", end=" → ")

        image = Image.open(img_path).convert("RGB")
        detections = model.predict(image, threshold=THRESHOLD)

        result = detections_to_dict(detections, class_names)
        json_path = DETECTIONS_DIR / f"{stem}.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

        bbox_ann, label_ann = build_annotators(image.size)
        labels = [
            f"{class_names[cid] if cid < len(class_names) else f'class_{cid}'} {conf:.2f}"
            for cid, conf in zip(detections.class_id, detections.confidence)
        ]
        annotated = image.copy()
        annotated = bbox_ann.annotate(annotated, detections)
        annotated = label_ann.annotate(annotated, detections, labels)

        vis_path = DETECTIONS_DIR / f"{stem}_vis.png"
        annotated.save(vis_path)
        print(f"{len(detections)} detections  ({json_path.name}, {vis_path.name})")

    print(f"\nDone. Results saved to {DETECTIONS_DIR}/")


if __name__ == "__main__":
    main()
