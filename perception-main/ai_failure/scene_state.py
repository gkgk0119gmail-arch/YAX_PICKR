#!/usr/bin/env python3
"""
Box / partition scene state from one top-camera frame (Grounding DINO, no training).

  partition = best "grid of cardboard dividers." hit
  box       = a "cardboard box." hit that contains the partition and is not much larger
              (a table-sized hit around everything is ignored)
  state     = partition_in_box | partition_only | empty_box | nothing

Usage (ai_bubble/.venv):
    HF_HUB_OFFLINE=1 ../ai_bubble/.venv/bin/python scene_state.py [--image img.jpg]
"""
import argparse
import json
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from camera import grab_color

HERE = Path(__file__).resolve().parent
MID = "IDEA-Research/grounding-dino-base"
P_BOX, P_PART = "cardboard box.", "grid of cardboard dividers."
TABLE_FRAC = 0.50     # a "box" covering more than this of the image (and not hugging a partition) is the table
HUG = 1.35            # box area / partition area for "partition inside this box"
KOR = {"partition_in_box": "파티션이 상자 안에 있음", "partition_only": "파티션만 있음 (상자 밖)",
       "empty_box": "빈 상자 (파티션 없음)", "nothing": "상자·파티션 없음"}


class SceneState:
    def __init__(self):
        self.proc = AutoProcessor.from_pretrained(MID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(MID).to("cuda").eval()

    def detect(self, img, prompt, thr=0.3):
        W, H = img.size
        inp = self.proc(images=img, text=prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            r = self.proc.post_process_grounded_object_detection(self.model(**inp), inp.input_ids, threshold=thr,
                                                                 text_threshold=0.25, target_sizes=[(H, W)])[0]
        return [(float(s), [float(v) for v in b]) for b, s in zip(r["boxes"].tolist(), r["scores"].tolist())]

    def __call__(self, img_bgr):
        img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        W, H = img.size
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
        contains = lambda o, i, m=10: o[0] - m <= i[0] and o[1] - m <= i[1] and o[2] + m >= i[2] and o[3] + m >= i[3]

        parts = self.detect(img, P_PART)
        boxes = self.detect(img, P_BOX)
        part = max(parts, default=None)
        box = None
        if part:
            hugging = [b for b in boxes if contains(b[1], part[1]) and area(b[1]) <= HUG * area(part[1])]
            box = max(hugging, default=None)
            state = "partition_in_box" if box else "partition_only"
        else:
            real = [b for b in boxes if area(b[1]) < TABLE_FRAC * W * H]
            box = max(real, default=None)
            state = "empty_box" if box else "nothing"
        others = [b for b in boxes if b is not box and area(b[1]) < TABLE_FRAC * W * H
                  and not (part and contains(b[1], part[1]))]
        return {"state": state, "partition": part, "box": box, "other_boxes": others}


def draw(img, res):
    out = img.copy()
    if res["box"]:
        s, b = res["box"]
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 4)
        cv2.putText(out, f"box {s:.2f}", (int(b[0]) + 6, int(b[1]) + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    if res["partition"]:
        s, b = res["partition"]
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0), 3)
        cv2.putText(out, f"partition {s:.2f}", (int(b[0]) + 6, int(b[3]) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 200, 0), 2)
    for s, b in res["other_boxes"]:
        cv2.rectangle(out, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 140, 255), 2)
        cv2.putText(out, f"other box {s:.2f}", (int(b[0]) + 4, int(b[1]) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 140, 255), 2)
    cv2.rectangle(out, (0, 0), (720, 50), (0, 0, 0), -1)
    cv2.putText(out, f"STATE: {res['state']}", (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path)
    ap.add_argument("--out", default=HERE / "data" / "scene", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.image:
        img, stem = cv2.imread(str(args.image)), args.image.stem
    else:
        img = grab_color()
        stem = f"scene_{len(list(args.out.glob('scene_*.jpg'))):03d}"
        cv2.imwrite(str(args.out / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    res = SceneState()(img)
    cv2.imwrite(str(args.out / f"{stem}_state.png"), draw(img, res))
    (args.out / f"{stem}_state.json").write_text(json.dumps(res, indent=1))
    print(f"{stem}: {res['state']} ({KOR[res['state']]})  partition={res['partition']}  box={res['box']}  "
          f"other boxes={len(res['other_boxes'])}")
    print(args.out / f"{stem}_state.png")


if __name__ == "__main__":
    main()
