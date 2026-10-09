"""Qwen3-VL cross-check of the partition scan: its own boxes for the open box and the partition, compared with
what the depth / VGGT pipeline measured. Run as its own step: the model is loaded, asked once and unloaded
again, so it never shares the GPU with VGGT / Grounding DINO.

  boxes = qwen_boxes(color)            -> {"box": [x1, y1, x2, y2] px, "partition": [...]}  (missing keys: not seen)
  ok, report = compare(boxes, sc)      -> IoU of each with the projected measurement
"""
import gc
import json
import re

import cv2
import numpy as np

MODEL = "Qwen/Qwen3-VL-8B-Instruct"
MIN_IOU = 0.5
PROMPT = ("Top-down photo of a table. Give the bounding box of (1) the open cardboard box and (2) the cardboard "
          "partition (a flat board with divider plates standing on it). Reply only with JSON: "
          '[{"label": "box" or "partition", "bbox_2d": [x1, y1, x2, y2]}].')


def qwen_boxes(color):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(MODEL)
    try:
        img = Image.fromarray(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=200, do_sample=False)
        txt = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    finally:                                           # free the GPU for VGGT / Grounding DINO
        del model
        gc.collect()
        torch.cuda.empty_cache()
    H, W = color.shape[:2]
    res = {"raw": txt}
    try:
        for o in json.loads(re.search(r"\[.*\]", txt, re.S).group(0)):
            lab = str(o.get("label", "")).lower()
            b = o.get("bbox_2d")
            if not b or len(b) != 4:
                continue
            px = [b[0] * W / 1000, b[1] * H / 1000, b[2] * W / 1000, b[3] * H / 1000]   # Qwen3-VL: 0..1000
            key = "partition" if "partition" in lab or "divider" in lab else "box" if "box" in lab else None
            if key and key not in res:
                res[key] = px
    except (AttributeError, ValueError, TypeError):
        pass
    return res


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def measured_px(sc):
    """Pixel boxes of the measured box (rim .. floor) and partition (board .. top) outlines."""
    import partition_scene as ps
    out = {}
    for key, r, zs in (("box", sc["box"], (sc["box"]["rim_z"], sc["box"]["floor_z"])),
                       ("partition", sc["part"], (sc["part"]["board_z"], sc["part"]["top_z"]))):
        C = r["corners"]
        q = ps.project(np.vstack([np.c_[C, np.full(4, z)] for z in zs]), sc["K"])
        out[key] = [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]
    return out


def compare(qb, sc, min_iou=MIN_IOU):
    meas = measured_px(sc)
    rep, ok = {}, True
    for key in ("box", "partition"):
        if key not in qb:
            rep[key] = "Qwen did not find it"
            ok = False
            continue
        v = iou(qb[key], meas[key])
        rep[key] = round(v, 2)
        ok &= v >= min_iou
    return ok, rep, meas


def draw(img, qb):
    for key, col in (("box", (255, 255, 255)), ("partition", (255, 255, 255))):
        if key in qb:
            x1, y1, x2, y2 = (int(v) for v in qb[key])
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, f"Qwen {key}", (x1 + 4, y2 - 8), 0, 0.6, col, 2)
    return img
