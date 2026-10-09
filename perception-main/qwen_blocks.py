"""Block count and rough boxes from Qwen3-VL (once per scan), as the prior for Grounding DINO.

Qwen looks at the whole top-camera picture and lists every bubble-wrapped block on the table (not the wooden
basket, not other objects) with a rough box and whether it stands or lies. Grounding DINO then gives the clean
box of each one (block_grasp.BlockFinder with finder.prior set). The 8B model needs ~17 GB of GPU memory: it is
loaded only for this call and freed afterwards (VGGT is parked on the CPU meanwhile).

  boxes = qwen_blocks.find(color_bgr)      # [dict(box=[x1, y1, x2, y2] px, pose="standing"|"lying")]
"""
import json
import re
import time

import cv2
import numpy as np
import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
PROMPT = (
    "This is a top-down photo of a table. Find EVERY cardboard block wrapped in clear bubble wrap that stands or "
    "lies on the table, including blocks that touch each other and blocks inside the wooden basket. Do NOT include "
    "the wooden basket or its dividers, the robot arm, the checkerboard, or any other object. "
    "Answer only with JSON: a list of {\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"block\", "
    "\"pose\": \"standing\" or \"lying\"}, one entry per block.")


def _parse(text, W, H):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    boxes = []
    for it in items:
        b = it.get("bbox_2d") if isinstance(it, dict) else None
        if not b or len(b) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in b)
        # Qwen3-VL answers in coordinates normalised to 0..1000
        x1, x2 = x1 / 1000 * W, x2 / 1000 * W
        y1, y2 = y1 / 1000 * H, y2 / 1000 * H
        if x2 - x1 > 8 and y2 - y1 > 8:
            boxes.append(dict(box=[x1, y1, x2, y2], pose=str(it.get("pose", "")).lower()))
    return boxes


def find(color_bgr, finder=None, log=print):
    """Rough block boxes (colour px). finder: its VGGT is moved to the CPU while Qwen is on the GPU."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    t0 = time.time()
    parked = None
    if finder is not None and getattr(finder, "vggt", None) is not None:
        parked = next(finder.vggt.parameters()).device
        finder.vggt.to("cpu")
        torch.cuda.empty_cache()
    model = proc = None
    try:
        proc = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
        t1 = time.time()
        img = Image.fromarray(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB))
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=1024, do_sample=False)
        text = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        boxes = _parse(text, img.width, img.height)
        log(f"[qwen] {len(boxes)} block(s) (load {t1 - t0:.0f} s, answer {time.time() - t1:.1f} s)")
        return boxes, text
    finally:
        del model, proc
        torch.cuda.empty_cache()
        if parked is not None:
            finder.vggt.to(parked)
