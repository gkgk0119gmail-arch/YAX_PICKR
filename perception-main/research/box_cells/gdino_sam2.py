"""Grounding DINO (text -> boxes) + SAM2 (box prompt -> mask, only inside each box)."""
import sys, time
import cv2, numpy as np, torch
from PIL import Image
from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,
                          Sam2Model, Sam2Processor)
from rect_fit import ransac_rect



img_path = sys.argv[1] if len(sys.argv) > 1 else "cam_now.jpg"
prompt = sys.argv[2] if len(sys.argv) > 2 else "bubble wrap."
out_path = sys.argv[3] if len(sys.argv) > 3 else "seg_result.png"
dev = "cuda"

img = Image.open(img_path).convert("RGB")
W, H = img.size

gd_id = "IDEA-Research/grounding-dino-base"
gd_proc = AutoProcessor.from_pretrained(gd_id)
gd = AutoModelForZeroShotObjectDetection.from_pretrained(gd_id).to(dev).eval()
t0 = time.time()
inp = gd_proc(images=img, text=prompt, return_tensors="pt").to(dev)
with torch.no_grad():
    r = gd_proc.post_process_grounded_object_detection(
        gd(**inp), inp.input_ids, threshold=0.3, text_threshold=0.25, target_sizes=[(H, W)])[0]
boxes = [(float(s), [float(v) for v in b]) for b, s in zip(r["boxes"], r["scores"])
         if (b[2] - b[0]) * (b[3] - b[1]) < 0.25 * W * H]          # drop whole-scene false positives
t_det = time.time() - t0
print(f"Grounding DINO '{prompt}': {len(boxes)} boxes ({t_det:.2f}s)")

sam_id = "facebook/sam2.1-hiera-large"
sam_proc = Sam2Processor.from_pretrained(sam_id)
sam = Sam2Model.from_pretrained(sam_id).to(dev).eval()
t0 = time.time()
masks = []
if boxes:
    sin = sam_proc(images=img, input_boxes=[[b for _, b in boxes]], return_tensors="pt").to(dev)
    with torch.no_grad():
        so = sam(**sin, multimask_output=False)
    masks = sam_proc.post_process_masks(so.pred_masks.cpu(), sin["original_sizes"])[0][:, 0].numpy()
t_seg = time.time() - t0
print(f"SAM2: {len(masks)} masks ({t_seg:.2f}s)")

vis = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
dim = (vis * 0.45).astype(np.uint8)
out = dim.copy()
colors = [(0, 200, 255), (255, 120, 0), (0, 220, 0), (255, 0, 200), (0, 0, 255)]
for i, ((s, (x1, y1, x2, y2)), m) in enumerate(zip(boxes, masks)):
    box = np.zeros(m.shape, bool)
    box[int(y1):int(y2) + 1, int(x1):int(x2) + 1] = True
    m = m.astype(bool) & box                                       # keep only what is inside the box
    out[m] = vis[m]                                                # object in full colour, rest dimmed
    c = colors[i % len(colors)]
    tint = out[m].astype(np.float32) * 0.6 + np.array(c) * 0.4
    out[m] = tint.astype(np.uint8)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, c, 2)
    cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
    info = f"#{i+1} {s:.2f}"
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)[:, 0, :]
        center, (rw, rh), ang, ratio, corners = ransac_rect(cnt)
        short, long_ = sorted((rw, rh))
        # gripper closes across the short side: jaw axis = direction of the short side
        jaw_deg = ang if rw <= rh else ang + 90
        cv2.polylines(out, [corners.round().astype(np.int32)], True, (255, 255, 255), 2)
        cx, cy = center
        a = np.radians(jaw_deg)
        hw = short / 2 + 6
        p1 = (int(cx - hw * np.cos(a)), int(cy - hw * np.sin(a)))
        p2 = (int(cx + hw * np.cos(a)), int(cy + hw * np.sin(a)))
        cv2.line(out, p1, p2, (0, 0, 255), 2)             # red: gripper closing direction
        cv2.circle(out, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        info += f" {short:.0f}x{long_:.0f}px yaw {jaw_deg % 180:.0f}deg fit {ratio*100:.0f}%"
    cv2.putText(out, info, (int(x1) - 30, int(y1) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
    print(info)
cv2.putText(out, f"GDINO '{prompt}' + SAM2.1 + RANSAC rect  (red = gripper closing axis)", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.imwrite(out_path, out)

# masks clipped to their boxes, for inpainting / copy-paste synthesis
clipped = []
for (_, (x1, y1, x2, y2)), m in zip(boxes, masks):
    box = np.zeros(m.shape, bool)
    box[int(y1):int(y2) + 1, int(x1):int(x2) + 1] = True
    clipped.append(m.astype(bool) & box)
np.savez_compressed(out_path.rsplit(".", 1)[0] + "_masks.npz",
                    masks=np.array(clipped), boxes=np.array([b for _, b in boxes]),
                    scores=np.array([s for s, _ in boxes]), image=img_path)
