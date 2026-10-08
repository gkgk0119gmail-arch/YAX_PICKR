#!/usr/bin/env python3
"""
Live RF-DETR object detection on RealSense camera stream.
- Draw ROI with mouse (left-drag). Right-click to clear ROI.
- Tune conf / max-det / averaging live via trackbars.
- Temporal averaging to smooth detections (toggle with 't').
"""

import argparse
import sys
import os
import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import pyrealsense2 as rs
import supervision as sv
from rfdetr import RFDETRMedium
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PERCEPTION_ROOT = Path(__file__).resolve().parent
PRETRAINED_WEIGHTS = PERCEPTION_ROOT / "models" / "rf-detr-medium.pth"
FINETUNED_WEIGHTS = PERCEPTION_ROOT / "ai" / "output" / "checkpoint_best_total.pth"

SAM2_CKPT = PERCEPTION_ROOT / "sam2" / "checkpoints" / "sam2.1_hiera_base_plus.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM_MASK_ALPHA = 0.5

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
WINDOW_NAME = "RF-DETR Live Detection"

COLOR_PALETTE = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])

# ---------------------------------------------------------------------------
# Temporal detection averager
# ---------------------------------------------------------------------------
class DetectionAverager:
    """
    Match detections across frames by IoU and smooth boxes over a sliding
    window.  A track qualifies if its total hits within the window >=
    min_hit_ratio * window.
    """

    def __init__(self, window=5, iou_thresh=0.3, min_hit_ratio=0.4,
                 max_miss=None):
        self.window = window
        self.iou_thresh = iou_thresh
        self.min_hit_ratio = min_hit_ratio
        self.max_miss = max_miss if max_miss is not None else window
        self.tracks = []
        self._next_id = 0

    @staticmethod
    def _iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ab = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (aa + ab - inter + 1e-9)

    def update(self, dets):

        # --- match dets to existing tracks via Hungarian on IoU cost ---
        matched_d = set()
        if self.tracks and dets:
            cost = np.zeros((len(self.tracks), len(dets)))
            for i, trk in enumerate(self.tracks):
                for j, d in enumerate(dets):
                    cost[i, j] = 1.0 - self._iou(trk["avg_box"], d["xyxy"])
            row_idx, col_idx = linear_sum_assignment(cost)
            matched_t = set()
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] < (1.0 - self.iou_thresh):
                    trk = self.tracks[r]
                    trk["history"].append(dets[c]["xyxy"])
                    trk["conf_history"].append(dets[c]["conf"])
                    trk["hit_bitmap"].append(True)
                    trk["miss_streak"] = 0
                    trk["class_id"] = dets[c].get("class_id", 0)
                    matched_t.add(r)
                    matched_d.add(c)

            # mark misses for unmatched tracks
            for i, trk in enumerate(self.tracks):
                if i not in matched_t:
                    trk["hit_bitmap"].append(False)
                    trk["miss_streak"] += 1
        else:
            for trk in self.tracks:
                trk["hit_bitmap"].append(False)
                trk["miss_streak"] += 1

        # --- create new tracks for unmatched detections ---
        for j in range(len(dets)):
            if j in matched_d:
                continue
            d = dets[j]
            self.tracks.append({
                "id": self._next_id,
                "history": deque([d["xyxy"]], maxlen=self.window),
                "conf_history": deque([d["conf"]], maxlen=self.window),
                "hit_bitmap": deque([True], maxlen=self.window),
                "miss_streak": 0,
                "avg_box": d["xyxy"],
                "class_id": d.get("class_id", 0),
            })
            self._next_id += 1

        # --- update averages & prune dead tracks ---
        keep = []
        for trk in self.tracks:
            if trk["miss_streak"] > self.max_miss:
                continue
            if trk["history"]:
                boxes = np.array(list(trk["history"]))
                trk["avg_box"] = boxes.mean(axis=0).tolist()
            keep.append(trk)
        self.tracks = keep

        # --- build candidates that pass hit-ratio threshold ---
        min_hits_needed = max(1, int(self.min_hit_ratio * self.window))
        candidates = []
        for trk in self.tracks:
            hits = sum(trk["hit_bitmap"])
            if hits >= min_hits_needed:
                avg_conf = float(np.mean(list(trk["conf_history"])))
                candidates.append({
                    "xyxy": trk["avg_box"],
                    "conf": avg_conf,
                    "class_id": trk.get("class_id", 0),
                })

        return candidates

    def reset(self):
        self.tracks.clear()
        self._next_id = 0


# ---------------------------------------------------------------------------
# Mouse-drawn ROI state
# ---------------------------------------------------------------------------
class RoiState:
    def __init__(self):
        self.drawing = False
        self.pt1 = None
        self.pt2 = None
        self.roi = None
        self.drag_preview = None

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.pt1 = (x, y)
            self.pt2 = (x, y)
            self.drag_preview = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.pt2 = (x, y)
            x1 = min(self.pt1[0], x); y1 = min(self.pt1[1], y)
            x2 = max(self.pt1[0], x); y2 = max(self.pt1[1], y)
            self.drag_preview = (x1, y1, x2, y2)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.pt1 is not None:
                x1 = min(self.pt1[0], x); y1 = min(self.pt1[1], y)
                x2 = max(self.pt1[0], x); y2 = max(self.pt1[1], y)
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    self.roi = (x1, y1, x2, y2)
                    print(f"ROI set: ({x1},{y1})-({x2},{y2})  "
                          f"size {x2-x1}x{y2-y1}")
                else:
                    self.roi = None
            self.drag_preview = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.roi = None
            self.drag_preview = None
            print("ROI cleared — using full frame")


# ---------------------------------------------------------------------------
# RealSense init (with retry + hw reset from cluster.py)
# ---------------------------------------------------------------------------
def get_device():
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        raise RuntimeError("No RealSense devices found")
    return devs[0]


def init_realsense():
    print("Initializing RealSense camera...")
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT,
                      rs.format.bgr8, CAMERA_FPS)
    while True:
        try:
            pipe = rs.pipeline()
            profile = pipe.start(cfg)
            for sensor in profile.get_device().sensors:
                if sensor.get_info(rs.camera_info.name) == "RGB Camera":
                    sensor.set_option(rs.option.enable_auto_exposure, 1)
                    break
            print("Camera initialized successfully.")
            return pipe, profile
        except RuntimeError as e:
            msg = str(e)
            print(f"Initialization error: {msg}")
            recoverable = [
                "get_xu(ctrl=1)", "Device or resource busy",
                "failed to set power state", "Couldn't resolve requests",
            ]
            if any(err in msg for err in recoverable):
                print("Hardware reset...")
                try:
                    get_device().hardware_reset()
                    time.sleep(2)
                except Exception:
                    time.sleep(2)
            else:
                raise


# ---------------------------------------------------------------------------
# SAM2 helpers (lazy-loaded)
# ---------------------------------------------------------------------------
def _pick_autocast_dtype():
    import torch
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def load_sam2_predictor(ckpt: Path, model_cfg: str, device: str):
    # The cloned `sam2/` repo lives under PERCEPTION_ROOT, and Python puts the
    # script's directory at sys.path[0]. That makes `import sam2` resolve to
    # the repo root and shadow the installed package, which SAM2 explicitly
    # rejects. Temporarily strip those entries before importing.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    blocked = {os.path.abspath(p) for p in (script_dir, str(PERCEPTION_ROOT), "")}
    saved_path = list(sys.path)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p) not in blocked]
    saved_mods = {k: sys.modules.pop(k) for k in list(sys.modules)
                  if k == "sam2" or k.startswith("sam2.")}
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2(model_cfg, str(ckpt), device=device)
        predictor = SAM2ImagePredictor(model)
    finally:
        sys.path[:] = saved_path
        for k, v in saved_mods.items():
            sys.modules.setdefault(k, v)
    return predictor


def run_sam2_on_dets(predictor, img_bgr, dets, device, autocast_dtype):
    """Return list of full-frame uint8 binary masks (one per detection)."""
    import torch
    if not dets:
        return []

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes = np.array([d["xyxy"] for d in dets], dtype=np.float32)

    masks = []
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                predictor.set_image(img_rgb)
                mask_preds, _, _ = predictor.predict(
                    box=boxes, multimask_output=False,
                )
        else:
            predictor.set_image(img_rgb)
            mask_preds, _, _ = predictor.predict(
                box=boxes, multimask_output=False,
            )

    # When N==1, SAM2 returns shape (1, H, W); when N>1 it's (N, 1, H, W).
    arr = np.asarray(mask_preds)
    if arr.ndim == 4:
        arr = arr[:, 0]
    elif arr.ndim == 3 and len(boxes) == 1:
        arr = arr  # already (1, H, W)
    for i in range(arr.shape[0]):
        masks.append((arr[i] > 0).astype(np.uint8))
    return masks


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------
def draw_detections(frame, detections, class_names, masks=None, mask_alpha=SAM_MASK_ALPHA):
    if not detections:
        return frame

    xyxy = np.array([d["xyxy"] for d in detections], dtype=np.float32)
    confidence = np.array([d["conf"] for d in detections], dtype=np.float32)
    class_ids = np.array([d.get("class_id", 0) for d in detections], dtype=int)

    kwargs = dict(xyxy=xyxy, confidence=confidence, class_id=class_ids)
    if masks is not None and len(masks) == len(detections) and len(masks) > 0:
        kwargs["mask"] = np.stack(masks, axis=0).astype(bool)

    sv_dets = sv.Detections(**kwargs)
    labels = [
        f"{class_names[cid] if cid < len(class_names) else f'cls{cid}'} {conf:.2f}"
        for cid, conf in zip(sv_dets.class_id, sv_dets.confidence)
    ]

    h, w = frame.shape[:2]
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(w, h))
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=(w, h))

    bbox_annotator = sv.BoxAnnotator(color=COLOR_PALETTE, thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        color=COLOR_PALETTE,
        text_color=sv.Color.BLACK,
        text_scale=text_scale,
        smart_position=True,
    )

    if "mask" in kwargs:
        mask_annotator = sv.MaskAnnotator(color=COLOR_PALETTE, opacity=mask_alpha)
        frame = mask_annotator.annotate(frame, sv_dets)

    frame = bbox_annotator.annotate(frame, sv_dets)
    frame = label_annotator.annotate(frame, sv_dets, labels)
    return frame


def run_detection(model, img_bgr, conf, max_det):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    detections = model.predict(img_rgb, threshold=conf)
    if detections.xyxy is None or len(detections.xyxy) == 0:
        return []
    xyxy = detections.xyxy.astype(np.float32)
    confs = detections.confidence.astype(float)
    class_ids = detections.class_id.astype(int)
    dets = []
    for i in range(len(xyxy)):
        dets.append({
            "xyxy": xyxy[i].tolist(),
            "conf": float(confs[i]),
            "class_id": int(class_ids[i]),
        })
    dets.sort(key=lambda d: d["conf"], reverse=True)
    return dets[:max_det]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Live RF-DETR detection on RealSense")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use the base pretrained COCO model instead of the fine-tuned checkpoint")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to custom checkpoint (overrides --pretrained)")
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--sam", action="store_true",
                        help="Run SAM2 on detection bboxes and overlay masks (toggle at runtime with 's')")
    args = parser.parse_args()

    if args.weights:
        weights = Path(args.weights)
    elif args.pretrained:
        weights = PRETRAINED_WEIGHTS
    else:
        weights = FINETUNED_WEIGHTS

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    print(f"Loading RF-DETR Medium (resolution={args.resolution}, weights={weights.name}) ...")
    model = RFDETRMedium(pretrain_weights=str(weights), resolution=args.resolution)
    model.optimize_for_inference()

    if args.pretrained and not args.weights:
        from rfdetr.util.coco_classes import COCO_CLASSES
        class_names = COCO_CLASSES
    else:
        class_names = model.class_names
    print(f"Classes: {class_names}")
    print("Warming up ...")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    model.predict(dummy, threshold=0.5)
    print("Warmup done.")

    sam_predictor = None
    sam_device = "cpu"
    sam_autocast_dtype = None
    if args.sam:
        if not SAM2_CKPT.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {SAM2_CKPT}")
        import torch
        sam_device = "cuda" if torch.cuda.is_available() else "cpu"
        sam_autocast_dtype = _pick_autocast_dtype()
        print(f"Loading SAM2 ({SAM2_CKPT.name}) on {sam_device} ...")
        sam_predictor = load_sam2_predictor(SAM2_CKPT, SAM2_CFG, sam_device)
        print("SAM2 ready.")
    use_sam = args.sam

    pipe, _ = init_realsense()

    # --- window + trackbars ---
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CAMERA_WIDTH, CAMERA_HEIGHT)

    cv2.createTrackbar("Conf x100",      WINDOW_NAME, 50, 100, lambda v: None)
    cv2.createTrackbar("Max Det",        WINDOW_NAME, 30, 100, lambda v: None)
    cv2.createTrackbar("Avg Window",     WINDOW_NAME,  5,  30, lambda v: None)
    cv2.createTrackbar("Min Hit% x100",  WINDOW_NAME, 40, 100, lambda v: None)

    roi_state = RoiState()
    cv2.setMouseCallback(WINDOW_NAME, roi_state.mouse_cb)

    averager = DetectionAverager(window=5, iou_thresh=0.3, min_hit_ratio=0.4)
    use_avg = False

    fps_filter = 0.0
    prev_time = time.time()

    print("\n--- Controls ---")
    print("  Left-drag    : draw ROI")
    print("  Right-click  : clear ROI (full frame)")
    print("  Trackbars    : Conf / MaxDet / Avg Window / Min Hit%")
    print("  t            : toggle temporal averaging")
    print("  s            : toggle SAM2 mask overlay (requires --sam to load model)")
    print("  q            : quit\n")

    try:
        while True:
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())

            # --- read trackbar values ---
            conf = max(cv2.getTrackbarPos("Conf x100", WINDOW_NAME), 1) / 100.0
            max_det = max(cv2.getTrackbarPos("Max Det", WINDOW_NAME), 1)
            avg_win = max(cv2.getTrackbarPos("Avg Window", WINDOW_NAME), 1)
            min_hit_pct = max(cv2.getTrackbarPos("Min Hit% x100", WINDOW_NAME), 1) / 100.0

            averager.window = avg_win
            averager.min_hit_ratio = min_hit_pct
            averager.max_miss = avg_win

            # --- crop to ROI if set ---
            roi = roi_state.roi
            if roi is not None:
                rx1, ry1, rx2, ry2 = roi
                crop = img[ry1:ry2, rx1:rx2].copy()
                dets = run_detection(model, crop, conf, max_det)
                for d in dets:
                    bx1, by1, bx2, by2 = d["xyxy"]
                    d["xyxy"] = [bx1 + rx1, by1 + ry1, bx2 + rx1, by2 + ry1]
            else:
                dets = run_detection(model, img, conf, max_det)

            # --- temporal averaging ---
            if use_avg:
                dets = averager.update(dets)
            else:
                averager.reset()

            # --- SAM2 segmentation on detection bboxes ---
            masks = None
            if use_sam and sam_predictor is not None and dets:
                try:
                    masks = run_sam2_on_dets(
                        sam_predictor, img, dets, sam_device, sam_autocast_dtype,
                    )
                except Exception as e:
                    print(f"[SAM2] inference error: {e}")
                    masks = None

            # --- visualise ---
            vis = img.copy()

            if roi is not None:
                cv2.rectangle(vis, (roi[0], roi[1]), (roi[2], roi[3]),
                              (0, 255, 255), 2)

            if roi_state.drag_preview is not None:
                dp = roi_state.drag_preview
                overlay = vis.copy()
                cv2.rectangle(overlay, (dp[0], dp[1]), (dp[2], dp[3]),
                              (0, 255, 255), 2)
                cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

            vis = draw_detections(vis, dets, class_names, masks=masks)

            # --- FPS ---
            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps_filter = 0.9 * fps_filter + 0.1 * (1.0 / dt if dt > 0 else 0)

            roi_str = (f"ROI {roi[2]-roi[0]}x{roi[3]-roi[1]}"
                       if roi else "FULL FRAME")
            avg_str = f"AVG:{avg_win}f hit>{min_hit_pct:.0%}" if use_avg else "AVG:off"
            sam_str = ("SAM:on" if (use_sam and sam_predictor is not None)
                       else ("SAM:off" if sam_predictor is not None else "SAM:--"))

            hud = (f"FPS {fps_filter:.1f} | {len(dets)} det | "
                   f"conf={conf:.2f} | {roi_str} | {avg_str} | {sam_str}")
            cv2.putText(vis, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 1, cv2.LINE_AA)

            help_txt = ("Left-drag: ROI | Right-click: clear | "
                        "t: toggle avg | s: toggle SAM | q: quit")
            cv2.putText(vis, help_txt, (10, CAMERA_HEIGHT - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2,
                        cv2.LINE_AA)
            cv2.putText(vis, help_txt, (10, CAMERA_HEIGHT - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
                        cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("t"):
                use_avg = not use_avg
                if not use_avg:
                    averager.reset()
                print(f"Temporal averaging: {'ON' if use_avg else 'OFF'}")
            elif key == ord("s"):
                if sam_predictor is None:
                    print("[SAM2] not loaded — relaunch with --sam to enable.")
                else:
                    use_sam = not use_sam
                    print(f"SAM2 overlay: {'ON' if use_sam else 'OFF'}")

    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
