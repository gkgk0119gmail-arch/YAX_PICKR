"""
Detection pipeline: RT-DETR detection + multi-frame tracking + SAM2 segmentation.

Handles:
- RealSense camera initialization and frame capture
- RT-DETR object detection with background filtering
- Multi-frame tracking with IoU matching
- SAM2 segmentation on confirmed tracks
- Depth-based filtering of overlapping masks

Usage:
    from detection_pipeline import DetectionPipeline, PipelineConfig
    
    config = PipelineConfig()
    pipeline = DetectionPipeline(config)
    result = pipeline.run()
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from ultralytics import YOLO

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class PipelineConfig:
    """Configuration for the detection pipeline."""
    
    # Camera settings
    WIDTH: int = 1280
    HEIGHT: int = 720
    FPS: int = 30
    
    # Depth settings
    DEPTH_WIDTH: int = 1280
    DEPTH_HEIGHT: int = 720
    DEPTH_UNITS: float = 0.0001
    LASER_POWER: int = 360
    
    # Depth filter settings
    FILTER_DECIMATION_ENABLED: bool = True
    FILTER_DISPARITY_ENABLED: bool = False
    FILTER_SPATIAL_ENABLED: bool = False
    FILTER_TEMPORAL_ENABLED: bool = True
    FILTER_HOLE_FILLING_ENABLED: bool = False
    FILTER_DECIMATION_MAGNITUDE: int = 1
    FILTER_SPATIAL_ALPHA: float = 0.25
    FILTER_SPATIAL_DELTA: int = 20
    FILTER_TEMPORAL_ALPHA: float = 0.2
    FILTER_TEMPORAL_DELTA: int = 20
    
    # Detection settings
    DET_WEIGHTS: Path = Path("/home/jeong/perception/models/rtdetr-l.pt")
    CONF: float = 0.1
    IOU: float = 0.3
    MAX_DET: int = 100
    
    # Capture settings
    NUM_FRAMES: int = 10
    FRAME_DELAY: float = 0.5
    WARMUP_FRAMES: int = 30
    
    # Crop settings
    CROP_X1: int | None = 358
    CROP_Y1: int | None = 33
    CROP_X2: int | None = 791
    CROP_Y2: int | None = 465
    CROP_CONFIG_FILE: Path = Path("/home/jeong/perception/app/data/crop_config.json")
    
    # Background filtering
    MAX_BOX_AREA_RATIO: float = 0.8
    MIN_BOX_AREA: int = 300
    EXCLUDE_CLASSES: list[str] = field(default_factory=lambda: [
        "dining table", "bed", "couch", "chair", "bench",
        "tv", "laptop", "refrigerator", "oven", "sink",
    ])
    FILTER_EDGE_BOXES: bool = False
    EDGE_MARGIN: int = 5
    
    # Custom NMS
    CUSTOM_NMS: bool = True
    CUSTOM_NMS_IOU: float = 0.1
    
    # Tracking
    TRACK_IOU_THRESHOLD: float = 0.3
    MIN_AVG_CONF: float = 0.1
    
    # Depth-based filtering
    DEPTH_OVERLAP_THRESHOLD: float = 0.3
    DEPTH_DIFF_THRESHOLD_M: float = 0.02
    
    # SAM2 settings
    SAM2_CKPT: Path = Path("/home/jeong/perception/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
    SAM2_CFG: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    MASK_ALPHA: float = 0.5
    
    # Output
    OUTPUT_DIR: Path = Path("/home/jeong/perception/app/data")


# =============================================================================
# TRACKING
# =============================================================================
@dataclass
class TrackedObject:
    """Represents an object tracked across multiple frames."""
    track_id: int
    detections: list[tuple[int, list[float], float, str]] = field(default_factory=list)
    
    def add_detection(self, frame_idx: int, xyxy: list[float], conf: float, name: str):
        self.detections.append((frame_idx, xyxy, conf, name))
    
    def get_avg_confidence(self, total_frames: int) -> float:
        return sum(det[2] for det in self.detections) / total_frames
    
    def get_best_detection(self) -> tuple[list[float], float, str]:
        best = max(self.detections, key=lambda d: d[2])
        return best[1], best[2], best[3]
    
    def get_avg_bbox(self) -> list[float]:
        if not self.detections:
            return [0, 0, 0, 0]
        boxes = [det[1] for det in self.detections]
        return [sum(b[i] for b in boxes) / len(boxes) for i in range(4)]


class MultiFrameTracker:
    """Tracks objects across multiple frames using IoU matching."""
    
    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.tracks: list[TrackedObject] = []
        self.next_id = 0
    
    def _compute_iou(self, box1: list[float], box2: list[float]) -> float:
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0
    
    def update(self, frame_idx: int, detections: list[dict]):
        if not detections:
            return
        
        track_boxes = [t.get_avg_bbox() if t.detections else None for t in self.tracks]
        used_tracks, used_dets = set(), set()
        
        iou_pairs = []
        for det_idx, det in enumerate(detections):
            for track_idx, track_box in enumerate(track_boxes):
                if track_box is None:
                    continue
                iou = self._compute_iou(det["xyxy"], track_box)
                if iou >= self.iou_threshold:
                    iou_pairs.append((iou, det_idx, track_idx))
        
        iou_pairs.sort(reverse=True)
        for iou, det_idx, track_idx in iou_pairs:
            if det_idx in used_dets or track_idx in used_tracks:
                continue
            det = detections[det_idx]
            self.tracks[track_idx].add_detection(frame_idx, det["xyxy"], det["conf"], det["name"])
            used_tracks.add(track_idx)
            used_dets.add(det_idx)
        
        for det_idx, det in enumerate(detections):
            if det_idx not in used_dets:
                new_track = TrackedObject(track_id=self.next_id)
                new_track.add_detection(frame_idx, det["xyxy"], det["conf"], det["name"])
                self.tracks.append(new_track)
                self.next_id += 1
    
    def get_confirmed_tracks(self, total_frames: int, min_avg_conf: float) -> list[TrackedObject]:
        return [t for t in self.tracks if t.get_avg_confidence(total_frames) >= min_avg_conf]


# =============================================================================
# REALSENSE CAMERA
# =============================================================================
def _get_realsense_device():
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        raise RuntimeError("No RealSense devices found")
    return devs[0]


def _setup_depth_filters(cfg: PipelineConfig) -> dict:
    filters = {k: None for k in ['decimation', 'spatial', 'temporal', 
                                  'hole_filling', 'depth_to_disparity', 'disparity_to_depth']}
    
    if cfg.FILTER_DECIMATION_ENABLED:
        filters['decimation'] = rs.decimation_filter()
        filters['decimation'].set_option(rs.option.filter_magnitude, cfg.FILTER_DECIMATION_MAGNITUDE)
    if cfg.FILTER_SPATIAL_ENABLED:
        filters['spatial'] = rs.spatial_filter()
        filters['spatial'].set_option(rs.option.filter_smooth_alpha, cfg.FILTER_SPATIAL_ALPHA)
        filters['spatial'].set_option(rs.option.filter_smooth_delta, cfg.FILTER_SPATIAL_DELTA)
    if cfg.FILTER_TEMPORAL_ENABLED:
        filters['temporal'] = rs.temporal_filter()
        filters['temporal'].set_option(rs.option.filter_smooth_alpha, cfg.FILTER_TEMPORAL_ALPHA)
        filters['temporal'].set_option(rs.option.filter_smooth_delta, cfg.FILTER_TEMPORAL_DELTA)
    if cfg.FILTER_HOLE_FILLING_ENABLED:
        filters['hole_filling'] = rs.hole_filling_filter()
    if cfg.FILTER_DISPARITY_ENABLED:
        filters['depth_to_disparity'] = rs.disparity_transform(True)
        filters['disparity_to_depth'] = rs.disparity_transform(False)
    
    return filters


def _apply_depth_filters(depth_frame, filters: dict):
    depth = depth_frame
    for key in ['decimation', 'depth_to_disparity', 'spatial', 'temporal', 
                'disparity_to_depth', 'hole_filling']:
        if filters[key]:
            depth = filters[key].process(depth)
    return depth


def init_realsense(cfg: PipelineConfig) -> tuple[rs.pipeline, rs.align, float, dict]:
    """Initialize RealSense pipeline. Returns (pipeline, align, depth_scale, filters)."""
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.color, cfg.WIDTH, cfg.HEIGHT, rs.format.bgr8, cfg.FPS)
    rs_cfg.enable_stream(rs.stream.depth, cfg.DEPTH_WIDTH, cfg.DEPTH_HEIGHT, rs.format.z16, cfg.FPS)
    
    pipeline = None
    recoverable_errors = ["get_xu(ctrl=1)", "Device or resource busy", "failed to set power state",
                          "Couldn't resolve requests", "Frame didn't arrive"]
    
    while True:
        try:
            pipeline = rs.pipeline()
            profile = pipeline.start(rs_cfg)
            
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_accuracy)
            depth_sensor.set_option(rs.option.depth_units, cfg.DEPTH_UNITS)
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
            depth_sensor.set_option(rs.option.laser_power, cfg.LASER_POWER)
            
            for sensor in profile.get_device().sensors:
                if sensor.get_info(rs.camera_info.name) == 'RGB Camera':
                    sensor.set_option(rs.option.enable_auto_exposure, 1)
                    break
            
            depth_scale = depth_sensor.get_depth_scale()
            print(f"[Camera] Depth scale: {depth_scale:.6f} m/unit")
            
            align = rs.align(rs.stream.color)
            depth_filters = _setup_depth_filters(cfg)
            
            print(f"[Camera] Warming up ({cfg.WARMUP_FRAMES} frames)...")
            for _ in range(cfg.WARMUP_FRAMES):
                pipeline.wait_for_frames()
            
            print("[Camera] Initialized successfully.")
            return pipeline, align, depth_scale, depth_filters
            
        except RuntimeError as e:
            msg = str(e)
            print(f"[Camera] Error: {msg}")
            if pipeline:
                try: pipeline.stop()
                except: pass
                pipeline = None
            
            if any(err in msg for err in recoverable_errors):
                print("[Camera] Doing hardware reset...")
                try:
                    _get_realsense_device().hardware_reset()
                    time.sleep(3)
                except Exception as reset_err:
                    print(f"[Camera] Reset failed: {reset_err}")
                    time.sleep(2)
            else:
                raise


def get_camera_intrinsics(pipeline: rs.pipeline) -> dict:
    profile = pipeline.get_active_profile()
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    return {"fx": intrinsics.fx, "fy": intrinsics.fy, 
            "cx": intrinsics.ppx, "cy": intrinsics.ppy,
            "width": intrinsics.width, "height": intrinsics.height}


def get_color_and_depth_frame(pipeline: rs.pipeline, align: rs.align, 
                               depth_filters: dict | None = None):
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
    
    if not color:
        return None, None, None
    
    if depth and depth_filters:
        depth = _apply_depth_filters(depth, depth_filters)
    
    return (np.asanyarray(color.get_data()), 
            np.asanyarray(depth.get_data()) if depth else None, 
            depth)


# =============================================================================
# DETECTION
# =============================================================================
def run_detection(det_model: YOLO, img_bgr: np.ndarray, conf: float, 
                  iou: float, max_det: int, device: str) -> list[dict]:
    results = det_model.predict(source=img_bgr, conf=conf, iou=iou, max_det=max_det,
                                 verbose=False, device=0 if device == "cuda" else "cpu")
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []
    
    xyxy = r0.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    cfs = r0.boxes.conf.detach().cpu().numpy().astype(float)
    cls = r0.boxes.cls.detach().cpu().numpy().astype(int)
    
    dets = [{"xyxy": xyxy[i].tolist(), "conf": float(cfs[i]), "cls": int(cls[i]),
             "name": det_model.names[int(cls[i])]} for i in range(len(xyxy))]
    dets.sort(key=lambda d: d["conf"], reverse=True)
    for j, d in enumerate(dets):
        d["idx"] = j
    return dets


def filter_background(dets: list[dict], img_shape: tuple[int, int], 
                      cfg: PipelineConfig) -> list[dict]:
    h, w = img_shape
    img_area = h * w
    filtered = []
    
    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        box_area = (x2 - x1) * (y2 - y1)
        
        if d["name"].lower() in [c.lower() for c in cfg.EXCLUDE_CLASSES]:
            continue
        if box_area / img_area > cfg.MAX_BOX_AREA_RATIO or box_area < cfg.MIN_BOX_AREA:
            continue
        if cfg.FILTER_EDGE_BOXES:
            m = cfg.EDGE_MARGIN
            if x1 <= m or y1 <= m or x2 >= w - m or y2 >= h - m:
                continue
        filtered.append(d)
    
    for j, d in enumerate(filtered):
        d["idx"] = j
    return filtered


def compute_iou(box1: list[float], box2: list[float]) -> float:
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0


def custom_nms(dets: list[dict], iou_threshold: float) -> list[dict]:
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
    keep, suppressed = [], set()
    
    for i, det in enumerate(dets):
        if i in suppressed:
            continue
        keep.append(det)
        for j in range(i + 1, len(dets)):
            if j not in suppressed and compute_iou(det["xyxy"], dets[j]["xyxy"]) > iou_threshold:
                suppressed.add(j)
    
    for j, d in enumerate(keep):
        d["idx"] = j
    return keep


def draw_detections(img_bgr: np.ndarray, dets: list[dict]) -> np.ndarray:
    vis = img_bgr.copy()
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = map(int, d["xyxy"])
        conf = d["conf"]
        color = (0, 255, 0) if conf >= 0.5 else (0, 255, 255) if conf >= 0.2 else (0, 128, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        # Use index as primary label for clarity when there are duplicate classes
        label = f'[{i}] {conf:.2f}'
        cv2.putText(vis, label, (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return vis


# =============================================================================
# SAM2 SEGMENTATION
# =============================================================================
def pick_autocast_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except: pass
        return torch.float16
    return torch.float32


def load_sam2(ckpt: Path, model_cfg: str, device: str) -> SAM2ImagePredictor:
    return SAM2ImagePredictor(build_sam2(model_cfg, str(ckpt), device=device))


def run_sam2_on_boxes(predictor: SAM2ImagePredictor, img_bgr: np.ndarray,
                      dets: list[dict], device: str, autocast_dtype: torch.dtype) -> list[np.ndarray]:
    if not dets:
        return []
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    masks = []
    
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                predictor.set_image(img_rgb)
        else:
            predictor.set_image(img_rgb)
        
        for d in dets:
            box = np.array(d["xyxy"], dtype=np.float32)
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    mask_preds, _, _ = predictor.predict(box=box, multimask_output=False)
            else:
                mask_preds, _, _ = predictor.predict(box=box, multimask_output=False)
            masks.append((mask_preds[0] > 0).astype(np.uint8))
    
    return masks


# =============================================================================
# VISUALIZATION & FILTERING
# =============================================================================
def generate_colors(n: int) -> list[tuple[int, int, int]]:
    colors = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        color = cv2.cvtColor(np.uint8([[[hue, 255, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(color[0]), int(color[1]), int(color[2])))
    return colors


def draw_segmentations(img_bgr: np.ndarray, dets: list[dict], 
                       masks: list[np.ndarray], alpha: float = 0.5) -> np.ndarray:
    vis = img_bgr.copy()
    if not masks:
        return vis
    
    colors = generate_colors(len(masks))
    for i, (d, mask) in enumerate(zip(dets, masks)):
        overlay = vis.copy()
        overlay[mask > 0] = colors[i]
        vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)
        x1, y1 = int(d["xyxy"][0]), int(d["xyxy"][1])
        # Use index as primary label for clarity when there are duplicate classes
        label = f'[{i}] {d["conf"]:.2f}'
        cv2.putText(vis, label, (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def compute_mask_overlap(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    min_area = min(np.sum(mask1 > 0), np.sum(mask2 > 0))
    return intersection / min_area if min_area > 0 else 0


def get_mask_median_depth(mask: np.ndarray, depth: np.ndarray, depth_scale: float) -> float:
    valid = depth[mask > 0]
    valid = valid[valid > 0]
    return float(np.median(valid)) * depth_scale if len(valid) > 0 else float('nan')


def filter_overlapping_by_depth(dets: list[dict], masks: list[np.ndarray], depth: np.ndarray,
                                 depth_scale: float, overlap_thresh: float, 
                                 depth_diff_thresh: float) -> tuple[list[dict], list[np.ndarray]]:
    if len(masks) <= 1:
        return dets, masks
    
    n = len(masks)
    areas = [np.sum(m > 0) for m in masks]
    depths = [get_mask_median_depth(m, depth, depth_scale) for m in masks]
    
    print(f"  [Depth] Analyzing {n} masks:")
    for i in range(n):
        print(f"    Mask {i}: {dets[i].get('name', '?')}, area={areas[i]}, depth={depths[i]:.3f}m")
    
    to_remove = set()
    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            
            overlap = compute_mask_overlap(masks[i], masks[j])
            if np.isnan(depths[i]) or np.isnan(depths[j]) or overlap < overlap_thresh:
                continue
            
            depth_diff = abs(depths[i] - depths[j])
            print(f"    Pair ({i},{j}): overlap={overlap:.3f}, depth_diff={depth_diff*1000:.1f}mm", end="")
            
            if depth_diff >= depth_diff_thresh:
                print(f" -> SKIP")
                continue
            
            idx_to_remove = i if areas[i] < areas[j] else j
            to_remove.add(idx_to_remove)
            print(f" -> REMOVE mask {idx_to_remove}")
    
    return ([d for i, d in enumerate(dets) if i not in to_remove],
            [m for i, m in enumerate(masks) if i not in to_remove])


# =============================================================================
# UTILITIES
# =============================================================================
def load_crop_config(cfg: PipelineConfig) -> tuple[int, int, int, int] | None:
    if all(v is not None for v in [cfg.CROP_X1, cfg.CROP_Y1, cfg.CROP_X2, cfg.CROP_Y2]):
        return cfg.CROP_X1, cfg.CROP_Y1, cfg.CROP_X2, cfg.CROP_Y2
    if cfg.CROP_CONFIG_FILE.exists():
        with open(cfg.CROP_CONFIG_FILE) as f:
            data = json.load(f)
        return data["x1"], data["y1"], data["x2"], data["y2"]
    return None


def crop_image(img: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop is None:
        return img
    x1, y1, x2, y2 = crop
    return img[y1:y2, x1:x2]


def uncrop_mask(mask: np.ndarray, crop: tuple[int, int, int, int], 
                full_shape: tuple[int, int]) -> np.ndarray:
    x1, y1, x2, y2 = crop
    full_mask = np.zeros(full_shape, dtype=mask.dtype)
    full_mask[y1:y2, x1:x2] = mask
    return full_mask


# =============================================================================
# PIPELINE RESULT
# =============================================================================
class PipelineResult(NamedTuple):
    """Result from the detection pipeline."""
    detections: list[dict]
    masks: list[np.ndarray]
    depth_frame: rs.depth_frame
    depth_array: np.ndarray
    depth_scale: float
    crop: tuple[int, int, int, int] | None
    full_shape: tuple[int, int]
    intrinsics: dict
    session_dir: Path
    color_image: np.ndarray | None = None  # Full color image (BGR)


# =============================================================================
# DETECTION PIPELINE CLASS
# =============================================================================
class DetectionPipeline:
    """Main pipeline: camera capture + detection + tracking + segmentation."""
    
    def __init__(self, config: PipelineConfig | None = None):
        self.cfg = config or PipelineConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._det_model = None
        self._sam2_predictor = None
        self._autocast_dtype = None
        self._pipeline = None
        self._align = None
        self._depth_scale = None
        self._depth_filters = None
        self._intrinsics = None
    
    def _init_models(self):
        if self._det_model is None:
            print("[Init] Loading RT-DETR model...")
            if not self.cfg.DET_WEIGHTS.exists():
                raise FileNotFoundError(f"Weights not found: {self.cfg.DET_WEIGHTS}")
            self._det_model = YOLO(str(self.cfg.DET_WEIGHTS))
        
        if self._sam2_predictor is None:
            print("[Init] Loading SAM2...")
            if not self.cfg.SAM2_CKPT.exists():
                raise FileNotFoundError(f"SAM2 not found: {self.cfg.SAM2_CKPT}")
            self._autocast_dtype = pick_autocast_dtype()
            self._sam2_predictor = load_sam2(self.cfg.SAM2_CKPT, self.cfg.SAM2_CFG, self.device)
    
    def _init_camera(self):
        if self._pipeline is None:
            print("[Init] Starting RealSense camera...")
            self._pipeline, self._align, self._depth_scale, self._depth_filters = init_realsense(self.cfg)
            self._intrinsics = get_camera_intrinsics(self._pipeline)
            print(f"[Init] Intrinsics: fx={self._intrinsics['fx']:.1f}, fy={self._intrinsics['fy']:.1f}")
    
    def run(self, save_frames: bool = True) -> PipelineResult | None:
        """Run full detection pipeline. Returns PipelineResult or None."""
        print("=" * 60)
        print("RT-DETR Multi-Frame Capture with Tracking")
        print("=" * 60)
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print(f"[Init] Device: {self.device}")
        
        self._init_models()
        self._init_camera()
        
        crop = load_crop_config(self.cfg)
        if crop:
            print(f"[Crop] Region: {crop[0]},{crop[1]} -> {crop[2]},{crop[3]}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.cfg.OUTPUT_DIR / f"capture_{timestamp}"
        if save_frames:
            session_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Output] {session_dir}")
        
        tracker = MultiFrameTracker(self.cfg.TRACK_IOU_THRESHOLD)
        stored_frames = []
        
        print(f"\n[Capture] {self.cfg.NUM_FRAMES} frames, conf={self.cfg.CONF}")
        print("-" * 60)
        
        captured = 0
        try:
            for idx in range(self.cfg.NUM_FRAMES):
                color, depth_np, depth_frame = get_color_and_depth_frame(
                    self._pipeline, self._align, self._depth_filters)
                if color is None:
                    continue
                
                cropped = crop_image(color, crop)
                cropped_depth = crop_image(depth_np, crop) if depth_np is not None else None
                stored_frames.append((idx, cropped.copy(), 
                                      cropped_depth.copy() if cropped_depth is not None else None,
                                      depth_np.copy() if depth_np is not None else None, 
                                      depth_frame,
                                      color.copy()))
                
                t0 = time.time()
                dets = run_detection(self._det_model, cropped, self.cfg.CONF, 
                                     self.cfg.IOU, self.cfg.MAX_DET, self.device)
                det_ms = (time.time() - t0) * 1000
                
                dets = filter_background(dets, cropped.shape[:2], self.cfg)
                if self.cfg.CUSTOM_NMS:
                    dets = custom_nms(dets, self.cfg.CUSTOM_NMS_IOU)
                
                tracker.update(idx, dets)
                
                if save_frames:
                    cv2.imwrite(str(session_dir / f"frame_{idx:04d}_det.png"), 
                               draw_detections(cropped, dets))
                
                names = [d["name"] for d in dets[:5]]
                print(f"[{idx:3d}] {len(dets):3d} dets | {det_ms:5.1f}ms | tracks={len(tracker.tracks)} | {names}")
                captured += 1
                
                if idx < self.cfg.NUM_FRAMES - 1:
                    time.sleep(self.cfg.FRAME_DELAY)
        except KeyboardInterrupt:
            print("\n[Interrupt]")
        finally:
            self._pipeline.stop()
            self._pipeline = None
        
        print("-" * 60)
        confirmed = tracker.get_confirmed_tracks(captured, self.cfg.MIN_AVG_CONF)
        print(f"[Tracking] {len(tracker.tracks)} tracks, {len(confirmed)} confirmed")
        
        for t in tracker.tracks:
            avg = t.get_avg_confidence(captured)
            _, best_conf, best_name = t.get_best_detection()
            status = "✓" if avg >= self.cfg.MIN_AVG_CONF else "✗"
            print(f"  {status} Track {t.track_id}: avg={avg:.3f}, best={best_name} ({best_conf:.2f})")
        
        if not confirmed or not stored_frames:
            print("[SAM] No objects to segment.")
            return None
        
        print("-" * 60)
        print(f"[SAM] Segmenting {len(confirmed)} objects...")
        
        last_idx, last_frame, last_depth, full_depth, last_depth_frame, full_color = stored_frames[-1]
        
        final_dets = []
        for t in confirmed:
            _, best_conf, best_name = t.get_best_detection()
            final_dets.append({
                "xyxy": t.get_avg_bbox(), "conf": t.get_avg_confidence(captured),
                "name": best_name, "track_id": t.track_id, "idx": len(final_dets)
            })
        
        t1 = time.time()
        masks = run_sam2_on_boxes(self._sam2_predictor, last_frame, final_dets,
                                   self.device, self._autocast_dtype)
        print(f"[SAM] Done in {(time.time() - t1) * 1000:.1f}ms ({len(masks)} masks)")
        
        if last_depth is not None and len(masks) > 1:
            print("[Depth] Filtering overlaps...")
            orig = len(masks)
            final_dets, masks = filter_overlapping_by_depth(
                final_dets, masks, last_depth, self._depth_scale,
                self.cfg.DEPTH_OVERLAP_THRESHOLD, self.cfg.DEPTH_DIFF_THRESHOLD_M)
            if len(masks) < orig:
                print(f"[Depth] {orig} -> {len(masks)} masks")
        
        if save_frames:
            vis = draw_segmentations(last_frame, final_dets, masks, self.cfg.MASK_ALPHA)
            cv2.imwrite(str(session_dir / f"frame_{last_idx:04d}_seg.png"), vis)
            cv2.imwrite(str(session_dir / "final_segmentation.png"), vis)
            print(f"[SAM] Saved: {session_dir / 'final_segmentation.png'}")
        
        return PipelineResult(
            detections=final_dets, masks=masks, depth_frame=last_depth_frame,
            depth_array=full_depth, depth_scale=self._depth_scale, crop=crop,
            full_shape=(self.cfg.HEIGHT, self.cfg.WIDTH), intrinsics=self._intrinsics,
            session_dir=session_dir, color_image=full_color
        )
