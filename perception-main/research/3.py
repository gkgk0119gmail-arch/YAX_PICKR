"""
Multi-frame capture with RT-DETR detection using low confidence threshold.
Tracks bboxes across frames and only runs SAM on consistently detected objects.

Run crop_selector.py first to define the crop region interactively.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from ultralytics import YOLO

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import meshcat
import meshcat.geometry as geom


# =========================
# TRACKING
# =========================
@dataclass
class TrackedObject:
    """Represents an object tracked across multiple frames."""
    track_id: int
    # List of (frame_idx, xyxy, conf, name) for frames where detected
    detections: list[tuple[int, list[float], float, str]] = field(default_factory=list)
    
    def add_detection(self, frame_idx: int, xyxy: list[float], conf: float, name: str):
        self.detections.append((frame_idx, xyxy, conf, name))
    
    def get_avg_confidence(self, total_frames: int) -> float:
        """Average confidence across all frames (0 for missing frames)."""
        total_conf = sum(det[2] for det in self.detections)
        return total_conf / total_frames
    
    def get_last_bbox(self) -> list[float]:
        """Get the most recent bounding box."""
        return self.detections[-1][1]
    
    def get_best_detection(self) -> tuple[list[float], float, str]:
        """Get detection with highest confidence."""
        best = max(self.detections, key=lambda d: d[2])
        return best[1], best[2], best[3]  # xyxy, conf, name
    
    def get_avg_bbox(self) -> list[float]:
        """Get average bounding box across all detections."""
        if not self.detections:
            return [0, 0, 0, 0]
        boxes = [det[1] for det in self.detections]
        avg = [sum(b[i] for b in boxes) / len(boxes) for i in range(4)]
        return avg


class MultiFrameTracker:
    """Tracks objects across multiple frames using IoU matching."""
    
    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.tracks: list[TrackedObject] = []
        self.next_id = 0
    
    def _compute_iou(self, box1: list[float], box2: list[float]) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def update(self, frame_idx: int, detections: list[dict]):
        """Match detections to existing tracks or create new ones."""
        if not detections:
            return
        
        # Get current track boxes (use average bbox for matching stability)
        track_boxes = []
        for track in self.tracks:
            if track.detections:
                track_boxes.append(track.get_avg_bbox())
            else:
                track_boxes.append(None)
        
        # Match detections to tracks
        used_tracks = set()
        used_dets = set()
        
        # Compute IoU matrix
        iou_pairs = []
        for det_idx, det in enumerate(detections):
            det_box = det["xyxy"]
            for track_idx, track_box in enumerate(track_boxes):
                if track_box is None:
                    continue
                iou = self._compute_iou(det_box, track_box)
                if iou >= self.iou_threshold:
                    iou_pairs.append((iou, det_idx, track_idx))
        
        # Sort by IoU descending and greedily assign
        iou_pairs.sort(reverse=True)
        for iou, det_idx, track_idx in iou_pairs:
            if det_idx in used_dets or track_idx in used_tracks:
                continue
            # Match found
            det = detections[det_idx]
            self.tracks[track_idx].add_detection(
                frame_idx, det["xyxy"], det["conf"], det["name"]
            )
            used_tracks.add(track_idx)
            used_dets.add(det_idx)
        
        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in used_dets:
                new_track = TrackedObject(track_id=self.next_id)
                new_track.add_detection(frame_idx, det["xyxy"], det["conf"], det["name"])
                self.tracks.append(new_track)
                self.next_id += 1
    
    def get_confirmed_tracks(self, total_frames: int, min_avg_conf: float) -> list[TrackedObject]:
        """Get tracks with average confidence above threshold."""
        confirmed = []
        for track in self.tracks:
            avg_conf = track.get_avg_confidence(total_frames)
            if avg_conf >= min_avg_conf:
                confirmed.append(track)
        return confirmed


# =========================
# CONFIG
# =========================
@dataclass
class Config:
    # Camera settings (color stream)
    WIDTH: int = 1280
    HEIGHT: int = 720
    FPS: int = 30
    
    # Depth settings
    DEPTH_WIDTH: int = 1280
    DEPTH_HEIGHT: int = 720
    DEPTH_UNITS: float = 0.0001  # 0.1mm precision
    LASER_POWER: int = 360  # Max power for better depth
    
    # Depth filter settings (matching cluster.py)
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
    
    # Detection settings - LOW confidence for capturing everything
    DET_WEIGHTS: Path = Path("/home/jeong/perception/models/rtdetr-l.pt")
    CONF: float = 0.1 # Very low confidence to detect everything
    IOU: float = 0.3
    MAX_DET: int = 100
    
    # Capture settings
    NUM_FRAMES: int = 10
    FRAME_DELAY: float = 0.5
    WARMUP_FRAMES: int = 30
    
    # Crop settings - set these directly OR use crop_config.json from crop_selector.py
    # If all are None, loads from crop_config.json. If that doesn't exist, uses full image.
    CROP_X1: int | None = 358
    CROP_Y1: int | None = 33
    CROP_X2: int | None = 791
    CROP_Y2: int | None = 465
    
    CROP_CONFIG_FILE: Path = Path("/home/jeong/perception/app/data/crop_config.json")
    
    # Background filtering
    # Max box area as fraction of image (filter out boxes larger than this)
    MAX_BOX_AREA_RATIO: float = 0.8  # Boxes > 50% of image are likely background
    # Min box area in pixels (filter out tiny noise)
    MIN_BOX_AREA: int = 300
    # Exclude these class names (common background classes)
    EXCLUDE_CLASSES: list[str] = field(default_factory=lambda: [
        "dining table", "bed", "couch", "chair", "bench",
        "tv", "laptop", "refrigerator", "oven", "sink",
    ])
    # Filter boxes touching image edges (likely background)
    FILTER_EDGE_BOXES: bool = False
    EDGE_MARGIN: int = 5  # Pixels from edge to consider "touching"
    
    # Custom NMS to merge overlapping boxes (regardless of class)
    CUSTOM_NMS: bool = True
    CUSTOM_NMS_IOU: float = 0.2  # Merge boxes with >20% overlap
    
    # Multi-frame tracking settings
    TRACK_IOU_THRESHOLD: float = 0.3  # IoU threshold for matching boxes across frames
    MIN_AVG_CONF: float = 0.1  # Minimum average confidence to run SAM
    
    # Depth-based segmentation filtering
    # Overlap = intersection / min(area1, area2) - catches when smaller mask is inside larger
    DEPTH_OVERLAP_THRESHOLD: float = 0.3  # Masks with overlap > this are considered overlapping
    DEPTH_DIFF_THRESHOLD_M: float = 0.02  # Depth difference < this (meters) means same object
    
    # SAM2 settings
    SAM2_CKPT: Path = Path("/home/jeong/perception/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
    SAM2_CFG: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    MASK_ALPHA: float = 0.5  # Overlay transparency
    
    # Output settings
    OUTPUT_DIR: Path = Path("/home/jeong/perception/app/data")


CFG = Config()


def load_crop_config(cfg: Config) -> tuple[int, int, int, int] | None:
    """Load crop coordinates from config or file. Returns (x1, y1, x2, y2) or None."""
    # First check if coordinates are set directly in config
    if all(v is not None for v in [cfg.CROP_X1, cfg.CROP_Y1, cfg.CROP_X2, cfg.CROP_Y2]):
        return cfg.CROP_X1, cfg.CROP_Y1, cfg.CROP_X2, cfg.CROP_Y2
    
    # Try loading from crop_config.json
    if cfg.CROP_CONFIG_FILE.exists():
        with open(cfg.CROP_CONFIG_FILE) as f:
            data = json.load(f)
        return data["x1"], data["y1"], data["x2"], data["y2"]
    
    # No crop defined - use full image
    return None


def get_realsense_device():
    """Get the first available RealSense device."""
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        raise RuntimeError("No RealSense devices found")
    return devs[0]


def setup_depth_filters(cfg: Config) -> dict:
    """Setup depth filter chain based on configuration.
    
    Returns dict with filter objects (or None if disabled).
    """
    filters = {
        'decimation': None,
        'spatial': None,
        'temporal': None,
        'hole_filling': None,
        'depth_to_disparity': None,
        'disparity_to_depth': None,
    }
    
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


def apply_depth_filters(depth_frame, filters: dict):
    """Apply filter chain to depth frame.
    
    Returns filtered depth frame.
    """
    depth = depth_frame
    
    if filters['decimation']:
        depth = filters['decimation'].process(depth)
    if filters['depth_to_disparity']:
        depth = filters['depth_to_disparity'].process(depth)
    if filters['spatial']:
        depth = filters['spatial'].process(depth)
    if filters['temporal']:
        depth = filters['temporal'].process(depth)
    if filters['disparity_to_depth']:
        depth = filters['disparity_to_depth'].process(depth)
    if filters['hole_filling']:
        depth = filters['hole_filling'].process(depth)
    
    return depth


def init_realsense(cfg: Config) -> tuple[rs.pipeline, rs.align, float, dict]:
    """Initialize RealSense pipeline with color and depth streams.
    
    Includes retry logic with hardware reset for camera connection failures.
    
    Returns (pipeline, align_processor, depth_scale, depth_filters).
    """
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.color, cfg.WIDTH, cfg.HEIGHT, rs.format.bgr8, cfg.FPS)
    rs_cfg.enable_stream(rs.stream.depth, cfg.DEPTH_WIDTH, cfg.DEPTH_HEIGHT, rs.format.z16, cfg.FPS)
    
    pipeline = None
    profile = None
    
    # Errors that can be recovered with a hardware reset
    recoverable_errors = [
        "get_xu(ctrl=1)",
        "Device or resource busy",
        "failed to set power state",
        "Couldn't resolve requests",
        "Frame didn't arrive",  # Timeout during warmup
    ]
    
    def do_hardware_reset():
        """Perform hardware reset and wait for re-enumeration."""
        print("[Camera] Camera busy or in bad state, doing hardware reset...")
        try:
            dev = get_realsense_device()
            dev.hardware_reset()
            print("[Camera] Reset triggered, waiting 3s for re-enumeration...")
            time.sleep(3)
        except Exception as reset_err:
            print(f"[Camera] Failed to reset device: {reset_err}")
            time.sleep(2)
    
    # Initialization loop with retry mechanism
    while True:
        try:
            pipeline = rs.pipeline()
            profile = pipeline.start(rs_cfg)
            
            # Configure depth sensor for high accuracy
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_accuracy)
            depth_sensor.set_option(rs.option.depth_units, cfg.DEPTH_UNITS)
            
            # Exposure settings for depth/IR sensor
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
            
            # Laser power (higher = better depth in low light / far objects)
            depth_sensor.set_option(rs.option.laser_power, cfg.LASER_POWER)
            
            # Configure color sensor exposure
            for sensor in profile.get_device().sensors:
                if sensor.get_info(rs.camera_info.name) == 'RGB Camera':
                    sensor.set_option(rs.option.enable_auto_exposure, 1)
                    print("[Camera] Color sensor auto-exposure enabled")
                    break
            
            # Get depth scale (now using our configured depth units)
            depth_scale = depth_sensor.get_depth_scale()
            print(f"[Camera] Depth scale: {depth_scale:.6f} m/unit")
            
            # Create align processor to align depth to color
            align = rs.align(rs.stream.color)
            
            # Setup depth filters
            depth_filters = setup_depth_filters(cfg)
            print(f"[Camera] Depth filters: decimation={cfg.FILTER_DECIMATION_ENABLED}, "
                  f"temporal={cfg.FILTER_TEMPORAL_ENABLED}, spatial={cfg.FILTER_SPATIAL_ENABLED}")

            # Warmup frames (inside try block so timeout triggers reset)
            print(f"[Camera] Warming up ({cfg.WARMUP_FRAMES} frames)...")
            for _ in range(cfg.WARMUP_FRAMES):
                pipeline.wait_for_frames()
            
            print("[Camera] Initialized successfully.")
            break
            
        except RuntimeError as e:
            msg = str(e)
            print(f"[Camera] Error: {msg}")
            
            # Try to stop pipe if it was partially started
            if pipeline:
                try:
                    pipeline.stop()
                except:
                    pass
                pipeline = None
            
            if any(err in msg for err in recoverable_errors):
                do_hardware_reset()
            else:
                # Fatal error (e.g. no device connected at all)
                raise RuntimeError(f"Fatal error initializing camera: {e}")

    return pipeline, align, depth_scale, depth_filters


def get_color_and_depth_frame(
    pipeline: rs.pipeline,
    align: rs.align,
    depth_filters: dict | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, rs.depth_frame | None]:
    """Get aligned color and depth frames from RealSense.
    
    Returns (color_bgr, depth_raw, depth_frame) where:
      - depth_raw is uint16 in sensor units
      - depth_frame is the RealSense depth frame object (for rs.pointcloud)
    """
    frames = pipeline.wait_for_frames()
    
    # Align depth to color
    aligned_frames = align.process(frames)
    
    color = aligned_frames.get_color_frame()
    depth = aligned_frames.get_depth_frame()
    
    if not color:
        return None, None, None
    
    # Apply depth filters if provided
    if depth and depth_filters:
        depth = apply_depth_filters(depth, depth_filters)
    
    color_np = np.asanyarray(color.get_data())
    depth_np = np.asanyarray(depth.get_data()) if depth else None
    
    return color_np, depth_np, depth


def crop_image(img: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    """Crop image to specified region (x1, y1, x2, y2) or return original."""
    if crop is None:
        return img
    x1, y1, x2, y2 = crop
    return img[y1:y2, x1:x2]


def run_detection(
    det_model: YOLO,
    img_bgr: np.ndarray,
    conf: float,
    iou: float,
    max_det: int,
    device: str,
) -> list[dict]:
    """Run RT-DETR detection on image."""
    results = det_model.predict(
        source=img_bgr,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False,
        device=0 if device == "cuda" else "cpu",
    )
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []

    xyxy = r0.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    cfs = r0.boxes.conf.detach().cpu().numpy().astype(float)
    cls = r0.boxes.cls.detach().cpu().numpy().astype(int)

    dets: list[dict] = []
    for i in range(len(xyxy)):
        dets.append({
            "xyxy": xyxy[i].tolist(),
            "conf": float(cfs[i]),
            "cls": int(cls[i]),
            "name": det_model.names[int(cls[i])],
        })

    dets.sort(key=lambda d: d["conf"], reverse=True)
    for j, d in enumerate(dets):
        d["idx"] = j
    return dets


def filter_background(
    dets: list[dict],
    img_shape: tuple[int, int],
    cfg: Config,
) -> list[dict]:
    """Filter out likely background detections."""
    h, w = img_shape
    img_area = h * w
    filtered = []
    
    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        box_w = x2 - x1
        box_h = y2 - y1
        box_area = box_w * box_h
        
        # Filter by class name
        if d["name"].lower() in [c.lower() for c in cfg.EXCLUDE_CLASSES]:
            continue
        
        # Filter by max area (too large = background)
        if box_area / img_area > cfg.MAX_BOX_AREA_RATIO:
            continue
        
        # Filter by min area (too small = noise)
        if box_area < cfg.MIN_BOX_AREA:
            continue
        
        # Filter boxes touching edges
        if cfg.FILTER_EDGE_BOXES:
            margin = cfg.EDGE_MARGIN
            if x1 <= margin or y1 <= margin or x2 >= w - margin or y2 >= h - margin:
                continue
        
        filtered.append(d)
    
    # Re-index
    for j, d in enumerate(filtered):
        d["idx"] = j
    
    return filtered


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def custom_nms(dets: list[dict], iou_threshold: float) -> list[dict]:
    """Apply NMS across all classes - keep highest confidence box when boxes overlap."""
    if not dets:
        return []
    
    # Sort by confidence (highest first)
    dets = sorted(dets, key=lambda d: d["conf"], reverse=True)
    
    keep = []
    suppressed = set()
    
    for i, det in enumerate(dets):
        if i in suppressed:
            continue
        
        keep.append(det)
        
        # Suppress all lower-confidence boxes that overlap with this one
        for j in range(i + 1, len(dets)):
            if j in suppressed:
                continue
            if compute_iou(det["xyxy"], dets[j]["xyxy"]) > iou_threshold:
                suppressed.add(j)
    
    # Re-index
    for j, d in enumerate(keep):
        d["idx"] = j
    
    return keep


def draw_detections(img_bgr: np.ndarray, dets: list[dict]) -> np.ndarray:
    """Draw detection boxes and labels on image."""
    vis = img_bgr.copy()
    
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        conf = d["conf"]
        
        # Color based on confidence
        if conf >= 0.5:
            color = (0, 255, 0)
        elif conf >= 0.2:
            color = (0, 255, 255)
        else:
            color = (0, 128, 255)
        
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f'{d["name"]} {conf:.2f}'
        cv2.putText(vis, label, (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    
    return vis


def pick_autocast_dtype() -> torch.dtype:
    """Pick best dtype for autocast."""
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def load_sam2(ckpt: Path, model_cfg: str, device: str) -> SAM2ImagePredictor:
    """Load SAM2 model."""
    model = build_sam2(model_cfg, str(ckpt), device=device)
    return SAM2ImagePredictor(model)


def run_sam2_on_boxes(
    predictor: SAM2ImagePredictor,
    img_bgr: np.ndarray,
    dets: list[dict],
    device: str,
    autocast_dtype: torch.dtype,
) -> list[np.ndarray]:
    """Run SAM2 segmentation on each detection box. Returns list of masks."""
    if not dets:
        return []
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    masks = []
    with torch.inference_mode():
        # Set image once
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                predictor.set_image(img_rgb)
        else:
            predictor.set_image(img_rgb)
        
        # Run prediction for each box
        for d in dets:
            box = np.array(d["xyxy"], dtype=np.float32)
            
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    mask_preds, scores, _ = predictor.predict(box=box, multimask_output=False)
            else:
                mask_preds, scores, _ = predictor.predict(box=box, multimask_output=False)
            
            mask = (mask_preds[0] > 0).astype(np.uint8)
            masks.append(mask)
    
    return masks


def generate_colors(n: int) -> list[tuple[int, int, int]]:
    """Generate n distinct colors."""
    colors = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        color = cv2.cvtColor(np.uint8([[[hue, 255, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(color[0]), int(color[1]), int(color[2])))
    return colors


def draw_segmentations(
    img_bgr: np.ndarray,
    dets: list[dict],
    masks: list[np.ndarray],
    alpha: float = 0.5,
) -> np.ndarray:
    """Draw segmentation masks with labels on image."""
    vis = img_bgr.copy()
    
    if not masks:
        return vis
    
    colors = generate_colors(len(masks))
    
    # Draw each mask
    for i, (d, mask) in enumerate(zip(dets, masks)):
        color = colors[i]
        
        # Create colored overlay
        overlay = vis.copy()
        overlay[mask > 0] = color
        vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)
        
        # Draw contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)
        
        # Draw label
        x1, y1, x2, y2 = map(int, d["xyxy"])
        label = f'{d["name"]} {d["conf"]:.2f}'
        cv2.putText(vis, label, (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    return vis


def compute_mask_overlap(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute overlap ratio between two masks.
    
    Returns intersection / min(area1, area2), which catches cases where
    a smaller mask is mostly contained within a larger one.
    """
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    area1 = np.sum(mask1 > 0)
    area2 = np.sum(mask2 > 0)
    min_area = min(area1, area2)
    return intersection / min_area if min_area > 0 else 0


def get_mask_median_depth(mask: np.ndarray, depth: np.ndarray, depth_scale: float) -> float:
    """Get median depth (in meters) of pixels within a mask.
    
    Returns NaN if no valid depth pixels in mask.
    """
    mask_pixels = depth[mask > 0]
    # Filter out zero (invalid) depth values
    valid_depths = mask_pixels[mask_pixels > 0]
    if len(valid_depths) == 0:
        return float('nan')
    return float(np.median(valid_depths)) * depth_scale


def get_camera_intrinsics(pipeline: rs.pipeline) -> dict:
    """Get color camera intrinsics from pipeline profile.
    
    Returns dict with fx, fy, cx, cy, width, height.
    """
    profile = pipeline.get_active_profile()
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()
    return {
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.ppx,
        "cy": intrinsics.ppy,
        "width": intrinsics.width,
        "height": intrinsics.height,
    }


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: dict,
    depth_scale: float,
) -> np.ndarray:
    """Convert depth image to 3D point cloud.
    
    Args:
        depth: HxW uint16 depth image in sensor units
        intrinsics: Camera intrinsics dict with fx, fy, cx, cy
        depth_scale: Meters per depth unit
        
    Returns:
        Nx3 array of 3D points (X, Y, Z in meters, camera frame)
    """
    h, w = depth.shape
    
    # Create pixel coordinate grids
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)
    
    # Convert depth to meters
    z = depth.astype(np.float32) * depth_scale
    
    # Project to 3D (standard pinhole camera model)
    x = (u - intrinsics["cx"]) * z / intrinsics["fx"]
    y = (v - intrinsics["cy"]) * z / intrinsics["fy"]
    
    # Stack to Nx3
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    
    return points


def uncrop_mask(
    mask: np.ndarray,
    crop: tuple[int, int, int, int],
    full_shape: tuple[int, int],
) -> np.ndarray:
    """Uncrop a mask from cropped coordinates to full image coordinates.
    
    Args:
        mask: HxW mask in cropped image coordinates
        crop: (x1, y1, x2, y2) crop coordinates
        full_shape: (height, width) of full image
        
    Returns:
        Full-size mask with the cropped mask placed at correct location
    """
    x1, y1, x2, y2 = crop
    full_h, full_w = full_shape
    
    full_mask = np.zeros((full_h, full_w), dtype=mask.dtype)
    full_mask[y1:y2, x1:x2] = mask
    
    return full_mask


# Camera to robot base transformation matrix
T_BASE_CAMERA = np.array([
    [-0.99870602, -0.05068679,  0.00413055,  0.46783590],
    [-0.05061085,  0.99857739,  0.01678431,  0.11403400],
    [-0.00497542,  0.01655354, -0.99985062,  1.02680489],
    [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
])


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform Nx3 points using a 4x4 transformation matrix.
    
    Args:
        points: Nx3 array of 3D points
        T: 4x4 homogeneous transformation matrix
        
    Returns:
        Nx3 array of transformed points
    """
    # Convert to homogeneous coordinates (Nx4)
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    points_h = np.hstack([points, ones])
    
    # Apply transformation: (4x4) @ (4xN) -> (4xN)
    transformed_h = (T @ points_h.T).T
    
    # Return xyz (drop homogeneous coordinate)
    return transformed_h[:, :3]


def visualize_pointcloud_meshcat(
    depth_frame: rs.depth_frame,
    rs_pc: rs.pointcloud,
    masks: list[np.ndarray],
    dets: list[dict],
    crop: tuple[int, int, int, int] | None,
    full_shape: tuple[int, int],
) -> meshcat.Visualizer:
    """Visualize point cloud with segmented objects highlighted in meshcat.
    
    Uses rs.pointcloud() for proper lens distortion handling.
    Points are transformed from camera frame to robot base frame using T_BASE_CAMERA.
    
    Args:
        depth_frame: RealSense depth frame object
        rs_pc: RealSense pointcloud object
        masks: List of masks (in cropped image coordinates)
        dets: List of detection dicts
        crop: Crop coordinates (x1, y1, x2, y2) or None
        full_shape: (height, width) of full image
        
    Returns:
        Meshcat visualizer instance
    """
    h, w = full_shape
    
    # Use RealSense SDK to compute point cloud (properly handles lens distortion!)
    print("[Meshcat] Computing point cloud with RealSense SDK (distortion-corrected)...")
    points_rs = rs_pc.calculate(depth_frame)
    points_camera = np.asanyarray(points_rs.get_vertices()).view(np.float32).reshape(-1, 3)
    
    # Filter valid points (non-zero depth, reasonable range)
    valid_mask = (points_camera[:, 2] > 0.1) & (points_camera[:, 2] < 3.0) & np.isfinite(points_camera[:, 2])
    
    print(f"[Meshcat] Point cloud: {valid_mask.sum():,} valid points out of {len(points_camera):,}")
    
    # Transform all points from camera frame to robot base frame
    points = transform_points(points_camera, T_BASE_CAMERA)
    
    # Uncrop masks to full image coordinates
    full_masks = []
    for mask in masks:
        if crop is not None:
            full_mask = uncrop_mask(mask, crop, (h, w))
        else:
            full_mask = mask
        full_masks.append(full_mask.flatten())
    
    # Create meshcat visualizer
    vis = meshcat.Visualizer().open()
    
    # Draw robot base coordinate frame at origin (X=red, Y=green, Z=blue)
    L = 0.1
    X_axis = np.array([[0, 0, 0], [L, 0, 0]], dtype=np.float32).T
    Y_axis = np.array([[0, 0, 0], [0, L, 0]], dtype=np.float32).T
    Z_axis = np.array([[0, 0, 0], [0, 0, L]], dtype=np.float32).T
    vis["frame"]["X"].set_object(
        geom.Line(geom.PointsGeometry(X_axis), geom.LineBasicMaterial(color=0xff0000, linewidth=3))
    )
    vis["frame"]["Y"].set_object(
        geom.Line(geom.PointsGeometry(Y_axis), geom.LineBasicMaterial(color=0x00ff00, linewidth=3))
    )
    vis["frame"]["Z"].set_object(
        geom.Line(geom.PointsGeometry(Z_axis), geom.LineBasicMaterial(color=0x0000ff, linewidth=3))
    )
    
    # Create combined mask of all segmented regions
    all_masked = np.zeros(h * w, dtype=bool)
    for fm in full_masks:
        all_masked |= (fm > 0)
    
    # Background points: valid AND not in any mask
    bg_mask = valid_mask & ~all_masked
    bg_points = points[bg_mask]
    
    # Subsample background for performance
    max_bg_points = 100000
    if bg_points.shape[0] > max_bg_points:
        indices = np.random.choice(bg_points.shape[0], size=max_bg_points, replace=False)
        bg_points = bg_points[indices]
    
    # Visualize background point cloud (gray)
    if bg_points.shape[0] > 0:
        vis["pointcloud"]["background"].set_object(
            geom.Points(
                geom.PointsGeometry(bg_points.T.astype(np.float32)),
                geom.PointsMaterial(size=0.002, color=0x888888)
            )
        )
        print(f"[Meshcat] Background: {bg_points.shape[0]:,} points (gray)")
    
    # Generate distinct colors for each mask
    colors = generate_colors(len(full_masks))
    
    # Visualize each segmented object with a different color
    for i, (fm, det, color) in enumerate(zip(full_masks, dets, colors)):
        obj_mask = valid_mask & (fm > 0)
        obj_points = points[obj_mask]
        
        if obj_points.shape[0] == 0:
            print(f"[Meshcat] Object {i} ({det['name']}): No valid points")
            continue
        
        # Convert BGR color to hex
        hex_color = (color[2] << 16) | (color[1] << 8) | color[0]
        
        vis["pointcloud"][f"object_{i}"].set_object(
            geom.Points(
                geom.PointsGeometry(obj_points.T.astype(np.float32)),
                geom.PointsMaterial(size=0.005, color=hex_color)
            )
        )
        print(f"[Meshcat] Object {i} ({det['name']}): {obj_points.shape[0]:,} points (color: #{hex_color:06x})")
    
    print(f"[Meshcat] Visualization URL: {vis.url()}")
    print(f"[Meshcat] Points shown in robot base frame (transformed from camera)")
    
    return vis


def filter_overlapping_by_depth(
    dets: list[dict],
    masks: list[np.ndarray],
    depth: np.ndarray,
    depth_scale: float,
    overlap_threshold: float,
    depth_diff_threshold_m: float,
) -> tuple[list[dict], list[np.ndarray]]:
    """Filter overlapping segmentations based on depth.
    
    For pairs of masks that overlap significantly (overlap > threshold):
    - Overlap = intersection / min(area1, area2) - catches when smaller mask
      is mostly contained in larger mask
    - If depth difference is very small (< depth_diff_threshold_m), they're likely
      the same object, so keep only the larger mask.
    
    Returns filtered (dets, masks).
    """
    if len(masks) <= 1:
        return dets, masks
    
    n = len(masks)
    
    # Compute mask areas and depths
    areas = [np.sum(m > 0) for m in masks]
    depths = [get_mask_median_depth(m, depth, depth_scale) for m in masks]
    
    # Debug: print mask info
    print(f"  [Depth] Analyzing {n} masks:")
    for i in range(n):
        name = dets[i].get("name", "?")
        print(f"    Mask {i}: {name}, area={areas[i]}, depth={depths[i]:.3f}m")
    
    # Track which indices to remove
    to_remove = set()
    
    # Check all pairs
    for i in range(n):
        if i in to_remove:
            continue
        for j in range(i + 1, n):
            if j in to_remove:
                continue
            
            # Check if masks overlap significantly (using intersection/min_area)
            overlap = compute_mask_overlap(masks[i], masks[j])
            
            # Check depth difference
            depth_i = depths[i]
            depth_j = depths[j]
            
            # Skip if either has invalid depth
            if np.isnan(depth_i) or np.isnan(depth_j):
                print(f"    Pair ({i},{j}): overlap={overlap:.3f}, SKIP (invalid depth)")
                continue
            
            depth_diff = abs(depth_i - depth_j)
            
            # Debug output for all pairs
            if overlap > 0.01:  # Only print if there's some overlap
                print(f"    Pair ({i},{j}): overlap={overlap:.3f}, depth_diff={depth_diff*1000:.1f}mm", end="")
            
            if overlap < overlap_threshold:
                if overlap > 0.01:
                    print(f" -> SKIP (overlap < {overlap_threshold})")
                continue
            
            if depth_diff >= depth_diff_threshold_m:
                print(f" -> SKIP (depth_diff >= {depth_diff_threshold_m*1000:.0f}mm)")
                continue
            
            # Same depth and overlapping - remove the smaller mask
            if areas[i] < areas[j]:
                to_remove.add(i)
                print(f" -> REMOVE mask {i} (smaller)")
            else:
                to_remove.add(j)
                print(f" -> REMOVE mask {j} (smaller)")
    
    # Filter out removed indices
    filtered_dets = [d for i, d in enumerate(dets) if i not in to_remove]
    filtered_masks = [m for i, m in enumerate(masks) if i not in to_remove]
    
    return filtered_dets, filtered_masks


def main():
    print("=" * 60)
    print("RT-DETR Multi-Frame Capture with Tracking")
    print("=" * 60)
    
    # Load crop configuration
    crop = load_crop_config(CFG)
    if crop:
        x1, y1, x2, y2 = crop
        print(f"[Crop] Using region: ({x1}, {y1}) -> ({x2}, {y2}) = {x2-x1}x{y2-y1}")
    else:
        print("[Crop] No crop defined, using full image")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = CFG.OUTPUT_DIR / f"capture_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[Output] Saving to: {session_dir}")
    
    # Setup PyTorch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Init] Using device: {device}")
    
    if not CFG.DET_WEIGHTS.exists():
        raise FileNotFoundError(f"Detection weights not found: {CFG.DET_WEIGHTS}")
    if not CFG.SAM2_CKPT.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {CFG.SAM2_CKPT}")
    
    # Load detection model
    print(f"[Init] Loading RT-DETR model...")
    det_model = YOLO(str(CFG.DET_WEIGHTS))
    
    # Load SAM2 (loaded but only used at the end)
    print(f"[Init] Loading SAM2...")
    autocast_dtype = pick_autocast_dtype()
    sam2_predictor = load_sam2(CFG.SAM2_CKPT, CFG.SAM2_CFG, device)
    
    # Initialize RealSense
    print("[Init] Starting RealSense camera...")
    pipeline, align, depth_scale, depth_filters = init_realsense(CFG)
    
    # Initialize tracker
    tracker = MultiFrameTracker(iou_threshold=CFG.TRACK_IOU_THRESHOLD)
    
    # Get camera intrinsics (for point cloud generation)
    intrinsics = get_camera_intrinsics(pipeline)
    print(f"[Init] Camera intrinsics: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}, "
          f"cx={intrinsics['cx']:.1f}, cy={intrinsics['cy']:.1f}")
    
    # RealSense pointcloud object (handles lens distortion properly!)
    rs_pointcloud = rs.pointcloud()
    
    # Store frames for later SAM processing (color and depth)
    # Now stores: (frame_idx, cropped_color, cropped_depth, full_depth_np, depth_frame)
    stored_frames: list[tuple[int, np.ndarray, np.ndarray | None, np.ndarray | None, rs.depth_frame | None]] = []
    
    print(f"\n[Capture] {CFG.NUM_FRAMES} frames, conf={CFG.CONF}, min_avg_conf={CFG.MIN_AVG_CONF}")
    print("-" * 60)
    
    captured_count = 0
    try:
        for frame_idx in range(CFG.NUM_FRAMES):
            # Capture frame (color + depth with filters applied)
            color_bgr, depth_raw, depth_frame = get_color_and_depth_frame(pipeline, align, depth_filters)
            if color_bgr is None:
                print(f"[Warning] Frame {frame_idx}: Failed, skipping...")
                continue
            
            # Crop to specified region
            cropped = crop_image(color_bgr, crop)
            cropped_depth = crop_image(depth_raw, crop) if depth_raw is not None else None
            
            # Store frame for later SAM processing (with cropped and full depth + depth_frame)
            stored_frames.append((
                frame_idx,
                cropped.copy(),
                cropped_depth.copy() if cropped_depth is not None else None,
                depth_raw.copy() if depth_raw is not None else None,  # Full depth array
                depth_frame,  # RealSense depth frame for rs.pointcloud()
            ))
            
            # Run detection on cropped image
            t0 = time.time()
            dets = run_detection(det_model, cropped, CFG.CONF, CFG.IOU, CFG.MAX_DET, device)
            det_time_ms = (time.time() - t0) * 1000
            
            # Filter out background detections
            dets = filter_background(dets, cropped.shape[:2], CFG)
            
            # Apply custom NMS to remove duplicate boxes on same object
            if CFG.CUSTOM_NMS:
                dets = custom_nms(dets, CFG.CUSTOM_NMS_IOU)
            
            # Update tracker with detections
            tracker.update(frame_idx, dets)

            # Draw detections and save
            vis_det = draw_detections(cropped, dets)
            det_path = session_dir / f"frame_{frame_idx:04d}_det.png"
            cv2.imwrite(str(det_path), vis_det)
            
            # Print progress (no SAM timing since we defer it)
            det_names = [d["name"] for d in dets[:5]]
            if len(dets) > 5:
                det_names.append(f"...+{len(dets)-5}")
            print(f"[{frame_idx:3d}] {len(dets):3d} dets | det={det_time_ms:5.1f}ms | tracks={len(tracker.tracks)} | {det_names}")
            
            captured_count += 1
            
            # Delay between frames
            if frame_idx < CFG.NUM_FRAMES - 1:
                time.sleep(CFG.FRAME_DELAY)
    
    except KeyboardInterrupt:
        print("\n[Interrupt] Stopping capture...")
    
    finally:
        pipeline.stop()
    
    print("-" * 60)
    
    # Get confirmed tracks (average confidence > threshold)
    confirmed_tracks = tracker.get_confirmed_tracks(captured_count, CFG.MIN_AVG_CONF)
    
    print(f"[Tracking] Total tracks: {len(tracker.tracks)}, Confirmed (avg_conf >= {CFG.MIN_AVG_CONF}): {len(confirmed_tracks)}")
    
    # Print track summary
    for track in tracker.tracks:
        avg_conf = track.get_avg_confidence(captured_count)
        num_dets = len(track.detections)
        _, best_conf, best_name = track.get_best_detection()
        status = "✓" if avg_conf >= CFG.MIN_AVG_CONF else "✗"
        print(f"  {status} Track {track.track_id}: avg_conf={avg_conf:.3f} ({num_dets}/{captured_count} frames), best={best_name} ({best_conf:.2f})")
    
    # Run SAM only on confirmed tracks using the last frame
    if confirmed_tracks and stored_frames:
        print("-" * 60)
        print(f"[SAM] Running segmentation on {len(confirmed_tracks)} confirmed objects...")
        
        # Use the last captured frame for SAM
        last_frame_idx, last_frame, last_depth, full_depth, last_depth_frame = stored_frames[-1]
        
        # Build detection list from confirmed tracks (use average bbox and best detection info)
        final_dets = []
        for track in confirmed_tracks:
            avg_bbox = track.get_avg_bbox()
            _, best_conf, best_name = track.get_best_detection()
            avg_conf = track.get_avg_confidence(captured_count)
            final_dets.append({
                "xyxy": avg_bbox,
                "conf": avg_conf,  # Use average confidence for display
                "name": best_name,
                "track_id": track.track_id,
                "idx": len(final_dets),
            })
        
        # Run SAM2 on confirmed detections
        t1 = time.time()
        masks = run_sam2_on_boxes(sam2_predictor, last_frame, final_dets, device, autocast_dtype)
        seg_time_ms = (time.time() - t1) * 1000
        
        print(f"[SAM] Segmentation completed in {seg_time_ms:.1f}ms ({len(masks)} masks)")
        
        # Filter overlapping segmentations by depth
        if last_depth is not None and len(masks) > 1:
            print(f"[Depth] Filtering overlapping segmentations...")
            original_count = len(masks)
            final_dets, masks = filter_overlapping_by_depth(
                final_dets,
                masks,
                last_depth,
                depth_scale,
                CFG.DEPTH_OVERLAP_THRESHOLD,
                CFG.DEPTH_DIFF_THRESHOLD_M,
            )
            if len(masks) < original_count:
                print(f"[Depth] Filtered {original_count} -> {len(masks)} masks")
            else:
                print(f"[Depth] No overlapping same-depth masks found")
        elif last_depth is None:
            print("[Depth] No depth data available, skipping depth filtering")
        
        # Save segmentation result for the last frame
        vis_seg = draw_segmentations(last_frame, final_dets, masks, CFG.MASK_ALPHA)
        seg_path = session_dir / f"frame_{last_frame_idx:04d}_seg.png"
        cv2.imwrite(str(seg_path), vis_seg)
        
        # Also save a final summary image
        final_path = session_dir / "final_segmentation.png"
        cv2.imwrite(str(final_path), vis_seg)
        print(f"[SAM] Saved final segmentation: {final_path}")
        
        # Visualize point cloud with segmented objects in meshcat
        if last_depth_frame is not None and masks:
            print("-" * 60)
            print("[Meshcat] Generating 3D point cloud visualization...")
            vis = visualize_pointcloud_meshcat(
                depth_frame=last_depth_frame,
                rs_pc=rs_pointcloud,
                masks=masks,
                dets=final_dets,
                crop=crop,
                full_shape=(CFG.HEIGHT, CFG.WIDTH),
            )
            print("-" * 60)
            input("[Meshcat] Press Enter to close visualization and exit...")
        else:
            print("[Meshcat] Skipped: No depth data or masks available")
    else:
        print("[SAM] No confirmed objects to segment.")
    
    print("-" * 60)
    print(f"[Done] Saved to: {session_dir}")


if __name__ == "__main__":
    main()
