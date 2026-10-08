"""
Cluttered Bin Segmentation using SAM2 Automatic Mask Generator.

This script segments all objects in a cluttered bin without requiring
object detection prompts. SAM2's automatic mask generator samples points
across the image and generates masks for all detected objects.

Key features:
- Automatic segmentation of all objects (no training/prompts needed)
- Depth-based filtering to help separate overlapping objects
- Configurable parameters for different clutter densities
- Saves individual segment masks and combined visualization

Usage:
    python cluttered_bin_segmentation.py
    
    # Or with custom parameters:
    python cluttered_bin_segmentation.py --points-per-side 64 --crop-layers 1
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
import torch


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class ClutteredBinConfig:
    """Configuration for cluttered bin segmentation."""
    
    # Camera settings
    WIDTH: int = 1280
    HEIGHT: int = 720
    FPS: int = 30
    DEPTH_WIDTH: int = 1280
    DEPTH_HEIGHT: int = 720
    DEPTH_UNITS: float = 0.0001
    LASER_POWER: int = 360
    WARMUP_FRAMES: int = 30
    
    # Depth filter settings
    FILTER_TEMPORAL_ENABLED: bool = True
    FILTER_TEMPORAL_ALPHA: float = 0.2
    FILTER_TEMPORAL_DELTA: int = 20
    
    # Crop settings (set to None to use full frame)
    CROP_X1: int | None = 358
    CROP_Y1: int | None = 33
    CROP_X2: int | None = 791
    CROP_Y2: int | None = 465
    CROP_CONFIG_FILE: Path = Path("/home/jeong/perception/app/data/crop_config.json")
    
    # SAM2 Automatic Mask Generator settings
    SAM2_CKPT: Path = Path("/home/jeong/perception/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
    SAM2_CFG: str = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    
    # --- Automatic mask generation parameters ---
    # For cluttered bins, balance between finding objects and avoiding over-segmentation
    POINTS_PER_SIDE: int = 24  # 24x24=576 points (lower = fewer but cleaner masks)
    POINTS_PER_BATCH: int = 64  # GPU memory vs speed tradeoff
    
    # Quality thresholds - HIGHER = fewer but more confident masks
    # These are the most impactful parameters for reducing false positives
    PRED_IOU_THRESH: float = 0.86  # Default 0.8, higher = more confident
    STABILITY_SCORE_THRESH: float = 0.92  # Default 0.95, higher = more stable masks
    
    # NMS threshold - LOWER = more aggressive duplicate removal
    BOX_NMS_THRESH: float = 0.4  # Default 0.7, lower removes more overlaps
    
    # Crop layers for multi-scale - 0 = no crops (avoids duplicate detections)
    CROP_N_LAYERS: int = 0  # Set to 0 to reduce over-segmentation
    CROP_NMS_THRESH: float = 0.7
    
    # Minimum mask region area in SAM (removes tiny artifacts early)
    MIN_MASK_REGION_AREA: int = 500
    
    # Use mask-to-mask refinement for better quality
    USE_M2M: bool = True
    
    # --- Post-processing settings ---
    # Minimum area for a valid segment (pixels) - HIGHER = removes small fragments
    MIN_SEGMENT_AREA: int = 2000  # Increased from 500
    # Maximum area ratio (filter out background/bin)
    MAX_AREA_RATIO: float = 0.4  # Reduced from 0.5
    
    # Overlap-based merging - merge masks that overlap significantly
    MERGE_OVERLAP_THRESH: float = 0.7  # Merge if IoU > this
    
    # Edge filtering - remove masks touching image borders (likely bin edges)
    FILTER_EDGE_MASKS: bool = True
    EDGE_MARGIN: int = 5
    
    # Depth-based filtering
    USE_DEPTH_FILTERING: bool = True
    DEPTH_BACKGROUND_THRESHOLD_M: float = 2.0  # Objects farther than this are background (meters)
    
    # Visualization
    MASK_ALPHA: float = 0.6
    
    # Output
    OUTPUT_DIR: Path = Path("/home/jeong/perception/app/data")


# =============================================================================
# REALSENSE CAMERA (reused from detection_pipeline)
# =============================================================================
def _get_realsense_device():
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        raise RuntimeError("No RealSense devices found")
    return devs[0]


def init_realsense(cfg: ClutteredBinConfig) -> tuple[rs.pipeline, rs.align, float, dict]:
    """Initialize RealSense pipeline."""
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
            
            # Setup temporal filter for depth
            filters = {}
            if cfg.FILTER_TEMPORAL_ENABLED:
                filters['temporal'] = rs.temporal_filter()
                filters['temporal'].set_option(rs.option.filter_smooth_alpha, cfg.FILTER_TEMPORAL_ALPHA)
                filters['temporal'].set_option(rs.option.filter_smooth_delta, cfg.FILTER_TEMPORAL_DELTA)
            
            print(f"[Camera] Warming up ({cfg.WARMUP_FRAMES} frames)...")
            for _ in range(cfg.WARMUP_FRAMES):
                pipeline.wait_for_frames()
            
            print("[Camera] Initialized successfully.")
            return pipeline, align, depth_scale, filters
            
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


def get_rgbd_frame(pipeline: rs.pipeline, align: rs.align, filters: dict):
    """
    Capture aligned color and depth frames.
    
    Returns:
        color: BGR image array
        depth: Depth array (raw units)
        depth_frame: RealSense depth frame object (for pointcloud extraction)
    """
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    
    if not color_frame or not depth_frame:
        return None, None, None
    
    if filters.get('temporal'):
        depth_frame = filters['temporal'].process(depth_frame)
    
    color = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data())
    
    return color, depth, depth_frame


# =============================================================================
# SAM2 AUTOMATIC MASK GENERATOR
# =============================================================================
def pick_autocast_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except: pass
        return torch.float16
    return torch.float32


def load_sam2_auto_mask_generator(cfg: ClutteredBinConfig, device: str):
    """Load SAM2 with automatic mask generator for cluttered scenes."""
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    
    print(f"[SAM2] Loading model from {cfg.SAM2_CKPT}...")
    sam2_model = build_sam2(cfg.SAM2_CFG, str(cfg.SAM2_CKPT), device=device)
    
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=cfg.POINTS_PER_SIDE,
        points_per_batch=cfg.POINTS_PER_BATCH,
        pred_iou_thresh=cfg.PRED_IOU_THRESH,
        stability_score_thresh=cfg.STABILITY_SCORE_THRESH,
        box_nms_thresh=cfg.BOX_NMS_THRESH,
        crop_n_layers=cfg.CROP_N_LAYERS,
        crop_nms_thresh=cfg.CROP_NMS_THRESH,
        min_mask_region_area=cfg.MIN_MASK_REGION_AREA,
        output_mode="binary_mask",
        use_m2m=cfg.USE_M2M,
    )
    print(f"[SAM2] Automatic mask generator ready")
    print(f"  - Points per side: {cfg.POINTS_PER_SIDE} ({cfg.POINTS_PER_SIDE**2} points)")
    print(f"  - Crop layers: {cfg.CROP_N_LAYERS}")
    print(f"  - IoU threshold: {cfg.PRED_IOU_THRESH}")
    print(f"  - Stability threshold: {cfg.STABILITY_SCORE_THRESH}")
    
    return mask_generator


def generate_masks(mask_generator, img_rgb: np.ndarray, device: str) -> list[dict]:
    """Generate all masks for the image using automatic mask generator."""
    autocast_dtype = pick_autocast_dtype()
    
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                masks = mask_generator.generate(img_rgb)
        else:
            masks = mask_generator.generate(img_rgb)
    
    return masks


# =============================================================================
# POST-PROCESSING & FILTERING
# =============================================================================
def filter_masks_by_area(masks: list[dict], img_shape: tuple[int, int], 
                         min_area: int, max_ratio: float) -> list[dict]:
    """Filter masks by area constraints."""
    total_area = img_shape[0] * img_shape[1]
    filtered = []
    
    for mask in masks:
        area = mask['area']
        if area < min_area:
            continue
        if area / total_area > max_ratio:
            continue
        filtered.append(mask)
    
    return filtered


def filter_edge_masks(masks: list[dict], img_shape: tuple[int, int], 
                      margin: int = 5) -> list[dict]:
    """Remove masks that touch image borders (likely bin edges)."""
    h, w = img_shape
    filtered = []
    
    for mask in masks:
        seg = mask['segmentation']
        # Check if mask touches edges
        touches_top = np.any(seg[:margin, :])
        touches_bottom = np.any(seg[-margin:, :])
        touches_left = np.any(seg[:, :margin])
        touches_right = np.any(seg[:, -margin:])
        
        # Count how many edges it touches
        edge_count = sum([touches_top, touches_bottom, touches_left, touches_right])
        
        # Only filter if it touches 2+ edges (likely the bin container)
        if edge_count < 2:
            filtered.append(mask)
    
    return filtered


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def merge_overlapping_masks(masks: list[dict], iou_thresh: float = 0.5) -> list[dict]:
    """Merge masks that overlap significantly, keeping the larger one."""
    if len(masks) <= 1:
        return masks
    
    # Sort by area (largest first)
    masks = sorted(masks, key=lambda m: m['area'], reverse=True)
    
    keep = []
    merged_indices = set()
    
    for i, mask_i in enumerate(masks):
        if i in merged_indices:
            continue
        
        seg_i = mask_i['segmentation']
        
        # Check against all smaller masks
        for j in range(i + 1, len(masks)):
            if j in merged_indices:
                continue
            
            seg_j = masks[j]['segmentation']
            iou = compute_mask_iou(seg_i, seg_j)
            
            if iou > iou_thresh:
                # Merge: keep the larger mask (mask_i), discard smaller (mask_j)
                merged_indices.add(j)
        
        keep.append(mask_i)
    
    return keep


def filter_contained_masks(masks: list[dict], containment_thresh: float = 0.8) -> list[dict]:
    """Remove masks that are mostly contained within another mask."""
    if len(masks) <= 1:
        return masks
    
    # Sort by area (largest first)
    masks = sorted(masks, key=lambda m: m['area'], reverse=True)
    
    keep = []
    removed_indices = set()
    
    for i, mask_i in enumerate(masks):
        if i in removed_indices:
            continue
        
        seg_i = mask_i['segmentation']
        area_i = mask_i['area']
        
        # Check if any smaller mask is contained within this one
        for j in range(i + 1, len(masks)):
            if j in removed_indices:
                continue
            
            seg_j = masks[j]['segmentation']
            area_j = masks[j]['area']
            
            # Check how much of mask_j is inside mask_i
            intersection = np.logical_and(seg_i, seg_j).sum()
            containment = intersection / area_j if area_j > 0 else 0
            
            if containment > containment_thresh:
                # mask_j is mostly inside mask_i - remove it
                removed_indices.add(j)
        
        keep.append(mask_i)
    
    return keep


def filter_masks_by_depth(masks: list[dict], depth: np.ndarray, 
                          depth_scale: float, max_depth_m: float,
                          debug: bool = True) -> list[dict]:
    """Filter out masks that are too far (background)."""
    if depth is None:
        print("    [Depth] No depth data, skipping depth filter")
        return masks
    
    if debug:
        print(f"    [Depth] depth_scale={depth_scale:.6f} m/unit, max_depth={max_depth_m}m")
        print(f"    [Depth] depth array range: {depth.min()} - {depth.max()}")
    
    filtered = []
    for i, mask in enumerate(masks):
        seg = mask['segmentation']
        depth_values = depth[seg > 0]
        valid_depths = depth_values[depth_values > 0]
        
        if len(valid_depths) == 0:
            if debug:
                print(f"    [Depth] Mask {i}: no valid depth values (all zeros)")
            # Keep mask even without depth info
            mask['median_depth_m'] = 0.0
            filtered.append(mask)
            continue
        
        median_depth_m = np.median(valid_depths) * depth_scale
        
        if debug and i < 5:  # Print first 5 for debugging
            print(f"    [Depth] Mask {i}: median_raw={np.median(valid_depths):.0f}, "
                  f"median_m={median_depth_m:.3f}m, valid_count={len(valid_depths)}")
        
        if median_depth_m < max_depth_m:
            mask['median_depth_m'] = median_depth_m
            filtered.append(mask)
        elif debug and i < 10:
            print(f"    [Depth] Mask {i}: FILTERED OUT (depth {median_depth_m:.2f}m > {max_depth_m}m)")
    
    return filtered


def sort_masks_by_depth(masks: list[dict]) -> list[dict]:
    """Sort masks by depth (closest first)."""
    return sorted(masks, key=lambda m: m.get('median_depth_m', float('inf')))


# =============================================================================
# VISUALIZATION
# =============================================================================
def generate_colors(n: int) -> list[tuple[int, int, int]]:
    """Generate distinct colors for visualization."""
    colors = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        color = cv2.cvtColor(np.uint8([[[hue, 255, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(color[0]), int(color[1]), int(color[2])))
    return colors


def visualize_masks(img_bgr: np.ndarray, masks: list[dict], alpha: float = 0.5) -> np.ndarray:
    """Create visualization with all masks overlaid.
    
    Labels use the ORIGINAL mask index (position in input list), which matches
    the segment_id used in point cloud extraction and grasp detection.
    """
    vis = img_bgr.copy()
    if not masks:
        return vis
    
    colors = generate_colors(len(masks))
    
    # Create (original_idx, mask) pairs, then sort by area for drawing order
    # Largest masks drawn first so smaller ones appear on top
    indexed_masks = [(i, mask) for i, mask in enumerate(masks)]
    sorted_indexed = sorted(indexed_masks, key=lambda x: x[1]['area'], reverse=True)
    
    for orig_idx, mask in sorted_indexed:
        seg = mask['segmentation']
        # Use color based on ORIGINAL index for consistency
        color = colors[orig_idx % len(colors)]
        
        # Apply colored overlay
        overlay = vis.copy()
        overlay[seg > 0] = color
        vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)
        
        # Draw contour
        contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 255), 1)
        
        # Add label at centroid - use ORIGINAL index to match segment_id
        if len(contours) > 0:
            M = cv2.moments(contours[0])
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                depth_str = f"{mask.get('median_depth_m', 0)*100:.0f}cm" if 'median_depth_m' in mask else ""
                label = f"[{orig_idx}] {depth_str}"
                cv2.putText(vis, label, (cx - 20, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    
    return vis


def visualize_masks_grid(img_bgr: np.ndarray, masks: list[dict], max_cols: int = 4) -> np.ndarray:
    """Create a grid showing individual masks."""
    if not masks:
        return img_bgr
    
    h, w = img_bgr.shape[:2]
    n = len(masks)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols
    
    # Scale down for grid
    scale = 0.5
    small_h, small_w = int(h * scale), int(w * scale)
    
    grid = np.zeros((rows * small_h, cols * small_w, 3), dtype=np.uint8)
    colors = generate_colors(n)
    
    for i, mask in enumerate(masks):
        row, col = i // cols, i % cols
        
        # Create individual mask visualization
        vis = img_bgr.copy()
        seg = mask['segmentation']
        overlay = vis.copy()
        overlay[seg > 0] = colors[i]
        vis = cv2.addWeighted(vis, 0.5, overlay, 0.5, 0)
        
        # Draw contour
        contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)
        
        # Add label
        depth_str = f"{mask.get('median_depth_m', 0)*100:.0f}cm" if 'median_depth_m' in mask else ""
        iou_str = f"IoU:{mask.get('predicted_iou', 0):.2f}"
        cv2.putText(vis, f"[{i}] {depth_str} {iou_str}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Resize and place in grid
        small = cv2.resize(vis, (small_w, small_h))
        grid[row*small_h:(row+1)*small_h, col*small_w:(col+1)*small_w] = small
    
    return grid


# =============================================================================
# UTILITIES
# =============================================================================
def load_crop_config(cfg: ClutteredBinConfig) -> tuple[int, int, int, int] | None:
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


# =============================================================================
# MAIN PIPELINE
# =============================================================================
class ClutteredBinSegmenter:
    """Segments all objects in a cluttered bin using SAM2 automatic mask generator."""
    
    def __init__(self, config: ClutteredBinConfig | None = None):
        self.cfg = config or ClutteredBinConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._mask_generator = None
        self._pipeline = None
        self._align = None
        self._depth_scale = None
        self._filters = None
    
    def _init_model(self):
        if self._mask_generator is None:
            self._mask_generator = load_sam2_auto_mask_generator(self.cfg, self.device)
    
    def _init_camera(self):
        if self._pipeline is None:
            self._pipeline, self._align, self._depth_scale, self._filters = init_realsense(self.cfg)
    
    def run(self, save_outputs: bool = True) -> dict[str, Any]:
        """
        Run cluttered bin segmentation.
        
        Returns:
            Dictionary with keys:
                - 'masks': List of mask dictionaries
                - 'color_image': BGR color image
                - 'depth_image': Depth array
                - 'session_dir': Path to saved outputs
        """
        print("=" * 60)
        print("Cluttered Bin Segmentation with SAM2")
        print("=" * 60)
        
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        print(f"[Init] Device: {self.device}")
        
        # Initialize
        self._init_model()
        self._init_camera()
        
        # Load crop region
        crop = load_crop_config(self.cfg)
        if crop:
            print(f"[Crop] Region: {crop[0]},{crop[1]} -> {crop[2]},{crop[3]}")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.cfg.OUTPUT_DIR / f"cluttered_seg_{timestamp}"
        if save_outputs:
            session_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Output] {session_dir}")
        
        # Capture frame
        print("\n[Capture] Capturing RGBD frame...")
        color_bgr, depth, _ = get_rgbd_frame(self._pipeline, self._align, self._filters)
        
        if color_bgr is None:
            print("[Error] Failed to capture frame")
            self._pipeline.stop()
            self._pipeline = None
            return None
        
        # Stop camera after capture
        self._pipeline.stop()
        self._pipeline = None
        
        # Apply crop
        color_bgr_cropped = crop_image(color_bgr, crop)
        depth_cropped = crop_image(depth, crop) if depth is not None else None
        
        if save_outputs:
            cv2.imwrite(str(session_dir / "color_input.png"), color_bgr_cropped)
            if depth_cropped is not None:
                depth_vis = cv2.normalize(depth_cropped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                cv2.imwrite(str(session_dir / "depth_input.png"), depth_vis)
        
        # Convert to RGB for SAM2
        color_rgb = cv2.cvtColor(color_bgr_cropped, cv2.COLOR_BGR2RGB)
        
        # Generate masks
        print("-" * 60)
        print(f"[SAM2] Generating masks...")
        t0 = time.time()
        masks = generate_masks(self._mask_generator, color_rgb, self.device)
        gen_time = time.time() - t0
        print(f"[SAM2] Generated {len(masks)} raw masks in {gen_time:.1f}s")
        
        # Post-process masks
        print("-" * 60)
        print("[Filter] Applying filters...")
        
        # 1. Filter by area
        original_count = len(masks)
        masks = filter_masks_by_area(
            masks, color_rgb.shape[:2], 
            self.cfg.MIN_SEGMENT_AREA, 
            self.cfg.MAX_AREA_RATIO
        )
        print(f"  Area filter: {original_count} -> {len(masks)}")
        
        # 2. Filter edge-touching masks (bin container)
        if self.cfg.FILTER_EDGE_MASKS:
            original_count = len(masks)
            masks = filter_edge_masks(masks, color_rgb.shape[:2], self.cfg.EDGE_MARGIN)
            print(f"  Edge filter: {original_count} -> {len(masks)}")
        
        # 3. Merge overlapping masks
        if self.cfg.MERGE_OVERLAP_THRESH > 0:
            original_count = len(masks)
            masks = merge_overlapping_masks(masks, self.cfg.MERGE_OVERLAP_THRESH)
            print(f"  Overlap merge: {original_count} -> {len(masks)}")
        
        # 4. Remove contained masks (smaller masks inside larger ones)
        original_count = len(masks)
        masks = filter_contained_masks(masks, containment_thresh=0.7)
        print(f"  Containment filter: {original_count} -> {len(masks)}")
        
        # Filter by depth
        if self.cfg.USE_DEPTH_FILTERING and depth_cropped is not None:
            original_count = len(masks)
            masks = filter_masks_by_depth(
                masks, depth_cropped, 
                self._depth_scale, 
                self.cfg.DEPTH_BACKGROUND_THRESHOLD_M,
                debug=True
            )
            print(f"  Depth filter: {original_count} -> {len(masks)}")
        else:
            # Assign default depth values when not filtering
            for mask in masks:
                mask['median_depth_m'] = 0.0
            
            # Sort by depth (closest first)
            masks = sort_masks_by_depth(masks)
        
        print(f"[Result] Final: {len(masks)} segments")
        
        # Print segment details
        for i, mask in enumerate(masks):
            area = mask['area']
            iou = mask.get('predicted_iou', 0)
            depth_m = mask.get('median_depth_m', 0)
            print(f"  [{i}] area={area:6d}px, IoU={iou:.2f}, depth={depth_m*100:.1f}cm")
        
        # Save outputs
        if save_outputs:
            print("-" * 60)
            print("[Save] Saving outputs...")
            
            # Combined visualization
            vis_combined = visualize_masks(color_bgr_cropped, masks, self.cfg.MASK_ALPHA)
            cv2.imwrite(str(session_dir / "segmentation_combined.png"), vis_combined)
            
            # Grid visualization
            if len(masks) > 0:
                vis_grid = visualize_masks_grid(color_bgr_cropped, masks)
                cv2.imwrite(str(session_dir / "segmentation_grid.png"), vis_grid)
            
            # Save individual masks
            masks_dir = session_dir / "masks"
            masks_dir.mkdir(exist_ok=True)
            
            for i, mask in enumerate(masks):
                seg = mask['segmentation'].astype(np.uint8) * 255
                cv2.imwrite(str(masks_dir / f"mask_{i:03d}.png"), seg)
            
            # Save mask metadata
            metadata = []
            for i, mask in enumerate(masks):
                metadata.append({
                    'id': i,
                    'area': int(mask['area']),
                    'bbox': mask['bbox'],  # XYWH format
                    'predicted_iou': float(mask.get('predicted_iou', 0)),
                    'stability_score': float(mask.get('stability_score', 0)),
                    'median_depth_m': float(mask.get('median_depth_m', 0)),
                })
            
            with open(session_dir / "segments_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"[Save] Saved to: {session_dir}")
            print(f"  - segmentation_combined.png")
            print(f"  - segmentation_grid.png")
            print(f"  - masks/ ({len(masks)} individual masks)")
            print(f"  - segments_metadata.json")
        
        return {
            'masks': masks,
            'color_image': color_bgr_cropped,
            'depth_image': depth_cropped,
            'depth_scale': self._depth_scale,
            'session_dir': session_dir,
        }


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Cluttered bin segmentation with SAM2')
    
    # SAM parameters
    parser.add_argument('--points-per-side', type=int, default=24,
                       help='Points to sample per side (24=576, 32=1024 points)')
    parser.add_argument('--crop-layers', type=int, default=0,
                       help='Number of crop layers for multi-scale (0=off, 1=4 crops)')
    parser.add_argument('--iou-thresh', type=float, default=0.86,
                       help='Predicted IoU threshold (higher=fewer masks)')
    parser.add_argument('--stability-thresh', type=float, default=0.92,
                       help='Stability score threshold (higher=fewer masks)')
    
    # Post-processing parameters
    parser.add_argument('--min-area', type=int, default=2000,
                       help='Minimum segment area in pixels')
    parser.add_argument('--merge-overlap', type=float, default=0.7,
                       help='Merge masks with IoU above this (0=disable)')
    
    # Depth filtering
    parser.add_argument('--max-depth', type=float, default=2.0,
                       help='Maximum depth in meters (filter background)')
    parser.add_argument('--no-depth-filter', action='store_true',
                       help='Disable depth-based filtering')
    
    # Region of interest
    parser.add_argument('--no-crop', action='store_true',
                       help='Use full frame instead of crop region')
    parser.add_argument('--no-edge-filter', action='store_true',
                       help='Disable edge mask filtering')
    
    # Presets for common scenarios
    parser.add_argument('--aggressive', action='store_true',
                       help='Aggressive filtering (fewer segments)')
    parser.add_argument('--sensitive', action='store_true',
                       help='Sensitive detection (more segments)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Apply presets
    if args.aggressive:
        # Aggressive = fewer segments, higher quality
        args.iou_thresh = 0.90
        args.stability_thresh = 0.95
        args.min_area = 3000
        args.merge_overlap = 0.5
        args.points_per_side = 16
    elif args.sensitive:
        # Sensitive = more segments, catch small objects
        args.iou_thresh = 0.75
        args.stability_thresh = 0.85
        args.min_area = 1000
        args.merge_overlap = 0.8
        args.points_per_side = 32
    
    # Create config with CLI arguments
    cfg = ClutteredBinConfig(
        POINTS_PER_SIDE=args.points_per_side,
        CROP_N_LAYERS=args.crop_layers,
        PRED_IOU_THRESH=args.iou_thresh,
        STABILITY_SCORE_THRESH=args.stability_thresh,
        MIN_SEGMENT_AREA=args.min_area,
        MERGE_OVERLAP_THRESH=args.merge_overlap,
        DEPTH_BACKGROUND_THRESHOLD_M=args.max_depth,
        USE_DEPTH_FILTERING=not args.no_depth_filter,
        FILTER_EDGE_MASKS=not args.no_edge_filter,
    )
    
    if args.no_crop:
        cfg.CROP_X1 = None
        cfg.CROP_Y1 = None
        cfg.CROP_X2 = None
        cfg.CROP_Y2 = None
    
    print(f"[Config] Points: {cfg.POINTS_PER_SIDE}x{cfg.POINTS_PER_SIDE}={cfg.POINTS_PER_SIDE**2}")
    print(f"[Config] IoU thresh: {cfg.PRED_IOU_THRESH}, Stability: {cfg.STABILITY_SCORE_THRESH}")
    print(f"[Config] Min area: {cfg.MIN_SEGMENT_AREA}px, Merge overlap: {cfg.MERGE_OVERLAP_THRESH}")
    
    # Run segmentation
    segmenter = ClutteredBinSegmenter(cfg)
    result = segmenter.run()
    
    if result:
        print("\n" + "=" * 60)
        print(f"SUCCESS: Found {len(result['masks'])} segments")
        print(f"Output: {result['session_dir']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
