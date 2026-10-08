"""
Point Cloud Capture for Grasp Pose Sampling

This module provides modular point cloud generation from RealSense cameras,
optimized for robotic grasping applications.

Camera/filter parameters match those used in ``cluster.py`` and
``grasp_detection_server.py`` (imported when available) so that point clouds
produced here are directly comparable to the live detection server.

Key behaviors:
1. SHARED PARAMETERS: 1280x720@30fps, 0.1mm depth units, max laser power
2. SHARED FILTER CHAIN: decimation + temporal (matches cluster.py)
3. NO ALIGN: point cloud stays in depth-camera frame (matches cluster.py)
4. HIGH ACCURACY visual preset with auto-exposure on depth + color
5. MULTI-FRAME AVERAGING (optional): smooth post-filter depth for noise viz
6. STATISTICAL OUTLIER REMOVAL: clean noise for grasp sampling
7. CONSISTENCY FILTERING: reject pixels with high temporal variance

Usage:
    from utils.realsense_pc import RealsensePointCloud

    pc_capture = RealsensePointCloud()
    points_camera, points_base, color_image = pc_capture.capture()
    pc_capture.close()
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.spatial import cKDTree

# Pull camera/filter parameters from cluster.py so this module stays
# bit-identical with grasp_detection_server.py. Fall back to the same
# literal values if cluster.py is not importable (e.g. when this module is
# used standalone outside the project root).
try:
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from cluster import (  # noqa: E402
        CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, DEPTH_UNITS, LASER_POWER,
        FILTER_DECIMATION_ENABLED, FILTER_DISPARITY_ENABLED, FILTER_SPATIAL_ENABLED,
        FILTER_TEMPORAL_ENABLED, FILTER_HOLE_FILLING_ENABLED,
        FILTER_DECIMATION_MAGNITUDE, FILTER_SPATIAL_ALPHA, FILTER_SPATIAL_DELTA,
        FILTER_TEMPORAL_ALPHA, FILTER_TEMPORAL_DELTA,
    )
except Exception:  # pragma: no cover - fallback for standalone use
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30
    DEPTH_UNITS = 0.0001
    LASER_POWER = 360
    FILTER_DECIMATION_ENABLED = True
    FILTER_DISPARITY_ENABLED = False
    FILTER_SPATIAL_ENABLED = False
    FILTER_TEMPORAL_ENABLED = True
    FILTER_HOLE_FILLING_ENABLED = False
    FILTER_DECIMATION_MAGNITUDE = 1
    FILTER_SPATIAL_ALPHA = 0.25
    FILTER_SPATIAL_DELTA = 20
    FILTER_TEMPORAL_ALPHA = 0.2
    FILTER_TEMPORAL_DELTA = 20


@dataclass
class RealsenseConfig:
    """Configuration for RealSense point cloud capture.

    Defaults are sourced from ``cluster.py`` so this module behaves
    identically to ``grasp_detection_server.py``.
    """
    # --- Camera settings (match cluster.py) ---
    width: int = CAMERA_WIDTH
    height: int = CAMERA_HEIGHT
    fps: int = CAMERA_FPS
    depth_units: float = DEPTH_UNITS      # 0.1mm precision
    laser_power: int = LASER_POWER        # Max power (150-360 for D455)

    # --- Auto-exposure (match cluster.py) ---
    enable_auto_exposure_depth: bool = True
    enable_auto_exposure_color: bool = True

    # --- Capture settings ---
    num_frames_avg: int = 15              # Frames averaged for noise stats
    warmup_frames: int = 30               # Match cluster.py warmup

    # --- Filter chain (match cluster.py) ---
    filter_decimation_enabled: bool = FILTER_DECIMATION_ENABLED
    filter_disparity_enabled: bool = FILTER_DISPARITY_ENABLED
    filter_spatial_enabled: bool = FILTER_SPATIAL_ENABLED
    filter_temporal_enabled: bool = FILTER_TEMPORAL_ENABLED
    filter_hole_filling_enabled: bool = FILTER_HOLE_FILLING_ENABLED

    # --- Filter parameters (match cluster.py) ---
    filter_decimation_magnitude: int = FILTER_DECIMATION_MAGNITUDE
    filter_spatial_alpha: float = FILTER_SPATIAL_ALPHA
    filter_spatial_delta: int = FILTER_SPATIAL_DELTA
    filter_temporal_alpha: float = FILTER_TEMPORAL_ALPHA
    filter_temporal_delta: int = FILTER_TEMPORAL_DELTA

    # --- Outlier removal settings ---
    outlier_mean_k: int = 50              # k-neighbors for mean distance
    outlier_std_ratio: float = 1.0        # Threshold multiplier

    # --- Depth range (meters) ---
    depth_min: float = 0.3                # D455 minimum reliable depth
    depth_max: float = 2.0                # Limit to workspace

    # --- Consistency threshold ---
    consistency_threshold: float = 0.015  # 1.5% relative noise threshold

    # --- Transform to robot base frame (default identity) ---
    T_base_camera: Optional[np.ndarray] = None


# Default camera-to-robot-base transform
DEFAULT_T_BASE_CAMERA = np.array([
    [ -0.99870602,  -0.05068679,   0.00413055,   0.46783590],
    [ -0.05061085,   0.99857739,   0.01678431,   0.11403400],
    [ -0.00497542,   0.01655354,  -0.99985062,   1.02680489],
    [  0.00000000,   0.00000000,   0.00000000,   1.00000000],
])


def statistical_outlier_removal(points: np.ndarray, mean_k: int = 50, 
                                std_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove statistical outliers based on mean distance to k-nearest neighbors.
    Points with distance > mean + std_ratio * std are considered outliers.
    
    This is CRITICAL for grasp pose sampling - outliers cause invalid grasps.
    
    Args:
        points: Nx3 point cloud
        mean_k: Number of neighbors for mean distance calculation
        std_ratio: Points beyond mean + ratio*std are removed
        
    Returns:
        Tuple of (filtered_points, inlier_mask)
    """
    if len(points) < mean_k + 1:
        return points, np.ones(len(points), dtype=bool)
    
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=mean_k + 1)  # +1 because point is its own neighbor
    mean_distances = distances[:, 1:].mean(axis=1)   # Exclude self-distance
    
    global_mean = mean_distances.mean()
    global_std = mean_distances.std()
    threshold = global_mean + std_ratio * global_std
    
    inlier_mask = mean_distances < threshold
    return points[inlier_mask], inlier_mask


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Transform points using a 4x4 transformation matrix.
    
    Args:
        points: Nx3 point cloud
        T: 4x4 transformation matrix
        
    Returns:
        Transformed Nx3 point cloud
    """
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ points.T).T + t


class RealsensePointCloud:
    """
    High-precision point cloud capture from RealSense cameras.
    
    Optimized for robotic grasping with multi-frame averaging,
    consistency filtering, and statistical outlier removal.
    """
    
    def __init__(self, config: Optional[RealsenseConfig] = None, verbose: bool = True):
        """
        Initialize RealSense camera for point cloud capture.
        
        Args:
            config: Configuration object. Uses defaults if None.
            verbose: Print status messages
        """
        self.config = config or RealsenseConfig()
        self.verbose = verbose
        
        # Set default transform if not provided
        if self.config.T_base_camera is None:
            self.config.T_base_camera = DEFAULT_T_BASE_CAMERA
        
        # Pipeline state
        self.pipe = None
        self.profile = None
        self.depth_sensor = None
        self.depth_scale = None

        # Filters (match cluster.py filter chain)
        self.decimation = None
        self.spatial = None
        self.temporal = None
        self.hole_filling = None
        self.depth_to_disparity = None
        self.disparity_to_depth = None
        self.pc = None

        self._initialize_camera()
    
    def _log(self, msg: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(msg)
    
    def _get_device(self):
        """Get first available RealSense device."""
        ctx = rs.context()
        devs = ctx.query_devices()
        if len(devs) == 0:
            raise RuntimeError("No RealSense devices found")
        return devs[0]
    
    def _initialize_camera(self):
        """Initialize camera with retry/reset logic."""
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, self.config.width, self.config.height, 
                         rs.format.z16, self.config.fps)
        cfg.enable_stream(rs.stream.color, self.config.width, self.config.height, 
                         rs.format.bgr8, self.config.fps)
        
        self._log("=" * 70)
        self._log("HIGH-PRECISION Point Cloud Capture for Grasp Pose Sampling")
        self._log("=" * 70)
        self._log(f"Resolution: {self.config.width}x{self.config.height} @ {self.config.fps}fps")
        self._log(f"Multi-frame averaging: {self.config.num_frames_avg} frames")
        
        while True:
            try:
                self.pipe = rs.pipeline()
                self.profile = self.pipe.start(cfg)
                self.depth_sensor = self.profile.get_device().first_depth_sensor()

                # High accuracy preset
                self.depth_sensor.set_option(rs.option.visual_preset,
                                            rs.rs400_visual_preset.high_accuracy)
                self._log("✓ High accuracy preset enabled")

                # Fine depth units
                self.depth_sensor.set_option(rs.option.depth_units, self.config.depth_units)
                self._log(f"✓ Depth units: {self.config.depth_units*1000:.1f}mm")

                # Auto-exposure on depth sensor (matches cluster.py)
                if self.config.enable_auto_exposure_depth:
                    try:
                        self.depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
                        self._log("✓ Depth auto-exposure enabled")
                    except Exception as e:
                        self._log(f"[warn] could not enable depth auto-exposure: {e}")

                # Laser power (match cluster.py: clamp to device range, tolerate failure)
                if self.depth_sensor.supports(rs.option.laser_power):
                    try:
                        lp_range = self.depth_sensor.get_option_range(rs.option.laser_power)
                        lp_value = float(min(max(self.config.laser_power, lp_range.min),
                                             lp_range.max))
                        self.depth_sensor.set_option(rs.option.laser_power, lp_value)
                        self._log(f"✓ Laser power: {lp_value:.0f}mW "
                                  f"(range {lp_range.min:.0f}-{lp_range.max:.0f}mW)")
                    except Exception as e:
                        self._log(f"[warn] could not set laser_power: {e}")
                else:
                    self._log("[info] depth sensor does not support laser_power option")

                # Enable emitter
                if self.depth_sensor.supports(rs.option.emitter_enabled):
                    self.depth_sensor.set_option(rs.option.emitter_enabled, 1)
                    self._log("✓ IR emitter enabled")

                # Color sensor auto-exposure (matches cluster.py)
                if self.config.enable_auto_exposure_color:
                    for sensor in self.profile.get_device().sensors:
                        try:
                            if sensor.get_info(rs.camera_info.name) == 'RGB Camera':
                                sensor.set_option(rs.option.enable_auto_exposure, 1)
                                self._log("✓ Color auto-exposure enabled")
                                break
                        except Exception:
                            continue

                self.depth_scale = self.depth_sensor.get_depth_scale()
                self._log(f"✓ Depth scale: {self.depth_scale}")
                self._log("Camera initialized successfully.")
                break

            except RuntimeError as e:
                msg = str(e)
                self._log(f"Initialization error: {msg}")

                recoverable_errors = [
                    "get_xu(ctrl=1)",
                    "Device or resource busy",
                    "failed to set power state",
                    "Couldn't resolve requests",
                ]

                if any(err in msg for err in recoverable_errors):
                    self._log("Camera busy or in bad state, doing hardware reset...")

                    if self.pipe:
                        try:
                            self.pipe.stop()
                        except:
                            pass

                    try:
                        dev = self._get_device()
                        dev.hardware_reset()
                        self._log("Reset triggered, waiting 2s for re-enumeration...")
                        time.sleep(2)
                    except Exception as reset_err:
                        self._log(f"Failed to reset device: {reset_err}")
                        time.sleep(2)
                else:
                    raise RuntimeError(f"Fatal error initializing camera: {e}")

        # Setup filter chain (mirror cluster.py: decimation + temporal only by default)
        if self.config.filter_decimation_enabled:
            self.decimation = rs.decimation_filter()
            self.decimation.set_option(rs.option.filter_magnitude,
                                       self.config.filter_decimation_magnitude)
            self._log(f"✓ Decimation filter: magnitude={self.config.filter_decimation_magnitude}")
        if self.config.filter_spatial_enabled:
            self.spatial = rs.spatial_filter()
            self.spatial.set_option(rs.option.filter_smooth_alpha,
                                    self.config.filter_spatial_alpha)
            self.spatial.set_option(rs.option.filter_smooth_delta,
                                    self.config.filter_spatial_delta)
            self._log(f"✓ Spatial filter: alpha={self.config.filter_spatial_alpha}, "
                      f"delta={self.config.filter_spatial_delta}")
        if self.config.filter_temporal_enabled:
            self.temporal = rs.temporal_filter()
            self.temporal.set_option(rs.option.filter_smooth_alpha,
                                     self.config.filter_temporal_alpha)
            self.temporal.set_option(rs.option.filter_smooth_delta,
                                     self.config.filter_temporal_delta)
            self._log(f"✓ Temporal filter: alpha={self.config.filter_temporal_alpha}, "
                      f"delta={self.config.filter_temporal_delta}")
        if self.config.filter_hole_filling_enabled:
            self.hole_filling = rs.hole_filling_filter()
            self._log("✓ Hole filling filter enabled")
        if self.config.filter_disparity_enabled:
            self.depth_to_disparity = rs.disparity_transform(True)
            self.disparity_to_depth = rs.disparity_transform(False)
            self._log("✓ Disparity transform enabled")

        self.pc = rs.pointcloud()
    
    def capture(self, save_debug_images: bool = True, 
                output_dir: str = "data") -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Capture a high-precision point cloud with multi-frame averaging.
        
        Args:
            save_debug_images: Save color, depth, and noise images
            output_dir: Directory for debug images
            
        Returns:
            Tuple of (points_camera, points_base, color_image)
            - points_camera: Nx3 points in camera frame
            - points_base: Nx3 points in robot base frame
            - color_image: HxWx3 BGR color image (or None)
        """
        MAX_CAPTURE_RETRIES = 3
        depth_frames_raw = []
        color_frame_final = None
        depth_frame = None
        depth = None

        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, self.config.width, self.config.height,
                         rs.format.z16, self.config.fps)
        cfg.enable_stream(rs.stream.color, self.config.width, self.config.height,
                         rs.format.bgr8, self.config.fps)

        for capture_attempt in range(MAX_CAPTURE_RETRIES):
            try:
                # Warm-up (matches cluster.py: 30 frames by default)
                self._log(f"\nWarming up camera ({self.config.warmup_frames} frames)...")
                for i in range(self.config.warmup_frames):
                    self.pipe.wait_for_frames()
                    if (i + 1) % 20 == 0:
                        self._log(f"  {i + 1}/{self.config.warmup_frames}")

                # Multi-frame capture through the same filter chain cluster.py uses.
                # Each frame is passed through (decimation, [disparity], [spatial],
                # temporal, [disparity], [hole_filling]) — exactly the chain in
                # grasp_detection_server.py._run_detection_with_pipeline.
                self._log(f"\nCapturing {self.config.num_frames_avg} frames "
                          f"(filter chain matches cluster.py)...")
                depth_frames_raw = []
                color_frame_final = None
                depth = None

                for i in range(self.config.num_frames_avg):
                    frames = self.pipe.wait_for_frames()

                    depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    if not depth_frame:
                        continue

                    depth = depth_frame
                    if self.decimation:
                        depth = self.decimation.process(depth)
                    if self.depth_to_disparity:
                        depth = self.depth_to_disparity.process(depth)
                    if self.spatial:
                        depth = self.spatial.process(depth)
                    if self.temporal:
                        depth = self.temporal.process(depth)
                    if self.disparity_to_depth:
                        depth = self.disparity_to_depth.process(depth)
                    if self.hole_filling:
                        depth = self.hole_filling.process(depth)

                    depth_data = np.asanyarray(depth.get_data()).astype(np.float32)
                    depth_frames_raw.append(depth_data)
                    if color_frame:
                        color_frame_final = color_frame

                    if (i + 1) % 5 == 0:
                        self._log(f"  Captured frame {i + 1}/{self.config.num_frames_avg}")

                if not depth_frames_raw or depth is None:
                    raise RuntimeError("No depth frames received from pipeline")

                # Per-pixel temporal stats (for noise visualization / consistency)
                self._log("\nComputing depth statistics with consistency filtering...")
                depth_stack = np.stack(depth_frames_raw, axis=0)
                depth_median = np.median(depth_stack, axis=0)
                depth_std = np.std(depth_stack, axis=0)
                depth_mean = np.mean(depth_stack, axis=0)

                consistency_mask = ((depth_mean > 0) &
                                   (depth_std < self.config.consistency_threshold * depth_mean))
                inconsistent_count = (~consistency_mask & (depth_mean > 0)).sum()
                total_valid = max(int((depth_mean > 0).sum()), 1)
                self._log(f"  Rejected {inconsistent_count:,} inconsistent pixels "
                         f"({100*inconsistent_count/total_valid:.2f}%)")

                depth_for_viz = depth_median.copy()
                depth_for_viz[~consistency_mask] = 0
                depth_for_viz_u16 = depth_for_viz.astype(np.uint16)

                # Save debug images
                color_image = None
                if save_debug_images:
                    os.makedirs(output_dir, exist_ok=True)

                    if color_frame_final:
                        color_image = np.asanyarray(color_frame_final.get_data())
                        cv2.imwrite(os.path.join(output_dir, "realsense_color.png"), color_image)

                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(depth_for_viz_u16, alpha=0.03), cv2.COLORMAP_JET
                    )
                    cv2.imwrite(os.path.join(output_dir, "realsense_depth.png"), depth_colormap)

                    depth_std_pct = np.zeros_like(depth_std)
                    valid = depth_mean > 0
                    depth_std_pct[valid] = depth_std[valid] / depth_mean[valid] * 100
                    depth_std_norm = (depth_std_pct * 25.5).clip(0, 255).astype(np.uint8)
                    depth_std_colormap = cv2.applyColorMap(depth_std_norm, cv2.COLORMAP_HOT)
                    cv2.imwrite(os.path.join(output_dir, "realsense_depth_noise.png"), depth_std_colormap)
                    self._log(f"  Saved debug images to {output_dir}/")
                elif color_frame_final:
                    color_image = np.asanyarray(color_frame_final.get_data())

                self._log("Frame capture successful.")
                break

            except RuntimeError as e:
                msg = str(e)
                self._log(f"\nCapture error: {msg}")

                recoverable_errors = [
                    "Frame didn't arrive",
                    "get_xu(ctrl=1)",
                    "Device or resource busy",
                    "failed to set power state",
                ]

                if any(err in msg for err in recoverable_errors) and capture_attempt < MAX_CAPTURE_RETRIES - 1:
                    self._log(f"Attempting recovery ({capture_attempt + 1}/{MAX_CAPTURE_RETRIES})...")

                    try:
                        self.pipe.stop()
                    except:
                        pass

                    try:
                        dev = self._get_device()
                        dev.hardware_reset()
                        self._log("Hardware reset triggered, waiting 3s...")
                        time.sleep(3)
                    except Exception as reset_err:
                        self._log(f"Reset failed: {reset_err}")
                        time.sleep(2)

                    try:
                        self.pipe = rs.pipeline()
                        self.profile = self.pipe.start(cfg)
                        self.depth_sensor = self.profile.get_device().first_depth_sensor()
                        self.depth_sensor.set_option(rs.option.visual_preset,
                                                    rs.rs400_visual_preset.high_accuracy)
                        self.depth_sensor.set_option(rs.option.depth_units, self.config.depth_units)
                        if self.depth_sensor.supports(rs.option.laser_power):
                            try:
                                lp_range = self.depth_sensor.get_option_range(rs.option.laser_power)
                                lp_value = float(min(max(self.config.laser_power, lp_range.min),
                                                     lp_range.max))
                                self.depth_sensor.set_option(rs.option.laser_power, lp_value)
                            except Exception:
                                pass
                        self._log("Pipeline restarted.")
                    except Exception as restart_err:
                        self._log(f"Restart failed: {restart_err}")
                        time.sleep(2)
                else:
                    raise RuntimeError(f"Fatal capture error after {capture_attempt + 1} attempts: {e}")

        if not depth_frames_raw or depth is None:
            raise RuntimeError("No frames captured!")
        
        # Generate point cloud
        self._log("Computing point cloud with distortion correction...")
        points = self.pc.calculate(depth)
        vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        
        # Filter valid points
        valid_mask = ((vtx[:, 2] > self.config.depth_min) & 
                     (vtx[:, 2] < self.config.depth_max) & 
                     np.isfinite(vtx[:, 2]))
        vtx_valid = vtx[valid_mask]
        self._log(f"  Valid points (depth {self.config.depth_min}-{self.config.depth_max}m): "
                  f"{vtx_valid.shape[0]:,}")
        
        # Statistical outlier removal
        self._log(f"\nRemoving statistical outliers (k={self.config.outlier_mean_k}, "
                  f"std_ratio={self.config.outlier_std_ratio})...")
        vtx_clean, inlier_mask = statistical_outlier_removal(
            vtx_valid,
            mean_k=self.config.outlier_mean_k,
            std_ratio=self.config.outlier_std_ratio
        )
        outlier_count = (~inlier_mask).sum()
        self._log(f"  Removed {outlier_count:,} outliers ({100*outlier_count/len(vtx_valid):.2f}%)")
        self._log(f"  Final point cloud: {vtx_clean.shape[0]:,} points")
        
        # Print statistics
        self._log(f"\n{'='*70}")
        self._log("Point cloud statistics:")
        self._log(f"  X range: [{vtx_clean[:, 0].min():.4f}, {vtx_clean[:, 0].max():.4f}] m")
        self._log(f"  Y range: [{vtx_clean[:, 1].min():.4f}, {vtx_clean[:, 1].max():.4f}] m")
        self._log(f"  Z range: [{vtx_clean[:, 2].min():.4f}, {vtx_clean[:, 2].max():.4f}] m")
        
        # Transform to robot base frame
        self._log("\nTransforming points to robot base frame...")
        vtx_base = transform_points(vtx_clean, self.config.T_base_camera)
        
        self._log(f"Points in base frame:")
        self._log(f"  X range: [{vtx_base[:, 0].min():.4f}, {vtx_base[:, 0].max():.4f}] m")
        self._log(f"  Y range: [{vtx_base[:, 1].min():.4f}, {vtx_base[:, 1].max():.4f}] m")
        self._log(f"  Z range: [{vtx_base[:, 2].min():.4f}, {vtx_base[:, 2].max():.4f}] m")
        
        return vtx_clean, vtx_base, color_image
    
    def close(self):
        """Stop the pipeline and release resources."""
        if self.pipe:
            try:
                self.pipe.stop()
                self._log("Camera pipeline stopped.")
            except:
                pass


def visualize_pointcloud(points: np.ndarray, vis=None, name: str = "pointcloud",
                         color: int = 0x00aaff, point_size: float = 0.0005):
    """
    Visualize point cloud in Meshcat.
    
    Args:
        points: Nx3 point cloud
        vis: Meshcat visualizer (creates new if None)
        name: Object name in visualizer
        color: Point color (hex)
        point_size: Size of points in meters
        
    Returns:
        Meshcat visualizer instance
    """
    import meshcat
    import meshcat.geometry as g
    
    if vis is None:
        vis = meshcat.Visualizer().open()
    
    # Draw coordinate frame
    L = 0.3
    X_axis = np.array([[0, 0, 0], [L, 0, 0]], dtype=np.float32).T
    Y_axis = np.array([[0, 0, 0], [0, L, 0]], dtype=np.float32).T
    Z_axis = np.array([[0, 0, 0], [0, 0, L]], dtype=np.float32).T
    vis["base_frame"]["X"].set_object(g.Line(g.PointsGeometry(X_axis), 
                                             g.LineBasicMaterial(color=0xff0000, linewidth=3)))
    vis["base_frame"]["Y"].set_object(g.Line(g.PointsGeometry(Y_axis), 
                                             g.LineBasicMaterial(color=0x00ff00, linewidth=3)))
    vis["base_frame"]["Z"].set_object(g.Line(g.PointsGeometry(Z_axis), 
                                             g.LineBasicMaterial(color=0x0000ff, linewidth=3)))
    
    # Subsample for visualization
    max_points = 1000000
    if points.shape[0] > max_points:
        indices = np.random.choice(points.shape[0], size=max_points, replace=False)
        vtx_display = points[indices]
    else:
        vtx_display = points
    
    points_3xN = vtx_display.T.astype(np.float32)
    vis[name].set_object(
        g.Points(
            g.PointsGeometry(points_3xN),
            g.PointsMaterial(size=point_size, color=color)
        )
    )
    
    print(f"Visualizing {vtx_display.shape[0]:,} points in Meshcat")
    print(f"Meshcat URL: {vis.url()}")
    
    return vis


# ============================================================================
# Main Script (when run directly)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Running point cloud capture demo...")
    print("=" * 70)
    
    # Create capturer with default config
    pc_capture = RealsensePointCloud()
    
    # Capture point cloud
    points_camera, points_base, color_image = pc_capture.capture()
    
    # Close camera
    pc_capture.close()
    
    # Visualize
    vis = visualize_pointcloud(points_base)
    
    print("\n" + "=" * 70)
    print("CAMERA PARAMETERS (match cluster.py / grasp_detection_server.py):")
    print(f"✓ Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
    print(f"✓ Depth units: {DEPTH_UNITS*1000:.1f}mm precision")
    print(f"✓ Laser power: {LASER_POWER}mW (max for D455)")
    print("✓ High-accuracy visual preset, auto-exposure (depth + color)")
    print(f"✓ Decimation filter: {'on' if FILTER_DECIMATION_ENABLED else 'off'} "
          f"(magnitude={FILTER_DECIMATION_MAGNITUDE})")
    print(f"✓ Temporal filter:   {'on' if FILTER_TEMPORAL_ENABLED else 'off'} "
          f"(alpha={FILTER_TEMPORAL_ALPHA}, delta={FILTER_TEMPORAL_DELTA})")
    print(f"✓ Spatial filter:    {'on' if FILTER_SPATIAL_ENABLED else 'off'}")
    print(f"✓ Disparity xform:   {'on' if FILTER_DISPARITY_ENABLED else 'off'}")
    print(f"✓ Hole filling:      {'on' if FILTER_HOLE_FILLING_ENABLED else 'off'}")
    print("✓ Multi-frame averaging with consistency filtering (noise viz)")
    print("✓ Statistical outlier removal")
    print("✓ Transformed to robot base frame")
    print("")
    print("Coordinate frame: Robot base (X=forward, Y=left, Z=up)")
    print("  Origin at robot base, Z=0 is table surface")
    print("")
    print("Debug images saved to data/:")
    print("  - realsense_color.png: RGB image")
    print("  - realsense_depth.png: Depth colormap")
    print("  - realsense_depth_noise.png: Noise heatmap (red = high variance)")
    print("=" * 70)
    input("\nPress Enter to exit...")
