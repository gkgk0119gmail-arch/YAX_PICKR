"""
Grasp detection and visualization for segmented point clouds.

Uses detection_pipeline.py for camera capture, RT-DETR detection, and SAM2 segmentation.
This file focuses on:
- Suction grasp candidate detection on segmented pointclouds
- Surface normal estimation and curvature analysis
- Meshcat 3D visualization

Run crop_selector.py first to define the crop region interactively.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import NamedTuple

import meshcat
import meshcat.geometry as geom
import numpy as np
import pyrealsense2 as rs
from scipy.spatial import cKDTree

from detection_pipeline import (
    DetectionPipeline,
    PipelineConfig,
    PipelineResult,
    generate_colors,
    uncrop_mask,
)


# =============================================================================
# SCENE DATA STRUCTURES
# =============================================================================
class SegmentPointCloud(NamedTuple):
    """Point cloud data for a single segmented object."""
    segment_id: int
    name: str
    points: np.ndarray  # Nx3 points in robot base frame
    color: tuple[int, int, int]  # BGR color for visualization


class ScenePointClouds(NamedTuple):
    """Extracted point clouds from a pipeline result."""
    segments: list[SegmentPointCloud]
    background_points: np.ndarray
    full_points: np.ndarray  # All points (for reference)
    valid_mask: np.ndarray   # Mask of valid depth points


# =============================================================================
# REALSENSE DEPTH FILTER CONFIGURATION
# =============================================================================
@dataclass
class DepthFilterConfig:
    """Configuration for RealSense depth post-processing filters.
    
    These filters run on the GPU/DSP and are faster than CPU-based outlier removal.
    """
    # Decimation filter - reduces resolution for faster processing
    decimation_enabled: bool = False
    decimation_magnitude: int = 2  # 2 = half resolution, 3 = third, etc.
    
    # Spatial filter - edge-preserving smoothing (crucial for flat surfaces)
    spatial_enabled: bool = False
    spatial_magnitude: int = 2  # Number of filter iterations (1-5)
    spatial_smooth_alpha: float = 0.5  # Smooth factor (0.25-1.0, lower = smoother)
    spatial_smooth_delta: int = 20  # Step size boundary (1-50 pixels)
    spatial_holes_fill: int = 3  # Hole filling mode (0=off, 1=2px, 2=4px, 3=8px, 4=16px, 5=unlimited)
    
    # Temporal filter - averages depth over more frames with stronger smoothing (for longer, smoother filtering)
    temporal_enabled: bool = True
    temporal_smooth_alpha: float = 0.15  # Lower value = smoother but more delay (0.0-1.0)
    temporal_smooth_delta: int = 50  # Increased threshold for outlier filtering
    temporal_persistence: int = 8  # Higher value = requires more frames to consider a pixel valid (longer memory)
    # Hole filling filter - fills invalid pixels with neighbor data
    hole_filling_enabled: bool = True
    hole_filling_mode: int = 1  # 0=fill from left, 1=farthest from around, 2=nearest from around


DEPTH_FILTER_CFG = DepthFilterConfig()


# =============================================================================
# GRASP DETECTION CONFIGURATION
# =============================================================================
@dataclass
class GraspDetectionConfig:
    """Configuration for suction grasp detection on segmented pointclouds."""
    
    # Workspace filtering (robot base frame)
    plane_normal: tuple[float, float, float] = (0.0, 0.0, -1.0)
    plane_d: float = 0.05
    plane_inlier_thresh: float = 0.004
    z_above_threshold: float = 0.01
    
    # Pre-processing
    voxel_size: float = 0.003
    sor_mean_k: int = 30
    sor_std_ratio: float = 1.0
    
    # Normal estimation
    normal_k_neighbors: int = 50
    camera_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    
    # Curvature filtering (legacy, kept for reference)
    curvature_threshold: float = 0.03
    
    # Plane fit filtering (replaces curvature - more practical for suction cups)
    plane_fit_rmse_threshold: float = 0.01 # 10mm - max RMSE for plane fit
    rim_gap_threshold: float = 0.005  # 5mm - max deviation at suction cup rim
    rim_inner_ratio: float = 0.7  # Check rim points between this ratio and 1.0 of radius
    
    # Grasp constraints
    gravity_vector: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    min_normal_z: float = 0.7
    
    # Scoring weights
    weight_flatness: float = 0.0
    weight_verticality: float = 0.1
    weight_centroid: float = 0.1
    weight_region: float = 0.3
    
    # Region-based scoring (suction cup contact area)
    suction_cup_radius: float = 0.025
    min_region_points: int = 5
    normal_consistency_threshold: float = 0.15
    planarity_threshold: float = 0.003
    
    # Output
    top_k_per_segment: int = 3


GRASP_CFG = GraspDetectionConfig()


# =============================================================================
# GRASP CANDIDATE DATA STRUCTURES
# =============================================================================
@dataclass
class SuctionGraspCandidate:
    """A candidate suction grasp pose."""
    position: np.ndarray
    normal: np.ndarray
    score: float
    
    # Point-level scores
    flatness_score: float = 0.0
    verticality_score: float = 0.0
    centroid_score: float = 0.0
    
    # Region-level scores
    normal_consistency_score: float = 0.0
    planarity_score: float = 0.0
    coverage_score: float = 0.0
    region_score: float = 0.0
    
    # Plane fit metrics (for suction seal quality)
    plane_rmse: float = 0.0  # RMSE of all points to fitted plane (meters)
    rim_max_gap: float = 0.0  # Maximum gap at suction cup rim (meters)
    plane_fit_score: float = 0.0  # Score based on RMSE (1.0 = perfect flat)
    rim_seal_score: float = 0.0  # Score based on rim gap (1.0 = perfect seal)
    
    # Geometric properties
    curvature: float = 0.0
    segment_id: int = -1
    
    def __repr__(self):
        return (f"SuctionGraspCandidate(pos=[{self.position[0]:.3f}, {self.position[1]:.3f}, "
                f"{self.position[2]:.3f}], score={self.score:.3f}, rmse={self.plane_rmse*1000:.2f}mm, "
                f"rim_gap={self.rim_max_gap*1000:.2f}mm)")


@dataclass
class SegmentGraspResult:
    """Result of grasp detection on a single segment."""
    segment_id: int
    candidates: list[SuctionGraspCandidate]
    
    # Intermediate data for visualization
    points_clean: np.ndarray | None = None
    normals: np.ndarray | None = None
    curvatures: np.ndarray | None = None
    valid_mask: np.ndarray | None = None


# =============================================================================
# POINT CLOUD PROCESSING
# =============================================================================
def voxel_grid_downsample(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Downsample point cloud using voxel grid filtering."""
    if len(points) == 0:
        return points, np.array([], dtype=int)
    
    voxel_indices = np.floor(points / voxel_size).astype(int)
    offset = voxel_indices.min(axis=0)
    voxel_indices_offset = voxel_indices - offset
    
    dims = voxel_indices_offset.max(axis=0) + 1
    linear_idx = (voxel_indices_offset[:, 0] * dims[1] * dims[2] + 
                  voxel_indices_offset[:, 1] * dims[2] + 
                  voxel_indices_offset[:, 2])
    
    unique_voxels, inverse = np.unique(linear_idx, return_inverse=True)
    
    downsampled = np.zeros((len(unique_voxels), 3), dtype=points.dtype)
    counts = np.zeros(len(unique_voxels), dtype=int)
    np.add.at(downsampled, inverse, points)
    np.add.at(counts, inverse, 1)
    downsampled /= counts[:, np.newaxis]
    
    return downsampled, inverse


def statistical_outlier_removal(points: np.ndarray, mean_k: int = 50, 
                                std_ratio: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Remove statistical outliers based on mean distance to k-nearest neighbors."""
    if len(points) < mean_k + 1:
        return points, np.ones(len(points), dtype=bool)
    
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=mean_k + 1)
    mean_distances = distances[:, 1:].mean(axis=1)
    
    global_mean = mean_distances.mean()
    global_std = mean_distances.std()
    threshold = global_mean + std_ratio * global_std
    
    inlier_mask = mean_distances < threshold
    return points[inlier_mask], inlier_mask


def estimate_normals_and_curvature(
    points: np.ndarray, 
    k: int = 30,
    orient_toward: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate surface normals and curvature using PCA on local neighborhoods."""
    n_points = len(points)
    if n_points == 0:
        return np.zeros((0, 3)), np.zeros(0)
    
    tree = cKDTree(points)
    k_actual = min(k, n_points)
    _, indices = tree.query(points, k=k_actual)
    
    normals = np.zeros((n_points, 3), dtype=np.float64)
    curvatures = np.zeros(n_points, dtype=np.float64)
    
    for i in range(n_points):
        neighbor_points = points[indices[i]]
        centroid = neighbor_points.mean(axis=0)
        centered = neighbor_points - centroid
        
        cov = centered.T @ centered / (k_actual - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # Sort eigenvalues/vectors in descending order
        sort_idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sort_idx]
        eigvecs = eigvecs[:, sort_idx]
        
        # Normal is eigenvector of smallest eigenvalue
        normals[i] = eigvecs[:, 2]
        
        # Curvature: λ3 / (λ1 + λ2 + λ3)
        eigsum = eigvals.sum()
        curvatures[i] = eigvals[2] / eigsum if eigsum > 1e-10 else 0.0
    
    # Orient normals toward viewpoint
    if orient_toward is not None:
        view_vectors = orient_toward - points
        flip_mask = (normals * view_vectors).sum(axis=1) < 0
        normals[flip_mask] *= -1
    
    return normals, curvatures


def filter_workspace_points(
    points: np.ndarray,
    plane_normal: tuple[float, float, float],
    plane_d: float,
    plane_inlier_thresh: float,
    z_above_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter points to workspace above the table plane."""
    n = np.array(plane_normal, dtype=np.float64)
    a, b, c = n
    d = plane_d
    
    dists = points @ n + d
    inlier_mask = np.abs(dists) < plane_inlier_thresh
    
    if abs(c) > 1e-6:
        z_plane = (-d - a * points[:, 0] - b * points[:, 1]) / c
        height_above = points[:, 2] - z_plane
        above_mask = (height_above >= z_above_threshold) & (~inlier_mask)
    else:
        above_mask = np.zeros(len(points), dtype=bool)
    
    return points[above_mask], above_mask


# =============================================================================
# PLANE FIT SCORING (for suction seal quality)
# =============================================================================
def fit_plane_least_squares(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a plane to points using least squares (SVD).
    
    Returns:
        plane_normal: Unit normal vector of the fitted plane
        d: Plane offset (plane equation: n · x + d = 0)
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    
    # SVD to find plane normal (smallest singular value direction)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    plane_normal = vh[2]  # Normal is the direction of smallest variance
    
    # Plane offset: d = -n · centroid
    d = -np.dot(plane_normal, centroid)
    
    return plane_normal, d


def compute_plane_fit_metrics(
    center_point: np.ndarray,
    neighbor_points: np.ndarray,
    config: GraspDetectionConfig,
) -> tuple[float, float, np.ndarray]:
    """Compute plane fit metrics for a grasp candidate.
    
    Args:
        center_point: The candidate grasp point
        neighbor_points: All points within suction_cup_radius
        config: Configuration with thresholds
        
    Returns:
        rmse: Root mean square error of all points to fitted plane
        rim_max_gap: Maximum deviation of rim points from plane
        plane_normal: Normal vector of fitted plane
    """
    if len(neighbor_points) < 3:
        return float('inf'), float('inf'), np.array([0, 0, 1])
    
    # Fit plane to all neighbor points
    try:
        plane_normal, d = fit_plane_least_squares(neighbor_points)
    except np.linalg.LinAlgError:
        return float('inf'), float('inf'), np.array([0, 0, 1])
    
    # Compute signed distances to plane
    distances = neighbor_points @ plane_normal + d
    
    # Metric 1: RMSE of all points to plane
    rmse = np.sqrt((distances ** 2).mean())
    
    # Metric 2: Check rim points (points near the edge of suction cup)
    # Compute distance from center for each neighbor
    center_dists = np.linalg.norm(neighbor_points - center_point, axis=1)
    
    # Rim points are those between rim_inner_ratio and 1.0 of the radius
    rim_inner = config.suction_cup_radius * config.rim_inner_ratio
    rim_mask = center_dists >= rim_inner
    
    if rim_mask.sum() > 0:
        rim_distances = np.abs(distances[rim_mask])
        rim_max_gap = rim_distances.max()
    else:
        # No rim points found, use max of all distances as fallback
        rim_max_gap = np.abs(distances).max()
    
    return rmse, rim_max_gap, plane_normal


def compute_plane_fit_scores_for_candidates(
    points: np.ndarray,
    valid_indices: np.ndarray,
    config: GraspDetectionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute plane fit scores for all valid candidate points.
    
    Args:
        points: All points in the segment (cleaned)
        valid_indices: Indices of points that passed initial filtering
        config: Configuration with thresholds
        
    Returns:
        plane_rmse: RMSE values for each candidate
        rim_max_gap: Max rim gap values for each candidate
        plane_fit_scores: Scores based on RMSE (0-1)
        rim_seal_scores: Scores based on rim gap (0-1)
        plane_normals: Fitted plane normals for each candidate
    """
    n_valid = len(valid_indices)
    
    if n_valid == 0:
        empty = np.array([])
        return empty, empty, empty, empty, np.zeros((0, 3))
    
    tree = cKDTree(points)
    
    plane_rmse = np.zeros(n_valid)
    rim_max_gap = np.zeros(n_valid)
    plane_fit_scores = np.zeros(n_valid)
    rim_seal_scores = np.zeros(n_valid)
    plane_normals = np.zeros((n_valid, 3))
    
    for i, idx in enumerate(valid_indices):
        center_point = points[idx]
        neighbor_indices = tree.query_ball_point(center_point, config.suction_cup_radius)
        
        if len(neighbor_indices) < config.min_region_points:
            plane_rmse[i] = float('inf')
            rim_max_gap[i] = float('inf')
            plane_fit_scores[i] = 0.0
            rim_seal_scores[i] = 0.0
            plane_normals[i] = np.array([0, 0, 1])
            continue
        
        neighbor_points = points[neighbor_indices]
        
        rmse, max_gap, normal = compute_plane_fit_metrics(
            center_point, neighbor_points, config
        )
        
        plane_rmse[i] = rmse
        rim_max_gap[i] = max_gap
        plane_normals[i] = normal
        
        # Convert to scores (1.0 = perfect, 0.0 = at or beyond threshold)
        # RMSE score: linear falloff from threshold
        if rmse <= config.plane_fit_rmse_threshold:
            plane_fit_scores[i] = 1.0 - (rmse / config.plane_fit_rmse_threshold)
        else:
            plane_fit_scores[i] = 0.0
        
        # Rim seal score: linear falloff from threshold
        if max_gap <= config.rim_gap_threshold:
            rim_seal_scores[i] = 1.0 - (max_gap / config.rim_gap_threshold)
        else:
            rim_seal_scores[i] = 0.0
    
    return plane_rmse, rim_max_gap, plane_fit_scores, rim_seal_scores, plane_normals


# =============================================================================
# REGION-BASED SCORING
# =============================================================================
def compute_region_scores(
    points: np.ndarray,
    normals: np.ndarray,
    curvatures: np.ndarray,
    valid_mask: np.ndarray,
    config: GraspDetectionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute region-based scores for valid grasp points within suction cup contact area."""
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)
    
    if n_valid == 0:
        return np.array([]), np.array([]), np.array([])
    
    tree = cKDTree(points)
    
    normal_consistency_scores = np.zeros(n_valid)
    planarity_scores = np.zeros(n_valid)
    coverage_scores = np.zeros(n_valid)
    
    voxel_area = config.voxel_size ** 2
    suction_area = np.pi * config.suction_cup_radius ** 2
    expected_points = suction_area / voxel_area
    
    for i, idx in enumerate(valid_indices):
        center_point = points[idx]
        neighbor_indices = tree.query_ball_point(center_point, config.suction_cup_radius)
        n_neighbors = len(neighbor_indices)
        
        # Coverage score
        if n_neighbors < config.min_region_points:
            continue
        
        coverage_scores[i] = min(n_neighbors / expected_points, 1.0)
        
        neighbor_points = points[neighbor_indices]
        neighbor_normals = normals[neighbor_indices]
        
        # Normal consistency score
        mean_normal = neighbor_normals.mean(axis=0)
        mean_normal_norm = np.linalg.norm(mean_normal)
        
        if mean_normal_norm < 1e-6:
            normal_consistency_scores[i] = 0.0
        else:
            mean_normal = mean_normal / mean_normal_norm
            alignments = np.abs(neighbor_normals @ mean_normal)
            normal_variance = 1.0 - alignments.mean()
            normal_consistency_scores[i] = max(0.0, 1.0 - normal_variance / config.normal_consistency_threshold)
        
        # Planarity score
        centroid = neighbor_points.mean(axis=0)
        centered = neighbor_points - centroid
        
        try:
            _, s, vh = np.linalg.svd(centered, full_matrices=False)
            plane_normal = vh[2]
            distances_to_plane = np.abs(centered @ plane_normal)
            rms_residual = np.sqrt((distances_to_plane ** 2).mean())
            planarity_scores[i] = max(0.0, 1.0 - rms_residual / config.planarity_threshold)
        except np.linalg.LinAlgError:
            planarity_scores[i] = 0.0
    
    return normal_consistency_scores, planarity_scores, coverage_scores


# =============================================================================
# GRASP DETECTION
# =============================================================================
def detect_grasps_on_segment(
    points: np.ndarray,
    segment_id: int,
    config: GraspDetectionConfig,
    verbose: bool = True,
) -> SegmentGraspResult:
    """Detect suction grasp candidates on a single segmented pointcloud.
    
    Uses plane fitting instead of curvature for filtering:
    - RMSE of local plane fit must be below threshold
    - Maximum rim gap must be below threshold for vacuum seal
    """
    empty_result = SegmentGraspResult(segment_id=segment_id, candidates=[])
    
    if len(points) < 50:
        if verbose:
            print(f"    [Segment {segment_id}] Too few points ({len(points)}), skipping")
        return empty_result
    
    # Workspace filtering
    points_ws, _ = filter_workspace_points(
        points, config.plane_normal, config.plane_d,
        config.plane_inlier_thresh, config.z_above_threshold
    )
    
    if len(points_ws) < 30:
        if verbose:
            print(f"    [Segment {segment_id}] Too few workspace points ({len(points_ws)}), skipping")
        return empty_result
    
    # Voxel downsampling
    points_ds, _ = voxel_grid_downsample(points_ws, config.voxel_size)
    
    if len(points_ds) < 20:
        if verbose:
            print(f"    [Segment {segment_id}] Too few points after downsampling ({len(points_ds)}), skipping")
        return empty_result
    
    # Statistical outlier removal
    points_clean, _ = statistical_outlier_removal(
        points_ds, 
        mean_k=min(config.sor_mean_k, len(points_ds) - 1),
        std_ratio=config.sor_std_ratio
    )
    
    if len(points_clean) < 10:
        if verbose:
            print(f"    [Segment {segment_id}] Too few points after SOR ({len(points_clean)}), skipping")
        return empty_result
    
    # Normal estimation (still needed for orientation and visualization)
    normals, curvatures = estimate_normals_and_curvature(
        points_clean,
        k=min(config.normal_k_neighbors, len(points_clean)),
        orient_toward=config.camera_position
    )
    
    # Initial filtering by normal direction only (surface must face up for suction)
    normal_z = normals[:, 2]
    normal_mask = normal_z > config.min_normal_z
    
    initial_count = normal_mask.sum()
    if initial_count == 0:
        if verbose:
            print(f"    [Segment {segment_id}] No points with upward-facing normals")
        return SegmentGraspResult(
            segment_id=segment_id, candidates=[],
            points_clean=points_clean, normals=normals,
            curvatures=curvatures, valid_mask=normal_mask
        )
    
    # Get initial candidate indices (only normal filtering)
    initial_indices = np.where(normal_mask)[0]
    
    # Compute plane fit metrics for all initial candidates
    plane_rmse, rim_max_gap, plane_fit_scores, rim_seal_scores, plane_normals = \
        compute_plane_fit_scores_for_candidates(points_clean, initial_indices, config)
    
    # No hard rejection - just use scores to rank candidates
    # Report stats for debugging
    if verbose:
        n_good_rmse = (plane_rmse <= config.plane_fit_rmse_threshold).sum()
        n_good_rim = (rim_max_gap <= config.rim_gap_threshold).sum()
        print(f"    [Segment {segment_id}] Plane fit stats: {initial_count} candidates, "
              f"{n_good_rmse} good RMSE (<{config.plane_fit_rmse_threshold*1000:.1f}mm), "
              f"{n_good_rim} good rim (<{config.rim_gap_threshold*1000:.1f}mm)")
        if len(plane_rmse) > 0:
            print(f"      RMSE range: [{plane_rmse.min()*1000:.2f}, {plane_rmse.max()*1000:.2f}] mm")
            print(f"      Rim gap range: [{rim_max_gap.min()*1000:.2f}, {rim_max_gap.max()*1000:.2f}] mm")
    
    # All points that passed normal filtering are valid candidates
    valid_mask = normal_mask.copy()
    valid_count = initial_count
    
    # Extract all candidate points
    valid_points = points_clean[initial_indices]
    valid_normals = normals[initial_indices]
    valid_curvatures = curvatures[initial_indices]
    
    # Plane fit metrics are already computed for all initial candidates
    valid_plane_rmse = plane_rmse
    valid_rim_max_gap = rim_max_gap
    valid_plane_fit_scores = plane_fit_scores
    valid_rim_seal_scores = rim_seal_scores
    
    # Compute other scores for valid points
    # Flatness score (legacy, weight should be 0)
    flatness_scores = np.clip(1.0 - (valid_curvatures / config.curvature_threshold), 0.0, 1.0)
    
    # Verticality score
    gravity = config.gravity_vector / np.linalg.norm(config.gravity_vector)
    verticality_scores = np.clip(valid_normals @ (-gravity), 0.0, 1.0)
    
    # Centroid score
    centroid = valid_points.mean(axis=0)
    dists = np.linalg.norm(valid_points - centroid, axis=1)
    max_dist = dists.max() if dists.max() > 1e-6 else 1.0
    centroid_scores = np.clip(1.0 - (dists / max_dist), 0.0, 1.0)
    
    # Region-based scores (normal consistency, planarity, coverage)
    # Use valid_mask which now contains all normal-filtered points
    normal_consistency_scores, planarity_scores, coverage_scores = compute_region_scores(
        points_clean, normals, curvatures, valid_mask, config
    )
    region_scores = (normal_consistency_scores + planarity_scores + coverage_scores) / 3.0
    
    # Combined score (flatness weight should be 0, using plane fit scores instead)
    total_scores = (
        config.weight_flatness * flatness_scores +
        config.weight_verticality * verticality_scores +
        config.weight_centroid * centroid_scores +
        config.weight_region * region_scores
    )
    
    # Add plane fit scores as bonus (they're already filtered, so these are all good)
    # The plane_fit and rim_seal scores range from 0-1, with higher being better
    plane_bonus = 0.5 * (valid_plane_fit_scores + valid_rim_seal_scores)
    total_scores = total_scores + plane_bonus
    
    # Select top k
    top_k = min(config.top_k_per_segment, len(total_scores))
    top_indices = np.argsort(total_scores)[::-1][:top_k]
    
    candidates = []
    for idx in top_indices:
        candidates.append(SuctionGraspCandidate(
            position=valid_points[idx].copy(),
            normal=valid_normals[idx].copy(),
            score=float(total_scores[idx]),
            flatness_score=float(flatness_scores[idx]),
            verticality_score=float(verticality_scores[idx]),
            centroid_score=float(centroid_scores[idx]),
            normal_consistency_score=float(normal_consistency_scores[idx]),
            planarity_score=float(planarity_scores[idx]),
            coverage_score=float(coverage_scores[idx]),
            region_score=float(region_scores[idx]),
            plane_rmse=float(valid_plane_rmse[idx]),
            rim_max_gap=float(valid_rim_max_gap[idx]),
            plane_fit_score=float(valid_plane_fit_scores[idx]),
            rim_seal_score=float(valid_rim_seal_scores[idx]),
            curvature=float(valid_curvatures[idx]),
            segment_id=segment_id
        ))
    
    if verbose:
        print(f"    [Segment {segment_id}] {len(points):,} pts -> {len(points_clean):,} clean -> "
              f"{valid_count} valid -> {len(candidates)} grasps")
        if candidates:
            best = candidates[0]
            print(f"      Best: score={best.score:.3f} "
                  f"(rmse={best.plane_rmse*1000:.2f}mm, rim_gap={best.rim_max_gap*1000:.2f}mm, "
                  f"region={best.region_score:.2f})")
    
    return SegmentGraspResult(
        segment_id=segment_id, candidates=candidates,
        points_clean=points_clean, normals=normals,
        curvatures=curvatures, valid_mask=valid_mask
    )


# =============================================================================
# COORDINATE TRANSFORMS
# =============================================================================
T_BASE_CAMERA = np.array([
    [-0.99870602, -0.05068679,  0.00413055,  0.46783590],
    [-0.05061085,  0.99857739,  0.01678431,  0.11403400],
    [-0.00497542,  0.01655354, -0.99985062,  1.02680489],
    [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
])


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform Nx3 points using a 4x4 transformation matrix."""
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    points_h = np.hstack([points, ones])
    transformed_h = (T @ points_h.T).T
    return transformed_h[:, :3]


# =============================================================================
# MESHCAT VISUALIZATION
# =============================================================================
def get_rank_color(rank: int, total: int) -> int:
    """Get color based on ranking within segment (White=best -> Dark gray=worst)."""
    if total <= 1:
        return 0xffffff  # White for best/only
    
    # Interpolate from white (255, 255, 255) to dark gray (64, 64, 64)
    t = rank / (total - 1)
    gray = int(255 - t * 191)  # 255 -> 64
    
    return (gray << 16) | (gray << 8) | gray


def visualize_normals_and_curvature(
    vis: meshcat.Visualizer,
    segment_results: list[SegmentGraspResult],
    normal_length: float = 0.015,
    normal_subsample: int = 3,
    curvature_threshold: float = 0.03,
):
    """Visualize surface normals and curvature for all segments."""
    print("-" * 60)
    print("[Normal Viz] Visualizing normals and curvature...")
    
    for result in segment_results:
        if result.points_clean is None or result.normals is None:
            continue
        
        seg_id = result.segment_id
        points = result.points_clean
        normals = result.normals
        curvatures = result.curvatures
        valid_mask = result.valid_mask
        
        n_points = len(points)
        if n_points == 0:
            continue
        
        # Color points by curvature
        c_norm = np.clip(curvatures / curvature_threshold, 0.0, 1.0)
        
        color_bins = [
            (0.0, 0.2, 0x0000ff, "flat"),
            (0.2, 0.4, 0x00ffff, "low_curv"),
            (0.4, 0.6, 0x00ff00, "med_curv"),
            (0.6, 0.8, 0xffff00, "high_curv"),
            (0.8, 1.01, 0xff0000, "curved"),
        ]
        
        for low, high, color, name in color_bins:
            mask = (c_norm >= low) & (c_norm < high)
            if mask.sum() > 0:
                vis["normals_viz"][f"seg_{seg_id}"][f"curv_{name}"].set_object(
                    geom.Points(
                        geom.PointsGeometry(points[mask].T.astype(np.float32)),
                        geom.PointsMaterial(size=0.004, color=color)
                    )
                )
        
        # Highlight valid grasp points
        if valid_mask is not None and valid_mask.sum() > 0:
            vis["normals_viz"][f"seg_{seg_id}"]["valid_points"].set_object(
                geom.Points(
                    geom.PointsGeometry(points[valid_mask].T.astype(np.float32)),
                    geom.PointsMaterial(size=0.006, color=0x00ff00)
                )
            )
        
        # Draw normals as line segments
        indices = np.arange(0, n_points, normal_subsample)
        starts = points[indices]
        ends = starts + normals[indices] * normal_length
        
        segments = np.zeros((len(indices) * 2, 3), dtype=np.float32)
        segments[0::2] = starts
        segments[1::2] = ends
        
        vis["normals_viz"][f"seg_{seg_id}"]["normals_all"].set_object(
            geom.LineSegments(
                geom.PointsGeometry(segments.T),
                geom.LineBasicMaterial(color=0x00ffff, linewidth=1)
            )
        )
        
        # Valid normals in white
        if valid_mask is not None and valid_mask.sum() > 0:
            valid_indices = np.where(valid_mask)[0]
            valid_starts = points[valid_indices]
            valid_ends = valid_starts + normals[valid_indices] * normal_length * 1.5
            
            valid_segments = np.zeros((len(valid_indices) * 2, 3), dtype=np.float32)
            valid_segments[0::2] = valid_starts
            valid_segments[1::2] = valid_ends
            
            vis["normals_viz"][f"seg_{seg_id}"]["normals_valid"].set_object(
                geom.LineSegments(
                    geom.PointsGeometry(valid_segments.T),
                    geom.LineBasicMaterial(color=0xffffff, linewidth=2)
                )
            )
        
        valid_count = valid_mask.sum() if valid_mask is not None else 0
        print(f"  [Segment {seg_id}] {n_points} pts, curvature=[{curvatures.min():.4f}, {curvatures.max():.4f}], "
              f"{valid_count} valid, {len(indices)} normals shown")
    
    print("[Normal Viz] Legend: Blue=flat → Red=curved | White lines=valid normals | Green=valid points")


def visualize_grasp_candidates(
    vis: meshcat.Visualizer,
    candidates: list[SuctionGraspCandidate],
    cup_radius: float = 0.01,
    cup_height: float = 0.02,
    arrow_length: float = 0.05,
):
    """Visualize all grasp candidates as suction cups with normal arrows.
    
    Each segment's candidates are colored from white (best) to dark gray (worst).
    """
    if not candidates:
        return
    
    # Group candidates by segment
    by_segment: dict[int, list[SuctionGraspCandidate]] = defaultdict(list)
    for c in candidates:
        by_segment[c.segment_id].append(c)
    
    # Sort each segment's candidates by score (best first)
    for seg_id in by_segment:
        by_segment[seg_id] = sorted(by_segment[seg_id], key=lambda x: x.score, reverse=True)
    
    total_visualized = 0
    
    for seg_id in sorted(by_segment.keys()):
        seg_candidates = by_segment[seg_id]
        n_candidates = len(seg_candidates)
        
        for rank, candidate in enumerate(seg_candidates):
            color = get_rank_color(rank, n_candidates)
            grasp_key = f"seg_{seg_id}_grasp_{rank}"
            
            # Suction cup cylinder
            vis["grasps"][grasp_key]["cup"].set_object(
                geom.Cylinder(cup_height, cup_radius),
                geom.MeshLambertMaterial(color=color, opacity=0.8)
            )
            
            # Orient cylinder along normal
            normal = candidate.normal
            y_axis = np.array([0, 1, 0])
            
            v = np.cross(y_axis, normal)
            s = np.linalg.norm(v)
            c = np.dot(y_axis, normal)
            
            if s > 1e-6:
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
            else:
                R = np.eye(3) if c > 0 else np.diag([1, -1, -1])
            
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = candidate.position + normal * cup_height / 2
            vis["grasps"][grasp_key]["cup"].set_transform(T)
            
            # Normal arrow
            arrow_pts = np.array([candidate.position, candidate.position + normal * arrow_length]).T.astype(np.float32)
            vis["grasps"][grasp_key]["normal"].set_object(
                geom.Line(geom.PointsGeometry(arrow_pts), 
                          geom.LineBasicMaterial(color=color, linewidth=3))
            )
            
            # Grasp point sphere
            vis["grasps"][grasp_key]["point"].set_object(
                geom.Sphere(0.005),
                geom.MeshLambertMaterial(color=color)
            )
            T_point = np.eye(4)
            T_point[:3, 3] = candidate.position
            vis["grasps"][grasp_key]["point"].set_transform(T_point)
            
            total_visualized += 1
    
    print(f"[Meshcat] Visualized {total_visualized} grasp candidates "
          f"(all top_k per segment, White=best → Dark=worst)")


# =============================================================================
# POINT CLOUD EXTRACTION
# =============================================================================
def apply_depth_filters(
    depth_frame: rs.depth_frame,
    config: DepthFilterConfig,
) -> rs.depth_frame:
    """Apply RealSense post-processing filters to depth frame.
    
    These filters run efficiently and improve depth quality:
    - Spatial: Edge-preserving smoothing (flattens surfaces)
    - Temporal: Multi-frame averaging (reduces noise for static scenes)
    - Hole filling: Fills invalid pixels with neighbor data
    """
    filtered = depth_frame
    filters_applied = []
    
    # Decimation filter (reduces resolution)
    if config.decimation_enabled:
        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, config.decimation_magnitude)
        filtered = decimation.process(filtered).as_depth_frame()
        filters_applied.append(f"decimation(mag={config.decimation_magnitude})")
    
    # Spatial filter (edge-preserving smoothing)
    if config.spatial_enabled:
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, config.spatial_magnitude)
        spatial.set_option(rs.option.filter_smooth_alpha, config.spatial_smooth_alpha)
        spatial.set_option(rs.option.filter_smooth_delta, config.spatial_smooth_delta)
        spatial.set_option(rs.option.holes_fill, config.spatial_holes_fill)
        filtered = spatial.process(filtered).as_depth_frame()
        filters_applied.append(f"spatial(mag={config.spatial_magnitude}, holes={config.spatial_holes_fill})")
    
    # Temporal filter (multi-frame averaging)
    if config.temporal_enabled:
        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, config.temporal_smooth_alpha)
        temporal.set_option(rs.option.filter_smooth_delta, config.temporal_smooth_delta)
        temporal.set_option(rs.option.holes_fill, config.temporal_persistence)
        filtered = temporal.process(filtered).as_depth_frame()
        filters_applied.append(f"temporal(alpha={config.temporal_smooth_alpha})")
    
    # Hole filling filter
    if config.hole_filling_enabled:
        hole_filling = rs.hole_filling_filter()
        hole_filling.set_option(rs.option.holes_fill, config.hole_filling_mode)
        filtered = hole_filling.process(filtered).as_depth_frame()
        filters_applied.append(f"hole_fill(mode={config.hole_filling_mode})")
    
    if filters_applied:
        print(f"[DepthFilter] Applied: {', '.join(filters_applied)}")
    
    return filtered


def extract_segment_pointclouds(
    result: PipelineResult,
    max_background_points: int = 100000,
    depth_filter_config: DepthFilterConfig | None = None,
) -> ScenePointClouds:
    """Extract point clouds from pipeline result.
    
    Computes the full point cloud using RealSense SDK (handles lens distortion),
    transforms to robot base frame, and separates into segments and background.
    
    Args:
        result: PipelineResult from DetectionPipeline
        max_background_points: Maximum background points to keep (for performance)
        depth_filter_config: Configuration for depth post-processing filters
        
    Returns:
        ScenePointClouds with segment and background point clouds
    """
    h, w = result.full_shape
    depth_filter_config = depth_filter_config or DEPTH_FILTER_CFG
    
    # Apply RealSense post-processing filters
    print("[PointCloud] Applying depth filters...")
    filtered_depth = apply_depth_filters(result.depth_frame, depth_filter_config)
    
    # Compute point cloud using RealSense SDK
    print("[PointCloud] Computing point cloud with RealSense SDK...")
    rs_pc = rs.pointcloud()
    points_rs = rs_pc.calculate(filtered_depth)
    points_camera = np.asanyarray(points_rs.get_vertices()).view(np.float32).reshape(-1, 3)
    
    # Filter valid points
    valid_mask = (
        (points_camera[:, 2] > 0.1) & 
        (points_camera[:, 2] < 3.0) & 
        np.isfinite(points_camera[:, 2])
    )
    print(f"[PointCloud] {valid_mask.sum():,} valid points out of {len(points_camera):,}")
    
    # Transform to robot base frame
    points = transform_points(points_camera, T_BASE_CAMERA)
    
    # Uncrop masks to full image coordinates
    full_masks = []
    for mask in result.masks:
        if result.crop is not None:
            full_mask = uncrop_mask(mask, result.crop, (h, w))
        else:
            full_mask = mask
        full_masks.append(full_mask.flatten())
    
    # Generate colors for segments
    colors = generate_colors(len(full_masks))
    
    # Extract segment point clouds
    segments: list[SegmentPointCloud] = []
    all_masked = np.zeros(h * w, dtype=bool)
    
    for i, (fm, det, color) in enumerate(zip(full_masks, result.detections, colors)):
        obj_mask = valid_mask & (fm > 0)
        obj_points = points[obj_mask]
        all_masked |= (fm > 0)
        
        if obj_points.shape[0] > 0:
            segments.append(SegmentPointCloud(
                segment_id=i,
                name=det.get("name", f"object_{i}"),
                points=obj_points,
                color=color,
            ))
            print(f"  [Segment {i}] {det['name']}: {obj_points.shape[0]:,} points")
    
    # Extract background points
    bg_mask = valid_mask & ~all_masked
    bg_points = points[bg_mask]
    
    if bg_points.shape[0] > max_background_points:
        indices = np.random.choice(bg_points.shape[0], size=max_background_points, replace=False)
        bg_points = bg_points[indices]
    
    print(f"[PointCloud] Background: {bg_points.shape[0]:,} points")
    
    return ScenePointClouds(
        segments=segments,
        background_points=bg_points,
        full_points=points,
        valid_mask=valid_mask,
    )


# =============================================================================
# GRASP DETECTION ON SEGMENTS
# =============================================================================
class GraspDetectionResult(NamedTuple):
    """Result of grasp detection on all segments."""
    candidates: list[SuctionGraspCandidate]
    segment_results: list[SegmentGraspResult]


def detect_grasps_on_segments(
    scene: ScenePointClouds,
    config: GraspDetectionConfig | None = None,
    verbose: bool = True,
) -> GraspDetectionResult:
    """Run grasp detection on all segment point clouds.
    
    Args:
        scene: ScenePointClouds from extract_segment_pointclouds
        config: Grasp detection configuration
        verbose: Print progress
        
    Returns:
        GraspDetectionResult with all candidates and per-segment results
    """
    config = config or GRASP_CFG
    
    if verbose:
        print("-" * 60)
        print("[Grasp Detection] Running suction grasp detection...")
        print(f"  Config: voxel={config.voxel_size*1000:.1f}mm, "
              f"suction_radius={config.suction_cup_radius*1000:.1f}mm, "
              f"min_normal_z={config.min_normal_z}")
        print(f"  Plane fit: RMSE<{config.plane_fit_rmse_threshold*1000:.1f}mm, "
              f"rim_gap<{config.rim_gap_threshold*1000:.1f}mm")
    
    all_candidates: list[SuctionGraspCandidate] = []
    segment_results: list[SegmentGraspResult] = []
    
    for seg in scene.segments:
        if verbose:
            print(f"  [Segment {seg.segment_id}] {seg.name} ({seg.points.shape[0]:,} points)")
        
        result = detect_grasps_on_segment(
            points=seg.points,
            segment_id=seg.segment_id,
            config=config,
            verbose=verbose,
        )

        print('number of points in segment: ', seg.points.shape[0])
        print('number of candidates: ', len(result.candidates))
        
        segment_results.append(result)
        all_candidates.extend(result.candidates)
    
    # Sort by score
    all_candidates = sorted(all_candidates, key=lambda x: x.score, reverse=True)
    
    if verbose:
        print("-" * 60)
        print(f"[Grasp Detection] Total: {len(all_candidates)} candidates "
              f"from {len(segment_results)} segments")
        
        if all_candidates:
            print("\n  Top 5 grasp candidates:")
            for i, c in enumerate(all_candidates[:5]):
                print(f"    {i+1}. Segment {c.segment_id}: score={c.score:.3f} "
                      f"@ [{c.position[0]:.3f}, {c.position[1]:.3f}, {c.position[2]:.3f}]")
    
    return GraspDetectionResult(
        candidates=all_candidates,
        segment_results=segment_results,
    )


# =============================================================================
# MESHCAT VISUALIZATION
# =============================================================================
def visualize_scene(
    scene: ScenePointClouds,
    grasp_result: GraspDetectionResult | None = None,
    show_normals: bool = True,
    show_grasps: bool = True,
    curvature_threshold: float = 0.03,
) -> meshcat.Visualizer:
    """Visualize point cloud scene and optionally grasp candidates.
    
    Args:
        scene: ScenePointClouds from extract_segment_pointclouds
        grasp_result: Optional GraspDetectionResult to visualize
        show_normals: Show surface normals and curvature
        show_grasps: Show grasp candidates as suction cups
        curvature_threshold: Threshold for curvature coloring
        
    Returns:
        Meshcat visualizer instance
    """
    vis = meshcat.Visualizer().open()
    
    # Draw coordinate frame at origin
    L = 0.1
    for axis, color, vec in [("X", 0xff0000, [L, 0, 0]), ("Y", 0x00ff00, [0, L, 0]), ("Z", 0x0000ff, [0, 0, L])]:
        pts = np.array([[0, 0, 0], vec], dtype=np.float32).T
        vis["frame"][axis].set_object(
            geom.Line(geom.PointsGeometry(pts), geom.LineBasicMaterial(color=color, linewidth=3))
        )
    
    # Background points
    if scene.background_points.shape[0] > 0:
        vis["pointcloud"]["background"].set_object(
            geom.Points(
                geom.PointsGeometry(scene.background_points.T.astype(np.float32)),
                geom.PointsMaterial(size=0.002, color=0x888888)
            )
        )
        print(f"[Meshcat] Background: {scene.background_points.shape[0]:,} points")
    
    # Segment point clouds
    for seg in scene.segments:
        hex_color = (seg.color[2] << 16) | (seg.color[1] << 8) | seg.color[0]
        vis["pointcloud"][f"segment_{seg.segment_id}"].set_object(
            geom.Points(
                geom.PointsGeometry(seg.points.T.astype(np.float32)),
                geom.PointsMaterial(size=0.002, color=hex_color)
            )
        )
        print(f"[Meshcat] Segment {seg.segment_id} ({seg.name}): {seg.points.shape[0]:,} points")
    
    # Visualize normals and grasp candidates if provided
    if grasp_result is not None:
        if show_normals:
            visualize_normals_and_curvature(
                vis, grasp_result.segment_results,
                curvature_threshold=curvature_threshold
            )
        
        if show_grasps and grasp_result.candidates:
            print("-" * 60)
            visualize_grasp_candidates(vis, grasp_result.candidates)
    
    print("-" * 60)
    print(f"[Meshcat] URL: {vis.url()}")
    print(f"[Meshcat] Points in robot base frame")
    
    return vis


# =============================================================================
# MAIN
# =============================================================================
def print_grasp_summary(grasp_result: GraspDetectionResult, detections: list[dict]):
    """Print a detailed summary of all detected grasps with score breakdowns."""
    if not grasp_result.candidates:
        return
    
    print("\n" + "=" * 80)
    print("GRASP DETECTION SUMMARY - ALL CANDIDATES")
    print("=" * 80)
    
    by_segment = defaultdict(list)
    for c in grasp_result.candidates:
        by_segment[c.segment_id].append(c)
    
    for seg_id in sorted(by_segment.keys()):
        seg_candidates = sorted(by_segment[seg_id], key=lambda x: x.score, reverse=True)
        det_name = detections[seg_id]["name"] if seg_id < len(detections) else "unknown"
        
        print(f"\n{'─' * 80}")
        print(f"SEGMENT {seg_id}: {det_name} ({len(seg_candidates)} candidates)")
        print(f"{'─' * 80}")
        
        for rank, c in enumerate(seg_candidates):
            print(f"\n  [{rank + 1}] TOTAL SCORE: {c.score:.4f}")
            print(f"      Position: [{c.position[0]:.4f}, {c.position[1]:.4f}, {c.position[2]:.4f}]")
            print(f"      Normal:   [{c.normal[0]:.4f}, {c.normal[1]:.4f}, {c.normal[2]:.4f}]")
            print(f"      ┌─────────────────────────────────────────────────────")
            print(f"      │ PLANE FIT METRICS (suction seal quality):")
            print(f"      │   Plane RMSE:      {c.plane_rmse*1000:.3f} mm  (threshold: {GRASP_CFG.plane_fit_rmse_threshold*1000:.1f} mm)")
            print(f"      │   Rim max gap:     {c.rim_max_gap*1000:.3f} mm  (threshold: {GRASP_CFG.rim_gap_threshold*1000:.1f} mm)")
            print(f"      │   Plane fit score: {c.plane_fit_score:.4f}")
            print(f"      │   Rim seal score:  {c.rim_seal_score:.4f}")
            print(f"      ├─────────────────────────────────────────────────────")
            print(f"      │ Point-level scores:")
            print(f"      │   Flatness:     {c.flatness_score:.4f}  (weight: {GRASP_CFG.weight_flatness})")
            print(f"      │   Verticality:  {c.verticality_score:.4f}  (weight: {GRASP_CFG.weight_verticality})")
            print(f"      │   Centroid:     {c.centroid_score:.4f}  (weight: {GRASP_CFG.weight_centroid})")
            print(f"      │ Region-level scores:")
            print(f"      │   Normal consistency: {c.normal_consistency_score:.4f}")
            print(f"      │   Planarity:          {c.planarity_score:.4f}")
            print(f"      │   Coverage:           {c.coverage_score:.4f}")
            print(f"      │   Region (avg):       {c.region_score:.4f}  (weight: {GRASP_CFG.weight_region})")
            print(f"      │ Legacy:")
            print(f"      │   Curvature:    {c.curvature:.5f}")
            print(f"      └─────────────────────────────────────────────────────")
            
            # Show weighted contribution
            weighted_flat = GRASP_CFG.weight_flatness * c.flatness_score
            weighted_vert = GRASP_CFG.weight_verticality * c.verticality_score
            weighted_cent = GRASP_CFG.weight_centroid * c.centroid_score
            weighted_region = GRASP_CFG.weight_region * c.region_score
            plane_bonus = 0.5 * (c.plane_fit_score + c.rim_seal_score)
            print(f"      Weighted: flat={weighted_flat:.3f} + vert={weighted_vert:.3f} "
                  f"+ cent={weighted_cent:.3f} + region={weighted_region:.3f} + plane_bonus={plane_bonus:.3f}")
    
    print("\n" + "=" * 80)


def main():
    # Step 1: Run detection pipeline (camera capture + detection + SAM)
    pipeline = DetectionPipeline(PipelineConfig())
    result = pipeline.run()
    
    if result is None:
        print("[Done] No objects detected.")
        return
    
    if result.depth_frame is None or not result.masks:
        print("[Done] No depth data or masks available.")
        return
    
    # Step 2: Extract point clouds from segments
    print("-" * 60)
    scene = extract_segment_pointclouds(result)
    
    if not scene.segments:
        print("[Done] No valid segment point clouds.")
        return
    
    # Step 3: Run grasp detection on segments
    grasp_result = detect_grasps_on_segments(scene, GRASP_CFG)
    
    # Step 4: Visualize scene with grasp candidates
    print("-" * 60)
    print("[Meshcat] Generating 3D visualization...")
    vis = visualize_scene(
        scene=scene,
        grasp_result=grasp_result,
        show_normals=True,
        show_grasps=True,
        curvature_threshold=GRASP_CFG.curvature_threshold,
    )
    
    # Print summary
    print_grasp_summary(grasp_result, result.detections)
    
    input("[Meshcat] Press Enter to close visualization and exit...")
    
    print("-" * 60)
    print(f"[Done] Saved to: {result.session_dir}")


if __name__ == "__main__":
    main()
