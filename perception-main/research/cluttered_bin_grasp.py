"""
Cluttered bin grasp detection pipeline.

Uses SAM2 for segmentation and computes suction grasp poses for each object.
Validates that the suction cup can form a complete seal before accepting grasps.

SCORING (max score = 1.5):
    total = 0.1×verticality     (alignment with gravity)
          + 0.1×centroid        (proximity to object center)
          + 0.3×flatness        (low surface curvature)
          + 0.5×plane_fit       (planarity under suction cup)
          + 0.5×rim_coverage    (suction cup seal quality)

Usage:
    python cluttered_bin_grasp.py              # Default Open3D visualization
    python cluttered_bin_grasp.py --meshcat    # Meshcat web visualization
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import torch
from scipy.spatial import cKDTree

from cluttered_bin_segmentation import (
    ClutteredBinConfig, load_sam2_auto_mask_generator, generate_masks,
    filter_masks_by_area, filter_edge_masks, merge_overlapping_masks,
    filter_contained_masks, filter_masks_by_depth, sort_masks_by_depth,
    load_crop_config, crop_image, visualize_masks,
    init_realsense, get_rgbd_frame,
)
from view_clutter import build_occupancy_volume, create_gripper_mesh
from visualization import visualize_results

# Camera to robot base transform
T_BASE_CAMERA = np.array([
    [ -0.99870602,  -0.05068679,   0.00413055,   0.48695190],
    [ -0.05061085,   0.99857739,   0.01678431,   0.10477300],
    [ -0.00497542,   0.01655354,  -0.99985062,   1.02676089],
    [  0.00000000,   0.00000000,   0.00000000,   1.00000000],
])


@dataclass
class GraspConfig:
    """Configuration for suction grasp detection."""
    # Workspace filtering (robot base frame)
    plane_normal: tuple[float, float, float] = (0.0, 0.0, -1.0)
    plane_d: float = 0.05
    plane_inlier_thresh: float = 0.004
    z_above_threshold: float = 0.01
    
    # Pre-processing
    voxel_size: float = 0.003
    sor_mean_k: int = 30
    sor_std_ratio: float = 1.0
    normal_k_neighbors: int = 50
    camera_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    
    # Grasp constraints
    gravity_vector: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    min_normal_z: float = 0.7
    curvature_threshold: float = 0.03
    
    # Suction cup parameters (3cm diameter = 1.5cm radius)
    suction_cup_radius: float = 0.015
    plane_fit_rmse_threshold: float = 0.01
    rim_gap_threshold: float = 0.005
    
    # Seal validation - CRITICAL: reject if suction cup overhangs the object
    min_rim_coverage_ratio: float = 0.9  # Require 60% of rim sectors to have points
    n_rim_sectors: int = 8  # Divide rim into 8 angular sectors
    rim_inner_ratio: float = 0.7  # Rim starts at 70% of radius
    
    # Scoring weights
    # Total = w1*verticality + w2*centroid + w3*flatness + w4*plane_fit + w5*rim_coverage
    # Max possible score = 0.1 + 0.1 + 0.3 + 0.5 + 0.5 = 1.5
    weight_verticality: float = 0.1
    weight_centroid: float = 0.1
    weight_flatness: float = 0.3
    weight_plane_fit: float = 0.5
    weight_rim_coverage: float = 0.5
    
    # Collision checking
    # NOTE: We only check collisions for:
    #   1. Approach path (center axis of suction cup, upper 30%)
    #   2. Gripper body volume
    # We do NOT check the full suction cup volume - it's expected to be near the object!
    collision_voxel_size: float = 0.008  # Voxel size for collision occupancy volume (8mm)
    collision_ray_depth: float = 0.15    # Max ray depth for occupancy casting
    collision_check_radius: float = 0.012  # Radius to check for collisions (should be >= voxel_size)
    gripper_cup_height: float = 0.06     # Suction cup cylinder height (6cm)
    gripper_body_size: tuple = (0.06, 0.03, 0.13)  # Body (width, depth, height)
    
    # Output
    top_k_per_segment: int = 10


GRASP_CFG = GraspConfig()


@dataclass
class GraspCandidate:
    """A candidate suction grasp pose."""
    position: np.ndarray
    normal: np.ndarray
    score: float
    segment_id: int = -1
    
    # Metrics
    plane_rmse: float = 0.0
    rim_coverage: float = 0.0  # Fraction of rim sectors with points
    seal_valid: bool = False
    collision_free: bool = False  # True if gripper doesn't collide with scene
    
    def __repr__(self):
        return f"Grasp(pos=[{self.position[0]:.3f},{self.position[1]:.3f},{self.position[2]:.3f}], score={self.score:.3f}, collision_free={self.collision_free})"


class SegmentResult(NamedTuple):
    """Result of grasp detection on a segment."""
    segment_id: int
    candidates: list[GraspCandidate]
    points: np.ndarray | None = None


class SegmentPointCloud(NamedTuple):
    """Point cloud for a segmented object."""
    segment_id: int
    points: np.ndarray
    colors: np.ndarray
    mask: np.ndarray


# =============================================================================
# POINT CLOUD PROCESSING
# =============================================================================
def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform Nx3 points using 4x4 matrix."""
    ones = np.ones((len(points), 1), dtype=points.dtype)
    return ((T @ np.hstack([points, ones]).T).T)[:, :3]


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel grid downsampling."""
    if len(points) == 0:
        return points
    voxel_idx = np.floor(points / voxel_size).astype(int)
    offset = voxel_idx.min(axis=0)
    voxel_idx -= offset
    dims = voxel_idx.max(axis=0) + 1
    linear = voxel_idx[:, 0] * dims[1] * dims[2] + voxel_idx[:, 1] * dims[2] + voxel_idx[:, 2]
    unique, inverse = np.unique(linear, return_inverse=True)
    out = np.zeros((len(unique), 3), dtype=points.dtype)
    counts = np.zeros(len(unique), dtype=int)
    np.add.at(out, inverse, points)
    np.add.at(counts, inverse, 1)
    return out / counts[:, None]


def remove_outliers(points: np.ndarray, k: int = 30, std_ratio: float = 1.0) -> np.ndarray:
    """Statistical outlier removal."""
    if len(points) < k + 1:
        return points
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    thresh = mean_dists.mean() + std_ratio * mean_dists.std()
    return points[mean_dists < thresh]


def estimate_normals(points: np.ndarray, k: int = 30, orient_toward: np.ndarray = None):
    """Estimate normals and curvature via PCA."""
    n = len(points)
    if n == 0:
        return np.zeros((0, 3)), np.zeros(0)
    
    tree = cKDTree(points)
    k = min(k, n)
    _, indices = tree.query(points, k=k)
    
    normals = np.zeros((n, 3))
    curvatures = np.zeros(n)
    
    for i in range(n):
        neighbors = points[indices[i]]
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered / (k - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        normals[i] = eigvecs[:, idx[2]]
        curvatures[i] = eigvals[idx[2]] / (eigvals.sum() + 1e-10)
    
    if orient_toward is not None:
        flip = ((orient_toward - points) * normals).sum(axis=1) < 0
        normals[flip] *= -1
    
    return normals, curvatures


def filter_workspace(points: np.ndarray, cfg: GraspConfig) -> np.ndarray:
    """Filter to workspace above table plane."""
    n = np.array(cfg.plane_normal)
    dists = points @ n + cfg.plane_d
    inlier = np.abs(dists) < cfg.plane_inlier_thresh
    
    if abs(n[2]) > 1e-6:
        z_plane = (-cfg.plane_d - n[0] * points[:, 0] - n[1] * points[:, 1]) / n[2]
        above = (points[:, 2] - z_plane >= cfg.z_above_threshold) & ~inlier
        return points[above]
    return points[~inlier]


# =============================================================================
# SCORING FUNCTIONS - Each returns a score in [0, 1]
# =============================================================================
def score_verticality(normal: np.ndarray, gravity: np.ndarray) -> float:
    """
    Score how aligned the surface normal is with the gravity vector.
    
    Args:
        normal: Surface normal at grasp point (unit vector)
        gravity: Gravity direction (unit vector, typically [0, 0, -1])
    
    Returns:
        Score in [0, 1], where 1 = perfectly aligned with -gravity (pointing up)
    """
    return float(np.clip(np.dot(normal, -gravity), 0, 1))


def score_centroid_proximity(point: np.ndarray, center: np.ndarray, max_dist: float) -> float:
    """
    Score how close the grasp point is to the object's center (in XY plane).
    
    Args:
        point: Grasp point position
        center: Object center position
        max_dist: Maximum XY distance in the point cloud (for normalization)
    
    Returns:
        Score in [0, 1], where 1 = at center, 0 = at max distance
    """
    dist_xy = np.linalg.norm(point[:2] - center[:2])
    return float(np.clip(1.0 - dist_xy / max(max_dist, 1e-6), 0, 1))


def score_flatness(curvature: float, threshold: float) -> float:
    """
    Score surface flatness based on local curvature.
    
    Args:
        curvature: Local surface curvature (from PCA eigenvalue ratio)
        threshold: Curvature threshold above which score is 0
    
    Returns:
        Score in [0, 1], where 1 = flat (curvature=0), 0 = curved (>= threshold)
    """
    return float(np.clip(1.0 - curvature / threshold, 0, 1))


def score_plane_fit(rmse: float, threshold: float) -> float:
    """
    Score how well a plane fits the surface under the suction cup.
    
    Args:
        rmse: Root mean square error of plane fit (meters)
        threshold: RMSE threshold above which score is 0
    
    Returns:
        Score in [0, 1], where 1 = perfect plane fit, 0 = poor fit
    """
    if rmse == float('inf'):
        return 0.0
    return float(np.clip(1.0 - rmse / threshold, 0, 1))


def score_rim_coverage(
    center: np.ndarray,
    points: np.ndarray,
    tree: cKDTree,
    radius: float,
    n_sectors: int = 8,
    rim_inner_ratio: float = 0.7,
) -> float:
    """
    Score how well the suction cup rim has surface contact in all directions.
    
    Divides the rim into angular sectors and checks if each sector has points.
    This is CRITICAL for seal validation - a grasp is only valid if the suction
    cup doesn't overhang the object.
    
    Args:
        center: Grasp point position
        points: All points in the segment
        tree: KD-tree for neighbor queries
        radius: Suction cup radius
        n_sectors: Number of angular sectors to check
        rim_inner_ratio: Rim starts at this fraction of radius
    
    Returns:
        Score in [0, 1], fraction of sectors with valid contact
    """
    # Get points within suction cup radius
    neighbor_idx = tree.query_ball_point(center, radius)
    if len(neighbor_idx) < 3:
        return 0.0
    
    neighbors = points[neighbor_idx]
    
    # Convert to 2D relative coordinates (project to XY plane relative to center)
    rel = neighbors[:, :2] - center[:2]
    dists = np.linalg.norm(rel, axis=1)
    
    # Filter to rim region (outer ring of suction cup)
    rim_inner = radius * rim_inner_ratio
    rim_mask = dists >= rim_inner
    rim_points = rel[rim_mask]
    
    if len(rim_points) == 0:
        return 0.0
    
    # Compute angular sector for each rim point
    angles = np.arctan2(rim_points[:, 1], rim_points[:, 0])  # -pi to pi
    sectors = ((angles + np.pi) / (2 * np.pi) * n_sectors).astype(int) % n_sectors
    
    # Count unique sectors with points
    unique_sectors = len(np.unique(sectors))
    return float(unique_sectors / n_sectors)


def compute_plane_fit_rmse(
    center: np.ndarray,
    points: np.ndarray,
    tree: cKDTree,
    radius: float,
) -> float:
    """
    Compute RMSE of plane fit to surface under suction cup.
    
    Args:
        center: Grasp point position
        points: All points in the segment
        tree: KD-tree for neighbor queries
        radius: Suction cup radius
    
    Returns:
        RMSE of plane fit in meters, or inf if insufficient points
    """
    neighbor_idx = tree.query_ball_point(center, radius)
    
    if len(neighbor_idx) < 5:
        return float('inf')
    
    neighbors = points[neighbor_idx]
    
    # Fit plane to region using SVD
    centroid = neighbors.mean(axis=0)
    centered = neighbors - centroid
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        plane_normal = vh[2]
        dists = np.abs(centered @ plane_normal)
        return float(np.sqrt((dists ** 2).mean()))
    except np.linalg.LinAlgError:
        return float('inf')


# =============================================================================
# COLLISION CHECKING
# =============================================================================
class SceneCollisionChecker:
    """Checks gripper collision against scene point cloud + occupancy volume."""
    
    def __init__(
        self,
        background_points: np.ndarray,
        segment_points_list: list[np.ndarray],
        occupancy_points: np.ndarray,
        check_radius: float = 0.012,
    ):
        """
        Initialize collision checker with separate background, segments, and occupancy.
        
        When checking collision for a segment, we exclude that segment's own points
        to avoid false collisions with the object being grasped.
        
        Args:
            background_points: Nx3 array of background surface points
            segment_points_list: List of Mx3 arrays, one per segment
            occupancy_points: Kx3 array of occupancy volume points
            check_radius: Radius to search for collision points (should be >= voxel_size)
        """
        self.background_points = background_points if len(background_points) > 0 else np.zeros((0, 3))
        self.segment_points_list = segment_points_list
        self.occupancy_points = occupancy_points if len(occupancy_points) > 0 else np.zeros((0, 3))
        self.check_radius = check_radius
        
        # Pre-build a KD-tree for background + occupancy (always checked)
        base_points = []
        if len(self.background_points) > 0:
            base_points.append(self.background_points)
        if len(self.occupancy_points) > 0:
            base_points.append(self.occupancy_points)
        
        if base_points:
            self.base_collision_points = np.vstack(base_points)
            self.base_tree = cKDTree(self.base_collision_points)
        else:
            self.base_collision_points = np.zeros((0, 3))
            self.base_tree = None
        
        n_seg_pts = sum(len(pts) for pts in segment_points_list)
        print(f"[Collision] Checker initialized:")
        print(f"    Background: {len(self.background_points):,} pts")
        print(f"    Segments: {len(segment_points_list)} segments, {n_seg_pts:,} total pts")
        print(f"    Occupancy: {len(self.occupancy_points):,} pts")
        print(f"    Check radius: {check_radius*1000:.1f}mm")
    
    def _build_collision_tree_excluding_segment(self, exclude_segment_id: int) -> cKDTree | None:
        """Build collision tree excluding the specified segment's points."""
        collision_points = [self.base_collision_points] if len(self.base_collision_points) > 0 else []
        
        # Add all OTHER segments' points
        for i, seg_pts in enumerate(self.segment_points_list):
            if i != exclude_segment_id and len(seg_pts) > 0:
                collision_points.append(seg_pts)
        
        if collision_points:
            all_pts = np.vstack(collision_points)
            return cKDTree(all_pts)
        return None
    
    def check_gripper_collision(
        self,
        position: np.ndarray,
        normal: np.ndarray,
        cfg: GraspConfig,
        segment_id: int = -1,
        verbose: bool = False,
    ) -> tuple[bool, int]:
        """
        Check if gripper at given pose collides with scene/occupancy.
        
        Samples points throughout the gripper VOLUME (not just surface) to ensure
        nothing penetrates the gripper space.
        
        IMPORTANT: Excludes the segment being grasped from collision checking
        to avoid false positives from the object's own surface.
        
        Args:
            position: Grasp position (suction cup contact point)
            normal: Surface normal at grasp point (gripper extends along this direction)
            cfg: Grasp configuration with gripper dimensions
            segment_id: ID of segment being grasped (excluded from collision check)
            verbose: Print debug info
        
        Returns:
            (collision_free, n_collisions): True if no collision, number of collision points
        """
        # Build collision tree excluding the segment being grasped
        collision_tree = self._build_collision_tree_excluding_segment(segment_id)
        if collision_tree is None:
            return True, 0
        
        # Build gripper rotation matrix: align local Z with surface normal
        z_local = np.array([0.0, 0.0, 1.0])
        n = normal / np.linalg.norm(normal)
        v = np.cross(z_local, n)
        s, c = np.linalg.norm(v), np.dot(z_local, n)
        
        if s > 1e-6:
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
        else:
            R = np.eye(3) if c > 0 else np.diag([1, -1, -1])
        
        cup_radius = cfg.suction_cup_radius
        cup_height = cfg.gripper_cup_height
        body_w, body_d, body_h = cfg.gripper_body_size
        
        # Generate sample points throughout the gripper
        # 
        # KEY INSIGHT: The suction cup SHOULD be near the object surface - that's
        # the whole point! We don't want to reject grasps just because other objects
        # are nearby. We only care about:
        #   1. Is something BLOCKING the approach (along center axis)?
        #   2. Will the gripper BODY collide with something?
        #
        check_points_local = []
        
        # 1. Check APPROACH PATH only (center axis of suction cup)
        #    This catches cases where another object is directly above the grasp point
        #    Start at top of cup (where it meets body) - the cup itself will displace air/contact object
        approach_check_start = cup_height * 0.7  # Start at 70% up the cup (~42mm for 60mm cup)
        for h in np.linspace(approach_check_start, cup_height, 4):
            check_points_local.append(np.array([0, 0, h]))
        
        # 2. Sample gripper BODY volume (this is what we really care about!)
        body_sample_step = 0.015  # 15mm sampling
        for x in np.arange(-body_w/2, body_w/2 + body_sample_step, body_sample_step):
            for y in np.arange(-body_d/2, body_d/2 + body_sample_step, body_sample_step):
                for z in np.arange(cup_height, cup_height + body_h, body_sample_step):
                    check_points_local.append(np.array([x, y, z]))
        
        check_points_local = np.array(check_points_local)
        
        # Transform to world frame
        world_points = (R @ check_points_local.T).T + position
        
        # Check for collisions
        n_collisions = 0
        collision_positions = []
        
        for i, wp in enumerate(world_points):
            neighbors = collision_tree.query_ball_point(wp, self.check_radius)
            if len(neighbors) > 0:
                n_collisions += len(neighbors)
                if verbose and len(collision_positions) < 5:
                    collision_positions.append((wp, check_points_local[i]))
                # Early exit after confirming collision
                if n_collisions > 0 and not verbose:
                    return False, n_collisions
        
        if verbose and n_collisions > 0:
            # Categorize collision by gripper part based on local Z
            approach_collisions = sum(1 for _, lp in collision_positions if lp[2] < cup_height)
            body_collisions = len(collision_positions) - approach_collisions
            
            print(f"    [Collision] {n_collisions} collision points "
                  f"(approach path: {approach_collisions}, body: {body_collisions})")
            print(f"      Gripper at pos=[{position[0]:.3f},{position[1]:.3f},{position[2]:.3f}], "
                  f"normal=[{normal[0]:.2f},{normal[1]:.2f},{normal[2]:.2f}]")
            for wp, lp in collision_positions[:3]:
                part = "APPROACH" if lp[2] < cup_height else "BODY"
                dist_from_axis = np.sqrt(lp[0]**2 + lp[1]**2)
                print(f"      [{part}] world=[{wp[0]:.3f},{wp[1]:.3f},{wp[2]:.3f}] "
                      f"local=[{lp[0]:.3f},{lp[1]:.3f},{lp[2]:.3f}] "
                      f"(height={lp[2]*1000:.0f}mm from contact)")
        
        return n_collisions == 0, n_collisions


def build_scene_collision_data(
    segments: list,
    background_points: np.ndarray,
    T_base_camera: np.ndarray,
    cfg: GraspConfig,
) -> tuple[SceneCollisionChecker, np.ndarray]:
    """
    Build collision checker from scene data.
    
    Args:
        segments: List of SegmentPointCloud objects
        background_points: Background surface points
        T_base_camera: Camera to base transform
        cfg: Grasp configuration
    
    Returns:
        (SceneCollisionChecker, occupancy_points): Collision checker and occupancy volume points
    """
    # Build occupancy volume via ray casting
    viz_data = {
        'T_base_camera': T_base_camera,
        'background_points': background_points,
        'segments': [{'points': seg.points} for seg in segments],
    }
    
    occupancy_pts = build_occupancy_volume(
        viz_data,
        voxel_size=cfg.collision_voxel_size,
        max_ray_depth=cfg.collision_ray_depth,
    )
    
    # Create list of segment points (indexed by segment_id)
    # We need to map segment_id to index in the list
    max_seg_id = max(seg.segment_id for seg in segments) if segments else -1
    segment_points_list = [np.zeros((0, 3)) for _ in range(max_seg_id + 1)]
    for seg in segments:
        segment_points_list[seg.segment_id] = seg.points
    
    collision_checker = SceneCollisionChecker(
        background_points=background_points,
        segment_points_list=segment_points_list,
        occupancy_points=occupancy_pts,
        check_radius=cfg.collision_check_radius,
    )
    
    return collision_checker, occupancy_pts


# =============================================================================
# CENTROID ESTIMATION
# # =============================================================================
# def compute_object_center(points: np.ndarray, normals: np.ndarray, cfg: GraspConfig) -> np.ndarray:
#     """Estimate object center using multiple methods."""
#     if len(points) == 0:
#         return np.zeros(3)
    
#     # Top surface centroid (top 30% by Z)
#     z_thresh = np.percentile(points[:, 2], 70)
#     top_mask = points[:, 2] >= z_thresh
#     top_center = points[top_mask].mean(axis=0) if top_mask.any() else points.mean(axis=0)
    
#     # Upward-facing surface centroid
#     upward_mask = normals[:, 2] > cfg.min_normal_z
#     plane_center = points[upward_mask].mean(axis=0) if upward_mask.sum() >= 3 else points.mean(axis=0)
    
#     # Density-based center
#     if len(points) >= 10:
#         tree = cKDTree(points)
#         sample = points[np.random.choice(len(points), min(200, len(points)), replace=False)]
#         densities = [len(tree.query_ball_point(p, cfg.suction_cup_radius)) for p in sample]
#         density_center = sample[np.argmax(densities)]
#     else:
#         density_center = points.mean(axis=0)
    
#     # Weighted combination
#     return 0.4 * plane_center + 0.4 * density_center + 0.2 * top_center


def compute_object_center(points: np.ndarray) -> np.ndarray:
    """Simple geometric centroid."""
    if len(points) == 0:
        return np.zeros(3)
    return points.mean(axis=0)



# =============================================================================
# GRASP DETECTION
# =============================================================================
def detect_grasps(
    points: np.ndarray,
    segment_id: int,
    cfg: GraspConfig,
    collision_checker: SceneCollisionChecker = None,
    verbose: bool = True,
) -> SegmentResult:
    """
    Detect suction grasp candidates on a segment point cloud.
    
    GRASP SAMPLING PIPELINE:
    1. Preprocessing: Filter workspace, voxel downsample, remove outliers
    2. Normal estimation: PCA-based normals + curvature
    3. Filter upward-facing: Keep points with normal.z > min_normal_z (~45° from vertical)
    4. Score each point: verticality + centroid proximity + flatness + seal quality
    5. Seal validation: Check rim coverage (suction cup doesn't overhang object)
    6. Collision check: Verify gripper doesn't hit other objects/background
    
    Args:
        points: Nx3 array of segment points
        segment_id: ID of the segment
        cfg: Grasp configuration
        collision_checker: Optional collision checker for scene/occupancy collision
        verbose: Print debug info
    
    Returns:
        SegmentResult with candidate grasps
    """
    empty = SegmentResult(segment_id, [])
    prefix = f"    [Seg {segment_id}]"
    
    if len(points) < 50:
        if verbose:
            print(f"{prefix} SKIP: only {len(points)} points (need 50)")
        return empty
    
    # ===== STEP 1: Preprocessing =====
    pts = filter_workspace(points, cfg)
    if verbose:
        print(f"{prefix} Step 1a: {len(points):,} -> {len(pts):,} pts after workspace filter")
    if len(pts) < 30:
        if verbose:
            print(f"{prefix} SKIP: only {len(pts)} pts after workspace filter (need 30)")
        return empty
    
    pts = voxel_downsample(pts, cfg.voxel_size)
    if verbose:
        print(f"{prefix} Step 1b: -> {len(pts):,} pts after voxel downsample ({cfg.voxel_size*1000:.1f}mm)")
    if len(pts) < 20:
        if verbose:
            print(f"{prefix} SKIP: only {len(pts)} pts after downsample (need 20)")
        return empty
    
    pts = remove_outliers(pts, min(cfg.sor_mean_k, len(pts) - 1), cfg.sor_std_ratio)
    if verbose:
        print(f"{prefix} Step 1c: -> {len(pts):,} pts after outlier removal")
    if len(pts) < 10:
        if verbose:
            print(f"{prefix} SKIP: only {len(pts)} pts after outlier removal (need 10)")
        return empty
    
    # ===== STEP 2: Normal estimation =====
    normals, curvatures = estimate_normals(pts, min(cfg.normal_k_neighbors, len(pts)), cfg.camera_position)
    if verbose:
        print(f"{prefix} Step 2: Estimated normals (k={min(cfg.normal_k_neighbors, len(pts))})")
    
    # ===== STEP 3: Filter upward-facing points =====
    # min_normal_z = 0.7 means angle from vertical = arccos(0.7) ≈ 45.6°
    valid_mask = normals[:, 2] > cfg.min_normal_z
    n_upward = valid_mask.sum()
    if verbose:
        angle_thresh = np.degrees(np.arccos(cfg.min_normal_z))
        print(f"{prefix} Step 3: {n_upward}/{len(pts)} pts face upward "
              f"(normal.z > {cfg.min_normal_z} = within {angle_thresh:.1f}° of vertical)")
    
    if not valid_mask.any():
        if verbose:
            print(f"{prefix} SKIP: no upward-facing points")
        return SegmentResult(segment_id, [], pts)
    
    valid_pts = pts[valid_mask]
    valid_normals = normals[valid_mask]
    valid_curvatures = curvatures[valid_mask]
    
    # Build KD-tree for neighbor queries
    tree = cKDTree(pts)
    
    # Compute object center for centroid scoring
    center = compute_object_center(valid_pts)
    if verbose:
        print(f"{prefix} Step 4: Object center at [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    
    # ===== STEP 4: Compute individual scores for each valid point =====
    n = len(valid_pts)
    gravity = cfg.gravity_vector / np.linalg.norm(cfg.gravity_vector)
    
    # Pre-compute max XY distance for centroid normalization
    dists_xy = np.linalg.norm(valid_pts[:, :2] - center[:2], axis=1)
    max_dist = max(dists_xy.max(), 1e-6)
    
    # Score arrays
    verticality_scores = np.zeros(n)
    centroid_scores = np.zeros(n)
    flatness_scores = np.zeros(n)
    plane_fit_scores = np.zeros(n)
    rim_coverage_scores = np.zeros(n)
    plane_rmses = np.zeros(n)
    seal_valid = np.zeros(n, dtype=bool)
    
    # Compute all scores for each candidate
    for i in range(n):
        # Score 1: Verticality (how aligned with gravity)
        verticality_scores[i] = score_verticality(valid_normals[i], gravity)
        
        # Score 2: Centroid proximity (prefer center of object)
        centroid_scores[i] = score_centroid_proximity(valid_pts[i], center, max_dist)
        
        # Score 3: Flatness (low curvature)
        flatness_scores[i] = score_flatness(valid_curvatures[i], cfg.curvature_threshold)
        
        # Score 4: Plane fit (planarity under suction cup)
        rmse = compute_plane_fit_rmse(valid_pts[i], pts, tree, cfg.suction_cup_radius)
        plane_rmses[i] = rmse
        plane_fit_scores[i] = score_plane_fit(rmse, cfg.plane_fit_rmse_threshold)
        
        # Score 5: Rim coverage (seal quality)
        rim_coverage_scores[i] = score_rim_coverage(
            valid_pts[i], pts, tree,
            cfg.suction_cup_radius,
            cfg.n_rim_sectors,
            cfg.rim_inner_ratio
        )
        
        # Seal validation: reject if suction cup would overhang object
        seal_valid[i] = rim_coverage_scores[i] >= cfg.min_rim_coverage_ratio
    
    if verbose:
        print(f"{prefix}        Scoring {n} upward points:")
        print(f"{prefix}          verticality:   min={verticality_scores.min():.2f}, max={verticality_scores.max():.2f}, mean={verticality_scores.mean():.2f}")
        print(f"{prefix}          centroid:      min={centroid_scores.min():.2f}, max={centroid_scores.max():.2f}, mean={centroid_scores.mean():.2f}")
        print(f"{prefix}          flatness:      min={flatness_scores.min():.2f}, max={flatness_scores.max():.2f}, mean={flatness_scores.mean():.2f}")
        print(f"{prefix}          plane_fit:     min={plane_fit_scores.min():.2f}, max={plane_fit_scores.max():.2f}, mean={plane_fit_scores.mean():.2f}")
        print(f"{prefix}          rim_coverage:  min={rim_coverage_scores.min():.2f}, max={rim_coverage_scores.max():.2f}, mean={rim_coverage_scores.mean():.2f}")
    
    # ===== STEP 5: Seal validation =====
    n_valid = seal_valid.sum()
    n_rejected_seal = n - n_valid
    
    if verbose:
        print(f"{prefix} Step 5: Seal validation (rim coverage >= {cfg.min_rim_coverage_ratio*100:.0f}%)")
        print(f"{prefix}          {n_valid} seal-valid, {n_rejected_seal} rejected (suction cup would overhang)")
        valid_rmses = plane_rmses[plane_rmses < np.inf]
        if len(valid_rmses) > 0:
            print(f"{prefix}          plane RMSE:   min={valid_rmses.min()*1000:.2f}mm, max={valid_rmses.max()*1000:.2f}mm")
    
    # ===== Aggregate scores with weights =====
    # Total = w1*verticality + w2*centroid + w3*flatness + w4*plane_fit + w5*rim_coverage
    total = (
        cfg.weight_verticality * verticality_scores +
        cfg.weight_centroid * centroid_scores +
        cfg.weight_flatness * flatness_scores +
        cfg.weight_plane_fit * plane_fit_scores +
        cfg.weight_rim_coverage * rim_coverage_scores
    )
    
    # CRITICAL: Reject candidates where suction cup would overhang
    total[~seal_valid] = -np.inf
    
    # ===== STEP 6: Collision checking =====
    # Get top candidates (more than top_k to allow for collision filtering)
    n_candidates_to_check = min(cfg.top_k_per_segment * 5, n)
    top_idx = np.argsort(total)[::-1][:n_candidates_to_check]
    
    if verbose:
        print(f"{prefix} Step 6: Collision checking top {n_candidates_to_check} candidates")
    
    candidates = []
    n_collision_rejected = 0
    
    for rank, i in enumerate(top_idx):
        if total[i] == -np.inf:
            continue
        
        pos = valid_pts[i].copy()
        nrm = valid_normals[i].copy()
        
        # Check collision with scene + occupancy (excluding current segment)
        collision_free = True
        if collision_checker is not None:
            # More verbose output for debugging
            if verbose and rank < 5:
                print(f"{prefix}          Candidate #{rank}: pos=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}], "
                      f"normal=[{nrm[0]:.2f},{nrm[1]:.2f},{nrm[2]:.2f}], score={total[i]:.3f}")
            
            collision_free, n_collisions = collision_checker.check_gripper_collision(
                pos, nrm, cfg,
                segment_id=segment_id,  # Exclude this segment from collision check
                verbose=(verbose and n_collision_rejected < 3),  # Debug first few rejections
            )
            if not collision_free:
                n_collision_rejected += 1
                if verbose and n_collision_rejected <= 3:
                    print(f"{prefix}          -> REJECTED: {n_collisions} collision points")
                continue  # Skip this grasp
            elif verbose and rank < 5:
                print(f"{prefix}          -> ACCEPTED (collision-free)")
        
        candidates.append(GraspCandidate(
            position=pos,
            normal=nrm,
            score=float(total[i]),
            segment_id=segment_id,
            plane_rmse=float(plane_rmses[i]),
            rim_coverage=float(rim_coverage_scores[i]),
            seal_valid=True,
            collision_free=collision_free,
        ))
        
        # Stop once we have enough collision-free candidates
        if len(candidates) >= cfg.top_k_per_segment:
            break
    
    if verbose:
        print(f"{prefix} RESULT: {len(candidates)} grasps found, {n_collision_rejected} collision-rejected")
        if candidates:
            best = candidates[0]
            print(f"{prefix}   Best: score={best.score:.3f}, rim={best.rim_coverage*100:.0f}%, "
                  f"rmse={best.plane_rmse*1000:.1f}mm, pos=[{best.position[0]:.3f},{best.position[1]:.3f},{best.position[2]:.3f}]")
    
    return SegmentResult(segment_id, candidates, pts)


# =============================================================================
# POINT CLOUD EXTRACTION
# =============================================================================
class PointCloudData(NamedTuple):
    """Full point cloud data with segments and background."""
    segments: list[SegmentPointCloud]
    background_points: np.ndarray
    background_colors: np.ndarray


def extract_segment_pointclouds(depth_frame, masks, crop, full_shape, save_path=None) -> PointCloudData:
    """Extract point clouds for each mask and background."""
    h, w = full_shape
    
    pc = rs.pointcloud()
    pts_rs = pc.calculate(depth_frame)
    pts_cam = np.asanyarray(pts_rs.get_vertices()).view(np.float32).reshape(-1, 3)
    
    # Save raw pointcloud for later use
    if save_path is not None:
        np.save(save_path, pts_cam)
        print(f"  Saved pointcloud to {save_path}")
    
    valid = (pts_cam[:, 2] > 0.1) & (pts_cam[:, 2] < 3.0) & np.isfinite(pts_cam[:, 2])
    pts = transform_points(pts_cam, T_BASE_CAMERA)
    
    # Create crop region mask (restrict all points to cropped area)
    if crop:
        crop_region = np.zeros((h, w), dtype=bool)
        crop_region[crop[1]:crop[3], crop[0]:crop[2]] = True
        crop_mask = crop_region.flatten()
    else:
        crop_mask = np.ones(len(pts), dtype=bool)
    
    # Track which points belong to any segment
    any_segment_mask = np.zeros(len(pts), dtype=bool)
    
    segments = []
    for i, mask_dict in enumerate(masks):
        seg_mask = mask_dict['segmentation']
        
        if crop:
            full_mask = np.zeros((h, w), dtype=bool)
            full_mask[crop[1]:crop[3], crop[0]:crop[2]] = seg_mask
        else:
            full_mask = seg_mask
        
        obj_mask = valid & full_mask.flatten()
        any_segment_mask |= obj_mask
        obj_pts = pts[obj_mask]
        
        if len(obj_pts) < 100:
            continue
        
        np.random.seed(i)
        colors = np.tile(np.random.rand(3), (len(obj_pts), 1))
        segments.append(SegmentPointCloud(i, obj_pts, colors, seg_mask))
    
    # Extract background points (valid, in crop region, but not in any segment)
    background_mask = valid & crop_mask & ~any_segment_mask
    background_pts = pts[background_mask]
    # Grey color for background
    background_colors = np.tile([0.3, 0.3, 0.3], (len(background_pts), 1))
    
    return PointCloudData(segments, background_pts, background_colors)


# =============================================================================
# MAIN PIPELINE
# =============================================================================
class PipelineResult(NamedTuple):
    segments: list[SegmentPointCloud]
    grasp_results: list[SegmentResult]
    masks: list[dict]
    session_dir: Path
    background_points: np.ndarray = None
    background_colors: np.ndarray = None
    occupancy_points: np.ndarray = None
    grasp_cfg: GraspConfig = None


def run_pipeline(seg_cfg: ClutteredBinConfig = None, grasp_cfg: GraspConfig = None, save: bool = True):
    """Run complete grasp detection pipeline."""
    seg_cfg = seg_cfg or ClutteredBinConfig()
    grasp_cfg = grasp_cfg or GRASP_CFG
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Cluttered Bin Grasp Detection")
    print(f"Suction cup: {grasp_cfg.suction_cup_radius*2*1000:.0f}mm diameter")
    print(f"Min rim coverage: {grasp_cfg.min_rim_coverage_ratio*100:.0f}%")
    print("=" * 60)
    
    # Initialize
    mask_gen = load_sam2_auto_mask_generator(seg_cfg, device)
    pipeline, align, depth_scale, filters = init_realsense(seg_cfg)
    crop = load_crop_config(seg_cfg)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = seg_cfg.OUTPUT_DIR / f"grasp_{timestamp}"
    if save:
        session_dir.mkdir(parents=True, exist_ok=True)
    
    # Capture
    print("\n[Capture] Getting frame...")
    color, depth, depth_frame = get_rgbd_frame(pipeline, align, filters)
    
    # Get intrinsics before stopping pipeline
    profile = pipeline.get_active_profile()
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    depth_intr = depth_stream.get_intrinsics()
    intrinsics = {
        'fx': depth_intr.fx,
        'fy': depth_intr.fy,
        'cx': depth_intr.ppx,
        'cy': depth_intr.ppy,
        'width': depth_intr.width,
        'height': depth_intr.height,
    }
    
    pipeline.stop()
    
    if color is None:
        print("[Error] Capture failed")
        return None
    
    full_shape = (seg_cfg.HEIGHT, seg_cfg.WIDTH)
    color_crop = crop_image(color, crop)
    depth_crop = crop_image(depth, crop) if depth is not None else None
    
    if save:
        cv2.imwrite(str(session_dir / "color.png"), color_crop)
        if depth_crop is not None:
            np.save(session_dir / "depth.npy", depth_crop)
            # Normalize depth for visualization
            depth_vis = (depth_crop.astype(float) / depth_crop.max() * 255).astype(np.uint8)
            cv2.imwrite(str(session_dir / "depth.png"), depth_vis)
    
    # Segment
    print("\n[SAM2] Generating masks...")
    color_rgb = cv2.cvtColor(color_crop, cv2.COLOR_BGR2RGB)
    masks = generate_masks(mask_gen, color_rgb, device)
    print(f"  Raw: {len(masks)} masks")
    
    # Save raw segmentation
    if save:
        cv2.imwrite(str(session_dir / "seg_00_raw.png"), 
                    visualize_masks(color_crop, masks, seg_cfg.MASK_ALPHA))
    
    # Filter masks
    masks = filter_masks_by_area(masks, color_rgb.shape[:2], seg_cfg.MIN_SEGMENT_AREA, seg_cfg.MAX_AREA_RATIO)
    if save:
        cv2.imwrite(str(session_dir / "seg_01_area.png"),
                    visualize_masks(color_crop, masks, seg_cfg.MASK_ALPHA))
    
    if seg_cfg.FILTER_EDGE_MASKS:
        masks = filter_edge_masks(masks, color_rgb.shape[:2], seg_cfg.EDGE_MARGIN)
        if save:
            cv2.imwrite(str(session_dir / "seg_02_edge.png"),
                        visualize_masks(color_crop, masks, seg_cfg.MASK_ALPHA))
    
    if seg_cfg.MERGE_OVERLAP_THRESH > 0:
        masks = merge_overlapping_masks(masks, seg_cfg.MERGE_OVERLAP_THRESH)
        if save:
            cv2.imwrite(str(session_dir / "seg_03_merge.png"),
                        visualize_masks(color_crop, masks, seg_cfg.MASK_ALPHA))
    
    masks = filter_contained_masks(masks, 0.7)
    if save:
        cv2.imwrite(str(session_dir / "seg_04_contain.png"),
                    visualize_masks(color_crop, masks, seg_cfg.MASK_ALPHA))
    
    if seg_cfg.USE_DEPTH_FILTERING and depth_crop is not None:
        masks = filter_masks_by_depth(masks, depth_crop, depth_scale, seg_cfg.DEPTH_BACKGROUND_THRESHOLD_M, False)
        if save:
            cv2.imwrite(str(session_dir / "seg_05_depth.png"),
                        visualize_masks(color_crop, masks, seg_cfg.MASK_ALPHA))
    
    masks = sort_masks_by_depth(masks)
    print(f"  Final: {len(masks)} segments")
    
    if not masks:
        return None
    
    if save:
        cv2.imwrite(str(session_dir / "segmentation.png"),
                    visualize_masks(color_crop, masks, seg_cfg.MASK_ALPHA))
    
    # Extract point clouds
    print("\n[PointCloud] Extracting segments...")
    pc_save_path = session_dir / "pointcloud.npy" if save else None
    pc_data = extract_segment_pointclouds(depth_frame, masks, crop, full_shape, pc_save_path)
    segments = pc_data.segments
    print(f"  {len(segments)} valid segments, {len(pc_data.background_points):,} background points")
    
    if not segments:
        return None
    
    # Build collision checker (scene + occupancy volume)
    print("\n[Collision] Building scene collision checker...")
    collision_checker, occupancy_pts = build_scene_collision_data(
        segments,
        pc_data.background_points,
        T_BASE_CAMERA,
        grasp_cfg,
    )
    
    # Detect grasps with collision checking
    print("\n[Grasp] Detecting grasps with collision filtering...")
    results = []
    for seg in segments:
        res = detect_grasps(seg.points, seg.segment_id, grasp_cfg, collision_checker)
        results.append(res)
    
    # Summary
    total = sum(len(r.candidates) for r in results)
    print(f"\n[Done] {total} valid grasps from {len(segments)} segments")
    print(f"  Output: {session_dir}")
    
    # Save visualization data for later viewing
    if save:
        viz_data = {
            'segments': [
                {
                    'segment_id': seg.segment_id,
                    'points': seg.points,
                    'colors': seg.colors,
                }
                for seg in segments
            ],
            'grasp_results': [
                {
                    'segment_id': res.segment_id,
                    'candidates': [
                        {
                            'position': c.position,
                            'normal': c.normal,
                            'score': c.score,
                            'segment_id': c.segment_id,
                            'plane_rmse': c.plane_rmse,
                            'rim_coverage': c.rim_coverage,
                            'seal_valid': c.seal_valid,
                            'collision_free': c.collision_free,
                        }
                        for c in res.candidates
                    ],
                }
                for res in results
            ],
            'background_points': pc_data.background_points,
            'background_colors': pc_data.background_colors,
            'suction_cup_diameter': grasp_cfg.suction_cup_radius * 2,
            'color_image': color_crop,
            'depth_image': depth_crop if depth_crop is not None else np.array([]),
            'depth_scale': depth_scale,
            'intrinsics': intrinsics,
            'crop': crop if crop else [0, 0, seg_cfg.WIDTH, seg_cfg.HEIGHT],
            'T_base_camera': T_BASE_CAMERA,
        }
        np.savez(session_dir / "viz_data.npz", **viz_data)
        print(f"  Saved viz_data.npz for later viewing")
    
    return PipelineResult(
        segments, results, masks, session_dir,
        pc_data.background_points, pc_data.background_colors,
        occupancy_pts, grasp_cfg,
    )


def main():
    parser = argparse.ArgumentParser(description="Cluttered bin grasp detection")
    parser.add_argument("--meshcat", action="store_true", 
                        help="Use meshcat web visualization instead of Open3D")
    parser.add_argument("--meshcat-url", type=str, default=None,
                        help="ZMQ URL for meshcat server (e.g., tcp://127.0.0.1:6000)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization")
    args = parser.parse_args()
    
    result = run_pipeline()
    
    if result is None:
        print("No objects detected.")
        return
    
    if args.no_viz:
        print("[Done] Skipping visualization (--no-viz)")
        return
    
    # Visualize with occupancy volume and gripper meshes
    visualize_results(
        result.segments, 
        result.grasp_results, 
        result.grasp_cfg,
        result.background_points,
        result.background_colors,
        result.occupancy_points,
        use_meshcat=args.meshcat,
        meshcat_url=args.meshcat_url,
    )


if __name__ == "__main__":
    main()
