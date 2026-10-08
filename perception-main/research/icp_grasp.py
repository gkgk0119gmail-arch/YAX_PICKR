"""
ICP-based pose estimation: match 3D models to detected/segmented objects.
Includes grasp planning for suction cup gripper.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tf
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

from detection_pipeline import (
    DetectionPipeline,
    PipelineConfig,
    PipelineResult,
    generate_colors,
    uncrop_mask,
)


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class ICPConfig:
    """Configuration for ICP pose estimation."""
    
    # Model loading
    model_dir: Path = Path("/home/jeong/perception/data")
    model_voxel_size: float = 0.002  # 2mm voxel for model downsampling
    
    # Segment processing
    segment_voxel_size: float = 0.002
    min_segment_points: int = 100
    
    # ICP parameters
    icp_max_iterations: int = 100
    icp_distance_scales: tuple[float, ...] = (0.05, 0.02, 0.01, 0.005)
    
    # Multi-pose initialization (crucial for symmetric objects)
    num_initial_rotations: int = 8
    try_flipped: bool = True
    
    # === VALIDATION THRESHOLDS ===
    # Fitness: fraction of model points with correspondence in segment
    min_fitness: float = 0.2
    
    # RMSE: average distance between matched points (meters)
    max_rmse: float = 0.005  # 5mm
    
    # Coverage: fraction of segment points covered by model
    # LOW coverage = model is too small for segment (e.g., can vs book)
    min_coverage: float = 0.3  # At least 30% of segment must be covered
    coverage_distance: float = 0.01  # Points within 10mm are "covered"
    
    # Chamfer: average bidirectional distance (most reliable metric!)
    # Measures actual shape similarity after alignment
    max_chamfer: float = 0.010  # 10mm - stricter catches wrong matches


ICP_CFG = ICPConfig()

# Grasp labels file
GRASP_LABELS_FILE = Path("/home/jeong/perception/app/grasp_labels.json")


# =============================================================================
# GRASP PLANNING
# =============================================================================
@dataclass
class GraspCandidate:
    """A candidate grasp pose in object frame."""
    name: str                    # Human-readable name (e.g., "top", "side_x")
    position: np.ndarray         # Contact point in object frame
    approach: np.ndarray         # Approach direction (unit vector, points INTO object)
    quality: float = 1.0         # Base quality score (higher = preferred)
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)
        self.approach = np.asarray(self.approach, dtype=np.float64)
        self.approach = self.approach / np.linalg.norm(self.approach)


@dataclass 
class GraspConfig:
    """Grasp configuration for a specific object model."""
    model_name: str
    candidates: list[GraspCandidate] = field(default_factory=list)


def load_grasp_labels(labels_file: Path = GRASP_LABELS_FILE) -> dict[str, GraspConfig]:
    """Load grasp configurations from JSON labels file."""
    if not labels_file.exists():
        print(f"[Grasp] Warning: No labels file at {labels_file}")
        return {}
    
    with open(labels_file, 'r') as f:
        data = json.load(f)
    
    configs = {}
    for model_name, grasp_list in data.items():
        candidates = [
            GraspCandidate(
                name=g["name"],
                position=g["position"],
                approach=g["approach"],
                quality=g.get("quality", 1.0),
            )
            for g in grasp_list
        ]
        configs[model_name] = GraspConfig(
            model_name=model_name,
            candidates=candidates,
        )
        print(f"[Grasp] Loaded {len(candidates)} grasps for '{model_name}'")
    
    return configs


@dataclass
class GraspResult:
    """Selected grasp in world frame."""
    model_name: str
    grasp_name: str
    position: np.ndarray         # Grasp contact point in world frame
    approach: np.ndarray         # Approach direction in world frame
    gripper_pose: np.ndarray     # 4x4 transform for gripper (Z = approach direction)
    score: float                 # Selection score
    

# Gripper dimensions (suction cup)
GRIPPER_CUP_DIAMETER = 0.03      # 3cm diameter
GRIPPER_CUP_HEIGHT = 0.015       # 1.5cm height
GRIPPER_ARM_LENGTH = 0.12        # 12cm arm length
GRIPPER_ARM_WIDTH = 0.025        # 2.5cm arm width
GRIPPER_ARM_HEIGHT = 0.02        # 2cm arm height


def select_best_grasp(
    model_name: str,
    object_pose: np.ndarray,  # 4x4 transform from object to world
    grasp_configs: dict[str, GraspConfig],
    min_upward_component: float = 0.3,  # Minimum Z component of approach in world
) -> GraspResult | None:
    """
    Select the best grasp for an object based on its pose.
    
    Strategy:
    1. Transform all grasp candidates to world frame
    2. Score each by how well the approach direction aligns with -Z (gravity)
       - Suction cups work best approaching from above
    3. Return the highest-scoring valid grasp
    
    Args:
        model_name: Name of the object model
        object_pose: 4x4 transformation matrix (object frame -> world frame)
        grasp_configs: Dictionary of grasp configurations
        min_upward_component: Minimum -Z component required for valid grasp
        
    Returns:
        GraspResult with the selected grasp, or None if no valid grasp found
    """
    if model_name not in grasp_configs:
        print(f"[Grasp] No grasp config for model '{model_name}'")
        return None
    
    config = grasp_configs[model_name]
    R = object_pose[:3, :3]  # Rotation matrix
    t = object_pose[:3, 3]   # Translation
    
    best_grasp: GraspResult | None = None
    best_score = -float('inf')
    
    print(f"[Grasp] Evaluating {len(config.candidates)} candidates for '{model_name}':")
    
    for candidate in config.candidates:
        # Transform to world frame
        pos_world = R @ candidate.position + t
        approach_world = R @ candidate.approach
        approach_world = approach_world / np.linalg.norm(approach_world)
        
        # Score: how much does approach point downward (-Z)?
        # approach_world should point INTO object, so gripper comes from -approach direction
        # We want gripper to come from above, meaning -approach should have positive Z
        # Equivalently: approach should have negative Z
        upward_score = -approach_world[2]  # Higher = gripper approaches more from above
        
        # Combined score: upward alignment * base quality
        score = upward_score * candidate.quality
        
        print(f"  {candidate.name}: approach_world={approach_world}, "
              f"upward={upward_score:.2f}, score={score:.2f}")
        
        # Check if this is a valid grasp (gripper can approach from above)
        if upward_score < min_upward_component:
            print(f"    -> SKIP (approach not upward enough)")
            continue
        
        if score > best_score:
            best_score = score
            
            # Compute gripper pose (Z-axis = approach direction, pointing into object)
            gripper_pose = compute_gripper_pose(pos_world, approach_world)
            
            best_grasp = GraspResult(
                model_name=model_name,
                grasp_name=candidate.name,
                position=pos_world,
                approach=approach_world,
                gripper_pose=gripper_pose,
                score=score,
            )
    
    if best_grasp:
        print(f"  -> SELECTED: {best_grasp.grasp_name} (score={best_grasp.score:.2f})")
    else:
        print(f"  -> NO VALID GRASP (none facing upward)")
    
    return best_grasp


def compute_gripper_pose(position: np.ndarray, approach: np.ndarray) -> np.ndarray:
    """
    Compute 4x4 gripper pose matrix.
    
    Convention:
    - Gripper Z-axis points along approach direction (into the object)
    - Gripper origin is at the contact point
    - X and Y are orthogonal, with Y preferring to point "up" in world
    """
    z_axis = approach / np.linalg.norm(approach)
    
    # Choose Y to be as "up" as possible (world Z direction)
    world_up = np.array([0, 0, 1])
    
    # If approach is nearly vertical, use world X as reference instead
    if abs(np.dot(z_axis, world_up)) > 0.9:
        world_up = np.array([1, 0, 0])
    
    # Gram-Schmidt to get orthonormal basis
    x_axis = np.cross(world_up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Build transformation matrix
    T = np.eye(4)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = position
    
    return T


# =============================================================================
# TRANSFORMS
# =============================================================================
T_BASE_CAMERA = np.array([
    [-0.99870602, -0.05068679,  0.00413055,  0.46783590],
    [-0.05061085,  0.99857739,  0.01678431,  0.11403400],
    [-0.00497542,  0.01655354, -0.99985062,  1.02680489],
    [ 0.0,         0.0,         0.0,         1.0        ],
], dtype=np.float64)


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform Nx3 points using 4x4 transformation matrix."""
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    points_h = np.hstack([points, ones])
    return (T @ points_h.T).T[:, :3]


# =============================================================================
# MODEL LOADING
# =============================================================================
def load_obj_as_pointcloud(filepath: Path, voxel_size: float) -> o3d.geometry.PointCloud:
    """Load OBJ file as downsampled point cloud."""
    mesh = o3d.io.read_triangle_mesh(str(filepath))
    
    if not mesh.has_vertices():
        raise ValueError(f"OBJ file has no vertices: {filepath}")
    
    # Sample points from mesh surface
    if mesh.has_triangles() and len(mesh.triangles) > 0:
        n_points = max(10000, len(mesh.vertices) * 5)
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
    
    # Downsample
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 4, max_nn=30
    ))
    
    return pcd


def load_all_models(model_dir: Path, voxel_size: float) -> dict[str, o3d.geometry.PointCloud]:
    """Load all OBJ files from directory."""
    models = {}
    obj_files = list(model_dir.glob("*.obj"))
    
    if not obj_files:
        print(f"[Models] No OBJ files found in {model_dir}")
        return models
    
    print(f"[Models] Loading {len(obj_files)} OBJ files from {model_dir}")
    
    for obj_path in obj_files:
        try:
            pcd = load_obj_as_pointcloud(obj_path, voxel_size)
            model_name = obj_path.stem
            models[model_name] = pcd
            
            bbox = pcd.get_axis_aligned_bounding_box()
            extent = bbox.get_extent()
            print(f"  [{model_name}] {len(pcd.points):,} pts, "
                  f"size: {extent[0]*1000:.1f} x {extent[1]*1000:.1f} x {extent[2]*1000:.1f} mm")
        except Exception as e:
            print(f"  [Warning] Failed to load {obj_path.name}: {e}")
    
    return models


# =============================================================================
# SEGMENT EXTRACTION
# =============================================================================
class SegmentPointCloud(NamedTuple):
    """Point cloud for a single segmented object."""
    segment_id: int
    name: str
    points: np.ndarray
    pcd: o3d.geometry.PointCloud
    color: tuple[int, int, int]


def extract_segment_pointclouds(
    result: PipelineResult,
    voxel_size: float = 0.002,
    min_points: int = 100,
) -> list[SegmentPointCloud]:
    """Extract point clouds from pipeline result."""
    h, w = result.full_shape
    
    # Apply depth filtering
    print("[PointCloud] Applying depth filters...")
    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.15)
    temporal.set_option(rs.option.filter_smooth_delta, 50)
    hole_fill = rs.hole_filling_filter()
    hole_fill.set_option(rs.option.holes_fill, 1)
    
    filtered = temporal.process(result.depth_frame)
    filtered = hole_fill.process(filtered)
    
    # Compute point cloud
    print("[PointCloud] Computing point cloud...")
    rs_pc = rs.pointcloud()
    points_rs = rs_pc.calculate(filtered)
    points_camera = np.asanyarray(points_rs.get_vertices()).view(np.float32).reshape(-1, 3)
    
    # Filter valid points
    valid_mask = (points_camera[:, 2] > 0.1) & (points_camera[:, 2] < 3.0) & np.isfinite(points_camera[:, 2])
    print(f"[PointCloud] {valid_mask.sum():,} valid points")
    
    # Transform to robot base frame
    points = transform_points(points_camera, T_BASE_CAMERA)
    
    # Process masks
    full_masks = []
    for mask in result.masks:
        if result.crop is not None:
            full_mask = uncrop_mask(mask, result.crop, (h, w))
        else:
            full_mask = mask
        full_masks.append(full_mask.flatten())
    
    colors = generate_colors(len(full_masks))
    segments: list[SegmentPointCloud] = []
    
    for i, (fm, det, color) in enumerate(zip(full_masks, result.detections, colors)):
        obj_mask = valid_mask & (fm > 0)
        obj_points = points[obj_mask]
        
        if obj_points.shape[0] < min_points:
            print(f"  [Segment {i}] {det['name']}: skipped ({obj_points.shape[0]} < {min_points} pts)")
            continue
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_points.astype(np.float64))
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 4, max_nn=30
        ))
        
        segments.append(SegmentPointCloud(
            segment_id=i,
            name=det.get("name", f"object_{i}"),
            points=np.asarray(pcd.points),
            pcd=pcd,
            color=color,
        ))
        print(f"  [Segment {i}] {det['name']}: {len(pcd.points):,} pts")
    
    return segments


# =============================================================================
# ICP MATCHING
# =============================================================================
class ICPResult(NamedTuple):
    """Result of ICP alignment."""
    model_name: str
    transformation: np.ndarray
    fitness: float
    rmse: float
    coverage: float
    chamfer: float
    position: np.ndarray
    rotation_matrix: np.ndarray


def compute_coverage(model_points: np.ndarray, segment_points: np.ndarray, threshold: float) -> float:
    """What fraction of SEGMENT points are within threshold distance of MODEL points."""
    if len(model_points) == 0 or len(segment_points) == 0:
        return 0.0
    
    model_tree = o3d.geometry.KDTreeFlann(
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model_points))
    )
    
    covered = 0
    for pt in segment_points:
        _, _, dist = model_tree.search_knn_vector_3d(pt, 1)
        if len(dist) > 0 and np.sqrt(dist[0]) < threshold:
            covered += 1
    
    return covered / len(segment_points)


def compute_chamfer(points_a: np.ndarray, points_b: np.ndarray) -> float:
    """Chamfer distance: average bidirectional nearest-neighbor distance."""
    if len(points_a) == 0 or len(points_b) == 0:
        return float('inf')
    
    # A -> B
    pcd_b = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_b))
    tree_b = o3d.geometry.KDTreeFlann(pcd_b)
    dist_a = [np.sqrt(tree_b.search_knn_vector_3d(pt, 1)[2][0]) for pt in points_a]
    
    # B -> A
    pcd_a = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_a))
    tree_a = o3d.geometry.KDTreeFlann(pcd_a)
    dist_b = [np.sqrt(tree_a.search_knn_vector_3d(pt, 1)[2][0]) for pt in points_b]
    
    return (np.mean(dist_a) + np.mean(dist_b)) / 2


def generate_initial_transforms(model_pcd: o3d.geometry.PointCloud, 
                                 segment_pcd: o3d.geometry.PointCloud,
                                 config: ICPConfig) -> list[np.ndarray]:
    """Generate multiple initial poses for robust ICP."""
    model_center = np.asarray(model_pcd.get_center())
    segment_center = np.asarray(segment_pcd.get_center())
    
    transforms = []
    for i in range(config.num_initial_rotations):
        angle = 2 * np.pi * i / config.num_initial_rotations
        c, s = np.cos(angle), np.sin(angle)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        T = np.eye(4)
        T[:3, :3] = Rz
        T[:3, 3] = segment_center - Rz @ model_center
        transforms.append(T)
        
        if config.try_flipped:
            Rx_flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            R_flipped = Rz @ Rx_flip
            T_flipped = np.eye(4)
            T_flipped[:3, :3] = R_flipped
            T_flipped[:3, 3] = segment_center - R_flipped @ model_center
            transforms.append(T_flipped)
    
    return transforms


def run_icp(model_pcd: o3d.geometry.PointCloud, segment_pcd: o3d.geometry.PointCloud,
            initial_T: np.ndarray, config: ICPConfig) -> tuple[np.ndarray, float, float]:
    """Run multi-scale ICP."""
    current_T = initial_T.copy()
    
    for dist_thresh in config.icp_distance_scales:
        result = o3d.pipelines.registration.registration_icp(
            model_pcd, segment_pcd, dist_thresh, current_T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=config.icp_max_iterations)
        )
        current_T = result.transformation
    
    return result.transformation, result.fitness, result.inlier_rmse


def match_model_to_segment(
    model_name: str,
    model_pcd: o3d.geometry.PointCloud,
    segment: SegmentPointCloud,
    config: ICPConfig,
) -> tuple[ICPResult | None, str]:
    """Match model to segment using multi-pose ICP + validation."""
    
    # Try multiple initial poses
    initial_transforms = generate_initial_transforms(model_pcd, segment.pcd, config)
    
    best_T, best_fitness, best_rmse = None, -1.0, float('inf')
    for init_T in initial_transforms:
        T, fitness, rmse = run_icp(model_pcd, segment.pcd, init_T, config)
        if fitness > best_fitness:
            best_T, best_fitness, best_rmse = T, fitness, rmse
    
    if best_T is None:
        return None, "ICP failed"
    
    # === VALIDATION ===
    
    # 1. Fitness check
    if best_fitness < config.min_fitness:
        return None, f"low fitness ({best_fitness:.3f} < {config.min_fitness})"
    
    # 2. RMSE check
    if best_rmse > config.max_rmse:
        return None, f"high RMSE ({best_rmse*1000:.1f}mm > {config.max_rmse*1000:.1f}mm)"
    
    # Transform model for validation
    model_pts = np.asarray(model_pcd.points)
    model_transformed = transform_points(model_pts.astype(np.float64), best_T)
    segment_pts = np.asarray(segment.pcd.points)
    
    # 3. Coverage check (catches small model fitting inside large segment)
    coverage = compute_coverage(model_transformed, segment_pts, config.coverage_distance)
    if coverage < config.min_coverage:
        return None, f"low coverage ({coverage:.1%} < {config.min_coverage:.0%})"
    
    # 4. Chamfer check (most reliable - actual shape similarity)
    chamfer = compute_chamfer(model_transformed, segment_pts)
    if chamfer > config.max_chamfer:
        return None, f"high chamfer ({chamfer*1000:.1f}mm > {config.max_chamfer*1000:.1f}mm)"
    
    # All checks passed
    return ICPResult(
        model_name=model_name,
        transformation=best_T,
        fitness=best_fitness,
        rmse=best_rmse,
        coverage=coverage,
        chamfer=chamfer,
        position=best_T[:3, 3],
        rotation_matrix=best_T[:3, :3],
    ), "ok"


def match_all_models_to_segments(
    models: dict[str, o3d.geometry.PointCloud],
    segments: list[SegmentPointCloud],
    config: ICPConfig,
) -> dict[int, ICPResult]:
    """Match all models to all segments."""
    results: dict[int, ICPResult] = {}
    
    n_poses = config.num_initial_rotations * (2 if config.try_flipped else 1)
    print(f"\n[ICP] Matching {len(models)} models to {len(segments)} segments")
    print(f"[ICP] {n_poses} initial poses | min_fitness={config.min_fitness} | "
          f"min_coverage={config.min_coverage:.0%} | max_chamfer={config.max_chamfer*1000:.0f}mm")
    print("-" * 60)
    
    for segment in segments:
        seg_extent = np.sort(segment.pcd.get_axis_aligned_bounding_box().get_extent())[::-1]
        print(f"\n[Segment {segment.segment_id}] {segment.name}")
        print(f"  {len(segment.pcd.points)} pts, size: {seg_extent[0]*1000:.0f}x{seg_extent[1]*1000:.0f}x{seg_extent[2]*1000:.0f}mm")
        
        best_result: ICPResult | None = None
        
        for model_name, model_pcd in models.items():
            model_extent = np.sort(model_pcd.get_axis_aligned_bounding_box().get_extent())[::-1]
            print(f"  Testing '{model_name}' ({model_extent[0]*1000:.0f}x{model_extent[1]*1000:.0f}x{model_extent[2]*1000:.0f}mm)...", end=" ")
            
            result, reason = match_model_to_segment(model_name, model_pcd, segment, config)
            
            if result is not None:
                print(f"✓ fit={result.fitness:.2f} cov={result.coverage:.0%} chamfer={result.chamfer*1000:.1f}mm")
                if best_result is None or result.fitness > best_result.fitness:
                    best_result = result
            else:
                print(f"✗ {reason}")
        
        if best_result is not None:
            results[segment.segment_id] = best_result
            print(f"  → MATCHED: '{best_result.model_name}'")
        else:
            print(f"  → No match")
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================
def visualize_gripper(
    vis: meshcat.Visualizer,
    name: str,
    grasp: GraspResult,
    color: int = 0x00ff00,
    opacity: float = 0.85,
) -> None:
    """
    Visualize the suction cup gripper at the grasp pose as a cylinder cup.
    
    The gripper Z-axis points into the object (approach direction).
    """
    gripper_group = vis["grippers"][name]
    
    # Suction cup - cylinder centered just behind contact point
    cup_offset = -GRIPPER_CUP_HEIGHT / 2  # Move back by half height
    cup_transform = grasp.gripper_pose.copy()
    cup_transform[:3, 3] += cup_transform[:3, 2] * cup_offset
    
    gripper_group["cup"].set_object(
        geom.Cylinder(GRIPPER_CUP_HEIGHT, GRIPPER_CUP_DIAMETER / 2),
        geom.MeshLambertMaterial(color=color, opacity=opacity)
    )
    # Cylinder in meshcat is Y-up, we need Z-up (along approach)
    # Rotate 90° around X to make cylinder axis align with Z
    cup_rotation = tf.rotation_matrix(np.pi/2, [1, 0, 0])
    gripper_group["cup"].set_transform(cup_transform @ cup_rotation)
    
    # Approach arrow (white line showing gripper approach direction)
    arrow_length = 0.05
    arrow_start = grasp.position - grasp.approach * arrow_length
    arrow_end = grasp.position
    axis_pts = np.array([arrow_start, arrow_end], dtype=np.float32).T
    gripper_group["arrow"].set_object(
        geom.Line(geom.PointsGeometry(axis_pts), 
                  geom.LineBasicMaterial(color=0xffffff, linewidth=3))
    )
    
    print(f"[Meshcat] Grasp '{grasp.grasp_name}' at pos=[{grasp.position[0]:.3f}, "
          f"{grasp.position[1]:.3f}, {grasp.position[2]:.3f}]")


def visualize_results(
    segments: list[SegmentPointCloud],
    models: dict[str, o3d.geometry.PointCloud],
    results: dict[int, ICPResult],
    grasps: dict[int, GraspResult] | None = None,
) -> meshcat.Visualizer:
    """Visualize in Meshcat."""
    vis = meshcat.Visualizer().open()
    
    # Coordinate frame
    L = 0.1
    for axis, color, vec in [("X", 0xff0000, [L,0,0]), ("Y", 0x00ff00, [0,L,0]), ("Z", 0x0000ff, [0,0,L])]:
        pts = np.array([[0,0,0], vec], dtype=np.float32).T
        vis["frame"][axis].set_object(geom.Line(geom.PointsGeometry(pts), geom.LineBasicMaterial(color=color, linewidth=3)))
    
    # Models side-by-side near origin
    model_spacing = 0.15
    model_names = list(models.keys())
    
    for i, name in enumerate(model_names):
        pcd = models[name]
        pts = np.asarray(pcd.points).copy()
        x_offset = (i - (len(model_names) - 1) / 2) * model_spacing
        pts[:, 0] += x_offset
        
        vis["models"][name].set_object(geom.Points(
            geom.PointsGeometry(pts.T.astype(np.float32)),
            geom.PointsMaterial(size=0.003, color=0x00ffff)
        ))
        
        print(f"[Meshcat] Model '{name}' at X={x_offset*100:.0f}cm (cyan)")
    
    # Segments
    for seg in segments:
        hex_color = (seg.color[2] << 16) | (seg.color[1] << 8) | seg.color[0]
        vis["segments"][f"seg_{seg.segment_id}"].set_object(geom.Points(
            geom.PointsGeometry(seg.points.T.astype(np.float32)),
            geom.PointsMaterial(size=0.0015, color=hex_color)
        ))
        print(f"[Meshcat] Segment {seg.segment_id} ({seg.name})")
        
        # Fitted model in RED
        if seg.segment_id in results:
            r = results[seg.segment_id]
            model_pts = np.asarray(models[r.model_name].points)
            fitted_pts = transform_points(model_pts.astype(np.float64), r.transformation)
            
            vis["fitted"][f"fit_{seg.segment_id}"].set_object(geom.Points(
                geom.PointsGeometry(fitted_pts.T.astype(np.float32)),
                geom.PointsMaterial(size=0.006, color=0xff0000)
            ))
            
            # Pose axes
            for axis, color, d in [("X", 0xff0000, [1,0,0]), ("Y", 0x00ff00, [0,1,0]), ("Z", 0x0000ff, [0,0,1])]:
                end = r.position + r.rotation_matrix @ np.array(d) * 0.05
                pts = np.array([r.position, end], dtype=np.float32).T
                vis["poses"][f"seg_{seg.segment_id}"][axis].set_object(
                    geom.Line(geom.PointsGeometry(pts), geom.LineBasicMaterial(color=color, linewidth=2))
                )
            
            print(f"[Meshcat] Fitted '{r.model_name}' → segment {seg.segment_id} (RED)")
            
            # Visualize grasp if available (green cylinder cup)
            if grasps and seg.segment_id in grasps:
                grasp = grasps[seg.segment_id]
                visualize_gripper(vis, f"seg_{seg.segment_id}", grasp)
    
    print("-" * 60)
    print(f"[Meshcat] URL: {vis.url()}")
    print("Legend: Cyan=models, Colors=segments, RED=fitted model, GREEN=suction grasp")
    return vis


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="ICP pose estimation")
    parser.add_argument("--models", type=Path, default=ICP_CFG.model_dir)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()
    
    config = ICPConfig(model_dir=args.models)
    
    print("=" * 60)
    print("ICP POSE ESTIMATION")
    print("=" * 60)
    
    # Load models
    models = load_all_models(config.model_dir, config.model_voxel_size)
    if not models:
        print("[Error] No models loaded")
        return
    
    # Load grasp labels
    print("\n" + "-" * 60)
    grasp_configs = load_grasp_labels(GRASP_LABELS_FILE)
    if not grasp_configs:
        print("[Warning] No grasp labels loaded - grasp planning will be skipped")
    
    # Run detection pipeline
    print("\n" + "-" * 60)
    pipeline = DetectionPipeline(PipelineConfig())
    result = pipeline.run()
    
    if result is None or result.depth_frame is None or not result.masks:
        print("[Done] No detections")
        return
    
    # Extract segments
    print("\n" + "-" * 60)
    segments = extract_segment_pointclouds(result, config.segment_voxel_size, config.min_segment_points)
    if not segments:
        print("[Done] No valid segments")
        return
    
    # ICP matching
    icp_results = match_all_models_to_segments(models, segments, config)
    
    # Grasp planning (using loaded labels)
    print("\n" + "-" * 60)
    print("[Grasp Planning]")
    grasp_results: dict[int, GraspResult] = {}
    for seg in segments:
        if seg.segment_id in icp_results:
            r = icp_results[seg.segment_id]
            grasp = select_best_grasp(r.model_name, r.transformation, grasp_configs)
            if grasp:
                grasp_results[seg.segment_id] = grasp
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for seg in segments:
        if seg.segment_id in icp_results:
            r = icp_results[seg.segment_id]
            print(f"[{seg.segment_id}] {seg.name} → {r.model_name} "
                  f"(fit={r.fitness:.2f}, cov={r.coverage:.0%}, chamfer={r.chamfer*1000:.1f}mm)")
            print(f"    Object pose: [{r.position[0]:.3f}, {r.position[1]:.3f}, {r.position[2]:.3f}]")
            
            if seg.segment_id in grasp_results:
                g = grasp_results[seg.segment_id]
                print(f"    Grasp: '{g.grasp_name}' at [{g.position[0]:.3f}, {g.position[1]:.3f}, {g.position[2]:.3f}]")
                print(f"    Approach: [{g.approach[0]:.3f}, {g.approach[1]:.3f}, {g.approach[2]:.3f}]")
            else:
                print(f"    Grasp: NO VALID GRASP")
        else:
            print(f"[{seg.segment_id}] {seg.name} → NO MATCH")
    
    # Visualize
    if not args.no_viz:
        vis = visualize_results(segments, models, icp_results, grasp_results)
        input("\n[Meshcat] Press Enter to exit...")
    
    print(f"\n[Done] Session: {result.session_dir}")


if __name__ == "__main__":
    main()
