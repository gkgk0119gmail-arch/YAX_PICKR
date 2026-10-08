"""
ICP-based pose estimation: match 3D models to detected/segmented objects.
Uses FPFH + RANSAC + ICP for robust global registration.
Includes grasp planning for suction cup gripper.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

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

from fpfh_ransac import (
    preprocess_point_cloud,
    ransac_registration,
    icp_refinement,
    multi_start_icp,
    compute_alignment_score,
    center_point_cloud,
    # Parameters we may want to use
    MODEL_VOXEL_SIZE,
    SCENE_VOXEL_SIZE,
    ICP_DISTANCE_THRESHOLD,
    MULTI_START_N_ROTATIONS,
    MULTI_START_AXES,
)

from visualization import (
    visualize_pipeline_results_open3d,
    visualize_pipeline_results_meshcat,
)

from utils import (
    load_grasp_labels,
    load_all_models,
    transform_points,
    save_segments,
    T_BASE_CAMERA,
    GraspCandidate,
    GraspConfig,
)


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class ICPConfig:
    """Configuration for FPFH+RANSAC+ICP pose estimation."""
    
    # Model loading
    model_dir: Path = Path("/home/jeong/perception/data")
    model_voxel_size: float = MODEL_VOXEL_SIZE  # Use same as fpfh_ransac
    
    # Segment processing
    segment_voxel_size: float = SCENE_VOXEL_SIZE  # Use same as fpfh_ransac
    min_segment_points: int = 100
    
    # ICP parameters (from fpfh_ransac)
    icp_distance_threshold: float = ICP_DISTANCE_THRESHOLD
    
    # Multi-start ICP
    use_multi_start_icp: bool = True
    multi_start_n_rotations: int = MULTI_START_N_ROTATIONS
    multi_start_axes: list = field(default_factory=lambda: list(MULTI_START_AXES))
    
    # === VALIDATION THRESHOLDS (based on compute_alignment_score) ===
    # Overall score from compute_alignment_score (0-1, higher is better)
    # Quality guide: >0.8 excellent, >0.6 good, >0.3 moderate, >0.15 poor
    min_overall_score: float = 0.15  # Minimum to consider a valid match
    
    # Bidirectional fitness (F1-like score of coverage & precision)
    min_bidirectional_fitness: float = 0.35
    
    # Chamfer distance: average bidirectional nearest-neighbor distance (meters)
    # Lower is better - measures actual shape similarity
    max_chamfer_distance: float = 0.01  # 10mm max
    
    # Model-to-Scene coverage: fraction of model points with nearby scene points
    # Higher is better - ensures model is well-represented in scene
    min_m2s_coverage: float = 0.25  # At least 30% of model covered


ICP_CFG = ICPConfig()

# =============================================================================
# GRASP PLANNING
# =============================================================================
@dataclass
class GraspResult:
    """Selected grasp in world frame."""
    model_name: str
    grasp_name: str
    position: np.ndarray         # Grasp contact point in world frame
    approach: np.ndarray         # Approach direction in world frame
    gripper_pose: np.ndarray     # 4x4 transform for gripper (Z = approach direction)
    score: float                 # Selection score
    

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
    color_image: np.ndarray | None = None,
) -> list[SegmentPointCloud]:
    """
    Extract point clouds from pipeline result with colors.
    
    Args:
        result: PipelineResult from detection pipeline
        voxel_size: Voxel size for downsampling
        min_points: Minimum points for a valid segment
        color_image: BGR color image for coloring points (optional)
        
    Returns:
        List of SegmentPointCloud namedtuples with colored point clouds
    """
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
    
    # Get colors from image if available
    point_colors = None
    if color_image is not None:
        # Map 3D points back to 2D image coordinates for color lookup
        # Points are in (w*h, 3) order matching depth image pixels
        pixel_indices = np.arange(w * h)
        pixel_y = pixel_indices // w
        pixel_x = pixel_indices % w
        
        # Get BGR colors and convert to RGB normalized
        bgr_colors = color_image[pixel_y, pixel_x]  # (N, 3) BGR
        point_colors = bgr_colors[:, ::-1].astype(np.float64) / 255.0  # RGB normalized
    
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
        
        # Add colors to segment - use real colors if available, otherwise use segment color
        if point_colors is not None:
            seg_colors = point_colors[obj_mask]
            pcd.colors = o3d.utility.Vector3dVector(seg_colors)
        else:
            # Fall back to segment color (BGR -> RGB normalized)
            rgb_color = [color[2]/255.0, color[1]/255.0, color[0]/255.0]
            pcd.paint_uniform_color(rgb_color)
        
        # NOTE: Commented out - preprocess_point_cloud() will do this anyway
        # pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        #     radius=voxel_size * 4, max_nn=30
        # ))
        
        segments.append(SegmentPointCloud(
            segment_id=i,
            name=det.get("name", f"object_{i}"),
            points=np.asarray(pcd.points),
            pcd=pcd,
            color=color,
        ))
        has_colors = pcd.has_colors()
        print(f"  [Segment {i}] {det['name']}: {len(pcd.points):,} pts, colors={has_colors}")
    
    return segments


# =============================================================================
# FPFH + RANSAC + ICP MATCHING
# =============================================================================
class RegistrationResult(NamedTuple):
    """Result of FPFH+RANSAC+ICP registration."""
    model_name: str
    transformation: np.ndarray
    # Alignment scores from compute_alignment_score
    overall_score: float         # Composite 0-1 score (higher = better)
    bidirectional_fitness: float # F1-like score of coverage & precision
    chamfer_distance: float      # Symmetric chamfer distance (meters, lower = better)
    m2s_coverage: float          # Model-to-scene coverage (fraction)
    s2m_precision: float         # Scene-to-model precision (fraction)
    m2s_rmse: float              # Model-to-scene RMSE (meters)
    s2m_rmse: float              # Scene-to-model RMSE (meters)
    combined_rmse: float         # Average RMSE (meters)
    position: np.ndarray
    rotation_matrix: np.ndarray


def match_model_to_segment_fpfh(
    model_name: str,
    model_pcd: o3d.geometry.PointCloud,
    segment: SegmentPointCloud,
    config: ICPConfig,
) -> tuple[RegistrationResult | None, str, dict | None]:
    """
    Match model to segment using FPFH + RANSAC + ICP.
    
    Steps:
    1. Center model at origin (critical for RANSAC rotations)
    2. Preprocess both model and segment (compute FPFH features)
    3. RANSAC global registration using feature matching
    4. ICP refinement (optionally multi-start)
    5. Score using compute_alignment_score
    6. Adjust transform to account for model centering
    7. Validate and return result
    
    Returns:
        (result, reason, scores) - scores dict is returned even for rejections
    """
    print(f"    [FPFH+RANSAC+ICP] Processing...")
    
    # Step 0: Center model at origin (critical for RANSAC to work properly)
    # RANSAC rotations happen around origin, so model must be centered
    model_centered, model_centroid = center_point_cloud(model_pcd)
    
    # Step 1: Preprocess model and segment to compute FPFH features
    model_down, model_fpfh = preprocess_point_cloud(
        model_centered, config.model_voxel_size, name=f"model '{model_name}'"
    )
    
    segment_down, segment_fpfh = preprocess_point_cloud(
        segment.pcd, config.segment_voxel_size, name=f"segment {segment.segment_id}"
    )
    
    # Orient segment normals consistently (towards a point "above" the workspace)
    # The exact location doesn't matter much - just needs to be outside the object
    segment_down.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 2]))
    
    # Step 2: RANSAC global registration
    try:
        ransac_result = ransac_registration(
            model_down,
            segment_down,
            model_fpfh,
            segment_fpfh
        )
    except Exception as e:
        return None, f"RANSAC failed: {e}", None
    
    if ransac_result.fitness < 0.05:
        return None, f"RANSAC low fitness ({ransac_result.fitness:.3f})", None
    
    # Step 3: ICP refinement
    if config.use_multi_start_icp:
        icp_result = multi_start_icp(
            model_down,
            segment_down,
            ransac_result.transformation,
            config.icp_distance_threshold,
            config.multi_start_n_rotations,
            config.multi_start_axes,
            visualize_all=False  # Don't visualize during batch matching
        )
    else:
        icp_result = icp_refinement(
            model_down,
            segment_down,
            ransac_result.transformation,
            config.icp_distance_threshold
        )
    
    # Step 4: Compute the final transformation
    # The ICP result transforms the CENTERED model to the scene.
    # We need to adjust for the centering: T_final = T_icp @ T_center
    # where T_center translates by -centroid (to center the model)
    T_center = np.eye(4)
    T_center[:3, 3] = -model_centroid
    final_transform = icp_result.transformation @ T_center
    
    # Step 5: Compute alignment score
    # Note: We score using the centered model with icp_result.transformation
    # (not final_transform) because that's what was actually aligned
    scores = compute_alignment_score(
        model_down,
        segment_down,
        icp_result.transformation,
        inlier_threshold=config.icp_distance_threshold,
        verbose=False  # Don't print during batch matching
    )
    
    # Step 6: Validate using alignment scores
    overall_score = scores['overall_score']
    bidirectional_fitness = scores['bidirectional_fitness']
    chamfer_distance = scores['chamfer_distance']
    m2s_coverage = scores['m2s_coverage']
    
    # Rejection checks (ordered by computational cost / reliability)
    if m2s_coverage < config.min_m2s_coverage:
        return None, f"low m2s_coverage ({m2s_coverage:.1%} < {config.min_m2s_coverage:.0%})", scores
    
    if chamfer_distance > config.max_chamfer_distance:
        return None, f"high chamfer ({chamfer_distance*1000:.1f}mm > {config.max_chamfer_distance*1000:.0f}mm)", scores
    
    if bidirectional_fitness < config.min_bidirectional_fitness:
        return None, f"low bidirectional fitness ({bidirectional_fitness:.3f} < {config.min_bidirectional_fitness})", scores
    
    if overall_score < config.min_overall_score:
        return None, f"low overall score ({overall_score:.3f} < {config.min_overall_score})", scores
    
    # All checks passed
    return RegistrationResult(
        model_name=model_name,
        transformation=final_transform,
        overall_score=overall_score,
        bidirectional_fitness=bidirectional_fitness,
        chamfer_distance=chamfer_distance,
        m2s_coverage=m2s_coverage,
        s2m_precision=scores['s2m_precision'],
        m2s_rmse=scores['m2s_rmse'],
        s2m_rmse=scores['s2m_rmse'],
        combined_rmse=scores['combined_rmse'],
        position=final_transform[:3, 3],
        rotation_matrix=final_transform[:3, :3],
    ), "ok", scores


def match_all_models_to_segments(
    models: dict[str, o3d.geometry.PointCloud],
    segments: list[SegmentPointCloud],
    config: ICPConfig,
) -> dict[int, RegistrationResult]:
    """Match all models to all segments using FPFH+RANSAC+ICP."""
    results: dict[int, RegistrationResult] = {}
    
    print(f"\n[Registration] Matching {len(models)} models to {len(segments)} segments")
    print(f"[Registration] Using FPFH + RANSAC + {'Multi-Start ' if config.use_multi_start_icp else ''}ICP")
    print(f"[Registration] Rejection thresholds:")
    print(f"    min_overall_score={config.min_overall_score} | "
          f"min_bidir_fitness={config.min_bidirectional_fitness}")
    print(f"    min_m2s_coverage={config.min_m2s_coverage:.0%} | "
          f"max_chamfer={config.max_chamfer_distance*1000:.0f}mm")
    print("-" * 60)
    
    for segment in segments:
        seg_extent = np.sort(segment.pcd.get_axis_aligned_bounding_box().get_extent())[::-1]
        print(f"\n[Segment {segment.segment_id}] {segment.name}")
        print(f"  {len(segment.pcd.points)} pts, size: {seg_extent[0]*1000:.0f}x{seg_extent[1]*1000:.0f}x{seg_extent[2]*1000:.0f}mm")
        
        best_result: RegistrationResult | None = None
        
        for model_name, model_pcd in models.items():
            model_extent = np.sort(model_pcd.get_axis_aligned_bounding_box().get_extent())[::-1]
            print(f"  Testing '{model_name}' ({model_extent[0]*1000:.0f}x{model_extent[1]*1000:.0f}x{model_extent[2]*1000:.0f}mm)...")
            
            result, reason, scores = match_model_to_segment_fpfh(model_name, model_pcd, segment, config)
            
            if result is not None:
                print(f"    ✓ score={result.overall_score:.3f} | "
                      f"bidir={result.bidirectional_fitness:.2f} | "
                      f"cov={result.m2s_coverage:.0%} | "
                      f"prec={result.s2m_precision:.0%} | "
                      f"chamfer={result.chamfer_distance*1000:.1f}mm | "
                      f"rmse={result.combined_rmse*1000:.2f}mm")
                # Select based on overall_score (highest wins)
                if best_result is None or result.overall_score > best_result.overall_score:
                    best_result = result
            else:
                if scores is not None:
                    print(f"    ✗ {reason}")
                    print(f"      score={scores['overall_score']:.3f} | "
                          f"bidir={scores['bidirectional_fitness']:.2f} | "
                          f"cov={scores['m2s_coverage']:.0%} | "
                          f"prec={scores['s2m_precision']:.0%} | "
                          f"chamfer={scores['chamfer_distance']*1000:.1f}mm | "
                          f"rmse={scores['combined_rmse']*1000:.2f}mm")
                else:
                    print(f"    ✗ {reason}")
        
        if best_result is not None:
            results[segment.segment_id] = best_result
            print(f"  → MATCHED: '{best_result.model_name}' (score={best_result.overall_score:.3f})")
        else:
            print(f"  → No match")
    
    return results


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="FPFH+RANSAC+ICP pose estimation")
    parser.add_argument("--models", type=Path, default=ICP_CFG.model_dir)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--no-multi-start", action="store_true", help="Disable multi-start ICP")
    parser.add_argument("--open3d", action="store_true", help="Use Open3D visualization instead of Meshcat")
    args = parser.parse_args()
    
    config = ICPConfig(model_dir=args.models)
    if args.no_multi_start:
        config.use_multi_start_icp = False
    
    print("=" * 60)
    print("FPFH + RANSAC + ICP POSE ESTIMATION")
    
    # Load models
    models = load_all_models(config.model_dir, config.model_voxel_size)
    if not models:
        print("[Error] No models loaded")
        return
    
    # Load grasp labels
    grasp_configs = load_grasp_labels()

    # Run detection pipeline
    pipeline = DetectionPipeline(PipelineConfig())
    result = pipeline.run()
    
    if result is None or result.depth_frame is None or not result.masks:
        print("[Done] No detections")
        return
    
    # Extract segments with colors
    segments = extract_segment_pointclouds(
        result, 
        config.segment_voxel_size, 
        config.min_segment_points,
        color_image=result.color_image,
    )
    if not segments:
        print("[Done] No valid segments")
        return
    
    # Save segments for debugging
    save_segments(segments)
    
    # FPFH + RANSAC + ICP matching
    registration_results = match_all_models_to_segments(models, segments, config)
    
    # Grasp planning (using loaded labels)
    print("\n[Grasp Planning]")
    grasp_results: dict[int, GraspResult] = {}
    for seg in segments:
        if seg.segment_id in registration_results:
            r = registration_results[seg.segment_id]
            grasp = select_best_grasp(r.model_name, r.transformation, grasp_configs)
            if grasp:
                grasp_results[seg.segment_id] = grasp
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for seg in segments:
        if seg.segment_id in registration_results:
            r = registration_results[seg.segment_id]
            print(f"\n[{seg.segment_id}] {seg.name} → {r.model_name}")
            print(f"    ┌─ Alignment Scores ─────────────────────────────────────")
            print(f"    │ Overall Score:        {r.overall_score:.4f}  (0-1, higher=better)")
            print(f"    │ Bidirectional Fit:    {r.bidirectional_fitness:.4f}  (F1 of coverage & precision)")
            print(f"    │ Chamfer Distance:     {r.chamfer_distance*1000:.2f} mm  (lower=better)")
            print(f"    │ Model→Scene Coverage: {r.m2s_coverage:.1%}  (model points matched)")
            print(f"    │ Scene→Model Precision:{r.s2m_precision:.1%}  (scene points matched)")
            print(f"    │ Model→Scene RMSE:     {r.m2s_rmse*1000:.2f} mm")
            print(f"    │ Scene→Model RMSE:     {r.s2m_rmse*1000:.2f} mm")
            print(f"    │ Combined RMSE:        {r.combined_rmse*1000:.2f} mm")
            print(f"    └────────────────────────────────────────────────────────")
            print(f"    Object pose: [{r.position[0]:.3f}, {r.position[1]:.3f}, {r.position[2]:.3f}]")
            
            if seg.segment_id in grasp_results:
                g = grasp_results[seg.segment_id]
                print(f"    Grasp: '{g.grasp_name}' at [{g.position[0]:.3f}, {g.position[1]:.3f}, {g.position[2]:.3f}]")
                print(f"    Approach: [{g.approach[0]:.3f}, {g.approach[1]:.3f}, {g.approach[2]:.3f}]")
            else:
                print(f"    Grasp: NO VALID GRASP")
        else:
            print(f"\n[{seg.segment_id}] {seg.name} → NO MATCH")
    
    # Visualize
    if not args.no_viz:
        if args.open3d:
            visualize_pipeline_results_open3d(segments, models, registration_results, grasp_results)
        else:
            vis = visualize_pipeline_results_meshcat(segments, models, registration_results, grasp_results)
            input("\n[Meshcat] Press Enter to exit...")
    
    print(f"\n[Done] Session: {result.session_dir}")


if __name__ == "__main__":
    main()
