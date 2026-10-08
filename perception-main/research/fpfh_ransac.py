"""
FPFH + RANSAC 3D Object Recognition
====================================
A simple and clean implementation using:
1. Fast Point Feature Histograms (FPFH) for local descriptors
2. RANSAC for global registration
3. ICP for refinement

Based on Open3D's global registration pipeline.
"""

import numpy as np
import open3d as o3d
import os
import time
import glob
from pathlib import Path
from typing import Tuple, List, Optional
from visualization import (
    visualize_point_cloud,
    visualize_registration,
    visualize_correspondences,
    visualize_before_after,
    visualize_all_icp_candidates
)
from utils.loaders import load_obj_as_pointcloud, _load_obj_manual

# =============================================================================
# PARAMETERS
# =============================================================================

# Model parameters
MODEL_NUM_SAMPLES = 100000          # Points to sample from mesh
MODEL_VOXEL_SIZE = 0.003            # Voxel size for model downsampling

# Scene parameters  
SCENE_REMOVE_PLANE = True           # Remove dominant plane (table/floor)
SCENE_PLANE_THRESHOLD = 0.005       # RANSAC threshold for plane fitting
SCENE_VOXEL_SIZE = 0.003            # Voxel size for scene downsampling

# FPFH parameters
FPFH_RADIUS_FACTOR = 5.0            # FPFH radius = factor * voxel_size
FPFH_MAX_NN = 100                   # Max neighbors for FPFH

# Normal estimation parameters
NORMAL_RADIUS_FACTOR = 2.0          # Normal radius = factor * voxel_size
NORMAL_MAX_NN = 30                  # Max neighbors for normals

# RANSAC parameters
RANSAC_DISTANCE_THRESHOLD = 0.01    # Inlier distance threshold
RANSAC_N_POINTS = 3               # Points to sample for each iteration
RANSAC_MAX_ITERATIONS = 8000000     # Maximum RANSAC iterations
RANSAC_CONFIDENCE = 0.9999          # Confidence level

# ICP parameters
ICP_DISTANCE_THRESHOLD = 0.005      # ICP correspondence threshold
ICP_MAX_ITERATIONS = 100            # Maximum ICP iterations
ICP_SCALES = [1.0]                  # Scale factors for multi-scale ICP (coarse-to-fine)
                                    # e.g., [4.0, 2.0, 1.0, 0.5] runs coarse-to-fine
                                    # [1.0] = single-scale (default, same as before)

# Multi-start ICP parameters
MULTI_START_ICP = True              # Enable multi-start ICP for better rotation handling
MULTI_START_N_ROTATIONS = 10        # Number of rotation samples around each axis
MULTI_START_AXES = ['x', 'y', 'z']            # Axes to rotate around: 'x', 'y', 'z', or combinations

# Visualization flags
VIS_MODEL = False                    # Show model after preprocessing
VIS_SCENE = False                    # Show scene after preprocessing
VIS_FEATURES = True                 # Show FPFH feature matching
VIS_RANSAC = True                   # Show RANSAC result
VIS_ICP = True                      # Show ICP refinement
VIS_FINAL = True                    # Show final result
VIS_ALL_ICP = True                  # Show ALL ICP candidate transformations with scores

huber_loss = False
icp_use_point_to_plane = True
# =============================================================================
# PREPROCESSING
# =============================================================================

def load_model(mesh_path: str, num_samples: int = MODEL_NUM_SAMPLES) -> o3d.geometry.PointCloud:
    """
    Load model from mesh and sample points.
    
    Supports OBJ files with non-standard MTL files (uses manual parser as fallback).
    Also loads texture colors if available.
    """
    print(f"\n{'='*60}")
    print("LOADING MODEL")
    print(f"{'='*60}")
    
    mesh_path = Path(mesh_path)
    mesh = None
    vertex_colors = None
    
    # For OBJ files, try the robust loader first
    if mesh_path.suffix.lower() == '.obj':
        try:
            mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
        except Exception:
            pass
        
        if mesh is None or not mesh.has_vertices():
            mesh, vertex_colors = _load_obj_manual(mesh_path)
            if vertex_colors is not None and len(vertex_colors) == len(mesh.vertices):
                mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    else:
        # For other formats (STL, PLY, etc.), use standard loader
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    
    if mesh is None or not mesh.has_triangles():
        raise ValueError(f"Could not load mesh from {mesh_path}")
    
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(number_of_points=num_samples)
    
    print(f"   Mesh: {mesh_path}")
    print(f"   Sampled {len(pcd.points)} points")
    print(f"   Has colors: {pcd.has_colors()}")
    
    return pcd


def load_scene(pcd_path: str) -> o3d.geometry.PointCloud:
    """Load scene point cloud."""
    print(f"\n{'='*60}")
    print("LOADING SCENE")
    print(f"{'='*60}")
    
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"   Scene: {pcd_path}")
    print(f"   Loaded {len(pcd.points)} points")
    
    return pcd


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, 
                           voxel_size: float,
                           name: str = "point cloud") -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """
    Preprocess point cloud: downsample, estimate normals, compute FPFH.
    
    Returns:
        Tuple of (downsampled_pcd, fpfh_features)
    """
    print(f"\n   Preprocessing {name}...")
    
    # Step 1: Voxel downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"      Downsampled: {len(pcd.points)} -> {len(pcd_down.points)} points")
    
    # Step 2: Estimate normals
    radius_normal = NORMAL_RADIUS_FACTOR * voxel_size
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal,
            max_nn=NORMAL_MAX_NN
        )
    )
    print(f"      Normals estimated (radius={radius_normal:.4f})")
    
    # Step 3: Compute FPFH features
    radius_fpfh = FPFH_RADIUS_FACTOR * voxel_size
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_fpfh,
            max_nn=FPFH_MAX_NN
        )
    )
    print(f"      FPFH computed (radius={radius_fpfh:.4f}, dim={fpfh.dimension()})")
    
    return pcd_down, fpfh


def remove_plane(pcd: o3d.geometry.PointCloud, 
                 threshold: float = SCENE_PLANE_THRESHOLD) -> o3d.geometry.PointCloud:
    """Remove dominant plane from scene."""
    print(f"\n   Removing dominant plane...")
    
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=threshold,
        ransac_n=3,
        num_iterations=1000
    )
    
    pcd_without_plane = pcd.select_by_index(inliers, invert=True)
    print(f"      Plane removed: {len(pcd.points)} -> {len(pcd_without_plane.points)} points")
    
    return pcd_without_plane


def center_point_cloud(pcd: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Center point cloud at origin and return the centroid."""
    centroid = pcd.get_center()
    pcd_centered = o3d.geometry.PointCloud(pcd)
    pcd_centered.translate(-centroid)
    return pcd_centered, centroid


# =============================================================================
# REGISTRATION
# =============================================================================

def ransac_registration(model_down: o3d.geometry.PointCloud,
                        scene_down: o3d.geometry.PointCloud,
                        model_fpfh: o3d.pipelines.registration.Feature,
                        scene_fpfh: o3d.pipelines.registration.Feature) -> o3d.pipelines.registration.RegistrationResult:
    """
    Perform RANSAC-based global registration using FPFH features.
    """
    print(f"\n{'='*60}")
    print("RANSAC GLOBAL REGISTRATION")
    print(f"{'='*60}")
    
    distance_threshold = RANSAC_DISTANCE_THRESHOLD
    
    print(f"   Distance threshold: {distance_threshold}")
    print(f"   Max iterations: {RANSAC_MAX_ITERATIONS}")
    
    start_time = time.time()
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        model_down,
        scene_down,
        model_fpfh,
        scene_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=RANSAC_N_POINTS,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            RANSAC_MAX_ITERATIONS, 
            RANSAC_CONFIDENCE
        )
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n   RANSAC Results:")
    print(f"      Time: {elapsed:.2f} seconds")
    print(f"      Fitness: {result.fitness:.4f}")
    print(f"      Inlier RMSE: {result.inlier_rmse:.6f}")
    print(f"      Correspondences: {len(result.correspondence_set)}")
    
    return result


def icp_refinement(model_down: o3d.geometry.PointCloud,
                   scene_down: o3d.geometry.PointCloud,
                   initial_transform: np.ndarray,
                   threshold: float = ICP_DISTANCE_THRESHOLD,
                   scales: List[float] = None) -> o3d.pipelines.registration.RegistrationResult:
    """
    Refine registration using (multi-scale) ICP.
    
    Args:
        model_down: Downsampled model point cloud
        scene_down: Downsampled scene point cloud
        initial_transform: Initial transformation (e.g., from RANSAC)
        threshold: Base ICP distance threshold
        scales: List of scale factors for multi-scale ICP (default: ICP_SCALES)
                e.g., [4.0, 2.0, 1.0, 0.5] runs coarse-to-fine
                [1.0] = single-scale (equivalent to standard ICP)
    
    Returns:
        Final ICP registration result
    """
    if scales is None:
        scales = ICP_SCALES
    
    print(f"\n{'='*60}")
    print("ICP REFINEMENT" + (" (multi-scale)" if len(scales) > 1 else ""))
    print(f"{'='*60}")
    
    print(f"   Base threshold: {threshold}")
    print(f"   Scales: {scales}")
    print(f"   Max iterations per scale: {ICP_MAX_ITERATIONS}")
    
    start_time = time.time()

    # Set up estimation method
    if huber_loss:
        loss = o3d.pipelines.registration.TukeyLoss(k=0.1) 
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    else:
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    if icp_use_point_to_plane:
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    
    # Multi-scale ICP: iterate through scales (coarse to fine)
    current_transform = initial_transform.copy()
    result = None
    
    for i, scale in enumerate(scales):
        scaled_threshold = threshold * scale
        
        result = o3d.pipelines.registration.registration_icp(
            model_down,
            scene_down,
            scaled_threshold,
            current_transform,
            estimation_method,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITERATIONS)
        )
        
        current_transform = result.transformation
        
        if len(scales) > 1:
            print(f"      Scale {scale:.1f}x (threshold={scaled_threshold:.4f}): "
                  f"fitness={result.fitness:.4f}, rmse={result.inlier_rmse:.6f}")
    
    elapsed = time.time() - start_time
    
    print(f"\n   ICP Results:")
    print(f"      Time: {elapsed:.2f} seconds")
    print(f"      Fitness: {result.fitness:.4f}")
    print(f"      Inlier RMSE: {result.inlier_rmse:.6f}")
    
    return result


def multi_start_icp(model_down: o3d.geometry.PointCloud,
                    scene_down: o3d.geometry.PointCloud,
                    initial_transform: np.ndarray,
                    threshold: float = ICP_DISTANCE_THRESHOLD,
                    n_rotations: int = MULTI_START_N_ROTATIONS,
                    axes: List[str] = None,
                    scales: List[float] = None,
                    visualize_all: bool = False) -> o3d.pipelines.registration.RegistrationResult:
    """
    Try ICP from multiple rotated starting positions to escape local minima.
    
    When RANSAC finds approximately the right position but wrong rotation,
    this tries multiple rotations around the detected center and picks the best.
    Optionally uses multi-scale ICP (coarse-to-fine) for each candidate.
    
    Args:
        model_down: Downsampled model point cloud
        scene_down: Downsampled scene point cloud
        initial_transform: Initial transformation from RANSAC
        threshold: Base ICP distance threshold
        n_rotations: Number of rotation samples per axis
        axes: List of axes to rotate around ('x', 'y', 'z')
        scales: List of scale factors for multi-scale ICP (default: ICP_SCALES)
        visualize_all: If True, visualize each candidate transformation with its score
    
    Returns:
        Best ICP registration result
    """
    if axes is None:
        axes = MULTI_START_AXES
    if scales is None:
        scales = ICP_SCALES
    
    print(f"\n{'='*60}")
    print("MULTI-START ICP REFINEMENT" + (" (multi-scale)" if len(scales) > 1 else ""))
    print(f"{'='*60}")
    print(f"   Rotations per axis: {n_rotations}")
    print(f"   Axes: {axes}")
    print(f"   Base threshold: {threshold}")
    print(f"   Scales: {scales}")
    
    start_time = time.time()
    
    # Extract rotation and translation from initial transform
    initial_rotation = initial_transform[:3, :3]
    initial_translation = initial_transform[:3, 3]
    
    # Generate rotation angles
    angles = np.linspace(0, 2 * np.pi, n_rotations, endpoint=False)
    
    # Build list of candidate transforms
    candidate_transforms = [initial_transform.copy()]  # Include original
    
    for axis in axes:
        for angle in angles:
            if angle == 0:  # Skip 0 rotation (already included as initial)
                continue
            
            # Create rotation matrix around specified axis
            if axis.lower() == 'x':
                R_delta = o3d.geometry.get_rotation_matrix_from_xyz([angle, 0, 0])
            elif axis.lower() == 'y':
                R_delta = o3d.geometry.get_rotation_matrix_from_xyz([0, angle, 0])
            elif axis.lower() == 'z':
                R_delta = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, angle])
            else:
                continue
            
            # Apply rotation to original rotation: R_new = R_delta @ R_initial
            R_new = R_delta @ initial_rotation
            
            # Build new transform keeping same translation
            T_new = np.eye(4)
            T_new[:3, :3] = R_new
            T_new[:3, 3] = initial_translation
            
            candidate_transforms.append(T_new)
    
    print(f"   Testing {len(candidate_transforms)} candidate poses...")
    
    # Set up ICP estimation method
    if icp_use_point_to_plane:
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    
    # Run ICP from each candidate and track best
    best_result = None
    best_score = -np.inf
    best_idx = 0
    
    # Collect all results if visualizing
    all_results = [] if visualize_all else None
    
    for i, T_init in enumerate(candidate_transforms):
        # Multi-scale ICP for this candidate
        current_transform = T_init.copy()
        result = None
        
        for scale in scales:
            scaled_threshold = threshold * scale
            result = o3d.pipelines.registration.registration_icp(
                model_down,
                scene_down,
                scaled_threshold,
                current_transform,
                estimation_method,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=ICP_MAX_ITERATIONS)
            )
            current_transform = result.transformation
        
        # Score: prioritize high fitness, penalize high RMSE
        # Higher fitness is better, lower RMSE is better
        icp_score = result.fitness - result.inlier_rmse * 10

        # print(f"      Candidate {i}: icp_score={icp_score:.4f}, fitness={result.fitness:.4f}, rmse={result.inlier_rmse:.6f}")
        
        if visualize_all:
            # Compute comprehensive alignment score for visualization
            alignment_scores = compute_alignment_score(
                model_down, scene_down, result.transformation,
                inlier_threshold=threshold, verbose=False
            )
            all_results.append({
                'transformation': result.transformation.copy(),
                'icp_score': icp_score,
                'icp_fitness': result.fitness,
                'icp_rmse': result.inlier_rmse,
                'alignment_scores': alignment_scores
            })
        
        if icp_score > best_score:
            best_score = icp_score
            best_result = result
            best_idx = i
    
    # Visualize all candidates if requested
    if visualize_all and all_results:
        visualize_all_icp_candidates(model_down, scene_down, all_results, best_idx)
    
    elapsed = time.time() - start_time
    
    print(f"\n   Multi-Start ICP Results:")
    print(f"      Time: {elapsed:.2f} seconds")
    print(f"      Best candidate: {best_idx} / {len(candidate_transforms)}")
    print(f"      Best Fitness: {best_result.fitness:.4f}")
    print(f"      Best Inlier RMSE: {best_result.inlier_rmse:.6f}")
    
    return best_result


def compute_alignment_score(model: o3d.geometry.PointCloud,
                            scene: o3d.geometry.PointCloud,
                            transformation: np.ndarray,
                            inlier_threshold: float = 0.005,
                            verbose: bool = True) -> dict:
    """
    Compute comprehensive bidirectional alignment score.
    
    This evaluates how well the model aligns with the scene by checking:
    1. Model-to-Scene: How many model points have nearby scene points (coverage)
    2. Scene-to-Model: How many scene points (near model) have nearby model points (precision)
    3. Bidirectional consistency: Average of both directions
    
    A perfect alignment has:
    - High model coverage (model points match scene)
    - High precision (scene points in region match model)
    - Low RMSE in both directions
    - High bidirectional score
    
    Args:
        model: Model point cloud (will be transformed)
        scene: Scene point cloud
        transformation: 4x4 transformation matrix
        inlier_threshold: Distance threshold for considering a point as "matched"
        verbose: Print detailed scores
    
    Returns:
        Dictionary with all alignment metrics
    """
    # Transform model to scene frame
    model_transformed = o3d.geometry.PointCloud(model)
    model_transformed.transform(transformation)
    
    model_pts = np.asarray(model_transformed.points)
    scene_pts = np.asarray(scene.points)
    
    n_model = len(model_pts)
    n_scene = len(scene_pts)
    
    # Build KD-trees for fast nearest neighbor queries
    scene_tree = o3d.geometry.KDTreeFlann(scene)
    model_tree = o3d.geometry.KDTreeFlann(model_transformed)
    
    # =========================================================================
    # 1. MODEL-TO-SCENE (Forward): For each model point, find nearest scene point
    # =========================================================================
    m2s_distances = []
    m2s_inliers = 0
    
    for pt in model_pts:
        [_, idx, dist_sq] = scene_tree.search_knn_vector_3d(pt, 1)
        dist = np.sqrt(dist_sq[0])
        m2s_distances.append(dist)
        if dist < inlier_threshold:
            m2s_inliers += 1
    
    m2s_distances = np.array(m2s_distances)
    m2s_coverage = m2s_inliers / n_model  # What fraction of model is covered
    m2s_rmse = np.sqrt(np.mean(m2s_distances[m2s_distances < inlier_threshold]**2)) if m2s_inliers > 0 else np.inf
    m2s_mean_dist = np.mean(m2s_distances)
    
    # =========================================================================
    # 2. SCENE-TO-MODEL (Backward): For scene points near model, find nearest model point
    # =========================================================================
    # First, find scene points that are within the model's bounding region
    model_center = model_transformed.get_center()
    model_extent = np.linalg.norm(model_transformed.get_max_bound() - model_transformed.get_min_bound())
    search_radius = model_extent * 0.6  # Search within 60% of model extent from center
    
    # Find scene points near the model
    scene_near_model_indices = []
    for i, pt in enumerate(scene_pts):
        if np.linalg.norm(pt - model_center) < search_radius:
            scene_near_model_indices.append(i)
    
    s2m_distances = []
    s2m_inliers = 0
    
    for idx in scene_near_model_indices:
        pt = scene_pts[idx]
        [_, nn_idx, dist_sq] = model_tree.search_knn_vector_3d(pt, 1)
        dist = np.sqrt(dist_sq[0])
        s2m_distances.append(dist)
        if dist < inlier_threshold:
            s2m_inliers += 1
    
    n_scene_near = len(scene_near_model_indices)
    s2m_distances = np.array(s2m_distances) if len(s2m_distances) > 0 else np.array([np.inf])
    s2m_precision = s2m_inliers / n_scene_near if n_scene_near > 0 else 0  # What fraction of local scene matches
    s2m_rmse = np.sqrt(np.mean(s2m_distances[s2m_distances < inlier_threshold]**2)) if s2m_inliers > 0 else np.inf
    s2m_mean_dist = np.mean(s2m_distances) if n_scene_near > 0 else np.inf
    
    # =========================================================================
    # 3. COMBINED SCORES
    # =========================================================================
    # Bidirectional fitness: Harmonic mean of coverage and precision (like F1 score)
    if m2s_coverage + s2m_precision > 0:
        bidirectional_fitness = 2 * (m2s_coverage * s2m_precision) / (m2s_coverage + s2m_precision)
    else:
        bidirectional_fitness = 0.0
    
    # Combined RMSE: average of both (when valid)
    if m2s_rmse != np.inf and s2m_rmse != np.inf:
        combined_rmse = (m2s_rmse + s2m_rmse) / 2
    elif m2s_rmse != np.inf:
        combined_rmse = m2s_rmse
    elif s2m_rmse != np.inf:
        combined_rmse = s2m_rmse
    else:
        combined_rmse = np.inf
    
    # Overall score: High is good (range roughly 0-1)
    # Penalize poor RMSE, reward high bidirectional fitness
    if combined_rmse != np.inf and combined_rmse > 0:
        overall_score = bidirectional_fitness * (1.0 - min(combined_rmse / inlier_threshold, 1.0))
    else:
        overall_score = 0.0
    
    # Chamfer distance (symmetric): average of mean distances both ways
    chamfer_distance = (m2s_mean_dist + s2m_mean_dist) / 2
    
    # =========================================================================
    # Build result dictionary
    # =========================================================================
    scores = {
        # Model-to-Scene metrics
        'm2s_coverage': m2s_coverage,           # Fraction of model covered by scene
        'm2s_inliers': m2s_inliers,             # Number of model inliers
        'm2s_rmse': m2s_rmse,                   # RMSE of model inliers
        'm2s_mean_dist': m2s_mean_dist,         # Mean distance model->scene
        
        # Scene-to-Model metrics
        's2m_precision': s2m_precision,         # Fraction of local scene matching model
        's2m_inliers': s2m_inliers,             # Number of scene inliers
        's2m_rmse': s2m_rmse,                   # RMSE of scene inliers
        's2m_mean_dist': s2m_mean_dist,         # Mean distance scene->model
        'n_scene_near_model': n_scene_near,     # Scene points in model region
        
        # Combined metrics
        'bidirectional_fitness': bidirectional_fitness,  # F1-like score of coverage & precision
        'combined_rmse': combined_rmse,                   # Average RMSE
        'chamfer_distance': chamfer_distance,             # Symmetric Chamfer distance
        'overall_score': overall_score,                   # Final composite score (0-1, higher is better)
        
        # Counts
        'n_model_points': n_model,
        'n_scene_points': n_scene,
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("ALIGNMENT SCORE")
        print(f"{'='*60}")
        print(f"\n   Model → Scene (Coverage):")
        print(f"      Coverage: {m2s_coverage:.4f} ({m2s_inliers}/{n_model} points)")
        print(f"      RMSE: {m2s_rmse:.6f}")
        print(f"      Mean distance: {m2s_mean_dist:.6f}")
        
        print(f"\n   Scene → Model (Precision):")
        print(f"      Precision: {s2m_precision:.4f} ({s2m_inliers}/{n_scene_near} local points)")
        print(f"      RMSE: {s2m_rmse:.6f}")
        print(f"      Mean distance: {s2m_mean_dist:.6f}")
        
        print(f"\n   Combined Metrics:")
        print(f"      Bidirectional Fitness (F1): {bidirectional_fitness:.4f}")
        print(f"      Combined RMSE: {combined_rmse:.6f}")
        print(f"      Chamfer Distance: {chamfer_distance:.6f}")
        print(f"      ★ Overall Score: {overall_score:.4f}")
        
        # Quality assessment
        print(f"\n   Quality Assessment:")
        if overall_score > 0.8:
            print(f"      ✓ EXCELLENT alignment")
        elif overall_score > 0.6:
            print(f"      ✓ GOOD alignment")
        elif overall_score > 0.3:
            print(f"      ~ MODERATE alignment - may need verification")
        elif overall_score > 0.15:
            print(f"      ✗ POOR alignment - likely incorrect pose")
        else:
            print(f"      ✗ VERY POOR alignment - wrong detection")
    
    return scores


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description="FPFH + RANSAC 3D Object Recognition")
    parser.add_argument("--model", type=str, default=None, help="Path to model file (STL/OBJ)")
    parser.add_argument("--scene", type=str, default=None, help="Path to scene PCD file")
    parser.add_argument("--segment", type=int, default=0, help="Segment index to load from /data/segments/")
    parser.add_argument("--use-segments", action="store_true", help="Load from saved segments")
    args = parser.parse_args()
    
    print("=" * 60)
    print("FPFH + RANSAC 3D Object Recognition")
    print("=" * 60)
    
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(script_dir)  # /home/jeong/perception
    
    # Default paths
    if args.use_segments or (args.model is None and args.scene is None):
        # Use can model by default (loads from /data/can/3DModel.obj)
        model_path = os.path.join(data_dir, "data/can/3DModel.obj")
        
        # Find segment files (check both .ply and .pcd)
        segment_dir = os.path.join(data_dir, "data/segments")
        segment_files = sorted(
            glob.glob(os.path.join(segment_dir, f"segment_{args.segment}_*.ply")) +
            glob.glob(os.path.join(segment_dir, f"segment_{args.segment}_*.pcd"))
        )
        
        if segment_files:
            scene_path = segment_files[0]
            print(f"[Using saved segment] {scene_path}")
        else:
            print(f"[Warning] No segment files found in {segment_dir}")
            print(f"[Warning] Run 6.py first to generate segments")
            # Fallback to T-shape (legacy)
            model_path = os.path.join(script_dir, "data/model/T/Tshape.STL")
            scene_path = os.path.join(script_dir, "data/scene/T/scene2.pcd")
    else:
        model_path = args.model if args.model else os.path.join(data_dir, "data/can/3DModel.obj")
        scene_path = args.scene if args.scene else os.path.join(script_dir, "data/scene/T/scene2.pcd")
    
    # Legacy paths (commented out)
    # model_path = os.path.join(script_dir, "data/model/T/Tshape.STL")
    # scene_path = os.path.join(script_dir, "data/scene/T/scene2.pcd")
    # model_path = os.path.join(script_dir, "data/model/bi/6914.STL")
    # scene_path = os.path.join(script_dir, "data/scene/bi/scene5.pcd")
    
    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    
    model_pcd = load_model(model_path, MODEL_NUM_SAMPLES)
    scene_pcd = load_scene(scene_path)
    
    # =========================================================================
    # STEP 2: PREPROCESS MODEL
    # =========================================================================
    
    print(f"\n{'='*60}")
    print("PREPROCESSING MODEL")
    print(f"{'='*60}")
    
    # Center model at origin
    model_centered, model_centroid = center_point_cloud(model_pcd)
    print(f"   Centered model (centroid was: {model_centroid})")
    
    # Preprocess
    model_down, model_fpfh = preprocess_point_cloud(
        model_centered, 
        MODEL_VOXEL_SIZE, 
        "model"
    )
    
    if VIS_MODEL:
        visualize_point_cloud(
            model_down,
            title="Model - Preprocessed (with normals)",
            show_normals=True,
            color=[0.0, 0.8, 0.7]
        )
    
    # =========================================================================
    # STEP 3: PREPROCESS SCENE
    # =========================================================================
    
    print(f"\n{'='*60}")
    print("PREPROCESSING SCENE")
    print(f"{'='*60}")
    
    # Skip plane removal for segmented objects (already isolated)
    is_segment = "segment" in scene_path.lower()
    
    # Remove plane if requested (and not a segment)
    if SCENE_REMOVE_PLANE and not is_segment:
        scene_no_plane = remove_plane(scene_pcd, SCENE_PLANE_THRESHOLD)
    else:
        scene_no_plane = scene_pcd
        if is_segment:
            print(f"   Skipping plane removal (already segmented)")
    
    # Preprocess
    scene_down, scene_fpfh = preprocess_point_cloud(
        scene_no_plane,
        SCENE_VOXEL_SIZE,
        "scene"
    )
    
    # Orient normals towards camera (assumed at origin)
    scene_down.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))
    
    if VIS_SCENE:
        visualize_point_cloud(
            scene_down,
            title="Scene - Preprocessed (with normals)",
            show_normals=True,
            color=[0.0, 0.7, 0.8]
        )
    
    # =========================================================================
    # STEP 4: VISUALIZE FPFH CORRESPONDENCES
    # =========================================================================
    
    if VIS_FEATURES:
        visualize_correspondences(
            model_down,
            scene_down,
            model_fpfh,
            scene_fpfh,
            title="FPFH Feature Correspondences"
        )
    
    # =========================================================================
    # STEP 5: RANSAC REGISTRATION
    # =========================================================================
    
    ransac_result = ransac_registration(
        model_down,
        scene_down,
        model_fpfh,
        scene_fpfh
    )
    
    if VIS_RANSAC:
        visualize_registration(
            model_down,
            scene_down,
            ransac_result.transformation,
            title="RANSAC Registration Result"
        )
    
    # =========================================================================
    # STEP 6: ICP REFINEMENT
    # =========================================================================
    
    if MULTI_START_ICP:
        icp_result = multi_start_icp(
            model_down,
            scene_down,
            ransac_result.transformation,
            ICP_DISTANCE_THRESHOLD,
            MULTI_START_N_ROTATIONS,
            MULTI_START_AXES,
            visualize_all=VIS_ALL_ICP
        )
    else:
        icp_result = icp_refinement(
            model_down,
            scene_down,
            ransac_result.transformation,
            ICP_DISTANCE_THRESHOLD
        )
    
    if VIS_ICP:
        visualize_before_after(
            model_down,
            scene_down,
            ransac_result.transformation,
            icp_result.transformation,
            title="ICP Refinement (Red=Before, Green=After)"
        )
    
    # =========================================================================
    # STEP 7: FINAL RESULT
    # =========================================================================
    
    print(f"\n{'='*60}")
    print("FINAL RESULT")
    print(f"{'='*60}")
    
    final_transform = icp_result.transformation
    
    print("\n   Final Transformation Matrix:")
    print(final_transform)
    
    print(f"\n   Final Fitness: {icp_result.fitness:.4f}")
    print(f"   Final RMSE: {icp_result.inlier_rmse:.6f}")
    
    # =========================================================================
    # STEP 7.5: COMPUTE ALIGNMENT SCORE
    # =========================================================================
    
    alignment_scores = compute_alignment_score(
        model_down,
        scene_down,
        final_transform,
        inlier_threshold=ICP_DISTANCE_THRESHOLD,
        verbose=True
    )
    
    if VIS_FINAL:
        # Show with original scene (with colors if available)
        print(f"\n   Showing: Final Result")
        print(f"   GREEN = Detected model")
        print(f"   (Close window to exit)")
        
        scene_vis = o3d.geometry.PointCloud(scene_pcd)
        if not scene_vis.has_colors():
            scene_vis.paint_uniform_color([0.6, 0.6, 0.6])
        
        model_vis = o3d.geometry.PointCloud(model_down)
        model_vis.transform(final_transform)
        model_vis.paint_uniform_color([0.0, 0.9, 0.0])
        
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
        coord_frame.transform(final_transform)
        
        o3d.visualization.draw_geometries(
            [scene_vis, model_vis, coord_frame],
            window_name="Final Result - FPFH + RANSAC + ICP",
            width=1280,
            height=960
        )
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
