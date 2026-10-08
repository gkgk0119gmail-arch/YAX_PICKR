"""
Test script for grasp detection from saved session data.

Implements proper 6DOF grasp pose sampling:
- Position (x, y, z) from surface points
- Orientation (R) from surface normal + roll angle sampling
- Spatial diversity via Non-Maximum Suppression (NMS)

The gripper has a rectangular body, so different roll angles around the 
approach axis can have very different collision profiles.

Usage:
    python test.py
    python test.py --n-rolls 16      # More roll angles
    python test.py --nms-radius 0.02 # Spread grasps by 2cm
    python test.py --show-all-rolls  # Visualize all rolls for top point
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from cluttered_bin_segmentation import (
    load_sam2_auto_mask_generator,
    generate_masks,
    filter_masks_by_area,
    filter_edge_masks,
    merge_overlapping_masks,
    filter_contained_masks,
    filter_masks_by_depth,
    sort_masks_by_depth,
    visualize_masks,
)
from cluttered_bin_grasp import (
    SegmentPointCloud,
    GraspConfig,
    T_BASE_CAMERA,
    transform_points,
    voxel_downsample,
    remove_outliers,
    estimate_normals,
    filter_workspace,
    compute_object_center,
    score_verticality,
    score_centroid_proximity,
    score_flatness,
    score_plane_fit,
    score_rim_coverage,
    compute_plane_fit_rmse,
    build_scene_collision_data,
)
from visualization import (
    GraspPose6D,
    visualize_grasp_6d,
    visualize_all_rolls_for_point,
    visualize_grasp_poses_o3d,
    visualize_analysis_meshcat,
)
from scipy.spatial import cKDTree


# =============================================================================
# PARAMETERS
# =============================================================================

DEPTH_SCALE = 0.0001
FULL_WIDTH = 1280
FULL_HEIGHT = 720
CROP = (358, 33, 791, 465)

MIN_SEGMENT_AREA = 2000
MAX_AREA_RATIO = 0.4
EDGE_MARGIN = 5
MERGE_OVERLAP_THRESH = 0.7
CONTAIN_THRESH = 0.7
DEPTH_BACKGROUND_M = 2.0
MASK_ALPHA = 0.6

SAM2_CKPT = Path("/home/jeong/perception/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"




def normal_to_rotation(normal: np.ndarray, roll: float = 0.0) -> np.ndarray:
    """Compute rotation matrix from surface normal and roll angle."""
    z = normal / np.linalg.norm(normal)
    
    # Choose initial X perpendicular to Z
    if abs(z[0]) < abs(z[1]):
        x_init = np.array([1.0, 0.0, 0.0])
    else:
        x_init = np.array([0.0, 1.0, 0.0])
    
    # Gram-Schmidt
    x = x_init - np.dot(x_init, z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    
    # Apply roll
    c, s = np.cos(roll), np.sin(roll)
    x_rolled = c * x + s * y
    y_rolled = -s * x + c * y
    
    return np.column_stack([x_rolled, y_rolled, z])


def sample_roll_angles(n_samples: int = 8) -> np.ndarray:
    """Generate roll angles evenly distributed in [0, 2π)."""
    return np.linspace(0, 2 * np.pi, n_samples, endpoint=False)


# =============================================================================
# NON-MAXIMUM SUPPRESSION FOR SPATIAL DIVERSITY
# =============================================================================

def nms_by_position(
    indices: np.ndarray,
    positions: np.ndarray,
    scores: np.ndarray,
    radius: float,
    max_keep: int = 50,
    verbose: bool = False,
) -> list[int]:
    """Non-Maximum Suppression based on spatial proximity.
    
    Keeps highest-scoring points while ensuring minimum distance between them.
    
    Args:
        indices: Original indices into positions array
        positions: Nx3 array of point positions
        scores: N array of scores (higher = better)
        radius: Minimum distance between kept points
        max_keep: Maximum number of points to keep
        verbose: Print debug info about suppressions
    
    Returns:
        List of indices (into original array) of kept points
    """
    # Sort by score descending
    order = np.argsort(scores)[::-1]
    
    kept = []
    kept_positions = []
    kept_scores = []
    n_suppressed = 0
    
    for rank, idx in enumerate(order):
        if len(kept) >= max_keep:
            break
        
        pos = positions[idx]
        score = scores[idx]
        
        # Check if too close to any already kept point
        too_close = False
        suppressor_idx = -1
        for i, kp in enumerate(kept_positions):
            dist = np.linalg.norm(pos - kp)
            if dist < radius:
                too_close = True
                suppressor_idx = i
                break
        
        if not too_close:
            kept.append(indices[idx])
            kept_positions.append(pos)
            kept_scores.append(score)
            if verbose:
                print(f"    NMS kept #{len(kept)}: score={score:.3f}, "
                      f"pos=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}]")
        else:
            n_suppressed += 1
            if verbose and n_suppressed <= 5:
                print(f"    NMS suppressed: score={score:.3f} (rank #{rank+1}) "
                      f"- too close to kept #{suppressor_idx+1} (score={kept_scores[suppressor_idx]:.3f})")
    
    if verbose:
        print(f"    NMS: {n_suppressed} points suppressed")
        if kept_scores:
            print(f"    Kept scores: max={max(kept_scores):.3f}, min={min(kept_scores):.3f}, "
                  f"range={max(kept_scores)-min(kept_scores):.3f}")
    
    return kept


# =============================================================================
# COLLISION CHECKING WITH CLEARANCE
# =============================================================================

def check_gripper_collision_with_clearance(
    pose: GraspPose6D,
    collision_tree: cKDTree,
    cfg: GraspConfig,
    check_radius: float = 0.012,
) -> tuple[bool, int, float]:
    """Check collision and compute clearance (min distance to obstacle).
    
    Returns:
        (collision_free, n_collisions, clearance)
        clearance: Minimum distance to nearest collision point (higher = safer)
    """
    if collision_tree is None:
        return True, 0, float('inf')
    
    R = pose.rotation
    t = pose.position
    
    cup_radius = cfg.suction_cup_radius
    cup_height = cfg.gripper_cup_height
    body_w, body_d, body_h = cfg.gripper_body_size
    
    # Generate sample points
    check_points_local = []
    
    # Approach path
    for h in np.linspace(cup_height * 0.7, cup_height, 4):
        check_points_local.append(np.array([0, 0, h]))
    
    # Gripper body
    body_step = 0.015
    for x in np.arange(-body_w/2 + body_step/2, body_w/2, body_step):
        for y in np.arange(-body_d/2 + body_step/2, body_d/2, body_step):
            for z in np.arange(cup_height, cup_height + body_h, body_step):
                check_points_local.append(np.array([x, y, z]))
    
    # Suction cup rim
    for h in [cup_height * 0.5, cup_height * 0.7, cup_height * 0.9]:
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            x = cup_radius * 0.9 * np.cos(angle)
            y = cup_radius * 0.9 * np.sin(angle)
            check_points_local.append(np.array([x, y, h]))
    
    check_points_local = np.array(check_points_local)
    world_points = (R @ check_points_local.T).T + t
    
    # Check collisions AND compute clearance
    n_collisions = 0
    min_clearance = float('inf')
    
    for wp in world_points:
        # Check collision
        neighbors = collision_tree.query_ball_point(wp, check_radius)
        if len(neighbors) > 0:
            n_collisions += len(neighbors)
        
        # Compute distance to nearest point (clearance)
        dist, _ = collision_tree.query(wp, k=1)
        min_clearance = min(min_clearance, dist)
    
    collision_free = n_collisions == 0
    return collision_free, n_collisions, min_clearance


def build_collision_tree_excluding_segment(
    background_points: np.ndarray,
    segment_points_list: list[np.ndarray],
    occupancy_points: np.ndarray,
    exclude_segment_id: int,
) -> cKDTree | None:
    """Build collision KD-tree excluding target segment."""
    collision_points = []
    
    if len(background_points) > 0:
        collision_points.append(background_points)
    if len(occupancy_points) > 0:
        collision_points.append(occupancy_points)
    
    for i, seg_pts in enumerate(segment_points_list):
        if i != exclude_segment_id and len(seg_pts) > 0:
            collision_points.append(seg_pts)
    
    if collision_points:
        return cKDTree(np.vstack(collision_points))
    return None


# =============================================================================
# GRASP SAMPLING WITH ROLL OPTIMIZATION
# =============================================================================

def evaluate_all_rolls(
    position: np.ndarray,
    normal: np.ndarray,
    point_score: float,
    collision_tree: cKDTree,
    cfg: GraspConfig,
    roll_angles: np.ndarray,
    segment_id: int = -1,
    plane_rmse: float = 0.0,
    rim_coverage: float = 0.0,
) -> list[GraspPose6D]:
    """Evaluate ALL roll angles for a grasp point.
    
    Returns list of poses with collision/clearance info for each roll.
    """
    poses = []
    
    for roll in roll_angles:
        R = normal_to_rotation(normal, roll)
        
        pose = GraspPose6D(
            position=position.copy(),
            rotation=R,
            score=point_score,
            point_score=point_score,
            roll_angle=roll,
            segment_id=segment_id,
            plane_rmse=plane_rmse,
            rim_coverage=rim_coverage,
            seal_valid=True,
        )
        
        collision_free, n_collisions, clearance = check_gripper_collision_with_clearance(
            pose, collision_tree, cfg, cfg.collision_check_radius
        )
        
        pose.collision_free = collision_free
        pose.n_collisions = n_collisions
        pose.clearance = clearance
        
        # Adjust score based on clearance (prefer rolls with more space)
        # Add small bonus for clearance to break ties
        pose.score = point_score + 0.01 * min(clearance, 0.1)  # Cap at 10cm
        
        poses.append(pose)
    
    return poses


def find_best_roll_for_grasp(
    position: np.ndarray,
    normal: np.ndarray,
    point_score: float,
    collision_tree: cKDTree,
    cfg: GraspConfig,
    roll_angles: np.ndarray,
    segment_id: int = -1,
    plane_rmse: float = 0.0,
    rim_coverage: float = 0.0,
    random_roll: bool = False,
    verbose: bool = False,
) -> GraspPose6D | None:
    """Find best roll angle for a grasp point.
    
    Args:
        random_roll: If True, pick randomly among collision-free rolls (for diversity)
        verbose: Print clearances for all rolls
    """
    poses = evaluate_all_rolls(
        position, normal, point_score, collision_tree, cfg, roll_angles,
        segment_id, plane_rmse, rim_coverage
    )
    
    if verbose:
        print(f"      Roll clearances: ", end="")
        for p in poses:
            roll_deg = np.degrees(p.roll_angle)
            status = "OK" if p.collision_free else "X"
            print(f"{roll_deg:.0f}°:{p.clearance*1000:.0f}mm({status}) ", end="")
        print()
    
    # Separate collision-free and colliding
    collision_free_poses = [p for p in poses if p.collision_free]
    
    if not collision_free_poses:
        # All collide - return the one with least collisions
        poses.sort(key=lambda p: p.n_collisions)
        return poses[0] if poses else None
    
    if random_roll and len(collision_free_poses) > 1:
        # Pick randomly among collision-free rolls
        # Weight by clearance so higher clearance is more likely
        clearances = np.array([p.clearance for p in collision_free_poses])
        # Softmax-like weighting
        weights = clearances / clearances.sum()
        idx = np.random.choice(len(collision_free_poses), p=weights)
        return collision_free_poses[idx]
    else:
        # Pick highest clearance among collision-free
        collision_free_poses.sort(key=lambda p: -p.clearance)
        return collision_free_poses[0]




def extract_pointclouds(masks: list[dict], pts_cam: np.ndarray, crop: tuple, full_shape: tuple):
    """Extract segment and background point clouds."""
    h, w = full_shape
    valid = (pts_cam[:, 2] > 0.1) & (pts_cam[:, 2] < 3.0) & np.isfinite(pts_cam[:, 2])
    pts = transform_points(pts_cam, T_BASE_CAMERA)
    
    crop_region = np.zeros((h, w), dtype=bool)
    crop_region[crop[1]:crop[3], crop[0]:crop[2]] = True
    crop_mask = crop_region.flatten()
    
    any_seg = np.zeros(len(pts), dtype=bool)
    segments = []
    
    for i, m in enumerate(masks):
        seg_mask = m['segmentation']
        full_mask = np.zeros((h, w), dtype=bool)
        full_mask[crop[1]:crop[3], crop[0]:crop[2]] = seg_mask
        
        obj_mask = valid & full_mask.flatten()
        any_seg |= obj_mask
        obj_pts = pts[obj_mask]
        
        if len(obj_pts) < 100:
            continue
        
        np.random.seed(i)
        colors = np.tile(np.random.rand(3), (len(obj_pts), 1))
        segments.append(SegmentPointCloud(i, obj_pts, colors, seg_mask))
    
    bg_mask = valid & crop_mask & ~any_seg
    bg_pts = pts[bg_mask]
    bg_colors = np.tile([0.3, 0.3, 0.3], (len(bg_pts), 1))
    
    return segments, bg_pts, bg_colors


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test 6DOF grasp detection")
    parser.add_argument("session", nargs="?", default="grasp_20260129_074636")
    parser.add_argument("--meshcat", action="store_true")
    parser.add_argument("--top-k", type=int, default=10, help="Grasps to visualize")
    parser.add_argument("--n-rolls", type=int, default=8, help="Roll angles per point")
    parser.add_argument("--nms-radius", type=float, default=0.015, help="NMS radius for spatial diversity (m)")
    parser.add_argument("--no-nms", action="store_true", help="Disable NMS (use top-k by score only)")
    parser.add_argument("--show-all-rolls", action="store_true", help="Show all rolls for top point")
    parser.add_argument("--random-roll", action="store_true", help="Pick random roll among collision-free (weighted by clearance)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output (NMS + roll selection)")
    args = parser.parse_args()
    
    session_dir = Path(__file__).parent / "data" / args.session
    
    # Load data
    color_bgr = cv2.imread(str(session_dir / "color.png"))
    depth = np.load(session_dir / "depth.npy")
    pts_cam = np.load(session_dir / "pointcloud.npy")
    print(f"[Load] Color: {color_bgr.shape}, Depth: {depth.shape}, PointCloud: {pts_cam.shape}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = GraspConfig()
    
    # Load/generate masks
    masks_path = session_dir / "masks_raw.npz"
    if masks_path.exists():
        print("\n[Segment] Loading saved masks...")
        data = np.load(masks_path, allow_pickle=True)
        masks = [{'segmentation': m, 'area': int(m.sum())} for m in data['masks']]
        print(f"  Loaded: {len(masks)} masks")
    else:
        print("\n[Segment] Running SAM2...")
        from types import SimpleNamespace
        sam_cfg = SimpleNamespace(SAM2_CKPT=SAM2_CKPT, SAM2_CFG=SAM2_CFG,
            POINTS_PER_SIDE=24, POINTS_PER_BATCH=64, PRED_IOU_THRESH=0.86,
            STABILITY_SCORE_THRESH=0.92, BOX_NMS_THRESH=0.4, CROP_N_LAYERS=0,
            CROP_NMS_THRESH=0.7, MIN_MASK_REGION_AREA=500, USE_M2M=True)
        mask_gen = load_sam2_auto_mask_generator(sam_cfg, device)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        masks = generate_masks(mask_gen, color_rgb, device)
        print(f"  Raw: {len(masks)} masks")
        np.savez_compressed(masks_path, masks=[m['segmentation'] for m in masks])
    
    # Filter masks
    cv2.imwrite(str(session_dir / "test_seg_00_raw.png"),
                visualize_masks(color_bgr, masks, MASK_ALPHA))
    
    img_shape = color_bgr.shape[:2]
    masks = filter_masks_by_area(masks, img_shape, MIN_SEGMENT_AREA, MAX_AREA_RATIO)
    cv2.imwrite(str(session_dir / "test_seg_01_area.png"),
                visualize_masks(color_bgr, masks, MASK_ALPHA))
    
    masks = filter_edge_masks(masks, img_shape, EDGE_MARGIN)
    cv2.imwrite(str(session_dir / "test_seg_02_edge.png"),
                visualize_masks(color_bgr, masks, MASK_ALPHA))
    
    masks = merge_overlapping_masks(masks, MERGE_OVERLAP_THRESH)
    cv2.imwrite(str(session_dir / "test_seg_03_merge.png"),
                visualize_masks(color_bgr, masks, MASK_ALPHA))
    
    masks = filter_contained_masks(masks, CONTAIN_THRESH)
    cv2.imwrite(str(session_dir / "test_seg_04_contain.png"),
                visualize_masks(color_bgr, masks, MASK_ALPHA))
    
    masks = filter_masks_by_depth(masks, depth, DEPTH_SCALE, DEPTH_BACKGROUND_M, False)
    cv2.imwrite(str(session_dir / "test_seg_05_depth.png"),
                visualize_masks(color_bgr, masks, MASK_ALPHA))
    
    masks = sort_masks_by_depth(masks)
    print(f"  Filtered: {len(masks)} masks")
    
    if not masks:
        print("No valid masks!")
        exit(1)
    
    # Extract point clouds
    print("\n[PointCloud] Extracting...")
    segments, bg_pts, bg_colors = extract_pointclouds(
        masks, pts_cam, CROP, (FULL_HEIGHT, FULL_WIDTH)
    )
    print(f"  {len(segments)} segments, {len(bg_pts):,} background points")
    
    # Build collision data
    print("\n[Collision] Building scene...")
    _, occupancy_pts = build_scene_collision_data(segments, bg_pts, T_BASE_CAMERA, cfg)
    
    segment_points_list = [seg.points for seg in segments]
    
    # ==========================================================================
    # SINGLE SEGMENT ANALYSIS
    # ==========================================================================
    segment_id = 3
    points = segments[segment_id].points
    segment_colors = segments[segment_id].colors
    
    print(f"\n[Analysis] Segment {segment_id}: {len(points)} points")
    
    # Preprocessing
    pts_workspace = filter_workspace(points, cfg)
    print(f"  {len(pts_workspace)} pts after workspace filter")
    
    pts_voxel = voxel_downsample(pts_workspace, cfg.voxel_size)
    print(f"  {len(pts_voxel)} pts after voxel downsample")
    
    pts = remove_outliers(pts_voxel, min(cfg.sor_mean_k, len(pts_voxel) - 1), cfg.sor_std_ratio)
    print(f"  {len(pts)} pts after outlier removal")
    
    normals, curvatures = estimate_normals(pts, min(cfg.normal_k_neighbors, len(pts)), cfg.camera_position)
    
    valid_mask = normals[:, 2] > cfg.min_normal_z
    n_upward = valid_mask.sum()
    angle_thresh = np.degrees(np.arccos(cfg.min_normal_z))
    print(f"  {n_upward}/{len(pts)} pts face upward (within {angle_thresh:.1f}° of vertical)")
    
    if not valid_mask.any():
        print("  SKIP: no upward-facing points")
        exit(1)
    
    valid_pts = pts[valid_mask]
    valid_normals = normals[valid_mask]
    valid_curvatures = curvatures[valid_mask]
    
    tree = cKDTree(pts)
    center = compute_object_center(valid_pts)
    print(f"  Object center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    
    # ==========================================================================
    # COMPUTE POINT-BASED SCORES
    # ==========================================================================
    n = len(valid_pts)
    gravity = cfg.gravity_vector / np.linalg.norm(cfg.gravity_vector)
    
    verticality_scores = np.array([score_verticality(valid_normals[i], gravity) for i in range(n)])
    
    dists_xy = np.linalg.norm(valid_pts[:, :2] - center[:2], axis=1)
    max_dist = max(dists_xy.max(), 1e-6)
    centroid_scores = np.array([score_centroid_proximity(valid_pts[i], center, max_dist) for i in range(n)])
    
    flatness_scores = np.array([score_flatness(valid_curvatures[i], cfg.curvature_threshold) for i in range(n)])
    
    plane_rmses = np.array([compute_plane_fit_rmse(valid_pts[i], pts, tree, cfg.suction_cup_radius) for i in range(n)])
    plane_fit_scores = np.array([score_plane_fit(plane_rmses[i], cfg.plane_fit_rmse_threshold) for i in range(n)])
    
    rim_coverage_scores = np.array([
        score_rim_coverage(valid_pts[i], pts, tree, cfg.suction_cup_radius, cfg.n_rim_sectors, cfg.rim_inner_ratio)
        for i in range(n)
    ])
    
    seal_valid = rim_coverage_scores >= cfg.min_rim_coverage_ratio
    
    print(f"\n  Point-based scores ({n} upward points):")
    print(f"    verticality:   min={verticality_scores.min():.2f}, max={verticality_scores.max():.2f}")
    print(f"    centroid:      min={centroid_scores.min():.2f}, max={centroid_scores.max():.2f}")
    print(f"    flatness:      min={flatness_scores.min():.2f}, max={flatness_scores.max():.2f}")
    print(f"    plane_fit:     min={plane_fit_scores.min():.2f}, max={plane_fit_scores.max():.2f}")
    print(f"    rim_coverage:  min={rim_coverage_scores.min():.2f}, max={rim_coverage_scores.max():.2f}")
    print(f"    seal_valid:    {seal_valid.sum()}/{n} ({seal_valid.sum()/n*100:.1f}%)")
    
    point_scores = (
        cfg.weight_verticality * verticality_scores +
        cfg.weight_centroid * centroid_scores +
        cfg.weight_flatness * flatness_scores +
        cfg.weight_plane_fit * plane_fit_scores +
        cfg.weight_rim_coverage * rim_coverage_scores
    )
    point_scores[~seal_valid] = -np.inf
    
    # ==========================================================================
    # SPATIAL DIVERSITY VIA NMS (or skip if --no-nms)
    # ==========================================================================
    
    # Get seal-valid indices sorted by score
    seal_valid_indices = np.where(seal_valid)[0]
    seal_valid_scores = point_scores[seal_valid]
    seal_valid_positions = valid_pts[seal_valid]
    
    # Sort by score descending for reference
    score_order = np.argsort(seal_valid_scores)[::-1]
    
    if args.no_nms:
        print(f"\n[No NMS] Using top {args.top_k * 5} by score (no spatial diversity)...")
        # Just take top-k by score
        diverse_indices = [seal_valid_indices[i] for i in score_order[:args.top_k * 5]]
        print(f"  {len(seal_valid_indices)} seal-valid -> {len(diverse_indices)} top by score")
        print(f"  Score range: {seal_valid_scores[score_order[0]]:.3f} to {seal_valid_scores[score_order[min(len(score_order)-1, args.top_k*5-1)]]:.3f}")
    else:
        print(f"\n[NMS] Applying Non-Maximum Suppression (radius={args.nms_radius*1000:.0f}mm)...")
        print(f"  NOTE: NMS trades score for spatial diversity. Use --no-nms to compare.")
        
        # NMS to get spatially diverse points
        diverse_indices = nms_by_position(
            seal_valid_indices,
            seal_valid_positions,
            seal_valid_scores,
            radius=args.nms_radius,
            max_keep=args.top_k * 5,
            verbose=args.verbose,
        )
        
        print(f"  {len(seal_valid_indices)} seal-valid -> {len(diverse_indices)} after NMS")
        
        # Show score comparison
        if diverse_indices:
            diverse_scores = [point_scores[i] for i in diverse_indices]
            top_scores = [seal_valid_scores[i] for i in score_order[:len(diverse_indices)]]
            print(f"  NMS scores:    max={max(diverse_scores):.3f}, min={min(diverse_scores):.3f}")
            print(f"  Top-k scores:  max={max(top_scores):.3f}, min={min(top_scores):.3f}")
            score_loss = max(top_scores) - min(diverse_scores)
            print(f"  Score trade-off: {score_loss:.3f} (lower = better diversity without losing quality)")
    
    # ==========================================================================
    # 6DOF GRASP SAMPLING WITH ROLL OPTIMIZATION
    # ==========================================================================
    print(f"\n[6DOF Grasp] Sampling {args.n_rolls} roll angles per point...")
    
    roll_angles = sample_roll_angles(args.n_rolls)
    print(f"  Roll angles: {[f'{np.degrees(r):.0f}°' for r in roll_angles]}")
    
    collision_tree = build_collision_tree_excluding_segment(
        bg_pts, segment_points_list, occupancy_pts, segment_id
    )
    
    # Show all rolls for top point if requested
    if args.show_all_rolls and len(diverse_indices) > 0:
        top_idx = diverse_indices[0]
        print(f"\n[All Rolls Mode] Evaluating point #{top_idx}")
        
        all_roll_poses = evaluate_all_rolls(
            position=valid_pts[top_idx],
            normal=valid_normals[top_idx],
            point_score=float(point_scores[top_idx]),
            collision_tree=collision_tree,
            cfg=cfg,
            roll_angles=roll_angles,
            segment_id=segment_id,
            plane_rmse=float(plane_rmses[top_idx]),
            rim_coverage=float(rim_coverage_scores[top_idx]),
        )
        
        # Build scene for viz
        other_segments_pts = np.vstack([seg.points for seg in segments[1:]]) if len(segments) > 1 else np.zeros((0, 3))
        viz_scene_pts = np.vstack([p for p in [bg_pts, other_segments_pts] if len(p) > 0]) if len(bg_pts) > 0 or len(other_segments_pts) > 0 else np.zeros((0, 3))
        
        visualize_all_rolls_for_point(
            valid_pts[top_idx], valid_normals[top_idx], all_roll_poses,
            segments[segment_id].points, segments[segment_id].colors,
            viz_scene_pts, occupancy_pts, cfg
        )
        
        print("\n[Done] All rolls visualized.")
        exit(0)
    
    # Find best roll for each diverse point
    print(f"  Processing {len(diverse_indices)} spatially diverse points...")
    if args.random_roll:
        print(f"  NOTE: --random-roll enabled, picking randomly among collision-free rolls")
    
    grasp_poses = []
    for rank, idx in enumerate(diverse_indices):
        pose = find_best_roll_for_grasp(
            position=valid_pts[idx],
            normal=valid_normals[idx],
            point_score=float(point_scores[idx]),
            collision_tree=collision_tree,
            cfg=cfg,
            roll_angles=roll_angles,
            segment_id=segment_id,
            plane_rmse=float(plane_rmses[idx]),
            rim_coverage=float(rim_coverage_scores[idx]),
            random_roll=args.random_roll,
            verbose=args.verbose and rank < 3,  # Only verbose for first 3
        )
        
        if pose is not None:
            grasp_poses.append(pose)
            roll_deg = np.degrees(pose.roll_angle)
            status = "OK" if pose.collision_free else f"COLL({pose.n_collisions})"
            print(f"    #{rank+1}: score={pose.point_score:.3f}, roll={roll_deg:.0f}°, "
                  f"clearance={pose.clearance*1000:.1f}mm, {status}")
        
        if len([p for p in grasp_poses if p.collision_free]) >= args.top_k:
            break
    
    # Sort: collision-free first, then by score
    grasp_poses.sort(key=lambda p: (not p.collision_free, -p.score))
    
    n_collision_free = sum(1 for p in grasp_poses if p.collision_free)
    print(f"\n  Found {len(grasp_poses)} grasps ({n_collision_free} collision-free)")
    
    # ==========================================================================
    # MESHCAT VISUALIZATION
    # ==========================================================================
    print("\n[Viz] Starting meshcat...")
    other_seg_pts = [seg.points for seg in segments[1:]]
    scene_pts_list = [bg_pts, occupancy_pts] + other_seg_pts
    scene_pts = np.vstack([p for p in scene_pts_list if len(p) > 0])
    
    visualize_analysis_meshcat(
        pts=pts,
        normals=normals,
        curvatures=curvatures,
        valid_pts=valid_pts,
        valid_mask=valid_mask,
        scores={
            'verticality': verticality_scores,
            'centroid': centroid_scores,
            'flatness': flatness_scores,
            'combined': point_scores,
        },
        scene_pts=scene_pts,
    )
    
    # ==========================================================================
    # OPEN3D VISUALIZATION
    # ==========================================================================
    other_segments_pts = np.vstack([seg.points for seg in segments[1:]]) if len(segments) > 1 else np.zeros((0, 3))
    viz_scene_pts = np.vstack([p for p in [bg_pts, other_segments_pts] if len(p) > 0]) if len(bg_pts) > 0 or len(other_segments_pts) > 0 else np.zeros((0, 3))
    
    print("  TIP: Use --show-all-rolls to see all roll angles for top point")
    visualize_grasp_poses_o3d(
        grasp_poses=grasp_poses,
        segment=segments[segment_id],
        scene_pts=viz_scene_pts,
        occupancy_pts=occupancy_pts,
        cfg=cfg,
        top_k=args.top_k,
    )
