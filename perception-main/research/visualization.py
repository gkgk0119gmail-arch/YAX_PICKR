"""
Visualization module for cluttered bin grasp detection.

Supports both Open3D (default) and Meshcat visualization backends.

Usage:
    from visualization import visualize_results, visualize_results_meshcat
    from visualization import GraspPose6D, visualize_grasp_6d
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import open3d as o3d

from view_clutter import create_gripper_mesh


# Rank-based colors for grasps (best=red, worst=green)
RANK_COLORS = [
    [1.0, 0.0, 0.0],  # Rank 0: Red (best)
    [1.0, 0.5, 0.0],  # Rank 1: Orange
    [1.0, 1.0, 0.0],  # Rank 2: Yellow
    [0.5, 1.0, 0.0],  # Rank 3: Lime
    [0.0, 1.0, 0.0],  # Rank 4+: Green
]

RANK_COLOR_NAMES = ["RED", "ORANGE", "YELLOW", "LIME", "GREEN"]


class SegmentPointCloud(NamedTuple):
    """Point cloud for a segmented object."""
    segment_id: int
    points: np.ndarray
    colors: np.ndarray
    mask: np.ndarray = None


@dataclass
class GraspCandidate:
    """A candidate suction grasp pose."""
    position: np.ndarray
    normal: np.ndarray
    score: float
    segment_id: int = -1
    plane_rmse: float = 0.0
    rim_coverage: float = 0.0
    seal_valid: bool = False
    collision_free: bool = False


# =============================================================================
# 6DOF GRASP POSE
# =============================================================================

@dataclass
class GraspPose6D:
    """Full 6DOF grasp pose in SE(3).
    
    Gripper frame:
    - Z-axis: approach direction (into object)
    - X-axis: gripper body width direction
    - Y-axis: gripper body depth direction
    """
    position: np.ndarray      # (3,) contact point
    rotation: np.ndarray      # (3, 3) rotation matrix
    
    score: float = 0.0
    point_score: float = 0.0
    roll_angle: float = 0.0
    clearance: float = 0.0    # Min distance to collision (higher = safer)
    
    segment_id: int = -1
    plane_rmse: float = 0.0
    rim_coverage: float = 0.0
    seal_valid: bool = False
    collision_free: bool = False
    n_collisions: int = 0
    
    @property
    def normal(self) -> np.ndarray:
        return self.rotation[:, 2]
    
    @property
    def transform(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.position
        return T
    
    def __repr__(self):
        roll_deg = np.degrees(self.roll_angle)
        status = "OK" if self.collision_free else f"COLL({self.n_collisions})"
        return (f"GraspPose6D(pos=[{self.position[0]:.3f},{self.position[1]:.3f},{self.position[2]:.3f}], "
                f"roll={roll_deg:.0f}°, score={self.score:.3f}, {status})")


class SegmentResult(NamedTuple):
    """Result of grasp detection on a segment."""
    segment_id: int
    candidates: list[GraspCandidate]
    points: np.ndarray | None = None


def get_rank_color(rank: int) -> list[float]:
    """Get color for a given rank (0=best)."""
    return RANK_COLORS[min(rank, len(RANK_COLORS) - 1)]


def get_rank_color_name(rank: int) -> str:
    """Get color name for a given rank."""
    return RANK_COLOR_NAMES[min(rank, len(RANK_COLOR_NAMES) - 1)]


def compute_grasp_ranking(segments, results) -> tuple[list, dict]:
    """
    Compute global grasp ranking across all segments.
    
    Returns:
        best_grasps: List of (segment_id, grasp) tuples sorted by score
        global_rank: Dict mapping segment_id to global rank
    """
    result_map = {r.segment_id: r for r in results}
    
    best_grasps = []
    for seg in segments:
        res = result_map.get(seg.segment_id)
        if res and res.candidates:
            best_grasps.append((seg.segment_id, res.candidates[0]))
    
    # Sort by score (highest first)
    best_grasps.sort(key=lambda x: x[1].score, reverse=True)
    global_rank = {seg_id: rank for rank, (seg_id, _) in enumerate(best_grasps)}
    
    return best_grasps, global_rank


def print_grasp_ranking(best_grasps: list):
    """Print grasp ranking table to console.
    
    The normal vector shown IS the surface normal at the grasp point, expressed
    in the WORLD (robot base) frame. This is also the gripper approach direction:
    the gripper's Z-axis aligns with this normal, approaching from above.
    
    For a flat horizontal surface facing up: normal ≈ [0, 0, 1]
    For a tilted surface: normal will have X/Y components showing the tilt direction
    """
    print("\n" + "=" * 60)
    print("GRASP RANKING (best to worst)")
    print("=" * 60)
    print("  normal = surface normal at grasp point (world frame)")
    print("         = gripper approach direction (Z-axis aligns with this)")
    print("-" * 60)
    for rank, (seg_id, grasp) in enumerate(best_grasps):
        color_name = get_rank_color_name(rank)
        n = grasp.normal
        # Calculate angle from vertical (Z-up)
        angle_from_vertical = np.degrees(np.arccos(np.clip(n[2], -1, 1)))
        print(f"  #{rank+1} [Seg {seg_id}] score={grasp.score:.3f} "
              f"rim={grasp.rim_coverage*100:.0f}% "
              f"-> {color_name}")
        print(f"      pos=[{grasp.position[0]:.3f}, {grasp.position[1]:.3f}, {grasp.position[2]:.3f}]")
        print(f"      normal=[{n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}] ({angle_from_vertical:.1f}° from vertical)")
    print("=" * 60 + "\n")


# =============================================================================
# OPEN3D VISUALIZATION
# =============================================================================
def create_scored_gripper_mesh_o3d(
    position: np.ndarray,
    normal: np.ndarray,
    suction_cup_radius: float,
    gripper_cup_height: float,
    gripper_body_size: tuple,
    rank: int,
) -> o3d.geometry.TriangleMesh:
    """Create Open3D gripper mesh colored by rank."""
    gripper = create_gripper_mesh(
        position, normal,
        diameter=suction_cup_radius * 2,
        cup_height=gripper_cup_height,
        body_size=gripper_body_size,
    )
    gripper.paint_uniform_color(get_rank_color(rank))
    return gripper


def visualize_results_o3d(
    segments: list,
    results: list,
    grasp_cfg,
    background_points: np.ndarray = None,
    background_colors: np.ndarray = None,
    occupancy_points: np.ndarray = None,
):
    """
    Visualize using Open3D.
    
    Args:
        segments: List of SegmentPointCloud objects
        results: List of SegmentResult objects
        grasp_cfg: GraspConfig with gripper dimensions
        background_points: Optional background point cloud
        background_colors: Optional background colors
        occupancy_points: Optional occupancy volume points (shown in black)
    """
    geoms = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)]
    result_map = {r.segment_id: r for r in results}
    
    # Add occupancy volume as black points
    if occupancy_points is not None and len(occupancy_points) > 0:
        occ_pcd = o3d.geometry.PointCloud()
        occ_pcd.points = o3d.utility.Vector3dVector(occupancy_points)
        occ_pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(occupancy_points), 3)))
        geoms.append(occ_pcd)
        print(f"[Viz-O3D] Occupancy: {len(occupancy_points):,} points (black)")
    
    # Add background pointcloud in grey
    if background_points is not None and len(background_points) > 0:
        bg_pcd = o3d.geometry.PointCloud()
        bg_pcd.points = o3d.utility.Vector3dVector(background_points)
        if background_colors is not None:
            bg_pcd.colors = o3d.utility.Vector3dVector(background_colors)
        else:
            bg_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        geoms.append(bg_pcd)
        print(f"[Viz-O3D] Background: {len(background_points):,} points (grey)")
    
    # Compute ranking
    best_grasps, global_rank = compute_grasp_ranking(segments, results)
    print_grasp_ranking(best_grasps)
    
    # Add segment pointclouds and gripper meshes
    for seg in segments:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(seg.points)
        pcd.colors = o3d.utility.Vector3dVector(seg.colors)
        geoms.append(pcd)
        
        res = result_map.get(seg.segment_id)
        if res and res.candidates:
            best = res.candidates[0]
            rank = global_rank.get(seg.segment_id, 0)
            gripper = create_scored_gripper_mesh_o3d(
                best.position, best.normal,
                grasp_cfg.suction_cup_radius,
                grasp_cfg.gripper_cup_height,
                grasp_cfg.gripper_body_size,
                rank,
            )
            geoms.append(gripper)
    
    print(f"[Viz-O3D] {len(segments)} segments, {len(best_grasps)} grasps")
    o3d.visualization.draw_geometries(geoms, window_name="Grasp Detection", width=1200, height=800)


# =============================================================================
# 6DOF GRASP VISUALIZATION (Open3D)
# =============================================================================

def create_gripper_mesh_6d(pose: GraspPose6D, cfg) -> o3d.geometry.TriangleMesh:
    """Create gripper mesh from 6DOF pose.
    
    Args:
        pose: GraspPose6D with position and rotation
        cfg: GraspConfig with gripper dimensions
    """
    diameter = cfg.suction_cup_radius * 2
    cup_height = cfg.gripper_cup_height
    body_w, body_d, body_h = cfg.gripper_body_size
    
    cup = o3d.geometry.TriangleMesh.create_cylinder(radius=diameter/2, height=cup_height)
    cup.paint_uniform_color([1.0, 0.3, 0.3])
    cup.translate([0, 0, cup_height/2])
    
    body = o3d.geometry.TriangleMesh.create_box(width=body_w, height=body_d, depth=body_h)
    body.paint_uniform_color([0.4, 0.4, 0.4])
    body.translate([-body_w/2, -body_d/2, cup_height])
    
    gripper = cup + body
    gripper.rotate(pose.rotation, center=[0, 0, 0])
    gripper.translate(pose.position)
    
    return gripper


def visualize_grasp_6d(
    pose: GraspPose6D,
    segment_pts: np.ndarray,
    segment_colors: np.ndarray,
    scene_pts: np.ndarray,
    occupancy_pts: np.ndarray,
    cfg,
    grasp_rank: int,
    total_grasps: int,
    extra_info: str = "",
):
    """Visualize a 6DOF grasp pose in Open3D.
    
    Args:
        pose: GraspPose6D to visualize
        segment_pts: Points of the target segment
        segment_colors: Colors of the target segment
        scene_pts: Background/other segment points
        occupancy_pts: Occupancy volume points (shown in black)
        cfg: GraspConfig with gripper dimensions
        grasp_rank: Current grasp rank (1-indexed)
        total_grasps: Total number of grasps
        extra_info: Additional info for window title
    """
    geoms = []
    
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))
    
    # Grasp frame
    grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
    grasp_frame.rotate(pose.rotation, center=[0, 0, 0])
    grasp_frame.translate(pose.position)
    geoms.append(grasp_frame)
    
    if len(scene_pts) > 0:
        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(scene_pts)
        scene_pcd.colors = o3d.utility.Vector3dVector(
            np.tile([0.2, 0.2, 0.2], (len(scene_pts), 1))
        )
        geoms.append(scene_pcd)
    
    if len(occupancy_pts) > 0:
        occ_pcd = o3d.geometry.PointCloud()
        occ_pcd.points = o3d.utility.Vector3dVector(occupancy_pts)
        occ_pcd.colors = o3d.utility.Vector3dVector(np.zeros((len(occupancy_pts), 3)))
        geoms.append(occ_pcd)
    
    seg_pcd = o3d.geometry.PointCloud()
    seg_pcd.points = o3d.utility.Vector3dVector(segment_pts)
    seg_pcd.colors = o3d.utility.Vector3dVector(segment_colors)
    geoms.append(seg_pcd)
    
    gripper = create_gripper_mesh_6d(pose, cfg)
    if pose.collision_free:
        gripper.paint_uniform_color([0.2, 0.8, 0.2])
    else:
        gripper.paint_uniform_color([0.9, 0.2, 0.2])
    geoms.append(gripper)
    
    status = "COLLISION-FREE" if pose.collision_free else f"COLLISION ({pose.n_collisions} pts)"
    roll_deg = np.degrees(pose.roll_angle)
    window_title = f"Grasp {grasp_rank}/{total_grasps}: roll={roll_deg:.0f}° - {status}"
    if extra_info:
        window_title += f" [{extra_info}]"
    
    print(f"\n[Open3D] {window_title}")
    print(f"  Position:  [{pose.position[0]:.3f}, {pose.position[1]:.3f}, {pose.position[2]:.3f}]")
    print(f"  Normal:    [{pose.normal[0]:.2f}, {pose.normal[1]:.2f}, {pose.normal[2]:.2f}]")
    print(f"  Roll:      {roll_deg:.1f}°")
    print(f"  Clearance: {pose.clearance*1000:.1f}mm")
    print(f"  Score:     {pose.score:.3f}")
    print("  Close window to continue...")
    
    o3d.visualization.draw_geometries(geoms, window_name=window_title, width=1200, height=800)


def visualize_all_rolls_for_point(
    position: np.ndarray,
    normal: np.ndarray,
    roll_poses: list[GraspPose6D],
    segment_pts: np.ndarray,
    segment_colors: np.ndarray,
    scene_pts: np.ndarray,
    occupancy_pts: np.ndarray,
    cfg,
):
    """Visualize all roll angles for a single grasp point (in sequence).
    
    Args:
        position: Grasp point position
        normal: Surface normal at grasp point
        roll_poses: List of GraspPose6D for each roll angle
        segment_pts: Points of the target segment
        segment_colors: Colors of the target segment
        scene_pts: Background/other segment points
        occupancy_pts: Occupancy volume points
        cfg: GraspConfig with gripper dimensions
    """
    print(f"\n[All Rolls] Showing {len(roll_poses)} roll angles for single point")
    print(f"  Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
    
    for i, pose in enumerate(roll_poses):
        roll_deg = np.degrees(pose.roll_angle)
        status = "OK" if pose.collision_free else f"COLL({pose.n_collisions})"
        print(f"    Roll {roll_deg:3.0f}°: clearance={pose.clearance*1000:5.1f}mm, {status}")
    
    # Show each roll in sequence
    for i, pose in enumerate(roll_poses):
        visualize_grasp_6d(
            pose, segment_pts, segment_colors, scene_pts, occupancy_pts, cfg,
            grasp_rank=i+1, total_grasps=len(roll_poses),
            extra_info=f"clearance={pose.clearance*1000:.1f}mm"
        )


def visualize_grasp_poses_o3d(
    grasp_poses: list[GraspPose6D],
    segment: SegmentPointCloud,
    scene_pts: np.ndarray,
    occupancy_pts: np.ndarray,
    cfg,
    top_k: int = 10,
):
    """Visualize top-K grasp poses using Open3D.
    
    Args:
        grasp_poses: List of GraspPose6D sorted by score
        segment: Target segment point cloud
        scene_pts: Background/other segment points
        occupancy_pts: Occupancy volume points
        cfg: GraspConfig with gripper dimensions
        top_k: Number of grasps to visualize
    """
    print("\n" + "="*60)
    print("OPEN3D 6DOF GRASP VISUALIZATION")
    print("="*60)
    
    grasps_to_show = grasp_poses[:top_k]
    
    print(f"\n[Open3D] Showing {len(grasps_to_show)} spatially diverse grasps...")
    print("  GREEN = collision-free, RED = collision")
    
    for rank, pose in enumerate(grasps_to_show):
        visualize_grasp_6d(
            pose, segment.points, segment.colors,
            scene_pts, occupancy_pts, cfg,
            grasp_rank=rank + 1, total_grasps=len(grasps_to_show),
        )
    
    print("\n[Done] All grasps visualized.")


# =============================================================================
# MESHCAT ANALYSIS VISUALIZATION
# =============================================================================

def visualize_analysis_meshcat(
    pts: np.ndarray,
    normals: np.ndarray,
    curvatures: np.ndarray,
    valid_pts: np.ndarray,
    valid_mask: np.ndarray,
    scores: dict,
    scene_pts: np.ndarray,
) -> str:
    """Visualize grasp analysis pipeline in meshcat.
    
    Shows side-by-side views of different scoring stages.
    
    Args:
        pts: All segment points after preprocessing
        normals: Surface normals for all points
        curvatures: Curvature values for all points
        valid_pts: Points that pass upward-facing filter
        valid_mask: Boolean mask of valid points
        scores: Dict with keys: 'verticality', 'centroid', 'flatness', 'combined'
        scene_pts: Background/other segment points
    
    Returns:
        Meshcat viewer URL
    """
    import meshcat
    import meshcat.geometry as g
    
    vis = meshcat.Visualizer()
    vis.open()
    vis.delete()
    
    scene_colors = np.tile([0.15, 0.15, 0.15], (len(scene_pts), 1))
    spacing = 0.35
    
    def add_scene(name, y_offset):
        sc = scene_pts.copy()
        sc[:, 1] += y_offset
        vis[f"{name}_scene"].set_object(g.PointCloud(sc.T, scene_colors.T, size=0.002))
    
    # 1: Curvature visualization
    y1 = -2 * spacing
    add_scene("1", y1)
    pts1 = pts.copy()
    pts1[:, 1] += y1
    curv_norm = (curvatures - curvatures.min()) / (curvatures.max() - curvatures.min() + 1e-10)
    colors1 = np.zeros((len(pts), 3))
    colors1[:, 0] = curv_norm
    colors1[:, 2] = 1 - curv_norm
    vis["1_curvature"].set_object(g.PointCloud(pts1.T, colors1.T, size=0.004))
    
    # 2: Normals visualization
    y2 = -1 * spacing
    add_scene("2", y2)
    pts2 = pts.copy()
    pts2[:, 1] += y2
    colors2 = (normals + 1) / 2
    vis["2_normals"].set_object(g.PointCloud(pts2.T, colors2.T, size=0.004))
    
    # 3: Filtered (upward-facing) points
    y3 = 0
    add_scene("3", y3)
    pts3 = valid_pts.copy()
    pts3[:, 1] += y3
    colors3 = np.tile([0.4, 0.8, 0.4], (len(valid_pts), 1))
    vis["3_filtered"].set_object(g.PointCloud(pts3.T, colors3.T, size=0.004))
    
    # 4: Verticality scores
    y4 = 1 * spacing
    add_scene("4", y4)
    pts4 = valid_pts.copy()
    pts4[:, 1] += y4
    colors4 = np.zeros((len(valid_pts), 3))
    colors4[:, 1] = scores['verticality']
    colors4[:, 2] = 1 - scores['verticality']
    vis["4_verticality"].set_object(g.PointCloud(pts4.T, colors4.T, size=0.005))
    
    # 5: Centroid scores
    y5 = 2 * spacing
    add_scene("5", y5)
    pts5 = valid_pts.copy()
    pts5[:, 1] += y5
    colors5 = np.zeros((len(valid_pts), 3))
    colors5[:, 0] = scores['centroid']
    colors5[:, 2] = 1 - scores['centroid']
    vis["5_centroid"].set_object(g.PointCloud(pts5.T, colors5.T, size=0.005))
    
    # 6: Flatness scores
    y6 = 3 * spacing
    add_scene("6", y6)
    pts6 = valid_pts.copy()
    pts6[:, 1] += y6
    colors6 = np.zeros((len(valid_pts), 3))
    colors6[:, 0] = 1 - scores['flatness']
    colors6[:, 1] = scores['flatness']
    colors6[:, 2] = scores['flatness']
    vis["6_flatness"].set_object(g.PointCloud(pts6.T, colors6.T, size=0.005))
    
    # 7: Combined scores
    combined = scores['combined']
    valid_scores = combined[combined > -np.inf]
    if len(valid_scores) > 0:
        combined_norm = (combined - valid_scores.min()) / (valid_scores.max() - valid_scores.min() + 1e-10)
        combined_norm[combined == -np.inf] = 0
    else:
        combined_norm = np.zeros(len(valid_pts))
    
    y7 = 4 * spacing
    add_scene("7", y7)
    pts7 = valid_pts.copy()
    pts7[:, 1] += y7
    colors7 = np.zeros((len(valid_pts), 3))
    colors7[:, 0] = combined_norm
    colors7[:, 1] = combined_norm
    colors7[:, 2] = 1 - combined_norm
    vis["7_combined"].set_object(g.PointCloud(pts7.T, colors7.T, size=0.005))
    
    print(f"\n  Meshcat URL: {vis.url()}")
    return vis.url()


# =============================================================================
# MESHCAT VISUALIZATION
# =============================================================================
def visualize_results_meshcat(
    segments: list,
    results: list,
    grasp_cfg,
    background_points: np.ndarray = None,
    background_colors: np.ndarray = None,
    occupancy_points: np.ndarray = None,
    zmq_url: str = None,
):
    """
    Visualize using Meshcat (web-based 3D viewer).
    
    Args:
        segments: List of SegmentPointCloud objects
        results: List of SegmentResult objects
        grasp_cfg: GraspConfig with gripper dimensions
        background_points: Optional background point cloud
        background_colors: Optional background colors
        occupancy_points: Optional occupancy volume points
        zmq_url: Optional ZMQ URL for meshcat server
    """
    try:
        import meshcat
        import meshcat.geometry as g
        import meshcat.transformations as tf
    except ImportError:
        print("[Error] meshcat not installed. Install with: pip install meshcat")
        print("        Falling back to Open3D visualization...")
        visualize_results_o3d(segments, results, grasp_cfg, background_points, 
                              background_colors, occupancy_points)
        return
    
    # Create meshcat visualizer
    if zmq_url:
        vis = meshcat.Visualizer(zmq_url=zmq_url)
    else:
        vis = meshcat.Visualizer()
    
    vis.delete()  # Clear previous geometry
    
    print(f"[Viz-Meshcat] Open viewer at: {vis.url()}")
    
    result_map = {r.segment_id: r for r in results}
    
    # Add coordinate frame
    vis["frame"].set_object(g.triad(scale=0.1))
    
    # Add occupancy volume as black points
    if occupancy_points is not None and len(occupancy_points) > 0:
        # Subsample if too many points
        if len(occupancy_points) > 50000:
            idx = np.random.choice(len(occupancy_points), 50000, replace=False)
            occ_pts = occupancy_points[idx]
        else:
            occ_pts = occupancy_points
        
        occ_colors = np.zeros((len(occ_pts), 3), dtype=np.float32)
        vis["occupancy"].set_object(
            g.PointCloud(occ_pts.T.astype(np.float32), occ_colors.T, size=0.003)
        )
        print(f"[Viz-Meshcat] Occupancy: {len(occ_pts):,} points (black)")
    
    # Add background pointcloud in grey
    if background_points is not None and len(background_points) > 0:
        if len(background_points) > 50000:
            idx = np.random.choice(len(background_points), 50000, replace=False)
            bg_pts = background_points[idx]
            bg_cols = background_colors[idx] if background_colors is not None else None
        else:
            bg_pts = background_points
            bg_cols = background_colors
        
        if bg_cols is None:
            bg_cols = np.full((len(bg_pts), 3), 0.5, dtype=np.float32)
        
        vis["background"].set_object(
            g.PointCloud(bg_pts.T.astype(np.float32), bg_cols.T.astype(np.float32), size=0.003)
        )
        print(f"[Viz-Meshcat] Background: {len(bg_pts):,} points (grey)")
    
    # Compute ranking
    best_grasps, global_rank = compute_grasp_ranking(segments, results)
    print_grasp_ranking(best_grasps)
    
    # Add segment pointclouds
    for seg in segments:
        pts = seg.points
        colors = seg.colors
        
        # Subsample large segments
        if len(pts) > 20000:
            idx = np.random.choice(len(pts), 20000, replace=False)
            pts = pts[idx]
            colors = colors[idx]
        
        vis[f"segment_{seg.segment_id}"].set_object(
            g.PointCloud(pts.T.astype(np.float32), colors.T.astype(np.float32), size=0.004)
        )
    
    # Add gripper meshes
    for seg in segments:
        res = result_map.get(seg.segment_id)
        if res and res.candidates:
            best = res.candidates[0]
            rank = global_rank.get(seg.segment_id, 0)
            color = get_rank_color(rank)
            
            # Create gripper components
            _add_gripper_meshcat(
                vis, f"gripper_{seg.segment_id}",
                best.position, best.normal,
                grasp_cfg.suction_cup_radius,
                grasp_cfg.gripper_cup_height,
                grasp_cfg.gripper_body_size,
                color,
            )
    
    print(f"[Viz-Meshcat] {len(segments)} segments, {len(best_grasps)} grasps")
    print(f"[Viz-Meshcat] Viewer URL: {vis.url()}")
    
    # Keep server running
    input("Press Enter to close meshcat viewer...")


def _add_gripper_meshcat(
    vis,
    name: str,
    position: np.ndarray,
    normal: np.ndarray,
    suction_cup_radius: float,
    cup_height: float,
    body_size: tuple,
    color: list[float],
):
    """Add gripper geometry to meshcat visualizer.
    
    IMPORTANT: Meshcat's g.Cylinder has its axis along Y, not Z like Open3D!
    We need to rotate by -90° around X to convert from Y-up to Z-up.
    """
    import meshcat.geometry as g
    import meshcat.transformations as tf
    
    # Compute rotation matrix to align local Z with surface normal
    z = np.array([0, 0, 1])
    n = normal / np.linalg.norm(normal)
    v = np.cross(z, n)
    s, c = np.linalg.norm(v), np.dot(z, n)
    
    if s > 1e-6:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    else:
        R = np.eye(3) if c > 0 else np.diag([1, -1, -1])
    
    # Build 4x4 transform: position + rotation to align with normal
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    
    # Color as hex for meshcat
    color_hex = int(color[0] * 255) << 16 | int(color[1] * 255) << 8 | int(color[2] * 255)
    material = g.MeshLambertMaterial(color=color_hex, opacity=0.8)
    
    body_w, body_d, body_h = body_size
    
    # Meshcat cylinder is Y-axis aligned, need to rotate -90° around X to make it Z-aligned
    # Rotation matrix for -90° around X: [[1,0,0], [0,0,1], [0,-1,0]]
    R_y_to_z = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ])
    
    # Suction cup (cylinder) - centered at cup_height/2 above contact
    cup_T = T @ tf.translation_matrix([0, 0, cup_height / 2]) @ R_y_to_z
    vis[f"{name}/cup"].set_object(
        g.Cylinder(cup_height, suction_cup_radius),
        material
    )
    vis[f"{name}/cup"].set_transform(cup_T)
    
    # Body (box) - no rotation needed, Box is already axis-aligned
    body_T = T @ tf.translation_matrix([0, 0, cup_height + body_h / 2])
    vis[f"{name}/body"].set_object(
        g.Box([body_w, body_d, body_h]),
        g.MeshLambertMaterial(color=0x666666, opacity=0.8)
    )
    vis[f"{name}/body"].set_transform(body_T)


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================
def visualize_results(
    segments: list,
    results: list,
    grasp_cfg,
    background_points: np.ndarray = None,
    background_colors: np.ndarray = None,
    occupancy_points: np.ndarray = None,
    use_meshcat: bool = False,
    meshcat_url: str = None,
):
    """
    Visualize grasp detection results.
    
    Args:
        segments: List of SegmentPointCloud objects
        results: List of SegmentResult objects
        grasp_cfg: GraspConfig with gripper dimensions
        background_points: Optional background point cloud
        background_colors: Optional background colors
        occupancy_points: Optional occupancy volume points
        use_meshcat: If True, use meshcat instead of Open3D
        meshcat_url: Optional ZMQ URL for meshcat server
    """
    if use_meshcat:
        visualize_results_meshcat(
            segments, results, grasp_cfg,
            background_points, background_colors, occupancy_points,
            zmq_url=meshcat_url,
        )
    else:
        visualize_results_o3d(
            segments, results, grasp_cfg,
            background_points, background_colors, occupancy_points,
        )
