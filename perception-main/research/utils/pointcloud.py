"""
Point cloud processing utilities.
"""
from __future__ import annotations

from pathlib import Path

import open3d as o3d


# =============================================================================
# CONSTANTS
# =============================================================================
SEGMENT_SAVE_DIR = Path("/home/jeong/perception/data/segments")

# Default parameters
DEFAULT_PLANE_THRESHOLD = 0.005


# =============================================================================
# POINT CLOUD PROCESSING
# =============================================================================
def remove_plane(
    pcd: o3d.geometry.PointCloud, 
    threshold: float = DEFAULT_PLANE_THRESHOLD
) -> o3d.geometry.PointCloud:
    """
    Remove dominant plane (e.g., table/floor) from scene using RANSAC.
    
    Args:
        pcd: Input point cloud
        threshold: RANSAC distance threshold for plane fitting
        
    Returns:
        Point cloud with plane removed
    """
    print(f"\n   Removing dominant plane...")
    
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=threshold,
        ransac_n=3,
        num_iterations=1000
    )
    
    pcd_without_plane = pcd.select_by_index(inliers, invert=True)
    print(f"      Plane removed: {len(pcd.points)} -> {len(pcd_without_plane.points)} points")
    
    return pcd_without_plane


def save_segments(segments, save_dir: Path = SEGMENT_SAVE_DIR) -> None:
    """
    Save all segment point clouds to PLY files with colors.
    
    Args:
        segments: List of SegmentPointCloud namedtuples with fields:
            segment_id, name, points, pcd, color
        save_dir: Directory to save PLY files
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Segments] Saving {len(segments)} segments to {save_dir}")
    
    for seg in segments:
        # Save as PLY file (preserves colors)
        ply_path = save_dir / f"segment_{seg.segment_id}_{seg.name}.ply"
        o3d.io.write_point_cloud(str(ply_path), seg.pcd)
        has_colors = seg.pcd.has_colors()
        print(f"  [{seg.segment_id}] {seg.name}: {len(seg.pcd.points)} pts, "
              f"colors={has_colors} -> {ply_path.name}")
