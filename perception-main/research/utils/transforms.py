"""
Coordinate transformation utilities.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import open3d as o3d


# =============================================================================
# CAMERA-TO-ROBOT TRANSFORM
# =============================================================================
# Calibrated transform from camera frame to robot base frame
T_BASE_CAMERA = np.array([
    [-0.99870602, -0.05068679,  0.00413055,  0.46783590],
    [-0.05061085,  0.99857739,  0.01678431,  0.11403400],
    [-0.00497542,  0.01655354, -0.99985062,  1.02680489],
    [ 0.0,         0.0,         0.0,         1.0        ],
], dtype=np.float64)


# =============================================================================
# POINT TRANSFORMATIONS
# =============================================================================
def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Transform Nx3 points using 4x4 transformation matrix.
    
    Args:
        points: (N, 3) array of 3D points
        T: (4, 4) transformation matrix
        
    Returns:
        (N, 3) array of transformed points
    """
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    points_h = np.hstack([points, ones])
    return (T @ points_h.T).T[:, :3]


def center_point_cloud(pcd: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Center point cloud at origin and return the original centroid.
    
    Args:
        pcd: Input point cloud
        
    Returns:
        Tuple of (centered_pcd, original_centroid)
    """
    centroid = pcd.get_center()
    pcd_centered = o3d.geometry.PointCloud(pcd)
    pcd_centered.translate(-centroid)
    return pcd_centered, centroid
