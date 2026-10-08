"""
Geometric / Heuristic Suction Grasp Detection Pipeline

This module implements a point cloud → graspable surface detection algorithm
for suction-based grasping. It finds local planar patches with low curvature
and normals aligned with the approach direction.

Pipeline:
1. Pre-processing: Voxel grid downsampling, Statistical Outlier Removal
2. Normal Estimation: Covariance-based PCA for surface normals
3. Curvature/Roughness Filtering: Reject high-curvature surfaces
4. Segmentation: Euclidean cluster extraction
5. Ranking & Selection: Score by flatness, verticality, and centroid proximity

Industry Use: Manufacturing, depalletizing, bin picking

Usage:
    from utils.suction_grasp_detection import SuctionGraspDetector
    from utils.realsense_pc import RealsensePointCloud
    
    # Capture point cloud
    pc_capture = RealsensePointCloud()
    points_camera, points_base, _ = pc_capture.capture()
    pc_capture.close()
    
    # Detect grasp candidates
    detector = SuctionGraspDetector()
    grasp_candidates = detector.detect(points_base)
    
    # Get best grasp
    best_grasp = grasp_candidates[0]
    print(f"Position: {best_grasp.position}, Normal: {best_grasp.normal}")
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from scipy.spatial import cKDTree
import warnings


@dataclass
class SuctionGraspCandidate:
    """A candidate suction grasp pose."""
    position: np.ndarray          # 3D position (grasp center)
    normal: np.ndarray            # Surface normal (approach direction)
    score: float                  # Overall grasp quality score
    
    # Component scores
    flatness_score: float = 0.0   # How flat the surface is (0-1)
    verticality_score: float = 0.0  # How vertical the normal is (0-1)
    centroid_score: float = 0.0   # How close to cluster centroid (0-1)
    
    # Geometric properties
    curvature: float = 0.0        # Local curvature
    cluster_id: int = -1          # Which cluster this belongs to
    
    def __repr__(self):
        return (f"SuctionGraspCandidate(pos=[{self.position[0]:.3f}, {self.position[1]:.3f}, "
                f"{self.position[2]:.3f}], score={self.score:.3f})")


@dataclass
class SuctionGraspConfig:
    """Configuration for suction grasp detection."""
    
    # =========================================================================
    # Workspace filtering (same as cluster.py)
    # =========================================================================
    # Dominant plane parameters (table surface in robot base frame)
    plane_normal: Tuple[float, float, float] = (0.0, 0.0, -1.0)  # Plane normal (pointing down)
    plane_d: float = 0.05              # Plane offset
    plane_inlier_thresh: float = 0.004 # Points within 4mm are "on" the plane
    z_above_threshold: float = 0.01   # Min height above plane (25mm)
    
    # Bounding box in XY (robot base frame)
    # Format: ((x_min, x_max), (y_min, y_max)) or None to disable
    bbox_xy: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
    
    # =========================================================================
    # Step 1: Pre-processing
    # =========================================================================
    voxel_size: float = 0.003          # 3mm voxel grid downsampling
    sor_mean_k: int = 50               # Statistical outlier removal: k neighbors
    sor_std_ratio: float = 1.0         # SOR threshold multiplier
    
    # Step 2: Normal estimation
    normal_k_neighbors: int = 30       # Number of neighbors for normal estimation
    orient_normals_toward_camera: bool = True  # Point normals toward camera
    camera_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    
    # Step 3: Curvature filtering
    curvature_threshold: float = 0.05  # Max curvature for valid grasp points
    
    # Step 4: Clustering
    cluster_eps: float = 0.02          # 20mm cluster radius
    cluster_min_points: int = 100      # Minimum points per cluster
    
    # Step 5: Ranking weights
    weight_flatness: float = 0.4       # Weight for flatness score
    weight_verticality: float = 0.4    # Weight for normal alignment with gravity
    weight_centroid: float = 0.2       # Weight for proximity to cluster center
    
    # Grasp constraints
    gravity_vector: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    min_normal_z: float = 0.7          # Minimum Z component of normal (cos ~45°)
    
    # Output
    max_candidates: int = 50           # Maximum number of grasp candidates to return
    top_k_per_cluster: int = 10        # Top candidates per cluster before merging


def filter_workspace(points: np.ndarray, 
                     plane_normal: Tuple[float, float, float],
                     plane_d: float,
                     plane_inlier_thresh: float,
                     z_above_threshold: float,
                     bbox_xy: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter point cloud to workspace: points above the table plane and within bounding box.
    
    Uses the same logic as cluster.py:
    1. Compute height above plane for each point
    2. Keep points that are >= z_above_threshold above plane AND not on plane surface
    3. Filter to XY bounding box if specified
    
    Args:
        points: Nx3 point cloud in robot base frame
        plane_normal: Plane normal vector (a, b, c)
        plane_d: Plane offset d (plane equation: ax + by + cz + d = 0)
        plane_inlier_thresh: Points within this distance are considered "on" the plane
        z_above_threshold: Minimum height above plane to keep
        bbox_xy: Optional ((x_min, x_max), (y_min, y_max)) bounding box
        
    Returns:
        Tuple of (filtered_points, mask)
    """
    n = np.array(plane_normal, dtype=np.float64)
    a, b, c = n
    d = plane_d
    
    # Compute signed distance to plane
    dists = points @ n + d
    inlier_mask = np.abs(dists) < plane_inlier_thresh
    
    # Compute height above plane in Z direction
    if abs(c) > 1e-6:
        # z_plane = (-d - a*x - b*y) / c for each point
        z_plane = (-d - a * points[:, 0] - b * points[:, 1]) / c
        height_above = points[:, 2] - z_plane
        above_mask = (height_above >= z_above_threshold) & (~inlier_mask)
    else:
        # Degenerate case: plane is vertical
        above_mask = np.zeros(len(points), dtype=bool)
    
    # Apply bounding box filter
    if bbox_xy is not None:
        (x_min, x_max), (y_min, y_max) = bbox_xy
        bbox_mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
        )
        above_mask = above_mask & bbox_mask
    
    return points[above_mask], above_mask


def voxel_grid_downsample(points: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample point cloud using voxel grid filtering.
    
    Each voxel keeps the centroid of all points within it.
    
    Args:
        points: Nx3 point cloud
        voxel_size: Size of voxel cubes in meters
        
    Returns:
        Tuple of (downsampled_points, voxel_indices)
    """
    if len(points) == 0:
        return points, np.array([], dtype=int)
    
    # Compute voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(int)
    
    # Create unique key for each voxel
    # Offset to handle negative coordinates
    offset = voxel_indices.min(axis=0)
    voxel_indices_offset = voxel_indices - offset
    
    # Compute linear index
    dims = voxel_indices_offset.max(axis=0) + 1
    linear_idx = (voxel_indices_offset[:, 0] * dims[1] * dims[2] + 
                  voxel_indices_offset[:, 1] * dims[2] + 
                  voxel_indices_offset[:, 2])
    
    # Find unique voxels and compute centroids
    unique_voxels, inverse = np.unique(linear_idx, return_inverse=True)
    
    # Compute centroid for each voxel
    downsampled = np.zeros((len(unique_voxels), 3), dtype=points.dtype)
    counts = np.zeros(len(unique_voxels), dtype=int)
    
    np.add.at(downsampled, inverse, points)
    np.add.at(counts, inverse, 1)
    
    downsampled /= counts[:, np.newaxis]
    
    return downsampled, inverse


def statistical_outlier_removal(points: np.ndarray, mean_k: int = 50, 
                                std_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove statistical outliers based on mean distance to k-nearest neighbors.
    
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
    distances, _ = tree.query(points, k=mean_k + 1)
    mean_distances = distances[:, 1:].mean(axis=1)
    
    global_mean = mean_distances.mean()
    global_std = mean_distances.std()
    threshold = global_mean + std_ratio * global_std
    
    inlier_mask = mean_distances < threshold
    return points[inlier_mask], inlier_mask


def estimate_normals_and_curvature(points: np.ndarray, k: int = 30,
                                   orient_toward: Optional[np.ndarray] = None
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate surface normals and curvature using PCA on local neighborhoods.
    
    For each point P, find k-nearest neighbors and compute the 3x3 covariance matrix.
    The eigenvector corresponding to the smallest eigenvalue is the surface normal.
    Curvature is computed as: C = λ3 / (λ1 + λ2 + λ3)
    
    Args:
        points: Nx3 point cloud
        k: Number of neighbors for local PCA
        orient_toward: Optional point to orient normals toward (e.g., camera position)
        
    Returns:
        Tuple of (normals, curvatures, eigenvalues)
        - normals: Nx3 surface normals
        - curvatures: N curvature values
        - eigenvalues: Nx3 eigenvalues (λ1, λ2, λ3) sorted descending
    """
    n_points = len(points)
    if n_points == 0:
        return (np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3)))
    
    # Build KD-tree for neighbor queries
    tree = cKDTree(points)
    
    # Query k-nearest neighbors
    k_actual = min(k, n_points)
    _, indices = tree.query(points, k=k_actual)
    
    normals = np.zeros((n_points, 3), dtype=np.float64)
    curvatures = np.zeros(n_points, dtype=np.float64)
    eigenvalues = np.zeros((n_points, 3), dtype=np.float64)
    
    for i in range(n_points):
        # Get neighbors
        neighbor_points = points[indices[i]]
        
        # Center the points
        centroid = neighbor_points.mean(axis=0)
        centered = neighbor_points - centroid
        
        # Compute covariance matrix
        cov = centered.T @ centered / (k_actual - 1)
        
        # Eigendecomposition (eigenvalues in ascending order by default)
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # Sort eigenvalues/vectors in descending order
        sort_idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sort_idx]
        eigvecs = eigvecs[:, sort_idx]
        
        # The smallest eigenvalue's eigenvector is the surface normal
        normal = eigvecs[:, 2]  # λ3 (smallest)
        
        # Compute curvature: C = λ3 / (λ1 + λ2 + λ3)
        eigsum = eigvals.sum()
        if eigsum > 1e-10:
            curvature = eigvals[2] / eigsum
        else:
            curvature = 0.0
        
        normals[i] = normal
        curvatures[i] = curvature
        eigenvalues[i] = eigvals
    
    # Orient normals consistently
    if orient_toward is not None:
        # Make normals point toward the camera/viewpoint
        view_vectors = orient_toward - points
        flip_mask = (normals * view_vectors).sum(axis=1) < 0
        normals[flip_mask] *= -1
    
    return normals, curvatures, eigenvalues


def euclidean_cluster_extraction(points: np.ndarray, eps: float = 0.02, 
                                 min_points: int = 100) -> np.ndarray:
    """
    Segment point cloud into clusters using DBSCAN (fast, sklearn-based).
    
    Args:
        points: Nx3 point cloud
        eps: Maximum distance between points in same cluster (meters)
        min_points: Minimum number of points for a valid cluster
        
    Returns:
        Cluster labels (-1 for noise/small clusters)
    """
    n_points = len(points)
    if n_points == 0:
        return np.array([], dtype=int)
    
    # Use sklearn's optimized DBSCAN with ball_tree algorithm
    from sklearn.cluster import DBSCAN
    
    clustering = DBSCAN(
        eps=eps, 
        min_samples=min_points,
        algorithm='ball_tree',  # Efficient for 3D points
        n_jobs=-1  # Use all CPU cores
    ).fit(points)
    
    return clustering.labels_


class SuctionGraspDetector:
    """
    Geometric suction grasp detector using point cloud analysis.
    
    Finds flat, upward-facing surfaces suitable for suction grasping
    by analyzing local surface geometry (normals, curvature) and
    clustering objects in the scene.
    """
    
    def __init__(self, config: Optional[SuctionGraspConfig] = None, verbose: bool = True):
        """
        Initialize the suction grasp detector.
        
        Args:
            config: Detection configuration. Uses defaults if None.
            verbose: Print status messages
        """
        self.config = config or SuctionGraspConfig()
        self.verbose = verbose
        
        # Store intermediate results for debugging/visualization
        self.points_downsampled = None
        self.normals = None
        self.curvatures = None
        self.cluster_labels = None
        self.valid_mask = None
    
    def _log(self, msg: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(msg)
    
    def detect(self, points: np.ndarray) -> List[SuctionGraspCandidate]:
        """
        Detect suction grasp candidates from a point cloud.
        
        Args:
            points: Nx3 point cloud in robot base frame
            
        Returns:
            List of SuctionGraspCandidate sorted by score (best first)
        """
        self._log("=" * 70)
        self._log("SUCTION GRASP DETECTION - Geometric Pipeline")
        self._log("=" * 70)
        
        if len(points) == 0:
            self._log("ERROR: Empty point cloud!")
            return []
        
        self._log(f"Input: {len(points):,} points")
        
        # ================================================================
        # Step 0: Workspace Filtering (same as cluster.py)
        # ================================================================
        self._log("\n[Step 0] Workspace filtering...")
        self._log(f"  Plane: n={self.config.plane_normal}, d={self.config.plane_d}")
        self._log(f"  Above-plane threshold: {self.config.z_above_threshold*1000:.1f}mm")
        if self.config.bbox_xy is not None:
            (x_min, x_max), (y_min, y_max) = self.config.bbox_xy
            self._log(f"  Bounding box: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}]")
        
        points_workspace, workspace_mask = filter_workspace(
            points,
            plane_normal=self.config.plane_normal,
            plane_d=self.config.plane_d,
            plane_inlier_thresh=self.config.plane_inlier_thresh,
            z_above_threshold=self.config.z_above_threshold,
            bbox_xy=self.config.bbox_xy
        )
        
        self._log(f"  Filtered: {len(points):,} → {len(points_workspace):,} points "
                  f"({100*len(points_workspace)/len(points):.1f}%)")
        
        if len(points_workspace) == 0:
            self._log("WARNING: No points in workspace after filtering!")
            return []
        
        # ================================================================
        # Step 1: Pre-processing
        # ================================================================
        self._log("\n[Step 1] Pre-processing...")
        
        # Voxel grid downsampling
        self._log(f"  Voxel grid downsampling (voxel_size={self.config.voxel_size*1000:.1f}mm)...")
        points_ds, _ = voxel_grid_downsample(points_workspace, self.config.voxel_size)
        self._log(f"    {len(points_workspace):,} → {len(points_ds):,} points")
        
        # Statistical outlier removal
        self._log(f"  Statistical outlier removal (k={self.config.sor_mean_k})...")
        points_clean, inlier_mask = statistical_outlier_removal(
            points_ds, 
            mean_k=self.config.sor_mean_k,
            std_ratio=self.config.sor_std_ratio
        )
        outliers = (~inlier_mask).sum()
        self._log(f"    Removed {outliers:,} outliers ({100*outliers/len(points_ds):.1f}%)")
        self._log(f"    Remaining: {len(points_clean):,} points")
        
        self.points_downsampled = points_clean
        
        # ================================================================
        # Step 2: Normal Estimation
        # ================================================================
        self._log("\n[Step 2] Normal estimation...")
        self._log(f"  Computing normals with k={self.config.normal_k_neighbors} neighbors...")
        
        orient_point = self.config.camera_position if self.config.orient_normals_toward_camera else None
        normals, curvatures, eigenvalues = estimate_normals_and_curvature(
            points_clean,
            k=self.config.normal_k_neighbors,
            orient_toward=orient_point
        )
        
        self.normals = normals
        self.curvatures = curvatures
        
        self._log(f"  Curvature range: [{curvatures.min():.4f}, {curvatures.max():.4f}]")
        self._log(f"  Mean curvature: {curvatures.mean():.4f}")
        
        # ================================================================
        # Step 3: Curvature and Roughness Filtering
        # ================================================================
        self._log("\n[Step 3] Curvature filtering...")
        
        # Filter by curvature
        curvature_mask = curvatures < self.config.curvature_threshold
        self._log(f"  Curvature < {self.config.curvature_threshold}: "
                  f"{curvature_mask.sum():,} / {len(curvatures):,} points")
        
        # Filter by normal direction (must point upward for top-down grasp)
        # Normal Z component should be positive (pointing up) and above threshold
        normal_z = normals[:, 2]
        normal_mask = normal_z > self.config.min_normal_z
        self._log(f"  Normal Z > {self.config.min_normal_z}: "
                  f"{normal_mask.sum():,} / {len(normals):,} points")
        
        # Combined validity mask
        valid_mask = curvature_mask & normal_mask
        self._log(f"  Valid grasp points: {valid_mask.sum():,}")
        
        self.valid_mask = valid_mask
        
        if valid_mask.sum() == 0:
            self._log("WARNING: No valid grasp points found!")
            return []
        
        # ================================================================
        # Step 4: Segmentation (Euclidean Cluster Extraction)
        # ================================================================
        self._log("\n[Step 4] Euclidean cluster extraction...")
        
        cluster_labels = euclidean_cluster_extraction(
            points_clean,
            eps=self.config.cluster_eps,
            min_points=self.config.cluster_min_points
        )
        
        self.cluster_labels = cluster_labels
        
        n_clusters = cluster_labels.max() + 1 if len(cluster_labels) > 0 else 0
        self._log(f"  Found {n_clusters} clusters")
        
        for i in range(n_clusters):
            cluster_mask = (cluster_labels == i)
            cluster_size = cluster_mask.sum()
            if cluster_size > 0:
                centroid = points_clean[cluster_mask].mean(axis=0)
                self._log(f"    Cluster {i}: {cluster_size:,} points, centroid: {centroid}")
            else:
                self._log(f"    Cluster {i}: {cluster_size:,} points")
        
        # ================================================================
        # Step 5: Ranking and Selection
        # ================================================================
        self._log("\n[Step 5] Ranking and selection...")
        
        gravity = self.config.gravity_vector / np.linalg.norm(self.config.gravity_vector)
        
        # VECTORIZED scoring for all valid points at once
        valid_indices = np.where(valid_mask)[0]
        valid_points = points_clean[valid_indices]
        valid_normals = normals[valid_indices]
        valid_curvatures = curvatures[valid_indices]
        valid_cluster_ids = cluster_labels[valid_indices]
        
        n_valid = len(valid_indices)
        self._log(f"  Scoring {n_valid:,} valid points...")
        
        # Flatness scores (vectorized)
        flatness_scores = 1.0 - (valid_curvatures / self.config.curvature_threshold)
        flatness_scores = np.clip(flatness_scores, 0.0, 1.0)
        
        # Verticality scores (vectorized dot product)
        verticality_scores = valid_normals @ (-gravity)
        verticality_scores = np.clip(verticality_scores, 0.0, 1.0)
        
        # Centroid scores per cluster (vectorized per cluster)
        centroid_scores = np.zeros(n_valid, dtype=np.float64)
        
        for cluster_id in range(n_clusters):
            cluster_mask_local = valid_cluster_ids == cluster_id
            if not cluster_mask_local.any():
                continue
            
            cluster_pts = valid_points[cluster_mask_local]
            centroid = cluster_pts.mean(axis=0)
            
            # Distances to centroid
            dists = np.linalg.norm(cluster_pts - centroid, axis=1)
            max_dist = dists.max()
            
            if max_dist > 1e-6:
                scores = 1.0 - (dists / max_dist)
            else:
                scores = np.ones(len(dists))
            
            centroid_scores[cluster_mask_local] = scores
        
        centroid_scores = np.clip(centroid_scores, 0.0, 1.0)
        
        # Combined scores (vectorized)
        total_scores = (
            self.config.weight_flatness * flatness_scores +
            self.config.weight_verticality * verticality_scores +
            self.config.weight_centroid * centroid_scores
        )
        
        self._log(f"  Total candidates before filtering: {n_valid:,}")
        
        # Select top k per cluster efficiently using argsort
        final_candidates = []
        
        for cluster_id in range(n_clusters):
            cluster_mask_local = valid_cluster_ids == cluster_id
            cluster_indices_local = np.where(cluster_mask_local)[0]
            
            if len(cluster_indices_local) == 0:
                continue
            
            # Get scores for this cluster and find top k
            cluster_scores = total_scores[cluster_indices_local]
            top_k = min(self.config.top_k_per_cluster, len(cluster_scores))
            top_indices = np.argsort(cluster_scores)[::-1][:top_k]
            
            # Create candidate objects only for top k
            for local_idx in top_indices:
                global_idx = cluster_indices_local[local_idx]
                
                candidate = SuctionGraspCandidate(
                    position=valid_points[global_idx].copy(),
                    normal=valid_normals[global_idx].copy(),
                    score=float(total_scores[global_idx]),
                    flatness_score=float(flatness_scores[global_idx]),
                    verticality_score=float(verticality_scores[global_idx]),
                    centroid_score=float(centroid_scores[global_idx]),
                    curvature=float(valid_curvatures[global_idx]),
                    cluster_id=cluster_id
                )
                final_candidates.append(candidate)
        
        # Sort all candidates globally and limit
        final_candidates = sorted(final_candidates, key=lambda x: x.score, reverse=True)
        final_candidates = final_candidates[:self.config.max_candidates]
        
        self._log(f"  Final candidates: {len(final_candidates)}")
        
        if final_candidates:
            self._log(f"\n  Top 5 grasp candidates:")
            for i, c in enumerate(final_candidates[:5]):
                self._log(f"    {i+1}. Score={c.score:.3f} (flat={c.flatness_score:.2f}, "
                          f"vert={c.verticality_score:.2f}, cent={c.centroid_score:.2f}) "
                          f"@ [{c.position[0]:.3f}, {c.position[1]:.3f}, {c.position[2]:.3f}]")
        
        self._log("\n" + "=" * 70)
        
        return final_candidates
    
    def get_best_grasp(self, candidates: Optional[List[SuctionGraspCandidate]] = None
                       ) -> Optional[SuctionGraspCandidate]:
        """
        Get the best grasp candidate.
        
        Args:
            candidates: List of candidates (runs detect() if None)
            
        Returns:
            Best SuctionGraspCandidate or None if no candidates
        """
        if candidates is None or len(candidates) == 0:
            return None
        return candidates[0]
    
    def visualize_normals_and_curvature(self, vis=None, 
                                         normal_length: float = 0.01,
                                         normal_subsample: int = 10,
                                         curvature_colormap: str = "jet"):
        """
        Visualize surface normals and curvature.
        
        Args:
            vis: Meshcat visualizer (creates new if None)
            normal_length: Length of normal arrows in meters
            normal_subsample: Show every Nth normal (for performance)
            curvature_colormap: Colormap for curvature ("jet", "hot", "viridis")
            
        Returns:
            Meshcat visualizer
        """
        import meshcat
        import meshcat.geometry as g
        
        if self.points_downsampled is None or self.normals is None or self.curvatures is None:
            print("ERROR: Run detect() first to compute normals and curvature")
            return None
        
        if vis is None:
            vis = meshcat.Visualizer().open()
        
        # Clear previous
        vis["normals_viz"].delete()
        vis["curvature_viz"].delete()
        
        points = self.points_downsampled
        normals = self.normals
        curvatures = self.curvatures
        
        # =====================================================================
        # Visualize points colored by curvature (using binned color groups)
        # =====================================================================
        # Normalize curvature to [0, 1] for coloring
        c_min, c_max = curvatures.min(), curvatures.max()
        if c_max > c_min:
            c_norm = (curvatures - c_min) / (c_max - c_min)
        else:
            c_norm = np.zeros_like(curvatures)
        
        # Bin points by curvature into color groups (meshcat doesn't support vertex colors)
        # Blue (flat) -> Cyan -> Green -> Yellow -> Red (curved)
        color_bins = [
            (0.0, 0.2, 0x0000ff, "blue (flat)"),      # Blue
            (0.2, 0.4, 0x00ffff, "cyan"),             # Cyan
            (0.4, 0.6, 0x00ff00, "green"),            # Green
            (0.6, 0.8, 0xffff00, "yellow"),           # Yellow
            (0.8, 1.01, 0xff0000, "red (curved)"),    # Red
        ]
        
        for i, (low, high, color, name) in enumerate(color_bins):
            mask = (c_norm >= low) & (c_norm < high)
            if mask.sum() > 0:
                bin_pts = points[mask].T.astype(np.float32)
                vis["curvature_viz"][f"bin_{i}_{name}"].set_object(
                    g.Points(
                        g.PointsGeometry(bin_pts),
                        g.PointsMaterial(size=0.003, color=color)
                    )
                )
        
        print(f"Curvature range: [{c_min:.4f}, {c_max:.4f}]")
        print(f"  Blue = flat (C~0), Red = curved (C~{c_max:.3f})")
        
        # =====================================================================
        # Visualize normals as arrows (subsampled for performance)
        # =====================================================================
        indices = np.arange(0, len(points), normal_subsample)
        n_normals = len(indices)
        
        # Create line segments for all normals at once
        # Each normal is a line from point to point + normal * length
        starts = points[indices]
        ends = starts + normals[indices] * normal_length
        
        # Interleave starts and ends for line segments
        # Format: [start0, end0, start1, end1, ...]
        segments = np.zeros((n_normals * 2, 3), dtype=np.float32)
        segments[0::2] = starts
        segments[1::2] = ends
        
        vis["normals_viz"]["arrows"].set_object(
            g.LineSegments(
                g.PointsGeometry(segments.T),
                g.LineBasicMaterial(color=0x00ffff, linewidth=1)  # Cyan
            )
        )
        
        print(f"Showing {n_normals} normals (every {normal_subsample}th point)")
        print(f"Normal length: {normal_length*1000:.1f}mm")
        print(f"Meshcat URL: {vis.url()}")
        
        return vis
    
    def visualize(self, candidates: List[SuctionGraspCandidate], vis=None):
        """
        Visualize grasp detection results in Meshcat.
        
        Shows the best grasp candidate per cluster, colored by global score ranking:
        Red (best) -> Orange -> Yellow -> Green -> Cyan -> Blue (worst)
        
        Args:
            candidates: List of grasp candidates
            vis: Meshcat visualizer (creates new if None)
        """
        import meshcat
        import meshcat.geometry as g
        
        if vis is None:
            vis = meshcat.Visualizer().open()
        
        # Clear previous visualization
        vis["grasp_detection"].delete()
        
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
        
        # Draw downsampled point cloud colored by cluster
        if self.points_downsampled is not None and self.cluster_labels is not None:
            n_clusters = self.cluster_labels.max() + 1 if len(self.cluster_labels) > 0 else 0
            
            # Color palette for clusters
            cluster_colors = [
                0x1f77b4, 0xff7f0e, 0x2ca02c, 0xd62728, 0x9467bd,
                0x8c564b, 0xe377c2, 0x7f7f7f, 0xbcbd22, 0x17becf
            ]
            
            for cluster_id in range(n_clusters):
                mask = self.cluster_labels == cluster_id
                if mask.sum() == 0:
                    continue
                    
                cluster_pts = self.points_downsampled[mask]
                color = cluster_colors[cluster_id % len(cluster_colors)]
                
                # Draw cluster points
                vis["grasp_detection"]["clusters"][f"cluster_{cluster_id}"].set_object(
                    g.Points(
                        g.PointsGeometry(cluster_pts.T.astype(np.float32)),
                        g.PointsMaterial(size=0.003, color=color)
                    )
                )
                
                # Draw cluster centroid as a sphere
                centroid = cluster_pts.mean(axis=0)
                vis["grasp_detection"]["centroids"][f"centroid_{cluster_id}"].set_object(
                    g.Sphere(0.008),  # 8mm radius sphere
                    g.MeshLambertMaterial(color=color)
                )
                T_centroid = np.eye(4)
                T_centroid[:3, 3] = centroid
                vis["grasp_detection"]["centroids"][f"centroid_{cluster_id}"].set_transform(T_centroid)
            
            # Draw noise points (unclustered)
            noise_mask = self.cluster_labels == -1
            if noise_mask.sum() > 0:
                noise_pts = self.points_downsampled[noise_mask].T.astype(np.float32)
                vis["grasp_detection"]["clusters"]["noise"].set_object(
                    g.Points(
                        g.PointsGeometry(noise_pts),
                        g.PointsMaterial(size=0.001, color=0x888888)
                    )
                )
        
        # Draw valid grasp points
        if self.points_downsampled is not None and self.valid_mask is not None:
            valid_pts = self.points_downsampled[self.valid_mask].T.astype(np.float32)
            vis["grasp_detection"]["valid_points"].set_object(
                g.Points(
                    g.PointsGeometry(valid_pts),
                    g.PointsMaterial(size=0.004, color=0x00ff00)
                )
            )
        
        # Draw grasp candidates - one best grasp per cluster
        # Color by global ranking: Red (best) -> Orange -> Yellow -> Green -> Cyan -> Blue (worst)
        
        def get_rank_color(rank: int, total: int) -> int:
            """
            Get color based on global ranking.
            Rank 0 (best) = Red (0xff0000)
            Last rank = Blue (0x0000ff)
            Interpolates through: Red -> Orange -> Yellow -> Green -> Cyan -> Blue
            """
            if total <= 1:
                return 0xff0000  # Red for single candidate
            
            # Normalize rank to [0, 1]
            t = rank / (total - 1)
            
            # Color stops: Red -> Orange -> Yellow -> Green -> Cyan -> Blue
            # Each segment is 0.2 of the gradient
            if t < 0.2:
                # Red (255, 0, 0) -> Orange (255, 128, 0)
                s = t / 0.2
                r, g, b = 255, int(128 * s), 0
            elif t < 0.4:
                # Orange (255, 128, 0) -> Yellow (255, 255, 0)
                s = (t - 0.2) / 0.2
                r, g, b = 255, int(128 + 127 * s), 0
            elif t < 0.6:
                # Yellow (255, 255, 0) -> Green (0, 255, 0)
                s = (t - 0.4) / 0.2
                r, g, b = int(255 * (1 - s)), 255, 0
            elif t < 0.8:
                # Green (0, 255, 0) -> Cyan (0, 255, 255)
                s = (t - 0.6) / 0.2
                r, g, b = 0, 255, int(255 * s)
            else:
                # Cyan (0, 255, 255) -> Blue (0, 0, 255)
                s = (t - 0.8) / 0.2
                r, g, b = 0, int(255 * (1 - s)), 255
            
            return (r << 16) | (g << 8) | b
        
        # Filter to get only the best candidate per cluster
        # Candidates are already sorted by score, so first occurrence of each cluster_id is the best
        best_per_cluster = []
        seen_clusters = set()
        for candidate in candidates:
            if candidate.cluster_id not in seen_clusters:
                seen_clusters.add(candidate.cluster_id)
                best_per_cluster.append(candidate)
        
        # Sort by score for global ranking (should already be sorted, but ensure it)
        best_per_cluster = sorted(best_per_cluster, key=lambda x: x.score, reverse=True)
        
        n_to_show = len(best_per_cluster)
        
        for i, candidate in enumerate(best_per_cluster):
            # Get color based on global ranking
            cylinder_color = get_rank_color(i, n_to_show)
            
            # Draw suction cup position
            radius = 0.01  # 10mm suction cup visualization
            height = 0.02
            
            # Create cylinder for suction cup with rank-based color
            vis["grasp_detection"]["candidates"][f"grasp_{i}"]["cup"].set_object(
                g.Cylinder(height, radius),
                g.MeshLambertMaterial(color=cylinder_color)
            )
            
            # Orient cylinder along normal
            # Meshcat Cylinder is along Y-axis by default, we need to rotate to align with normal
            normal = candidate.normal
            y_axis = np.array([0, 1, 0])  # Default cylinder axis
            
            # Compute rotation from y_axis to normal using Rodrigues' formula
            v = np.cross(y_axis, normal)
            s = np.linalg.norm(v)
            c = np.dot(y_axis, normal)
            
            if s > 1e-6:
                # Rodrigues' rotation formula as matrix
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
            else:
                # Vectors are parallel or anti-parallel
                if c > 0:
                    R = np.eye(3)  # Same direction
                else:
                    # Opposite direction - rotate 180° around X or Z axis
                    R = np.diag([1, -1, -1])
            
            # Build 4x4 transform
            # Position the cylinder so its base is at the grasp point
            # Cylinder center is at origin, so offset by half height along normal
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = candidate.position + normal * height / 2
            
            vis["grasp_detection"]["candidates"][f"grasp_{i}"]["cup"].set_transform(T)
            
            # Draw normal arrow (same color as cylinder)
            arrow_length = 0.05
            arrow_start = candidate.position
            arrow_end = candidate.position + normal * arrow_length
            arrow_pts = np.array([arrow_start, arrow_end]).T.astype(np.float32)
            
            vis["grasp_detection"]["candidates"][f"grasp_{i}"]["normal"].set_object(
                g.Line(g.PointsGeometry(arrow_pts), 
                       g.LineBasicMaterial(color=cylinder_color, linewidth=3))
            )
        
        print(f"Visualized {n_to_show} grasp candidates (1 per cluster, color: Red=best → Blue=worst)")
        print(f"Meshcat URL: {vis.url()}")
        
        return vis


# ============================================================================
# Main Script (when run directly)
# ============================================================================

if __name__ == "__main__":
    try:
        from utils.realsense_pc import RealsensePointCloud
    except ModuleNotFoundError:
        from realsense_pc import RealsensePointCloud
    
    print("=" * 70)
    print("Suction Grasp Detection Demo")
    print("=" * 70)
    
    # Capture point cloud
    print("\n[1] Capturing point cloud from RealSense...")
    pc_capture = RealsensePointCloud()
    points_camera, points_base, color_image = pc_capture.capture()
    pc_capture.close()
    
    # Detect suction grasps
    print("\n[2] Running suction grasp detection...")
    detector = SuctionGraspDetector()
    candidates = detector.detect(points_base)
    
    # Print results
    if candidates:
        best = candidates[0]
        print(f"\nBest grasp: pos=[{best.position[0]:.4f}, {best.position[1]:.4f}, {best.position[2]:.4f}], "
              f"score={best.score:.3f}, cluster={best.cluster_id}")
    else:
        print("No valid grasp candidates found!")
    
    # Visualize everything
    print("\n[3] Visualizing...")
    vis = detector.visualize_normals_and_curvature(normal_length=0.015, normal_subsample=5)
    vis = detector.visualize(candidates, vis=vis)
    
    print("\n" + "=" * 70)
    input("Press Enter to exit...")
