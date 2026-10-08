import numpy as np
import time
import cv2
from pathlib import Path

# =============================================================================
# CONFIGURATION - All tunable parameters in one place
# =============================================================================

# --- Calibration: Robot base to camera transform ---
# T_base_camera = np.array([
#     [-0.99870602, -0.05068679,  0.00413055,  0.48886996],
#     [-0.05061085,  0.99857739,  0.01678431,  0.10034969],
#     [-0.00497542,  0.01655354, -0.99985062,  1.02528209],
#     [ 0.0, 0.0, 0.0, 1.0]
# ])
# T_base_camera = np.array([
#     [-0.99870602, -0.05068679,  0.00413055,  0.46886996],
#     [-0.05061085,  0.99857739,  0.01678431,  0.10034969],
#     [-0.00497542,  0.01655354, -0.99985062,  1.02528209],
#     [ 0.0, 0.0, 0.0, 1.0]
# ])

T_base_camera = np.array([
    [ -0.99870602,  -0.05068679,   0.00413055,   0.45301565],
    [ -0.05061085,   0.99857739,   0.01678431,   0.12875332],
    [ -0.00497542,   0.01655354,  -0.99985062,   1.02653152],
    [  0.00000000,   0.00000000,   0.00000000,   1.00000000],
])


# --- Camera Settings ---
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
DEPTH_UNITS = 0.0001  # 0.1mm precision
LASER_POWER = 360     # Max power for better depth

# --- Depth Filter Settings ---
# Enable/disable individual filters
FILTER_DECIMATION_ENABLED = True
FILTER_DISPARITY_ENABLED = False      # depth_to_disparity + disparity_to_depth
FILTER_SPATIAL_ENABLED = False
FILTER_TEMPORAL_ENABLED = True
FILTER_HOLE_FILLING_ENABLED = False

# Filter parameters
FILTER_DECIMATION_MAGNITUDE = 1
FILTER_SPATIAL_ALPHA = 0.25
FILTER_SPATIAL_DELTA = 20
FILTER_TEMPORAL_ALPHA = 0.2
FILTER_TEMPORAL_DELTA = 20

# --- Grasp Detection Parameters ---
# Plane fitting
RANSAC_MAX_ITERS = 800
RANSAC_INLIER_THRESH = 0.004  # meters - tolerance for points on plane

# Default dominant plane (table surface in base frame)
# Set to None to auto-detect via RANSAC
DEFAULT_PLANE_NORMAL = (0.0, 0.0, -1.0)
DEFAULT_PLANE_D = 0.08

# Above-plane filtering
Z_ABOVE_THRESHOLD = 0.025  # meters - minimum height above plane to consider

# Subplane clustering
DISTANCE_EPS = 0.01        # meters - thickness tolerance for plane slabs
MIN_POINTS = 400           # minimum points per detected subplane
ANGLE_TOL_DEG = 10.0       # degrees - max angle deviation from target normal
SPATIAL_VOXEL_M = 0.02     # meters - grid size for spatial clustering
NORMAL_K = 30              # k-neighbors for local normal estimation

# Region of interest (base frame coordinates)
# Format: ((x_min, x_max), (y_min, y_max))
BBOX_XY = ((0.25, 0.75), (-0.25, 0.25))

# --- Visualization ---
VIZ_STRIDE = 4             # downsample factor for point cloud visualization

# =============================================================================

# =============================================================================
# AI MODEL (lazy-loaded singleton)
# =============================================================================

_ai_model = None
_ai_class_names = None

# Try to import AI config; fall back to defaults if config.py not on path
try:
    from config import USE_AI, AI_CONFIDENCE_THRESHOLD, AI_RESOLUTION
except ImportError:
    USE_AI = False
    AI_CONFIDENCE_THRESHOLD = 0.5
    AI_RESOLUTION = 640

# ROI margins — see config.py for rationale.  Safe fallbacks if config
# isn't importable (e.g. when this module is used standalone).
try:
    from config import BBOX_MARGIN_M, BBOX_MARGIN_PX
except ImportError:
    BBOX_MARGIN_M = 0.20
    BBOX_MARGIN_PX = 100

_FINETUNED_WEIGHTS = Path(__file__).parent / "ai" / "output" / "checkpoint_best_total.pth"


def get_ai_model(force=False):
    """Load the RF-DETR model (once). Returns model or None."""
    global _ai_model, _ai_class_names
    if _ai_model is not None:
        return _ai_model
    if not force and not USE_AI:
        return None
    if not _FINETUNED_WEIGHTS.exists():
        print(f"[AI] Weights not found: {_FINETUNED_WEIGHTS}")
        return None
    from rfdetr import RFDETRMedium
    print(f"[AI] Loading RF-DETR Medium (resolution={AI_RESOLUTION}) ...")
    _ai_model = RFDETRMedium(pretrain_weights=str(_FINETUNED_WEIGHTS), resolution=AI_RESOLUTION)
    _ai_model.optimize_for_inference()
    _ai_class_names = _ai_model.class_names
    print(f"[AI] Classes: {_ai_class_names}")
    print("[AI] Warming up ...")
    _ai_model.predict(np.zeros((480, 640, 3), dtype=np.uint8), threshold=0.5)
    print("[AI] Model ready.")
    return _ai_model


def run_ai_detection(color_bgr):
    """Run AI detector on a BGR image. Returns list of [x1,y1,x2,y2] or empty list."""
    model = get_ai_model()
    if model is None:
        return []
    img_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    detections = model.predict(img_rgb, threshold=AI_CONFIDENCE_THRESHOLD)
    if detections.xyxy is None or len(detections.xyxy) == 0:
        return []
    return detections.xyxy.astype(np.float32).tolist()


def compute_point_labels_from_bboxes(pixel_coords, bboxes):
    """
    Assign each point a bbox label based on containment in color-image pixel space.

    Parameters:
        pixel_coords: (N, 2) float array of (px, py) in color image
        bboxes: list of [x1, y1, x2, y2]

    Returns:
        (N,) int32 array — bbox index if inside exactly one bbox, -1 otherwise
    """
    labels = np.full(len(pixel_coords), -1, dtype=np.int32)
    if not bboxes:
        return labels
    px, py = pixel_coords[:, 0], pixel_coords[:, 1]
    hit_count = np.zeros(len(pixel_coords), dtype=np.int32)
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        inside = (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)
        labels[inside & (hit_count == 0)] = i
        labels[inside & (hit_count > 0)] = -1
        hit_count[inside] += 1
    return labels


def project_pts_to_color_pixels(pts_cam, color_intrinsics, depth_to_color_extrinsic=None):
    """
    Project depth-camera-frame 3D points to color image pixel coordinates.

    Parameters:
        pts_cam: (N, 3) points in depth camera frame
        color_intrinsics: dict with keys fx, fy, cx, cy
        depth_to_color_extrinsic: dict with keys R (3x3 array), t (3, array),
                                  or None to assume identity (aligned cameras)

    Returns:
        (N, 2) float array of (px, py)
    """
    if depth_to_color_extrinsic is not None:
        R = np.asarray(depth_to_color_extrinsic['R'])
        t = np.asarray(depth_to_color_extrinsic['t'])
        pts = (R @ pts_cam.T).T + t
    else:
        pts = pts_cam
    fx = color_intrinsics['fx']
    fy = color_intrinsics['fy']
    cx = color_intrinsics['cx']
    cy = color_intrinsics['cy']
    z = pts[:, 2]
    z_safe = np.where(z > 1e-6, z, 1e-6)
    px = (pts[:, 0] / z_safe) * fx + cx
    py = (pts[:, 1] / z_safe) * fy + cy
    return np.stack([px, py], axis=1)


def _split_component_by_labels(comp_local_idx, inlier_labels, seg_inliers):
    """
    Split a BFS component into sub-components by AI point labels.

    Points with label -1 (ambiguous/outside) are assigned to the nearest
    labeled neighbor within the component.

    Returns list of index arrays. If <= 1 distinct label, returns
    [comp_local_idx] unchanged (no split).
    """
    from scipy.spatial import cKDTree

    labels = inlier_labels[comp_local_idx].copy()
    unique_positive = np.unique(labels[labels >= 0])

    if len(unique_positive) <= 1:
        return [comp_local_idx]

    unlabeled_mask = labels < 0
    if unlabeled_mask.any() and not unlabeled_mask.all():
        labeled_mask = ~unlabeled_mask
        labeled_pts = seg_inliers[comp_local_idx[labeled_mask]]
        labeled_labels = labels[labeled_mask]
        unlabeled_pts = seg_inliers[comp_local_idx[unlabeled_mask]]
        tree = cKDTree(labeled_pts)
        _, nearest_idx = tree.query(unlabeled_pts)
        labels[unlabeled_mask] = labeled_labels[nearest_idx]

    sub_groups = []
    for lbl in np.unique(labels[labels >= 0]):
        mask = labels == lbl
        if mask.any():
            sub_groups.append(comp_local_idx[mask])

    remaining = labels < 0
    if remaining.any():
        sub_groups.append(comp_local_idx[remaining])

    return sub_groups


# =============================================================================

# ---------------------------
# Plane fitting
# ----------------------------
def fit_plane_ransac_from_depth(
    pts, max_iters=800, inlier_thresh=0.004, roi=None
):

    best_inliers = -1
    best_plane = None
    rng = np.random.default_rng(123)

    for _ in range(max_iters):
        # Randomly sample 3 non-collinear points
        idx = rng.choice(len(pts), size=3, replace=False)
        p0, p1, p2 = pts[idx]
        v1, v2 = p1 - p0, p2 - p0
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n /= norm
        d = -np.dot(n, p0)

        # Point-to-plane distance
        dists = np.abs(pts @ n + d)
        inliers = np.count_nonzero(dists < inlier_thresh)
        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = (n.copy(), d)

    # Refine with least squares on inliers
    if best_plane is None:
        raise RuntimeError("Plane RANSAC failed")
    n, d = best_plane
    dists = np.abs(pts @ n + d)
    inlier_mask = dists < inlier_thresh
    P = pts[inlier_mask]
    
    # PCA: plane normal is eigenvector of smallest eigenvalue of covariance
    Pc = P - P.mean(axis=0, keepdims=True)
    C = Pc.T @ Pc
    w, V = np.linalg.eigh(C)
    n_refined = V[:, np.argmin(w)]
    n_refined /= np.linalg.norm(n_refined)
    d_refined = -np.dot(n_refined, P.mean(axis=0))
    
    # Make normal point approximately toward camera (positive z) for consistency
    if n_refined[2] < 0:
        n_refined = -n_refined
        d_refined = -d_refined

    return n_refined.astype(np.float32), float(d_refined), inlier_mask


import numpy as np

def find_parallel_subplanes(points_xyz: np.ndarray,
                             n_target: np.ndarray,
                             distance_eps: float = 0.01,
                             min_points: int = 500,
                             angle_tol_deg: float = 10.0,
                             spatial_voxel_m: float = 0.02):
    """
    Find all planar patches whose normals are close to n_target, allowing different offsets d.
    - points_xyz: (N,3) point cloud
    - n_target: (3,) target normal (not necessarily normalized)
    - distance_eps: thickness tolerance along n_target (meters)
    - min_points: minimum inliers per plane to keep
    - angle_tol_deg: max angle difference between fitted normal and n_target
    - spatial_voxel_m: 2D grid size (meters) to split disjoint patches within a slab
    Returns: list of dicts { 'n': (3,), 'd': float, 'indices': (K,) }
    """
    assert points_xyz.ndim == 2 and points_xyz.shape[1] == 3
    n_unit = n_target / np.linalg.norm(n_target)

    # 1) Project onto the target normal (planes parallel to n_unit form 1D modes)
    s = points_xyz @ n_unit  # shape (N,)
    order = np.argsort(s)
    s_sorted = s[order]
    pts_sorted = points_xyz[order]

    # 2) Simple 1D clustering by gaps in s
    gaps = np.diff(s_sorted)
    split_positions = np.where(gaps > distance_eps)[0]
    starts = np.r_[0, split_positions + 1]
    ends = np.r_[split_positions, len(s_sorted) - 1]

    results = []
    cos_angle_thresh = np.cos(np.deg2rad(angle_tol_deg))
    print('starts', starts)

    for start, end in zip(starts, ends):
        seg_idx_sorted = np.arange(start, end + 1)
        seg = pts_sorted[seg_idx_sorted]
        print()
        print('start', start, 'end', end)
        print('seg.shape', seg.shape, seg.mean(axis=0))
        if seg.shape[0] < min_points:
            continue

        # 3) Fit plane by PCA
        centroid = seg.mean(axis=0)
        Pc = seg - centroid
        C = Pc.T @ Pc
        w, V = np.linalg.eigh(C)
        n_fit = V[:, 0]  # eigenvector for smallest eigenvalue

        print('n_fit', n_fit)
        print('centroid', centroid)

        # Make normal direction consistent with n_target
        if np.dot(n_fit, n_unit) < 0:
            n_fit = -n_fit

        print('n_fit', np.dot(n_fit, n_unit))

        # Check angle to target normal
        cos_ang = float(np.clip(np.dot(n_fit, n_unit), -1.0, 1.0))
        print('cos_ang', cos_ang, cos_angle_thresh)
        if cos_ang < cos_angle_thresh:
            continue
        

        d_fit = -float(np.dot(n_fit, centroid))
        print('d_fit', d_fit)
        # 4) Tighten inliers by true point-to-plane distance (<= distance_eps)
        dist_all = np.abs(seg @ n_fit + d_fit)
        inlier_mask_all = dist_all <= distance_eps
        print('inlier_mask_all', inlier_mask_all.sum())
        if inlier_mask_all.sum() < min_points:
            continue



        # 5) Split spatially within the plane using a grid-based BFS in (u,v)
        seg_inliers = seg[inlier_mask_all]
        seg_inliers_idx_sorted = seg_idx_sorted[inlier_mask_all]

        # Tangent basis from PCA (eigenvectors for largest eigenvalues)
        t1 = V[:, 1]
        t2 = V[:, 2]
        u = seg_inliers @ t1
        v = seg_inliers @ t2

        iu = np.floor(u / spatial_voxel_m).astype(np.int64)
        iv = np.floor(v / spatial_voxel_m).astype(np.int64)

        bins = {}
        for j, key in enumerate(zip(iu, iv)):
            if key in bins:
                bins[key].append(j)
            else:
                bins[key] = [j]

        visited = set()
        # Local function to iterate 8-neighborhood
        def neighbors(k):
            a, b = k
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    if da == 0 and db == 0:
                        continue
                    yield (a + da, b + db)

        from collections import deque

        for start_key in list(bins.keys()):
            if start_key in visited:
                continue
            queue = deque([start_key])
            visited.add(start_key)
            comp_bins = []
            while queue:
                k = queue.popleft()
                comp_bins.append(k)
                for nk in neighbors(k):
                    if nk in bins and nk not in visited:
                        visited.add(nk)
                        queue.append(nk)

            # Collect points in this connected component
            comp_local_idx = []
            for bkey in comp_bins:
                comp_local_idx.extend(bins[bkey])
            if not comp_local_idx:
                continue
            comp_local_idx = np.array(comp_local_idx, dtype=np.int64)

            comp_pts = seg_inliers[comp_local_idx]
            if comp_pts.shape[0] < min_points:
                continue

            # Refine plane on component and re-check alignment and distance
            comp_centroid = comp_pts.mean(axis=0)
            Pc_c = comp_pts - comp_centroid
            Cc = Pc_c.T @ Pc_c
            w_c, V_c = np.linalg.eigh(Cc)
            n_fit_c = V_c[:, 0]
            if np.dot(n_fit_c, n_unit) < 0:
                n_fit_c = -n_fit_c
            cos_ang_c = float(np.clip(np.dot(n_fit_c, n_unit), -1.0, 1.0))
            if cos_ang_c < cos_angle_thresh:
                continue
            d_fit_c = -float(np.dot(n_fit_c, comp_centroid))
            dist_c = np.abs(comp_pts @ n_fit_c + d_fit_c)
            inlier_mask_c = dist_c <= distance_eps
            if inlier_mask_c.sum() < min_points:
                continue

            # Map back to original indices
            final_local_idx = comp_local_idx[inlier_mask_c]
            orig_indices = order[seg_inliers_idx_sorted[final_local_idx]]
            results.append({'n': n_fit_c.astype(np.float64),
                            'd': d_fit_c,
                            'indices': orig_indices})

    return results


def find_parallel_subplanes_v2(
    points_xyz: np.ndarray,
    n_target: np.ndarray,
    distance_eps: float = 0.01,
    min_points: int = 500,
    angle_tol_deg: float = 10.0,
    spatial_voxel_m: float = 0.02,
    normal_k: int = 30,
    point_labels: np.ndarray = None,
):
    """
    Find planar patches parallel to n_target, robust to box sides.
    
    Key improvement over v1: Estimates local surface normals first, then only 
    clusters points whose local normal is close to n_target. This excludes 
    box side points that would otherwise connect top surfaces in 1D projection.
    
    Parameters:
        points_xyz: (N, 3) point cloud
        n_target: (3,) target normal vector (not necessarily normalized)
        distance_eps: thickness tolerance along n_target (meters)
        min_points: minimum inliers per plane to keep
        angle_tol_deg: max angle difference between local/fitted normal and n_target
        spatial_voxel_m: 2D grid size (meters) to split disjoint patches within a slab
        normal_k: number of neighbors for local normal estimation
        
    Returns:
        list of dicts { 'n': (3,), 'd': float, 'indices': (K,) }
    """
    from scipy.spatial import cKDTree
    from collections import deque
    
    assert points_xyz.ndim == 2 and points_xyz.shape[1] == 3
    n_unit = n_target / np.linalg.norm(n_target)
    cos_angle_thresh = np.cos(np.deg2rad(angle_tol_deg))
    
    # print(f"[v2] Input points: {len(points_xyz)}, normal_k={normal_k}, angle_tol={angle_tol_deg}°")
    
    # 1) Estimate local surface normals using PCA on k-nearest neighbors
    tree = cKDTree(points_xyz)
    
    # Query k nearest neighbors for each point (batch query)
    _, indices = tree.query(points_xyz, k=normal_k)
    
    # Vectorized normal estimation
    # Gather neighbor points for all points at once
    neighbor_pts = points_xyz[indices]  # Shape: (N, k, 3)
    centroids = neighbor_pts.mean(axis=1, keepdims=True)  # (N, 1, 3)
    centered = neighbor_pts - centroids  # (N, k, 3)
    
    # Compute covariance matrices: (N, 3, 3)
    # cov[i] = centered[i].T @ centered[i]
    cov = np.einsum('nki,nkj->nij', centered, centered)
    
    # Compute eigenvalues/eigenvectors for all covariance matrices
    eigvals, eigvecs = np.linalg.eigh(cov)  # eigvecs: (N, 3, 3), sorted ascending
    
    # Smallest eigenvector (first column) is the normal
    normals = eigvecs[:, :, 0]  # (N, 3)
    
    # Make normals consistent with n_target direction
    dots = normals @ n_unit
    flip_mask = dots < 0
    normals[flip_mask] *= -1
    dots = np.abs(dots)
    
    # 2) Filter points whose local normal is close to target normal
    normal_mask = dots >= cos_angle_thresh
    num_filtered = normal_mask.sum()
    # print(f"[v2] Points with normal close to target: {num_filtered} / {len(points_xyz)} ({100*num_filtered/len(points_xyz):.1f}%)")
    
    if num_filtered < min_points:
        print(f"[v2] Not enough points after normal filtering")
        return []
    
    filtered_pts = points_xyz[normal_mask]
    filtered_to_orig = np.where(normal_mask)[0]  # Map filtered index -> original index
    filtered_labels = point_labels[normal_mask] if point_labels is not None else None
    
    # 3) 1D clustering on filtered points (now safe - no side points!)
    s = filtered_pts @ n_unit
    order = np.argsort(s)
    s_sorted = s[order]
    pts_sorted = filtered_pts[order]
    sorted_labels = filtered_labels[order] if filtered_labels is not None else None
    
    gaps = np.diff(s_sorted)
    split_positions = np.where(gaps > distance_eps)[0]
    starts = np.r_[0, split_positions + 1]
    ends = np.r_[split_positions + 1, len(s_sorted)]
    
    # print(f"[v2] Found {len(starts)} 1D clusters")
    
    results = []
    
    for seg_i, (start, end) in enumerate(zip(starts, ends)):
        seg_idx_sorted = np.arange(start, end)
        seg = pts_sorted[seg_idx_sorted]
        
        if len(seg) < min_points:
            continue
        
        # Fit plane by PCA
        centroid = seg.mean(axis=0)
        centered = seg - centroid
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        n_fit = eigvecs[:, 0]
        
        if np.dot(n_fit, n_unit) < 0:
            n_fit = -n_fit
        
        # Check angle to target normal
        cos_ang = float(np.clip(np.dot(n_fit, n_unit), -1.0, 1.0))
        if cos_ang < cos_angle_thresh:
            continue
        
        d_fit = -float(np.dot(n_fit, centroid))
        
        # Tighten inliers by true point-to-plane distance
        dist_all = np.abs(seg @ n_fit + d_fit)
        inlier_mask_all = dist_all <= distance_eps
        if inlier_mask_all.sum() < min_points:
            continue
        
        # 4) Spatial clustering within the slab using grid-based BFS
        seg_inliers = seg[inlier_mask_all]
        seg_inliers_idx_sorted = seg_idx_sorted[inlier_mask_all]
        inlier_labels = sorted_labels[seg_inliers_idx_sorted] if sorted_labels is not None else None
        
        # Tangent basis from PCA
        t1 = eigvecs[:, 1]
        t2 = eigvecs[:, 2]
        u = seg_inliers @ t1
        v = seg_inliers @ t2
        
        iu = np.floor(u / spatial_voxel_m).astype(np.int64)
        iv = np.floor(v / spatial_voxel_m).astype(np.int64)
        
        bins = {}
        for j, key in enumerate(zip(iu, iv)):
            if key in bins:
                bins[key].append(j)
            else:
                bins[key] = [j]
        
        visited = set()
        
        def neighbors(k):
            a, b = k
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    if da == 0 and db == 0:
                        continue
                    yield (a + da, b + db)
        
        for start_key in list(bins.keys()):
            if start_key in visited:
                continue
            queue = deque([start_key])
            visited.add(start_key)
            comp_bins = []
            while queue:
                k = queue.popleft()
                comp_bins.append(k)
                for nk in neighbors(k):
                    if nk in bins and nk not in visited:
                        visited.add(nk)
                        queue.append(nk)
            
            # Collect points in this connected component
            comp_local_idx = []
            for bkey in comp_bins:
                comp_local_idx.extend(bins[bkey])
            if not comp_local_idx:
                continue
            comp_local_idx = np.array(comp_local_idx, dtype=np.int64)
            
            # Optionally split by AI labels before validation
            if inlier_labels is not None:
                sub_groups = _split_component_by_labels(comp_local_idx, inlier_labels, seg_inliers)
            else:
                sub_groups = [comp_local_idx]

            for sub_idx in sub_groups:
                comp_pts = seg_inliers[sub_idx]
                if len(comp_pts) < min_points:
                    continue
                
                # Refine plane on component
                comp_centroid = comp_pts.mean(axis=0)
                centered_c = comp_pts - comp_centroid
                cov_c = centered_c.T @ centered_c
                eigvals_c, eigvecs_c = np.linalg.eigh(cov_c)
                n_fit_c = eigvecs_c[:, 0]
                
                if np.dot(n_fit_c, n_unit) < 0:
                    n_fit_c = -n_fit_c
                
                cos_ang_c = float(np.clip(np.dot(n_fit_c, n_unit), -1.0, 1.0))
                if cos_ang_c < cos_angle_thresh:
                    continue
                
                d_fit_c = -float(np.dot(n_fit_c, comp_centroid))
                dist_c = np.abs(comp_pts @ n_fit_c + d_fit_c)
                inlier_mask_c = dist_c <= distance_eps
                if inlier_mask_c.sum() < min_points:
                    continue
                
                # Map back to original indices
                final_local_idx = sub_idx[inlier_mask_c]
                sorted_idx = order[seg_inliers_idx_sorted[final_local_idx]]
                orig_indices = filtered_to_orig[sorted_idx]
                
                results.append({
                    'n': n_fit_c.astype(np.float64),
                    'd': d_fit_c,
                    'indices': orig_indices,
                })
    
    print(f"[v2] Final subplanes found: {len(results)}")
    return results


def fit_rectangle_to_plane_points(points_3d: np.ndarray, plane_normal: np.ndarray):
    """
    Fit a minimum area bounding rectangle to 3D points lying on a plane.
    
    Parameters:
        points_3d: (N, 3) array of 3D points
        plane_normal: (3,) plane normal vector
        
    Returns:
        dict with:
            - 'center': (3,) center of the rectangle in 3D
            - 'corners': (4, 3) array of 4 corner points in 3D
            - 'width': width of the rectangle (meters)
            - 'height': height of the rectangle (meters)
            - 'angle': rotation angle in degrees
    """
    if len(points_3d) < 3:
        return None
    
    # Normalize plane normal
    n = plane_normal / np.linalg.norm(plane_normal)
    
    # Create orthonormal basis on the plane
    # Find a vector not parallel to n
    if abs(n[0]) < 0.9:
        v = np.array([1.0, 0.0, 0.0])
    else:
        v = np.array([0.0, 1.0, 0.0])
    
    # Gram-Schmidt to get tangent vectors
    t1 = v - np.dot(v, n) * n
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    t2 /= np.linalg.norm(t2)
    
    # Project 3D points to 2D plane coordinates
    centroid = points_3d.mean(axis=0)
    pts_centered = points_3d - centroid
    u = pts_centered @ t1  # x-coordinate in plane
    v = pts_centered @ t2  # y-coordinate in plane
    pts_2d = np.stack([u, v], axis=1).astype(np.float32)
    
    # Use OpenCV to find minimum area bounding rectangle
    rect = cv2.minAreaRect(pts_2d)
    box_2d = cv2.boxPoints(rect)  # 4 corners in 2D
    
    # Extract rectangle properties
    (cx_2d, cy_2d), (width, height), angle = rect
    
    # Transform 2D center back to 3D
    center_3d = centroid + cx_2d * t1 + cy_2d * t2
    
    # Transform 2D corners back to 3D
    corners_3d = np.zeros((4, 3), dtype=np.float64)
    for i, (uc, vc) in enumerate(box_2d):
        corners_3d[i] = centroid + uc * t1 + vc * t2
    
    return {
        'center': center_3d,
        'corners': corners_3d,
        'width': float(width),
        'height': float(height),
        'angle': float(angle),
    }


def detect_grasp_planes(
    pts_cam: np.ndarray,
    inlier_thresh: float = RANSAC_INLIER_THRESH,
    z_above_threshold: float = Z_ABOVE_THRESHOLD,
    distance_eps: float = DISTANCE_EPS,
    min_points: int = MIN_POINTS,
    angle_tol_deg: float = ANGLE_TOL_DEG,
    spatial_voxel_m: float = SPATIAL_VOXEL_M,
    normal_k: int = NORMAL_K,
    n_plane_base: tuple = DEFAULT_PLANE_NORMAL,
    d_plane_base: float = DEFAULT_PLANE_D,
    bbox_xy: tuple = BBOX_XY,
    bbox_pixels: tuple = None,
    bbox_margin_m: float = BBOX_MARGIN_M,
    bbox_margin_px: int = BBOX_MARGIN_PX,
    color_image=None,
    color_intrinsics=None,
    depth_to_color_extrinsic=None,
    T_base_camera_override: np.ndarray = None,
):
    """
    High-level routine to extract graspable subplanes above the dominant support plane.
    
    Use rs.pointcloud().calculate() to get pts_cam for proper lens distortion handling.

    Parameters:
      - pts_cam: (N, 3) point cloud in camera frame from rs.pointcloud().calculate()
      - n_plane_base: Dominant plane normal in base frame. 
                      Defaults to (0.0, 0.0, -1.0). Set to None to run RANSAC detection.
      - d_plane_base: Dominant plane offset in base frame. 
                      Defaults to -0.5695. Set to None to run RANSAC detection.
      - bbox_xy: Optional bounding box for filtering subplane centroids.
                 Format: ((x_min, x_max), (y_min, y_max)) in base frame coordinates.
                 If None, no filtering is applied.

    Output dict keys:
      - plane: {"n": [nx, ny, nz], "d": d}  (can be stored for future calls)
      - subplanes: list of {"center": [x, y, z], "n": [nx, ny, nz], "d": d, "num_points": int}
      - centers: list of [x, y, z] (duplicate of subplanes[i]["center"]) for convenience
      - pts_base: (N, 3) point cloud in base frame (for visualization)
      - above_mask: boolean mask of points above the plane
      - seq: int sequence (always 0 here; publishers can overwrite)
      - timestamp: float
    """
    t0 = time.time()
    # print(f"[detect_grasp_planes] Input point cloud: {pts_cam.shape[0]:,} points")

    # 1) Transform points to base frame first
    t1 = time.time()
    T_bc = T_base_camera_override if T_base_camera_override is not None else T_base_camera
    pts_h = np.hstack([pts_cam, np.ones((len(pts_cam), 1))])
    pts_base = (pts_h @ T_bc.T)[:, :3]
    t2 = time.time()
    # print(f"[detect_grasp_planes] Transform points to base frame: {(t2-t1)*1000:.1f} ms")

    # 2) Determine dominant plane in base frame
    if n_plane_base is None or d_plane_base is None:
        # Run RANSAC to detect dominant plane directly in base frame
        t1 = time.time()
        n_plane_base, d_plane_base, _ = fit_plane_ransac_from_depth(pts_base, max_iters=800, inlier_thresh=inlier_thresh)
        t2 = time.time()
        print(f"[detect_grasp_planes] Fit dominant plane (RANSAC): {(t2-t1)*1000:.1f} ms")
        n_plane_base = n_plane_base.astype(np.float64)
        print(f"✓ Dominant plane in base frame: ({n_plane_base[0]:.4f})x + ({n_plane_base[1]:.4f})y + ({n_plane_base[2]:.4f})z + ({d_plane_base:.4f}) = 0")
    else:
        # Use provided/default plane parameters (skip RANSAC)
        n_plane_base = np.array(n_plane_base, dtype=np.float64)
        d_plane_base = float(d_plane_base)
        # print(f"[detect_grasp_planes] Using provided dominant plane (RANSAC skipped)")
        # print(f"✓ Dominant plane in base frame: ({n_plane_base[0]:.4f})x + ({n_plane_base[1]:.4f})y + ({n_plane_base[2]:.4f})z + ({d_plane_base:.4f}) = 0")

    # 3) Keep points above the plane by z-height
    t1 = time.time()
    a, b, c = float(n_plane_base[0]), float(n_plane_base[1]), float(n_plane_base[2])
    d = d_plane_base
    dists = pts_base @ n_plane_base + d
    inlier_mask = np.abs(dists) < inlier_thresh
    if abs(c) > 1e-6:
        z_plane = (-d - a * pts_base[:, 0] - b * pts_base[:, 1]) / c
        height_above_plane_z = pts_base[:, 2] - z_plane
        above_mask = (height_above_plane_z >= z_above_threshold) & (~inlier_mask)
    else:
        above_mask = np.zeros_like(inlier_mask)
    pts_above = pts_base[above_mask]
    t2 = time.time()
    # print(f"[detect_grasp_planes] Filter above-plane points: {(t2-t1)*1000:.1f} ms ({pts_above.shape[0]} points)")

    # 4) AI-assisted labeling (optional)
    above_labels = None
    ai_bboxes = []
    above_pixel_coords = None  # (M, 2) projection of pts_above to color pixels, lazy
    if color_image is not None and color_intrinsics is not None and get_ai_model() is not None:
        t_ai = time.time()
        ai_bboxes = run_ai_detection(color_image)
        if ai_bboxes:
            pixel_coords = project_pts_to_color_pixels(pts_cam, color_intrinsics, depth_to_color_extrinsic)
            all_labels = compute_point_labels_from_bboxes(pixel_coords, ai_bboxes)
            above_labels = all_labels[above_mask]
            above_pixel_coords = pixel_coords[above_mask]
            n_labeled = int((above_labels >= 0).sum())
            print(f"[AI] {len(ai_bboxes)} bboxes, {n_labeled}/{len(above_labels)} above-plane points labeled ({time.time()-t_ai:.0f}ms)")
        else:
            print(f"[AI] No detections")

    # 4b) ROI pre-crop: shrink the input to find_parallel_subplanes_v2 to the
    # user-selected ROI plus a generous margin. The post-hoc centroid filter
    # (below, and in detection_server.py) is left in place as a correctness
    # backstop — the margin just guarantees that any object whose centroid
    # lies inside the ROI keeps its full footprint in the cropped cloud so
    # the detected subplane and its centroid are bit-identical.
    roi_to_above = None  # set only when we actually crop
    if pts_above.shape[0] > 0 and (bbox_xy is not None or bbox_pixels is not None):
        roi_mask = np.ones(pts_above.shape[0], dtype=bool)

        if bbox_xy is not None:
            (xmn, xmx), (ymn, ymx) = bbox_xy
            m = float(bbox_margin_m)
            roi_mask &= (
                (pts_above[:, 0] >= xmn - m) & (pts_above[:, 0] <= xmx + m) &
                (pts_above[:, 1] >= ymn - m) & (pts_above[:, 1] <= ymx + m)
            )

        if bbox_pixels is not None and color_intrinsics is not None:
            if above_pixel_coords is None:
                above_pixel_coords = project_pts_to_color_pixels(
                    pts_cam[above_mask], color_intrinsics, depth_to_color_extrinsic
                )
            x1p, y1p, x2p, y2p = bbox_pixels
            mp = float(bbox_margin_px)
            roi_mask &= (
                (above_pixel_coords[:, 0] >= x1p - mp) & (above_pixel_coords[:, 0] <= x2p + mp) &
                (above_pixel_coords[:, 1] >= y1p - mp) & (above_pixel_coords[:, 1] <= y2p + mp)
            )

        if not roi_mask.all():
            roi_to_above = np.flatnonzero(roi_mask)
            pts_above_roi = pts_above[roi_mask]
            above_labels_roi = above_labels[roi_mask] if above_labels is not None else None
        else:
            pts_above_roi = pts_above
            above_labels_roi = above_labels
    else:
        pts_above_roi = pts_above
        above_labels_roi = above_labels

    # 5) Find parallel subplanes among points above the plane (within ROI)
    t1 = time.time()
    if pts_above_roi.shape[0] == 0:
        subplanes = []
    else:
        subplanes = find_parallel_subplanes_v2(
            pts_above_roi,
            n_plane_base,
            distance_eps=distance_eps,
            min_points=min_points,
            angle_tol_deg=angle_tol_deg,
            spatial_voxel_m=spatial_voxel_m,
            normal_k=normal_k,
            point_labels=above_labels_roi,
        )
    t2 = time.time()
    # print(f"[detect_grasp_planes] Find parallel subplanes (v2): {(t2-t1)*1000:.1f} ms")

    # Remap subplane indices from pts_above_roi space back to pts_above space
    # so the rest of this function (which indexes pts_above) and downstream
    # visualisation (which uses indices_in_above) keep working unchanged.
    if roi_to_above is not None:
        for sp in subplanes:
            sp["indices"] = roi_to_above[sp["indices"]]

    # 6) Compute centers and fit rectangles for each detected subplane
    t1 = time.time()
    centers = []
    enriched_subplanes = []
    
    # Parse bbox if provided
    if bbox_xy is not None:
        (x_min, x_max), (y_min, y_max) = bbox_xy
        # print(f"[detect_grasp_planes] Filtering centroids to bbox: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}]")
    
    for sp in subplanes:
        idx = sp["indices"]
        comp_pts = pts_above[idx]
        if comp_pts.size == 0:
            continue
        center = comp_pts.mean(axis=0)
        
        # Filter by bounding box if specified
        if bbox_xy is not None:
            cx, cy, cz = center
            if not (x_min <= cx <= x_max and y_min <= cy <= y_max):
                continue  # Skip subplanes with centroid outside bbox
        
        centers.append(center.tolist())
        print('center', center)
        
        # Fit rectangle to the subplane points
        rect_info = fit_rectangle_to_plane_points(comp_pts, sp["n"])
        
        # Calculate Yaw from rectangle corners for robot alignment
        # IMPORTANT: cv2.boxPoints returns corners in order, but edge 0->1 could be
        # either the short or long side depending on rectangle orientation.
        # We must consistently pick the same edge (e.g., always the longer edge).
        yaw_rad = 0.0
        if rect_info and rect_info['corners'] is not None:
            c = rect_info['corners']
            
            # Compute both adjacent edge vectors
            edge01 = c[1] - c[0]  # Edge from corner 0 to corner 1
            edge12 = c[2] - c[1]  # Edge from corner 1 to corner 2
            
            # Compute lengths (in XY plane for yaw calculation)
            len01 = np.linalg.norm(edge01[:2])  # Only XY components matter for yaw
            len12 = np.linalg.norm(edge12[:2])
            
            # Always use the LONGER edge for consistent yaw
            # (For square boxes, they're equal so it doesn't matter)
            if len01 >= len12:
                edge_vec = edge01
            else:
                edge_vec = edge12
            
            # Calculate angle in the XY plane (base frame yaw)
            yaw_rad = float(np.arctan2(edge_vec[1], edge_vec[0]))
            print(f'yaw_rad={yaw_rad:.4f}, yaw_deg={np.degrees(yaw_rad):.1f}, edge_lens=({len01*1000:.1f}mm, {len12*1000:.1f}mm)')
        
        nz = float(n_plane_base[2])
        if abs(nz) > 1e-6:
            z_plane_at_center = (-d - a * center[0] - b * center[1]) / nz
            object_height = float(center[2] - z_plane_at_center)
        else:
            object_height = 0.0

        enriched_subplanes.append({
            "center": center.tolist(),
            "n": sp["n"].astype(float).tolist(),
            "d": float(sp["d"]),
            "num_points": int(len(idx)),
            "indices_in_above": idx.tolist(),  # Keep indices for visualization
            "yaw": yaw_rad,  # Robot grasp yaw in radians
            "object_height": object_height,  # meters, measured from support plane
            "rectangle": {
                "center": rect_info["center"].tolist() if rect_info else None,
                "corners": rect_info["corners"].tolist() if rect_info else None,
                "width": rect_info["width"] if rect_info else None,
                "height": rect_info["height"] if rect_info else None,
                "angle": rect_info["angle"] if rect_info else None,
            } if rect_info else None,
        })
    t2 = time.time()
    # print(f"[detect_grasp_planes] Compute subplane centers and rectangles: {(t2-t1)*1000:.1f} ms")

    result = {
        "plane": {"n": n_plane_base.astype(float).tolist(), "d": float(d_plane_base)},
        "subplanes": enriched_subplanes,
        "centers": centers,
        "seq": 0,
        "timestamp": time.time(),
        "processing_ms": int((time.time() - t0) * 1000),
        "above_mask": above_mask,  # Include mask for visualization
        "bbox_xy": bbox_xy,  # Include bbox for visualization
        "pts_base": pts_base,  # Include for visualization
    }

    print(f"[detect_grasp_planes] Total detection time: {(time.time() - t0) * 1000:.1f} ms")

    return result


    



def visualize_dominant_plane(vis, n_plane, d_plane, size=0.5, grid_div=10):
    """
    Visualize the dominant plane with a grid and normal arrow.
    
    Parameters:
        vis: meshcat visualizer
        n_plane: (3,) plane normal vector
        d_plane: plane offset (n·p + d = 0)
        size: size of the plane visualization in meters
        grid_div: number of grid divisions
    """
    import meshcat.geometry as g
    import meshcat.transformations as tf
    
    n = np.array(n_plane, dtype=np.float64)
    n = n / np.linalg.norm(n)
    
    # Find a point on the plane: p = -d * n (closest point to origin)
    plane_center = -d_plane * n
    
    # Create orthonormal basis on the plane
    if abs(n[0]) < 0.9:
        v = np.array([1.0, 0.0, 0.0])
    else:
        v = np.array([0.0, 1.0, 0.0])
    t1 = v - np.dot(v, n) * n
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    
    # Draw grid lines on the plane
    half = size / 2
    for i in range(grid_div + 1):
        offset = -half + i * (size / grid_div)
        
        # Lines parallel to t1
        p1 = plane_center + offset * t2 - half * t1
        p2 = plane_center + offset * t2 + half * t1
        vertices = np.array([p1, p2]).T.astype(np.float32)
        vis[f"dominant_plane/grid_t1_{i}"].set_object(
            g.Line(g.PointsGeometry(vertices), g.LineBasicMaterial(color=0x444444, linewidth=1))
        )
        
        # Lines parallel to t2
        p1 = plane_center + offset * t1 - half * t2
        p2 = plane_center + offset * t1 + half * t2
        vertices = np.array([p1, p2]).T.astype(np.float32)
        vis[f"dominant_plane/grid_t2_{i}"].set_object(
            g.Line(g.PointsGeometry(vertices), g.LineBasicMaterial(color=0x444444, linewidth=1))
        )
    
    # Draw normal arrow (shaft + cone)
    arrow_length = 0.15
    arrow_radius = 0.008
    cone_radius = 0.02
    cone_height = 0.04
    
    # Arrow shaft (cylinder)
    shaft_end = plane_center + (arrow_length - cone_height) * n
    shaft_center = (plane_center + shaft_end) / 2
    
    # Build rotation matrix to align Z-axis with normal
    z_axis = np.array([0, 0, 1])
    if np.allclose(n, z_axis):
        R = np.eye(3)
    elif np.allclose(n, -z_axis):
        R = np.diag([1, -1, -1])
    else:
        axis = np.cross(z_axis, n)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(z_axis, n), -1, 1))
        # Rodrigues formula
        K = np.array([[0, -axis[2], axis[1]], 
                      [axis[2], 0, -axis[0]], 
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    # Shaft transform
    T_shaft = np.eye(4)
    T_shaft[:3, :3] = R
    T_shaft[:3, 3] = shaft_center
    
    vis["dominant_plane/normal_shaft"].set_object(
        g.Cylinder(arrow_length - cone_height, arrow_radius),
        g.MeshBasicMaterial(color=0xFF0000)  # Red
    )
    vis["dominant_plane/normal_shaft"].set_transform(T_shaft)
    
    # Cone (arrow head)
    cone_center = plane_center + (arrow_length - cone_height / 2) * n
    T_cone = np.eye(4)
    T_cone[:3, :3] = R
    T_cone[:3, 3] = cone_center
    
    vis["dominant_plane/normal_cone"].set_object(
        g.Cylinder(cone_height, 0, cone_radius),  # Cone: height, radiusTop=0, radiusBottom
        g.MeshBasicMaterial(color=0xFF0000)
    )
    vis["dominant_plane/normal_cone"].set_transform(T_cone)
    
    # Draw a small sphere at plane center
    vis["dominant_plane/center"].set_object(
        g.Sphere(0.012),
        g.MeshBasicMaterial(color=0xFFFF00)  # Yellow
    )
    vis["dominant_plane/center"].set_transform(
        np.array([
            [1, 0, 0, plane_center[0]],
            [0, 1, 0, plane_center[1]],
            [0, 0, 1, plane_center[2]],
            [0, 0, 0, 1]
        ])
    )
    
    # Add text label (as a small marker)
    print(f"[Viz] Dominant plane: n=({n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}), d={d_plane:.4f}")
    # print(f"[Viz] Plane center (closest to origin): ({plane_center[0]:.3f}, {plane_center[1]:.3f}, {plane_center[2]:.3f})")


def visualize_bbox(vis, bbox_xy, z_height=0.1):
    """
    Visualize the 2D bounding box as a rectangular frame in 3D.
    
    Parameters:
        vis: meshcat visualizer
        bbox_xy: ((x_min, x_max), (y_min, y_max))
        z_height: height at which to draw the bbox (default 0.1m above origin)
    """
    import meshcat.geometry as g
    
    if bbox_xy is None:
        return
    
    (x_min, x_max), (y_min, y_max) = bbox_xy
    
    # Define 4 corners of the bbox
    corners = np.array([
        [x_min, y_min, z_height],
        [x_max, y_min, z_height],
        [x_max, y_max, z_height],
        [x_min, y_max, z_height],
    ], dtype=np.float32)
    
    # Draw 4 edges
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        vertices = np.array([p1, p2]).T.astype(np.float32)
        vis[f"bbox/edge_{i}"].set_object(
            g.Line(g.PointsGeometry(vertices), g.LineBasicMaterial(color=0x00FF00, linewidth=3))
        )
    
    # Draw corner markers
    for i, corner in enumerate(corners):
        vis[f"bbox/corner_{i}"].set_object(
            g.Sphere(0.01),
            g.MeshBasicMaterial(color=0x00FF00)
        )
        vis[f"bbox/corner_{i}"].set_transform(
            np.array([
                [1, 0, 0, corner[0]],
                [0, 1, 0, corner[1]],
                [0, 0, 1, corner[2]],
                [0, 0, 0, 1]
            ])
        )
    
    print(f"[Viz] BBox: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}] at z={z_height:.3f}")


def visualize_grasp_detection(vis, result, viz_stride=VIZ_STRIDE):
    """
    Visualize grasp detection results using pts_base from the result.
    
    Parameters:
        vis: meshcat visualizer
        result: output from detect_grasp_planes (must include 'pts_base')
        viz_stride: downsample stride for visualization (e.g., 4 means show 1/4 of points)
    """
    import meshcat.geometry as g
    
    # Clear all previous visualization groups for live updates
    vis["dominant_plane"].delete()
    vis["bbox"].delete()
    vis["markers"].delete()
    
    # Visualize the dominant plane first
    plane_info = result.get('plane', {})
    if plane_info:
        visualize_dominant_plane(
            vis, 
            np.array(plane_info['n']), 
            plane_info['d'],
            size=0.6,
            grid_div=12
        )
    
    # Visualize the bounding box if present
    bbox_xy = result.get('bbox_xy')
    if bbox_xy is not None:
        plane_d = plane_info.get('d', 0.05) if plane_info else 0.05
        visualize_bbox(vis, bbox_xy, z_height=-plane_d + 0.02)
    
    # Get point cloud and masks from result
    pts_base = result['pts_base']
    above_mask = np.array(result['above_mask'])
    N_full = len(pts_base)
    
    # Create colors array
    colors_full = np.ones((N_full, 3), dtype=np.float32) * 0.7  # Gray background
    
    # Color palette for subplanes
    palette = np.array([
        [1.0, 0.0, 0.0],  # red
        [0.0, 0.0, 1.0],  # blue
        [1.0, 1.0, 0.0],  # yellow
        [1.0, 0.0, 1.0],  # magenta
        [0.0, 1.0, 1.0],  # cyan
        [1.0, 0.5, 0.0],  # orange
        [0.5, 0.0, 1.0],  # purple
        [0.0, 0.5, 1.0],  # sky blue
    ], dtype=np.float32)
    
    # Apply colors to subplane points
    global_above_idx = np.flatnonzero(above_mask)
    for k, sp in enumerate(result['subplanes']):
        local_idx = np.array(sp['indices_in_above'])
        if len(local_idx) > 0:
            global_idx = global_above_idx[local_idx]
            valid_idx = global_idx[global_idx < N_full]
            colors_full[valid_idx] = palette[k % len(palette)]
    
    # Downsample for visualization
    pts_viz = pts_base[::viz_stride]
    colors_viz = colors_full[::viz_stride]
    
    # Update point cloud
    vis["point_cloud"].set_object(
        g.PointCloud(
            position=pts_viz.T.astype(np.float32),
            color=colors_viz.T.astype(np.float32),
            size=0.002,
        )
    )
    
    # Draw rectangle centers and corners for each subplane
    for i, sp in enumerate(result['subplanes']):
        rect = sp.get('rectangle')
        if rect is None or rect['corners'] is None:
            continue
        
        # Get color for this subplane (convert to hex)
        color_rgb = palette[i % len(palette)]
        color_hex = int(color_rgb[0] * 255) << 16 | int(color_rgb[1] * 255) << 8 | int(color_rgb[2] * 255)
        
        # Draw rectangle center as a larger sphere (green)
        rect_center = rect['center']
        vis[f"markers/rect_center_{i}"].set_object(
            g.Sphere(0.008),
            g.MeshBasicMaterial(color=0x00FF00)  # Green for center
        )
        vis[f"markers/rect_center_{i}"].set_transform(
            np.array([
                [1, 0, 0, rect_center[0]],
                [0, 1, 0, rect_center[1]],
                [0, 0, 1, rect_center[2]],
                [0, 0, 0, 1]
            ])
        )
        
        # Draw 4 corners as smaller spheres
        corners = rect['corners']
        for j, corner in enumerate(corners):
            vis[f"markers/corner_{i}_{j}"].set_object(
                g.Sphere(0.005),
                g.MeshBasicMaterial(color=color_hex)
            )
            vis[f"markers/corner_{i}_{j}"].set_transform(
                np.array([
                    [1, 0, 0, corner[0]],
                    [0, 1, 0, corner[1]],
                    [0, 0, 1, corner[2]],
                    [0, 0, 0, 1]
                ])
            )
        
        # Draw rectangle edges as lines
        for j in range(4):
            p1 = np.array(corners[j])
            p2 = np.array(corners[(j + 1) % 4])
            # Create line geometry using LineSegments
            vertices = np.array([p1, p2]).T.astype(np.float32)  # Shape: (3, 2)
            vis[f"markers/edge_{i}_{j}"].set_object(
                g.Line(g.PointsGeometry(vertices), g.LineBasicMaterial(color=color_hex, linewidth=2))
            )


if __name__ == "__main__":
    import argparse
    import pyrealsense2 as rs
    import meshcat
    import meshcat.geometry as g

    parser = argparse.ArgumentParser(description="Run grasp plane detection on RealSense camera")
    parser.add_argument("--ai", action="store_true", help="Enable AI-assisted object splitting (RF-DETR)")
    parser.add_argument(
        "--flange", action="store_true",
        help="Eye-in-hand mode: load T_flange_camera from flange_camera_transform.npz "
             "and compute T_base_camera live from the robot's current flange pose."
    )
    parser.add_argument(
        "--flange-file", default="flange_camera_transform.npz",
        help="Path to the eye-in-hand calibration .npz (default: flange_camera_transform.npz)"
    )
    parser.add_argument(
        "--robot-ip", default="192.168.58.2",
        help="Fairino robot IP (default: 192.168.58.2)"
    )
    args = parser.parse_args()

    if args.ai:
        print("AI mode enabled via --ai flag")
        get_ai_model(force=True)

    # ------------------------------------------------------------------
    # Eye-in-hand mode: compute live T_base_camera = T_base_flange @ T_flange_camera
    # ------------------------------------------------------------------
    T_base_camera_runtime = None  # None -> detect_grasp_planes uses the module constant
    if args.flange:
        import sys as _sys
        from scipy.spatial.transform import Rotation as _R

        _SDK = "/home/jeong/fr5sdk/linux"
        if _SDK not in _sys.path:
            _sys.path.insert(0, _SDK)
        from fairino import Robot as _Robot

        print(f"[--flange] Loading eye-in-hand calibration from {args.flange_file}")
        _cal = np.load(args.flange_file, allow_pickle=True)
        T_flange_camera = _cal["T_flange_camera"]
        _euler_conv = str(_cal["euler_convention"]) if "euler_convention" in _cal.files else "xyz"
        print(f"[--flange] T_flange_camera (m):\n{T_flange_camera}")
        print(f"[--flange] Euler convention: '{_euler_conv}'")

        print(f"[--flange] Connecting to robot at {args.robot_ip} ...")
        _robot = _Robot.RPC(args.robot_ip)
        time.sleep(1.0)
        _err, _pose = _robot.GetActualToolFlangePose(flag=1)
        if _err != 0 or _pose is None:
            print(f"[--flange] ERROR: cannot read flange pose (err={_err}). Aborting.")
            exit(1)
        print(f"[--flange] Flange pose (mm, deg): "
              f"x={_pose[0]:.2f} y={_pose[1]:.2f} z={_pose[2]:.2f}  "
              f"rx={_pose[3]:.2f} ry={_pose[4]:.2f} rz={_pose[5]:.2f}")

        _T_bf = np.eye(4)
        _T_bf[:3, 3] = np.asarray(_pose[:3], dtype=np.float64) / 1000.0
        _T_bf[:3, :3] = _R.from_euler(_euler_conv, _pose[3:], degrees=True).as_matrix()
        T_base_camera_runtime = _T_bf @ T_flange_camera
        print(f"[--flange] T_base_flange (m):\n{_T_bf}")
        print(f"[--flange] T_base_camera (live):\n{T_base_camera_runtime}")

        try:
            _robot.CloseRPC()
        except Exception:
            pass

    # ============================================================================
    # Setup RealSense D455 Camera (from realsense_pc.py)
    # ============================================================================
    print("Initializing RealSense camera...")
    
    def get_device():
        ctx = rs.context()
        devs = ctx.query_devices()
        if len(devs) == 0:
            raise RuntimeError("No RealSense devices found")
        return devs[0]

    pipe = None
    profile = None
    depth_sensor = None
    fx = fy = cx = cy = 0.0
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS)
    cfg.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)

    # Initialization loop with retry mechanism
    while True:
        try:
            pipe = rs.pipeline()
            profile = pipe.start(cfg)
            
            # Get camera intrinsics
            depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intrinsics = depth_profile.get_intrinsics()
            fx, fy = intrinsics.fx, intrinsics.fy
            cx, cy = intrinsics.ppx, intrinsics.ppy
            print(f"Depth intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

            color_intr = color_profile.get_intrinsics()
            _color_intrinsics = {'fx': color_intr.fx, 'fy': color_intr.fy,
                                 'cx': color_intr.ppx, 'cy': color_intr.ppy}
            print(f"Color intrinsics: fx={color_intr.fx:.2f}, fy={color_intr.fy:.2f}, cx={color_intr.ppx:.2f}, cy={color_intr.ppy:.2f}")

            d2c_ext = depth_profile.get_extrinsics_to(color_profile)
            _depth_to_color_ext = {
                'R': np.array(d2c_ext.rotation).reshape(3, 3),
                't': np.array(d2c_ext.translation),
            }
            print(f"Depth-to-color translation: {_depth_to_color_ext['t']}")
            
            # Configure depth sensor for high accuracy
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_accuracy)
            depth_sensor.set_option(rs.option.depth_units, DEPTH_UNITS)
            
            # Exposure settings for depth/IR sensor
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1)  # Auto-exposure ON
            
            # Laser power (higher = better depth in low light / far objects).
            # Not every device supports this option (e.g. some D405 SKUs), so
            # be tolerant of unsupported / out-of-range failures.
            if depth_sensor.supports(rs.option.laser_power):
                try:
                    lp_range = depth_sensor.get_option_range(rs.option.laser_power)
                    lp_value = float(min(max(LASER_POWER, lp_range.min), lp_range.max))
                    depth_sensor.set_option(rs.option.laser_power, lp_value)
                    print(f"Laser power: {lp_value:.0f} (range {lp_range.min:.0f}-{lp_range.max:.0f})")
                except Exception as e:
                    print(f"[warn] could not set laser_power: {e}")
            else:
                print("[info] depth sensor does not support laser_power option")
            
            # Configure color sensor exposure
            for sensor in profile.get_device().sensors:
                if sensor.get_info(rs.camera_info.name) == 'RGB Camera':
                    sensor.set_option(rs.option.enable_auto_exposure, 1)  # Auto-exposure ON
                    # For manual control:
                    # sensor.set_option(rs.option.enable_auto_exposure, 0)
                    # sensor.set_option(rs.option.exposure, 166)
                    # sensor.set_option(rs.option.gain, 64)
                    print(f"Color sensor auto-exposure enabled")
                    break
            
            print("Camera initialized successfully.")
            break
            
        except RuntimeError as e:
            msg = str(e)
            print(f"Initialization error: {msg}")
            
            # Errors that can be recovered with a hardware reset
            recoverable_errors = [
                "get_xu(ctrl=1)",
                "Device or resource busy",
                "failed to set power state",
                "Couldn't resolve requests",  # Camera in use or bad state
            ]
            
            if any(err in msg for err in recoverable_errors):
                print("Camera busy or in bad state, doing hardware reset...")
                
                # Try to stop pipe if it was partially started
                if pipe:
                    try:
                        pipe.stop()
                    except:
                        pass
                
                try:
                    dev = get_device()
                    dev.hardware_reset()
                    print("Reset triggered, waiting 2s for re-enumeration...")
                    time.sleep(2)
                except Exception as reset_err:
                    print(f"Failed to reset device: {reset_err}")
                    time.sleep(2)
            else:
                # Fatal error (e.g. no device connected at all)
                print(f"Fatal error initializing camera: {e}")
                exit(1)
    
    # Setup filters using centralized configuration
    decimation = spatial = temporal = hole_filling = None
    depth_to_disparity = disparity_to_depth = None
    
    if FILTER_DECIMATION_ENABLED:
        decimation = rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, FILTER_DECIMATION_MAGNITUDE)
    if FILTER_SPATIAL_ENABLED:
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_smooth_alpha, FILTER_SPATIAL_ALPHA)
        spatial.set_option(rs.option.filter_smooth_delta, FILTER_SPATIAL_DELTA)
    if FILTER_TEMPORAL_ENABLED:
        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, FILTER_TEMPORAL_ALPHA)
        temporal.set_option(rs.option.filter_smooth_delta, FILTER_TEMPORAL_DELTA)
    if FILTER_HOLE_FILLING_ENABLED:
        hole_filling = rs.hole_filling_filter()
    if FILTER_DISPARITY_ENABLED:
        depth_to_disparity = rs.disparity_transform(True)
        disparity_to_depth = rs.disparity_transform(False)
    
    # RealSense pointcloud object (handles lens distortion properly!)
    pc = rs.pointcloud()
    
    # Warm-up camera
    print("Warming up camera...")
    for _ in range(30):
        pipe.wait_for_frames()
    
    # Capture depth + color frames
    print("Capturing frames...")
    frames = pipe.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame:
        print("ERROR: Failed to capture depth frame")
        pipe.stop()
        exit(1)
    color_image = np.asanyarray(color_frame.get_data()) if color_frame else None
    
    # Apply filter chain (only enabled filters from configuration)
    depth = depth_frame
    if decimation:
        depth = decimation.process(depth)
    if depth_to_disparity:
        depth = depth_to_disparity.process(depth)
    if spatial:
        depth = spatial.process(depth)
    if temporal:
        depth = temporal.process(depth)
    if disparity_to_depth:
        depth = disparity_to_depth.process(depth)
    if hole_filling:
        depth = hole_filling.process(depth)
    
    # Use RealSense SDK to compute point cloud (properly handles lens distortion!)
    print("Computing point cloud with RealSense SDK (distortion-corrected)...")
    points = pc.calculate(depth)
    pts_cam = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    
    # Filter valid points (remove zeros)
    valid_mask = (pts_cam[:, 2] > 0) & np.isfinite(pts_cam[:, 2])
    pts_cam = pts_cam[valid_mask]
    print(f"Point cloud: {pts_cam.shape[0]:,} valid points")
    print(f"  X range: [{pts_cam[:, 0].min():.3f}, {pts_cam[:, 0].max():.3f}] m")
    print(f"  Y range: [{pts_cam[:, 1].min():.3f}, {pts_cam[:, 1].max():.3f}] m")
    print(f"  Z range: [{pts_cam[:, 2].min():.3f}, {pts_cam[:, 2].max():.3f}] m")
    
    pipe.stop()
    
    # ============================================================================
    # Run detect_grasp_planes (uses centralized configuration defaults)
    # ============================================================================
    print("\n" + "="*70)
    print("Running detect_grasp_planes...")
    print("="*70)
    detect_kwargs = {}
    if args.ai and color_image is not None:
        detect_kwargs['color_image'] = color_image
        detect_kwargs['color_intrinsics'] = _color_intrinsics
        detect_kwargs['depth_to_color_extrinsic'] = _depth_to_color_ext
    if T_base_camera_runtime is not None:
        detect_kwargs['T_base_camera_override'] = T_base_camera_runtime

    result = detect_grasp_planes(
        pts_cam,
        **detect_kwargs,
    )
    
    print("\n" + "="*70)
    print(f"RESULTS: Found {len(result['subplanes'])} graspable subplanes")
    print("="*70)
    for i, sp in enumerate(result['subplanes']):
        center = sp['center']
        plane = sp['n']
        d = sp['d']
        rect = sp.get('rectangle')
        
        print(f"\n--- Subplane {i+1} ---")
        print(f"  Points: {sp['num_points']}")
        print(f"  Plane: n=({plane[0]:.4f}, {plane[1]:.4f}, {plane[2]:.4f}), d={d:.4f}")
        print(f"  Centroid: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
        
        if rect and rect['corners']:
            corners = rect['corners']
            rect_center = rect['center']
            print(f"  Rectangle Center: ({rect_center[0]:.4f}, {rect_center[1]:.4f}, {rect_center[2]:.4f})")
            print(f"  Rectangle Size: width={rect['width']*1000:.1f}mm, height={rect['height']*1000:.1f}mm, angle={rect['angle']:.1f}°")
            print(f"  Grasp Yaw: {sp.get('yaw', 0.0):.4f} rad ({np.degrees(sp.get('yaw', 0.0)):.1f}°)")
            print(f"  Rectangle Corners:")
            for j, corner in enumerate(corners):
                print(f"    Corner {j+1}: ({corner[0]:.4f}, {corner[1]:.4f}, {corner[2]:.4f})")
        else:
            print(f"  Rectangle: Could not fit")
    
    # ============================================================================
    # Visualization
    # ============================================================================
    print("\nGenerating visualization...")
    
    # Launch Meshcat
    vis = meshcat.Visualizer()
    vis.open()
    vis.delete()
    
    visualize_grasp_detection(vis, result)
    
    print(f"\nMeshcat visualization ready at: {vis.url()}")
    print("Press Ctrl+C to exit...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")
        pass


#problems
#same height different orientation but together 
#same height same orientation


# ideas
# parallel subplanes 
# change voxel clustering method 
# new clustering method 


