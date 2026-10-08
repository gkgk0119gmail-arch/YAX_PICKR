import time
import cv2
import numpy as np
import pyrealsense2 as rs

#### PARAMETERS
# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ChArUco board settings
ARUCO_DICT = cv2.aruco.DICT_4X4_100
BOARD_SQUARES_X = 7
BOARD_SQUARES_Y = 5
BOARD_SQUARE_LENGTH = 0.04  # meters
BOARD_MARKER_LENGTH = 0.03  # meters

# Calibration settings
CALIBRATION_DURATION_SECONDS = 2.0  # ~300 frames at 30 FPS
MIN_CORNERS_FOR_POSE = 4  # cv2.solvePnP requires at least 4 points by default



#### INPUTS FROM MEASUREMENTS
corners_touched = np.array([

# [1,1],
# [2,1],
# [3,1],
# [4,1],
# [4,2],
# [3,2],
# [2,2],
# [1,2],

[1,1],
[6,1],
[6,4],
[1,4]


])
# Gripper tool offset: the gripper tip is 13cm along the flange Z-axis
# T_flange_tip transforms from flange frame to gripper tip frame
T_offset_correction = np.eye(4)
T_offset_correction[0, 3] = 0.00
T_offset_correction[1, 3] = 0.00
T_offset_correction[2, 3] = 0.13  # 13cm gripper length along Z

# Set to None to capture new T_cb, or provide a preset transformation matrix
T_cb = None 

# Uncomment the line below to force a new capture:
# T_cb = None

# List of TCP poses - add or remove matrices as needed
# Each matrix should be a 4x4 transformation matrix representing the TCP pose when touching a corner
TCP_POSES = [
    np.array([  # [1,1]
        [ 0.750620, -0.660584,  0.014082, -0.076902],
        [-0.660566, -0.750739, -0.006521,  0.740062],
        [ 0.014880, -0.004407, -0.999880,  0.249858],
        [ 0.000000,  0.000000,  0.000000,  1.000000]
    ]),
    np.array([  # [6,1]
        [ 0.757209, -0.653027,  0.013762,  0.128495],
        [-0.653041, -0.757312, -0.004105,  0.741921],
        [ 0.013103, -0.005879, -0.999897,  0.251077],
        [ 0.000000,  0.000000,  0.000000,  1.000000]
    ]),
    np.array([  # [6,4]
        [ 0.758031, -0.652085,  0.013219,  0.128080],
        [-0.652073, -0.758134, -0.005773,  0.624544],
        [ 0.013786, -0.004244, -0.999896,  0.249242],
        [ 0.000000,  0.000000,  0.000000,  1.000000]
    ]),
    np.array([  # [1,4]
        [ 0.750921, -0.660243,  0.014050, -0.077334],
        [-0.660242, -0.751034, -0.005373,  0.619339],
        [ 0.014099, -0.005242, -0.999887,  0.249393],
        [ 0.000000,  0.000000,  0.000000,  1.000000]
    ]),
]
# Print the difference between offset corrected vs non-offset corrected tcp_poses
print("\nDifference between offset-corrected and original TCP poses (Frobenius norm for each pose):")
for i, (orig_pose, offset_pose) in enumerate(zip(TCP_POSES, [pose @ T_offset_correction for pose in TCP_POSES])):
    diff = offset_pose - orig_pose
    frob_norm = np.linalg.norm(diff)
    print(f"TCP Pose {i+1}: Frobenius norm of diff = {frob_norm:.8f}")

# For more detail, optionally print both matrices per pose
print("\nDetailed difference (per-element) for each pose:")
for i, (orig_pose, offset_pose) in enumerate(zip(TCP_POSES, [pose @ T_offset_correction for pose in TCP_POSES])):
    print(f"TCP Pose {i+1} original:\n{orig_pose}")
    print(f"TCP Pose {i+1} offset-corrected:\n{offset_pose}")
    print(f"Difference:\n{offset_pose - orig_pose}\n")


# Apply offset correction to all TCP poses
TCP_POSES = [pose @ T_offset_correction for pose in TCP_POSES]




# Extract translation vectors from all TCP poses
tcp_translations = np.array([pose[0:3, 3] for pose in TCP_POSES])

# Validate that the number of TCP poses matches the number of corners
assert len(TCP_POSES) == len(corners_touched), \
    f"Mismatch: {len(TCP_POSES)} TCP poses but {len(corners_touched)} corners. They must be equal."

# Compute and print distances between consecutive TCP positions
print(f"\nDistances between consecutive TCP positions:")
for i in range(len(tcp_translations)):
    next_i = (i + 1) % len(tcp_translations)
    dist = np.linalg.norm(tcp_translations[next_i] - tcp_translations[i])
    print(f"  Distance t{i+1} -> t{next_i+1}: {dist:.6f} meters")



def get_camera_intrinsics(profile):
    """Extract camera matrix K and distortion coefficients D from RealSense profile."""
    color_stream = profile.get_stream(rs.stream.color)
    video_profile = color_stream.as_video_stream_profile()
    intr = video_profile.get_intrinsics()
    
    K = np.array([
        [intr.fx, 0.0, intr.ppx],
        [0.0, intr.fy, intr.ppy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    D = np.array(intr.coeffs[:5], dtype=np.float32).reshape(1, 5)
    
    return K, D


def create_charuco_board():
    """Create ChArUco board."""
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (BOARD_SQUARES_X, BOARD_SQUARES_Y),
        BOARD_SQUARE_LENGTH,
        BOARD_MARKER_LENGTH,
        dictionary
    )
    return board


def corner_indices_to_board_frame(corner_indices):
    """
    Convert chessboard corner indices to 3D coordinates in the board frame.
    
    Args:
        corner_indices: Nx2 array of corner indices [x_index, y_index]
        
    Returns:
        Nx3 array of 3D points in board frame (meters)
        
    The board frame origin is at the top-left corner of the board.
    X-axis points right, Y-axis points down, Z-axis points out of the board.
    Each corner is at position (x_index * SQUARE_LENGTH, y_index * SQUARE_LENGTH, 0).
    """
    corner_indices = np.asarray(corner_indices)
    num_corners = corner_indices.shape[0]
    
    points_board_frame = np.zeros((num_corners, 3), dtype=np.float32)
    points_board_frame[:, 0] = corner_indices[:, 0] * BOARD_SQUARE_LENGTH  # X
    points_board_frame[:, 1] = corner_indices[:, 1] * BOARD_SQUARE_LENGTH  # Y
    points_board_frame[:, 2] = 0.0  # Z (on the board surface)
    
    return points_board_frame


def solve_transformation_svd(points_source, points_target):
    """
    Solve for rigid transformation from source frame to target frame using SVD.
    Uses Procrustes/Kabsch/Umeyama algorithm.
    
    Args:
        points_source: Nx3 array of points in source frame
        points_target: Nx3 array of corresponding points in target frame
        
    Returns:
        T: 4x4 transformation matrix such that points_target = T @ points_source
    
    This solves: points_target = R @ points_source + t
    """
    points_source = np.asarray(points_source, dtype=np.float64)
    points_target = np.asarray(points_target, dtype=np.float64)
    
    # Compute centroids
    centroid_source = np.mean(points_source, axis=0)
    centroid_target = np.mean(points_target, axis=0)
    
    # Center the points
    points_source_centered = points_source - centroid_source
    points_target_centered = points_target - centroid_target
    
    # Compute cross-covariance matrix
    H = points_source_centered.T @ points_target_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = Vt.T @ U.T
    
    # Handle reflection case (ensure proper rotation, det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = centroid_target - R @ centroid_source
    
    # Build 4x4 transformation matrix
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T


def create_charuco_detector(board, K, D):
    """Create ChArUco detector."""
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    
    charuco_params = cv2.aruco.CharucoParameters()
    charuco_params.cameraMatrix = K
    charuco_params.distCoeffs = D
    
    detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)
    return detector


def estimate_board_pose(charuco_corners, charuco_ids, board, K, D):
    """Estimate board pose from detected ChArUco corners."""
    charuco_corners = np.ascontiguousarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2)
    charuco_ids = np.ascontiguousarray(charuco_ids, dtype=np.int32).reshape(-1, 1)
    
    # Get 3D object points for detected corners
    obj_points = board.getChessboardCorners()
    obj_points = obj_points[charuco_ids.reshape(-1)]
    
    # Solve PnP to get board pose
    success, rvec, tvec = cv2.solvePnP(obj_points, charuco_corners, K, D)
    
    return success, rvec, tvec


def rvec_tvec_to_transformation_matrix(rvec, tvec):
    """Convert rotation vector and translation vector to 4x4 transformation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(-1)
    return T


def filter_by_density(data, radius, label="data"):
    """
    Filter data by finding the densest cluster (mode) using a fixed radius.
    Conceptually similar to finding the core of a DBSCAN cluster.
    
    Args:
        data: Nx3 array of vectors
        radius: Search radius for neighbors (Euclidean distance)
        label: Name for logging
        
    Returns:
        inlier_mask: Boolean mask of points belonging to the densest cluster
    """
    data = np.asarray(data)
    n = len(data)
    if n < 2:
        return np.ones(n, dtype=bool)
        
    # Compute pairwise squared Euclidean distances
    # (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff = data[:, np.newaxis, :] - data[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    radius_sq = radius**2
    
    # Count neighbors for each point
    # (N, N) boolean matrix -> sum rows -> (N,) counts
    neighbor_counts = np.sum(dist_sq <= radius_sq, axis=1)
    
    # Find the point with the most neighbors (the mode center)
    best_idx = np.argmax(neighbor_counts)
    max_neighbors = neighbor_counts[best_idx]
    
    # Select all points that are within the radius of this mode center
    # This effectively selects the "dominant cluster"
    inlier_mask = dist_sq[best_idx] <= radius_sq
    
    n_outliers = n - np.sum(inlier_mask)
    if n_outliers > 0:
        print(f"  {label}: Kept {max_neighbors}/{n} samples in densest cluster (radius={radius})")
        print(f"  {label}: Removed {n_outliers} outliers")
    
    return inlier_mask


def average_rotations_svd(rotation_matrices):
    """
    Average multiple rotation matrices using SVD (Procrustes/Karcher mean approximation).
    
    This computes the mean rotation by:
    1. Summing all rotation matrices
    2. Using SVD to project the sum back onto SO(3)
    
    Args:
        rotation_matrices: List of 3x3 rotation matrices
        
    Returns:
        R_avg: 3x3 averaged rotation matrix
    """
    rotation_matrices = np.asarray(rotation_matrices)
    n = len(rotation_matrices)
    
    # Sum all rotation matrices
    R_sum = np.sum(rotation_matrices, axis=0)
    
    # SVD to project back onto SO(3)
    U, S, Vt = np.linalg.svd(R_sum)
    
    # Compute the closest rotation matrix
    R_avg = U @ Vt
    
    # Ensure proper rotation (det = +1, not -1)
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt
    
    return R_avg


def average_poses_proper(rvecs, tvecs, rvec_radius=0.1, tvec_radius=0.05):
    """
    Properly average camera poses with density-based outlier removal and SO(3) rotation averaging.
    
    Args:
        rvecs: Nx3 array of rotation vectors
        tvecs: Nx3 array of translation vectors
        rvec_radius: Clustering radius for rotation (radians), default 0.1 (~5.7 deg)
        tvec_radius: Clustering radius for translation (meters), default 0.05 (5 cm)
        
    Returns:
        rvec_avg: Averaged rotation vector
        tvec_avg: Averaged translation vector
        stats: Dictionary with statistics
    """
    rvecs = np.asarray(rvecs)
    tvecs = np.asarray(tvecs)
    n_original = len(rvecs)
    
    print(f"\n--- Outlier Detection (Density Clustering) ---")
    # Detect outliers in rotation and translation independently using density
    rvec_inliers = filter_by_density(rvecs, radius=rvec_radius, label="Rotation")
    tvec_inliers = filter_by_density(tvecs, radius=tvec_radius, label="Translation")
    
    # Intersection of inliers (must be valid in BOTH rotation and translation)
    common_inliers = rvec_inliers & tvec_inliers
    n_clean = np.sum(common_inliers)
    n_outliers = n_original - n_clean
    
    if n_clean < 3:
        print(f"WARNING: Too few inliers ({n_clean}/{n_original}), using all data.")
        common_inliers = np.ones(n_original, dtype=bool)
        n_clean = n_original
        n_outliers = 0
        
    rvecs_clean = rvecs[common_inliers]
    tvecs_clean = tvecs[common_inliers]
    
    print(f"Pose Averaging: {n_clean}/{n_original} samples used ({n_outliers} outliers removed)")
    
    # Print samples (restored as requested)
    print(f"\n--- Valid Pose Samples ({n_clean}) ---")
    rotation_matrices = []
    for i in range(n_clean):
        rvec = rvecs_clean[i]
        tvec = tvecs_clean[i]
        
        R, _ = cv2.Rodrigues(rvec)
        rotation_matrices.append(R)
        
        # Only print first 5 and last 5 if too many
            # Build 4x4 transformation matrix for display
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec
        
        print(f"  Sample {i+1}:")
        print(f"    rvec: [{rvec[0]:.4f}, {rvec[1]:.4f}, {rvec[2]:.4f}]")
        print(f"    tvec: [{tvec[0]:.4f}, {tvec[1]:.4f}, {tvec[2]:.4f}]")
        # Flatten print for T to save vertical space
        print(f"    T_cb: [[{T[0,0]:.4f}, {T[0,1]:.4f}, {T[0,2]:.4f}, {T[0,3]:.4f}], ...]")
        
    if n_clean > 20:
        print(f"  ... (printed 10/{n_clean} samples)")

    # Average rotations
    R_avg = average_rotations_svd(rotation_matrices)
    rvec_avg, _ = cv2.Rodrigues(R_avg)
    rvec_avg = rvec_avg.reshape(-1)
    
    # Average translations
    tvec_avg = np.mean(tvecs_clean, axis=0)
    
    stats = {
        'n_original': n_original,
        'n_clean': n_clean,
        'n_outliers': n_outliers,
        'rvec_std': np.std(rvecs_clean, axis=0),
        'tvec_std': np.std(tvecs_clean, axis=0),
    }
    
    return rvec_avg, tvec_avg, stats


def capture_camera_to_board_transform():
    """Capture T_cb by detecting ChArUco board with RealSense camera."""
    # Start RealSense pipeline
    print("Starting RealSense pipeline...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)
    
    profile = pipeline.start(config)
    
    # Warm up camera
    print("Warming up camera for 2 seconds...")
    warmup_start = time.time()
    while time.time() - warmup_start < 2.0:
        pipeline.wait_for_frames()
    
    # Get camera intrinsics
    K, D = get_camera_intrinsics(profile)
    print(f"Camera matrix K:\n{K}")
    print(f"Distortion coefficients D: {D.reshape(-1)}")
    
    # Create ChArUco board and detector
    board = create_charuco_board()
    detector = create_charuco_detector(board, K, D)
    
    # Collect poses for specified duration
    print(f"\nCollecting board poses for {CALIBRATION_DURATION_SECONDS} seconds...")
    print("Please keep the board visible and stable.")
    
    rvecs_list = []
    tvecs_list = []
    start_time = time.time()
    frame_count = 0
    detection_count = 0
    
    while time.time() - start_time < CALIBRATION_DURATION_SECONDS:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        image_bgr = np.asanyarray(color_frame.get_data())
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect ChArUco board
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(image_gray)
        
        frame_count += 1
        
        # Estimate pose if enough corners detected
        if charuco_ids is not None and len(charuco_ids) >= MIN_CORNERS_FOR_POSE:
            success, rvec, tvec = estimate_board_pose(charuco_corners, charuco_ids, board, K, D)
            
            if success:
                rvecs_list.append(rvec.reshape(-1))
                tvecs_list.append(tvec.reshape(-1))
                detection_count += 1
                print(f"  Frame {frame_count}: Detected {len(charuco_ids)} corners - rvec={rvec.reshape(-1)}, tvec={tvec.reshape(-1)}")
    
    pipeline.stop()
    
    # Compute average pose
    print(f"\nCollected {detection_count} valid poses from {frame_count} frames")
    
    if detection_count == 0:
        print("ERROR: No valid board detections! Make sure the board is visible.")
        return None, None, None
    
    rvecs_array = np.array(rvecs_list)
    tvecs_array = np.array(tvecs_list)
    
    # Proper averaging with outlier removal and SO(3) rotation averaging
    # Radii: 0.1 rad (~5.7 deg) for rotation, 0.05 m (5 cm) for translation
    rvec_avg, tvec_avg, stats = average_poses_proper(rvecs_array, tvecs_array, rvec_radius=0.1, tvec_radius=0.05)
    
    print(f"\n--- Final Averaged Pose ---")
    print(f"  Average rvec: {rvec_avg}")
    print(f"  Average tvec: {tvec_avg}")
    print(f"  (Used {stats['n_clean']}/{stats['n_original']} samples after removing {stats['n_outliers']} outliers)")
    
    # Convert to transformation matrix T_cb (camera to board)
    T_cb_captured = rvec_tvec_to_transformation_matrix(rvec_avg, tvec_avg)
    
    print(f"\nTransformation matrix T_cb (camera to board):")
    print(T_cb_captured)
    
    return T_cb_captured, K, D


def main():
    global T_cb
    
    # Check if T_cb needs to be captured
    if T_cb is None:
        print("=" * 60)
        print("T_cb is not set. Capturing camera-to-board transformation...")
        print("=" * 60)
        T_cb_result, K, D = capture_camera_to_board_transform()
        
        if T_cb_result is None:
            print("ERROR: Failed to capture T_cb. Exiting.")
            return
        
        T_cb = T_cb_result
        
        # Save to file
        output_file = "camera_to_board_transform.npz"
        np.savez(output_file, T_cb=T_cb, K=K, D=D)
        print(f"\nSaved transformation to {output_file}")
        print(f"  T_cb: 4x4 transformation matrix")
        print(f"  K: camera matrix")
        print(f"  D: distortion coefficients")
    else:
        print("=" * 60)
        print("Using preset T_cb (camera-to-board transformation):")
        print("=" * 60)
        print(T_cb)
        print()
    
    # Now proceed with robot-to-board calibration
    print("\n" + "=" * 60)
    print("Computing robot-to-board transformation (T_rb)...")
    print("=" * 60)
    
    # Convert user-defined corners to board frame coordinates
    print(f"\nCorner coordinates in board frame:")
    points_board = corner_indices_to_board_frame(corners_touched)
    for i, (corner_idx, point_3d) in enumerate(zip(corners_touched, points_board)):
        print(f"  Corner {corner_idx} -> {point_3d} meters")
    
    # Extract TCP positions in robot frame (using all TCP poses)
    points_robot = tcp_translations
    print(f"\nTCP positions in robot frame ({len(points_robot)} points):")
    for i, tcp_pos in enumerate(points_robot):
        print(f"  TCP {i+1}: {tcp_pos}")
    
    # Solve for T_rb (robot base to board) using SVD
    print(f"\nSolving for T_rb (robot base to board frame)...")
    T_rb = solve_transformation_svd(points_board, points_robot)
    
    print(f"\nTransformation matrix T_rb (robot base to board):")
    print(T_rb)
    
    # Verify the transformation by computing residuals
    print(f"\nVerification - transformed points vs actual TCP positions:")
    points_board_homogeneous = np.hstack([points_board, np.ones((len(points_board), 1))])
    points_transformed = (T_rb @ points_board_homogeneous.T).T[:, :3]
    
    residuals = points_robot - points_transformed
    residual_norms = np.linalg.norm(residuals, axis=1)
    
    for i in range(len(points_board)):
        print(f"  Point {i+1}: actual={points_robot[i]}, transformed={points_transformed[i]}, error={residual_norms[i]:.6f} m")
    
    print(f"\nMean residual error: {np.mean(residual_norms):.6f} m")
    print(f"Max residual error: {np.max(residual_norms):.6f} m")
    
    # Compute T_rc (robot base to camera) and T_cr (camera to robot base)
    print(f"\n" + "=" * 60)
    print("Computing final transformations T_rc and T_cr...")
    print("=" * 60)
    
    # T_bc: board to camera (inverse of T_cb)
    T_bc = np.linalg.inv(T_cb)
    
    # T_rc: robot base to camera = T_rb @ T_bc
    # (board to robot) composed with (board to camera)^-1
    T_rc = T_rb @ T_bc
    
    # T_cr: camera to robot base (inverse of T_rc)
    T_cr = np.linalg.inv(T_rc)
    
    print(f"\nTransformation matrix T_rc (robot base to camera):")
    print(T_rc)
    
    print(f"\nTransformation matrix T_cr (camera to robot base):")
    print(T_cr)
    
    # Verify full transformation chain consistency
    print(f"\n" + "=" * 60)
    print("Verifying full transformation chain consistency...")
    print("=" * 60)
    
    # Test 1: Verify that board corners can be transformed to robot frame via both paths
    print("\nTest 1: Board corners -> Robot frame (via direct T_rb vs via T_rc @ T_cb)")
    print("  Path A: T_rb @ point_board (direct)")
    print("  Path B: T_rc @ T_cb @ point_board (via camera)")
    
    chain_errors = []
    for i, point_board in enumerate(points_board):
        point_board_hom = np.append(point_board, 1.0)
        
        # Path A: Direct transformation from board to robot
        point_robot_direct = (T_rb @ point_board_hom)[:3]
        
        # Path B: Board -> Camera -> Robot
        point_camera = (T_cb @ point_board_hom)[:3]
        point_robot_via_camera = (T_rc @ np.append(point_camera, 1.0))[:3]
        
        chain_error = np.linalg.norm(point_robot_direct - point_robot_via_camera)
        chain_errors.append(chain_error)
        
        print(f"  Corner {i+1}:")
        print(f"    Direct:      {point_robot_direct}")
        print(f"    Via camera:  {point_robot_via_camera}")
        print(f"    Error:       {chain_error:.6f} m")
    
    print(f"\n  Mean chain error: {np.mean(chain_errors):.6f} m")
    print(f"  Max chain error:  {np.max(chain_errors):.6f} m")
    
    # Test 2: Verify transformation chain algebraically
    print("\nTest 2: Algebraic verification - T_rc should equal T_rb @ T_bc")
    T_rc_computed = T_rb @ T_bc
    T_rc_diff = np.linalg.norm(T_rc - T_rc_computed, 'fro')
    print(f"  Frobenius norm of (T_rc - T_rb @ T_bc): {T_rc_diff:.2e}")
    if T_rc_diff < 1e-10:
        print(f"  ✓ Transformation chain is algebraically consistent")
    else:
        print(f"  ✗ WARNING: Transformation chain has numerical inconsistency")
    
    # Test 3: Verify test points at various locations on the board
    print("\nTest 3: Testing additional points on the board")
    test_points_board = np.array([
        [0.0, 0.0, 0.0],  # Top-left corner
        [BOARD_SQUARES_X * BOARD_SQUARE_LENGTH, 0.0, 0.0],  # Top-right
        [0.0, BOARD_SQUARES_Y * BOARD_SQUARE_LENGTH, 0.0],  # Bottom-left
        [BOARD_SQUARES_X * BOARD_SQUARE_LENGTH, BOARD_SQUARES_Y * BOARD_SQUARE_LENGTH, 0.0],  # Bottom-right
        [BOARD_SQUARES_X * BOARD_SQUARE_LENGTH / 2, BOARD_SQUARES_Y * BOARD_SQUARE_LENGTH / 2, 0.0],  # Center
    ])
    
    test_labels = ["Top-left", "Top-right", "Bottom-left", "Bottom-right", "Center"]
    
    for label, test_point in zip(test_labels, test_points_board):
        test_point_hom = np.append(test_point, 1.0)
        
        # Transform via both paths
        point_robot_direct = (T_rb @ test_point_hom)[:3]
        point_robot_via_camera = (T_rc @ T_cb @ test_point_hom)[:3]
        
        error = np.linalg.norm(point_robot_direct - point_robot_via_camera)
        print(f"  {label:12s}: Direct={point_robot_direct}, Via camera={point_robot_via_camera}, Error={error:.6f} m")
    
    # Test 4: Verify inverse transformations
    print("\nTest 4: Verify inverse transformations")
    T_rc_inv_computed = np.linalg.inv(T_rc)
    T_cr_diff = np.linalg.norm(T_cr - T_rc_inv_computed, 'fro')
    print(f"  Frobenius norm of (T_cr - inv(T_rc)): {T_cr_diff:.2e}")
    if T_cr_diff < 1e-10:
        print(f"  ✓ Inverse transformations are consistent")
    else:
        print(f"  ✗ WARNING: Inverse transformation has numerical inconsistency")
    
    # Test 5: Round-trip transformation test
    print("\nTest 5: Round-trip transformation (Board -> Camera -> Robot -> Board)")
    test_point_board = points_board[0]  # Use first calibration point
    test_point_board_hom = np.append(test_point_board, 1.0)
    
    # Forward: Board -> Camera -> Robot -> Board
    point_camera = T_cb @ test_point_board_hom
    point_robot = T_rc @ point_camera
    point_board_roundtrip = np.linalg.inv(T_rb) @ point_robot
    
    roundtrip_error = np.linalg.norm(test_point_board - point_board_roundtrip[:3])
    print(f"  Original point:  {test_point_board}")
    print(f"  Round-trip:      {point_board_roundtrip[:3]}")
    print(f"  Error:           {roundtrip_error:.6f} m")
    
    if roundtrip_error < 1e-6:
        print(f"  ✓ Round-trip transformation is accurate")
    else:
        print(f"  ⚠ Round-trip error: {roundtrip_error:.6f} m")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    if np.max(chain_errors) < 0.001 and T_rc_diff < 1e-10 and roundtrip_error < 1e-6:
        print("✓ All verification tests passed!")
        print("  The transformation chain is consistent and accurate.")
    else:
        print("⚠ Some verification tests show discrepancies:")
        if np.max(chain_errors) >= 0.001:
            print(f"  - Chain error: {np.max(chain_errors):.6f} m (should be < 0.001 m)")
        if T_rc_diff >= 1e-10:
            print(f"  - Algebraic inconsistency: {T_rc_diff:.2e}")
        if roundtrip_error >= 1e-6:
            print(f"  - Round-trip error: {roundtrip_error:.6f} m")
        print("  This may indicate calibration errors or numerical issues.")
    
    # Save all transformations to file
    output_file = "robot_camera_calibration.npz"
    np.savez(
        output_file,
        T_cb=T_cb,
        T_rb=T_rb,
        T_rc=T_rc,
        T_cr=T_cr
    )
    print(f"\n" + "=" * 60)
    print(f"Saved all transformations to {output_file}:")
    print("=" * 60)
    print(f"  T_cb: camera to board")
    print(f"  T_rb: robot base to board")
    print(f"  T_rc: robot base to camera")
    print(f"  T_cr: camera to robot base")
    
    # Calculate and print ChArUco corners in robot frame (similar to charuco_corners.py)
    print(f"\n" + "=" * 60)
    print("CHARUCO CORNERS IN ROBOT FRAME (Calculated from T_rb)")
    print("=" * 60)
    
    # Get 3D object points for all corners (in board frame)
    # Create a temporary board object to get the corners
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board_obj = cv2.aruco.CharucoBoard(
        (BOARD_SQUARES_X, BOARD_SQUARES_Y),
        BOARD_SQUARE_LENGTH,
        BOARD_MARKER_LENGTH,
        dictionary
    )
    
    # all_corners_board is (N, 3) in board frame (z=0)
    all_corners_board = board_obj.getChessboardCorners()
    
    # Transform to robot frame: T_rb @ point_board
    # Add 1 for homogeneous coords
    all_corners_board_hom = np.hstack([all_corners_board, np.ones((len(all_corners_board), 1))])
    all_corners_robot = (T_rb @ all_corners_board_hom.T).T[:, :3]
    
    # Organize into dictionary for printing by grid coordinates
    # Grid: x=1..6, y=1..4 (for 7x5 squares)
    num_x_internal = BOARD_SQUARES_X - 1
    
    robot_corner_dict = {}
    for idx in range(len(all_corners_robot)):
        # Map index to 1-based (x, y) coordinates
        x = (idx % num_x_internal) + 1
        y = (idx // num_x_internal) + 1
        robot_corner_dict[(x, y)] = all_corners_robot[idx]
    
    # Print line-by-line
    print(f"Grid: x=1-{BOARD_SQUARES_X-1}, y=1-{BOARD_SQUARES_Y-1}")
    print("Format: [x,y] = (X, Y, Z) in meters")
    print()
    
    for y in range(1, BOARD_SQUARES_Y):
        for x in range(1, BOARD_SQUARES_X):
            pos = robot_corner_dict[(x, y)]
            print(f"[{x},{y}] = ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
    
    


if __name__ == "__main__":
    main()

# Transformation matrix T_rc (robot base to camera):
# [[ 0.02309381 -0.99956719  0.01822618  0.12027968]
#  [ 0.0214865   0.01872308  0.99959382 -0.08323177]
#  [-0.99950239 -0.02269281  0.02190958 -0.05417377]
#  [ 0.          0.          0.          1.        ]]

# Transformation matrix T_cr (camera to robot base):
# [[ 0.02309381  0.0214865  -0.99950237 -0.05513617]
#  [-0.99956711  0.01872308 -0.02269281  0.12055661]
#  [ 0.01822618  0.99959379  0.02190958  0.08219265]
#  [ 0.          0.          0.          1.        ]]