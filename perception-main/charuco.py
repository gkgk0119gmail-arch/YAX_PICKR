import sys
import time
from collections import deque

import cv2
import numpy as np
import pyrealsense2 as rs
try:
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
    MESHCAT_AVAILABLE = True
except Exception:
    MESHCAT_AVAILABLE = False


# Defaults (same as the larger script)
WIDTH = 640
HEIGHT = 480
FPS = 30
DICT = cv2.aruco.DICT_4X4_100
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LENGTH = 0.04
MARKER_LENGTH = 0.03
AXIS_LENGTH = SQUARE_LENGTH * 2.0
DISPLAY = True
USE_MESHCAT = False
PCD_STRIDE = 4
MAX_Z_METERS = 5.0


def build_camera_matrix_from_realsense(profile: rs.pipeline_profile):
    color_stream = profile.get_stream(rs.stream.color)
    video_profile = color_stream.as_video_stream_profile()
    intr = video_profile.get_intrinsics()
    width, height = intr.width, intr.height
    K = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]], dtype=np.float32)
    if intr.model in (rs.distortion.brown_conrady, rs.distortion.inverse_brown_conrady):
        coeffs = list(intr.coeffs[:5])
        while len(coeffs) < 5:
            coeffs.append(0.0)
        D = np.array(coeffs, dtype=np.float32).reshape(1, 5)
    else:
        D = np.zeros((1, 5), dtype=np.float32)
    return K, D, (width, height)


def get_depth_scale(profile: rs.pipeline_profile) -> float:
    try:
        for s in profile.get_device().sensors:
            if s.is_depth_sensor():
                return s.as_depth_sensor().get_depth_scale()
    except Exception:
        pass
    # Default RealSense depth scale if unknown
    return 0.001


def create_charuco_board():
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    aruco = cv2.aruco
    if hasattr(aruco, "CharucoBoard_create"):
        board = aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    elif hasattr(aruco, "CharucoBoard") and hasattr(aruco.CharucoBoard, "create"):
        board = aruco.CharucoBoard.create(SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    else:
        board = aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    return board


def build_charuco_detector(board, K, D):
    det_params = cv2.aruco.DetectorParameters()
    det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(cv2.aruco, "CharucoParameters"):
        charuco_params = cv2.aruco.CharucoParameters()
        charuco_params.cameraMatrix = K
        charuco_params.distCoeffs = D
        detector = cv2.aruco.CharucoDetector(board, charuco_params, det_params)
    else:
        detector = cv2.aruco.CharucoDetector(board, None, det_params)
    return detector


def _object_points_to_camera_points(obj_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    obj_points = np.asarray(obj_points, dtype=np.float32).reshape(-1, 3)
    cam_points = (obj_points @ R.T) + tvec.reshape(1, 3)
    return cam_points


def _update_meshcat_point_cloud(vis, K: np.ndarray, depth_m: np.ndarray, color_bgr: np.ndarray, mask: np.ndarray, stride: int = 4):
    if vis is None:
        return
    if depth_m is None or color_bgr is None:
        return
    h, w = depth_m.shape
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    ys = np.arange(0, h, stride, dtype=np.int32)
    xs = np.arange(0, w, stride, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = depth_m[grid_y, grid_x]
    valid = (z > 0) & np.isfinite(z) & (z < MAX_Z_METERS)
    if mask is not None and mask.shape[:2] == depth_m.shape:
        mask_s = mask[grid_y, grid_x].astype(bool)
    else:
        mask_s = np.zeros_like(valid, dtype=bool)

    u = grid_x.astype(np.float32)
    v = grid_y.astype(np.float32)
    X = (u - cx) / fx * z
    Y = (v - cy) / fy * z
    Z = z

    pts = np.stack([X[valid], Y[valid], Z[valid]], axis=0)
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    cols = rgb[grid_y, grid_x]
    cols = (cols[valid].astype(np.float32) / 255.0).T  # 3xN

    vis["point_cloud"].set_object(
        g.PointsGeometry(pts, cols),
        g.PointsMaterial(size=3.0, opacity=0.9)
    )

    masked_valid = valid & mask_s
    if np.any(masked_valid):
        pts_m = np.stack([X[masked_valid], Y[masked_valid], Z[masked_valid]], axis=0)
        vis["point_cloud_masked"].set_object(
            g.PointsGeometry(pts_m),
            g.PointsMaterial(size=5.0, color=0xFF0000, opacity=1.0)
        )
    else:
        # Clear masked geometry if no masked points this frame
        try:
            vis["point_cloud_masked"].delete()
        except Exception:
            pass


def average_rotations_svd(rotation_matrices):
    """
    Average multiple rotation matrices using SVD.
    """
    rotation_matrices = np.asarray(rotation_matrices)
    
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


def compute_average_pose(rvecs_buffer, tvecs_buffer):
    if not rvecs_buffer:
        return None, None
        
    # Average translation (simple mean)
    tvec_avg = np.mean(tvecs_buffer, axis=0)
    
    # Average rotation (SVD method)
    rotation_matrices = []
    for rvec in rvecs_buffer:
        R, _ = cv2.Rodrigues(np.array(rvec))
        rotation_matrices.append(R)
        
    R_avg = average_rotations_svd(rotation_matrices)
    rvec_avg, _ = cv2.Rodrigues(R_avg)
    
    return rvec_avg.reshape(-1), tvec_avg


def main() -> int:
    global DISPLAY
    print("Starting RealSense pipeline…")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    try:
        profile = pipeline.start(config)
    except Exception as exc:
        print(f"Failed to start RealSense pipeline: {exc}")
        return 1

    # Warmup frames
    for _ in range(5):
        pipeline.wait_for_frames()

    K, D, _ = build_camera_matrix_from_realsense(profile)
    depth_scale = get_depth_scale(profile)
    align = rs.align(rs.stream.color)
    meshcat_vis = None
    if USE_MESHCAT and MESHCAT_AVAILABLE:
        try:
            meshcat_vis = meshcat.Visualizer().open()
            print("MeshCat URL:", meshcat_vis.url())
        except Exception as exc:
            print(f"MeshCat unavailable: {exc}")
            meshcat_vis = None
    
    # Use a single predefined dictionary (no 4X4_100 auto-switching)
    board = create_charuco_board()
    charuco_detector = build_charuco_detector(board, K, D)
    
    # Pose buffer for averaging
    POSE_BUFFER_SIZE = 100
    rvecs_buffer = deque(maxlen=POSE_BUFFER_SIZE)
    tvecs_buffer = deque(maxlen=POSE_BUFFER_SIZE)
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame:
                continue
            image_bgr = np.asanyarray(color_frame.get_data())
            depth_image = None
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) * float(depth_scale)
            image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("data/image.png", image_bgr)

            # Detect ChArUco directly
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(image_gray)

            if DISPLAY:
                vis = image_bgr.copy()
                if marker_corners is not None and marker_ids is not None and len(marker_corners) > 0:
                    cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)

            ok = False
            rvec = None
            tvec = None
            if charuco_ids is not None and len(charuco_ids) >= 3:
                # Normalize shapes/dtypes for estimatePoseCharucoBoard
                try:
                    cc = np.ascontiguousarray(charuco_corners, dtype=np.float32)
                    if cc.ndim == 2 and cc.shape[1] == 2:
                        cc = cc.reshape(-1, 1, 2)
                    elif cc.ndim == 3 and cc.shape[1] == 1 and cc.shape[2] == 2:
                        pass
                    else:
                        cc = cc.reshape(-1, 1, 2)
                    ci = np.ascontiguousarray(charuco_ids, dtype=np.int32).reshape(-1, 1)

                    # estimate board pose from ChArUco
                    try:
                        # Try the modern CharucoDetector method first (OpenCV 4.7+)
                        if hasattr(board, 'getChessboardCorners'):
                            obj_points = board.getChessboardCorners()
                            obj_points = obj_points[ci.reshape(-1)]
                            ok, rvec, tvec = cv2.solvePnP(obj_points, cc, K, D)
                        else:
                            # Fallback for older versions
                            retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(cc, ci, board, K, D, None, None)
                            ok = bool(retval)
                    except AttributeError:
                        # Try without return value for very old versions
                        try:
                            rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(cc, ci, board, K, D)
                            ok = rvec is not None and tvec is not None
                        except Exception as e2:
                            # print(f"charuco pose estimation failed: {e2}")
                            ok = False
                except Exception as exc:
                    print(f"charuco pose error: {exc}")

            if ok and rvec is not None and tvec is not None:
                if DISPLAY:
                    try:
                        cv2.drawFrameAxes(vis, K, D, rvec, tvec, AXIS_LENGTH)
                    except Exception:
                        pass
                    cv2.putText(vis, "OK", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Add current pose to buffer
                rv = rvec.reshape(-1)
                tv = tvec.reshape(-1)
                rvecs_buffer.append(rv)
                tvecs_buffer.append(tv)
                
                # Calculate and print average pose
                avg_rvec, avg_tvec = compute_average_pose(rvecs_buffer, tvecs_buffer)
                
                if avg_rvec is not None:
                    print(f"Sample {len(rvecs_buffer)}/{POSE_BUFFER_SIZE} | Avg Pose: rvec={avg_rvec.tolist()} tvec={avg_tvec.tolist()}")
            else:
                if DISPLAY:
                    cv2.putText(vis, "no pose", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            # Build a default mask from detected markers (if any) to highlight in MeshCat
            mask = None
            try:
                if marker_corners is not None and len(marker_corners) > 0:
                    mask = np.zeros((image_bgr.shape[0], image_bgr.shape[1]), dtype=np.uint8)
                    for j in range(len(marker_corners)):
                        pts = np.asarray(marker_corners[j], dtype=np.float32).reshape(-1, 2)
                        cv2.fillConvexPoly(mask, pts.astype(np.int32), 255)
            except Exception:
                mask = None

            # Update MeshCat point cloud visualization (entire frame + highlighted mask)
            try:
                if depth_image is not None:
                    _update_meshcat_point_cloud(meshcat_vis, K, depth_image, image_bgr, mask, stride=PCD_STRIDE)
            except Exception as exc:
                # print(f"meshcat update error: {exc}")
                pass

            if DISPLAY:
                # Optionally overlay ChArUco corners
                try:
                    if charuco_corners is not None and charuco_ids is not None:
                        cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
                except Exception:
                    pass
                try:
                    cv2.imshow("ChArUco Pose (CharucoDetector)", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        break
                except cv2.error as e:
                    # OpenCV not built with GUI support - disable display mode
                    if "is not implemented" in str(e):
                        print("Warning: OpenCV not built with GUI support. Disabling display mode.")
                        print("To fix: sudo apt-get install libgtk2.0-dev pkg-config && pip install --force-reinstall opencv-python")
                        DISPLAY = False
                    else:
                        raise

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        if DISPLAY:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
