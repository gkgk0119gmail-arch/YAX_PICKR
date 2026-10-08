import time
import threading
import json
from collections import deque
import numpy as np
import pyrealsense2 as rs
import meshcat
from http.server import HTTPServer, BaseHTTPRequestHandler
import cluster
from cluster import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, DEPTH_UNITS, LASER_POWER,
    FILTER_DECIMATION_ENABLED, FILTER_DISPARITY_ENABLED, FILTER_SPATIAL_ENABLED,
    FILTER_TEMPORAL_ENABLED, FILTER_HOLE_FILLING_ENABLED,
    FILTER_DECIMATION_MAGNITUDE, FILTER_SPATIAL_ALPHA, FILTER_SPATIAL_DELTA,
    FILTER_TEMPORAL_ALPHA, FILTER_TEMPORAL_DELTA,
)
from config import BOX_YAW_DEG
from catalog import OBJECT_CATALOG, BOX_CATALOG, DEFAULT_BOX_KEY
from box_packing import pack_with_padding, _box_to_robot

RESULT_BUFFER = deque(maxlen=10)
RESULT_LOCK = threading.Lock()
SERVER_PORT = 5000

MATCH_TOLERANCE_CM = 3.0
# Max distance (metres) between subplane centers to be considered the same object
TEMPORAL_MATCH_RADIUS_M = 0.03


def classify_object(rect_width_m: float, rect_height_m: float) -> dict | None:
    """
    Match a detected rectangle (meters) against OBJECT_CATALOG entries (cm).

    The camera sees the top face, so rect (width, height) corresponds to
    catalog (width, length) — but either orientation may match.

    Returns the best-matching catalog entry as a dict with key, name, width,
    length, height (all in cm), or None if nothing is within tolerance.
    """
    rw_cm = rect_width_m * 100.0
    rh_cm = rect_height_m * 100.0
    detected = sorted([rw_cm, rh_cm])

    best_key = None
    best_err = float("inf")

    for key, item in OBJECT_CATALOG.items():
        catalog = sorted([item.width, item.length])
        err = abs(detected[0] - catalog[0]) + abs(detected[1] - catalog[1])
        if err < best_err:
            best_err = err
            best_key = key

    if best_err > MATCH_TOLERANCE_CM * 2:
        return None

    matched = OBJECT_CATALOG[best_key]
    return {
        "type": best_key,
        "name": matched.name,
        "width_cm": matched.width,
        "length_cm": matched.length,
        "height_cm": matched.height,
    }


def _smooth_subplanes(buffer: deque) -> list[dict]:
    """
    Match subplanes across buffered frames by center proximity and return
    temporally-smoothed subplanes using the median of rectangle dimensions,
    the mean of center positions, and the circular mean of yaw.

    Uses the most recent frame as the anchor: each anchor subplane is matched
    to the nearest subplane in every older frame within TEMPORAL_MATCH_RADIUS_M.

    Matching uses the Hungarian algorithm (optimal assignment) instead of greedy
    nearest-neighbor to avoid cross-contaminating tracks when objects are close.
    """
    from scipy.optimize import linear_sum_assignment

    if not buffer:
        return []

    anchor_result = buffer[-1]
    anchor_sps = anchor_result.get("subplanes", [])
    if not anchor_sps:
        return []

    anchor_centers = np.array([sp["center"] for sp in anchor_sps])

    tracks: list[list[dict]] = [[] for _ in anchor_sps]

    for result in buffer:
        sps = result.get("subplanes", [])
        if not sps:
            continue
        frame_centers = np.array([sp["center"] for sp in sps])

        # Build cost matrix: (num_anchor, num_frame_sps)
        cost = np.linalg.norm(
            anchor_centers[:, None, :] - frame_centers[None, :, :], axis=2
        )

        row_ind, col_ind = linear_sum_assignment(cost)
        for ai, fi in zip(row_ind, col_ind):
            if cost[ai, fi] <= TEMPORAL_MATCH_RADIUS_M:
                tracks[ai].append(sps[fi])

    smoothed = []
    for ai, observations in enumerate(tracks):
        if not observations:
            smoothed.append(anchor_sps[ai])
            continue

        centers = np.array([o["center"] for o in observations])
        yaws = np.array([o.get("yaw", 0.0) for o in observations])

        rect_widths = []
        rect_heights = []
        obj_heights = []
        for o in observations:
            r = o.get("rectangle")
            if r and r.get("width") is not None and r.get("height") is not None:
                rect_widths.append(r["width"])
                rect_heights.append(r["height"])
            oh = o.get("object_height")
            if oh is not None:
                obj_heights.append(oh)

        sp = dict(anchor_sps[ai])
        sp["center"] = centers.mean(axis=0).tolist()
        sp["yaw"] = float(np.arctan2(np.mean(np.sin(yaws)), np.mean(np.cos(yaws))))
        if rect_widths:
            sp["rectangle"] = dict(sp.get("rectangle") or {})
            sp["rectangle"]["width"] = float(np.median(rect_widths))
            sp["rectangle"]["height"] = float(np.median(rect_heights))
        if obj_heights:
            sp["object_height"] = float(np.median(obj_heights))

        smoothed.append(sp)

    return smoothed




def solve_packing(matched_targets: list[tuple[int, str]]) -> dict:
    """
    Run the bin-packing solver for matched detected objects.

    Uses the same padding-optimised packing as box_packing.solve() so that
    center_top positions are identical.

    Parameters
    ----------
    matched_targets : list of (target_index, item_key) pairs
        target_index is the index in the response targets list;
        item_key is the OBJECT_CATALOG key (e.g. "small").

    Returns a dict with box info, success status, and per-item placements
    including center_top in robot frame (metres).
    """
    box = BOX_CATALOG[DEFAULT_BOX_KEY]

    sorted_matches = sorted(
        matched_targets,
        key=lambda x: OBJECT_CATALOG[x[1]].volume,
        reverse=True,
    )
    items = [OBJECT_CATALOG[item_key] for _, item_key in sorted_matches]

    final_placed, best_padding = pack_with_padding(box, items)

    # Map placed items back to target indices via item name.
    # Items with the same name are interchangeable (same dimensions).
    name_to_targets: dict[str, list[tuple[int, str]]] = {}
    for target_idx, item_key in sorted_matches:
        name = OBJECT_CATALOG[item_key].name
        name_to_targets.setdefault(name, []).append((target_idx, item_key))

    placements = []
    for placed in final_placed:
        entries = name_to_targets.get(placed.name, [])
        if not entries:
            continue
        target_idx, item_key = entries.pop(0)
        item = OBJECT_CATALOG[item_key]

        cx, cy, cz = placed.center_top
        print(f"------placed.center_top={placed.center_top}")

        rx, ry, rz = _box_to_robot(cx, cy, cz)
        print('------box_to_robot', rx, ry, rz)

        max_catalog_dim = max(item.width, item.length)
        longest_edge_along_x = abs(placed.w - max_catalog_dim) < 0.1
        target_yaw_deg = (0.0 if longest_edge_along_x else 90.0) + BOX_YAW_DEG

        placements.append({
            "placement_order": len(placements) + 1,
            "target_index": target_idx,
            "type": item_key,
            "name": placed.name,
            "object_height_cm": item.height,
            "target_yaw_deg": target_yaw_deg,
            "placed_dims_cm": {"w": placed.w, "l": placed.l, "h": placed.h},
            "box_position_cm": {"x": placed.x, "y": placed.y, "z": placed.z},
            "center_top_box_cm": list(placed.center_top),
            "center_top_robot_m": [rx, ry, rz],
        })

    return {
        "box": DEFAULT_BOX_KEY,
        "box_dims_cm": {"w": box.width, "l": box.length, "h": box.height},
        "placed_count": len(placements),
        "total_count": len(matched_targets),
        "placements": placements,
    }


def get_device():
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        raise RuntimeError("No RealSense devices found")
    return devs[0]

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class GraspRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/detect':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            with RESULT_LOCK:
                if RESULT_BUFFER:
                    subplanes = _smooth_subplanes(RESULT_BUFFER)
                    targets = []
                    matched_targets = []
                    for i, sp in enumerate(subplanes):
                        target = {
                            "center": sp.get("center"),
                            "yaw": sp.get("yaw", 0.0),
                            "object_height_m": sp.get("object_height"),
                        }
                        rect = sp.get("rectangle")
                        if rect:
                            rw = rect.get("width")
                            rh = rect.get("height")
                            target["rectangle"] = {
                                "width": rw,
                                "height": rh,
                            }
                            if rw is not None and rh is not None:
                                match = classify_object(rw, rh)
                                if match:
                                    target["object"] = match
                                    matched_targets.append((i, match["type"]))
                        targets.append(target)

                    response_data = {"targets": targets}
                    if len(matched_targets) >= 2:
                        response_data["packing"] = solve_packing(matched_targets)
                else:
                    response_data = {"targets": []}
            
            try:
                json_str = json.dumps(response_data, cls=NumpyEncoder)
                self.wfile.write(json_str.encode())
            except Exception as e:
                print(f"Error serializing response: {e}")
                self.wfile.write(b'{"error": "serialization_failed"}')
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "not_found"}')

    def log_message(self, format, *args):
        # Suppress default logging to keep console clean
        pass

def run_detection_loop():
    
    # Initialize Meshcat visualizer once (persists across camera restarts)
    vis = None
    try:
        vis = meshcat.Visualizer()
        vis.open()
        print(f"Live visualization available at: {vis.url()}")
        vis.delete()
    except Exception as e:
        print(f"Failed to initialize Meshcat: {e}")
    
    # Outer loop to handle camera reconnection
    while True:
        pipe = None
        try:
            pipe = _init_camera_pipeline()
            if pipe is None:
                print("Failed to initialize camera, retrying in 2s...")
                time.sleep(2)
                continue
            
            _run_detection_with_pipeline(pipe, vis)
            
        except Exception as e:
            print(f"Detection loop crashed: {e}")
        finally:
            if pipe:
                try:
                    pipe.stop()
                except:
                    pass
        
        # Brief pause before attempting to reconnect
        print("Attempting to reconnect camera in 1s...")
        time.sleep(1)


def _init_camera_pipeline():
    """Initialize RealSense pipeline with retry on failure. Returns pipeline or None."""
    print("Initializing RealSense camera...")
    
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS)
    cfg.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)
    
    max_retries = 5
    for attempt in range(max_retries):
        pipe = None
        try:
            pipe = rs.pipeline()
            profile = pipe.start(cfg)
            
            # Get camera intrinsics
            depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            intrinsics = depth_profile.get_intrinsics()
            fx, fy = intrinsics.fx, intrinsics.fy
            cx, cy = intrinsics.ppx, intrinsics.ppy
            print(f"Camera intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
            
            # Configure depth sensor for high accuracy
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_sensor.set_option(rs.option.visual_preset, rs.rs400_visual_preset.high_accuracy)
            depth_sensor.set_option(rs.option.depth_units, DEPTH_UNITS)
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
            if depth_sensor.supports(rs.option.laser_power):
                try:
                    lp_range = depth_sensor.get_option_range(rs.option.laser_power)
                    lp_value = float(min(max(LASER_POWER, lp_range.min), lp_range.max))
                    depth_sensor.set_option(rs.option.laser_power, lp_value)
                except Exception as e:
                    print(f"[warn] could not set laser_power: {e}")
            else:
                print("[info] depth sensor does not support laser_power option")
            
            print("Camera initialized successfully.")
            
            # Warm up
            print("Warming up camera...")
            for _ in range(30):
                try:
                    pipe.wait_for_frames()
                except RuntimeError as e:
                    print(f"Error during warmup: {e}")
            
            return pipe
            
        except RuntimeError as e:
            msg = str(e)
            print(f"Initialization error (attempt {attempt + 1}/{max_retries}): {msg}")
            
            # Try to stop pipe if it was partially started
            if pipe:
                try:
                    pipe.stop()
                except:
                    pass
            
            # Hardware reset
            print("Camera busy or in bad state, doing hardware reset...")
            try:
                dev = get_device()
                dev.hardware_reset()
                print("Reset triggered, waiting 2s for re-enumeration...")
                time.sleep(2)
            except Exception as reset_err:
                print(f"Failed to reset device: {reset_err}")
                time.sleep(2)
    
    return None


def _run_detection_with_pipeline(pipe, vis):
    """Run the detection loop with an initialized pipeline. Raises on fatal errors."""
    
    # Setup filters using centralized configuration from cluster.py
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
    
    print("Starting continuous detection loop...")
    
    consecutive_errors = 0
    max_consecutive_errors = 10
    
    while True:
        try:
            frames = pipe.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            
            # Reset error counter on successful frame
            consecutive_errors = 0
            
            # Apply filter chain (only enabled filters)
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
            points = pc.calculate(depth)
            pts_cam = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            
            # Filter valid points
            valid_mask = (pts_cam[:, 2] > 0) & np.isfinite(pts_cam[:, 2])
            pts_cam = pts_cam[valid_mask]
            
            # Run detection (uses centralized defaults from cluster.py)
            result = cluster.detect_grasp_planes(pts_cam)
            
            # Live visualization
            if vis:
                try:
                    cluster.visualize_grasp_detection(vis, result)
                except Exception as viz_err:
                    print(f"Visualization error: {viz_err}")
            
            with RESULT_LOCK:
                RESULT_BUFFER.append(result)
                
        except Exception as e:
            msg = str(e)
            print(f"Error in detection loop: {msg}")
            consecutive_errors += 1
            
            # Check for fatal errors that require pipeline restart
            fatal_keywords = [
                "Device disconnected",
                "get_xu(ctrl=1)",
                "Device or resource busy",
                "wait_for_frames cannot be called before start",
                "failed to set power state",
            ]
            
            if any(kw in msg for kw in fatal_keywords):
                print("Fatal camera error detected, restarting pipeline...")
                raise  # Break out to trigger reconnection
            
            if consecutive_errors >= max_consecutive_errors:
                print(f"Too many consecutive errors ({consecutive_errors}), restarting pipeline...")
                raise
            
            time.sleep(0.1)

def run_server():
    server_address = ('0.0.0.0', SERVER_PORT)
    httpd = HTTPServer(server_address, GraspRequestHandler)
    print(f"Server running on port {SERVER_PORT}")
    httpd.serve_forever()

if __name__ == '__main__':
    # Start detection thread
    detection_thread = threading.Thread(target=run_detection_loop, daemon=True)
    detection_thread.start()
    
    # Start HTTP server in main thread
    try:
        run_server()
    except KeyboardInterrupt:
        print("Shutting down server...")
