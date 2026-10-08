import os
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO, SAM, YOLOWorld
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import matplotlib.pyplot as plt

def _make_transform(R, t):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return T

def _invert_transform(T):
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4, dtype=np.float32)
    T_inv[:3, :3] = R_inv.astype(np.float32)
    T_inv[:3, 3] = t_inv.astype(np.float32)
    return T_inv

def _rotation_matrix_from_euler_xyz(rx_deg, ry_deg, rz_deg):
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(rx), -np.sin(rx)],
            [0.0, np.sin(rx), np.cos(rx)],
        ],
        dtype=np.float32,
    )
    Ry = np.array(
        [
            [np.cos(ry), 0.0, np.sin(ry)],
            [0.0, 1.0, 0.0],
            [-np.sin(ry), 0.0, np.cos(ry)],
        ],
        dtype=np.float32,
    )
    Rz = np.array(
        [
            [np.cos(rz), -np.sin(rz), 0.0],
            [np.sin(rz), np.cos(rz), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    # Convention: R = Rz * Ry * Rx (intrinsic XYZ / ZYX extrinsic)
    return (Rz @ Ry @ Rx).astype(np.float32)

def _mad_inlier_mask(points, k=3.5):
    # Median Absolute Deviation per axis
    median = np.median(points, axis=0)
    abs_dev = np.abs(points - median)
    mad = np.median(abs_dev, axis=0) + 1e-9
    z = abs_dev / mad
    return np.all(z < k, axis=1)


def save_points_as_ply(path, points_xyz):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = int(points_xyz.shape[0])
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points_xyz:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

def run_detection(model, color_bgr):
    det_results = model(color_bgr, imgsz=640, conf=0.25, verbose=False)
    det = det_results[0]
    det_vis = det.plot()
    return det, det_vis

def run_largest_mask_segmentation(sam, color_bgr, det, width, height):
    boxes = getattr(det, "boxes", None)
    mask = None
    if boxes is not None and getattr(boxes, "xyxy", None) is not None and boxes.xyxy.shape[0] > 0:
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
        sam_results = sam(color_bgr, bboxes=xyxy.tolist(), imgsz=640, verbose=False)
        res = sam_results[0]
        masks = getattr(res, "masks", None)
        if masks is not None and getattr(masks, "data", None) is not None:
            mask_np = masks.data.detach().cpu().numpy()
            if mask_np.ndim == 2:
                mask_np = mask_np[0:1]
            areas = mask_np.reshape(mask_np.shape[0], -1).sum(axis=1)
            idx = int(np.argmax(areas))
            mask = (mask_np[idx] > 0.5).astype(np.uint8)
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask

def render_overlay(color_bgr, mask, width, height):
    if mask is None:
        return None
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    overlay = color_bgr.copy()
    alpha = 0.45
    color = (0, 128, 255)
    overlay[mask.astype(bool)] = (
        (1 - alpha) * overlay[mask.astype(bool)] + alpha * np.array(color, dtype=np.uint8)
    ).astype(np.uint8)
    return overlay

def extract_object_points(verts_img, depth, mask):
    valid_depth = depth > 0
    if mask is not None:
        obj_mask = (mask.astype(bool) & valid_depth)
        obj_points = verts_img[obj_mask]
    else:
        obj_points = verts_img[valid_depth]
    return obj_points

def live_loop(pipeline, align_to_color, detector, segmenter, width, height):
    window_name = "Live detection/segmentation"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except Exception:
        pass
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align_to_color.process(frames)
        color_frame = aligned.get_color_frame()
        if not color_frame:
            continue
        color_bgr = np.asanyarray(color_frame.get_data())

        det, det_vis = run_detection(detector, color_bgr)
        mask = None
        try:
            mask = run_largest_mask_segmentation(segmenter, color_bgr, det, width, height)
        except Exception:
            mask = None
        overlay = render_overlay(color_bgr, mask, width, height)
        vis = overlay if overlay is not None else det_vis

        cv2.imshow(window_name, vis)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

to_detect = "bottle"
sam_model = "sam2_b.pt" # "mobile_sam.pt"
yolo_model = "yolov8l-worldv2.pt" # "yolov8s-world.pt"
live=False
if __name__ == "__main__":
    width, height, fps = 640, 480, 30

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    align_to_color = rs.align(rs.stream.color)
    pc = rs.pointcloud()

    profile = pipeline.start(config)

    for _ in range(5):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    aligned = align_to_color.process(frames)
    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()

    depth = np.asanyarray(depth_frame.get_data())  # uint16
    color_bgr = np.asanyarray(color_frame.get_data())  # uint8 HxWx3

    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)
    verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    verts_img = verts.reshape(height, width, 3)

    world = YOLO(yolo_model)
    if hasattr(world, "set_classes") and callable(getattr(world, "set_classes")):
        world.set_classes([to_detect])
    else:
        inner = getattr(world, "model", None)
        if inner is not None and hasattr(inner, "set_classes") and callable(getattr(inner, "set_classes")):
            inner.set_classes([to_detect])

    if live:
        sam = SAM(sam_model)
        try:
            live_loop(pipeline, align_to_color, world, sam, width, height)
        finally:
            pipeline.stop()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        raise SystemExit(0)

    os.makedirs("data", exist_ok=True)
    cv2.imwrite(os.path.join("data", "monitor_color.png"), color_bgr)

    det, det_vis = run_detection(world, color_bgr)
    os.makedirs("data", exist_ok=True)
    cv2.imwrite(os.path.join("data", "monitor_det.png"), det_vis)

    sam = SAM(sam_model)
    mask = run_largest_mask_segmentation(sam, color_bgr, det, width, height)

    obj_points = extract_object_points(verts_img, depth, mask)

    out_ply = os.path.join("data", "monitor_points.ply")
    if obj_points.size == 0:
        pipeline.stop()
        raise SystemExit("No 3D points for the segmented object.")

    save_points_as_ply(out_ply, obj_points)

    overlay = render_overlay(color_bgr, mask, width, height)
    if overlay is None:
        overlay = det_vis
    cv2.imwrite(os.path.join("data", "monitor_seg.png"), overlay)


    print(f"Saved object point cloud: {out_ply}")
    print(f"Num points: {obj_points.shape[0]}")

    pipeline.stop()

    # Meshcat visualization: camera at origin and masked object points
    vis = meshcat.Visualizer().open()

    # Draw camera frame axes at origin (X right, Y down, Z forward)
    L = 0.2
    X = np.array([[0, 0, 0], [L, 0, 0]], dtype=np.float32).T
    Y = np.array([[0, 0, 0], [0, L, 0]], dtype=np.float32).T
    Z = np.array([[0, 0, 0], [0, 0, L]], dtype=np.float32).T
    vis["frame"]["X"].set_object(g.Line(g.PointsGeometry(X), g.LineBasicMaterial(color=0xff0000)))
    vis["frame"]["Y"].set_object(g.Line(g.PointsGeometry(Y), g.LineBasicMaterial(color=0x00ff00)))
    vis["frame"]["Z"].set_object(g.Line(g.PointsGeometry(Z), g.LineBasicMaterial(color=0x0000ff)))

    # Show full frame point cloud and highlight masked object (subsample if large)
    valid_depth = (depth > 0)
    if mask is not None:
        obj_mask = (mask.astype(bool) & valid_depth)
        bg_mask = (valid_depth & (~obj_mask))
    else:
        bg_mask = valid_depth

    bg_points = verts_img[bg_mask]

    # Subsample background points for performance
    max_show_bg = 400000
    if bg_points.shape[0] > max_show_bg:
        sel_bg = np.random.choice(bg_points.shape[0], size=max_show_bg, replace=False)
        show_bg = bg_points[sel_bg]
    else:
        show_bg = bg_points

    B = show_bg.T.astype(np.float32)
    vis["cloud"]["background"].set_object(
        g.Points(
            g.PointsGeometry(B),
            g.PointsMaterial(size=0.0009, color=0x444444)
        )
    )
    try:
        print(f"Meshcat background points shown: {B.shape[1]}")
    except Exception:
        pass

    # Object points overlay (if any)
    if (mask is not None) and (obj_points.size > 0):
        max_show_obj = 400000
        if obj_points.shape[0] > max_show_obj:
            sel_obj = np.random.choice(obj_points.shape[0], size=max_show_obj, replace=False)
            show_obj = obj_points[sel_obj]
        else:
            show_obj = obj_points

        P = show_obj.T.astype(np.float32)
        vis["cloud"]["object"].set_object(
            g.Points(
                g.PointsGeometry(P),
                g.PointsMaterial(size=0.0060, color=0xff3366)
            )
        )
        try:
            print(f"Meshcat object points shown: {P.shape[1]}")
        except Exception:
            pass

    print("Meshcat URL:", vis.url())
    input("Press Enter to close Meshcat...")

