"""Live view of both RealSense cameras: top D435i and wrist D405, colour + depth.

  perception-main/ai_bubble/.venv/bin/python perception-main/rs_dual_view.py
  q / Esc: quit    s: save a snapshot (colour PNG + raw depth .npy, in metres)

Cameras are picked by serial (/dev/videoN numbers move with plug order). Depth is aligned to
colour (same pixels), so the depth tile lines up with the colour tile. The D405's depth and
colour must share one resolution — it refuses to start otherwise. Its depth unit is 0.1 mm
vs the D435i's 1 mm; get_depth_scale() handles that.
"""
import time

import cv2
import numpy as np
import pyrealsense2 as rs

CAMS = {
    #  name      serial          depth (w, h, fps)  colour (w, h, fps)  depth range shown (m)
    "D435i top": ("261822074796", (848, 480, 30), (1280, 720, 30), (0.2, 1.5)),
    "D405 wrist": ("261122276195", (848, 480, 30), (848, 480, 30), (0.07, 0.5)),
}
TILE = (640, 360)


def start(serial, depth, colour):
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, depth[0], depth[1], rs.format.z16, depth[2])
    cfg.enable_stream(rs.stream.color, colour[0], colour[1], rs.format.bgr8, colour[2])
    prof = pipe.start(cfg)
    return pipe, prof.get_device().first_depth_sensor().get_depth_scale(), rs.align(rs.stream.color)


def depth_tile(d_m, rng):
    lo, hi = rng
    v = np.clip((d_m - lo) / (hi - lo), 0, 1)
    img = cv2.applyColorMap((255 - v * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img[d_m == 0] = 0                               # no depth → black
    return img


def label(img, text):
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


def main():
    connected = {s.get_info(rs.camera_info.serial_number) for s in rs.context().query_devices()}
    pipes = {}
    for name, (serial, d, c, rng) in CAMS.items():
        if serial not in connected:
            print(f"{name} ({serial}) not connected — skipped")
            continue
        pipes[name] = (*start(serial, d, c), rng)
    if not pipes:
        raise SystemExit("no RealSense camera found")

    last = {n: (np.zeros((TILE[1], TILE[0], 3), np.uint8),) * 2 for n in CAMS}
    raw = {}
    fps = {n: 0.0 for n in pipes}
    tprev = {n: time.time() for n in pipes}
    cv2.namedWindow("RealSense", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RealSense", 2 * TILE[0], 2 * TILE[1])
    while True:
        for n, (pipe, scale, align, rng) in pipes.items():
            fs = pipe.poll_for_frames()
            if not fs:
                continue
            fs = align.process(fs)
            df, cf = fs.get_depth_frame(), fs.get_color_frame()
            if not df or not cf:
                continue
            d_m = np.asanyarray(df.get_data()).astype(np.float32) * scale
            col = np.asanyarray(cf.get_data())
            raw[n] = (col.copy(), d_m)
            now = time.time()
            fps[n] = 0.9 * fps[n] + 0.1 / max(now - tprev[n], 1e-3)
            tprev[n] = now
            h, w = d_m.shape
            centre = d_m[h // 2 - 2:h // 2 + 3, w // 2 - 2:w // 2 + 3]
            centre = np.median(centre[centre > 0]) if (centre > 0).any() else 0.0
            ct = label(cv2.resize(col, TILE), f"{n} colour {fps[n]:4.1f} fps")
            dt = depth_tile(d_m, rng)
            cv2.drawMarker(dt, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
            dt = label(cv2.resize(dt, TILE), f"{n} depth  centre {centre * 100:5.1f} cm")
            last[n] = (ct, dt)
        grid = np.vstack([np.hstack(last[n]) for n in CAMS])
        cv2.imshow("RealSense", grid)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord("s"):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            for n, (col, d_m) in raw.items():
                tag = n.split()[0]
                cv2.imwrite(f"snap_{stamp}_{tag}.png", col)
                np.save(f"snap_{stamp}_{tag}_depth_m.npy", d_m)
            print(f"saved snap_{stamp}_*")
    for pipe, *_ in pipes.values():
        pipe.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
