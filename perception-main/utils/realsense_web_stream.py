from __future__ import annotations

"""Stream an Intel RealSense (D415 or D455) to the browser as MJPEG.

Why: viewing the camera over SSH normally needs cv2.imshow + an X11 display,
which is painful remotely. This serves the live color (and optionally
colorized depth) streams over plain HTTP, mirroring the WebDisplay used in
``perception/refine_eye_in_hand.py``. Just forward the port and open a browser:

    ssh -L 7000:localhost:7000 <host>
    # then locally: http://localhost:7000/

Examples:
    # Pick the camera model explicitly
    python perception/utils/realsense_web_stream.py --model 455
    python perception/utils/realsense_web_stream.py --model 415

    # Or pick a specific device by serial (use --list-devices to find it)
    python perception/utils/realsense_web_stream.py --serial 123456789012

    # Color only, custom port/quality
    python perception/utils/realsense_web_stream.py --no-depth --web-port 7001 --web-quality 70
"""

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    print(
        "ERROR: OpenCV (cv2) is not installed.\n"
        "Install with: pip install opencv-python"
    )
    raise

try:
    import pyrealsense2 as rs  # Intel RealSense SDK Python bindings
except Exception:  # pragma: no cover
    print(
        "ERROR: pyrealsense2 is not installed or failed to load.\n"
        "Install with one of:\n"
        "  - pip install pyrealsense2          # on Linux/Windows\n"
        "  - pip install pyrealsense2-macosx   # on macOS\n"
    )
    raise


# Substrings used to match a model name reported by the SDK. The D415 reports
# e.g. "Intel RealSense D415" and the D455 "Intel RealSense D455".
_MODEL_TAGS = {
    "415": "D415",
    "455": "D455",
}


# ===========================================================================
# Web MJPEG server (same shape as WebDisplay in refine_eye_in_hand.py):
# each named window is served as a multipart/x-mixed-replace JPEG stream.
# ===========================================================================
class WebDisplay:
    """Serve named windows as MJPEG streams.

    Open http://<host>:<port>/ in a browser. Every window pushed via
    imshow() appears as a live image.
    """

    def __init__(self, host="0.0.0.0", port=7000, quality=80, title="RealSense"):
        self._cond = threading.Condition()
        self._frames = {}      # name -> encoded jpeg bytes
        self._versions = {}    # name -> monotonically increasing int
        self._order = []       # window creation order (for stable layout)
        self._quality = int(quality)
        self.host = host
        self.port = port
        self.title = title
        self._server = None
        self._thread = None
        self._start()

    def _placeholder(self, name, width=640, height=360):
        img = np.full((int(height), int(width), 3), 24, dtype=np.uint8)
        cv2.putText(img, "waiting for frames...", (16, max(28, int(height) // 2 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
        cv2.putText(img, name, (16, max(52, int(height) // 2 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(
            ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
        return buf.tobytes() if ok else None

    def named_window(self, name, width=640, height=360):
        with self._cond:
            if name not in self._versions:
                self._order.append(name)
                self._frames[name] = self._placeholder(name, width, height)
                self._versions[name] = 1
                self._cond.notify_all()

    def imshow(self, name, img):
        ok, buf = cv2.imencode(
            ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
        if not ok:
            return
        data = buf.tobytes()
        with self._cond:
            if name not in self._versions:
                self._order.append(name)
            self._frames[name] = data
            self._versions[name] = self._versions.get(name, 0) + 1
            self._cond.notify_all()

    def shutdown(self):
        with self._cond:
            self._frames.clear()
            self._versions.clear()
            self._order.clear()
            self._cond.notify_all()
        try:
            if self._server is not None:
                self._server.shutdown()
        except Exception:
            pass

    def _start(self):
        disp = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):
                pass

            def do_GET(self):
                disp._handle_get(self)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def _handle_get(self, h):
        parsed = urlparse(h.path)
        path = parsed.path

        if path == "/":
            body = _web_page(self.title).encode("utf-8")
            h.send_response(200)
            h.send_header("Content-Type", "text/html; charset=utf-8")
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)
            return

        if path == "/layout":
            with self._cond:
                windows = list(self._order)
            body = json.dumps({"windows": windows}).encode("utf-8")
            h.send_response(200)
            h.send_header("Content-Type", "application/json")
            h.send_header("Cache-Control", "no-cache")
            h.send_header("Content-Length", str(len(body)))
            h.end_headers()
            h.wfile.write(body)
            return

        if path.startswith("/stream/"):
            name = unquote(path[len("/stream/"):])
            h.send_response(200)
            h.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame")
            h.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            h.send_header("Pragma", "no-cache")
            h.send_header("Connection", "close")
            h.end_headers()
            last_ver = -1
            try:
                while True:
                    with self._cond:
                        ver = self._versions.get(name)
                        if ver is None:
                            break
                        if ver == last_ver:
                            self._cond.wait(timeout=1.0)
                            ver = self._versions.get(name)
                            if ver is None:
                                break
                            if ver == last_ver:
                                continue
                        frame = self._frames.get(name)
                        last_ver = ver
                    if frame is None:
                        continue
                    h.wfile.write(b"--frame\r\n")
                    h.wfile.write(b"Content-Type: image/jpeg\r\n")
                    h.wfile.write(
                        ("Content-Length: %d\r\n\r\n" % len(frame)).encode())
                    h.wfile.write(frame)
                    h.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        h.send_response(404)
        h.end_headers()


def _web_page(title):
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #14161a; color: #e6e6e6;
         font-family: ui-sans-serif, system-ui, sans-serif; }
  header { padding: 10px 16px; background: #1d2026;
           border-bottom: 1px solid #2c2f36; position: sticky; top: 0;
           display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; }
  main { display: flex; flex-wrap: wrap; gap: 16px; padding: 16px;
         align-items: flex-start; }
  figure { margin: 0; background: #1d2026; border: 1px solid #2c2f36;
           border-radius: 8px; overflow: hidden; }
  figcaption { font-size: 12px; padding: 6px 10px; background: #23262d;
               border-bottom: 1px solid #2c2f36; }
  img { display: block; max-width: 100%; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__ &mdash; remote view</h1>
</header>
<main id="windows"></main>
<script>
const windowsEl = document.getElementById('windows');
const shown = new Map();

function syncLayout() {
  fetch('/layout').then(r => r.json()).then(data => {
    const names = data.windows || [];
    for (const name of names) {
      if (shown.has(name)) continue;
      const fig = document.createElement('figure');
      const cap = document.createElement('figcaption');
      cap.textContent = name;
      const img = document.createElement('img');
      const streamUrl = () =>
        '/stream/' + encodeURIComponent(name) + '?t=' + Date.now();
      img.src = streamUrl();
      img.alt = name;
      img.onerror = () => { setTimeout(() => { img.src = streamUrl(); }, 1000); };
      fig.appendChild(cap);
      fig.appendChild(img);
      windowsEl.appendChild(fig);
      shown.set(name, fig);
    }
    for (const [name, fig] of shown) {
      if (!names.includes(name)) {
        fig.remove();
        shown.delete(name);
      }
    }
  }).catch(() => {});
}
setInterval(syncLayout, 1000);
syncLayout();
</script>
</body>
</html>
""".replace("__TITLE__", title)


# ===========================================================================
# RealSense device selection
# ===========================================================================
def _list_connected_devices():
    ctx = rs.context()
    out = []
    for d in ctx.query_devices():
        try:
            out.append({
                "name": d.get_info(rs.camera_info.name),
                "serial": d.get_info(rs.camera_info.serial_number),
                "fw": d.get_info(rs.camera_info.firmware_version),
            })
        except Exception:
            out.append({"name": "<unknown>", "serial": "?", "fw": "?"})
    return out


def _pick_device(model=None, serial=None):
    """Return the serial number of the device to use, or None for any.

    Raises RuntimeError with a helpful message if no match is found.
    """
    devices = _list_connected_devices()
    if not devices:
        raise RuntimeError(
            "No RealSense device connected.\n"
            "- Use a USB 3.x cable plugged directly into the host.\n"
            "- Verify the SDK sees it: rs-enumerate-devices"
        )

    if serial:
        for d in devices:
            if d["serial"] == serial:
                return d["serial"]
        raise RuntimeError(
            f"No device with serial {serial!r}. Connected: "
            + ", ".join(f"{d['name']} (S/N {d['serial']})" for d in devices)
        )

    if model:
        tag = _MODEL_TAGS.get(str(model))
        if tag is None:
            raise RuntimeError(f"Unsupported --model {model!r}; use 415 or 455.")
        matches = [d for d in devices if tag in d["name"]]
        if not matches:
            raise RuntimeError(
                f"No {tag} found. Connected: "
                + ", ".join(f"{d['name']} (S/N {d['serial']})" for d in devices)
            )
        return matches[0]["serial"]

    # No filter: use the only device, or refuse if ambiguous.
    if len(devices) == 1:
        return devices[0]["serial"]
    raise RuntimeError(
        "Multiple RealSense devices connected; pick one with --model or "
        "--serial. Connected: "
        + ", ".join(f"{d['name']} (S/N {d['serial']})" for d in devices)
    )


# ===========================================================================
# Streaming loop
# ===========================================================================
def stream(
    *,
    model=None,
    serial=None,
    width=848,
    height=480,
    fps=30,
    show_depth=True,
    web_host="0.0.0.0",
    web_port=7000,
    web_quality=80,
):
    chosen_serial = _pick_device(model=model, serial=serial)

    devices = {d["serial"]: d for d in _list_connected_devices()}
    dev_name = devices.get(chosen_serial, {}).get("name", "RealSense")

    pipeline = rs.pipeline()
    config = rs.config()
    if chosen_serial:
        config.enable_device(chosen_serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, int(fps))
    if show_depth:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, int(fps))

    try:
        profile = pipeline.start(config)
    except Exception as start_err:
        print(
            f"Requested {width}x{height}@{fps} failed ({start_err}). "
            "Falling back to default device profile."
        )
        config = rs.config()
        if chosen_serial:
            config.enable_device(chosen_serial)
        profile = pipeline.start(config)

    colorizer = rs.colorizer() if show_depth else None
    # Align depth to color so the two views line up visually.
    align = rs.align(rs.stream.color) if show_depth else None

    web = WebDisplay(host=web_host, port=web_port, quality=web_quality,
                     title=f"{dev_name}")
    web.named_window("color", width, height)
    if show_depth:
        web.named_window("depth", width, height)

    shown_host = "localhost" if web_host in ("0.0.0.0", "") else web_host
    print(f"Streaming {dev_name} (S/N {chosen_serial}) to the web.")
    print(f"  Open: http://{shown_host}:{web_port}/")
    if web_host in ("0.0.0.0", ""):
        print(f"  Over SSH: ssh -L {web_port}:localhost:{web_port} <host>")
    print("  Press Ctrl+C to stop.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)

            color_frame = frames.get_color_frame()
            if color_frame is not None:
                color = np.asanyarray(color_frame.get_data())
                web.imshow("color", color)

            if show_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame is not None:
                    colorized = colorizer.process(depth_frame)
                    depth_viz = np.asanyarray(colorized.get_data())
                    web.imshow("depth", depth_viz)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        web.shutdown()
        try:
            pipeline.stop()
        except Exception:
            pass


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Stream an Intel RealSense (D415/D455) to the browser as MJPEG.")
    p.add_argument("--model", choices=["415", "455"], default=None,
                   help="Which camera to use (D415 or D455). Omit if only one is connected.")
    p.add_argument("--serial", default=None,
                   help="Select a device by serial number (overrides --model).")
    p.add_argument("--width", type=int, default=848, help="Stream width (default 848).")
    p.add_argument("--height", type=int, default=480, help="Stream height (default 480).")
    p.add_argument("--fps", type=int, default=30, help="Stream FPS (default 30).")
    p.add_argument("--no-depth", action="store_true",
                   help="Stream color only (skip colorized depth).")
    p.add_argument("--web-host", default="0.0.0.0",
                   help="Bind address for the web viewer (default 0.0.0.0).")
    p.add_argument("--web-port", type=int, default=7000,
                   help="Port for the web viewer (default 7000).")
    p.add_argument("--web-quality", type=int, default=80,
                   help="JPEG quality 1-100 (default 80).")
    p.add_argument("--list-devices", action="store_true",
                   help="List connected RealSense devices and exit.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.list_devices:
        devices = _list_connected_devices()
        if not devices:
            print("No RealSense devices found.")
            return 1
        print("Connected RealSense devices:")
        for d in devices:
            print(f"  - {d['name']} (S/N {d['serial']}, FW {d['fw']})")
        return 0

    try:
        stream(
            model=args.model,
            serial=args.serial,
            width=args.width,
            height=args.height,
            fps=args.fps,
            show_depth=not args.no_depth,
            web_host=args.web_host,
            web_port=args.web_port,
            web_quality=args.web_quality,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
