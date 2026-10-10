"""Live dashboard during a run, for the team / audience (layout: docs/1000086694.png).

  top bar      elapsed time, current step
  left         D435 (top) colour stream with the basket cells (filled ones blue) and the blocks still on the
               table: their grasp rectangle (centre, jaw, size, top height) projected from the CURRENT robot
               coordinates, so every update (fast top check, D405 re-measure) shows at once; the scan
               picture while the camera is closed
  numbering    Block N / cell N = the order the run will take them (fill_cells.planned_order -> plan())
  right top    D405 (wrist) stream with a crosshair and the gripper opening; COMPLETE filled / all cells
  right        detected blocks: checked once placed, short side a, long side b, height h, (x, y) base frame

The mission process cannot share a RealSense with another program, so it renders the dashboard itself: a
background thread takes the newest frame of each open camera (sources registered with set_source), draws
the page and saves one JPEG to /dev/shm/pickr_live.jpg about 8 times a second. live_viewer.py (a separate
window process) shows that file, so closing or crashing the window never touches the robot.

  live_view.start()                         # pickr.py does this (and opens the window)
  live_view.set_source("top", cam.latest)   # any function returning a BGR image or None
  live_view.status("cell 2: down to block")
  live_view.targets(block_xyz, cell_xyz)    # base frame (m), drawn on the top view
  live_view.scene(scene)                    # fill_cells scene (read live: placed blocks, filled cells)
  live_view.set_arm(p)                      # piper interface: gripper opening
"""
import os
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path("/dev/shm/pickr_live.jpg")
W, H = 1920, 1080
_src = {}
_state = dict(status="", targets=None, K=None, t0=time.time(), scene=None, blocks=[], scan_img=None, arm=None,
              cell_no={})
_run = False

# colours (RGB) as in the mock-up
BG, PANEL, LINE = (17, 20, 24), (22, 25, 30), (42, 46, 53)
TEXT, MUTED, DIM = (230, 232, 235), (138, 144, 153), (90, 96, 105)
BLUE, ORANGE, GREEN = (74, 144, 217), (240, 160, 80), (61, 220, 132)

_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans{}.ttf"     # English only: Korean text broke too easily
_fonts = {}


def _f(size, bold=False, mono=False):
    """DejaVu Sans / Sans Mono; PIL's default font if it is missing."""
    key = (size, bold, mono)
    if key not in _fonts:
        try:
            _fonts[key] = ImageFont.truetype(_FONT.format(("Mono" if mono else "") + ("-Bold" if bold else "")), size)
        except OSError:
            _fonts[key] = ImageFont.load_default(size)
    return _fonts[key]


# ------------------------------------------------------------------ API
def set_source(name, fn, K=None):
    """name 'top' or 'wrist'; fn() -> BGR image or None; fn=None removes it. K (top) for the markers."""
    if fn is None:
        _src.pop(name, None)
    else:
        _src[name] = fn
    if K is not None and name == "top":
        _state["K"] = K


def stage(i, names=("Partition", "Blocks", "Drag")):
    """Demo mode: which step runs (1-based; 0 = not started, len+1 = all done), drawn in the top bar."""
    _state["stage"] = (int(i), tuple(names))


def status(text):
    _state["status"] = str(text)


def plan(pairs):
    """[(block, cell)] in the order the run will do them: Block 1 goes into cell 1, ... (dashboard numbers;
    blocks / cells outside the plan follow). Cells keep their fill-order number for the whole run."""
    sc = _state["scene"] or {}
    first = [b for b, _ in pairs]
    _state["blocks"] = first + [b for b in sc.get("free", []) if not any(b is f for f in first)]
    _state["cell_no"] = {k: i for i, (_, k) in enumerate(pairs, 1)}
    _state["cell_of"] = {id(b): i for i, (b, _) in enumerate(pairs, 1)}


def targets(block_xyz=None, cell_xyz=None):
    _state["targets"] = (block_xyz, cell_xyz)


def scene(sc):
    """The scene of a (re)scan. Its blocks are numbered here (Block 1..n); the dict is read live afterwards:
    a block gone from sc["free"] has been placed, a cell in sc["occupied"] is filled."""
    _state["scene"] = sc
    _state["blocks"] = list(sc.get("free", []))
    _state["cell_no"], _state["cell_of"] = {}, {}
    img = None
    try:
        img = cv2.imread(str(Path(sc["run"]) / "color.jpg"))
    except Exception:
        pass
    _state["scan_img"] = img
    if sc.get("K") is not None:
        _state["K"] = sc["K"]


def set_arm(p):
    _state["arm"] = p


# ------------------------------------------------------------------ drawing
def _grab(name):
    fn = _src.get(name)
    if fn is None:
        return None
    try:
        return fn()
    except Exception:
        return None


def _fit(img, box):
    """Resize img to fit box (x0, y0, x1, y1), centred. Returns (PIL image, x, y, scale)."""
    x0, y0, x1, y1 = box
    s = min((x1 - x0) / img.shape[1], (y1 - y0) / img.shape[0])
    w, h = int(img.shape[1] * s), int(img.shape[0] * s)
    im = Image.fromarray(cv2.cvtColor(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    return im, x0 + (x1 - x0 - w) // 2, y0 + (y1 - y0 - h) // 2, s


def _rect_px(r):
    """Grasp rectangle of a block (top face: centre, jaw = across the short side, size) in top-camera pixels."""
    K = _state["K"]
    if K is None or not r.get("top_xyz_m") or not r.get("size_m"):
        return None
    from observer import project
    a = np.radians(r["jaw_yaw_deg"])
    j, l = np.array([np.cos(a), np.sin(a), 0.0]), np.array([-np.sin(a), np.cos(a), 0.0])
    c = np.asarray(r["top_xyz_m"], float)
    hs, hl = r["size_m"][0] / 2, r["size_m"][1] / 2
    P = np.array([c + sj * hs * j + sl * hl * l for sj, sl in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    return project(P, K)


def _on_table(sc):
    free = sc.get("free", [])
    return [(i, b) for i, b in enumerate(_state["blocks"], 1) if any(b is f for f in free)]


def _top_overlay(img):
    """Cells and markers drawn on the full-resolution top image (shapes only; labels later in Korean)."""
    sc = _state["scene"]
    if sc is not None:
        over = img.copy()
        for k, c in sc.get("num", {}).items():
            q = np.asarray(c.get("quad_px", []), np.int32)
            if len(q) != 4:
                continue
            if k in sc.get("occupied", ()):
                inner = (q - q.mean(0)) * 0.55 + q.mean(0)
                cv2.fillConvexPoly(over, inner.astype(np.int32), BLUE[::-1])
        cv2.addWeighted(over, 0.85, img, 0.15, 0, img)
        for k, c in sc.get("num", {}).items():
            q = np.asarray(c.get("quad_px", []), np.int32)
            if len(q) == 4:
                cv2.polylines(img, [q], True, (235, 235, 235), 3, cv2.LINE_AA)
        for _, b in _on_table(sc):
            try:
                q = _rect_px(b.get("robot", {}))
            except Exception:
                q = None
            if q is not None:
                cv2.polylines(img, [q.astype(np.int32)], True, ORANGE[::-1], 4, cv2.LINE_AA)
            elif "box" in b:
                x1, y1, x2, y2 = (int(round(v)) for v in b["box"])
                cv2.rectangle(img, (x1, y1), (x2, y2), ORANGE[::-1], 4, cv2.LINE_AA)
        if _state["K"] is not None:                          # --bul: defective blocks, red X
            from observer import project
            for b in sc.get("defects", []):
                try:
                    u, v = project(np.array(b["robot"]["top_xyz_m"], float)[None], _state["K"])[0].astype(int)
                except Exception:
                    continue
                cv2.line(img, (u - 45, v - 45), (u + 45, v + 45), (0, 0, 255), 9, cv2.LINE_AA)
                cv2.line(img, (u - 45, v + 45), (u + 45, v - 45), (0, 0, 255), 9, cv2.LINE_AA)
    if _state["targets"] and _state["K"] is not None:
        try:
            from observer import project
            for xyz, col in zip(_state["targets"], (ORANGE[::-1], (255, 0, 255))):
                if xyz is not None:
                    u, v = project(np.asarray(xyz, float)[None], _state["K"])[0].astype(int)
                    cv2.circle(img, (u, v), 30, col, 4, cv2.LINE_AA)
        except Exception:
            pass
    return img


def _left(page, d):
    d.text((32, 80), "D435 · Workspace", font=_f(28, bold=True), fill=TEXT)
    box = (32, 130, 1153, 970)
    d.rectangle(box, fill=(12, 14, 17), outline=LINE, width=2)
    img = _grab("top")
    live = img is not None
    if not live:
        img = _state["scan_img"]
    if img is None:
        d.text((560, 540), "waiting for the camera...", font=_f(26), fill=MUTED, anchor="mm")
        d.text((50, 938), "[D435 RGB stream]", font=_f(16, mono=True), fill=MUTED)
        return
    img = np.ascontiguousarray(img)
    d.text((1152, 112), f"{img.shape[1]}×{img.shape[0]} · {'30 fps' if live else 'scan'}", font=_f(18, mono=True),
           fill=MUTED, anchor="rs")
    im, ox, oy, s = _fit(_top_overlay(img.copy()), (box[0] + 2, box[1] + 2, box[2] - 2, box[3] - 40))
    page.paste(im, (ox, oy))
    d.text((50, 938), "[D435 RGB stream]" if live else "[scan picture · camera closed]", font=_f(16, mono=True), fill=MUTED)
    sc = _state["scene"]
    if sc is None:
        return
    P = lambda u, v: (ox + u * s, oy + v * s)
    quads = [np.asarray(c.get("quad_px", []), float) for c in sc.get("num", {}).values()]
    quads = [q for q in quads if len(q) == 4]
    if quads:
        allq = np.vstack(quads)
        x, y = P(allq[:, 0].min(), allq[:, 1].min())
        d.text((x, y - 8), "Placement grid", font=_f(19), fill=TEXT, anchor="ls", stroke_width=3, stroke_fill=BG)
    for k, c in sc.get("num", {}).items():
        q = np.asarray(c.get("quad_px", []), float)
        n = _state["cell_no"].get(k)
        if len(q) == 4 and (n is not None or not _state["cell_no"]):
            x, y = P(*q.mean(0))
            d.text((x, y - 6), str(n if n is not None else k), font=_f(26, bold=True), fill=(255, 255, 255),
                   anchor="mm", stroke_width=3, stroke_fill=BG)
            if n is not None:                            # big = fill order, small = the cell's own number
                d.text((x, y + 16), f"cell {k}", font=_f(13), fill=MUTED, anchor="mm", stroke_width=2, stroke_fill=BG)
        elif len(q) == 4 and k not in sc.get("occupied", set()):   # in the grid but no reachable block for it
            x, y = P(*q.mean(0))
            d.text((x, y - 8), "-", font=_f(26, bold=True), fill=MUTED, anchor="mm", stroke_width=3, stroke_fill=BG)
            d.text((x, y + 14), f"cell {k}: no block", font=_f(13), fill=MUTED, anchor="mm", stroke_width=2,
                   stroke_fill=BG)
    for i, b in _on_table(sc):
        try:
            q = _rect_px(b.get("robot", {}))
        except Exception:
            q = None
        if q is None and "box" not in b:
            continue
        if q is not None:                                # label centred above the rectangle's top corner
            x, y = P(q[:, 0].mean(), q[:, 1].min())
        else:
            x, y = P((b["box"][0] + b["box"][2]) / 2, b["box"][1])
        t = f"Block {i}"
        tw = d.textlength(t, font=_f(18))
        d.rounded_rectangle((x - tw / 2 - 8, y - 32, x + tw / 2 + 8, y - 6), 3, fill=ORANGE)
        d.text((x, y - 19), t, font=_f(18), fill=(30, 20, 10), anchor="mm")


def _wrist(page, d):
    d.text((1184, 80), "D405 · Gripper", font=_f(23, bold=True), fill=TEXT)
    d.text((1624, 104), "wrist", font=_f(16, mono=True), fill=MUTED, anchor="rs")
    box = (1184, 120, 1625, 450)
    d.rectangle(box, fill=(12, 14, 17), outline=LINE, width=2)
    img = _grab("wrist")
    if img is not None:
        im, ox, oy, _ = _fit(np.ascontiguousarray(img), (box[0] + 2, box[1] + 2, box[2] - 2, box[3] - 2))
        page.paste(im, (ox, oy))
    else:
        d.text(((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), "camera closed", font=_f(22), fill=MUTED, anchor="mm")
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    d.line((cx, box[1], cx, box[3]), fill=(110, 125, 145), width=1)
    d.line((box[0], cy, box[2], cy), fill=(110, 125, 145), width=1)
    w = None
    if _state["arm"] is not None:
        try:
            w = _state["arm"].GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0
        except Exception:
            w = None
    t = f"gripper: {w:.0f} mm" if w is not None else "gripper: — mm"
    tw = d.textlength(t, font=_f(16, mono=True))
    d.rectangle((box[0] + 6, box[3] - 30, box[0] + 18 + tw, box[3] - 8), fill=(12, 14, 17))
    d.text((box[0] + 12, box[3] - 19), t, font=_f(16, mono=True), fill=TEXT, anchor="lm")


def _complete(d):
    box = (1648, 82, 1888, 450)
    d.rounded_rectangle(box, 6, fill=PANEL, outline=LINE, width=2)
    sc = _state["scene"]
    cells = sc.get("num", {}) if sc else {}
    n, done = len(cells), len([k for k in cells if k in sc.get("occupied", ())]) if sc else 0
    cx = (box[0] + box[2]) // 2
    d.text((cx, 212), "COMPLETE", font=_f(20, mono=True), fill=MUTED, anchor="mm")
    a, b = str(done) if sc else "—", str(n) if sc else "—"
    big = _f(84)
    wa, ws, wb = d.textlength(a, font=big), d.textlength("/", font=big), d.textlength(b, font=big)
    x = cx - (wa + ws + wb) / 2
    d.text((x, 278), a, font=big, fill=BLUE, anchor="lm")
    d.text((x + wa, 278), "/", font=big, fill=MUTED, anchor="lm")
    d.text((x + wa + ws, 278), b, font=big, fill=TEXT, anchor="lm")
    if n:
        gap, x0, x1 = 6, box[0] + 22, box[2] - 22
        seg = (x1 - x0 - gap * (n - 1)) / n
        for i in range(n):
            xs = x0 + i * (seg + gap)
            d.rounded_rectangle((xs, 324, xs + seg, 331), 3, fill=BLUE if i < done else LINE)


def _blocks(d):
    sc = _state["scene"]
    blocks = _state["blocks"]
    d.text((1184, 476), "Detected blocks", font=_f(21, bold=True), fill=TEXT)
    d.text((1888, 497), f"{len(blocks)} found", font=_f(16), fill=MUTED, anchor="rs")
    if not blocks:
        return
    free = sc.get("free", []) if sc else []
    y0, y_end, gap = 515, 1060, 12
    h = min(118, (y_end - y0) / len(blocks) - gap)
    for i, b in enumerate(blocks, 1):
        r = b.get("robot", {})
        placed = not any(b is f for f in free)
        y = y0 + (i - 1) * (h + gap)
        d.rounded_rectangle((1184, y, 1879, y + h), 6, fill=PANEL, outline=LINE, width=2)
        cy = y + h / 2
        cb = min(26, h - 8)
        d.rounded_rectangle((1209, cy - cb / 2, 1209 + cb, cy + cb / 2), 3,
                            fill=BLUE if placed else (245, 245, 245))
        if placed:
            d.line((1209 + cb * 0.22, cy, 1209 + cb * 0.42, cy + cb * 0.22, 1209 + cb * 0.8, cy - cb * 0.25),
                   fill=(255, 255, 255), width=3)
        fb = _f(min(24, int(h * 0.45)), bold=True)
        d.text((1255, cy), f"Block {i}", font=fb, fill=MUTED if placed else TEXT, anchor="lm")
        tc = _state.get("cell_of", {}).get(id(b))
        if tc is not None:
            d.text((1265 + d.textlength(f"Block {i}", font=fb), cy), f"→ cell {tc}", font=_f(min(18, int(h * 0.35))),
                   fill=MUTED if placed else BLUE, anchor="lm")
        size = r.get("size_m") or [None, None]
        hh = r.get("height_m")
        xy = r.get("top_xyz_m")
        mm = lambda v: "—" if v is None else f"{v * 1000:.0f}"
        rows = [("a: ", f"{mm(size[0])} mm"), ("b: ", f"{mm(size[1])} mm"), ("h: ", f"{mm(hh)} mm"),
                ("(x, y): ", f"({mm(xy[0]) if xy else '—'}, {mm(xy[1]) if xy else '—'})")]
        fm = _f(16, mono=True)
        if h >= 100:                                     # 4 lines as in the mock-up
            for j, (k, v) in enumerate(rows):
                yy = y + 14 + j * 23
                d.text((1655, yy), k, font=fm, fill=MUTED)
                d.text((1655 + d.textlength(k, font=fm), yy), v, font=fm, fill=TEXT)
        else:                                            # many blocks: one line
            t = f"{mm(size[0])}×{mm(size[1])}×{mm(hh)} mm   ({mm(xy[0]) if xy else '—'}, {mm(xy[1]) if xy else '—'})"
            d.text((1860, cy), t, font=_f(min(16, int(h * 0.5)), mono=True), fill=TEXT, anchor="rm")


def render():
    page = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(page)
    d.rounded_rectangle((32, 24, 1888, 62), 4, fill=PANEL, outline=LINE, width=1)
    el = int(time.time() - _state["t0"])
    fm = _f(18, mono=True)
    d.text((48, 43), "Elapsed", font=_f(18), fill=MUTED, anchor="lm")
    d.text((130, 43), f"{el // 3600:02d}:{el // 60 % 60:02d}:{el % 60:02d}", font=fm, fill=TEXT, anchor="lm")
    d.text((300, 43), "Status", font=_f(18), fill=MUTED, anchor="lm")
    d.ellipse((378, 38, 388, 48), fill=GREEN)
    st = re.sub(r"cell (\d+)", lambda m: f"cell {_state['cell_no'].get(int(m.group(1)), m.group(1))}",
                _state["status"] or "ready")                # same numbers as on the picture
    stg = _state.get("stage")
    d.text((398, 43), st[:62 if stg else 120], font=fm, fill=GREEN, anchor="lm")
    if stg:                                             # demo: 1 Partition > 2 Blocks > 3 Drag, right-aligned
        cur, names = stg
        x = 1880
        for k in range(len(names), 0, -1):
            t = f"{k} {names[k - 1]}"
            fb = _f(17, bold=True)
            w = d.textlength(t, font=fb) + 20
            col = GREEN if k < cur else BLUE if k == cur else LINE
            d.rounded_rectangle((x - w, 29, x, 57), 6, fill=col)
            d.text((x - w / 2, 43), t, font=fb, fill=(255, 255, 255) if k <= cur else MUTED, anchor="mm")
            x -= w + 8
            if k > 1:
                d.text((x, 43), ">", font=fb, fill=MUTED, anchor="rm")
                x -= 18
    _left(page, d)
    _wrist(page, d)
    _complete(d)
    _blocks(d)
    return cv2.cvtColor(np.asarray(page), cv2.COLOR_RGB2BGR)


def _loop():
    tmp = OUT.with_suffix(".tmp.jpg")
    while _run:
        t = time.time()
        try:
            cv2.imwrite(str(tmp), render(), [cv2.IMWRITE_JPEG_QUALITY, 85])
            os.replace(tmp, OUT)                              # the viewer never reads a half-written file
        except Exception:
            pass
        time.sleep(max(0.0, 0.125 - (time.time() - t)))


def start():
    global _run
    if _run:
        return
    _run = True
    _state["t0"] = time.time()
    threading.Thread(target=_loop, daemon=True).start()


def stop():
    global _run
    _run = False
