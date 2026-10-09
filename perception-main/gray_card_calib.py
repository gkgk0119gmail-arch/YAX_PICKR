"""Grey-card colour/brightness calibration for the top D435i and wrist D405 (pyrealsense2).

Same procedure as WeGo's wego-piper-web (rs/piper_rs/hub.py calibrate_gray_card), using its
pure logic vendored in third_party/wego_graycard.py:
  1. auto-exposure on (auto white balance off — its result can't be read back) and settle
  2. auto-exposure off → the exposure it found is frozen
  3. white balance from the card's R/B balance, read back and repeat
  4. exposure snapped to whole 1/120 s flicker periods, gain scaled until the card reads 18 % grey
     (luma 118) — read back and repeat; then a 1 s flicker check
  5. measure again and report; refuse saturated / shadowed / too-small card readings

One change from WeGo's verdict: white balance only moves R against B, so the goal here is R = B
on the card (±3 %), not WeGo's "all three channels within 2 %". Measured on site (2026-10-09,
WB swept 2800–6500 K on the card): at R = B the D405 still reads G ~9 % high and the D435i
~17 % low — no white-balance setting gets under 2 %, so WeGo's check could never pass here.
The leftover green tint is reported (`green_tint_pct`), not treated as a failure.
Colour/brightness only — NOT geometric calibration (that needs a checkerboard / ChArUco).
The D405's colour comes from its stereo module, so its exposure also applies to its depth.

Run (needs a GUI OpenCV — conda env piper_sim; the pipeline venv's OpenCV is headless):
  python gray_card_calib.py
  click the card in a camera tile → its box turns green when the reading is usable
  c: calibrate the selected camera    +/-: box size    q / Esc: quit
Successful results are written to pac2026/camera_profile.yaml; the pipeline applies them
whenever it opens the cameras (RealSense settings reset on replug / power loss).
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from third_party import wego_graycard as gc  # noqa: E402

PAC = HERE.parent / "pac2026"
SETTLE_S, ROUNDS, STEP_WAIT = 2.0, 3, 0.4          # WeGo's _CAL_SETTLE_S / _CAL_ROUNDS / sleeps
WB_ROUNDS, RB_TOL = 6, 0.03                        # R/B balance target (see docstring)
#: brightness steps: the image is gamma-encoded, so a proportional exposure step under-corrects
#: (on site the D435i went 213 → 166 → 138 → 134 in WeGo's 3 rounds) — allow more rounds
EXP_ROUNDS = 8
#: Korea's 60 Hz mains → lights flicker at 120 Hz. Measured on site: D435i colour at 4.7 ms
#: exposure swung luma 110–120 (std 3.0) frame to frame; at 8.33 ms std 0.1. So exposure is
#: kept to whole periods and brightness is set with gain (WeGo's "exposure fixed, gain" mode).
FLICKER_PERIOD_US = 1e6 / 120
MAX_FLICKER = 2.0


def rb_error(r):
    return abs(r.b / r.r - 1.0) if r.r > 0 else 1.0


def green_tint_pct(r):
    rb = (r.r + r.b) / 2
    return (r.g / rb - 1.0) * 100 if rb > 0 else 0.0
TILE_W = 640
O = rs.option


def exposure_scale(max_value):
    """Raw exposure → µs. D435i RGB sensor: 100 µs units (max 10000); D405 stereo module: µs."""
    return 100 if max_value <= 20000 else 1


class Cam:
    def __init__(self, role, serial, color_size, depth_size, fps):
        self.role, self.serial = role, str(serial)
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, *color_size, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, *depth_size, rs.format.z16, fps)
        self.pipe = rs.pipeline()
        prof = self.pipe.start(cfg)
        dev = prof.get_device()
        self.name = dev.get_info(rs.camera_info.name)
        self.sensor = next(s for s in dev.query_sensors()
                           if any(p.stream_type() == rs.stream.color for p in s.get_stream_profiles()))
        self.exp_scale = exposure_scale(self.sensor.get_option_range(O.exposure).max)
        self.frame, self.count = None, 0
        self.lock = threading.Lock()
        self.run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.run:
            try:
                fs = self.pipe.wait_for_frames(2000)
            except RuntimeError:
                continue
            cf = fs.get_color_frame()
            if cf:
                with self.lock:
                    self.frame = np.asanyarray(cf.get_data()).copy()
                    self.count += 1

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def fresh(self, n=3, timeout=3.0):
        """A frame captured after the last setting change (n frames later)."""
        start, t0 = self.count, time.time()
        while self.count < start + n and time.time() - t0 < timeout:
            time.sleep(0.01)
        return self.latest()

    def get(self, opt):
        return self.sensor.get_option(getattr(O, opt))

    def set(self, opt, value):
        r = self.sensor.get_option_range(getattr(O, opt))
        self.sensor.set_option(getattr(O, opt), float(min(max(round(value), r.min), r.max)))

    def rng(self, opt):
        r = self.sensor.get_option_range(getattr(O, opt))
        return r.min, r.max

    def close(self):
        self.run = False
        time.sleep(0.1)
        self.pipe.stop()


def calibrate(cam, roi, log):
    """WeGo's calibrate_gray_card (adjust="exposure") on one camera; returns a result dict."""
    target = gc.TARGET_LUMA
    steps = []
    cam.set("enable_auto_white_balance", 0)
    cam.set("enable_auto_exposure", 1)
    time.sleep(SETTLE_S)
    before = gc.measure(cam.fresh(), roi)
    ok, why = before.usable
    if not ok:
        return {"ok": False, "verdict": why, "before": before.to_dict()}
    cam.set("enable_auto_exposure", 0)                 # freeze the exposure AE found
    lo, hi = cam.rng("white_balance")
    wb = cam.get("white_balance")
    cam.set("white_balance", wb)                       # AWB off falls back to this; let it land before measuring
    time.sleep(2 * STEP_WAIT)
    reading = gc.measure(cam.fresh(10), roi)
    for _ in range(WB_ROUNDS):
        if rb_error(reading) <= RB_TOL:
            break
        new = gc.white_balance_for(reading, wb, lo, hi)
        if abs(new - wb) < 10:
            break
        cam.set("white_balance", new)
        wb = cam.get("white_balance")
        time.sleep(2 * STEP_WAIT)
        reading = gc.measure(cam.fresh(10), roi)
        steps.append({"white_balance": round(wb), "b_over_r": round(reading.b / reading.r, 3),
                      "luma": round(reading.luma, 1)})
        log(f"  {cam.role}: WB {wb:.0f} K → B/R {reading.b / reading.r:.2f}, green {green_tint_pct(reading):+.0f}%, "
            f"luma {reading.luma:.0f}")

    # exposure: a whole number of mains-flicker periods (1/120 s in Korea), nearest to what AE chose;
    # brightness then comes from gain (WeGo's adjust="gain" step, same proportional model)
    period_raw = FLICKER_PERIOD_US / cam.exp_scale
    e_lo, e_hi = cam.rng("exposure")
    k = max(1, round(cam.get("exposure") / period_raw))
    g_lo, g_hi = cam.rng("gain")
    for _attempt in range(3):
        k = int(min(max(k, 1), e_hi // period_raw))
        cam.set("exposure", k * period_raw)
        # restart gain from the sensor default each time: left at a tiny value by an earlier run,
        # its proportional steps round to nothing and the search stalls (seen on site: gain 3)
        gain = max(cam.sensor.get_option_range(O.gain).default, g_lo, 1.0)
        cam.set("gain", gain)
        time.sleep(2 * STEP_WAIT)
        reading = gc.measure(cam.fresh(10), roi)
        prev = None                                    # (gain, luma) of the previous step
        for _ in range(EXP_ROUNDS):
            if abs(reading.luma - target) <= gc.LUMA_TOLERANCE / 2:   # aim tighter than the pass band
                break
            if prev is None:
                new = gc.exposure_for(reading, gain, max(g_lo, 1.0), g_hi, target)   # WeGo's first step
            else:
                # brightness is far from proportional to gain on the D435i (on site: gain 64 → 25 moved
                # luma only 150 → 128, i.e. luma ~ gain^0.17), so learn the exponent from the last two
                # readings (secant in log space) instead of assuming 1
                p_exp = np.log(reading.luma / prev[1]) / np.log(gain / prev[0]) if gain != prev[0] else 1.0
                p_exp = float(np.clip(p_exp, 0.05, 1.5))
                ratio = float(np.clip((target / reading.luma) ** (1 / p_exp), 0.25, 4.0))
                new = float(np.clip(gain * ratio, max(g_lo, 1.0), g_hi))
            if round(new) == round(gain):
                break
            prev = (gain, reading.luma)
            cam.set("gain", new)
            gain = cam.get("gain")
            time.sleep(2 * STEP_WAIT)
            reading = gc.measure(cam.fresh(10), roi)
            steps.append({"exposure_us": round(cam.get("exposure") * cam.exp_scale), "gain": round(gain),
                          "luma": round(reading.luma, 1)})
            log(f"  {cam.role}: exposure {cam.get('exposure') * cam.exp_scale / 1000:.2f} ms, gain {gain:.0f} "
                f"→ luma {reading.luma:.0f}")
        if abs(reading.luma - target) <= gc.LUMA_TOLERANCE:
            break
        # gain ran out of range: one flicker period more / less exposure and try again
        k += 1 if reading.luma < target else -1
        if k < 1:
            break

    # flicker check: the card must hold still over a second at the final setting
    lumas = []
    t0 = time.time()
    while time.time() - t0 < 1.0:
        lumas.append(gc.measure(cam.fresh(1), roi).luma)
    flicker = float(np.std(lumas))
    log(f"  {cam.role}: luma std over 1 s = {flicker:.1f}")

    usable, why = reading.usable
    bad = [] if usable else [why]
    if usable and abs(reading.luma - target) > gc.LUMA_TOLERANCE:
        bad.append(f"luma {reading.luma:.0f} (target {target:.0f})")
    if usable and rb_error(reading) > RB_TOL:
        bad.append(f"B/R {reading.b / reading.r:.2f} (target 1.00)")
    if flicker > MAX_FLICKER:
        bad.append(f"brightness flickers (std {flicker:.1f})")
    ok = not bad
    verdict = ("OK" if ok else " · ".join(bad)) + f" — green tint {green_tint_pct(reading):+.0f}% (not fixable by WB)"
    return {"ok": ok, "verdict": verdict, "serial": cam.serial, "camera": cam.name,
            "enable_auto_exposure": 0, "enable_auto_white_balance": 0,
            "exposure": round(cam.get("exposure")), "exposure_us": round(cam.get("exposure") * cam.exp_scale),
            "gain": round(cam.get("gain")), "white_balance": round(cam.get("white_balance")),
            "luma_std_1s": round(flicker, 2),
            "before": before.to_dict(), "after": reading.to_dict(), "steps": steps,
            "roi_xywh": [int(v) for v in roi], "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def save_profile(path, role, res):
    prof = yaml.safe_load(path.read_text()) if path.exists() else {}
    prof = prof or {}
    prof[role] = {k: res[k] for k in ("serial", "camera", "enable_auto_exposure", "enable_auto_white_balance",
                                      "exposure", "exposure_us", "gain", "white_balance", "luma_std_1s",
                                      "roi_xywh", "calibrated_at")}
    prof[role]["card_after"] = {k: res["after"][k] for k in ("b", "g", "r", "luma", "neutral_error_pct", "spread_pct")}
    path.write_text("# written by perception-main/gray_card_calib.py — applied by pac/cameras.py on open\n"
                    + yaml.safe_dump(prof, sort_keys=False, allow_unicode=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=PAC / "config.yaml")
    ap.add_argument("--cams", nargs="+", default=["top", "wrist"])
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())["camera"]
    out = a.config.parent / cfg.get("profile", "camera_profile.yaml")

    connected = {d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()}
    cams = []
    for role in a.cams:
        c = cfg[role]
        if str(c["serial"]) not in connected:
            print(f"{role} camera {c['serial']} not connected — skipped", flush=True)
            continue
        cams.append(Cam(role, c["serial"], c["color_size"], c["depth_size"], c["fps"]))
        print(f"{role}: {cams[-1].name} {c['serial']} opened", flush=True)
    if not cams:
        raise SystemExit("no camera")

    half = {cam.role: 40 if cam.role == "top" else 70 for cam in cams}   # box half-size, colour px
    roi_c = {cam.role: None for cam in cams}                              # box centre, colour px
    status = {cam.role: "click the grey card" for cam in cams}
    busy = {"role": None}
    sel = {"role": cams[0].role}
    layout = []                                         # (role, x0, y0, scale, w, h) per tile

    def log(msg):
        print(msg, flush=True)

    def roi_of(role, shape):
        if roi_c[role] is None:
            return None
        (cx, cy), h = roi_c[role], half[role]
        x0, y0 = max(0, cx - h), max(0, cy - h)
        return (x0, y0, min(shape[1], cx + h) - x0, min(shape[0], cy + h) - y0)

    def on_mouse(ev, x, y, *_):
        if ev != cv2.EVENT_LBUTTONDOWN:
            return
        for role, x0, y0, s, w, h in layout:
            if x0 <= x < x0 + w and y0 <= y < y0 + h:
                roi_c[role] = (int((x - x0) / s), int((y - y0) / s))
                sel["role"] = role
                status[role] = "press c to calibrate"

    def run_cal(cam, roi):
        log(f"{cam.role}: calibrating (ROI {roi}) …")
        try:
            res = calibrate(cam, roi, log)
        except Exception as e:                         # keep the GUI alive, report the failure
            res = {"ok": False, "verdict": f"error: {e}"}
        if res["ok"]:
            save_profile(out, cam.role, res)
            status[cam.role] = (f"OK  exp {res['exposure_us'] / 1000:.1f} ms  WB {res['white_balance']} K  "
                                f"luma {res['after']['luma']:.0f}  saved")
        else:
            status[cam.role] = f"FAILED: {res['verdict']}"
        log(f"{cam.role}: {'OK' if res['ok'] else 'FAILED'} — {res['verdict']} "
            + (f"| exposure {res['exposure_us']} us, gain {res['gain']}, WB {res['white_balance']} K, "
               f"luma {res['before']['luma']:.0f}→{res['after']['luma']:.0f}, "
               f"BGR {res['after']['b']:.0f}/{res['after']['g']:.0f}/{res['after']['r']:.0f}"
               + (f" → {out.name}" if res["ok"] else "") if "after" in res else ""))
        busy["role"] = None

    cv2.namedWindow("grey card", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("grey card", on_mouse)
    while True:
        tiles = []
        layout.clear()
        x_off = 0
        for cam in cams:
            img = cam.latest()
            if img is None:
                img = np.zeros((360, TILE_W, 3), np.uint8)
            s = TILE_W / img.shape[1]
            view = cv2.resize(img, (TILE_W, int(img.shape[0] * s)))
            roi = roi_of(cam.role, img.shape)
            lines = [f"{cam.role}: {cam.name}" + ("  [selected]" if sel["role"] == cam.role else "")]
            colour = (0, 200, 255)
            if roi is not None and roi[2] > 0 and roi[3] > 0:
                r = gc.measure(img, roi)
                ok, why = r.usable
                colour = (0, 255, 0) if ok else (0, 0, 255)
                lines.append(f"luma {r.luma:5.1f} (118)  neutral {r.neutral_error_pct:4.1f}%  "
                             f"spread {r.spread_pct:4.1f}%  clip {r.clipped_pct:3.0f}%")
                lines.append("usable" if ok else f"not usable: {why}")
                x, y, w, h = roi
                cv2.rectangle(view, (int(x * s), int(y * s)), (int((x + w) * s), int((y + h) * s)), colour, 2)
            lines.append("CALIBRATING…" if busy["role"] == cam.role else status[cam.role])
            for i, t in enumerate(lines):
                yy = 22 + 22 * i
                cv2.putText(view, t, (8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(view, t, (8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            layout.append((cam.role, x_off, 0, s, view.shape[1], view.shape[0]))
            x_off += view.shape[1]
            tiles.append(view)
        hmax = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
        cv2.imshow("grey card", np.hstack(tiles))
        k = cv2.waitKey(30) & 0xFF
        if k in (27, ord("q")):
            break
        role = sel["role"]
        if k in (ord("+"), ord("=")):
            half[role] = min(half[role] + 10, 300)
        elif k == ord("-"):
            half[role] = max(half[role] - 10, 10)
        elif k == ord("c") and busy["role"] is None:
            cam = next(c for c in cams if c.role == role)
            img = cam.latest()
            roi = roi_of(role, img.shape) if img is not None else None
            if roi is None:
                status[role] = "click the card first"
                continue
            busy["role"] = role
            threading.Thread(target=run_cal, args=(cam, roi), daemon=True).start()
    for cam in cams:
        cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
