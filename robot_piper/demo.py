"""Demo mode for the judges: the three modules in one process, one after the other.

  ./demo.sh                       (live window with the START / STOP button opens by itself)

  0  preflight once (CAN, cameras, arm at rest / teach mode off)
  1  partition: scan (Qwen check, VGGT) and plan right away, then WAIT for START in the live window;
     START -> the partition goes into the basket -> rest (no check afterwards: assumed done)
  -  5 s at rest: the team puts the blocks where the partition was
  2  blocks: fill_cells as in pickr.sh (scan, fill in the team order, empty cells left -> scan again and refill,
     at most fill_cells.RESCANS = 2 times) -> rest
  3  drag: grip the middle of the basket's right wall and pull it 30 cm to the right -> rest
  4  finale: the taught `clap` motion (pac2026/teach/clap.json) -> rest; if it fails the demo still counts as done
  STOP in the live window at any time: the arm holds where it is and this process ends at once.
A step that fails stops the demo there (the arm may not be at rest - it holds its pose).
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "robot_piper"), str(ROOT / "perception-main"),
                str(ROOT / "perception-main" / "research" / "box_cells"), str(ROOT)]
RESCANS = 1                    # demo: one recovery round for empty cells (fill_cells default 2)
STATE = Path("/dev/shm/pickr_demo_state.json")     # -> live_viewer.py (button)
CMD = Path("/dev/shm/pickr_demo_cmd")              # <- live_viewer.py ("start" / "stop")
_arm = {}


def set_state(state, note=""):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(state=state, pid=os.getpid(), note=note)))
    os.replace(tmp, STATE)


def hold_and_exit(why):
    """STOP: command the joints where they are (the arm holds), then end this process at once."""
    p = _arm.get("p")
    if p is not None:
        try:
            j = p.GetArmJointMsgs().joint_state
            q = [j.joint_1, j.joint_2, j.joint_3, j.joint_4, j.joint_5, j.joint_6]
            for _ in range(10):
                p.MotionCtrl_2(0x01, 0x01, 10, 0x00)
                p.JointCtrl(*[int(v) for v in q])
                time.sleep(0.01)
        except Exception:
            pass
    print(f"\n[demo] STOP: {why} - the arm holds where it is", flush=True)
    try:
        import runlog
        runlog.event("demo_stop", why=why)
        set_state("stopped", why)
    except Exception:
        pass
    os._exit(2)


def watch_stop():
    while True:
        try:
            if CMD.read_text().strip() == "stop":
                hold_and_exit("STOP button")
        except OSError:
            pass
        time.sleep(0.1)


def wait_start():
    while True:
        try:
            if CMD.read_text().strip() == "start":
                CMD.unlink()
                return
        except OSError:
            pass
        time.sleep(0.1)


class LiveCams:
    """Top D435i + wrist D405 open for a whole step: the live view shows both, scans take frames from them
    (no reopening). fill_cells opens its own cameras, so these are closed during the block step."""

    def __init__(self):
        import gdino_vggt_lines  # noqa: F401  (pac2026 on the path)
        import live_view
        from pac.cameras import RealSenseCam
        from pick_place import WristSeen
        cfg = yaml.safe_load((ROOT / "pac2026" / "config.yaml").read_text())["camera"]
        prof_path = ROOT / "pac2026" / cfg.get("profile", "camera_profile.yaml")
        prof = (yaml.safe_load(prof_path.read_text()) or {}).get("top") if prof_path.exists() else None
        c = cfg["top"]
        self.top = RealSenseCam(c["serial"], c["color_size"], c["depth_size"], c["fps"], cfg["depth_frames"], prof)
        live_view.set_source("top", self.top.latest, self.top.intrinsics)
        try:
            self.wrist = WristSeen()
            live_view.set_source("wrist", self.wrist.latest)
        except Exception as e:
            self.wrist = None
            print(f"[view] wrist D405 not shown: {e}")
        self.t0 = time.time()

    def grab(self):
        time.sleep(max(0.0, 1.0 - (time.time() - self.t0)))   # exposure settles after opening
        f = self.top.grab(with_depth=True)
        return f.color, f.depth_m, f.intrinsics

    def close(self):
        import live_view
        live_view.set_source("top", None)
        live_view.set_source("wrist", None)
        for c in (self.top, self.wrist):
            try:
                if c is not None:
                    c.close()
            except Exception:
                pass


def fail(step, why):
    import live_view
    msg = f"step {step} failed: {why}"
    print(f"[demo] {msg} - demo stopped here", flush=True)
    live_view.status(msg)
    set_state("done", msg)
    time.sleep(3)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-qwen", action="store_true", help="skip the Qwen check of the partition scan")
    ap.add_argument("--wait-s", type=float, default=5.0, help="seconds at rest before the block step")
    ap.add_argument("--no-view", action="store_true")
    args = ap.parse_args()
    import ik_pool
    ik_pool.start()                                       # planner IK workers: forked before any other thread
    import runlog
    runlog.start(sys.argv)
    import live_view
    live_view.start()
    live_view.stage(0)
    live_view.status("preparing the demo")
    try:
        CMD.unlink()
    except OSError:
        pass
    set_state("preparing", "preflight")
    if not args.no_view:
        try:
            subprocess.Popen([sys.executable, str(ROOT / "robot_piper" / "live_viewer.py")],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError as e:
            print(f"[view] live window not started: {e}")
    threading.Thread(target=watch_stop, daemon=True).start()
    loaded = {}

    def load():                                           # models load while the preflight runs
        try:
            import warnings
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error()
            hf_logging.disable_progress_bar()
            warnings.filterwarnings("ignore", category=FutureWarning)
            import fill_cells
            loaded["finder"] = fill_cells.load_finder()   # Grounding DINO + SAM2 + VGGT, shared by all steps
        except BaseException as e:
            loaded["error"] = e
    loader = threading.Thread(target=load, daemon=True)
    loader.start()

    # 0 ---------------------------------------------------------------- preflight (once)
    import pickr
    print("== DEMO preflight ==", flush=True)
    if not (pickr.check_can() and pickr.check_cameras() and pickr.check_arm(False)):
        fail(0, "preflight (CAN / cameras / arm)")
    from board_hover import connect
    p = connect()
    _arm["p"] = p
    live_view.set_arm(p)
    tcp = yaml.safe_load((ROOT / "pac2026" / "config.yaml").read_text())["robot"]["tcp_offset_m"]
    out = ROOT / "pac2026" / "runs" / f"demo_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)

    # 1 ---------------------------------------------------------------- partition: scan + plan now, move on START
    import move_partition as mp
    import partition_scene as ps
    live_view.stage(1)
    live_view.status("partition: scanning")
    set_state("preparing", "partition: scanning and planning")
    mpargs = mp.parser().parse_args([])
    cams = LiveCams()                                     # live view during the partition step
    color, depth, K = cams.grab()
    cv2.imwrite(str(out / "partition_color.jpg"), color)
    if loader.is_alive():
        live_view.status("loading the perception models")
    loader.join()
    if "error" in loaded:
        fail(1, f"loading the models: {loaded['error']}")
    finder = loaded["finder"]
    import fill_cells
    fill_cells.RESCANS = RESCANS
    ps._GD["proc"], ps._GD["model"] = finder.gd_proc, finder.gd
    qb = None
    if not args.no_qwen:                                  # Qwen on the GPU with VGGT parked on the CPU
        import qwen_check
        live_view.status("partition: Qwen check")
        qb = qwen_check.qwen_boxes(color, finder)
        print(f"[qwen] box {np.round(qb['box']).tolist() if 'box' in qb else '-'} px, partition "
              f"{np.round(qb['partition']).tolist() if 'partition' in qb else '-'} px", flush=True)
    live_view.status("partition: VGGT scan")
    try:
        sc = ps.scan(color, depth, K, part_h=mpargs.part_height)
        g = mp.plan(sc, tcp, sc["box"]["floor_z"], 1.0)
    except (RuntimeError, mp.PlanFail) as e:
        cv2.imwrite(str(out / "partition_plan.png"), color)
        fail(1, f"partition scan / plan: {e}")
    img = ps.draw(color, dict(sc, box=g["box_eff"]), g)
    if qb is not None:
        qok, qrep, _ = qwen_check.compare(qb, sc)
        qwen_check.draw(img, qb)
        print(f"[qwen] overlap with the measurement (IoU, need {qwen_check.MIN_IOU}): {qrep}", flush=True)
        cv2.imwrite(str(out / "partition_plan.png"), img)
        if not qok:
            fail(1, f"Qwen does not agree with the scan {qrep}")
    cv2.imwrite(str(out / "partition_plan.png"), img)
    print(f"[plan] grip the {g['along']} divider at ({g['xy'][0] * 1000:.0f}, {g['xy'][1] * 1000:.0f}) mm, "
          f"tilt {g['beta']} deg; release at ({g['xy_place'][0] * 1000:.0f}, {g['xy_place'][1] * 1000:.0f}); "
          f"picture {out / 'partition_plan.png'}", flush=True)

    live_view.status("READY - press START")
    set_state("ready", "partition planned - press START when the judges are here")
    print("[demo] READY - waiting for START in the live window", flush=True)
    runlog.event("demo_ready")
    wait_start()
    set_state("running", "step 1: partition")
    runlog.event("demo_start")
    print("[demo] START", flush=True)
    t_start = time.time()
    try:
        mp.run(p, sc, g, mpargs, tcp)                     # ends at the rest pose
    except SystemExit as e:
        fail(1, f"partition move: {e}")

    cams.close()                                          # fill_cells opens the cameras itself

    # -  ---------------------------------------------------------------- the team places the blocks
    live_view.stage(2)
    for i in range(int(round(args.wait_s)), 0, -1):
        live_view.status(f"place the blocks: {i} s")
        set_state("running", f"place the blocks ({i} s)")
        time.sleep(1)

    # 2 ---------------------------------------------------------------- blocks into the cells (+ up to 2 rescans)
    set_state("running", "step 2: blocks")
    try:
        results = fill_cells.main(["--go"], finder)       # ends at rest; empty cells -> rescan, at most RESCANS
    except SystemExit as e:
        fail(2, f"blocks: {e}")
    done = sum(r["status"] == "done" for r in results)
    print(f"[demo] blocks: {done} cell(s) filled", flush=True)
    if any(r["status"] == "HALT" for r in results):
        fail(2, "blocks: halted (the arm holds its pose)")

    # 3 ---------------------------------------------------------------- drag the basket to the right
    live_view.stage(3)
    set_state("running", "step 3: drag the basket")
    live_view.status("drag the basket: scanning")
    import drag_box
    cams = LiveCams()                                     # live view during the drag
    try:
        drag_box.main([], p=p, grab=cams.grab)            # right wall, 30 cm, ends at rest
    except SystemExit as e:
        fail(3, f"drag: {e}")
    finally:
        cams.close()

    # 4 ---------------------------------------------------------------- finale: clap (decoration, never fails the demo)
    set_state("running", "finale: clap")
    live_view.status("finale: clap")
    import teach_replay
    try:
        teach_replay.replay(p, "clap", vel=150.0)
    except (SystemExit, Exception) as e:
        print(f"[demo] clap skipped: {e}", flush=True)
        runlog.event("demo_clap_failed", why=str(e))

    live_view.stage(4)
    secs = time.time() - t_start
    live_view.status(f"demo finished in {secs:.0f} s")
    set_state("done", f"finished in {secs:.0f} s")
    runlog.event("demo_done", secs=secs)
    print(f"\n[demo] finished in {secs:.0f} s; debug log {runlog.folder()}", flush=True)
    time.sleep(5)                                         # the dashboard shows the end a moment longer


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        hold_and_exit("Ctrl+C")
