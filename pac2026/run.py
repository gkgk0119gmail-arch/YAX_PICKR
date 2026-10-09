#!/usr/bin/env python3
"""
One entry point for the whole pick & place pipeline, with a live monitoring dashboard.

  python run.py                           live cameras, robot from config (mock by default)
  python run.py --robot piper             real PiPER (CAN must be up)
  python run.py --robot sim               MuJoCo twin (start ./sim_tools/run.sh twin_server.py first)
  python run.py --image img.jpg [--depth depth.raw]     offline: run on a saved frame
  python run.py --step scan_box|partition|blocks        run only up to that vision step
  dashboard: http://localhost:8800  (--port, --no-dashboard; stays open after the run until Ctrl+C)

Run with the perception venv:
  HF_HUB_OFFLINE=1 ../perception-main/ai_bubble/.venv/bin/python run.py --image ...
"""
import argparse
import socket
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pac import perception as P                      # noqa: E402
from pac.calib import PixelToRobot                    # noqa: E402
from pac.cameras import CameraManager                 # noqa: E402
from pac.dashboard import Dashboard                   # noqa: E402
from pac.models import ModelHub                       # noqa: E402
from pac.pipeline import Pipeline                     # noqa: E402
from pac.robot import make_robot                      # noqa: E402
from pac.runlog import RunLog                         # noqa: E402


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "localhost"


def main():
    ap = argparse.ArgumentParser(description="PAC 2026 partition pick & place")
    ap.add_argument("--config", default=HERE / "config.yaml", type=Path)
    ap.add_argument("--robot", choices=["mock", "piper", "sim"])
    ap.add_argument("--image", type=Path, help="offline colour image instead of the live camera")
    ap.add_argument("--depth", type=Path, help="offline raw Z16 depth for --image")
    ap.add_argument("--step", choices=["scan_box", "partition", "blocks", "all"], default="all")
    ap.add_argument("--max-blocks", type=int)
    ap.add_argument("--mock-fail-rate", type=float, help="mock robot: chance a grasp misses (tests retries)")
    ap.add_argument("--sim-host", help="--robot sim: twin server host (default from config)")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("--no-hold", action="store_true", help="exit right after the run (dashboard closes)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.robot:
        cfg["robot"]["type"] = args.robot
    if args.sim_host:
        cfg["robot"]["sim_host"] = args.sim_host
    if args.mock_fail_rate is not None:
        cfg["robot"]["mock_fail_rate"] = args.mock_fail_rate
    max_blocks = args.max_blocks or cfg["run"]["max_blocks"]

    runs = HERE / cfg["run"]["log_dir"]
    dash = None
    if not args.no_dashboard:
        dash = Dashboard(runs, args.port)
        dash.serve()
        print(f"dashboard: http://localhost:{args.port}   (LAN: http://{lan_ip()}:{args.port})", flush=True)
    log = RunLog(runs, dash)
    cam_name = f"file {args.image.name}" if args.image else "live D435i"
    log.info(f"config {args.config.name}, robot={cfg['robot']['type']}, camera={cam_name}")

    cams = CameraManager(cfg, image=args.image, depth=args.depth)
    if dash:
        dash.frame_source = cams.latest
        dash.set_info(robot=cfg["robot"]["type"], camera=cam_name, max_blocks=max_blocks)
    hub = ModelHub(cfg, HERE)
    hub.warmup(log)
    first = cams.top(with_depth=False)
    calib = PixelToRobot.load_or_mock(HERE / cfg["calibration"]["file"], cfg["calibration"]["table_z_robot_m"],
                                      first.color.shape)
    if calib.mock:
        log.warn("calibration: MOCK pixel->robot mapping (no calib file) - robot positions are not real")
    robot = make_robot(cfg, log)
    pipe = Pipeline(cfg, cams, hub, robot, calib, log)

    status = "done"
    try:
        if args.step == "all":
            result = pipe.run(max_blocks)
            if result.get("stopped") == "dashboard stop":
                status = "stopped"
        else:
            f, box = pipe.step1_scan_box()
            result = {"box_state": box.state, "height_m": box.height_m}
            if args.step in ("partition", "blocks") and box.state == "partition_in_box":
                part = pipe.step2_partition(f, box)
                result["grid"] = f"{part.rows}x{part.cols}"
                result["cells"] = {c.id: c.state for c in part.cells}
            if args.step == "blocks":
                result["blocks"] = len(P.find_blocks(f, box, hub, cfg, log))
        log.json("result", result)
        log.info(f"result: {result}")
        log.info(f"log folder: {log.dir}")
    except KeyboardInterrupt:
        status = "stopped"
        log.info("interrupted - stopping robot", "error")
        robot.stop()
    except Exception as e:  # noqa: BLE001  - keep the dashboard up to show what went wrong
        status = "error"
        log.info(f"ERROR: {type(e).__name__}: {e}", "error")
        robot.stop()
        raise
    finally:
        if dash:
            dash.set_status(status)
            if not args.no_hold:
                print("run finished - dashboard stays open, Ctrl+C to quit", flush=True)
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
        cams.close()


if __name__ == "__main__":
    main()
