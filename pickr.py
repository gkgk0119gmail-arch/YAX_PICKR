#!/usr/bin/env python3
"""PICKR - one command for the whole pick & place: preflight checks, then fill basket cells.

  ./pickr.sh                    fill every empty cell (auto), asks once before the arm moves
  ./pickr.sh --cells 1 3        those cells (robot view: 1 far-left, 2 far-right, 3 near-left, 4 near-right)
  ./pickr.sh --dry-run          scan + plan + all safety checks, the arm does not move
  ./pickr.sh --yes              no confirmation prompt

The scene (basket, cells, blocks) is scanned ONCE at the start and remembered; cells are filled farthest from
the robot first and every placed block becomes an obstacle for the next plans.

Preflight: CAN up (asks for the password in a desktop window if needed), both cameras present, arm not in
teach mode (switched off when folded at rest), arm moved to the rest pose. Every round then plans straight
vertical grasps, checks every path, and checks with the wrist D405
before going down. A cell that cannot be done is skipped with the reason; a failed move stops everything.
"""
import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "robot_piper"), str(ROOT / "perception-main"), str(ROOT / "perception-main" / "research" / "box_cells")]
CFG = yaml.safe_load((ROOT / "pac2026" / "config.yaml").read_text())


def say(ok, msg):
    print(f"  [{'OK' if ok else '!!'}] {msg}", flush=True)
    return ok


def check_can():
    port = CFG["robot"]["can_port"]
    up = "UP" in subprocess.run(["ip", "-br", "link", "show", port], capture_output=True, text=True).stdout
    if up:
        return say(True, f"CAN {port} up")
    say(False, f"CAN {port} down - bringing it up (password window on the desktop)")
    subprocess.run(["pkexec", "sh", "-c", f"ip link set {port} down; ip link set {port} type can bitrate 1000000 "
                                          f"&& ip link set {port} up"])
    up = "UP" in subprocess.run(["ip", "-br", "link", "show", port], capture_output=True, text=True).stdout
    return say(up, f"CAN {port} {'up' if up else 'still down'}")


def check_cameras():
    import pyrealsense2 as rs
    have = {d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()}
    ok = True
    for role in ("top", "wrist"):
        s = str(CFG["camera"][role]["serial"])
        ok &= say(s in have, f"camera {role} ({s}) {'connected' if s in have else 'NOT FOUND'}")
    # USB speed: a RealSense on a USB 2 link (480 Mbps) stalls / times out with depth + colour
    for dev in Path("/sys/bus/usb/devices").glob("*"):
        try:
            if (dev / "idVendor").read_text().strip() == "8086":
                speed = int((dev / "speed").read_text().strip())
                name = (dev / "product").read_text().strip()
                ok &= say(speed >= 5000, f"{name} on USB port {dev.name}: {speed} Mbps"
                          + ("" if speed >= 5000 else " -> USB 2! plug it into a USB 3 port (blue), straight into the PC"))
        except (OSError, ValueError):
            pass
    return ok


def check_arm(dry):
    import hover_test as ht
    from board_hover import REST_Q, connect, enable, go_rest
    p = connect()
    time.sleep(0.3)
    _, js, st = ht.read(p)
    if not any(js):
        return say(False, "no feedback from the arm (power? CAN?)")
    js[1] = (js[1] + 180) % 360 - 180
    at_rest = np.abs(js - REST_Q).max() < 12
    say(True, f"arm: {st.ctrl_mode}, {st.arm_status}, joints {np.round(js, 1).tolist()}")
    if "TEACH" in str(st.ctrl_mode).upper():
        if not at_rest:
            return say(False, "teach mode and not folded at rest: fold the arm by hand, then run again")
        subprocess.run([sys.executable, str(ROOT / "robot_piper" / "teach_mode.py"), "off"], check=False)
        _, _, st = ht.read(p)
        if "TEACH" in str(st.ctrl_mode).upper():
            return say(False, "could not leave teach mode - press the teach button once")
        say(True, "teach mode switched off")
    if "NORMAL" not in str(st.arm_status).upper() and "EXCEED" not in str(st.arm_status).upper():
        return say(False, f"arm status {st.arm_status}")
    if not at_rest:
        if dry:
            return say(False, "arm not at rest (dry run does not move it)")
        enable(p)
        if not go_rest(p, 10):
            return say(False, "could not reach the rest pose")
        say(True, "moved to the rest pose")
    else:
        say(True, "arm at the rest pose")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="+", default=["auto"])
    ap.add_argument("--max", type=int, default=16)
    ap.add_argument("--speed", type=int, default=30, help="moves between poses, percent")
    ap.add_argument("--line-speed", type=int, default=40, help="straight up/down legs (streamed), percent")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--wait-hands", action="store_true", help="wait until no hand is in view before the scan")
    ap.add_argument("--collide", action="store_true", help="plan with 3D collision checks (off by default)")
    args = ap.parse_args()
    # models (Grounding DINO, SAM2, VGGT) load on the GPU while the preflight runs and the prompt waits
    loaded = {}

    def load():
        try:
            import fill_cells
            loaded["finder"] = fill_cells.load_finder()
        except BaseException as e:                       # re-raised in the main thread
            loaded["error"] = e
    loader = threading.Thread(target=load, daemon=True)
    loader.start()
    print("== PICKR preflight ==")
    if not (check_can() and check_cameras() and check_arm(args.dry_run)):
        sys.exit("preflight failed - nothing was started")
    if not args.dry_run and not args.yes:
        print(f"\nThe arm will move (speed {args.speed}%). Cells: {' '.join(args.cells)}. "
              "Keep hands out of the table area; e-stop in reach.")
        if input("Press Enter to start, anything else to cancel: ").strip():
            sys.exit("cancelled")
    if loader.is_alive():
        print("loading the perception models...", flush=True)
    loader.join()
    if "error" in loaded:
        raise loaded["error"]
    import fill_cells
    argv = ["--cells", *args.cells, "--max", str(args.max), "--speed", str(args.speed),
            "--line-speed", str(args.line_speed)]
    if not args.dry_run:
        argv.append("--go")
    if args.wait_hands:
        argv.append("--wait-hands")
    if args.collide:
        argv.append("--collide")
    results = fill_cells.main(argv, loaded["finder"])
    ok = sum(r["status"] in ("done", "planned") for r in results)
    print(f"\n{ok}/{len(results)} cell rounds {'planned' if args.dry_run else 'done'}; pictures in pac2026/runs/")


if __name__ == "__main__":
    main()
