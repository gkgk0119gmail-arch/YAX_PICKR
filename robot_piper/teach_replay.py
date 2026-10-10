"""Teach by hand, replay by interpolation.

  ./teach.sh record NAME     drag-teach mode: move the arm by hand and press keys (no Enter needed)
        c  record a waypoint (the 6 joint angles + gripper opening now)
        g  close the gripper (recorded as a step)      o  open the gripper (recorded as a step)
        u  undo the last step                          l  list the steps
        the gripper is opened / closed BY HAND too; every waypoint keeps its opening
        q  save and finish (at the rest pose it also leaves teach mode - that resets the arm, power drops briefly)
  ./teach.sh play NAME [--vel 20] [--dry-run]
        preflight (CAN, cameras, teach mode off, rest pose), then at once (--ask: wait for Enter): rest -> waypoints -> rest.
        Each leg is a minimum-jerk joint interpolation (smooth start / stop) streamed at 50 Hz, then the arm
        settles exactly on the waypoint; then the gripper: taught wider than now -> opens fully, narrower ->
        closes with full force (stops on the plate). Extra g / o steps run where they were recorded.
        Ctrl+C: the arm holds where it is.
  ./teach.sh list

Files: pac2026/teach/NAME.json
"""
import argparse
import json
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "robot_piper"), str(ROOT)]
DIR = ROOT / "pac2026" / "teach"
REST_Q = np.array([0.0, 0.0, 0.0, 0.0, 25.0, 0.0])
REST_TOL = 12.0
OPEN_MM = 100.0               # full stroke of the on-site gripper (max_range_config 100 mm)
CLOSE_MM = 0.0
ARM_LIM = np.array([[-150, 150], [0, 180], [-170, 0], [-100, 100], [-70, 70], [-170, 170]], float)  # arm, deg
GRIP_TOL = 5.0                # replay: a taught opening this much wider / narrower than now -> open / close


def joints(p):
    j = p.GetArmJointMsgs().joint_state
    q = np.array([j.joint_1, j.joint_2, j.joint_3, j.joint_4, j.joint_5, j.joint_6]) / 1000.0
    q[1] = (q[1] + 180) % 360 - 180                     # joint 2 reads a full turn off in teach mode
    return q


def grip_mm(p):
    return p.GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0


def in_teach(p):
    return "TEACH" in str(p.GetArmStatus().arm_status.ctrl_mode).upper()


def connect():
    from piper_sdk import C_PiperInterface_V2
    p = C_PiperInterface_V2("can0")
    p.ConnectPort()
    time.sleep(0.4)
    return p


def gripper_cmd(p, mm, effort, secs=3.0):
    """Drive the gripper to mm with the given effort until it gets there or stops moving (closed on something),
    at most secs; keeps commanding a moment longer so the force is fully applied. Returns the opening (mm)."""
    t0, hist = time.time(), []
    while time.time() - t0 < secs:
        p.GripperCtrl(int(round(mm * 1000)), effort, 0x01, 0)
        time.sleep(0.02)
        hist.append(grip_mm(p))
        there = abs(hist[-1] - mm) < 1.0
        stalled = len(hist) > 25 and np.ptp(hist[-25:]) < 0.3      # no motion for 0.5 s
        if there or stalled:
            break
    for _ in range(25):                                   # hold the command 0.5 s more (full force)
        p.GripperCtrl(int(round(mm * 1000)), effort, 0x01, 0)
        time.sleep(0.02)
    return grip_mm(p)


def clap(p, times, open_mm, period, effort):
    """Open / close the gripper on a fixed beat (period s per clap), without waiting for it to arrive."""
    for _ in range(times):
        for mm in (open_mm, CLOSE_MM):
            t_end = time.time() + period / 2
            while time.time() < t_end:
                p.GripperCtrl(int(round(mm * 1000)), effort, 0x01, 0)
                time.sleep(0.02)
    return gripper_cmd(p, CLOSE_MM, effort, secs=1.0)


# ------------------------------------------------------------------ record
def leave_teach(p):
    for _ in range(5):
        p.MotionCtrl_1(0x00, 0x00, 0x02)                  # end teach record
        time.sleep(0.05)
    time.sleep(0.3)
    for _ in range(3):
        p.MotionCtrl_1(0x02, 0x00, 0x00)                  # reset: leaves teach mode (power drops briefly)
        time.sleep(0.05)
    time.sleep(1.5)


def record(name, effort):
    """Drag-teach: the arm AND the gripper are moved by hand; each waypoint keeps the gripper opening too."""
    p = connect()
    if not in_teach(p):
        for _ in range(5):
            p.MotionCtrl_1(0x00, 0x00, 0x01)              # start drag-teach
            time.sleep(0.05)
        time.sleep(0.5)
    print(f"teach mode {'ON' if in_teach(p) else 'NOT on?'}: move the arm and open / close the gripper by hand.\n"
          "  c = waypoint (joints + gripper opening)   g / o = extra close / open step   u = undo   l = list\n"
          "  q = save + finish (at the rest pose it also leaves teach mode)   Ctrl+C = save + stop", flush=True)
    steps = []
    out = DIR / f"{name}.json"
    DIR.mkdir(parents=True, exist_ok=True)

    def save():
        out.write_text(json.dumps(dict(name=name, recorded=time.strftime("%Y-%m-%d %H:%M:%S"), steps=steps), indent=1))

    def show():
        for i, s in enumerate(steps, 1):
            print(f"   {i:2d}. " + (f"waypoint {np.round(s['q'], 1).tolist()}  gripper {s['grip_mm']:.0f} mm"
                                    if s["type"] == "waypoint" else f"gripper {s['action']}"), flush=True)

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    last = 0.0
    try:
        while True:
            if time.time() - last > 0.25:                 # live joint read-out on one line
                q = joints(p)
                print(f"\r   joints {np.round(q, 1).tolist()}  gripper {grip_mm(p):.0f} mm  steps {len(steps)}   ",
                      end="", flush=True)
                last = time.time()
            if not select.select([sys.stdin], [], [], 0.05)[0]:
                continue
            k = sys.stdin.read(1).lower()
            print()
            if k == "c":
                q = joints(p)
                steps.append(dict(type="waypoint", q=q.round(2).tolist(), grip_mm=round(grip_mm(p), 1)))
                print(f"[{len(steps)}] waypoint {np.round(q, 1).tolist()}", flush=True)
                save()
            elif k in ("g", "o"):
                action = "close" if k == "g" else "open"
                steps.append(dict(type="gripper", action=action))
                print(f"[{len(steps)}] gripper {action} - recorded, runs here during the replay", flush=True)
                save()
            elif k == "u" and steps:
                print(f"undo: {steps.pop()}", flush=True)
                save()
            elif k == "l":
                show()
            elif k == "q":
                q = joints(p)
                save()
                if np.abs(q - REST_Q).max() <= REST_TOL:
                    leave_teach(p)                        # safe only folded at rest (the reset drops the arm)
                else:
                    print("not at the rest pose: staying in teach mode (fold the arm before the replay)", flush=True)
                print(f"saved {len(steps)} step(s) -> {out}", flush=True)
                show()
                return
    except KeyboardInterrupt:
        save()
        print(f"\nCtrl+C: saved {len(steps)} step(s) -> {out} (the arm is still in teach mode)", flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ------------------------------------------------------------------ play
def min_jerk(t):
    return 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5


def settle(p, q, speed, timeout=8.0):
    """Keep commanding q until the arm is within 0.5 deg, or has stopped (0.5 s) within 3 deg (a taught pose
    a hair beyond a joint limit, gravity on a long reach). Returns (ok, max error)."""
    qi = (np.asarray(q) * 1000).round().astype(int)
    t0, hist = time.time(), []
    err = np.inf
    while time.time() - t0 < timeout:
        p.MotionCtrl_2(0x01, 0x01, speed, 0x00)
        p.JointCtrl(*[int(v) for v in qi])
        js = joints(p)
        hist.append(js)
        err = float(np.abs(js - q).max())
        still = len(hist) > 25 and np.ptp(np.array(hist[-25:]), 0).max() < 0.15
        if err < 0.5 or (still and err < 3.0):
            return True, err
        time.sleep(0.02)
    return False, err


def leg(p, a, b, vel, track_speed, name):
    """Stream a -> b (deg) on a minimum-jerk profile, peak joint speed ~vel deg/s, then settle on b."""
    a = np.asarray(a, float)
    b = np.clip(np.asarray(b, float), ARM_LIM[:, 0], ARM_LIM[:, 1])   # taught a hair past a limit
    dq = float(np.abs(b - a).max())
    T = max(0.6, 1.875 * dq / vel)                        # min-jerk peak speed = 1.875 * mean speed
    n = max(2, int(T * 50))
    t0 = time.time()
    for i in range(1, n + 1):
        q = a + (b - a) * min_jerk(i / n)
        qi = (q * 1000).round().astype(int)
        p.MotionCtrl_2(0x01, 0x01, track_speed, 0x00)
        p.JointCtrl(*[int(v) for v in qi])
        time.sleep(max(0.0, t0 + i * T / n - time.time()))
    ok, err = settle(p, b, track_speed)
    print(f"   {name}: {'ok' if ok else 'NOT reached'} ({err:.2f} deg, {T:.1f} s)", flush=True)
    import runlog
    runlog.event("teach_leg", name=name, q_from=a, q_to=b, ok=ok, err=err, secs=T, q_end=joints(p))
    if not ok:
        raise SystemExit(f"{name}: not reached (error {err:.1f} deg, {p.GetArmStatus().arm_status.arm_status}) "
                         "- the arm holds its pose")
    return b


def run_steps(p, steps, vel, track_speed=50, effort=5000, back_to_rest=True):
    """Replay taught steps on a connected, enabled arm, from wherever it is now. A leg that is not reached
    raises SystemExit (the arm holds its pose)."""
    q_now = joints(p)
    for i, s in enumerate(steps, 1):
        if s["type"] == "waypoint":
            q_now = leg(p, q_now, s["q"], vel, track_speed, f"step {i}: waypoint")
            w_rec, w_now = s.get("grip_mm"), grip_mm(p)  # gripper as it was taught here: wider -> open fully,
            if w_rec is not None and abs(w_rec - w_now) > GRIP_TOL:   # narrower -> close with full force
                act = "open" if w_rec > w_now else "close"
                w = gripper_cmd(p, OPEN_MM if act == "open" else CLOSE_MM, effort)
                print(f"   step {i}: gripper {act} (taught {w_rec:.0f} mm) -> {w:.0f} mm", flush=True)
        elif s["type"] == "clap":
            w = clap(p, s["times"], s["open_mm"], s["period"], effort)
            print(f"   step {i}: clap x{s['times']} -> {w:.0f} mm", flush=True)
        else:
            w = gripper_cmd(p, CLOSE_MM if s["action"] == "close" else OPEN_MM, effort)
            print(f"   step {i}: gripper {s['action']} -> {w:.0f} mm", flush=True)
    if back_to_rest:
        leg(p, q_now, REST_Q, vel, track_speed, "back to rest")


def replay(p, name, vel=20.0, **kw):
    """Replay pac2026/teach/NAME.json on an already connected arm (used by demo.py, no preflight / prompt)."""
    run_steps(p, json.loads((DIR / f"{name}.json").read_text())["steps"], vel, **kw)


def play(name, args):
    data = json.loads((DIR / f"{name}.json").read_text())
    steps = data["steps"]
    wps = [s for s in steps if s["type"] == "waypoint"]
    print(f"{name}: {len(wps)} waypoint(s), {len(steps) - len(wps)} gripper step(s), recorded {data.get('recorded')}")
    q_prev = REST_Q
    total = 0.0
    for s in steps:
        if s["type"] == "waypoint":
            dq = float(np.abs(np.asarray(s["q"]) - q_prev).max())
            total += max(0.6, 1.875 * dq / args.vel)
            q_prev = np.asarray(s["q"])
    total += max(0.6, 1.875 * float(np.abs(REST_Q - q_prev).max()) / args.vel)
    print(f"about {total:.0f} s of motion at peak {args.vel:.0f} deg/s (+ settling, gripper)")
    if args.dry_run:
        for i, s in enumerate(steps, 1):
            print(f"   {i:2d}. " + (f"waypoint {np.round(s['q'], 1).tolist()}  gripper {s.get('grip_mm', 0):.0f} mm"
                                    if s["type"] == "waypoint" else
                                    f"clap x{s['times']} ({s['open_mm']:.0f} mm, {s['period']:.2f} s)"
                                    if s["type"] == "clap" else f"gripper {s['action']}"))
        return
    import pickr
    if not (pickr.check_can() and pickr.check_cameras() and pickr.check_arm(False)):
        sys.exit("preflight failed - nothing was started")
    if args.ask and input("Press Enter to replay, anything else to cancel: ").strip():
        sys.exit("cancelled")
    import runlog
    runlog.start(sys.argv)
    from board_hover import enable
    p = connect()
    enable(p)
    try:
        run_steps(p, steps, args.vel, args.track_speed, args.effort, not args.no_return)
    except KeyboardInterrupt:
        q = joints(p)
        for _ in range(20):                               # hold right here
            p.MotionCtrl_2(0x01, 0x01, 10, 0x00)
            p.JointCtrl(*[int(v) for v in (q * 1000).round().astype(int)])
            time.sleep(0.02)
        sys.exit(f"\n[stop] Ctrl+C - the arm holds at {np.round(q, 1).tolist()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["record", "play", "list"])
    ap.add_argument("name", nargs="?", default="partition")
    ap.add_argument("--vel", type=float, default=20.0, help="play: peak joint speed, deg/s")
    ap.add_argument("--track-speed", type=int, default=50, help="play: controller speed %% while streaming")
    ap.add_argument("--effort", type=int, default=5000, help="gripper effort 0..5000")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ask", action="store_true", help="play: wait for Enter after the preflight")
    ap.add_argument("--yes", action="store_true", help="(default now; kept for old commands)")
    ap.add_argument("--no-return", action="store_true", help="play: stay at the last waypoint")
    args = ap.parse_args()
    if args.cmd == "list":
        for f in sorted(DIR.glob("*.json")):
            d = json.loads(f.read_text())
            print(f"{f.stem:20s} {len(d['steps'])} steps, recorded {d.get('recorded')}")
    elif args.cmd == "record":
        record(args.name, args.effort)
    else:
        play(args.name, args)


if __name__ == "__main__":
    main()
