"""Debug log of one ./pickr.sh run, written automatically to pac2026/runs/logs/pickr_<time>/:

  console.log    everything printed (stdout + stderr, also tracebacks / Ctrl-C), each line time-stamped
  events.jsonl   one JSON line per event: every arm move (start / target / reached joints, the joint trajectory
                 at ~10 Hz, closest approach to the rest pose, who called it), rest moves, gripper, D405 and
                 top-camera checks
  meta.json      command line, git commit + uncommitted files, config.yaml

Without start() every call here does nothing (the robot scripts can still be run on their own).
"""
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_dir = None
_ev = None
_t0 = time.time()


class _Tee:
    def __init__(self, stream, f):
        self.stream, self.f, self.bol = stream, f, True

    def write(self, s):
        self.stream.write(s)
        for part in s.splitlines(keepends=True):
            if self.bol:
                self.f.write(time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d} ")
            self.f.write(part)
            self.bol = part.endswith("\n")
        self.f.flush()
        return len(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()

    def __getattr__(self, k):
        return getattr(self.stream, k)


def _plain(v):
    if isinstance(v, np.ndarray):
        return np.round(v.astype(float), 3).tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items() if not isinstance(x, np.ndarray) or x.size < 200}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def start(argv):
    """Open the run folder and tee the console into it. Returns the folder."""
    global _dir, _ev, _t0
    _t0 = time.time()
    _dir = ROOT / "pac2026" / "runs" / "logs" / f"pickr_{time.strftime('%Y%m%d_%H%M%S')}"
    _dir.mkdir(parents=True, exist_ok=True)
    con = open(_dir / "console.log", "a", buffering=1)
    sys.stdout, sys.stderr = _Tee(sys.stdout, con), _Tee(sys.stderr, con)
    _ev = open(_dir / "events.jsonl", "a", buffering=1)

    def git(*a):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            return ""
    meta = dict(argv=list(argv), start=time.strftime("%Y-%m-%d %H:%M:%S"), commit=git("rev-parse", "--short", "HEAD"),
                branch=git("rev-parse", "--abbrev-ref", "HEAD"), uncommitted=git("status", "--short").splitlines())
    try:
        meta["config"] = (ROOT / "pac2026" / "config.yaml").read_text()
    except OSError:
        pass
    (_dir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"[log] debug log: {_dir}")
    return _dir


def folder():
    return _dir


def event(kind, **data):
    """One JSON line; 'caller' = the robot code that asked for it (file:line function)."""
    if _ev is None:
        return
    st = [f for f in traceback.extract_stack()[:-1] if "runlog.py" not in f.filename and f.name != "_log_move"][-4:]
    rec = dict(t=round(time.time() - _t0, 3), clock=time.strftime("%H:%M:%S"), kind=kind,
               caller=[f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in st], **_plain(data))
    try:
        _ev.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:                                  # logging must never stop the arm
        sys.__stderr__.write(f"[log] event not written: {e}\n")
