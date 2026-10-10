"""Say "인사" -> the arm replays the taught `hello` motion (pac2026/teach/hello.json).

  ./voice_hello.sh                 listen on the ReSpeaker, run the motion on every "인사"
  ./voice_hello.sh --test          only print what was heard (the arm never moves)
  ./voice_hello.sh --vel 150       replay speed (deg/s), passed to teach_replay play

Offline Korean speech recognition (Vosk, models/vosk-model-small-ko-0.22), free dictation; the exact word 인사
(or 인사를) triggers. Not prefixes (background chatter decoded as "인사위" fired) and not a keyword-only grammar
(this model has no [unk], so every sound became "인사"). While the arm moves the microphone is ignored;
Ctrl+C stops listening (and a running motion holds where it is).
"""
import argparse
import json
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from vosk import KaldiRecognizer, Model, SetLogLevel

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "vosk-model-small-ko-0.22"
KEYWORDS = ["인사", "인사를"]          # exact words (both in the model vocabulary)
RATE = 16000


def find_mic(name):
    """ALSA card of the first capture device whose name contains `name` (arecord -l), and its channel count."""
    for line in subprocess.run(["arecord", "-l"], capture_output=True, text=True).stdout.splitlines():
        m = re.match(r"card \d+: (\w+) \[(.*?)\]", line)
        if m and name.lower() in line.lower():
            dev = f"plughw:CARD={m.group(1)},DEV=0"
            hw = subprocess.run(["arecord", "-D", dev.replace("plughw", "hw"), "--dump-hw-params", "-d", "1", "/dev/null"],
                                capture_output=True, text=True).stderr
            ch = re.search(r"CHANNELS:\s*\[?(\d+)", hw)
            return dev, m.group(2), int(ch.group(1)) if ch else 1
    sys.exit(f"no capture device matching '{name}' - is the ReSpeaker plugged in? (arecord -l)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true", help="only print detections, never move the arm")
    ap.add_argument("--motion", default="hello", help="teach file to replay")
    ap.add_argument("--vel", type=float, default=150.0, help="replay peak joint speed, deg/s")
    ap.add_argument("--mic", default="ReSpeaker", help="input device name (substring)")
    ap.add_argument("--channel", type=int, default=0, help="mic channel (ReSpeaker 6-ch firmware: 0 = processed)")
    ap.add_argument("--conf", type=float, default=0.6, help="minimum word confidence")
    args = ap.parse_args()

    SetLogLevel(-1)
    if not MODEL.exists():
        sys.exit(f"missing {MODEL} - download vosk-model-small-ko-0.22 from alphacephei.com/vosk/models")
    rec = KaldiRecognizer(Model(str(MODEL)), RATE)
    rec.SetWords(True)

    dev, label, ch = find_mic(args.mic)
    audio = queue.Queue()
    mic = subprocess.Popen(["arecord", "-q", "-D", dev, "-f", "S16_LE", "-r", str(RATE), "-c", str(ch), "-t", "raw"],
                           stdout=subprocess.PIPE)

    def reader():                                         # 0.25 s blocks, one channel
        n = 4000 * ch * 2
        while (buf := mic.stdout.read(n)):
            audio.put(np.frombuffer(buf, np.int16).reshape(-1, ch)[:, args.channel].tobytes())

    threading.Thread(target=reader, daemon=True).start()

    cmd = [str(ROOT / "teach.sh"), "play", args.motion, "--yes", "--vel", str(args.vel)]
    print(f"mic: {label} [{dev}] ({ch} ch, using ch {args.channel})", flush=True)
    print(f"listening for {KEYWORDS}" + (" - TEST mode, the arm will not move" if args.test else f" -> {' '.join(cmd[1:])}"), flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # killed -> still stop arecord (finally)
    try:
        while True:
            if not rec.AcceptWaveform(audio.get()):
                continue
            res = json.loads(rec.Result())
            if args.test and res.get("text"):
                print(f"   ({res['text']})", flush=True)
            words = [w for w in res.get("result", [])
                     if w["word"] in KEYWORDS and w["conf"] >= args.conf]
            if not words:
                continue
            print(f"[{time.strftime('%H:%M:%S')}] heard '{words[0]['word']}' (conf {words[0]['conf']:.2f})", flush=True)
            if args.test:
                continue
            r = subprocess.run(cmd)
            print(f"motion done (exit {r.returncode}) - listening again", flush=True)
            while not audio.empty():                  # drop what the mic heard while the arm moved
                audio.get_nowait()
            rec.Reset()
    except KeyboardInterrupt:
        print("\nstopped listening")
    finally:
        mic.terminate()


if __name__ == "__main__":
    main()
