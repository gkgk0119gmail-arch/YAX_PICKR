"""Window that shows the live dashboard published by live_view.py (both RealSense views, cells, blocks, progress).
Started by pickr.py / demo.py; can also be opened by hand while a run is going:

  perception-main/ai_bubble/.venv/bin/python robot_piper/live_viewer.py

Demo mode (demo.py writes /dev/shm/pickr_demo_state.json): a big button on top -
  PREPARING (grey)  scan + plan running          START (green)  waiting for the judges: click to begin
  STOP (red)        running: click = the demo process is killed (the arm holds where it is)
F / F11 full screen, Esc leaves full screen, Q closes the window (the run goes on).
"""
import json
import os
import signal
import sys
import time
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/pickr_live.jpg")
STATE = Path("/dev/shm/pickr_demo_state.json")     # demo.py -> viewer: {"state": ..., "pid": ...}
CMD = Path("/dev/shm/pickr_demo_cmd")              # viewer -> demo.py: "start" / "stop"


def demo_state():
    try:
        st = json.loads(STATE.read_text())
        os.kill(int(st["pid"]), 0)                   # the demo process is still there
        return st
    except (OSError, ValueError, KeyError):
        return None


def main():
    root = tk.Tk()
    root.title("PICKR live dashboard")
    root.configure(bg="black")
    root.geometry("1600x960")                          # 16:9 dashboard (1920x1080) + the button bar
    bar = tk.Frame(root, bg="black")
    btn = tk.Button(bar, text="", font=("Sans", 22, "bold"), width=16, height=1, relief="flat", bd=0,
                    activeforeground="white", fg="white", bg="#444444")
    info = tk.Label(bar, text="", bg="black", fg="#aaaaaa", font=("Sans", 14))
    btn.pack(side="left", padx=16, pady=8)
    info.pack(side="left", padx=8)
    lab = tk.Label(root, bg="black", fg="white", text="waiting for the cameras...", font=("Sans", 18))
    lab.pack(side="bottom", fill="both", expand=True)
    st = dict(mtime=0.0, img=None, full=False, shown=False, mode=None)

    def full(on):
        st["full"] = on
        root.attributes("-fullscreen", on)
    root.bind("<f>", lambda e: full(not st["full"]))
    root.bind("<F11>", lambda e: full(not st["full"]))
    root.bind("<Escape>", lambda e: full(False))
    root.bind("<q>", lambda e: root.destroy())

    def click():
        ds = demo_state()
        if ds is None:
            return
        if ds["state"] == "ready":
            CMD.write_text("start")
        elif ds["state"] == "stopping":                 # going back to rest: HOLD = stop right there
            CMD.write_text("hold")
        elif ds["state"] == "running":
            CMD.write_text("stop")
            pid = int(ds["pid"])

            def kill(n=0):                              # the demo answers within 1.5 s ("stopping") and goes
                ds2 = demo_state()                      # back to rest; no answer (or 90 s gone): killed
                if ds2 is not None and ds2.get("state") == "stopping" and n < 90:
                    root.after(1000, kill, n + 1)
                    return
                try:
                    os.kill(pid, 0)
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
            root.after(1500, kill)
    btn.configure(command=click)

    def tick():
        ds = demo_state()
        if ds is not None and not st["shown"]:
            bar.pack(side="top", fill="x", before=lab)
            st["shown"] = True
        elif ds is None and st["shown"] and st["mode"] not in ("done", "stopped"):
            st["mode"] = "stopped"
            btn.configure(text="STOPPED", bg="#444444", state="disabled")
            info.configure(text="the demo process has ended")
        if ds is not None and ds["state"] != st["mode"]:
            st["mode"] = ds["state"]
            look = {"preparing": ("PREPARING...", "#555555", "disabled"),
                    "ready": ("START", "#1f9d55", "normal"),
                    "running": ("STOP", "#c0392b", "normal"),
                    "stopping": ("HOLD", "#8e44ad", "normal"),
                    "done": ("DONE", "#2d6cdf", "disabled")}.get(ds["state"], (ds["state"].upper(), "#555555", "disabled"))
            btn.configure(text=look[0], bg=look[1], activebackground=look[1], state=look[2], disabledforeground="white")
            info.configure(text=ds.get("note", ""))
        elif ds is not None:
            info.configure(text=ds.get("note", ""))
        try:
            m = SRC.stat().st_mtime
            if m != st["mtime"]:
                st["mtime"] = m
                im = Image.open(SRC)
                im.load()
                W, H = max(lab.winfo_width(), 100), max(lab.winfo_height(), 100)
                s = min(W / im.width, H / im.height)
                im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
                st["img"] = ImageTk.PhotoImage(im)
                lab.configure(image=st["img"], text="")
        except (OSError, ValueError):
            pass                                            # not written yet / being replaced
        root.after(60, tick)
    tick()
    root.mainloop()


if __name__ == "__main__":
    main()
