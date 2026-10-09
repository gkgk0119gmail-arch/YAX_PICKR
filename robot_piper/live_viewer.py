"""Window that shows the live dashboard published by live_view.py (both RealSense views, cells, blocks, progress).
Started by pickr.py; can also be opened by hand while a run is going:

  perception-main/ai_bubble/.venv/bin/python robot_piper/live_viewer.py

F / F11 full screen, Esc leaves full screen, Q closes the window (the run goes on).
"""
import sys
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/pickr_live.jpg")


def main():
    root = tk.Tk()
    root.title("PICKR live dashboard")
    root.configure(bg="black")
    root.geometry("1600x900")                          # 16:9, the dashboard is 1920x1080
    lab = tk.Label(root, bg="black", fg="white", text="waiting for the cameras...", font=("Sans", 18))
    lab.pack(fill="both", expand=True)
    st = dict(mtime=0.0, img=None, full=False)

    def full(on):
        st["full"] = on
        root.attributes("-fullscreen", on)
    root.bind("<f>", lambda e: full(not st["full"]))
    root.bind("<F11>", lambda e: full(not st["full"]))
    root.bind("<Escape>", lambda e: full(False))
    root.bind("<q>", lambda e: root.destroy())

    def tick():
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
