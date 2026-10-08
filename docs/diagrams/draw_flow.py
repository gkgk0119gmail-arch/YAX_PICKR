"""Basic-task feedback-loop diagram with hand-placed nodes and non-crossing feedback lanes."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Polygon

for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f and ("Regular" in f or "Medium" in f):
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK JP"

BLUE = ("#DCEBFF", "#2F6FD6")
ORANGE = ("#FFE7D1", "#E07B1F")
GREEN = ("#E3F5E1", "#3C9A3C")
YELLOW = ("#FFF6D6", "#C9A227")
RED = ("#FDE2E2", "#D64545")
GREY = ("#EEEEEE", "#888888")
INK, RED_L, GREEN_L = "#444444", "#D64545", "#3C9A3C"

fig, ax = plt.subplots(figsize=(12, 16.5), dpi=150)
ax.set_xlim(0, 12)
ax.set_ylim(0.7, 17)
ax.axis("off")


def box(x, y, w, h, title, sub, col, double=False):
    fc, ec = col
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                fc=fc, ec=ec, lw=2))
    if double:
        ax.add_patch(FancyBboxPatch((x - w / 2 + 0.07, y - h / 2 + 0.07), w - 0.14, h - 0.14,
                                    boxstyle="round,pad=0.02,rounding_size=0.14", fc="none", ec=ec, lw=1.2))
    if sub:
        ax.text(x, y + 0.13, title, ha="center", va="center", fontsize=14, fontweight="bold", color="#111111")
        ax.text(x, y - 0.2, sub, ha="center", va="center", fontsize=9.5, color="#333333")
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=13, color="#111111")
    return dict(l=x - w / 2, r=x + w / 2, t=y + h / 2, b=y - h / 2, x=x, y=y)


def diamond(x, y, w, h, title, sub, col):
    fc, ec = col
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)], closed=True,
                         fc=fc, ec=ec, lw=2))
    ax.text(x, y + (0.12 if sub else 0), title, ha="center", va="center", fontsize=12.5, fontweight="bold")
    if sub:
        ax.text(x, y - 0.2, sub, ha="center", va="center", fontsize=9, color="#333333")
    return dict(l=x - w / 2, r=x + w / 2, t=y + h / 2, b=y - h / 2, x=x, y=y)


def arrow(pts, color=INK, dashed=False, label=None, lpos=None, lcol=None, ha="left"):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys, color=color, lw=1.8, ls=(0, (5, 3)) if dashed else "-", zorder=1)
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=16), zorder=2)
    if label:
        ax.text(*lpos, label, fontsize=11, color=lcol or color, ha=ha, va="center",
                bbox=dict(fc="white", ec="none", pad=1.2))


# ---- title
ax.text(6, 16.55, "PAC 2026 기본과제 — 피드백 루프", ha="center", fontsize=22, fontweight="bold")
ax.text(6, 16.1, "파티션 칸별 블록 적재 · PICKR Lab", ha="center", fontsize=12, color="#666666")

X = 5.6
S1 = box(X, 15.1, 6.2, 0.85, "1. 상자 위치 · 높이", "D435i · Grounding DINO · RANSAC 테이블 평면 · 벽 높이 p99", BLUE)
Q1 = diamond(X, 13.75, 2.6, 1.15, "파티션이 상자 안?", None, YELLOW)
STOP = box(9.4, 13.75, 1.9, 0.65, "정지 / 알림", None, GREY)
S2 = box(X, 12.4, 6.6, 0.85, "2. 파티션 n×m · 빈 칸", "D435i · VGGT · Canny · RANSAC 직선 · SAM2+RANSAC 사각형 · DINOv3", BLUE)
Q2 = diamond(X, 11.05, 2.6, 1.15, "빈 칸 · 블록 남음?", None, YELLOW)
DONE = box(9.4, 11.05, 1.9, 0.75, "완료", "안착 개수 보고", GREEN)
S3 = box(X, 9.7, 6.6, 0.85, "3. 블록 집기", "D435i · RF-DETR · SAM2 · RANSAC 사각형 → PiPER 이동 · 그리퍼 닫기", GREEN)
S4 = diamond(X, 8.3, 4.0, 1.3, "4. 5 cm 들어 올려 확인", "D405 · 그리퍼 벌림 폭", ORANGE)
S5 = box(X, 6.9, 6.0, 0.85, "5. 목표 칸 위로 이동", "PiPER · z = 상자 높이 + 블록 높이 + 안전마진", GREEN)
S6 = diamond(X, 5.5, 4.2, 1.3, "6. 위치 · 파지 확인", "D435i 그리퍼 위치 · D405 파지", ORANGE)
FIX = box(9.25, 5.5, 1.7, 0.6, "보정 이동", None, RED)
S7 = box(X, 4.05, 6.4, 0.85, "7. 내려놓기 → 안착 확인", "PiPER 하강 · 그리퍼 열기 · D435i + DINOv3 + depth", BLUE, double=True)
UPD = box(3.7, 2.55, 2.3, 0.6, "빈 칸 목록 갱신", None, GREEN)
RE = box(7.5, 2.55, 2.5, 0.65, "재파지 · 90° 회전", None, RED)

# ---- main flow
for a, b, lab in [(S1, Q1, None), (Q1, S2, "예"), (S2, Q2, None), (Q2, S3, "예"), (S3, S4, None),
                  (S4, S5, "잡힘"), (S5, S6, None), (S6, S7, "OK")]:
    arrow([(X, a["b"]), (X, b["t"])], label=lab, lpos=(X + 0.1, (a["b"] + b["t"]) / 2) if lab else None)
arrow([(Q1["r"], Q1["y"]), (STOP["l"], STOP["y"])], color=RED_L, label="아니오", lpos=(Q1["r"] + 0.15, Q1["y"] + 0.22))
arrow([(Q2["r"], Q2["y"]), (DONE["l"], DONE["y"])], label="아니오", lpos=(Q2["r"] + 0.15, Q2["y"] + 0.22))
arrow([(S6["r"], S6["y"]), (FIX["l"], FIX["y"])], color=RED_L, dashed=True, label="오차 큼",
      lpos=(S6["r"] + 0.05, S6["y"] + 0.3))
arrow([(FIX["x"], FIX["b"]), (FIX["x"], S6["y"] - 0.62), (S6["x"] + 0.95, S6["y"] - 0.62),
       (S6["x"] + 0.95, S6["y"] - 0.33)], color=RED_L, dashed=True, label="재확인",
      lpos=(FIX["x"] + 0.12, S6["y"] - 0.62), ha="left")
arrow([(S7["x"] - 1.2, S7["b"]), (UPD["x"], UPD["t"])], label="정상 안착", lpos=(UPD["x"] - 1.45, 3.2))
arrow([(S7["x"] + 1.2, S7["b"]), (RE["x"], RE["t"])], color=RED_L, dashed=True, label="걸침",
      lpos=(RE["x"] + 0.12, 3.2))

# ---- left lanes: retries back to step 3 (red), next block back to Q2 (green, outermost)
lane = {"S4": 1.55, "S6": 1.15, "S7": 0.75}
entry = {"S4": S3["y"] - 0.25, "S6": S3["y"], "S7": S3["y"] + 0.25}
for key, node, lab in [("S4", S4, "안 잡힘"), ("S6", S6, "놓침"), ("S7", S7, "빈 칸 / 상자 밖")]:
    xl = lane[key]
    start = (node["l"], node["y"])
    arrow([start, (xl, node["y"]), (xl, entry[key]), (S3["l"], entry[key])], color=RED_L, dashed=True,
          label=lab, lpos=(xl + 0.08, node["y"] + 0.22))
arrow([(UPD["l"], UPD["y"]), (0.35, UPD["y"]), (0.35, Q2["y"]), (Q2["l"], Q2["y"])], color=GREEN_L, dashed=True,
      label="다음 블록", lpos=(0.45, Q2["y"] + 0.24), lcol=GREEN_L)

# ---- right lane: re-insert back to step 5
xr = 11.0
arrow([(RE["r"], RE["y"]), (xr, RE["y"]), (xr, S5["y"]), (S5["r"], S5["y"])], color=RED_L, dashed=True,
      label="재삽입", lpos=(xr - 0.1, S5["y"] + 0.24), ha="right")

# ---- legend (bottom row)
ly = 1.25
ax.add_patch(FancyBboxPatch((0.3, ly - 0.42), 11.35, 0.84, boxstyle="round,pad=0.02,rounding_size=0.15",
                            fc="white", ec="#BBBBBB", lw=1.2))
ax.text(0.65, ly, "범례", va="center", fontsize=12, fontweight="bold")
lx = 1.65
for txt, col in [("D435i (천장)", BLUE), ("D405 (손목)", ORANGE), ("PiPER 동작", GREEN), ("실패 처리", RED)]:
    ax.add_patch(FancyBboxPatch((lx, ly - 0.16), 0.5, 0.32, boxstyle="round,pad=0.01,rounding_size=0.08",
                                fc=col[0], ec=col[1], lw=1.5))
    ax.text(lx + 0.62, ly, txt, va="center", fontsize=11)
    lx += 1.95
ax.plot([lx - 0.1, lx + 0.45], [ly, ly], color=RED_L, lw=1.8, ls=(0, (5, 3)))
ax.text(lx + 0.55, ly, "점선 = 되돌아가기", va="center", fontsize=11, color="#555555")

fig.savefig("basic_task_flow.png", bbox_inches="tight", facecolor="white")
fig.savefig("basic_task_flow.svg", bbox_inches="tight", facecolor="white")
print("saved")
