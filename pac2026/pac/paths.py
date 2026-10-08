"""Make the existing perception-main modules importable without moving them yet."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]                 # pac2026/
PERCEPTION = ROOT.parent / "perception-main"
for sub in ("research/box_cells", "ai_failure", "ai_bubble"):
    p = str(PERCEPTION / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
