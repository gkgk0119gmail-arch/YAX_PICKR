#!/usr/bin/env python3
"""
Download every model the pipeline uses into the Hugging Face cache, plus our own trained weights.

Gated models (DINOv3, VGGT-1B-Commercial) need a Hugging Face account that was granted access:
    export HF_TOKEN=hf_...      (never commit it)
Our RF-DETR bubble-wrap weights are a GitHub Release asset of this repository (too big for git):
    gh release download models-v1 -p checkpoint_best_total.pth -D perception-main/ai_bubble/output
"""
import os
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ("IDEA-Research/grounding-dino-base", "Apache-2.0", False),
    ("facebook/sam2.1-hiera-large", "Apache-2.0", False),
    ("facebook/dinov3-vitb16-pretrain-lvd1689m", "DINOv3 License (gated)", True),
    ("facebook/VGGT-1B-Commercial", "VGGT License (gated)", True),
]


def main():
    for repo, lic, gated in MODELS:
        if gated and not os.environ.get("HF_TOKEN"):
            print(f"skip {repo} ({lic}): set HF_TOKEN of an account with access")
            continue
        print(f"downloading {repo} ({lic}) ...")
        snapshot_download(repo)
    out = ROOT / "perception-main/ai_bubble/output"
    if not (out / "checkpoint_best_total.pth").exists():
        out.mkdir(parents=True, exist_ok=True)
        print("downloading our RF-DETR weights from the GitHub release ...")
        subprocess.run(["gh", "release", "download", "models-v1", "-p", "checkpoint_best_total.pth",
                        "-D", str(out)], cwd=ROOT, check=False)
    print("done. RF-DETR base weights (rf-detr-medium.pth) download automatically on first use.")


if __name__ == "__main__":
    main()
