#!/usr/bin/env python3
"""
Fine-tune RF-DETR Medium on the synthetic bubble-wrap block dataset (make_dataset.py).

Blocks are small (~40-60 px in a 1280x720 frame), so the default resolution is higher
than ai/train.py's 640.

Usage:
    .venv/bin/python train.py [--epochs 30] [--batch-size 4] [--resolution 768]
"""
import argparse
from pathlib import Path

from rfdetr import RFDETRMedium

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="Fine-tune RF-DETR for bubble-wrap blocks")
    ap.add_argument("--dataset-dir", default=HERE / "dataset", type=Path)
    ap.add_argument("--output-dir", default=HERE / "output", type=Path)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum-steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-encoder", type=float, default=1.5e-4)
    ap.add_argument("--resolution", type=int, default=768)
    ap.add_argument("--checkpoint-interval", type=int, default=10)
    args = ap.parse_args()

    for k, v in vars(args).items():
        print(f"{k:20s} {v}")

    model = RFDETRMedium(resolution=args.resolution)
    model.train(
        dataset_dir=str(args.dataset_dir),
        output_dir=str(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        use_ema=True,
        checkpoint_interval=args.checkpoint_interval,
    )
    print(f"\nTraining complete. Checkpoints in {args.output_dir}")


if __name__ == "__main__":
    main()
