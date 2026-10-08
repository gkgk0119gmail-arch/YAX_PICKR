#!/usr/bin/env python3
"""
Fine-tune RF-DETR Medium on a custom box detection dataset.

Usage:
    python train.py [--dataset-dir DIR] [--output-dir DIR] [--epochs N]
                    [--batch-size N] [--lr FLOAT] [--resolution INT]
"""

import argparse
import os

from rfdetr import RFDETRMedium


def main():
    parser = argparse.ArgumentParser(description="Fine-tune RF-DETR for box detection")
    parser.add_argument("--dataset-dir",
                        default=os.path.join(os.path.dirname(__file__), "dataset"),
                        help="Path to COCO-format dataset (train/valid/test splits)")
    parser.add_argument("--output-dir",
                        default=os.path.join(os.path.dirname(__file__), "output"),
                        help="Directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-4)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze the DINOv2 backbone (faster, less GPU memory)")
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    output_dir = os.path.abspath(args.output_dir)

    print(f"Dataset:    {dataset_dir}")
    print(f"Output:     {output_dir}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch size: {args.batch_size} (grad accum {args.grad_accum_steps})")
    print(f"LR:         {args.lr} (encoder: {args.lr_encoder})")
    print(f"Resolution: {args.resolution}")
    print(f"Freeze enc: {args.freeze_encoder}")
    print()

    model = RFDETRMedium(resolution=args.resolution)

    model.train(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        use_ema=True,
        checkpoint_interval=args.checkpoint_interval,
        freeze_encoder=args.freeze_encoder,
    )

    print(f"\nTraining complete. Checkpoints saved to {output_dir}")


if __name__ == "__main__":
    main()
