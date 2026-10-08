#!/usr/bin/env python3
"""Plot training metrics from RF-DETR output/metrics.csv."""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=os.path.join(
        os.path.dirname(__file__), "output", "metrics.csv"))
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "output", "training_plot.png"))
    args = parser.parse_args()

    df = pd.read_csv(args.metrics)

    val = df.dropna(subset=["val/mAP_50_95"]).copy()
    train = df.dropna(subset=["train/loss"]).copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RF-DETR Fine-Tuning — Box Detection", fontsize=15, fontweight="bold")

    # -- mAP over epochs --
    ax = axes[0, 0]
    ax.plot(val["epoch"], val["val/mAP_50_95"], "o-", label="mAP@50:95", markersize=3)
    ax.plot(val["epoch"], val["val/mAP_50"], "s-", label="mAP@50", markersize=3)
    ax.plot(val["epoch"], val["val/mAP_75"], "^-", label="mAP@75", markersize=3)
    if "val/ema_mAP_50_95" in val.columns:
        ax.plot(val["epoch"], val["val/ema_mAP_50_95"], "x--", label="EMA mAP@50:95",
                markersize=3, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_title("Validation mAP")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # -- Recall / AR --
    ax = axes[0, 1]
    ax.plot(val["epoch"], val["val/mAR"], "o-", label="mAR@500", markersize=3)
    ax.plot(val["epoch"], val["val/recall"], "s-", label="Recall (F1 sweep)", markersize=3)
    ax.plot(val["epoch"], val["val/precision"], "^-", label="Precision (F1 sweep)", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / AR")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # -- Training loss --
    ax = axes[1, 0]
    ax.plot(train["epoch"], train["train/loss"], ".-", markersize=2, color="tab:red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    # -- Validation loss --
    ax = axes[1, 1]
    ax.plot(val["epoch"], val["val/loss"], ".-", markersize=3, color="tab:orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Validation Loss")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {args.out}")


if __name__ == "__main__":
    main()
