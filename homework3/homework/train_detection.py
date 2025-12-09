"""
Train script for Homework 3 — Detection (Segmentation + Depth)

Run using:
    python -m homework.train_detection
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import time

# Correct relative imports
from .datasets.drive_dataset import SuperTuxDataset, load_data
from .models import Detector, load_model, save_model
from .metrics import AccuracyMetric, ConfusionMatrix, DepthErrorMetric


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_banner(msg: str):
    print("\n" + "=" * 60)
    print(msg)
    print("=" * 60 + "\n")


# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------

def train_detection(
    dataset_path: str,
    batch_size: int = 16,
    epochs: int = 5,
    lr: float = 1e-3,
    num_workers: int = 2,
):
    device = get_device()
    print_banner(f"Training Detector on device: {device}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    model = load_model("detector", in_channels=3, num_classes=3)
    model = model.to(device)

    # --------------------------------------------------------
    # Loss functions
    # --------------------------------------------------------
    ce_loss = nn.CrossEntropyLoss()
    l1_loss = nn.L1Loss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    dataset_path = Path(dataset_path)
    train_data, val_data = load_data(dataset_path)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    cm = ConfusionMatrix(num_classes=3)
    depth_metric = DepthErrorMetric()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    print_banner("Starting Training")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for batch in train_loader:
            img = batch["image"].to(device)
            depth_tgt = batch["depth"].to(device)
            seg_tgt = batch["track"].to(device)

            optimizer.zero_grad()
            seg_logits, depth_pred = model(img)

            loss_seg = ce_loss(seg_logits, seg_tgt)
            loss_depth = l1_loss(depth_pred, depth_tgt)
            loss = loss_seg + loss_depth

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        dt = time.time() - t0
        avg_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch}]  Loss = {avg_loss:.4f}   ({dt:.1f} sec)")

        # Validation
        model.eval()
        cm.reset()
        depth_metric.reset()

        with torch.inference_mode():
            for batch in val_loader:
                img = batch["image"].to(device)
                depth_tgt = batch["depth"].to(device)
                seg_tgt = batch["track"].to(device)

                seg_pred, depth_pred = model.predict(img)
                cm.add(seg_pred, seg_tgt)
                depth_metric.add(depth_pred, depth_tgt, seg_tgt)

        iou = cm.iou().mean().item()
        mae = depth_metric.mae()
        mae_lane = depth_metric.mae_lane_only()

        print(f"  Val mIoU:       {iou:.4f}")
        print(f"  Val MAE:        {mae:.4f}")
        print(f"  Val MAE lanes:  {mae_lane:.4f}")

    # Save final model
    print_banner("Saving Model")
    path = save_model(model)
    print(f"Model saved to: {path}")
    return path


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_detection(
        dataset_path=args.dataset,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
