"""
Train script for Homework 3 — Classification

Run using:
    python -m homework.train_classification
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import time

from .models import load_model, save_model
from .datasets.classification_dataset import load_data
from .metrics import AccuracyMetric


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

def train_classification(
    batch_size: int = 32,
    epochs: int = 5,
    lr: float = 1e-3,
    num_workers: int = 2,
    transform: str = "default",  # pass e.g. "augmented" if you added augmentations
):
    device = get_device()
    print_banner(f"Training Classifier on device: {device}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    model = load_model("classifier", in_channels=3, num_classes=6)
    model = model.to(device)

    # --------------------------------------------------------
    # Loss + Optimizer
    # --------------------------------------------------------
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --------------------------------------------------------
    # Data loading
    # --------------------------------------------------------
    train_data, val_data = load_data(transform_pipeline=transform)

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
    # Training
    # --------------------------------------------------------
    print_banner("Starting Training")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()

        running_loss = 0.0

        for batch in train_loader:
            img = batch["image"].to(device)         # (B,3,64,64)
            labels = batch["label"].to(device)      # (B,)

            optimizer.zero_grad()

            logits = model(img)
            loss = ce_loss(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        dt = time.time() - t0
        avg_loss = running_loss / len(train_loader)

        print(f"[Epoch {epoch}]  Loss = {avg_loss:.4f}   ({dt:.1f} sec)")

        # ----------------------------------------------------
        # Validation accuracy
        # ----------------------------------------------------
        model.eval()
        acc = AccuracyMetric()

        with torch.inference_mode():
            for batch in val_loader:
                img = batch["image"].to(device)
                labels = batch["label"].to(device)

                pred = model.predict(img)
                acc.add(pred, labels)

        print(f"  Val Accuracy:  {acc.value():.4f}")

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------
    print_banner("Saving Model")
    path = save_model(model)
    print(f"Model saved to: {path}")
    return path


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--transform", type=str, default="default")

    args = parser.parse_args()

    train_classification(
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        transform=args.transform,
    )


if __name__ == "__main__":
    main()
