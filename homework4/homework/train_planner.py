"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from homework.models import MODEL_FACTORY, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric
import pathlib


# ------------------------------------------------------
# Waypoint loss (weighted Long/Lat + mask)
# ------------------------------------------------------
def waypoint_loss(pred, target, mask, long_weight=1.0, lat_weight=2.0):
    """
    pred:   (B, n_waypoints, 2)
    target: (B, n_waypoints, 2)
    mask:   (B, n_waypoints) or (B, n_waypoints, 1)
    """

    # --- Fix mask shape ---
    if mask.ndim == 3:         # (B, n, 1)
        mask = mask.squeeze(-1)   # (B, n)

    # Make mask float
    mask = mask.float()

    # --- Longitudinal MSE ---
    long_loss = F.mse_loss(
        pred[..., 0] * mask,
        target[..., 0] * mask,
        reduction="mean"
    )

    # --- Lateral MSE ---
    lat_loss = F.mse_loss(
        pred[..., 1] * mask,
        target[..., 1] * mask,
        reduction="mean"
    )

    return long_weight * long_loss + lat_weight * lat_loss


# ------------------------------------------------------
# Training function
# ------------------------------------------------------
def train(
    model_name="cnn_planner",
    transform_pipeline="img_only",
    num_workers=2,
    lr=1e-3,
    batch_size=64,
    num_epoch=40,
    device="cuda" if torch.cuda.is_available() else "cpu",
):

    # --------------------------------------------------
    # Delete old checkpoint if exists
    # --------------------------------------------------
    checkpoint_path = pathlib.Path(f"{model_name}.th")
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Deleted old checkpoint: {checkpoint_path}")

    # --------------------------------------------------
    # Data loaders
    # --------------------------------------------------
    train_loader = load_data(
        dataset_path="drive_data/train",
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = load_data(
        dataset_path="drive_data/val",
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=False,
    )

    print(f"Loaded {len(train_loader.dataset)} training samples")
    print(f"Loaded {len(val_loader.dataset)} validation samples")

    # --------------------------------------------------
    # Model (always use n_waypoints=3)
    # --------------------------------------------------
    model = MODEL_FACTORY[model_name](n_waypoints=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --------------------------------------------------
    # Train loop
    # --------------------------------------------------
    for epoch in range(1, num_epoch + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:

            # --- IMG ONLY ---
            if "image" in batch:
                x = batch["image"].to(device)               # (B,3,H,W)
                pred = model(x)

            # --- STATE ONLY ---
            else:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                pred = model(track_left, track_right)

            waypoints = batch["waypoints"][:, :3].to(device)
            mask = batch["waypoints_mask"][:, :3].to(device)

            loss = waypoint_loss(pred, waypoints, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * waypoints.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------
        model.eval()
        metric = PlannerMetric()

        with torch.no_grad():
            for batch in val_loader:

                if "image" in batch:
                    pred = model(batch["image"].to(device))
                else:
                    pred = model(
                        batch["track_left"].to(device),
                        batch["track_right"].to(device)
                    )

                target_waypoints = batch["waypoints"][:, :3].to(device)
                mask = batch["waypoints_mask"][:, :3].to(device)

                metric.add(pred, target_waypoints, mask)

        results = metric.compute()

        print(
            f"Epoch [{epoch}/{num_epoch}] "
            f"Train Loss: {avg_loss:.4f} | "
            f"Long: {results['longitudinal_error']:.3f} | "
            f"Lat:  {results['lateral_error']:.3f}"
        )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------
    save_model(model, model_name)
    print(f"Saved model → {model_name}.th")


if __name__ == "__main__":
    train()
