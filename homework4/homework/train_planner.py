"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import torch
from torch.utils.data import DataLoader
from homework.models import MODEL_FACTORY, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric
import pathlib

# ------------------------------------------------------
# Waypoint loss (weighted Long/Lat + mask)
# ------------------------------------------------------
import torch
import torch.nn.functional as F

def waypoint_loss(pred, target, mask, long_weight=1.0, lat_weight=2.0):
    """
    pred:   (B, n_waypoints, 2)
    target: (B, n_waypoints, 2)
    mask:   (B, n_waypoints, 1) boolean
    """
    # Ensure mask has correct shape
    mask_exp = mask.squeeze(-1).unsqueeze(-1).expand_as(pred)

    # Weighted longitudinal / lateral loss
    long_loss = F.mse_loss(pred[..., 0] * mask.squeeze(-1), target[..., 0] * mask.squeeze(-1), reduction='mean')
    lat_loss  = F.mse_loss(pred[..., 1] * mask.squeeze(-1), target[..., 1] * mask.squeeze(-1), reduction='mean')

    # Total loss with adjustable weights
    return long_weight * long_loss + lat_weight * lat_loss

# ------------------------------------------------------
# Training function
# ------------------------------------------------------
def train(
    model_name="mlp_planner",
    transform_pipeline="state_only",
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

    # --------------------------------------------------
    # Model: FORCE n_waypoints=3
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
            # STATE_ONLY pipeline uses track_left + track_right
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)

            waypoints = batch["waypoints"][:, :3].to(device)
            mask = batch["waypoints_mask"][:, :3].to(device)

            # Forward pass
            pred = model(track_left, track_right)

            # SAFETY: ensure pred matches target
            if pred.size(1) != waypoints.size(1):
                pred = pred[:, :waypoints.size(1), :]

            # Compute loss
            loss = waypoint_loss(pred, waypoints, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * track_left.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------
        model.eval()
        metric = PlannerMetric()

        with torch.no_grad():
            for batch in val_loader:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)

                pred = model(track_left, track_right)

                target_waypoints = batch["waypoints"][:, :3].to(device)
                if pred.size(1) != target_waypoints.size(1):
                    pred = pred[:, :target_waypoints.size(1), :]

                metric.add(pred, target_waypoints, batch["waypoints_mask"][:, :3].to(device))

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
    save_model(model, "mlp_planner")  # saves with a custom name

    print(f"Saved model → {model_name}.th")


if __name__ == "__main__":
    train()
