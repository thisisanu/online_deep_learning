"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

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
def waypoint_loss(pred, target, mask):
    """
    pred:   (B, n_waypoints, 2)
    target: (B, n_waypoints, 2)
    mask:   (B, n_waypoints)
    """
    # SAFETY: trim pred to match target
    if pred.size(1) != target.size(1):
        pred = pred[:, :target.size(1), :]

    # Ensure mask is broadcastable
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)  # (B, n_waypoints, 1)

    dx2 = (pred[..., 0] - target[..., 0]) ** 2
    dy2 = (pred[..., 1] - target[..., 1]) ** 2

    loss = 1.3 * dx2 + dy2

    # flatten any trailing singleton dimension in mask
    if mask.ndim == loss.ndim and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)

    mask = mask.expand_as(loss)
    return (loss * mask).mean()

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
            # ------------------------------
            # STATE_ONLY pipeline
            # ------------------------------
            state = batch["state"][:, :].to(device)
            waypoints = batch["waypoints"][:, :3].to(device)      # (B,3,2)
            mask = batch["waypoints_mask"][:, :3].to(device)      # (B,3)

            pred = model(state)

            # SAFETY: ensure pred matches target
            if pred.size(1) != waypoints.size(1):
                pred = pred[:, :waypoints.size(1), :]

            # Compute loss
            loss = waypoint_loss(pred, waypoints, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * state.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------
        model.eval()
        metric = PlannerMetric()

        with torch.no_grad():
            for batch in val_loader:
                state = batch["state"][:, :].to(device)
                pred = model(state)

                target_waypoints = batch["waypoints"][:, :3].to(device)
                if pred.size(1) != target_waypoints.size(1):
                    pred = pred[:, :target_waypoints.size(1), :]

                metric.add(
                    pred,
                    target_waypoints,
                    batch["waypoints_mask"][:, :3].to(device)
                )

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
    save_model(model)
    print(f"Saved model → {model_name}.th")


if __name__ == "__main__":
    train()
