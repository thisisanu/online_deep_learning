"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import torch
from torch.utils.data import DataLoader
from homework.models import MODEL_FACTORY, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import PlannerMetric


# ------------------------------------------------------
# Correct waypoint loss (weighted Long/Lat + mask)
# ------------------------------------------------------
def waypoint_loss(pred, target, mask):
    """
    pred:   (B, 3, 2)
    target: (B, 3, 2)
    mask:   (B, 3)
    """
    mask = mask.unsqueeze(-1)  # (B, 3, 1)

    dx = pred[..., 0] - target[..., 0]
    dy = pred[..., 1] - target[..., 1]

    # Weighted longitudinal & lateral loss
    loss = (1.3 * dx**2 + dy**2) * mask

    return loss.mean()


# ------------------------------------------------------
# FINAL TRAIN FUNCTION — PASSING VERSION
# ------------------------------------------------------
def train(
    model_name="mlp_planner",
    transform_pipeline="state_only",
    num_workers=2,              # prevents warnings/freezing
    lr=1e-3,
    batch_size=64,
    num_epoch=40,
    device="cuda" if torch.cuda.is_available() else "cpu",
):

    # --------------------------------------------------
    # Correct dataset paths
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
    # Model
    # --------------------------------------------------
    model = MODEL_FACTORY[model_name]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --------------------------------------------------
    # Train loop
    # --------------------------------------------------
    for epoch in range(1, num_epoch + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            mask = batch["waypoints_mask"].to(device)

            pred = model(track_left, track_right)

            # Debug shape check — safe to keep
            assert pred.shape == waypoints.shape, \
                f"Shape mismatch: pred {pred.shape}, target {waypoints.shape}"

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
                pred = model(
                    batch["track_left"].to(device),
                    batch["track_right"].to(device)
                )
                metric.add(pred,
                           batch["waypoints"].to(device),
                           batch["waypoints_mask"].to(device))

        results = metric.compute()

        print(
            f"Epoch [{epoch}/{num_epoch}] "
            f"Train Loss: {avg_loss:.4f} | "
            f"Long: {results['longitudinal_error']:.3f} | "
            f"Lat: {results['lateral_error']:.3f}"
        )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------
    save_model(model)
    print(f"Saved model → {model_name}.th")


if __name__ == "__main__":
    train()
