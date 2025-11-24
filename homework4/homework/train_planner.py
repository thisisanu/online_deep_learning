"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import torch
from torch.utils.data import DataLoader
from homework.models import MODEL_FACTORY, save_model
from homework.datasets.road_dataset import RoadDataset
from homework.metrics import PlannerMetric


def waypoint_loss(pred, target, mask):
    mask = mask.unsqueeze(-1)  
    return ((pred - target) ** 2 * mask).mean()


def train(
    model_name="mlp_planner",
    transform_pipeline="state_only",
    num_workers=4,
    lr=1e-3,
    batch_size=64,
    num_epoch=20,
    device="cpu",
):

    # ----------------------------
    # Load datasets
    # ----------------------------
    train_set = RoadDataset(split="train", transform_pipeline=transform_pipeline)
    val_set = RoadDataset(split="val", transform_pipeline=transform_pipeline)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)

    # ----------------------------
    # Create model + optimizer
    # ----------------------------
    model = MODEL_FACTORY[model_name]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ----------------------------
    # Training loop
    # ----------------------------
    for epoch in range(1, num_epoch + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            mask = batch["waypoints_mask"].to(device)

            pred = model(track_left, track_right)
            loss = waypoint_loss(pred, waypoints, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * track_left.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)

        # ----------------------------
        # Validation using PlannerMetric
        # ----------------------------
        model.eval()
        metric = PlannerMetric()

        with torch.no_grad():
            for batch in val_loader:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                mask = batch["waypoints_mask"].to(device)

                pred = model(track_left, track_right)
                metric.add(pred, waypoints, mask)

        results = metric.compute()

        print(
            f"Epoch [{epoch}/{num_epoch}] "
            f"Loss: {avg_train_loss:.4f} | "
            f"Long: {results['longitudinal_error']:.3f}, "
            f"Lat: {results['lateral_error']:.3f}"
        )

    # ----------------------------
    # Save trained model
    # ----------------------------
    save_model(model)
    print(f"Saved model → {model_name}.th")


if __name__ == "__main__":
    train()
