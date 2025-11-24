"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
import torch
from torch.utils.data import DataLoader
from homework.models import MLPPlanner, save_model
from homework.datasets.road_dataset import RoadDataset
from homework.metrics import compute_errors  # or compute_metrics depending on your repo
from homework.models import MODEL_FACTORY, save_model
from homework.datasets.road_dataset import RoadDataset
from homework.metrics import compute_errors

# ------------------------
# Loss function
# ------------------------
def waypoint_loss(pred, target, mask):
    """
    Computes MSE on valid waypoints only.
    pred, target: (B, n_waypoints, 2)
    mask: (B, n_waypoints) boolean
    """
    mask = mask.unsqueeze(-1)  # (B, n_waypoints, 1)
    return ((pred - target)**2 * mask).mean()


# ------------------------
# Training function
# ------------------------
def train(
    model_name="mlp_planner",
    transform_pipeline="state_only",
    num_workers=4,
    lr=1e-3,
    batch_size=64,
    num_epoch=20,
    device="cpu",
):
    # Load datasets with optional transform pipeline
    train_set = RoadDataset(split="train", transform_pipeline=transform_pipeline)
    val_set = RoadDataset(split="val", transform_pipeline=transform_pipeline)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)

    # Create model
    model = MODEL_FACTORY[model_name]().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epoch + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            mask = batch["waypoints_mask"].to(device)

            pred = model(track_left, track_right)
            mask_exp = mask.unsqueeze(-1)
            loss = ((pred - waypoints) ** 2 * mask_exp).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * track_left.size(0)

        avg_train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_long_err, val_lat_err, val_count = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                mask = batch["waypoints_mask"].to(device)

                pred = model(track_left, track_right)
                long_err, lat_err = compute_errors(pred, waypoints, mask)
                val_long_err += long_err
                val_lat_err += lat_err
                val_count += 1

        avg_long_err = val_long_err / val_count
        avg_lat_err = val_lat_err / val_count

        print(
            f"Epoch [{epoch}/{num_epoch}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Long: {avg_long_err:.3f}, Lat: {avg_lat_err:.3f}"
        )

    # Save model
    save_model(model)
    print(f"{model_name} saved!")



# Only call train when running this file directly
if __name__ == "__main__":
    train()
