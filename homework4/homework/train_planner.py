"""
Usage:
    python3 -m homework.train_planner --your_args here
"""
import torch
from torch.utils.data import DataLoader
from homework.models import MLPPlanner, save_model
from homework.datasets.road_dataset import RoadDataset
from homework.metrics import compute_errors  # or compute_metrics depending on your repo

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
def train(num_epochs=20, batch_size=64, lr=1e-3, device="cpu"):
    # Load datasets
    train_set = RoadDataset(split="train")
    val_set = RoadDataset(split="val")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # Create model and optimizer
    model = MLPPlanner().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
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

        # ------------------------
        # Validation
        # ------------------------
        model.eval()
        total_long_err, total_lat_err, val_count = 0.0, 0.0, 0

        with torch.no_grad():
            for batch in val_loader:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                mask = batch["waypoints_mask"].to(device)

                pred = model(track_left, track_right)
                long_err, lat_err = compute_errors(pred, waypoints, mask)  # ensure compute_errors returns longitudinal, lateral

                total_long_err += long_err
                total_lat_err += lat_err
                val_count += 1

        avg_long_err = total_long_err / val_count
        avg_lat_err = total_lat_err / val_count

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Long: {avg_long_err:.3f}, Lat: {avg_lat_err:.3f}"
        )

    # ------------------------
    # Save final model
    # ------------------------
    save_model(model)
    print("Model saved!")


# ------------------------
# Entry point
# ------------------------
if __name__ == "__main__":
    print("Time to train!")
    train()
