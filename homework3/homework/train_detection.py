import torch
import torch.nn as nn
import torch.optim as optim
import sys
from torch.utils.data import DataLoader
from pathlib import Path
from homework.datasets.road_dataset import load_data
from homework.models import Detector
from homework.metrics import ConfusionMatrix

# Ensure homework root is on sys.path
HOMEWORK_ROOT = Path(__file__).resolve().parent
sys.path.append(str(HOMEWORK_ROOT))
# -----------------------
# Hyperparameters
# -----------------------
batch_size = 16
lr = 1e-3
num_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Dataset and DataLoader
# -----------------------
data_dir = Path("datasets/drive_data")
train_data = load_data(data_dir / "train")
val_data = load_data(data_dir / "val")

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# -----------------------
# Model, Loss, Optimizer
# -----------------------
model = Detector().to(device)
seg_criterion = nn.CrossEntropyLoss()
depth_criterion = nn.L1Loss()  # MAE for depth
optimizer = optim.Adam(model.parameters(), lr=lr)

# -----------------------
# Track best model
# -----------------------
best_val_iou = 0.0
best_model_wts = None

# Path to homework folder
homework_dir = Path(__file__).resolve().parent
homework_model_path = homework_dir / "detector.th"

# -----------------------
# Training Loop
# -----------------------
for epoch in range(num_epochs):
    model.train()
    running_seg_loss = 0.0
    running_depth_loss = 0.0

    for batch in train_loader:
        images = batch['image'].to(device)
        seg_labels = batch['track'].to(device)
        depth_labels = batch['depth'].to(device)

        optimizer.zero_grad()
        seg_logits, depth_pred = model(images)

        seg_loss = seg_criterion(seg_logits, seg_labels)
        depth_loss = depth_criterion(depth_pred.squeeze(1), depth_labels)
        loss = seg_loss + depth_loss

        loss.backward()
        optimizer.step()

        running_seg_loss += seg_loss.item() * images.size(0)
        running_depth_loss += depth_loss.item() * images.size(0)

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Seg Loss: {running_seg_loss/len(train_loader.dataset):.4f}, "
          f"Depth Loss: {running_depth_loss/len(train_loader.dataset):.4f}")

    # -----------------------
    # Validation
    # -----------------------
    model.eval()
    confusion = ConfusionMatrix(num_classes=3)
    val_depth_error = 0.0
    val_depth_boundary_error = 0.0
    total_pixels = 0
    total_boundary_pixels = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            seg_labels = batch['track'].to(device)
            depth_labels = batch['depth'].to(device)

            seg_logits, depth_pred = model(images)
            seg_preds = seg_logits.argmax(dim=1)

            confusion.update(seg_labels.cpu().numpy(), seg_preds.cpu().numpy())

            abs_diff = torch.abs(depth_pred.squeeze(1) - depth_labels)
            val_depth_error += abs_diff.sum().item()
            total_pixels += abs_diff.numel()

            boundary_mask = (seg_labels == 1) | (seg_labels == 2)
            val_depth_boundary_error += abs_diff[boundary_mask].sum().item()
            total_boundary_pixels += boundary_mask.sum().item()

    val_depth_error /= total_pixels
    val_depth_boundary_error /= total_boundary_pixels
    val_iou = confusion.mean_iou()

    print(f"Val IoU: {val_iou:.4f}, Val Depth MAE: {val_depth_error:.4f}, "
          f"Val Depth MAE (boundary): {val_depth_boundary_error:.4f}")

    # -----------------------
    # Save checkpoint
    # -----------------------
    checkpoint_dir = homework_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_dir / f"detector_epoch{epoch+1}.pt")

    # -----------------------
    # Save best model safely
    # -----------------------
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        best_model_wts = model.state_dict()

        # Save detector.th in homework folder
        torch.save(best_model_wts, homework_model_path)
        print(f"Saved best model with val IoU: {best_val_iou:.4f} -> {homework_model_path}")

# -----------------------
# Final save at end of training (just in case)
# -----------------------
if best_model_wts is not None:
    torch.save(best_model_wts, homework_model_path)
    print(f"Final best model saved to {homework_model_path}")




