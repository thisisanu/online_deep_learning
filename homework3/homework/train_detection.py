import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys

from homework.datasets.road_dataset import load_data
from homework.models import Detector
from homework.metrics import ConfusionMatrix

# -----------------------
# Paths and setup
# -----------------------
HOMEWORK_ROOT = Path(__file__).resolve().parent
sys.path.append(str(HOMEWORK_ROOT))
data_dir = Path("drive_data")
homework_model_path = HOMEWORK_ROOT / "detector.th"
checkpoint_dir = HOMEWORK_ROOT / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Hyperparameters
# -----------------------
batch_size = 16
lr = 1e-3
num_epochs = 30

# -----------------------
# Dataset and DataLoader
# -----------------------
train_data = load_data(data_dir / "train", return_dataloader=False)
val_data = load_data(data_dir / "val", return_dataloader=False)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# -----------------------
# Model, loss, optimizer
# -----------------------
model = Detector().to(device)

# Weighted CE for class imbalance
class_weights = torch.tensor([0.2, 0.4, 0.4], device=device)
seg_criterion = nn.CrossEntropyLoss(weight=class_weights)
depth_criterion = nn.L1Loss()  # MAE

optimizer = optim.Adam(model.parameters(), lr=lr)

# Track best model
best_val_iou = 0.0
best_model_wts = None

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
        depth_loss = depth_criterion(depth_pred, depth_labels)
        loss = seg_loss + depth_loss

        loss.backward()
        optimizer.step()

        running_seg_loss += seg_loss.item() * images.size(0)
        running_depth_loss += depth_loss.item() * images.size(0)

    avg_seg_loss = running_seg_loss / len(train_loader.dataset)
    avg_depth_loss = running_depth_loss / len(train_loader.dataset)
    print(f"[Epoch {epoch+1}/{num_epochs}] "
          f"Seg Loss: {avg_seg_loss:.4f}, Depth Loss: {avg_depth_loss:.4f}")

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
            confusion.add(seg_preds, seg_labels)

            abs_diff = torch.abs(depth_pred - depth_labels)
            val_depth_error += abs_diff.sum().item()
            total_pixels += abs_diff.numel()

            boundary_mask = (seg_labels == 1) | (seg_labels == 2)
            val_depth_boundary_error += abs_diff[boundary_mask].sum().item()
            total_boundary_pixels += boundary_mask.sum().item()

    val_depth_error /= total_pixels
    val_depth_boundary_error /= total_boundary_pixels
    metrics = confusion.compute()
    val_iou = metrics["iou"]
    val_acc = metrics["accuracy"]

    print(f"[Val] IoU: {val_iou:.4f}, Accuracy: {val_acc:.4f}, "
          f"Depth MAE: {val_depth_error:.4f}, "
          f"Depth MAE (boundary): {val_depth_boundary_error:.4f}")

    # -----------------------
    # Save checkpoint
    # -----------------------
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_dir / f"detector_epoch{epoch+1}.pt")

    # -----------------------
    # Save best model
    # -----------------------
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        best_model_wts = model.state_dict()
        torch.save(best_model_wts, homework_model_path)
        print(f"[INFO] Saved best model with IoU {best_val_iou:.4f} -> {homework_model_path}")

# -----------------------
# Final save
# -----------------------
if best_model_wts is not None:
    torch.save(best_model_wts, homework_model_path)
    print(f"[INFO] Final best model saved to {homework_model_path}")
