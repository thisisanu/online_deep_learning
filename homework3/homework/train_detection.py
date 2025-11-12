import sys
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# -----------------------------
# Fix imports
# -----------------------------
homework_path = Path(__file__).resolve().parent
sys.path.insert(0, str(homework_path))

from homework.datasets.road_dataset import load_data
from homework.models import Detector  # make sure Detector is compatible with UNet output
from homework.metrics import ConfusionMatrix

# -----------------------------
# Hyperparameters
# -----------------------------
batch_size = 16
lr_seg = 1e-3
lr_depth = 5e-4
num_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mixed_precision = True
num_workers = 4

# -----------------------------
# Dataset
# -----------------------------
data_dir = Path("/content/online_deep_learning/homework3/drive_data")
train_data = load_data(data_dir / "train", transform_pipeline="aug", return_dataloader=False)
val_data = load_data(data_dir / "val", transform_pipeline="default", return_dataloader=False)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# -----------------------------
# Class weights for segmentation
# -----------------------------
all_labels = []
for sample in train_data:
    track = sample['track']
    track = torch.from_numpy(track.copy()).long() if isinstance(track, np.ndarray) else track.long()
    all_labels.append(track.flatten())
all_labels = torch.cat(all_labels)

num_classes = 3
class_counts = Counter(all_labels.tolist())
print("Class counts:", class_counts)

# Increase weights for lane/curb (classes 1 & 2)
weights = torch.tensor([1.0, 2.5, 2.5], device=device)
seg_criterion = nn.CrossEntropyLoss(weight=weights)
depth_criterion = nn.L1Loss()
print("Segmentation weights:", weights)

# -----------------------------
# Model
# -----------------------------
model = Detector(in_channels=3, num_classes=num_classes).to(device)

optimizer = optim.Adam([
    {'params': model.parameters(), 'lr': lr_seg}
])

scaler = torch.amp.GradScaler(enabled=(mixed_precision and device.type=="cuda"))

# -----------------------------
# Checkpoints
# -----------------------------
homework_model_path = homework_path / "detector_best.th"
checkpoint_dir = homework_path / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

best_val_iou = 0.0
best_model_wts = None

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(num_epochs):
    model.train()
    running_seg_loss = 0.0
    running_depth_loss = 0.0

    for batch in train_loader:
        images = batch['image'].to(device)
        seg_labels = batch['track']
        depth_labels = batch['depth'].to(device)

        seg_labels = torch.from_numpy(seg_labels.copy()).long().to(device) if isinstance(seg_labels, np.ndarray) else seg_labels.long().to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=mixed_precision):
            seg_logits, depth_pred = model(images)
            seg_loss = seg_criterion(seg_logits, seg_labels)
            depth_loss = depth_criterion(depth_pred.squeeze(1), depth_labels)

            # Emphasize segmentation
            loss = seg_loss * 3.0 + depth_loss * 0.3

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_seg_loss += seg_loss.item() * images.size(0)
        running_depth_loss += depth_loss.item() * images.size(0)

    print(f"Epoch {epoch+1}/{num_epochs} | Seg Loss: {running_seg_loss/len(train_loader.dataset):.4f} | Depth Loss: {running_depth_loss/len(train_loader.dataset):.4f}")

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    confusion = ConfusionMatrix(num_classes=num_classes)
    val_depth_error = 0.0
    val_depth_boundary_error = 0.0
    total_pixels = 0
    total_boundary_pixels = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            seg_labels = batch['track']
            depth_labels = batch['depth'].to(device)

            seg_labels = torch.from_numpy(seg_labels.copy()).long().to(device) if isinstance(seg_labels, np.ndarray) else seg_labels.long().to(device)

            seg_logits, depth_pred = model(images)
            seg_preds = seg_logits.argmax(dim=1)

            confusion.add(seg_preds, seg_labels)

            abs_diff = torch.abs(depth_pred.squeeze(1) - depth_labels)
            val_depth_error += abs_diff.sum().item()
            total_pixels += abs_diff.numel()

            boundary_mask = (seg_labels == 1) | (seg_labels == 2)
            val_depth_boundary_error += abs_diff[boundary_mask].sum().item()
            total_boundary_pixels += boundary_mask.sum().item()

    val_depth_error /= total_pixels
    val_depth_boundary_error = val_depth_boundary_error / total_boundary_pixels if total_boundary_pixels > 0 else 0.0

    metrics = confusion.compute()
    val_iou = metrics["iou"]
    val_acc = metrics["accuracy"]

    print(f"Val Acc: {val_acc:.4f} | Val IoU: {val_iou:.4f} | Val Depth MAE: {val_depth_error:.4f} | Depth MAE (boundary): {val_depth_boundary_error:.4f}")

    # -----------------------------
    # Save checkpoint
    # -----------------------------
    checkpoint_path = checkpoint_dir / f"detector_epoch{epoch+1}.pt"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

    # -----------------------------
    # Save best model by IoU
    # -----------------------------
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        best_model_wts = model.state_dict()
        torch.save(best_model_wts, homework_model_path)
        print(f"Saved best model with Val IoU: {best_val_iou:.4f} -> {homework_model_path}")

# -----------------------------
# Final save
# -----------------------------
if best_model_wts is not None:
    torch.save(best_model_wts, homework_model_path)
    print(f"Final best model saved to {homework_model_path}")
