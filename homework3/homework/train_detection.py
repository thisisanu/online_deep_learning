import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from collections import Counter
import numpy as np

from homework.datasets.road_dataset import load_data
from homework.models import Detector
from homework.metrics import ConfusionMatrix

# -----------------------
# Hyperparameters
# -----------------------
batch_size = 16
lr_seg = 1e-3       # Segmentation LR
lr_depth = 5e-4     # Depth LR
num_epochs = 20     # Reduced
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mixed_precision = True
num_workers = 2

# -----------------------
# Dataset and DataLoader
# -----------------------
data_dir = Path("/content/online_deep_learning/homework3/drive_data")
train_data = load_data(data_dir / "train", transform_pipeline="aug", return_dataloader=False)
val_data = load_data(data_dir / "val", transform_pipeline="default", return_dataloader=False)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# -----------------------
# Compute class weights for segmentation
# -----------------------
num_classes = 3
# Boost lane classes
weights = torch.tensor([0.1, 1.5, 1.5], device=device)
seg_criterion = nn.CrossEntropyLoss(weight=weights)

# -----------------------
# Dice Loss function
# -----------------------
def dice_loss(pred, target, eps=1e-6):
    pred = torch.softmax(pred, dim=1)
    target_onehot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1]).permute(0,3,1,2).float()
    intersection = (pred * target_onehot).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target_onehot.sum(dim=(2,3))
    loss = 1 - (2 * intersection + eps) / (union + eps)
    return loss.mean()

# -----------------------
# Model, Loss, Optimizer
# -----------------------
model = Detector(in_channels=3, num_classes=num_classes).to(device)
depth_criterion = nn.L1Loss()

optimizer = optim.Adam([
    {'params': model.segmentation_head.parameters(), 'lr': lr_seg},
    {'params': model.depth_head.parameters(), 'lr': lr_depth},
])

scaler = torch.amp.GradScaler(enabled=(mixed_precision and device.type=="cuda"))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# -----------------------
# Checkpoints
# -----------------------
homework_dir = Path.cwd()
homework_model_path = homework_dir / "detector_best.th"
checkpoint_dir = homework_dir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
        seg_labels = batch['track']
        depth_labels = batch['depth'].to(device)

        if isinstance(seg_labels, np.ndarray):
            seg_labels = torch.from_numpy(seg_labels.copy()).long().to(device)
        else:
            seg_labels = seg_labels.long().to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(enabled=(mixed_precision and device.type=="cuda")):
            seg_logits, depth_pred = model(images)
            seg_loss = seg_criterion(seg_logits, seg_labels) + dice_loss(seg_logits, seg_labels)
            depth_loss = depth_criterion(depth_pred.squeeze(1), depth_labels)
            # Focus more on segmentation
            loss = seg_loss * 2.0 + depth_loss * 0.5

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_seg_loss += seg_loss.item() * images.size(0)
        running_depth_loss += depth_loss.item() * images.size(0)

    scheduler.step()  # Update LR

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Seg Loss: {running_seg_loss/len(train_loader.dataset):.4f}, "
          f"Depth Loss: {running_depth_loss/len(train_loader.dataset):.4f}")

    # -----------------------
    # Validation
    # -----------------------
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

            if isinstance(seg_labels, np.ndarray):
                seg_labels = torch.from_numpy(seg_labels.copy()).long().to(device)
            else:
                seg_labels = seg_labels.long().to(device)

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

    print(f"Val IoU: {val_iou:.4f}, Val Depth MAE: {val_depth_error:.4f}, "
          f"Val Depth MAE (boundary): {val_depth_boundary_error:.4f}")

    # -----------------------
    # Save checkpoint
    # -----------------------
    checkpoint_path = checkpoint_dir / f"detector_epoch{epoch+1}.pt"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

    # -----------------------
    # Save best model
    # -----------------------
    if val_iou > best_val_iou:
        best_val_iou = val_iou
        best_model_wts = model.state_dict()
        torch.save(best_model_wts, homework_model_path)
        print(f"Saved best model with val IoU: {best_val_iou:.4f} -> {homework_model_path}")

# -----------------------
# Final save
# -----------------------
if best_model_wts is not None:
    torch.save(best_model_wts, homework_model_path)
    print(f"Final best model saved to {homework_model_path}")
