import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from collections import Counter

from homework.datasets.road_dataset import load_data
from homework.models import Detector
from homework.metrics import ConfusionMatrix

# -----------------------
# Hyperparameters
# -----------------------
batch_size = 16
lr = 1e-3
num_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mixed_precision = True  # Enable FP16 training
num_workers = 2  # safer for Colab

# -----------------------
# Dataset and DataLoader
# -----------------------
data_dir = Path("drive_data")  # adjust if needed
train_data = load_data(data_dir / "train", transform_pipeline="aug", return_dataloader=False)
val_data = load_data(data_dir / "val", transform_pipeline="default", return_dataloader=False)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# -----------------------
# Compute class weights for segmentation
# -----------------------
all_labels = []
for sample in train_data:
    all_labels.append(sample["track"].flatten())
all_labels = torch.cat(all_labels)
class_counts = Counter(all_labels.tolist())
num_classes = 3
total = sum(class_counts.values())
weights = [total / (num_classes * class_counts.get(cls, 1)) for cls in range(num_classes)]
weights = torch.tensor(weights, device=device)

# -----------------------
# Model, Loss, Optimizer
# -----------------------
model = Detector(in_channels=3, num_classes=num_classes).to(device)
seg_criterion = nn.CrossEntropyLoss(weight=weights)
depth_criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=lr)

scaler = torch.cuda.amp.GradScaler(enabled=(mixed_precision and device.type=="cuda"))

# -----------------------
# Track best model
# -----------------------
homework_dir = Path.cwd()  # Colab-friendly
homework_model_path = homework_dir / "detector.th"
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
        seg_labels = batch['track'].to(device)
        depth_labels = batch['depth'].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(mixed_precision and device.type=="cuda")):
            seg_logits, depth_pred = model(images)
            seg_loss = seg_criterion(seg_logits, seg_labels)
            depth_loss = depth_criterion(depth_pred.squeeze(1), depth_labels)
            loss = seg_loss + depth_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_seg_loss += seg_loss.item() * images.size(0)
        running_depth_loss += depth_loss.item() * images.size(0)

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
            seg_labels = batch['track'].to(device)
            depth_labels = batch['depth'].to(device)

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
        torch.save(best_model_wts, homework_model_path)
        print(f"Saved best model with val IoU: {best_val_iou:.4f} -> {homework_model_path}")

# -----------------------
# Final save at end of training
# -----------------------
if best_model_wts is not None:
    torch.save(best_model_wts, homework_model_path)
    print(f"Final best model saved to {homework_model_path}")


