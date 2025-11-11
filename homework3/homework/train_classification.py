import sys
import os
import argparse
import copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Fix imports: add homework/ to sys.path
# -----------------------------
homework_path = Path(__file__).resolve().parent
sys.path.insert(0, str(homework_path))

# Import local modules
from datasets.classification_dataset import SuperTuxClassificationDataset, get_class_names
from models import Classifier, save_model

# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Train SuperTux classification model")
parser.add_argument("--data_dir", type=str, default="./classification_data", help="Path to dataset")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
args = parser.parse_args()

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Get class names from labels.csv
# -----------------------------
print("[train_classification] Looking for labels.csv at:", os.path.join(args.data_dir, "train", "labels.csv"))
class_names = get_class_names(args.data_dir, split="train")
print(f"Found classes: {class_names}")

# -----------------------------
# Data loaders
# -----------------------------
train_dataset = SuperTuxClassificationDataset(root_dir=args.data_dir, split="train")
val_dataset = SuperTuxClassificationDataset(root_dir=args.data_dir, split="val")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

dataloaders = {"train": train_loader, "val": val_loader}
dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

# -----------------------------
# Model, criterion, optimizer
# -----------------------------
model = Classifier(in_channels=3, num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# -----------------------------
# Sanity check: inspect a single batch
# -----------------------------
print("\n[Sanity Check] Running one batch through the model...")

sample_inputs, sample_labels = next(iter(train_loader))
print(f"Input batch shape: {sample_inputs.shape}")   # Expected: [batch_size, 3, H, W]
print(f"Labels shape: {sample_labels.shape}")       # Expected: [batch_size]

sample_inputs = sample_inputs.to(device)
sample_labels = sample_labels.to(device)

model.eval()  # Ensure eval mode for sanity check
with torch.inference_mode():
    outputs = model(sample_inputs)
    print(f"Output logits shape: {outputs.shape}")  # Expected: [batch_size, num_classes]
    _, preds = torch.max(outputs, 1)
    print(f"Predicted classes shape: {preds.shape}") # Expected: [batch_size]

print("[Sanity Check] Passed! Shapes look correct.\n")

# -----------------------------
# Training loop
# -----------------------------
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(args.epochs):
    print(f"\nEpoch {epoch+1}/{args.epochs}")
    print("-" * 20)

    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
            loader = train_loader
        else:
            model.eval()
            loader = val_loader

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Save best weights
        if phase == "val" and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

print(f"\nTraining complete. Best val Acc: {best_acc:.4f}")

# -----------------------------
# Save best model
# -----------------------------
model.load_state_dict(best_model_wts)

# Primary save (using existing helper)
save_model(model)  # Saves to homework/classifier.th

# Ensure explicit safety copy in homework/ directory
weights_path = homework_path / "classifier.th"
torch.save(model.state_dict(), weights_path)

print(f"Saved model with validation accuracy: {best_acc:.3f}")
print(f"Model weights saved to: {weights_path.resolve()}")
