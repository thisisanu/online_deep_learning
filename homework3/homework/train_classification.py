import os
import sys
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Fix imports: Add 'homework/' to sys.path
# -----------------------------
from homework.datasets import SuperTuxClassificationDataset, load_data, get_class_names
from homework.models import Classifier, save_model

# -----------------------------
# Argument parser
# -----------------------------
parser = argparse.ArgumentParser(description="Train SuperTux classification model")
parser.add_argument("--data_dir", type=str, default="./classification_data", help="Path to dataset")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
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

# Safety copy in homework/ directory
homework_dir = Path(__file__).resolve().parent
weights_path = homework_dir / "classifier.th"
if not weights_path.exists():
    print(f"Warning: {weights_path} not found after save_model. Creating safety copy...")
    torch.save(model.state_dict(), weights_path)

print(f"Saved model with validation accuracy: {best_acc:.3f}")

