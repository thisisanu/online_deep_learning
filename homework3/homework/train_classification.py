import os
import sys
import argparse
import copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

# -----------------------------
# Fix imports: Add 'homework/' to sys.path
# -----------------------------
homework_path = Path(__file__).resolve().parent / "homework"
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
parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
args = parser.parse_args()

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Get class names
# -----------------------------
labels_csv_path = Path(args.data_dir) / "train" / "labels.csv"
print(f"[train_classification] Looking for labels.csv at: {labels_csv_path}")
class_names = get_class_names(args.data_dir, split="train")
print(f"Found classes: {class_names}")

# -----------------------------
# Data transforms
# -----------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# -----------------------------
# Data loaders
# -----------------------------
train_dataset = SuperTuxClassificationDataset(root_dir=args.data_dir, split="train", transform=train_transform)
val_dataset = SuperTuxClassificationDataset(root_dir=args.data_dir, split="val", transform=val_transform)

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
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# -----------------------------
# Training loop
# -----------------------------
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(args.epochs):
    print(f"\nEpoch {epoch+1}/{args.epochs}")
    print("-" * 20)

    for phase in ["train", "val"]:
        model.train() if phase == "train" else model.eval()
        loader = dataloaders[phase]

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

        if phase == "train":
            scheduler.step()

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
model.eval()

save_model(model)  # saves homework/classifier.th
print(f"Saved model with validation accuracy: {best_acc:.3f}")
