import sys
import os
import copy
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# -----------------------------
# Fix imports
# -----------------------------
homework_path = Path(__file__).resolve().parent
sys.path.insert(0, str(homework_path))

from datasets.classification_dataset import SuperTuxClassificationDataset, get_class_names

# -----------------------------
# Hyperparameters
# -----------------------------
DATA_DIR = "./classification_data"
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Classes
# -----------------------------
class_names = get_class_names(DATA_DIR, split="train")
num_classes = len(class_names)
print(f"Found classes: {class_names}")

# -----------------------------
# Data transforms
# -----------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# Dataset & loaders
# -----------------------------
train_dataset = SuperTuxClassificationDataset(DATA_DIR, split="train", transform=train_transforms)
val_dataset = SuperTuxClassificationDataset(DATA_DIR, split="val", transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

# -----------------------------
# Model
# -----------------------------
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Sanity check
# -----------------------------
print("\n[Sanity Check] Running one batch...")
inputs, labels = next(iter(train_loader))
inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
model.eval()
with torch.inference_mode():
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    print(f"Input: {inputs.shape}, Output: {outputs.shape}, Preds: {preds.shape}")
print("[Sanity Check Passed]\n")

# -----------------------------
# Training loop
# -----------------------------
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}\n" + "-"*25)
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
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase=="train"):
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

        # Save best model
        if phase == "val" and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

print(f"\nTraining complete. Best val Acc: {best_acc:.4f}")

# -----------------------------
# Save model
# -----------------------------
model.load_state_dict(best_model_wts)
save_path = homework_path / "classifier.th"
torch.save(model.state_dict(), save_path)
print(f"Saved model weights to: {save_path.resolve()}")
