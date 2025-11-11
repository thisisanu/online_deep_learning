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
best_model_wts = copy.deepcopy(model.state_d_
