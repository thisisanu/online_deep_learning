import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import argparse
import os

# ✅ Import your dataset loader
from homework.datasets.classification_dataset import load_data, LABEL_NAMES


# ------------------------- #
# Device setup
# ------------------------- #
def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


# ------------------------- #
# Model creation
# ------------------------- #
def create_model(num_classes, pretrained=True):
    model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


# ------------------------- #
# Training one epoch
# ------------------------- #
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()  # ✅ switch to training mode
    running_loss, running_corrects = 0.0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc.item()


# ------------------------- #
# Evaluation
# ------------------------- #
def evaluate(model, dataloader, criterion, device):
    model.eval()  # ✅ switch to eval mode before inference
    running_loss, running_corrects = 0.0, 0

    with torch.inference_mode():  # ✅ safe mode for validation
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    loss = running_loss / len(dataloader.dataset)
    acc = running_corrects.double() / len(dataloader.dataset)
    return loss, acc.item()


# ------------------------- #
# Main Function
# ------------------------- #
def main():
    parser = argparse.ArgumentParser(description="SuperTux Classification Training")
    parser.add_argument("--data_dir", type=str, default="./classification_data",
                        help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    device = setup_device()

    # ✅ Use your SuperTuxDataset-based loader
    print("Loading SuperTux classification dataset...")
    train_loader = load_data(os.path.join(args.data_dir, "train"),
                             transform_pipeline="aug",
                             batch_size=args.batch_size,
                             shuffle=True)
    val_loader = load_data(os.path.join(args.data_dir, "val"),
                           transform_pipeline="default",
                           batch_size=args.batch_size,
                           shuffle=False)
    dataloaders = {"train": train_loader, "val": val_loader}

    num_classes = len(LABEL_NAMES)
    model = create_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Model: ResNet18 with {num_classes} output classes")

    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, dataloaders["train"], criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, dataloaders["val"], criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # ✅ Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Saved best model (val_acc={val_acc:.4f})")

    print("\nTraining complete ✅")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
