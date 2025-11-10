import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb

from homework.datasets.classification_dataset import load_data
from homework.models import load_model, save_model
from homework.metrics import AccuracyMetric


def train(
    data_dir: str = "./classification_data",
    exp_dir: str = "logs",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
):
    """
    Main training loop for classification using SuperTuxDataset.
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Prepare paths
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(exist_ok=True)

    data_dir = Path(data_dir)
    train_path = data_dir / "train"
    val_path = data_dir / "val"

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_model("classifier", with_weights=False).to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Metrics
    train_acc = AccuracyMetric()
    val_acc = AccuracyMetric()

    # Load data using your provided dataset code
    train_data = load_data(
        str(train_path), transform_pipeline="aug", shuffle=True, batch_size=batch_size
    )
    val_data = load_data(
        str(val_path), transform_pipeline="default", shuffle=False, batch_size=batch_size
    )

    # TensorBoard setup
    writer = tb.SummaryWriter(str(exp_dir / "train"), flush_secs=1)

    best_val_acc = 0.0
    global_step = 0

    for epoch in range(num_epoch):
        print(f"Epoch {epoch+1}/{num_epoch}")

        # --- Training ---
        model.train()
        train_acc.reset()
        for imgs, labels in train_data:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Update metrics
            preds = logits.argmax(dim=1)
            train_acc.add(preds.cpu(), labels.cpu())

            # Log training loss
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/accuracy", train_acc.get_value(), global_step)
            global_step += 1

        # --- Validation ---
        model.eval()
        val_acc.reset()
        val_loss = 0.0
        num_batches = 0
        with torch.inference_mode():
            for imgs, labels in val_data:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                val_loss += criterion(logits, labels).item()
                preds = logits.argmax(dim=1)
                val_acc.add(preds.cpu(), labels.cpu())
                num_batches += 1

        val_loss /= num_batches
        val_accuracy = val_acc.get_value()

        print(f"Train Accuracy: {train_acc.get_value():.3f}, Val Accuracy: {val_accuracy:.3f}")

        # Log validation metrics
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_accuracy, epoch)

        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            save_model(model)
            print(f"Saved best model (Val Acc = {best_val_acc:.3f})")

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train classifier using SuperTuxDataset")
    parser.add_argument("--data_dir", type=str, default="./classification_data", help="Base dataset directory")
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    train(**vars(args))
