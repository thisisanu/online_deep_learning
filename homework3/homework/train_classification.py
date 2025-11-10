import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb

from homework.models import load_model, save_model
from homework.datasets.classification_dataset import load_data
from homework.metrics import AccuracyMetric


def train(
    data_dir: str = "./classification_data",
    exp_dir: str = "logs",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    """
    Main training loop for the classifier
    """
    # Set random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create directories
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(exist_ok=True)

    data_dir = Path(data_dir)
    train_path = data_dir / "train"
    val_path = data_dir / "val"

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = load_model("classifier", with_weights=False).to(device)

    # Create optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Create metrics
    train_accuracy = AccuracyMetric()
    val_accuracy = AccuracyMetric()

    # Load datasets
    train_data = load_data(
        str(train_path),
        transform_pipeline="aug",
        shuffle=True,
        batch_size=batch_size,
    )
    val_data = load_data(
        str(val_path),
        transform_pipeline="default",
        shuffle=False,
        batch_size=batch_size,
    )

    # Setup TensorBoard
    logger = tb.SummaryWriter(str(exp_dir / "train"), flush_secs=1)

    # Training loop
    global_step = 0
    best_val_acc = 0.0

    for epoch in range(num_epoch):
        print(f"\nEpoch {epoch+1}/{num_epoch}")
        print("-" * 30)

        # --- Training Phase ---
        model.train()  # ✅ enable training mode
        train_accuracy.reset()

        for imgs, labels in train_data:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            train_accuracy.add(preds.cpu(), labels.cpu())

            logger.add_scalar("train/loss", loss.item(), global_step)
            logger.add_scalar("train/accuracy", train_accuracy.get_value(), global_step)
            global_step += 1

        # --- Validation Phase ---
        model.eval()  # ✅ switch to eval mode
        val_accuracy.reset()
        val_loss = 0.0
        num_batches = 0

        with torch.inference_mode():  # ✅ no gradient computation
            for imgs, labels in val_data:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                val_accuracy.add(preds.cpu(), labels.cpu())
                num_batches += 1

        val_loss /= num_batches
        val_acc = val_accuracy.get_value()

        print(f"Train Accuracy: {train_accuracy.get_value():.4f}")
        print(f"Val Accuracy:   {val_acc:.4f}")

        logger.add_scalar("val/loss", val_loss, epoch)
        logger.add_scalar("val/accuracy", val_acc, epoch)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model)
            print(f"✅ Saved best model (Val Acc = {best_val_acc:.4f})")

    print("\n🎯 Training complete.")
    print(f"Best validation accuracy achieved: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Classification Model")

    parser.add_argument("--data_dir", type=str, default="./classification_data",
                        help="Base directory for classification data")
    parser.add_argument("--exp_dir", type=str, default="logs",
                        help="Experiment log directory")
    parser.add_argument("--num_epoch", type=int, default=10,
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for dataloaders")
    parser.add_argument("--seed", type=int, default=2024,
                        help="Random seed for reproducibility")

    args = parser.parse_args()
    train(**vars(args))
