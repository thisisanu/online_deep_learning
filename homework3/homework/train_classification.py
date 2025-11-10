import argparse
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb

# Absolute imports for Colab
from models import load_model, save_model
from datasets.classification_dataset import load_data
from metrics import AccuracyMetric

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

    # Check if directories exist
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Train or validation directory not found.\nTrain: {train_path}\nVal: {val_path}"
        )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = load_model("classifier", with_weights=False).to(device)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Metrics
    train_accuracy = AccuracyMetric()
    val_accuracy = AccuracyMetric()
    
    # Data loaders
    train_data = load_data(str(train_path), transform_pipeline="aug", shuffle=True, batch_size=batch_size)
    val_data = load_data(str(val_path), transform_pipeline="default", shuffle=False, batch_size=batch_size)

    # TensorBoard logger
    logger = tb.SummaryWriter(str(exp_dir / "train"), flush_secs=1)

    # Training loop
    global_step = 0
    best_val_acc = 0

    for epoch in range(num_epoch):
        print(f"\nEpoch {epoch+1}/{num_epoch}")
        
        # Training
        model.train()
        train_accuracy.reset()
        
        for img, label in train_data:
            img, label = img.to(device), label.to(device)
            
            logits = model(img)
            loss = criterion(logits, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pred = model.predict(img)
            train_accuracy.add(pred.cpu(), label.cpu())
            
            # Log
            logger.add_scalar("train/loss", loss.item(), global_step)
            logger.add_scalar("train/accuracy", train_accuracy.get_value(), global_step)
            
            global_step += 1
        
        # Validation
        model.eval()
        val_accuracy.reset()
        val_loss = 0
        num_val_batches = 0
        
        with torch.inference_mode():
            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                
                logits = model(img)
                val_loss += criterion(logits, label).item()
                
                pred = model.predict(img)
                val_accuracy.add(pred.cpu(), label.cpu())
                
                num_val_batches += 1
        
        val_loss /= max(1, num_val_batches)
        val_acc = val_accuracy.get_value()
        
        # Log validation metrics
        logger.add_scalar("val/loss", val_loss, global_step)
        logger.add_scalar("val/accuracy", val_acc, global_step)
        
        print(f"Train Accuracy: {train_accuracy.get_value():.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model)
            print(f"Saved model with validation accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default="./classification_data", help="Base directory of classification data")
    parser.add_argument("--exp_dir", type=str, default="logs", help="Directory for logs")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)
    
    args = parser.parse_args()
    train(**vars(args))
