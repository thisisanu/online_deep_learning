import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.classification_dataset import load_data
from .metrics import AccuracyMetric

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
    
    # Create model
    model = load_model("classifier", with_weights=False).to(device)
    
    # Create optimizer 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create metrics
    train_accuracy = AccuracyMetric()
    val_accuracy = AccuracyMetric()
    
    # Create data loaders
    train_data = load_data(str(train_path), transform_pipeline="aug", 
                          shuffle=True, batch_size=batch_size)
    val_data = load_data(str(val_path), transform_pipeline="default", 
                        shuffle=False, batch_size=batch_size)

    # Setup tensorboard
    logger = tb.SummaryWriter(str(exp_dir / "train"), flush_secs=1)

    # Training loop
    global_step = 0
    best_val_acc = 0
    
    for epoch in range(num_epoch):
        print(f"Epoch {epoch}")
        
        # Training
        model.train()
        train_accuracy.reset()
        
        for img, label in train_data:
            img, label = img.to(device), label.to(device)
            
            # Forward pass
            logits = model(img)
            loss = criterion(logits, label)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
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
                
                # Forward pass
                logits = model(img)
                val_loss += criterion(logits, label).item()
                
                # Update metrics
                pred = model.predict(img)
                val_accuracy.add(pred.cpu(), label.cpu())
                
                num_val_batches += 1
        
        val_loss /= num_val_batches
        val_acc = val_accuracy.get_value()
        
        # Log validation metrics
        logger.add_scalar("val/loss", val_loss, global_step)
        logger.add_scalar("val/accuracy", val_acc, global_step)
        
        print(f"Train accuracy: {train_accuracy.get_value():.3f}")
        print(f"Val accuracy: {val_acc:.3f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model)
            print(f"Saved model with validation accuracy: {val_acc:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default="./classification_data", help="Base directory of classification data")
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)
    
    args = parser.parse_args()
    train(**vars(args))
