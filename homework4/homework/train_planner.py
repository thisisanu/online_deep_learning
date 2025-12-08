"""
Usage:
    python3 -m homework.train_planner --model mlp_planner --epochs 50 --lr 0.001
    python3 -m homework.train_planner --model transformer_planner --epochs 50 --lr 0.001
    python3 -m homework.train_planner --model cnn_planner --epochs 50 --lr 0.001
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from .models import MODEL_FACTORY, save_model
from .metrics import PlannerMetric
from .datasets.road_dataset import load_data


class WeightedMSELoss(nn.Module):
    def __init__(self, lateral_weight=2.0, longitudinal_weight=1.5):
        super().__init__()
        self.lateral_weight = lateral_weight
        self.longitudinal_weight = longitudinal_weight
        
    def forward(self, predictions, targets):
        longitudinal_loss = nn.functional.mse_loss(predictions[:, 0], targets[:, 0])
        lateral_loss = nn.functional.mse_loss(predictions[:, 1], targets[:, 1])
        return self.longitudinal_weight * longitudinal_loss + self.lateral_weight * lateral_loss


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        predictions = model(**batch)
        waypoints = batch['waypoints']
        waypoints_mask = batch['waypoints_mask']
        valid_predictions = predictions[waypoints_mask]
        valid_targets = waypoints[waypoints_mask]
        loss = criterion(valid_predictions, valid_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        if device.type in ['cuda', 'mps']:
            torch.cuda.empty_cache() if device.type == 'cuda' else None
    return total_loss / num_batches


def evaluate(model, dataloader, metric, device):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for batch in dataloader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            predictions = model(**batch)
            metric.add(predictions, batch['waypoints'], batch['waypoints_mask'])
    return metric.compute()


def train(model_name, epochs=50, lr=0.001, batch_size=32, data_path='drive_data', 
          transform_pipeline=None, **kwargs):
    if model_name == "linear_planner":
        model_name = "mlp_planner"

    if transform_pipeline is None:
        if model_name == "cnn_planner":
            transform_pipeline = "default"
        else:
            transform_pipeline = "state_only"

    if model_name == "transformer_planner":
        device = torch.device('cpu')
        print("Forcing CPU for transformer model to avoid MPS buffer issues")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    if model_name == "transformer_planner":
        batch_size = min(batch_size, 8)
        print(f"Reduced batch size to {batch_size} for transformer model")

    model = MODEL_FACTORY[model_name]()
    model = model.to(device)
    print(f"Model: {model_name}")

    train_loader = load_data(
        f"{data_path}/train", 
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=0 if model_name == "transformer_planner" else 2,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = load_data(
        f"{data_path}/val", 
        transform_pipeline=transform_pipeline,
        return_dataloader=True,
        num_workers=0 if model_name == "transformer_planner" else 2,
        batch_size=batch_size,
        shuffle=False
    )

    criterion = WeightedMSELoss(lateral_weight=2.0)
    print("Using weighted MSE loss for better lateral error performance")

    effective_lr = lr if model_name != "transformer_planner" else min(lr, 1e-4)
    effective_wd = 1e-3 if model_name != "cnn_planner" else 1e-4
    optimizer = optim.AdamW(
        model.parameters(),
        lr=effective_lr,
        weight_decay=effective_wd
    )
    print(f"Using AdamW optimizer, lr={effective_lr}, weight_decay={effective_wd}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.7)

    metric = PlannerMetric()

    print("Starting training...")
    best_longitudinal_error = float('inf')
    best_lateral_error = float('inf')

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        metrics = evaluate(model, val_loader, metric, device)
        longitudinal_error = metrics['longitudinal_error']
        lateral_error = metrics['lateral_error']

        scheduler.step(lateral_error)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Long Error: {longitudinal_error:.4f}, "
              f"Val Lat Error: {lateral_error:.4f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        save_model_flag = False
        combined_error = longitudinal_error + lateral_error
        best_combined = best_longitudinal_error + best_lateral_error
        if combined_error < best_combined or (lateral_error < best_lateral_error and longitudinal_error < 0.25):
            if lateral_error < best_lateral_error:
                best_lateral_error = lateral_error
                print(f"New best lateral error: {lateral_error:.4f}")
            if longitudinal_error < best_longitudinal_error:
                best_longitudinal_error = longitudinal_error
                print(f"New best longitudinal error: {longitudinal_error:.4f}")
            save_model_flag = True
        if save_model_flag:
            save_path = save_model(model)
            print(f"Saved best model to {save_path}")

    print("Training completed!")
    print(f"Best lateral error: {best_lateral_error:.4f}")
    print(f"Best longitudinal error: {best_longitudinal_error:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description='Train planner models')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['mlp_planner', 'transformer_planner', 'cnn_planner'],
                       help='Model to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--data_path', type=str, default='drive_data', help='Path to dataset')
    args = parser.parse_args()
    train(args.model, args.epochs, args.lr, args.batch_size, args.data_path)

if __name__ == "__main__":
    main()
