from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb


def test_logging(logger: tb.SummaryWriter):
    """
    Your code here - finish logging the dummy loss and accuracy

    For training, log the training loss every iteration and the average accuracy every epoch
    Call the loss 'train_loss' and accuracy 'train_accuracy'

    For validation, log only the average accuracy every epoch
    Call the accuracy 'val_accuracy'

    Make sure the logging is in the correct spot so the global_step is set correctly,
    for epoch=0, iteration=0: global_step=0
    """
    # strongly simplified training loop
    global_step = 0
    for epoch in range(10):
        metrics = {"train_acc": [], "val_acc": []}

        # Training loop
        torch.manual_seed(epoch)
        for iteration in range(20):
            dummy_train_loss = 0.9 ** (epoch + iteration / 20.0)
            dummy_train_accuracy = epoch / 10.0 + torch.randn(10)
            
            # Log training loss
            logger.add_scalar("train_loss", dummy_train_loss, global_step)

            # Collect accuracy
            metrics["train_acc"].append(dummy_train_accuracy)

            global_step += 1

        # Log average training accuracy at next training step of the epoch
        avg_train_acc = sum(metrics["train_acc"]) / len(metrics["train_acc"])
        # Compute the average accuracy across all elements
        avg_train_acc = avg_train_acc.mean().item()

        logger.add_scalar("train_accuracy", avg_train_acc, global_step - 1)

        # Validation loop
        torch.manual_seed(epoch)  # deterministic
        for iteration in range(10):
            dummy_val_accuracy = epoch / 10.0 + torch.randn(10)
            metrics["val_acc"].append(dummy_val_accuracy)

        # Log average validation accuracy at same global_step
        avg_val_acc = sum(metrics["val_acc"]) / len(metrics["val_acc"])
        # Compute scalar average
        avg_val_acc = torch.stack(metrics["val_acc"]).mean().item()

        logger.add_scalar("val_accuracy", avg_val_acc, global_step - 1)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    args = parser.parse_args()

    log_dir = Path(args.exp_dir) / f"logger_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    test_logging(logger)
