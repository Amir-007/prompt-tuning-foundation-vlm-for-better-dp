"""
MaPLe Training Script
======================
Training utilities for MaPLe (Multi-modal Prompt Learning).

Follows MaPLe paper hyperparameters:
- Learning rate: 0.0035 (lower than CoOp's 0.002)
- Context length: 2 (shorter than CoOp's 16)
- Prompt depth: 9 (deep multi-modal prompts)
- Optimizer: SGD with momentum
- Scheduler: Cosine annealing with warmup
"""

import json
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from maple_model import CustomCLIPMaPLe, compute_harmonic_mean


# Evaluation Functions.
@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    """
    Evaluate model accuracy on a dataset.

    Args:
        model: MaPLe model
        loader: DataLoader
        device: torch device

    Returns:
        Accuracy percentage (0-100)
    """
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_base_new(
    model: nn.Module,
    base_loader,
    new_loader,
    device: torch.device,
    base_classes: List[str],
    new_classes: List[str],
) -> Dict[str, float]:
    """
    Evaluate model on both base (seen) and new (unseen) classes.

    Returns dict with:
    - base_acc: Accuracy on base classes
    - new_acc: Accuracy on new classes
    - h_mean: Harmonic mean of base and new accuracies
    """
    # For base-to-new evaluation, use separate class sets.
    # Assume the model already matches the class set.
    base_acc = evaluate(model, base_loader, device)
    new_acc = evaluate(model, new_loader, device)
    h_mean = compute_harmonic_mean(base_acc, new_acc)

    return {
        "base_acc": base_acc,
        "new_acc": new_acc,
        "h_mean": h_mean,
    }


# Warmup + Cosine Annealing Schedule.
class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.
    """
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        warmup_lr: float = 1e-5,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup phase.
            alpha = self.current_epoch / self.warmup_epochs
            return self.warmup_lr + alpha * (self.base_lr - self.warmup_lr)
        else:
            # Cosine annealing phase.
            import math
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            return self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# Training Function.
def train_maple(
    model: CustomCLIPMaPLe,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 0.0035,
    warmup_epochs: int = 1,
    warmup_lr: float = 1e-5,
    eval_freq: int = 5,
    checkpoint_dir: Optional[str] = None,
    checkpoint_freq: int = 25,
    verbose: bool = True,
) -> Dict:
    """
    Train MaPLe model.

    Args:
        model: CustomCLIPMaPLe model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: torch device
        epochs: Number of training epochs
        lr: Base learning rate
        warmup_epochs: Number of warmup epochs
        warmup_lr: Initial warmup learning rate
        eval_freq: Evaluate every N epochs
        checkpoint_dir: Directory to save checkpoints
        checkpoint_freq: Save checkpoint every N epochs
        verbose: Print training progress

    Returns:
        History dict with training metrics
    """
    model = model.to(device)

    # Freeze all parameters except the prompt learner.
    for param in model.parameters():
        param.requires_grad_(False)

    # Enable gradients for prompt learner parameters.
    for param in model.prompt_learner.parameters():
        param.requires_grad_(True)

    # Count trainable parameters.
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    if verbose:
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
        print(f"  ({100.0 * trainable_params / total_params:.4f}% of total)")

    # Optimizer: SGD with momentum (as per MaPLe paper)
    optimizer = optim.SGD(
        [p for p in model.prompt_learner.parameters() if p.requires_grad],
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    # Create the learning-rate scheduler.
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        base_lr=lr,
        warmup_lr=warmup_lr,
    )

    # Use cross-entropy loss.
    criterion = nn.CrossEntropyLoss()

    # Track training history.
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "best_val_acc": 0.0,
        "best_epoch": 0,
    }

    # Main training loop.
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", disable=not verbose)
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass.
            logits = model(images)
            loss = criterion(logits, labels)

            # Backward pass.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        # Update the learning rate.
        scheduler.step()
        current_lr = scheduler.get_lr()

        # Compute average training loss.
        avg_train_loss = epoch_loss / n_batches

        # Run evaluation.
        if epoch % eval_freq == 0 or epoch == epochs:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    logits = model(images)
                    loss = criterion(logits, labels)

                    val_loss += loss.item() * labels.size(0)
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            avg_val_loss = val_loss / val_total if val_total > 0 else 0
            val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0

            # Update the best validation accuracy.
            if val_acc > history["best_val_acc"]:
                history["best_val_acc"] = val_acc
                history["best_epoch"] = epoch

            # Store metrics in history.
            history["epoch"].append(epoch)
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)

            if verbose:
                print(
                    f"  Epoch {epoch:3d}/{epochs} | "
                    f"Train Loss: {avg_train_loss:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f} | "
                    f"Val Acc: {val_acc:.2f}% | "
                    f"LR: {current_lr:.6f}"
                )

        # Save a checkpoint.
        if checkpoint_dir and epoch % checkpoint_freq == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.prompt_learner.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                checkpoint_path,
            )
            if verbose:
                print(f"  Saved checkpoint: {checkpoint_path}")

    return history


def run_maple_experiment(
    model: CustomCLIPMaPLe,
    train_loader,
    test_loader,
    device: torch.device,
    config: Dict,
    output_dir: str,
    exp_name: str,
    verbose: bool = True,
) -> Dict:
    """
    Run a complete MaPLe experiment with training and evaluation.

    Args:
        model: MaPLe model
        train_loader: Training data loader
        test_loader: Test data loader
        device: torch device
        config: Experiment configuration dict
        output_dir: Output directory for results
        exp_name: Experiment name for saving
        verbose: Print progress

    Returns:
        Results dict with accuracies and history
    """
    epochs = config.get("epochs", 50)
    lr = config.get("lr", 0.0035)
    warmup_epochs = config.get("warmup_epochs", 1)
    warmup_lr = config.get("warmup_lr", 1e-5)
    eval_freq = config.get("eval_freq", 5)

    # Create the checkpoint directory.
    checkpoint_dir = os.path.join(output_dir, "checkpoints", exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Run training.
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training: {exp_name}")
        print(f"{'='*60}")

    history = train_maple(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,  # Use test as validation for simplicity
        device=device,
        epochs=epochs,
        lr=lr,
        warmup_epochs=warmup_epochs,
        warmup_lr=warmup_lr,
        eval_freq=eval_freq,
        checkpoint_dir=checkpoint_dir,
        verbose=verbose,
    )

    # Run final evaluation.
    final_acc = evaluate(model, test_loader, device)

    if verbose:
        print(f"\n  Final Test Accuracy: {final_acc:.2f}%")
        print(f"  Best Val Accuracy: {history['best_val_acc']:.2f}% (epoch {history['best_epoch']})")

    # Save model weights.
    weights_dir = os.path.join(output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    weights_path = os.path.join(weights_dir, f"{exp_name}_weights.pt")
    torch.save(model.prompt_learner.state_dict(), weights_path)

    # Save training history.
    history_dir = os.path.join(output_dir, "History_And_Csv")
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, f"history_{exp_name}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Remove checkpoints after successful training.
    import shutil
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    results = {
        "exp_name": exp_name,
        "final_acc": final_acc,
        "best_val_acc": history["best_val_acc"],
        "best_epoch": history["best_epoch"],
        "weights_path": weights_path,
        "history_path": history_path,
        "history": history,
    }

    return results


# Epoch Schedule (following MaPLe paper)
def get_epochs_for_shots(num_shots: int) -> int:
    """Get recommended number of epochs based on shot count."""
    if num_shots == 1:
        return 50
    elif num_shots == 4:
        return 100
    elif num_shots == 8:
        return 100
    elif num_shots == 16:
        return 200
    else:
        return 100
