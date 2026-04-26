"""
Utility helpers: seeding, checkpointing, and training-history plotting.
"""

import os
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ─── Checkpointing ────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    path: str,
) -> None:
    """Save model and optimizer state to *path*."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch":      epoch,
            "val_loss":   val_loss,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: torch.optim.Optimizer = None,
    device: str = "cpu",
) -> int:
    """
    Load checkpoint into *model* (and optionally *optimizer*).

    Returns the saved epoch number.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(
        f"[Checkpoint] Loaded from '{path}'  "
        f"(epoch {checkpoint['epoch']}, val_loss {checkpoint['val_loss']:.4f})"
    )
    return checkpoint["epoch"]


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_history(history: Dict[str, list], save_path: str = None) -> None:
    """
    Plot train / validation loss and accuracy curves.

    If *save_path* is given, saves the figure to that path instead of
    displaying it interactively.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot_history] matplotlib not installed. Skipping plot.")
        return

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"],   label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"],   label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[plot_history] Saved to '{save_path}'")
    else:
        plt.show()

    plt.close(fig)
