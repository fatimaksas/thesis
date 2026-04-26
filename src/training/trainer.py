"""
Training loop (Trainer class).

Handles:
  - Train / validation epochs
  - Learning-rate scheduling (cosine annealing or step-decay)
  - Early stopping
  - Checkpoint saving (best model by validation loss)
  - History logging
"""

import os
import sys
import time
from typing import Tuple, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config as cfg

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_metrics, print_metrics
from ..utils.helpers import save_checkpoint


class Trainer:
    """
    Encapsulates the full training procedure.

    Parameters
    ----------
    model        : the CNNViTHybrid (or any nn.Module)
    train_loader : DataLoader for training split
    val_loader   : DataLoader for validation split
    device       : torch device string or object
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        device:       str = None,
        num_epochs:   int = cfg.NUM_EPOCHS,
        lr:           float = cfg.LEARNING_RATE,
        weight_decay: float = cfg.WEIGHT_DECAY,
        scheduler:    str = cfg.LR_SCHEDULER,
        patience:     int = cfg.EARLY_STOP_PATIENCE,
        checkpoint_dir: str = cfg.CHECKPOINT_DIR,
    ) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.num_epochs   = num_epochs
        self.patience     = patience
        self.checkpoint_dir = checkpoint_dir

        # Device
        if device is None:
            device = cfg.DEVICE
        self.device = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )
        self.model.to(self.device)

        # Loss & optimiser
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        # LR scheduler
        if scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs, eta_min=1e-7
            )
        elif scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.LR_STEP_SIZE,
                gamma=cfg.LR_GAMMA,
            )
        else:
            self.scheduler = None

        os.makedirs(checkpoint_dir, exist_ok=True)

        # History
        self.history: Dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
        }

    # ── One training epoch ────────────────────────────────────────────────
    def _train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(
            tqdm(self.train_loader, desc="  Train", leave=False)
        ):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss   = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=cfg.GRAD_CLIP_MAX_NORM)
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds      = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

        return total_loss / total, correct / total

    # ── One evaluation epoch ──────────────────────────────────────────────
    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> Tuple[float, float, dict]:
        self.model.eval()
        total_loss = 0.0
        all_labels, all_preds, all_probs = [], [], []

        for images, labels in tqdm(loader, desc="  Eval ", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            logits = self.model(images)
            loss   = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

        n = len(all_labels)
        metrics = compute_metrics(
            all_labels, all_preds, all_probs,
            num_classes=cfg.NUM_CLASSES,
        )
        return total_loss / n, metrics["accuracy"], metrics

    # ── Full training loop ────────────────────────────────────────────────
    def fit(self) -> Dict[str, list]:
        """
        Train the model for up to `num_epochs` epochs.

        Returns the history dict (train/val loss and accuracy per epoch).
        """
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_epoch = 0

        print(f"\n[Trainer] Starting training on {self.device}")
        print(
            f"[Trainer] Epochs: {self.num_epochs}  "
            f"Batch: {self.train_loader.batch_size}  "
            f"LR: {self.optimizer.param_groups[0]['lr']:.2e}\n"
        )

        for epoch in range(1, self.num_epochs + 1):
            t0 = time.time()

            # Train
            train_loss, train_acc = self._train_epoch()

            # Validate
            val_loss, val_acc, val_metrics = self._eval_epoch(self.val_loader)

            elapsed = time.time() - t0

            # Log
            print(
                f"Epoch [{epoch:03d}/{self.num_epochs}]  "
                f"Train loss: {train_loss:.4f}  acc: {train_acc:.4f}  │  "
                f"Val loss: {val_loss:.4f}  acc: {val_acc:.4f}  "
                f"AUC: {val_metrics['auc']:.4f}  "
                f"({elapsed:.1f}s)"
            )

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # LR step
            if self.scheduler is not None:
                self.scheduler.step()

            # Checkpoint
            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                best_epoch     = epoch
                epochs_no_improve = 0
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    os.path.join(self.checkpoint_dir, "best_model.pth"),
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(
                        f"\n[Trainer] Early stopping triggered at epoch {epoch}. "
                        f"Best epoch: {best_epoch}."
                    )
                    break

        print(f"\n[Trainer] Training complete. Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")
        return self.history
