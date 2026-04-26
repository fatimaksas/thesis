"""
main.py – Entry point for training and evaluating the CNN-ViT Hybrid model.

Run
---
    python main.py              # train with default config.py settings
    python main.py --eval-only  # load best checkpoint and evaluate on test set
    python main.py --epochs 30  # override number of epochs
"""

import argparse
import os
import sys

import torch

import config as cfg
from src.data      import build_dataloaders
from src.models    import CNNViTHybrid
from src.training  import Trainer, compute_metrics
from src.training.metrics import print_metrics
from src.utils     import set_seed, load_checkpoint, plot_history


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="CNN-ViT Hybrid for combined medical imaging datasets."
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; load best checkpoint and evaluate on test set."
    )
    parser.add_argument(
        "--epochs", type=int, default=cfg.NUM_EPOCHS,
        help=f"Number of training epochs (default: {cfg.NUM_EPOCHS})."
    )
    parser.add_argument(
        "--batch-size", type=int, default=cfg.BATCH_SIZE,
        help=f"Mini-batch size (default: {cfg.BATCH_SIZE})."
    )
    parser.add_argument(
        "--lr", type=float, default=cfg.LEARNING_RATE,
        help=f"Initial learning rate (default: {cfg.LEARNING_RATE})."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use: 'cuda' or 'cpu' (auto-detected if not set)."
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of DataLoader worker processes (default: 4)."
    )
    parser.add_argument(
        "--no-pretrained", action="store_true",
        help="Train CNN backbone from scratch (no ImageNet weights)."
    )
    parser.add_argument(
        "--freeze-backbone", action="store_true",
        help="Freeze CNN backbone weights (only train the ViT head)."
    )
    return parser.parse_args()


# ─── Evaluation helper ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_on_test(model, test_loader, device):
    model.eval()
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
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
        class_names=cfg.CLASS_NAMES,
    )
    metrics["loss"] = total_loss / n
    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Reproducibility ────────────────────────────────────────────────────
    set_seed(cfg.RANDOM_SEED)

    # ── Device ─────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Using device: {device}")

    # ── Data ───────────────────────────────────────────────────────────────
    print("[Main] Loading datasets...")
    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size  = args.batch_size,
        num_workers = args.workers,
    )

    # Print class distribution
    print(
        f"[Main] Train set class counts: "
        f"{train_loader.dataset.class_counts}"
    )
    print(
        f"[Main] Train set source counts: "
        f"{train_loader.dataset.source_counts}"
    )

    # ── Model ──────────────────────────────────────────────────────────────
    print("[Main] Building CNN-ViT Hybrid model...")
    model = CNNViTHybrid(
        num_classes     = cfg.NUM_CLASSES,
        pretrained      = not args.no_pretrained,
        freeze_backbone = args.freeze_backbone,
    )
    print(f"[Main] Trainable parameters: {model.count_parameters():,}")

    checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth")

    # ── Evaluation-only mode ───────────────────────────────────────────────
    if args.eval_only:
        if not os.path.isfile(checkpoint_path):
            print(
                f"[Main] No checkpoint found at '{checkpoint_path}'. "
                "Please train first."
            )
            sys.exit(1)
        load_checkpoint(model, checkpoint_path, device=str(device))
        metrics = evaluate_on_test(model, test_loader, device)
        print_metrics(metrics, split="Test")
        return

    # ── Training ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        device       = str(device),
        num_epochs   = args.epochs,
        lr           = args.lr,
    )
    history = trainer.fit()

    # Save training curves
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    plot_history(
        history,
        save_path=os.path.join(cfg.RESULTS_DIR, "training_history.png"),
    )

    # ── Final test evaluation ──────────────────────────────────────────────
    print("[Main] Evaluating best model on test set...")
    if os.path.isfile(checkpoint_path):
        load_checkpoint(model, checkpoint_path, device=str(device))

    metrics = evaluate_on_test(model, test_loader, device)
    print_metrics(metrics, split="Test")

    # Save metrics to text file
    metrics_path = os.path.join(cfg.RESULTS_DIR, "test_metrics.txt")
    with open(metrics_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"[Main] Test metrics saved to '{metrics_path}'")


if __name__ == "__main__":
    main()
