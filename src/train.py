"""
Training pipeline with Weights & Biases logging.

Usage
-----
    # Single run with defaults
    python -m src.train

    # Override hyperparameters
    python -m src.train --backbone resnet50 --lr 3e-4 --epochs 30

    # Run as part of a W&B sweep (called automatically by sweep agent)
    python -m src.train --sweep
"""
import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import wandb

from src.config import (
    DEFAULT_HPARAMS, WANDB_PROJECT, WANDB_ENTITY,
    MODELS_DIR, PROCESSED_DATA_DIR, CLASS_NAMES,
)
from src.dataset import get_dataloaders
from src.model import build_model, count_parameters
from src.evaluate import evaluate_model


def get_optimizer(model: nn.Module, hparams: dict):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if hparams["optimizer"] == "sgd":
        return optim.SGD(
            params, lr=hparams["learning_rate"],
            momentum=0.9, weight_decay=hparams["weight_decay"],
        )
    elif hparams["optimizer"] == "adamw":
        return optim.AdamW(
            params, lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
        )
    return optim.Adam(
        params, lr=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
    )


def get_scheduler(optimizer, hparams: dict):
    if hparams["scheduler"] == "cosine":
        return CosineAnnealingLR(optimizer, T_max=hparams["epochs"])
    elif hparams["scheduler"] == "step":
        return StepLR(optimizer, step_size=hparams["step_size"], gamma=hparams["gamma"])
    return None


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def train(hparams: dict | None = None):
    """Full training loop with W&B tracking."""

    hp = {**DEFAULT_HPARAMS, **(hparams or {})}

    # ── W&B init ──────────────────────────────────────────────────────
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config=hp,
        reinit=True,
    )
    hp = dict(wandb.config)  # pick up sweep overrides

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=PROCESSED_DATA_DIR,
        batch_size=hp["batch_size"],
        num_workers=hp["num_workers"],
        augmentation=hp.get("augmentation", "medium"),
    )
    print(f"Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | "
          f"Test: {len(test_loader.dataset)}")

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(
        backbone=hp["backbone"],
        pretrained=hp["pretrained"],
        dropout=hp["dropout"],
        freeze_backbone=hp["freeze_backbone"],
        unfreeze_last_n=hp["unfreeze_last_n"],
        head_hidden_dim=hp.get("head_hidden_dim", 256),
        head_num_layers=hp.get("head_num_layers", 1),
    ).to(device)

    total_p, train_p = count_parameters(model)
    print(f"Parameters — total: {total_p:,}  trainable: {train_p:,}")
    wandb.config.update({"total_params": total_p, "trainable_params": train_p})

    # ── Training setup ────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, hp)
    scheduler = get_scheduler(optimizer, hp)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    patience = hp["early_stopping_patience"]

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(1, hp["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        # W&B logging
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "learning_rate": lr,
        })

        print(f"Epoch {epoch:3d}/{hp['epochs']}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  lr={lr:.2e}")

        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # ── Save best model ───────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / "best_model.pth"
    torch.save(best_state, save_path)
    print(f"Best model saved → {save_path}  (val_acc={best_val_acc:.4f})")

    # Log model artifact to W&B
    artifact = wandb.Artifact(
        name="best-model",
        type="model",
        metadata={"val_accuracy": best_val_acc, "backbone": hp["backbone"]},
    )
    artifact.add_file(str(save_path))
    run.log_artifact(artifact)

    # ── Test evaluation ───────────────────────────────────────────────
    model.load_state_dict(best_state)
    metrics = evaluate_model(model, test_loader, CLASS_NAMES, device)

    wandb.log({
        "test/accuracy": metrics["accuracy"],
        "test/macro_f1": metrics["macro_f1"],
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=metrics["y_true"],
            preds=metrics["y_pred"],
            class_names=CLASS_NAMES,
        ),
    })

    # Log per-class metrics
    for cls_name in CLASS_NAMES:
        wandb.log({
            f"test/precision_{cls_name}": metrics["per_class"][cls_name]["precision"],
            f"test/recall_{cls_name}": metrics["per_class"][cls_name]["recall"],
            f"test/f1_{cls_name}": metrics["per_class"][cls_name]["f1-score"],
        })

    wandb.summary["best_val_accuracy"] = best_val_acc
    wandb.summary["test_accuracy"] = metrics["accuracy"]
    wandb.summary["test_macro_f1"] = metrics["macro_f1"]

    run.finish()
    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", type=str, default=DEFAULT_HPARAMS["backbone"])
    p.add_argument("--lr", type=float, default=DEFAULT_HPARAMS["learning_rate"])
    p.add_argument("--epochs", type=int, default=DEFAULT_HPARAMS["epochs"])
    p.add_argument("--batch_size", type=int, default=DEFAULT_HPARAMS["batch_size"])
    p.add_argument("--dropout", type=float, default=DEFAULT_HPARAMS["dropout"])
    p.add_argument("--optimizer", type=str, default=DEFAULT_HPARAMS["optimizer"])
    p.add_argument("--freeze_backbone", action="store_true", default=DEFAULT_HPARAMS["freeze_backbone"])
    p.add_argument("--no_freeze", dest="freeze_backbone", action="store_false")
    p.add_argument("--unfreeze_last_n", type=int, default=DEFAULT_HPARAMS["unfreeze_last_n"])
    p.add_argument("--weight_decay", type=float, default=DEFAULT_HPARAMS["weight_decay"])
    p.add_argument("--sweep", action="store_true", help="Run as W&B sweep agent")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.sweep:
        # When called by a sweep agent, hparams come from wandb.config
        train()
    else:
        hp = {
            "backbone": args.backbone,
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "dropout": args.dropout,
            "optimizer": args.optimizer,
            "freeze_backbone": args.freeze_backbone,
            "unfreeze_last_n": args.unfreeze_last_n,
            "weight_decay": args.weight_decay,
        }
        train(hp)
