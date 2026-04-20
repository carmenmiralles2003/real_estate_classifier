"""
Systematic experimentation script to find the best model.
Runs multiple configurations and saves each model separately.
Uses mixed precision (AMP) for faster GPU training.
"""
import copy
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.amp import autocast, GradScaler
import wandb

from src.config import (
    DEFAULT_HPARAMS, WANDB_PROJECT, WANDB_ENTITY,
    MODELS_DIR, PROCESSED_DATA_DIR, CLASS_NAMES, NUM_CLASSES,
)
from src.dataset import get_dataloaders
from src.model import build_model, count_parameters
from src.evaluate import evaluate_model


EXPERIMENTS = [
    # ── Exp 1: EfficientNet-B0 full fine-tune, AdamW, label smoothing ─
    {
        "name": "effb0_full_finetune",
        "backbone": "efficientnet_b0",
        "freeze_backbone": False,
        "learning_rate": 3e-4,
        "optimizer": "adamw",
        "weight_decay": 1e-2,
        "dropout": 0.3,
        "epochs": 50,
        "batch_size": 64,
        "scheduler": "cosine",
        "augmentation": "medium",
        "head_hidden_dim": 512,
        "head_num_layers": 2,
        "early_stopping_patience": 10,
    },
    # ── Exp 2: ResNet50 full fine-tune ────────────────────────────────
    {
        "name": "resnet50_full_finetune",
        "backbone": "resnet50",
        "freeze_backbone": False,
        "learning_rate": 1e-4,
        "optimizer": "adamw",
        "weight_decay": 1e-2,
        "dropout": 0.4,
        "epochs": 50,
        "batch_size": 32,
        "scheduler": "cosine",
        "augmentation": "heavy",
        "head_hidden_dim": 512,
        "head_num_layers": 2,
        "early_stopping_patience": 10,
    },
    # ── Exp 3: EfficientNet-B3 (large model, more capacity) ──────────
    {
        "name": "effb3_full_finetune",
        "backbone": "efficientnet_b3",
        "freeze_backbone": False,
        "learning_rate": 1e-4,
        "optimizer": "adamw",
        "weight_decay": 1e-2,
        "dropout": 0.4,
        "epochs": 50,
        "batch_size": 32,
        "scheduler": "cosine",
        "augmentation": "heavy",
        "head_hidden_dim": 512,
        "head_num_layers": 2,
        "early_stopping_patience": 10,
    },
    # ── Exp 4: EfficientNet-B0 partial freeze (last 4 blocks) ────────
    {
        "name": "effb0_partial_unfreeze4",
        "backbone": "efficientnet_b0",
        "freeze_backbone": True,
        "unfreeze_last_n": 4,
        "learning_rate": 5e-4,
        "optimizer": "adamw",
        "weight_decay": 5e-3,
        "dropout": 0.3,
        "epochs": 50,
        "batch_size": 64,
        "scheduler": "cosine",
        "augmentation": "heavy",
        "head_hidden_dim": 512,
        "head_num_layers": 1,
        "early_stopping_patience": 10,
    },
    # ── Exp 5: EfficientNet-B0 heavy augment, SGD + momentum ─────────
    {
        "name": "effb0_sgd_heavy_aug",
        "backbone": "efficientnet_b0",
        "freeze_backbone": False,
        "learning_rate": 1e-2,
        "optimizer": "sgd",
        "weight_decay": 1e-3,
        "dropout": 0.5,
        "epochs": 50,
        "batch_size": 64,
        "scheduler": "cosine",
        "augmentation": "heavy",
        "head_hidden_dim": 256,
        "head_num_layers": 1,
        "early_stopping_patience": 10,
    },
    # ── Exp 6: ResNet50 partial freeze (last 3 children) ─────────────
    {
        "name": "resnet50_partial_unfreeze3",
        "backbone": "resnet50",
        "freeze_backbone": True,
        "unfreeze_last_n": 3,
        "learning_rate": 3e-4,
        "optimizer": "adamw",
        "weight_decay": 1e-2,
        "dropout": 0.3,
        "epochs": 50,
        "batch_size": 64,
        "scheduler": "cosine",
        "augmentation": "medium",
        "head_hidden_dim": 512,
        "head_num_layers": 2,
        "early_stopping_patience": 10,
    },
]


def get_optimizer(model, hp):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if hp["optimizer"] == "sgd":
        return optim.SGD(params, lr=hp["learning_rate"], momentum=0.9, weight_decay=hp["weight_decay"])
    elif hp["optimizer"] == "adamw":
        return optim.AdamW(params, lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
    return optim.Adam(params, lr=hp["learning_rate"], weight_decay=hp["weight_decay"])


def get_scheduler(optimizer, hp):
    if hp["scheduler"] == "cosine":
        return CosineAnnealingLR(optimizer, T_max=hp["epochs"])
    elif hp["scheduler"] == "step":
        return StepLR(optimizer, step_size=7, gamma=0.1)
    return None


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
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
        with autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def run_experiment(hp, experiment_name, device):
    """Run a single experiment and return metrics."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"{'='*70}")
    
    full_hp = {**DEFAULT_HPARAMS, **hp}
    full_hp["num_workers"] = 0  # Windows compatibility

    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config=full_hp,
        name=experiment_name,
        reinit=True,
    )

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=PROCESSED_DATA_DIR,
        batch_size=full_hp["batch_size"],
        num_workers=0,
        augmentation=full_hp.get("augmentation", "medium"),
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # Model
    model = build_model(
        backbone=full_hp["backbone"],
        pretrained=True,
        dropout=full_hp["dropout"],
        freeze_backbone=full_hp["freeze_backbone"],
        unfreeze_last_n=full_hp.get("unfreeze_last_n", 2),
        head_hidden_dim=full_hp.get("head_hidden_dim", 256),
        head_num_layers=full_hp.get("head_num_layers", 1),
    ).to(device)

    total_p, train_p = count_parameters(model)
    print(f"Backbone: {full_hp['backbone']} | Total: {total_p:,} | Trainable: {train_p:,}")

    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing!
    optimizer = get_optimizer(model, full_hp)
    scheduler = get_scheduler(optimizer, full_hp)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    patience = full_hp["early_stopping_patience"]

    for epoch in range(1, full_hp["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        wandb.log({
            "epoch": epoch, "train/loss": train_loss, "train/accuracy": train_acc,
            "val/loss": val_loss, "val/accuracy": val_acc, "learning_rate": lr,
        })

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            marker = " ★ BEST"
        else:
            patience_counter += 1

        print(f"  Epoch {epoch:3d}/{full_hp['epochs']}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  lr={lr:.2e}{marker}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Save model
    save_path = MODELS_DIR / f"{experiment_name}.pth"
    torch.save(best_state, save_path)
    print(f"  Model saved → {save_path} (best_val_acc={best_val_acc:.4f})")

    # Test evaluation
    model.load_state_dict(best_state)
    metrics = evaluate_model(model, test_loader, CLASS_NAMES, device)

    wandb.log({
        "test/accuracy": metrics["accuracy"],
        "test/macro_f1": metrics["macro_f1"],
    })
    for cls_name in CLASS_NAMES:
        wandb.log({
            f"test/f1_{cls_name}": metrics["per_class"][cls_name]["f1-score"],
        })

    wandb.summary["best_val_accuracy"] = best_val_acc
    wandb.summary["test_accuracy"] = metrics["accuracy"]
    wandb.summary["test_macro_f1"] = metrics["macro_f1"]

    # Log artifact
    artifact = wandb.Artifact(name=experiment_name, type="model",
                               metadata={"val_accuracy": best_val_acc, "test_accuracy": metrics["accuracy"]})
    artifact.add_file(str(save_path))
    run.log_artifact(artifact)
    run.finish()

    print(f"\n  RESULT: val_acc={best_val_acc:.4f}  test_acc={metrics['accuracy']:.4f}  "
          f"macro_f1={metrics['macro_f1']:.4f}")
    print(f"  {metrics['report_str']}")

    return {
        "name": experiment_name,
        "backbone": full_hp["backbone"],
        "val_acc": best_val_acc,
        "test_acc": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "save_path": str(save_path),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for exp in EXPERIMENTS:
        name = exp.pop("name")
        result = run_experiment(exp, name, device)
        results.append(result)
        
        print(f"\n{'─'*70}")
        print("LEADERBOARD SO FAR:")
        for i, r in enumerate(sorted(results, key=lambda x: -x["test_acc"])):
            print(f"  {i+1}. {r['name']:30s}  val={r['val_acc']:.4f}  test={r['test_acc']:.4f}  f1={r['macro_f1']:.4f}")
        print(f"{'─'*70}")

    # Copy best model as best_model.pth
    best = max(results, key=lambda x: x["val_acc"])
    best_src = Path(best["save_path"])
    best_dst = MODELS_DIR / "best_model.pth"
    shutil.copy2(best_src, best_dst)
    print(f"\n{'='*70}")
    print(f"WINNER: {best['name']}  val={best['val_acc']:.4f}  test={best['test_acc']:.4f}")
    print(f"Saved as {best_dst}")
    print(f"{'='*70}")

    # Save results
    with open(MODELS_DIR / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
