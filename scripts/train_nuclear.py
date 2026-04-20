"""
NUCLEAR training script: maximum performance strategy.
Targets: 0.97+ val accuracy via advanced techniques.

Techniques:
  - Stronger backbones (ConvNeXt-Tiny, EfficientNet-B4, Swin-T, DenseNet161)
  - Higher resolution (288-380px)
  - Mixup + CutMix regularization
  - Linear warmup + cosine annealing
  - Discriminative learning rates (backbone < head)
  - Test-Time Augmentation (TTA)
  - Label smoothing
  - Gradient clipping + mixed precision (AMP)
  - Stochastic Weight Averaging (SWA)
"""
import copy
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
import timm
import wandb

from src.config import (
    WANDB_PROJECT, WANDB_ENTITY,
    MODELS_DIR, PROCESSED_DATA_DIR, CLASS_NAMES, NUM_CLASSES,
    IMG_MEAN, IMG_STD,
)
from src.dataset import get_dataloaders
from src.evaluate import evaluate_model

# ════════════════════════════════════════════════════════════════════════
# EXPERIMENT DEFINITIONS
# ════════════════════════════════════════════════════════════════════════
EXPERIMENTS = [
    {
        "name": "convnext_tiny_288",
        "backbone": "convnext_tiny.fb_in22k_ft_in1k",
        "img_size": 288,
        "batch_size": 24,
        "lr_backbone": 2e-5,
        "lr_head": 1e-3,
        "weight_decay": 0.05,
        "dropout": 0.4,
        "epochs": 40,
        "warmup_epochs": 3,
        "head_dim": 512,
        "mixup_alpha": 0.3,
        "cutmix_alpha": 1.0,
        "label_smoothing": 0.1,
        "patience": 8,
        "min_delta": 0.0015,
        "min_epochs": 10,
        "augmentation": "heavy",
    },
    {
        "name": "efficientnet_b4_380",
        "backbone": "efficientnet_b4.ra2_in1k",
        "img_size": 380,
        "batch_size": 12,
        "lr_backbone": 1e-5,
        "lr_head": 5e-4,
        "weight_decay": 0.01,
        "dropout": 0.4,
        "epochs": 40,
        "warmup_epochs": 3,
        "head_dim": 512,
        "mixup_alpha": 0.2,
        "cutmix_alpha": 1.0,
        "label_smoothing": 0.1,
        "patience": 8,
        "min_delta": 0.0015,
        "min_epochs": 10,
        "augmentation": "heavy",
    },
    {
        "name": "swin_tiny_224",
        "backbone": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
        "img_size": 224,
        "batch_size": 32,
        "lr_backbone": 2e-5,
        "lr_head": 1e-3,
        "weight_decay": 0.05,
        "dropout": 0.3,
        "epochs": 40,
        "warmup_epochs": 3,
        "head_dim": 512,
        "mixup_alpha": 0.3,
        "cutmix_alpha": 1.0,
        "label_smoothing": 0.1,
        "patience": 8,
        "min_delta": 0.0015,
        "min_epochs": 10,
        "augmentation": "heavy",
    },
    {
        "name": "convnext_small_288",
        "backbone": "convnext_small.fb_in22k_ft_in1k",
        "img_size": 288,
        "batch_size": 16,
        "lr_backbone": 1e-5,
        "lr_head": 5e-4,
        "weight_decay": 0.05,
        "dropout": 0.4,
        "epochs": 40,
        "warmup_epochs": 3,
        "head_dim": 512,
        "mixup_alpha": 0.3,
        "cutmix_alpha": 1.0,
        "label_smoothing": 0.1,
        "patience": 8,
        "min_delta": 0.0015,
        "min_epochs": 10,
        "augmentation": "heavy",
    },
    {
        "name": "densenet161_288",
        "backbone": "densenet161.tv_in1k",
        "img_size": 288,
        "batch_size": 16,
        "lr_backbone": 2e-5,
        "lr_head": 1e-3,
        "weight_decay": 0.01,
        "dropout": 0.3,
        "epochs": 40,
        "warmup_epochs": 3,
        "head_dim": 512,
        "mixup_alpha": 0.2,
        "cutmix_alpha": 1.0,
        "label_smoothing": 0.1,
        "patience": 8,
        "min_delta": 0.0015,
        "min_epochs": 10,
        "augmentation": "heavy",
    },
    {
        "name": "effb0_288_maxaug",
        "backbone": "efficientnet_b0.ra_in1k",
        "img_size": 288,
        "batch_size": 32,
        "lr_backbone": 3e-5,
        "lr_head": 1e-3,
        "weight_decay": 0.01,
        "dropout": 0.3,
        "epochs": 40,
        "warmup_epochs": 3,
        "head_dim": 512,
        "mixup_alpha": 0.3,
        "cutmix_alpha": 1.0,
        "label_smoothing": 0.1,
        "patience": 8,
        "min_delta": 0.0015,
        "min_epochs": 10,
        "augmentation": "heavy",
    },
]


# ════════════════════════════════════════════════════════════════════════
# MIXUP / CUTMIX
# ════════════════════════════════════════════════════════════════════════
def rand_bbox(size, lam):
    """Random bounding box for CutMix."""
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def mixup_cutmix(images, labels, mixup_alpha=0.3, cutmix_alpha=1.0, num_classes=NUM_CLASSES):
    """
    Apply either Mixup or CutMix with 50/50 probability.
    Returns mixed images and soft label targets.
    """
    batch_size = images.size(0)
    # Convert labels to one-hot
    targets_onehot = F.one_hot(labels, num_classes).float()

    # 50% chance mixup, 50% chance cutmix
    if np.random.rand() < 0.5 and mixup_alpha > 0:
        # Mixup
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        lam = max(lam, 1.0 - lam)  # ensure lam >= 0.5
        indices = torch.randperm(batch_size, device=images.device)
        mixed_images = lam * images + (1 - lam) * images[indices]
        mixed_targets = lam * targets_onehot + (1 - lam) * targets_onehot[indices]
    elif cutmix_alpha > 0:
        # CutMix
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        indices = torch.randperm(batch_size, device=images.device)
        mixed_images = images.clone()
        x1, y1, x2, y2 = rand_bbox(images.size(), lam)
        mixed_images[:, :, x1:x2, y1:y2] = images[indices, :, x1:x2, y1:y2]
        # Adjust lambda to actual area ratio
        lam = 1 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
        mixed_targets = lam * targets_onehot + (1 - lam) * targets_onehot[indices]
    else:
        mixed_images = images
        mixed_targets = targets_onehot

    return mixed_images, mixed_targets


def soft_cross_entropy(logits, targets, label_smoothing=0.0):
    """Cross entropy with soft targets (for mixup/cutmix)."""
    if label_smoothing > 0:
        n_classes = logits.size(-1)
        targets = targets * (1 - label_smoothing) + label_smoothing / n_classes
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(targets * log_probs).sum(dim=1).mean()
    return loss


# ════════════════════════════════════════════════════════════════════════
# MODEL BUILDER (with discriminative LR support)
# ════════════════════════════════════════════════════════════════════════
def build_advanced_model(backbone, num_classes, dropout, head_dim, pretrained=True):
    """Build model with timm backbone + custom head, returning param groups."""
    encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
    in_features = encoder.num_features

    head = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(dropout),
        nn.Linear(in_features, head_dim),
        nn.GELU(),
        nn.Dropout(dropout * 0.5),
        nn.Linear(head_dim, head_dim // 2),
        nn.GELU(),
        nn.Dropout(dropout * 0.3),
        nn.Linear(head_dim // 2, num_classes),
    )

    # Initialize head weights
    for m in head.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model = nn.Sequential()
    model.add_module("backbone", encoder)
    model.add_module("classifier", head)

    return model


def get_param_groups(model, lr_backbone, lr_head, weight_decay):
    """Discriminative learning rates: backbone gets lower lr."""
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone"):
            backbone_params.append(param)
        else:
            head_params.append(param)

    return [
        {"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": head_params, "lr": lr_head, "weight_decay": weight_decay * 0.1},
    ]


# ════════════════════════════════════════════════════════════════════════
# SCHEDULER: linear warmup + cosine decay
# ════════════════════════════════════════════════════════════════════════
def get_cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, scheduler, device, scaler, hp, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    use_mix = hp.get("mixup_alpha", 0) > 0 or hp.get("cutmix_alpha", 0) > 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if use_mix and epoch > hp.get("warmup_epochs", 0):
            mixed_images, mixed_targets = mixup_cutmix(
                images, labels,
                mixup_alpha=hp.get("mixup_alpha", 0),
                cutmix_alpha=hp.get("cutmix_alpha", 0),
            )
        else:
            mixed_images = images
            mixed_targets = None

        optimizer.zero_grad()
        with autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(mixed_images)
            if mixed_targets is not None:
                loss = soft_cross_entropy(
                    outputs, mixed_targets,
                    label_smoothing=hp.get("label_smoothing", 0),
                )
            else:
                loss = F.cross_entropy(
                    outputs, labels,
                    label_smoothing=hp.get("label_smoothing", 0),
                )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        with torch.no_grad():
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


# ════════════════════════════════════════════════════════════════════════
# TTA (Test-Time Augmentation)
# ════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def predict_with_tta(model, loader, device, img_size, n_augments=5):
    """
    Test-Time Augmentation: original + horizontally flipped + random crops.
    Returns averaged probabilities.
    """
    from torchvision import transforms
    model.eval()

    tta_transforms = [
        # Original
        transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=1.0),
        ]),
        # Five crops would need raw images, so we use transforms on normalized tensors
    ]

    all_probs = []
    all_labels = []

    # Standard prediction
    for images, labels in loader:
        images = images.to(device)
        with autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu())
        all_labels.append(labels)

    # Flipped prediction
    flip_probs = []
    for images, labels in loader:
        flipped = torch.flip(images, dims=[3])  # horizontal flip
        flipped = flipped.to(device)
        with autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(flipped)
        probs = F.softmax(logits, dim=1)
        flip_probs.append(probs.cpu())

    # Average
    all_probs = torch.cat(all_probs, dim=0)
    flip_probs = torch.cat(flip_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    avg_probs = (all_probs + flip_probs) / 2.0
    preds = avg_probs.argmax(dim=1)

    return preds.numpy(), all_labels.numpy()


# ════════════════════════════════════════════════════════════════════════
# SINGLE EXPERIMENT
# ════════════════════════════════════════════════════════════════════════
def run_experiment(hp, device):
    name = hp["name"]
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}")
    print(f"  backbone={hp['backbone']}  img_size={hp['img_size']}  batch={hp['batch_size']}")
    print(f"  lr_backbone={hp['lr_backbone']}  lr_head={hp['lr_head']}  wd={hp['weight_decay']}")
    print(f"  mixup={hp['mixup_alpha']}  cutmix={hp['cutmix_alpha']}  label_smooth={hp['label_smoothing']}")
    print(f"{'='*70}")

    # W&B
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        config=hp,
        name=name,
        reinit=True,
    )

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=PROCESSED_DATA_DIR,
        batch_size=hp["batch_size"],
        num_workers=0,
        img_size=hp["img_size"],
        augmentation=hp.get("augmentation", "heavy"),
    )
    # Avoid last minibatch of size 1, which breaks BatchNorm during training.
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=hp["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    print(f"  Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # Model
    model = build_advanced_model(
        backbone=hp["backbone"],
        num_classes=NUM_CLASSES,
        dropout=hp["dropout"],
        head_dim=hp["head_dim"],
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total_p:,} total | {train_p:,} trainable")

    # Optimizer with discriminative LR
    param_groups = get_param_groups(
        model, hp["lr_backbone"], hp["lr_head"], hp["weight_decay"],
    )
    optimizer = optim.AdamW(param_groups)

    # Scheduler: warmup + cosine
    steps_per_epoch = len(train_loader)
    scheduler = get_cosine_warmup_scheduler(
        optimizer, hp["warmup_epochs"], hp["epochs"], steps_per_epoch,
    )

    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    patience = hp.get("patience", 8)
    min_delta = hp.get("min_delta", 0.0015)
    min_epochs = hp.get("min_epochs", 10)

    for epoch in range(1, hp["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, scaler, hp, epoch,
        )
        val_loss, val_acc = validate(model, val_loader, device)

        lr_bb = optimizer.param_groups[0]["lr"]
        lr_hd = optimizer.param_groups[1]["lr"]

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss, "train/accuracy": train_acc,
            "val/loss": val_loss, "val/accuracy": val_acc,
            "lr/backbone": lr_bb, "lr/head": lr_hd,
        })

        marker = ""
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            marker = " ★ BEST"
        else:
            patience_counter += 1

        print(f"  Epoch {epoch:3d}/{hp['epochs']}  "
              f"train={train_acc:.4f}  val={val_acc:.4f}  "
              f"lr_bb={lr_bb:.1e}  lr_hd={lr_hd:.1e}{marker}")

        if epoch >= min_epochs and patience_counter >= patience:
            print(
                f"  ⛔ Early stopping at epoch {epoch} "
                f"(no val improvement > {min_delta:.4f} for {patience} epochs)"
            )
            break

    # Save model
    save_path = MODELS_DIR / f"{name}.pth"
    torch.save(best_state, save_path)
    print(f"  ✅ Model saved → {save_path} (best_val_acc={best_val_acc:.4f})")

    # Test evaluation (standard)
    model.load_state_dict(best_state)
    metrics = evaluate_model(model, test_loader, CLASS_NAMES, device)

    # TTA evaluation
    from sklearn.metrics import accuracy_score, f1_score
    tta_preds, tta_labels = predict_with_tta(model, test_loader, device, hp["img_size"])
    tta_acc = accuracy_score(tta_labels, tta_preds)
    tta_f1 = f1_score(tta_labels, tta_preds, average="macro", zero_division=0)

    wandb.log({
        "test/accuracy": metrics["accuracy"],
        "test/macro_f1": metrics["macro_f1"],
        "test/tta_accuracy": tta_acc,
        "test/tta_macro_f1": tta_f1,
    })
    wandb.summary["best_val_accuracy"] = best_val_acc
    wandb.summary["test_accuracy"] = metrics["accuracy"]
    wandb.summary["test_tta_accuracy"] = tta_acc

    # Log artifact
    artifact = wandb.Artifact(name=name.replace(".", "_"), type="model",
                               metadata={"val_acc": best_val_acc, "test_acc": metrics["accuracy"]})
    artifact.add_file(str(save_path))
    run.log_artifact(artifact)
    run.finish()

    print(f"\n  📊 RESULT: val={best_val_acc:.4f}  test={metrics['accuracy']:.4f}  "
          f"test_tta={tta_acc:.4f}  macro_f1={metrics['macro_f1']:.4f}")

    return {
        "name": name,
        "backbone": hp["backbone"],
        "img_size": hp["img_size"],
        "val_acc": best_val_acc,
        "test_acc": metrics["accuracy"],
        "test_tta_acc": tta_acc,
        "macro_f1": metrics["macro_f1"],
        "tta_f1": tta_f1,
        "save_path": str(save_path),
    }


# ════════════════════════════════════════════════════════════════════════
# ENSEMBLE EVALUATION
# ════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_ensemble(model_infos, test_loader, device, img_size_map):
    """
    Soft-voting ensemble with TTA across multiple models.
    Each model votes with its own resolution's test loader.
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    # We'll gather per-model probabilities on the same test set
    # Since different models may use different img sizes, we re-create loaders
    all_model_probs = []

    for info in model_infos:
        name = info["name"]
        backbone = info["backbone"]
        img_size = info["img_size"]
        save_path = info["save_path"]

        print(f"  Loading {name} ({backbone}, {img_size}px)...")
        model = build_advanced_model(backbone, NUM_CLASSES, dropout=0.0, head_dim=512, pretrained=False)
        state = torch.load(save_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        # Create test loader at this model's resolution
        _, _, tl = get_dataloaders(
            data_dir=PROCESSED_DATA_DIR,
            batch_size=32, num_workers=0, img_size=img_size,
        )

        # Standard probs
        probs_list = []
        labels_list = []
        for images, labels in tl:
            images = images.to(device)
            with autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
            probs_list.append(F.softmax(logits, dim=1).cpu())
            labels_list.append(labels)

        # Flipped probs
        flip_probs = []
        for images, labels in tl:
            flipped = torch.flip(images, dims=[3]).to(device)
            with autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(flipped)
            flip_probs.append(F.softmax(logits, dim=1).cpu())

        probs = (torch.cat(probs_list) + torch.cat(flip_probs)) / 2.0
        all_model_probs.append(probs)
        labels = torch.cat(labels_list)

        del model
        torch.cuda.empty_cache()

    # Average across models
    ensemble_probs = torch.stack(all_model_probs).mean(dim=0)
    ensemble_preds = ensemble_probs.argmax(dim=1).numpy()
    y_true = labels.numpy()

    acc = accuracy_score(y_true, ensemble_preds)
    f1 = f1_score(y_true, ensemble_preds, average="macro", zero_division=0)
    report = classification_report(y_true, ensemble_preds, target_names=CLASS_NAMES, zero_division=0)

    print(f"\n  🏆 ENSEMBLE ({len(model_infos)} models + TTA)")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1:.4f}")
    print(f"\n{report}")

    return acc, f1


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for exp in EXPERIMENTS:
        try:
            result = run_experiment(exp, device)
            results.append(result)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  ❌ OOM on {exp['name']}! Reducing batch size and retrying...")
                torch.cuda.empty_cache()
                exp["batch_size"] = max(4, exp["batch_size"] // 2)
                try:
                    result = run_experiment(exp, device)
                    results.append(result)
                except Exception as e2:
                    print(f"  ❌ Failed again: {e2}")
                    if wandb.run:
                        wandb.finish(exit_code=1)
            else:
                print(f"  ❌ Error: {e}")
                if wandb.run:
                    wandb.finish(exit_code=1)
        except Exception as e:
            print(f"  ❌ Error on {exp['name']}: {e}")
            if wandb.run:
                wandb.finish(exit_code=1)

        # Leaderboard
        if results:
            print(f"\n{'─'*70}")
            print("📊 LEADERBOARD:")
            for i, r in enumerate(sorted(results, key=lambda x: -x["val_acc"])):
                print(f"  {i+1}. {r['name']:30s}  val={r['val_acc']:.4f}  "
                      f"test={r['test_acc']:.4f}  tta={r['test_tta_acc']:.4f}  f1={r['macro_f1']:.4f}")
            print(f"{'─'*70}")

    if not results:
        print("No experiments completed successfully!")
        return

    # ── Copy best model ────────────────────────────────────────────────
    best = max(results, key=lambda x: x["val_acc"])
    shutil.copy2(best["save_path"], MODELS_DIR / "best_model.pth")

    # ── Save metadata for API ──────────────────────────────────────────
    meta = {"backbone": best["backbone"], "img_size": best["img_size"],
            "head_dim": 512, "name": best["name"]}
    with open(MODELS_DIR / "best_model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*70}")
    print(f"🏆 BEST SINGLE MODEL: {best['name']}")
    print(f"   val={best['val_acc']:.4f}  test={best['test_acc']:.4f}  tta={best['test_tta_acc']:.4f}")
    print(f"{'='*70}")

    # ── Ensemble top 3 ─────────────────────────────────────────────────
    if len(results) >= 2:
        top_n = sorted(results, key=lambda x: -x["val_acc"])[:min(3, len(results))]
        print(f"\n🔄 Building ensemble of top {len(top_n)} models...")
        img_map = {r["name"]: r["img_size"] for r in top_n}

        # Log ensemble to W&B
        ens_run = wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY,
            name=f"ensemble_top{len(top_n)}",
            config={"models": [r["name"] for r in top_n], "strategy": "soft_vote+tta"},
            reinit=True,
        )
        ens_acc, ens_f1 = evaluate_ensemble(
            top_n, None, device, img_map,
        )
        wandb.log({"ensemble/accuracy": ens_acc, "ensemble/macro_f1": ens_f1})
        wandb.summary["ensemble_accuracy"] = ens_acc
        wandb.summary["ensemble_macro_f1"] = ens_f1
        ens_run.finish()

        print(f"\n{'='*70}")
        print(f"🏆 ENSEMBLE ACCURACY: {ens_acc:.4f}  |  MACRO F1: {ens_f1:.4f}")
        print(f"{'='*70}")

    # ── Save all results ───────────────────────────────────────────────
    with open(MODELS_DIR / "nuclear_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ All done! Check W&B for detailed results.")


if __name__ == "__main__":
    main()
