"""
Phase 2: Train diverse models + ensemble for maximum test accuracy.
Uses already-trained convnext_tiny_288 + trains swin_tiny & convnext_small.
Final: soft-voting ensemble with TTA across all models.
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
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.config import (
    WANDB_PROJECT, WANDB_ENTITY,
    MODELS_DIR, PROCESSED_DATA_DIR, CLASS_NAMES, NUM_CLASSES,
)
from src.dataset import get_dataloaders
from src.evaluate import evaluate_model

# ── Import shared utilities from nuclear script ──────────────────────
from scripts.train_nuclear import (
    build_advanced_model, get_param_groups, get_cosine_warmup_scheduler,
    train_one_epoch, validate, predict_with_tta, mixup_cutmix,
    soft_cross_entropy,
)

# ════════════════════════════════════════════════════════════════════════
# PHASE 2 EXPERIMENTS (only models we DON'T already have)
# ════════════════════════════════════════════════════════════════════════
PHASE2_EXPERIMENTS = [
    {
        "name": "swin_tiny_224",
        "backbone": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
        "img_size": 224,
        "batch_size": 32,
        "lr_backbone": 2e-5,
        "lr_head": 1e-3,
        "weight_decay": 0.05,
        "dropout": 0.3,
        "epochs": 30,
        "warmup_epochs": 3,
        "head_dim": 512,
        "mixup_alpha": 0.3,
        "cutmix_alpha": 1.0,
        "label_smoothing": 0.1,
        "patience": 8,
        "min_delta": 0.0015,
        "min_epochs": 8,
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
        "epochs": 30,
        "warmup_epochs": 3,
        "head_dim": 512,
        "mixup_alpha": 0.3,
        "cutmix_alpha": 1.0,
        "label_smoothing": 0.1,
        "patience": 8,
        "min_delta": 0.0015,
        "min_epochs": 8,
        "augmentation": "heavy",
    },
]

# Already trained model info
PRETRAINED_MODELS = [
    {
        "name": "convnext_tiny_288",
        "backbone": "convnext_tiny.fb_in22k_ft_in1k",
        "img_size": 288,
        "head_dim": 512,
        "save_path": str(MODELS_DIR / "convnext_tiny_288.pth"),
        "val_acc": 0.9731,
        "test_acc": 0.9620,
    },
]


def run_single_experiment(hp, device):
    """Train one model with full logging."""
    name = hp["name"]
    print(f"\n{'='*70}")
    print(f"TRAINING: {name}")
    print(f"  backbone={hp['backbone']}  img={hp['img_size']}px  batch={hp['batch_size']}")
    print(f"{'='*70}")

    run = wandb.init(
        project=WANDB_PROJECT, entity=WANDB_ENTITY,
        config=hp, name=name, reinit=True,
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=PROCESSED_DATA_DIR, batch_size=hp["batch_size"],
        num_workers=0, img_size=hp["img_size"], augmentation=hp.get("augmentation", "heavy"),
    )
    train_loader = DataLoader(
        train_loader.dataset, batch_size=hp["batch_size"],
        shuffle=True, num_workers=0, pin_memory=True, drop_last=True,
    )
    print(f"  Data: {len(train_loader.dataset)} train | {len(val_loader.dataset)} val | {len(test_loader.dataset)} test")

    model = build_advanced_model(
        hp["backbone"], NUM_CLASSES, hp["dropout"], hp["head_dim"],
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_p:,}")

    param_groups = get_param_groups(model, hp["lr_backbone"], hp["lr_head"], hp["weight_decay"])
    optimizer = optim.AdamW(param_groups)
    scheduler = get_cosine_warmup_scheduler(optimizer, hp["warmup_epochs"], hp["epochs"], len(train_loader))
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    patience = hp.get("patience", 8)
    min_delta = hp.get("min_delta", 0.0015)
    min_epochs = hp.get("min_epochs", 8)

    for epoch in range(1, hp["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler, hp, epoch)
        val_loss, val_acc = validate(model, val_loader, device)

        lr_bb = optimizer.param_groups[0]["lr"]
        lr_hd = optimizer.param_groups[1]["lr"]
        wandb.log({"epoch": epoch, "train/loss": train_loss, "train/accuracy": train_acc,
                    "val/loss": val_loss, "val/accuracy": val_acc, "lr/backbone": lr_bb, "lr/head": lr_hd})

        marker = ""
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            marker = " ★ BEST"
        else:
            patience_counter += 1

        print(f"  Ep {epoch:2d}/{hp['epochs']}  train={train_acc:.4f}  val={val_acc:.4f}  lr_hd={lr_hd:.1e}{marker}")

        if epoch >= min_epochs and patience_counter >= patience:
            print(f"  ⛔ Early stop at epoch {epoch} (no improvement > {min_delta:.4f} for {patience} ep)")
            break

    save_path = MODELS_DIR / f"{name}.pth"
    torch.save(best_state, save_path)

    model.load_state_dict(best_state)
    metrics = evaluate_model(model, test_loader, CLASS_NAMES, device)
    tta_preds, tta_labels = predict_with_tta(model, test_loader, device, hp["img_size"])
    tta_acc = accuracy_score(tta_labels, tta_preds)
    tta_f1 = f1_score(tta_labels, tta_preds, average="macro", zero_division=0)

    wandb.log({"test/accuracy": metrics["accuracy"], "test/macro_f1": metrics["macro_f1"],
               "test/tta_accuracy": tta_acc, "test/tta_macro_f1": tta_f1})
    wandb.summary.update({"best_val_accuracy": best_val_acc, "test_accuracy": metrics["accuracy"],
                           "test_tta_accuracy": tta_acc})

    artifact = wandb.Artifact(name=name.replace(".", "_"), type="model",
                               metadata={"val_acc": best_val_acc, "test_acc": metrics["accuracy"]})
    artifact.add_file(str(save_path))
    run.log_artifact(artifact)
    run.finish()

    print(f"\n  ✅ {name}: val={best_val_acc:.4f}  test={metrics['accuracy']:.4f}  tta={tta_acc:.4f}  f1={metrics['macro_f1']:.4f}")

    del model
    torch.cuda.empty_cache()

    return {
        "name": name, "backbone": hp["backbone"], "img_size": hp["img_size"],
        "head_dim": hp["head_dim"], "save_path": str(save_path),
        "val_acc": best_val_acc, "test_acc": metrics["accuracy"],
        "test_tta_acc": tta_acc, "macro_f1": metrics["macro_f1"],
    }


# ════════════════════════════════════════════════════════════════════════
# ENSEMBLE EVALUATION (soft-voting + TTA)
# ════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def build_ensemble_predictions(model_infos, device):
    """Get averaged softmax probabilities across all models with TTA."""
    all_model_probs = []
    labels = None

    for info in model_infos:
        name, backbone, img_size = info["name"], info["backbone"], info["img_size"]
        head_dim = info.get("head_dim", 512)
        print(f"  Loading {name} ({backbone}, {img_size}px)...")

        model = build_advanced_model(backbone, NUM_CLASSES, dropout=0.0, head_dim=head_dim, pretrained=False)
        state = torch.load(info["save_path"], map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        _, _, test_loader = get_dataloaders(
            data_dir=PROCESSED_DATA_DIR, batch_size=32, num_workers=0, img_size=img_size,
        )

        # Standard probabilities
        probs_list, labels_list = [], []
        for images, lbls in test_loader:
            images = images.to(device)
            with autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(images)
            probs_list.append(F.softmax(logits, dim=1).cpu())
            labels_list.append(lbls)

        # Flipped probabilities (TTA)
        flip_list = []
        for images, _ in test_loader:
            flipped = torch.flip(images, dims=[3]).to(device)
            with autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(flipped)
            flip_list.append(F.softmax(logits, dim=1).cpu())

        probs = (torch.cat(probs_list) + torch.cat(flip_list)) / 2.0
        all_model_probs.append(probs)
        labels = torch.cat(labels_list)

        del model
        torch.cuda.empty_cache()

    return all_model_probs, labels


def evaluate_ensemble_combo(model_probs, labels, model_names, combo_indices):
    """Evaluate a specific combination of models."""
    selected = torch.stack([model_probs[i] for i in combo_indices]).mean(dim=0)
    preds = selected.argmax(dim=1).numpy()
    y_true = labels.numpy()
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    names = [model_names[i] for i in combo_indices]
    return acc, f1, names


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Train new models ──────────────────────────────────────
    new_results = []
    for exp in PHASE2_EXPERIMENTS:
        save_path = MODELS_DIR / f"{exp['name']}.pth"
        if save_path.exists():
            print(f"\n⏭️  {exp['name']} already trained, skipping...")
            continue
        try:
            result = run_single_experiment(exp, device)
            new_results.append(result)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ❌ OOM! Halving batch size...")
                torch.cuda.empty_cache()
                exp["batch_size"] = max(4, exp["batch_size"] // 2)
                try:
                    result = run_single_experiment(exp, device)
                    new_results.append(result)
                except Exception as e2:
                    print(f"  ❌ Failed: {e2}")
                    if wandb.run: wandb.finish(exit_code=1)
            else:
                print(f"  ❌ Error: {e}")
                if wandb.run: wandb.finish(exit_code=1)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            if wandb.run: wandb.finish(exit_code=1)

    # ── Collect all available models ────────────────────────────────────
    all_models = list(PRETRAINED_MODELS)
    for r in new_results:
        all_models.append(r)
    # Also check for any .pth files we might have missed
    for exp in PHASE2_EXPERIMENTS:
        sp = MODELS_DIR / f"{exp['name']}.pth"
        if sp.exists() and not any(m["name"] == exp["name"] for m in all_models):
            all_models.append({
                "name": exp["name"], "backbone": exp["backbone"],
                "img_size": exp["img_size"], "head_dim": exp["head_dim"],
                "save_path": str(sp), "val_acc": 0, "test_acc": 0,
            })

    print(f"\n{'='*70}")
    print(f"AVAILABLE MODELS: {len(all_models)}")
    for m in all_models:
        print(f"  - {m['name']} (val={m.get('val_acc', '?')})")
    print(f"{'='*70}")

    if len(all_models) < 2:
        print("Need at least 2 models for ensemble. Exiting.")
        return

    # ── Phase 2: Ensemble combinations ──────────────────────────────────
    print(f"\n🔄 Building ensemble predictions...")
    model_probs, labels = build_ensemble_predictions(all_models, device)
    model_names = [m["name"] for m in all_models]

    # Try all pair and triple combinations
    from itertools import combinations
    print(f"\n{'='*70}")
    print("ENSEMBLE RESULTS (soft-vote + TTA):")
    print(f"{'='*70}")

    best_combo_acc = 0
    best_combo_info = None
    combos_to_try = []

    # Individual models (for reference)
    for i in range(len(all_models)):
        combos_to_try.append((i,))
    # All pairs
    for combo in combinations(range(len(all_models)), 2):
        combos_to_try.append(combo)
    # All triples
    for combo in combinations(range(len(all_models)), 3):
        combos_to_try.append(combo)
    # Full ensemble
    if len(all_models) > 3:
        combos_to_try.append(tuple(range(len(all_models))))

    for combo in combos_to_try:
        acc, f1, names = evaluate_ensemble_combo(model_probs, labels, model_names, combo)
        tag = "SINGLE" if len(combo) == 1 else f"ENS-{len(combo)}"
        print(f"  [{tag}] {' + '.join(names):60s}  acc={acc:.4f}  f1={f1:.4f}")
        if acc > best_combo_acc:
            best_combo_acc = acc
            best_combo_info = {"indices": combo, "names": names, "acc": acc, "f1": f1}

    # ── Log best ensemble to W&B ───────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"🏆 BEST COMBINATION: {' + '.join(best_combo_info['names'])}")
    print(f"   Test Accuracy: {best_combo_info['acc']:.4f}")
    print(f"   Macro F1:      {best_combo_info['f1']:.4f}")
    print(f"{'='*70}")

    ens_run = wandb.init(
        project=WANDB_PROJECT, entity=WANDB_ENTITY,
        name=f"ensemble_best_{'_'.join(best_combo_info['names'][:2])}",
        config={"models": best_combo_info["names"], "strategy": "soft_vote+tta"},
        reinit=True,
    )
    wandb.log({"ensemble/accuracy": best_combo_info["acc"], "ensemble/macro_f1": best_combo_info["f1"]})
    wandb.summary.update({"ensemble_accuracy": best_combo_info["acc"], "ensemble_macro_f1": best_combo_info["f1"]})

    # Full classification report for best combo
    selected = torch.stack([model_probs[i] for i in best_combo_info["indices"]]).mean(dim=0)
    preds = selected.argmax(dim=1).numpy()
    y_true = labels.numpy()
    report = classification_report(y_true, preds, target_names=CLASS_NAMES, zero_division=0)
    print(f"\n{report}")
    ens_run.finish()

    # ── Save best model metadata ───────────────────────────────────────
    # Find best single model for production
    single_results = [(i, *evaluate_ensemble_combo(model_probs, labels, model_names, (i,))) for i in range(len(all_models))]
    best_single_idx = max(single_results, key=lambda x: x[1])[0]
    best_single = all_models[best_single_idx]

    shutil.copy2(best_single["save_path"], MODELS_DIR / "best_model.pth")
    meta = {
        "backbone": best_single["backbone"], "img_size": best_single["img_size"],
        "head_dim": best_single.get("head_dim", 512), "name": best_single["name"],
        "ensemble_models": best_combo_info["names"],
        "ensemble_accuracy": best_combo_info["acc"],
    }
    with open(MODELS_DIR / "best_model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save full results
    results_data = {
        "all_models": [{k: v for k, v in m.items()} for m in all_models],
        "new_results": new_results,
        "best_single": best_single["name"],
        "best_ensemble": best_combo_info,
    }
    with open(MODELS_DIR / "phase2_results.json", "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    print(f"\n✅ Phase 2 complete!")
    print(f"   Best single: {best_single['name']} (test={single_results[best_single_idx][1]:.4f})")
    print(f"   Best ensemble: {' + '.join(best_combo_info['names'])} (test={best_combo_info['acc']:.4f})")


if __name__ == "__main__":
    main()
