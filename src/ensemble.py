"""
Ensemble inference: combine predictions from multiple trained models.

Usage
-----
    python -m src.ensemble --models models/eff_b0.pth models/resnet50.pth models/mobilenet.pth \
                           --backbones efficientnet_b0 resnet50 mobilenetv2_100
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from src.config import CLASS_NAMES, NUM_CLASSES, MODELS_DIR, PROCESSED_DATA_DIR, DEFAULT_HPARAMS
from src.model import load_model_for_inference
from src.dataset import get_dataloaders
from src.evaluate import evaluate_model, plot_confusion_matrix


class EnsembleModel:
    """
    Ensemble of multiple trained models using soft-voting (average probabilities).
    """

    def __init__(self, models: list, device: str = "cpu"):
        self.models = models
        self.device = device
        for m in self.models:
            m.to(device)
            m.eval()

    @torch.no_grad()
    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        """Average softmax probabilities across all models."""
        images = images.to(self.device)
        all_probs = []
        for model in self.models:
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)
        # Average across models
        avg_probs = torch.stack(all_probs).mean(dim=0)
        return avg_probs

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Returns averaged logits (log of averaged probs) for compatibility."""
        avg_probs = self.predict_proba(images)
        return torch.log(avg_probs + 1e-8)


def evaluate_ensemble(
    model_paths: list[str],
    backbones: list[str],
    data_dir: Path = PROCESSED_DATA_DIR,
    device: str = "cpu",
):
    """
    Load multiple models, build an ensemble, and evaluate on the test set.
    """
    assert len(model_paths) == len(backbones), \
        "Must provide one backbone name per model checkpoint"

    print(f"Building ensemble with {len(model_paths)} models...")
    models = []
    for path, backbone in zip(model_paths, backbones):
        print(f"  Loading {backbone} from {path}")
        m = load_model_for_inference(path, backbone=backbone, device=device)
        models.append(m)

    ensemble = EnsembleModel(models, device=device)

    # Get test loader
    _, _, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=DEFAULT_HPARAMS["batch_size"],
        num_workers=DEFAULT_HPARAMS["num_workers"],
    )

    print(f"Evaluating ensemble on {len(test_loader.dataset)} test images...")
    metrics = evaluate_model(ensemble, test_loader, CLASS_NAMES, device)

    print("\n" + "=" * 60)
    print("ENSEMBLE RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1:      {metrics['macro_f1']:.4f}")
    print("\n" + metrics["report_str"])

    # Save confusion matrix
    fig = plot_confusion_matrix(
        metrics["y_true"], metrics["y_pred"], CLASS_NAMES,
        save_path=MODELS_DIR / "ensemble_confusion_matrix.png",
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Ensemble evaluation")
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="Paths to model checkpoints",
    )
    parser.add_argument(
        "--backbones", nargs="+", required=True,
        help="Backbone names corresponding to each model",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    evaluate_ensemble(
        model_paths=args.models,
        backbones=args.backbones,
        device=args.device,
    )


if __name__ == "__main__":
    main()
