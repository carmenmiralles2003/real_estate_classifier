"""
Evaluation utilities: per-class metrics, confusion matrix, and visualization.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

from src.config import CLASS_NAMES


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    class_names: list[str] = CLASS_NAMES,
    device: str | torch.device = "cpu",
) -> dict:
    """
    Run inference on a DataLoader and compute per-class metrics.

    Returns
    -------
    dict with keys: accuracy, macro_f1, per_class, y_true, y_pred, report_str
    """
    model.eval()
    model.to(device)
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0,
    )
    report_str = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0,
    )

    per_class = {
        name: {
            "precision": report[name]["precision"],
            "recall": report[name]["recall"],
            "f1-score": report[name]["f1-score"],
            "support": report[name]["support"],
        }
        for name in class_names
    }

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class": per_class,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "report_str": report_str,
    }


def plot_confusion_matrix(
    y_true, y_pred, class_names=CLASS_NAMES, save_path=None,
):
    """Plot and optionally save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved → {save_path}")

    return fig
