"""
Transfer-learning model factory.

Supported backbones: efficientnet_b0, resnet50, mobilenetv2.
Uses timm for flexible model loading with pretrained ImageNet weights.
"""
import torch
import torch.nn as nn
import timm

from src.config import NUM_CLASSES, DEFAULT_HPARAMS


def build_model(
    backbone: str = DEFAULT_HPARAMS["backbone"],
    num_classes: int = NUM_CLASSES,
    pretrained: bool = DEFAULT_HPARAMS["pretrained"],
    dropout: float = DEFAULT_HPARAMS["dropout"],
    freeze_backbone: bool = DEFAULT_HPARAMS["freeze_backbone"],
    unfreeze_last_n: int = DEFAULT_HPARAMS["unfreeze_last_n"],
    head_hidden_dim: int = DEFAULT_HPARAMS["head_hidden_dim"],
    head_num_layers: int = DEFAULT_HPARAMS["head_num_layers"],
) -> nn.Module:
    """
    Builds a classifier on top of a pretrained backbone.

    Strategy
    --------
    1. Load a pretrained backbone from timm.
    2. Replace the classification head with a configurable MLP.
    3. Optionally freeze backbone, then unfreeze the last `unfreeze_last_n` blocks.

    Head variations (controlled by head_num_layers):
        1 layer:  BN → Drop → Linear(in, hidden) → ReLU → Drop → Linear(hidden, C)
        2 layers: BN → Drop → Linear(in, hidden) → ReLU → Drop →
                              Linear(hidden, hidden//2) → ReLU → Drop → Linear(hidden//2, C)
    """
    # Load backbone with no classifier head
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
    in_features = model.num_features

    # ── Freeze backbone ────────────────────────────────────────────────
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze last N children (blocks / layers)
        children = list(model.children())
        for child in children[-unfreeze_last_n:]:
            for param in child.parameters():
                param.requires_grad = True

    # ── Custom classification head ────────────────────────────────────
    layers = [
        nn.BatchNorm1d(in_features),
        nn.Dropout(dropout),
        nn.Linear(in_features, head_hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout / 2),
    ]

    if head_num_layers >= 2:
        mid_dim = head_hidden_dim // 2
        layers += [
            nn.Linear(head_hidden_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
        ]
        layers.append(nn.Linear(mid_dim, num_classes))
    else:
        layers.append(nn.Linear(head_hidden_dim, num_classes))

    classifier = nn.Sequential(*layers)

    # Wrap backbone + classifier
    full_model = nn.Sequential()
    full_model.add_module("backbone", model)
    full_model.add_module("classifier", classifier)

    return full_model


def count_parameters(model: nn.Module):
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_model_for_inference(
    checkpoint_path: str,
    backbone: str = DEFAULT_HPARAMS["backbone"],
    num_classes: int = NUM_CLASSES,
    device: str = "cpu",
) -> nn.Module:
    """Load a trained model from a checkpoint for inference."""
    model = build_model(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    m = build_model()
    total, trainable = count_parameters(m)
    print(f"Backbone: {DEFAULT_HPARAMS['backbone']}")
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
