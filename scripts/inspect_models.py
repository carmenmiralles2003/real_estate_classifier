"""Quick script to print architecture details for the report."""
import torch
from scripts.train_nuclear import build_advanced_model
from src.config import NUM_CLASSES

models_info = [
    ("convnext_small.fb_in22k_ft_in1k", "ConvNeXt-Small"),
    ("convnext_tiny.fb_in22k_ft_in1k", "ConvNeXt-Tiny"),
    ("swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", "Swin-Tiny"),
]

SEP = "=" * 70

for backbone, label in models_info:
    print(f"\n{SEP}")
    print(f"{label} ({backbone})")
    print(SEP)
    m = build_advanced_model(backbone, NUM_CLASSES, 0.4, 512, pretrained=False)

    bb = m.backbone
    total = sum(p.numel() for p in m.parameters())
    bb_total = sum(p.numel() for p in bb.parameters())
    head_total = sum(p.numel() for p in m.classifier.parameters())
    print(f"Total params: {total:,}")
    print(f"Backbone params: {bb_total:,}")
    print(f"Head params: {head_total:,}")
    print(f"num_features (backbone output dim): {bb.num_features}")

    # Per-stage params
    print("\nPer-component breakdown:")
    for name, child in bb.named_children():
        p = sum(pp.numel() for pp in child.parameters())
        if p > 0:
            if hasattr(child, "__len__"):
                for i, stage in enumerate(child):
                    sp = sum(pp.numel() for pp in stage.parameters())
                    if hasattr(stage, "blocks"):
                        n_blocks = len(stage.blocks)
                        print(f"  {name}[{i}]: {sp:>12,} params ({n_blocks} blocks)")
                    elif hasattr(stage, "layers"):
                        print(f"  {name}[{i}]: {sp:>12,} params (transformer layer)")
                    else:
                        print(f"  {name}[{i}]: {sp:>12,} params")
            else:
                print(f"  {name}: {p:>12,} params")

    # Head breakdown
    print("\nClassification Head:")
    for name, layer in m.classifier.named_children():
        p = sum(pp.numel() for pp in layer.parameters())
        if p > 0:
            print(f"  [{name}] {layer.__class__.__name__}: {p:,} params")
        else:
            print(f"  [{name}] {layer.__class__.__name__}")

    del m
    torch.cuda.empty_cache()
