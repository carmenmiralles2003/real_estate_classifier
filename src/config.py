"""
Central configuration for the real-estate image classification project.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# ── Dataset ────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Bedroom",
    "Coast",
    "Forest",
    "Highway",
    "Industrial",
    "Inside city",
    "Kitchen",
    "Living room",
    "Mountain",
    "Office",
    "Open country",
    "Store",
    "Street",
    "Suburb",
    "Tall building",
]
NUM_CLASSES = len(CLASS_NAMES)

# ── Image ──────────────────────────────────────────────────────────────
IMG_SIZE = 224
IMG_MEAN = [0.485, 0.456, 0.406]  # ImageNet stats
IMG_STD = [0.229, 0.224, 0.225]

# ── Training defaults ─────────────────────────────────────────────────
DEFAULT_HPARAMS = {
    "backbone": "efficientnet_b0",
    "pretrained": True,
    "freeze_backbone": True,
    "unfreeze_last_n": 2,       # unfreeze last N blocks when fine-tuning
    "dropout": 0.3,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "epochs": 25,
    "optimizer": "adam",         # "adam" | "sgd" | "adamw"
    "scheduler": "cosine",      # "cosine" | "step" | "none"
    "step_size": 7,
    "gamma": 0.1,
    "augmentation": "medium",   # "light" | "medium" | "heavy"
    "head_hidden_dim": 256,     # hidden layer size in classifier head
    "head_num_layers": 1,       # 1 or 2 hidden layers in head
    "early_stopping_patience": 5,
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "seed": 42,
    "num_workers": 4,
}

# ── Weights & Biases ──────────────────────────────────────────────────
WANDB_PROJECT = "real-estate-classifier"
WANDB_ENTITY = os.getenv("WANDB_ENTITY", "202525416-universidad-pontificia-comillas")

# ── API ────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
MODEL_PATH = MODELS_DIR / "best_model.pth"

# ── Streamlit ─────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", f"http://localhost:{API_PORT}")
