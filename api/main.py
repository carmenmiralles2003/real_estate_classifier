"""
FastAPI inference service for real-estate image classification.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import MODEL_PATH, MODELS_DIR, CLASS_NAMES, NUM_CLASSES, DEFAULT_HPARAMS
from src.model import load_model_for_inference
from src.dataset import get_val_transforms
from api.schemas import (
    PredictionResult,
    HealthResponse,
    ErrorResponse,
    ModelInfo,
    ModelListResponse,
    ModelSelectRequest,
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# ── Global state ──────────────────────────────────────────────────────
model = None
device = None
transform = None
model_name = "unknown"
model_backbone = DEFAULT_HPARAMS["backbone"]
model_img_size = 224
available_models = []


def load_model_catalog():
    """Load available model metadata from disk."""
    catalog = []
    results_path = MODELS_DIR / "phase2_results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)

        for item in results.get("all_models", []):
            save_path = item.get("save_path")
            checkpoint = Path(save_path).name if save_path else f"{item['name']}.pth"
            checkpoint_path = MODELS_DIR / checkpoint
            if not checkpoint_path.exists():
                continue

            catalog.append({
                "name": item["name"],
                "backbone": item["backbone"],
                "img_size": item["img_size"],
                "head_dim": item.get("head_dim", 256),
                "checkpoint": checkpoint,
                "val_acc": item.get("val_acc"),
                "test_acc": item.get("test_acc"),
                "test_tta_acc": item.get("test_tta_acc"),
                "macro_f1": item.get("macro_f1"),
            })

    meta_path = MODELS_DIR / "best_model_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

        meta_name = meta.get("name")
        if meta_name and not any(item["name"] == meta_name for item in catalog):
            checkpoint = meta.get("checkpoint", f"{meta_name}.pth")
            if (MODELS_DIR / checkpoint).exists():
                catalog.append({
                    "name": meta_name,
                    "backbone": meta.get("backbone", DEFAULT_HPARAMS["backbone"]),
                    "img_size": meta.get("img_size", 224),
                    "head_dim": meta.get("head_dim", 256),
                    "checkpoint": checkpoint,
                    "val_acc": None,
                    "test_acc": None,
                    "test_tta_acc": None,
                    "macro_f1": None,
                })

    return sorted(catalog, key=lambda item: item["name"])


def persist_selected_model(meta):
    """Persist current selection so restarts keep the chosen model."""
    meta_path = MODELS_DIR / "best_model_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "backbone": meta["backbone"],
            "img_size": meta["img_size"],
            "head_dim": meta["head_dim"],
            "name": meta["name"],
            "checkpoint": meta["checkpoint"],
        }, f, indent=2)


def set_active_model(meta):
    """Load the requested model checkpoint into memory."""
    global model, transform, model_name, model_backbone, model_img_size

    checkpoint_path = MODELS_DIR / meta["checkpoint"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    backbone = meta["backbone"]
    img_size = meta["img_size"]
    head_dim = meta.get("head_dim", 256)

    transform = get_val_transforms(img_size)

    try:
        from scripts.train_nuclear import build_advanced_model

        loaded_model = build_advanced_model(backbone, NUM_CLASSES, dropout=0.0, head_dim=head_dim, pretrained=False)
        state = torch.load(str(checkpoint_path), map_location=str(device), weights_only=True)
        loaded_model.load_state_dict(state)
        loaded_model.to(device)
        loaded_model.eval()
        model = loaded_model
    except Exception as primary_err:
        # Fallback: try loading with build_model using matching head params
        try:
            from src.model import build_model
            loaded_model = build_model(
                backbone=backbone,
                num_classes=NUM_CLASSES,
                pretrained=False,
                freeze_backbone=False,
                dropout=0.0,
                head_hidden_dim=head_dim,
                head_num_layers=2,
            )
            state = torch.load(str(checkpoint_path), map_location=str(device), weights_only=True)
            loaded_model.load_state_dict(state)
            loaded_model.to(device)
            loaded_model.eval()
            model = loaded_model
        except Exception:
            # Last resort: standard load
            model = load_model_for_inference(
                checkpoint_path=str(checkpoint_path),
                backbone=backbone,
                device=str(device),
            )

    model_name = meta["name"]
    model_backbone = backbone
    model_img_size = img_size
    persist_selected_model(meta)
    print(f"Active model set to {model_name} ({backbone}) @ {img_size}px on {device}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global device, available_models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    available_models = load_model_catalog()
    meta_path = MODELS_DIR / "best_model_meta.json"
    selected_meta = None

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        selected_meta = next((item for item in available_models if item["name"] == meta.get("name")), None)

    if selected_meta is None and available_models:
        selected_meta = next((item for item in available_models if item["checkpoint"] == MODEL_PATH.name), None)

    if selected_meta is None and available_models:
        selected_meta = available_models[0]

    if selected_meta is None:
        print("WARNING: No valid model metadata found. The /predict endpoint will return 503 until a model is available.")
    else:
        set_active_model(selected_meta)

    yield  # app runs here

    # Cleanup (nothing to clean)


app = FastAPI(
    title="Real-Estate Image Classifier API",
    description=(
        "Classifies property images into 15 scene types "
        "using fine-tuned transfer learning models (ConvNeXt, EfficientNet, Swin, etc.)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    """Basic root endpoint to avoid confusing 404 on /."""
    return {
        "message": "Real-Estate Image Classifier API",
        "docs": "/docs",
        "health": "/health",
        "models": "/models",
    }

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model availability."""
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        device=str(device),
        model_name=model_name,
        backbone=model_backbone,
        img_size=model_img_size,
    )


@app.get("/models", response_model=ModelListResponse, tags=["System"])
async def list_models():
    """List available models and the active selection."""
    return ModelListResponse(
        active_model=model_name,
        models=[ModelInfo(**item) for item in available_models],
    )


@app.post("/models/select", response_model=HealthResponse, tags=["System"])
async def select_model(request: ModelSelectRequest):
    """Switch the active model without restarting the API."""
    selected_meta = next((item for item in available_models if item["name"] == request.name), None)
    if selected_meta is None:
        raise HTTPException(status_code=404, detail=f"Model '{request.name}' not found")

    try:
        set_active_model(selected_meta)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load model '{request.name}': {exc}")

    return await health_check()


@app.post(
    "/predict",
    response_model=PredictionResult,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    tags=["Prediction"],
    summary="Classify a real-estate image",
)
async def predict(file: UploadFile = File(..., description="Image file (JPEG, PNG, or WebP)")):
    """
    Upload a property image and receive the predicted room type
    with a confidence score and full probability distribution.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train a model first.")

    # Validate content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Allowed: {ALLOWED_CONTENT_TYPES}",
        )

    # Read and validate size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum: {MAX_FILE_SIZE // (1024*1024)} MB")

    # Decode image
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image. Upload a valid JPEG/PNG/WebP.")

    # Preprocess and predict
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()

    idx = probs.argmax().item()
    probabilities = {name: round(probs[i].item(), 4) for i, name in enumerate(CLASS_NAMES)}

    return PredictionResult(
        model_name=model_name,
        predicted_class=CLASS_NAMES[idx],
        confidence=round(probs[idx].item(), 4),
        probabilities=probabilities,
    )
