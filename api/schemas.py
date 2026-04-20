"""
Pydantic schemas for the FastAPI real-estate classifier.
"""
from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    model_name: str = Field(..., example="convnext_tiny_288", description="Model used for this prediction")
    predicted_class: str = Field(..., example="kitchen", description="Predicted room type")
    confidence: float = Field(..., ge=0, le=1, example=0.92, description="Confidence score")
    probabilities: dict[str, float] = Field(
        ...,
        description="Per-class probability distribution",
        example={"bathroom": 0.02, "bedroom": 0.01, "dining_room": 0.03,
                 "exterior": 0.01, "kitchen": 0.92, "living_room": 0.01},
    )


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    model_loaded: bool = Field(..., example=True)
    device: str = Field(..., example="cpu")
    model_name: str = Field(..., example="convnext_tiny_288")
    backbone: str = Field(..., example="convnext_tiny.fb_in22k_ft_in1k")
    img_size: int = Field(..., example=288)


class ModelInfo(BaseModel):
    name: str = Field(..., example="convnext_tiny_288")
    backbone: str = Field(..., example="convnext_tiny.fb_in22k_ft_in1k")
    img_size: int = Field(..., example=288)
    head_dim: int = Field(..., example=512)
    checkpoint: str = Field(..., example="convnext_tiny_288.pth")
    val_acc: float | None = Field(default=None, example=0.9731)
    test_acc: float | None = Field(default=None, example=0.9620)
    test_tta_acc: float | None = Field(default=None, example=0.9605)
    macro_f1: float | None = Field(default=None, example=0.9650)


class ModelListResponse(BaseModel):
    active_model: str = Field(..., example="convnext_tiny_288")
    models: list[ModelInfo]


class ModelSelectRequest(BaseModel):
    name: str = Field(..., example="convnext_tiny_288")


class ErrorResponse(BaseModel):
    detail: str = Field(..., example="Invalid image format")
