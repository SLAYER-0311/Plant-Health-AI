"""
PlantHealth AI Backend - Pydantic Schemas
==========================================
Request and response models for the API.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    """Single prediction result."""
    class_name: str = Field(..., description="Predicted class name")
    plant: str = Field(..., description="Plant type extracted from class name")
    condition: str = Field(..., description="Disease or healthy status")
    confidence: float = Field(..., ge=0, le=100, description="Confidence percentage")
    class_index: int = Field(..., description="Class index in model output")


class OODInfo(BaseModel):
    """Out-of-Distribution detection information."""
    is_ood: bool = Field(..., description="Whether image is out-of-distribution (not a leaf)")
    max_probability: float = Field(..., description="Maximum class probability")
    entropy: float = Field(..., description="Prediction entropy (higher = more uncertain)")
    recommendation: str = Field(..., description="Human-readable recommendation")
    in_distribution_votes: int = Field(..., description="Number of methods that voted in-distribution")
    total_votes: int = Field(..., description="Total number of detection methods")


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    success: bool = Field(True, description="Whether prediction was successful")
    predictions: List[PredictionResult] = Field(..., description="Top-k predictions")
    top_prediction: Optional[PredictionResult] = Field(None, description="Most likely prediction (None if OOD)")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    ood_detection: Optional[OODInfo] = Field(None, description="Out-of-distribution detection results")
    warning: Optional[str] = Field(None, description="Warning message if OOD detected")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "predictions": [
                    {
                        "class_name": "Tomato___Early_blight",
                        "plant": "Tomato",
                        "condition": "Early blight",
                        "confidence": 95.5,
                        "class_index": 29
                    }
                ],
                "top_prediction": {
                    "class_name": "Tomato___Early_blight",
                    "plant": "Tomato",
                    "condition": "Early blight",
                    "confidence": 95.5,
                    "class_index": 29
                },
                "inference_time_ms": 45.2,
                "ood_detection": None,
                "warning": None
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = Field(False)
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device being used for inference")
    version: str = Field(..., description="API version")


class ClassInfo(BaseModel):
    """Information about a disease class."""
    index: int
    name: str
    plant: str
    condition: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    num_classes: int
    image_size: int
    device: str
    classes: List[ClassInfo]
