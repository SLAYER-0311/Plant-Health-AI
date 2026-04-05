"""
PlantHealth AI Backend - Prediction Router
============================================
API endpoints for plant disease prediction.
"""

import logging
from typing import List
from io import BytesIO

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from PIL import Image

from ..config import get_settings, Settings
from ..schemas.prediction import (
    PredictionResponse,
    PredictionResult,
    ErrorResponse,
    ModelInfoResponse,
    ClassInfo,
    OODInfo,
)
from ..services.classifier import get_classifier

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Prediction"])

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

def validate_image(file: UploadFile) -> None:
    """Validate uploaded image file."""
    if file.filename:
        ext = file.filename.rsplit(".", 1)[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="File must be an image."
        )

@router.options("/predict")
async def options_predict():
    """Handle OPTIONS request for CORS preflight."""
    logger.info("OPTIONS /predict request received")
    return {
        "methods": ["POST", "OPTIONS"],
    }

@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Predict plant disease",
    description="Upload an image of a plant leaf to predict the disease or healthy status.",
)
async def predict(
    file: UploadFile = File(..., description="Image file of plant leaf"),
    settings: Settings = Depends(get_settings),
) -> PredictionResponse:
    """Predict disease based on uploaded image."""
    logger.info(f"Request details: method=[POST], filename={file.filename}, content_type={file.content_type}")

    validate_image(file)
    try:
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size: 10 MB"
            )

        image = Image.open(BytesIO(contents))
        classifier = get_classifier()
        predictions, inference_time, ood_info_dict = classifier.predict(image, top_k=settings.top_k)

        # Check if OOD detected
        is_ood = ood_info_dict and ood_info_dict.get("is_ood", False)
        
        # List of supported plants in the training dataset
        supported_plants = {
            "Apple", "Blueberry", "Cherry", "Corn", "Grape", "Orange", 
            "Peach", "Pepper", "Potato", "Raspberry", "Soybean", 
            "Squash", "Strawberry", "Tomato"
        }
        
        # Convert ood_info_dict to OODInfo schema if available
        ood_info_obj = None
        warning_msg = None
        if ood_info_dict:
            scores = ood_info_dict.get("scores", {})
            recommendation = ood_info_dict.get("recommendation", "")
            
            ood_info_obj = OODInfo(
                is_ood=scores.get("is_ood", False),
                max_probability=scores.get("max_probability", 0.0),
                entropy=scores.get("entropy", 0.0),
                recommendation=recommendation,
                in_distribution_votes=scores.get("in_distribution_votes", 0),
                total_votes=scores.get("total_votes", 0),
            )
            
            if ood_info_obj.is_ood:
                warning_msg = recommendation

        # Handle predictions
        filtered_predictions = [p for p in predictions if p.confidence >= settings.confidence_threshold]
        if not filtered_predictions and predictions:
            filtered_predictions = predictions[:1]  # At least one prediction

        # Check if predicted plant is supported (for banana, mango, etc.)
        if filtered_predictions and not is_ood:
            top_plant = filtered_predictions[0].plant
            # Extract base plant name (remove extra text in parentheses)
            base_plant = top_plant.split("(")[0].strip().split(",")[0].strip()
            
            if base_plant not in supported_plants and filtered_predictions[0].confidence < 70:
                warning_msg = (
                    f"⚠️ Note: '{base_plant}' may not be in our supported plant list. "
                    f"This model is trained on: {', '.join(sorted(supported_plants))}. "
                    f"For best results, please upload images of these plants only."
                )

        # Determine top prediction
        top_prediction = filtered_predictions[0] if filtered_predictions else None

        return PredictionResponse(
            success=True,
            predictions=filtered_predictions,
            top_prediction=top_prediction,
            inference_time_ms=round(inference_time, 2),
            ood_detection=ood_info_obj,
            warning=warning_msg,
        )

    except Exception as ex:
        logger.error(f"Prediction failed: {ex}")
        raise HTTPException(status_code=500, detail="Prediction failed.")

@router.get("/model-info")
async def get_model_info(settings: Settings = Depends(get_settings)) -> ModelInfoResponse:
    classifier = get_classifier()
    class_info = [
        ClassInfo(
            index=idx,
            name=name,
            plant="TBD",
            condition="TBD"
        )
        for idx, name in enumerate(classifier.class_names)
    ]
    
    return ModelInfoResponse(
        model_name="ResNet50",
        num_classes=len(classifier.class_names),
        image_size=settings.image_size,
        device=str(classifier.device),
        classes=class_info
    )