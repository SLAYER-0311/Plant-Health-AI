"""
PlantHealth AI Backend - Classifier Service
=============================================
Model loading and inference service.
"""

import json
import time
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.models.resnet_transfer import PlantDiseaseResNet
from src.utils.ood_detection import OODDetector, create_default_detector

from ..config import get_settings
from ..schemas.prediction import PredictionResult

logger = logging.getLogger(__name__)


# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PlantDiseaseClassifier:
    """
    Plant disease classification service.
    
    Handles model loading, image preprocessing, and inference.
    """
    
    def __init__(self, enable_ood_detection: bool = True, ood_strict: bool = True):
        self.settings = get_settings()
        self.model: Optional[nn.Module] = None
        self.class_names: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.device: torch.device = torch.device("cpu")
        self.is_loaded: bool = False
        
        # OOD detection - Use strict mode by default for better detection
        self.enable_ood_detection = enable_ood_detection
        self.ood_detector: Optional[OODDetector] = None
        if enable_ood_detection:
            self.ood_detector = create_default_detector(strict=ood_strict)
            logger.info(f"OOD detection enabled (strict={ood_strict})")
        
    def load_model(self) -> bool:
        """
        Load the trained model and class names.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Determine device
            if self.settings.device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.settings.device)
            
            logger.info(f"Using device: {self.device}")
            
            # Load class names
            class_names_path = self.settings.class_names_file
            if class_names_path.exists():
                with open(class_names_path, 'r') as f:
                    data = json.load(f)
                    # Handle both list and dict formats
                    if isinstance(data, list):
                        self.class_names = data
                        self.class_to_idx = {name: i for i, name in enumerate(data)}
                    else:
                        self.class_names = data.get('class_names', [])
                        self.class_to_idx = data.get('class_to_idx', {})
                logger.info(f"Loaded {len(self.class_names)} class names")
            else:
                logger.warning(f"Class names file not found: {class_names_path}")
                # Use default class names if file not found
                self.class_names = self._get_default_class_names()
                self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
            
            # Load model
            model_path = self.settings.model_file
            if model_path.exists():
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Create model architecture
                self.model = self._create_model(len(self.class_names))
                
                # Load weights
                if 'model_state_dict' in checkpoint:
                    try:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        logger.info(f"✓ Successfully loaded model weights from checkpoint")
                    except Exception as e:
                        logger.error(f"Failed to load state dict: {e}")
                        logger.info("Attempting to load with strict=False...")
                        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"Model loaded from: {model_path}")
                logger.info(f"Model on device: {next(self.model.parameters()).device}")
                self.is_loaded = True
                return True
            else:
                logger.warning(f"Model file not found: {model_path}")
                logger.info("Creating model with random weights for testing...")
                
                # Create model with random weights for testing
                self.model = self._create_model(len(self.class_names))
                self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True
                return True
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _create_model(self, num_classes: int) -> nn.Module:
        """Create PlantDiseaseResNet model architecture matching trained model."""
        # Use the same architecture as the trained model
        model = PlantDiseaseResNet(
            num_classes=num_classes,
            dropout_rate=0.5,
            pretrained=False,  # We'll load our own weights
            freeze_backbone=False,
        )
        
        return model
    
    def _get_default_class_names(self) -> List[str]:
        """Get default class names for the PlantVillage dataset."""
        return [
            "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
            "Apple___healthy", "Blueberry___healthy",
            "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
            "Corn_(maize)___Common_rust_", "Corn_(maize)___Northern_Leaf_Blight",
            "Corn_(maize)___healthy", "Grape___Black_rot",
            "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
            "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)",
            "Peach___Bacterial_spot", "Peach___healthy",
            "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
            "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
            "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
            "Strawberry___Leaf_scorch", "Strawberry___healthy",
            "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
            "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
            "Tomato___healthy"
        ]
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model inference.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed tensor ready for inference
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = image.resize(
            (self.settings.image_size, self.settings.image_size),
            Image.Resampling.BILINEAR
        )
        
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # Convert to tensor (HWC -> CHW)
        tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def parse_class_name(self, class_name: str) -> Tuple[str, str]:
        """
        Parse class name into plant and condition.
        
        Args:
            class_name: Full class name (e.g., "Tomato___Early_blight")
            
        Returns:
            Tuple of (plant, condition)
        """
        if "___" in class_name:
            parts = class_name.split("___")
            plant = parts[0].replace("_", " ").replace(",", ", ")
            condition = parts[1].replace("_", " ")
        else:
            plant = class_name
            condition = "Unknown"
        
        return plant, condition
    
    def _check_image_complexity(self, image: Image.Image) -> Tuple[bool, Dict[str, float]]:
        """
        Check if image has sufficient complexity to be a real leaf.
        Simple uniform patterns (like cloth) will have low complexity.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (is_complex_enough, metrics dict)
        """
        # Convert to numpy array
        img_array = np.array(image.convert('RGB'))
        
        # Calculate standard deviation per channel
        std_r = np.std(img_array[:, :, 0])
        std_g = np.std(img_array[:, :, 1])
        std_b = np.std(img_array[:, :, 2])
        avg_std = (std_r + std_g + std_b) / 3
        
        # Calculate edge density (real leaves have more edges/textures)
        from scipy.ndimage import sobel
        gray = np.mean(img_array, axis=2)
        edges_x = sobel(gray, axis=0)
        edges_y = sobel(gray, axis=1)
        edge_magnitude = np.sqrt(edges_x**2 + edges_y**2)
        edge_density = np.mean(edge_magnitude)
        
        # Real leaves typically have:
        # - avg_std > 10 (variation in color)
        # - edge_density > 8 (texture and veins)
        
        is_complex = avg_std > 10 and edge_density > 8
        
        metrics = {
            'avg_std': float(avg_std),
            'edge_density': float(edge_density),
            'std_r': float(std_r),
            'std_g': float(std_g),
            'std_b': float(std_b),
        }
        
        return is_complex, metrics
    
    def predict(
        self, 
        image: Image.Image, 
        top_k: int = 5,
        return_ood_scores: bool = False
    ) -> Tuple[List[PredictionResult], float, Optional[Dict]]:
        """
        Run inference on an image with optional OOD detection.
        
        Args:
            image: PIL Image
            top_k: Number of top predictions to return
            return_ood_scores: If True, return detailed OOD detection scores
            
        Returns:
            Tuple of (predictions list, inference time in ms, ood_info dict)
            - predictions: List of PredictionResult objects (empty if OOD detected)
            - inference_time: Time taken for inference in milliseconds
            - ood_info: Dictionary with OOD detection results (None if disabled)
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Check image complexity BEFORE inference (detect uniform patterns like cloth)
        if self.enable_ood_detection:
            is_complex, complexity_metrics = self._check_image_complexity(image)
            if not is_complex:
                logger.warning(f"Low complexity image detected - likely not a leaf. "
                             f"STD: {complexity_metrics['avg_std']:.2f}, "
                             f"Edges: {complexity_metrics['edge_density']:.2f}")
                
                ood_info = {
                    "is_ood": True,
                    "scores": {
                        "is_ood": True,
                        "max_probability": 0.0,
                        "entropy": 0.0,
                        "in_distribution_votes": 0,
                        "total_votes": 1,
                        "complexity_check": "failed",
                        **complexity_metrics
                    },
                    "recommendation": (
                        "⚠️ WARNING: This image appears too uniform to be a plant leaf! "
                        "It looks like a solid color or simple pattern (e.g., cloth, paper). "
                        "Please upload a clear image of an actual plant leaf with visible texture and veins."
                    )
                }
                return [], 0.1, ood_info
        
        # Preprocess
        tensor = self.preprocess_image(image).to(self.device)
        
        # Inference
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        inference_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # OOD Detection with stricter voting for production
        ood_info = None
        if self.enable_ood_detection and self.ood_detector is not None:
            # Use majority voting, but if confidence is low, be more strict
            is_ood, ood_scores = self.ood_detector.detect(
                outputs, 
                return_scores=True,
                voting_strategy="majority"
            )
            
            # Additional check: If max probability is suspiciously high but entropy suggests confusion
            # This catches cases like solid green cloth that fool the confidence check
            max_prob = ood_scores['max_probability']
            entropy = ood_scores['entropy']
            
            # If confidence is 60-85% AND entropy is moderate, treat as suspicious
            if 0.60 <= max_prob <= 0.85 and entropy > 1.5:
                # Override: require at least 3/4 votes for acceptance
                if ood_scores['in_distribution_votes'] < 3:
                    is_ood = True
                    ood_scores['is_ood'] = True
                    logger.warning(f"Suspicious image detected - confidence {max_prob:.3f} but moderate entropy {entropy:.3f}")
            
            ood_info = {
                "is_ood": is_ood,
                "scores": ood_scores,
                "recommendation": self.ood_detector.get_recommendation(ood_scores),
            }
            
            if is_ood:
                logger.warning(f"OOD image detected! Max prob: {ood_scores['max_probability']:.3f}, "
                             f"Entropy: {ood_scores['entropy']:.3f}")
                # Return empty predictions for OOD images
                return [], inference_time, ood_info
        
        # Get top-k predictions
        probs = probabilities[0].cpu().numpy()
        top_indices = np.argsort(probs)[::-1][:top_k]
        
        predictions = []
        for idx in top_indices:
            class_name = self.class_names[idx]
            plant, condition = self.parse_class_name(class_name)
            
            predictions.append(PredictionResult(
                class_name=class_name,
                plant=plant,
                condition=condition,
                confidence=float(probs[idx] * 100),
                class_index=int(idx),
            ))
        
        return predictions, inference_time, ood_info


# Global classifier instance
_classifier: Optional[PlantDiseaseClassifier] = None


def get_classifier() -> PlantDiseaseClassifier:
    """Get or create the global classifier instance."""
    global _classifier
    
    if _classifier is None:
        _classifier = PlantDiseaseClassifier()
        _classifier.load_model()
    
    return _classifier
