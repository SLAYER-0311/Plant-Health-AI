"""
Models module for PlantHealth AI.
Contains CNN architectures for plant disease classification.
"""

from .custom_cnn import PlantDiseaseCNN, create_custom_cnn
from .resnet_transfer import PlantDiseaseResNet, create_resnet_model

__all__ = [
    "PlantDiseaseCNN",
    "create_custom_cnn",
    "PlantDiseaseResNet",
    "create_resnet_model",
]
