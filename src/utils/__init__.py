"""
Utility functions for PlantHealth AI.
"""

# Import OOD detection (required for inference)
from .ood_detection import OODDetector, create_default_detector

# Visualization imports are optional (only needed for training)
try:
    from .visualization import (
        plot_training_history,
        plot_confusion_matrix,
        plot_sample_predictions,
        plot_augmentations,
        plot_class_distribution,
    )
    __all__ = [
        "OODDetector",
        "create_default_detector",
        "plot_training_history",
        "plot_confusion_matrix",
        "plot_sample_predictions",
        "plot_augmentations",
        "plot_class_distribution",
    ]
except ImportError:
    # Visualization not available (matplotlib not installed)
    __all__ = [
        "OODDetector",
        "create_default_detector",
    ]
