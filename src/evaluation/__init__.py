"""
Evaluation module for PlantHealth AI.
Contains metrics computation and analysis functions.
"""

from .metrics import (
    compute_confusion_matrix,
    compute_classification_report,
    evaluate_model,
    get_predictions,
    per_class_accuracy,
)

__all__ = [
    "compute_confusion_matrix",
    "compute_classification_report",
    "evaluate_model",
    "get_predictions",
    "per_class_accuracy",
]
