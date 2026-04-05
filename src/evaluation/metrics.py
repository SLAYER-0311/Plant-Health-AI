"""
PlantHealth AI - Evaluation Metrics
====================================
Functions for computing classification metrics and generating evaluation reports.

Features:
- Confusion matrix computation
- Per-class precision, recall, F1-score
- Top-k accuracy
- Prediction extraction for visualization
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    top_k_accuracy_score,
)
from tqdm import tqdm


def get_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    return_probs: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Get model predictions on a dataset.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run inference on
        return_probs: Whether to return probability scores
        
    Returns:
        Tuple of (predictions, true_labels, probabilities or None)
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = [] if return_probs else None
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Getting predictions", leave=False):
            images = images.to(device, non_blocking=True)
            
            outputs = model(images)
            
            if return_probs:
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
            
            _, predicted = outputs.max(1)
            
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.numpy())
    
    predictions = np.concatenate(all_preds)
    true_labels = np.concatenate(all_labels)
    
    if return_probs:
        probabilities = np.concatenate(all_probs)
        return predictions, true_labels, probabilities
    
    return predictions, true_labels, None


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = False,
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Whether to normalize by row (true class)
        
    Returns:
        Confusion matrix array
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    return cm


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_dict: bool = False,
) -> str | Dict:
    """
    Compute classification report with per-class metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        output_dict: Return as dictionary if True
        
    Returns:
        Classification report string or dictionary
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=output_dict,
        zero_division=0,
    )


def per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute accuracy for each class.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        
    Returns:
        Dictionary mapping class name/index to accuracy
    """
    classes = np.unique(y_true)
    accuracies = {}
    
    for cls in classes:
        mask = y_true == cls
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).mean() * 100
            key = class_names[cls] if class_names else str(cls)
            accuracies[key] = acc
    
    return accuracies


def compute_topk_accuracy(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    k: int = 5,
) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities (N x num_classes)
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy as percentage
    """
    return top_k_accuracy_score(y_true, y_probs, k=k) * 100


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    print_report: bool = True,
) -> Dict[str, any]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        dataloader: Validation/test data loader
        device: Device to run inference on
        class_names: List of class names
        print_report: Whether to print the classification report
        
    Returns:
        Dictionary with all evaluation metrics
    """
    # Get predictions
    predictions, true_labels, probabilities = get_predictions(
        model, dataloader, device, return_probs=True
    )
    
    # Overall accuracy
    accuracy = accuracy_score(true_labels, predictions) * 100
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average=None, zero_division=0
    )
    
    # Macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted', zero_division=0
    )
    
    # Top-5 accuracy
    top5_accuracy = compute_topk_accuracy(true_labels, probabilities, k=5)
    
    # Confusion matrix
    cm = compute_confusion_matrix(true_labels, predictions)
    cm_normalized = compute_confusion_matrix(true_labels, predictions, normalize=True)
    
    # Per-class accuracy
    class_accuracies = per_class_accuracy(true_labels, predictions, class_names)
    
    # Classification report
    report = compute_classification_report(
        true_labels, predictions, class_names, output_dict=True
    )
    
    # Print report
    if print_report:
        print("\n" + "=" * 60)
        print("MODEL EVALUATION REPORT")
        print("=" * 60)
        print(f"\nOverall Accuracy: {accuracy:.2f}%")
        print(f"Top-5 Accuracy: {top5_accuracy:.2f}%")
        print(f"\nMacro Average:")
        print(f"  Precision: {macro_precision:.4f}")
        print(f"  Recall: {macro_recall:.4f}")
        print(f"  F1-Score: {macro_f1:.4f}")
        print(f"\nWeighted Average:")
        print(f"  Precision: {weighted_precision:.4f}")
        print(f"  Recall: {weighted_recall:.4f}")
        print(f"  F1-Score: {weighted_f1:.4f}")
        
        print("\n" + "-" * 60)
        print("Per-Class Classification Report:")
        print("-" * 60)
        print(compute_classification_report(true_labels, predictions, class_names))
    
    return {
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'per_class_support': support,
        'per_class_accuracy': class_accuracies,
        'confusion_matrix': cm,
        'confusion_matrix_normalized': cm_normalized,
        'classification_report': report,
        'predictions': predictions,
        'true_labels': true_labels,
        'probabilities': probabilities,
    }


def find_misclassified_samples(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    n_samples: int = 10,
    class_names: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Find the most confidently misclassified samples.
    
    Args:
        predictions: Predicted labels
        true_labels: True labels
        probabilities: Prediction probabilities
        n_samples: Number of samples to return
        class_names: Optional list of class names
        
    Returns:
        List of dictionaries with misclassified sample info
    """
    # Find misclassified indices
    misclassified = predictions != true_labels
    misclassified_indices = np.where(misclassified)[0]
    
    if len(misclassified_indices) == 0:
        return []
    
    # Get confidence for misclassified samples
    confidences = probabilities[misclassified_indices, predictions[misclassified_indices]]
    
    # Sort by confidence (most confident mistakes first)
    sorted_indices = np.argsort(confidences)[::-1][:n_samples]
    
    results = []
    for idx in sorted_indices:
        sample_idx = misclassified_indices[idx]
        pred = predictions[sample_idx]
        true = true_labels[sample_idx]
        conf = confidences[idx] * 100
        
        results.append({
            'index': int(sample_idx),
            'predicted': class_names[pred] if class_names else int(pred),
            'true': class_names[true] if class_names else int(true),
            'confidence': float(conf),
            'predicted_idx': int(pred),
            'true_idx': int(true),
        })
    
    return results


def find_most_confused_classes(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    top_n: int = 10,
) -> List[Dict]:
    """
    Find the most commonly confused class pairs.
    
    Args:
        confusion_matrix: Confusion matrix
        class_names: List of class names
        top_n: Number of pairs to return
        
    Returns:
        List of dictionaries with confused class pair info
    """
    cm = confusion_matrix.copy()
    
    # Zero out diagonal (correct predictions)
    np.fill_diagonal(cm, 0)
    
    # Find top confused pairs
    pairs = []
    for _ in range(top_n):
        if cm.max() == 0:
            break
        
        # Find max confusion
        idx = np.unravel_index(np.argmax(cm), cm.shape)
        true_class = idx[0]
        pred_class = idx[1]
        count = cm[idx]
        
        pairs.append({
            'true_class': class_names[true_class],
            'predicted_class': class_names[pred_class],
            'count': int(count),
            'true_idx': int(true_class),
            'pred_idx': int(pred_class),
        })
        
        # Zero out this pair
        cm[idx] = 0
    
    return pairs
