"""
PlantHealth AI - Visualization Utilities
=========================================
Plotting functions for training metrics, confusion matrices, and predictions.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss', 
                 'train_acc', 'val_acc'
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Plot Loss
    ax1 = axes[0]
    if 'train_loss' in history:
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, len(epochs))
    
    # Plot Accuracy
    ax2 = axes[1]
    if 'train_acc' in history:
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    if 'val_acc' in history:
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    
    ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, len(epochs))
    
    # Add best epoch marker
    if 'val_acc' in history and len(history['val_acc']) > 0:
        best_epoch = np.argmax(history['val_acc']) + 1
        best_acc = max(history['val_acc'])
        ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
        ax2.annotate(
            f'Best: {best_acc:.2f}%\nEpoch {best_epoch}',
            xy=(best_epoch, best_acc),
            xytext=(best_epoch + 1, best_acc - 5),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='green', alpha=0.7)
        )
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    figsize: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
    save_path: Optional[str] = None,
    top_n: Optional[int] = None,
) -> plt.Figure:
    """
    Plot a confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array (num_classes x num_classes)
        class_names: List of class names
        title: Plot title
        figsize: Figure size (auto-calculated if None)
        normalize: Whether to normalize the confusion matrix
        save_path: Optional path to save the figure
        top_n: If set, only show top N most confused classes
        
    Returns:
        matplotlib Figure object
    """
    if normalize:
        cm_display = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        fmt = '.2f'
    else:
        cm_display = cm
        fmt = 'd'
    
    # Auto-calculate figure size
    if figsize is None:
        n_classes = len(class_names)
        if n_classes <= 10:
            figsize = (10, 8)
        elif n_classes <= 20:
            figsize = (14, 12)
        else:
            figsize = (20, 18)
    
    # Truncate class names for display
    display_names = [name[:25] + '...' if len(name) > 28 else name for name in class_names]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        cm_display,
        annot=len(class_names) <= 15,  # Only show numbers for small matrices
        fmt=fmt,
        cmap='Blues',
        xticklabels=display_names,
        yticklabels=display_names,
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    return fig


def plot_sample_predictions(
    images: List[np.ndarray],
    true_labels: List[str],
    pred_labels: List[str],
    confidences: Optional[List[float]] = None,
    title: str = "Sample Predictions",
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a grid of images with their true and predicted labels.
    
    Args:
        images: List of image arrays (H, W, C) in range [0, 1]
        true_labels: List of true class names
        pred_labels: List of predicted class names
        confidences: Optional list of prediction confidences
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    n_images = len(images)
    n_cols = min(5, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes for easy iteration
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else list(axes)
    else:
        axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        if idx < n_images:
            # Display image
            img = images[idx]
            if img.max() > 1:
                img = img / 255.0
            ax.imshow(img)
            
            # Format labels
            true_label = true_labels[idx]
            pred_label = pred_labels[idx]
            
            # Truncate long labels
            true_short = true_label[:20] + '...' if len(true_label) > 23 else true_label
            pred_short = pred_label[:20] + '...' if len(pred_label) > 23 else pred_label
            
            # Color based on correct/incorrect
            is_correct = true_label == pred_label
            color = 'green' if is_correct else 'red'
            
            # Create title text
            if confidences is not None:
                title_text = f"True: {true_short}\nPred: {pred_short}\nConf: {confidences[idx]:.1f}%"
            else:
                title_text = f"True: {true_short}\nPred: {pred_short}"
            
            ax.set_title(title_text, fontsize=9, color=color, fontweight='bold')
            ax.axis('off')
        else:
            ax.axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    return fig


def plot_augmentations(
    original: np.ndarray,
    augmented: List[np.ndarray],
    title: str = "Data Augmentation Examples",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot original image alongside augmented versions.
    
    Args:
        original: Original image array (H, W, C)
        augmented: List of augmented image arrays
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    n_augmented = len(augmented)
    n_total = 1 + n_augmented
    n_cols = min(4, n_total)
    n_rows = (n_total + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    # Flatten axes
    if n_rows == 1:
        axes = list(axes) if n_cols > 1 else [axes]
    else:
        axes = axes.flatten()
    
    # Plot original
    axes[0].imshow(original if original.max() <= 1 else original / 255.0)
    axes[0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Plot augmented versions
    for idx, aug_img in enumerate(augmented):
        ax = axes[idx + 1]
        img = aug_img if aug_img.max() <= 1 else aug_img / 255.0
        ax.imshow(img)
        ax.set_title(f'Augmented {idx + 1}', fontsize=10)
        ax.axis('off')
    
    # Hide unused axes
    for idx in range(n_total, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    return fig


def plot_class_distribution(
    distribution: Dict[str, int],
    title: str = "Class Distribution",
    figsize: Tuple[int, int] = (14, 10),
    top_n: Optional[int] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a bar chart of class distribution.
    
    Args:
        distribution: Dictionary mapping class names to counts
        title: Plot title
        figsize: Figure size
        top_n: If set, only show top N classes
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    # Sort by count
    sorted_items = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    
    if top_n is not None:
        sorted_items = sorted_items[:top_n]
    
    classes = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]
    
    # Truncate class names for display
    display_names = [name[:30] + '...' if len(name) > 33 else name for name in classes]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar chart
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(counts)))
    bars = ax.barh(range(len(counts)), counts, color=colors)
    
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(display_names, fontsize=9)
    ax.invert_yaxis()  # Top to bottom
    
    ax.set_xlabel('Number of Images', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{count:,}',
            va='center',
            fontsize=8
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    return fig


def plot_learning_rate(
    lrs: List[float],
    title: str = "Learning Rate Schedule",
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot learning rate over training.
    
    Args:
        lrs: List of learning rates per epoch/step
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(range(1, len(lrs) + 1), lrs, 'b-', linewidth=2)
    ax.set_xlabel('Step/Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    return fig
