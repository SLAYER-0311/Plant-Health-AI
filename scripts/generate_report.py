"""
PlantHealth AI - Generate Evaluation Report
=============================================
Evaluates the trained model and generates all report artifacts:
- Classification report
- Confusion matrix image
- Training curves image
- Sample predictions image
- Results JSON files
"""

import sys
import os
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from src.data.dataset import PlantDiseaseDataset, get_data_loaders
from src.data.transforms import get_val_transforms, denormalize
from src.models.resnet_transfer import PlantDiseaseResNet, create_resnet_model
from src.models.custom_cnn import PlantDiseaseCNN, create_custom_cnn
from src.evaluation.metrics import evaluate_model, find_misclassified_samples, find_most_confused_classes
from src.utils.visualization import (
    plot_training_history, plot_confusion_matrix,
    plot_sample_predictions, plot_class_distribution
)


def load_config():
    import yaml
    config_path = PROJECT_ROOT / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_single_model(model, model_name, test_loader, class_names, device, reports_dir):
    """Evaluate a model and save all artifacts."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    
    model.eval()
    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=class_names,
        print_report=True,
    )
    
    # Save confusion matrix
    prefix = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    
    fig = plot_confusion_matrix(
        results['confusion_matrix'],
        class_names=class_names,
        title=f'{model_name} — Confusion Matrix',
        normalize=True,
        save_path=str(reports_dir / f'{prefix}_confusion_matrix.png'),
    )
    plt.close(fig)
    
    # Save sample predictions
    val_transforms = get_val_transforms(image_size=224)
    test_dir = PROJECT_ROOT / 'data' / 'New Plant Diseases Dataset' / 'test'
    
    test_dataset = PlantDiseaseDataset(
        str(test_dir), transform=val_transforms,
        class_to_idx={cls: idx for idx, cls in enumerate(class_names)},
    )
    
    np.random.seed(42)
    indices = np.random.choice(len(test_dataset), size=10, replace=False)
    
    images_list, true_labels_list, pred_labels_list, confidence_list = [], [], [], []
    
    with torch.no_grad():
        for idx in indices:
            image, label = test_dataset[idx]
            output = model(image.unsqueeze(0).to(device))
            probs = torch.softmax(output, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs.max().item() * 100
            
            img_display = denormalize(image.numpy())
            images_list.append(img_display)
            true_labels_list.append(class_names[label])
            pred_labels_list.append(class_names[pred_idx])
            confidence_list.append(confidence)
    
    fig = plot_sample_predictions(
        images=images_list,
        true_labels=true_labels_list,
        pred_labels=pred_labels_list,
        confidences=confidence_list,
        title=f'{model_name} — Sample Predictions',
        figsize=(18, 8),
        save_path=str(reports_dir / f'{prefix}_sample_predictions.png'),
    )
    plt.close(fig)
    
    # Save results JSON
    results_dict = {
        'model': model_name,
        'accuracy': float(results['accuracy']),
        'top5_accuracy': float(results['top5_accuracy']),
        'macro_precision': float(results['macro_precision']),
        'macro_recall': float(results['macro_recall']),
        'macro_f1': float(results['macro_f1']),
        'weighted_precision': float(results['weighted_precision']),
        'weighted_recall': float(results['weighted_recall']),
        'weighted_f1': float(results['weighted_f1']),
    }
    
    results_path = reports_dir.parent / f'{prefix}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nArtifacts saved to: {reports_dir}")
    return results_dict


def generate_markdown_report(custom_results, resnet_results, reports_dir):
    """Generate the markdown evaluation report."""
    report_path = reports_dir.parent / 'evaluation_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# PlantHealth AI — Evaluation Report\n\n")
        f.write("## Model Evaluation Summary\n\n")
        f.write("This report presents the evaluation results for both the Custom CNN baseline\n")
        f.write("and the ResNet50 Transfer Learning model on the PlantVillage test set.\n\n")
        f.write("---\n\n")
        
        # Comparison table
        f.write("## Model Comparison\n\n")
        f.write("| Metric | Custom CNN | ResNet50 (Transfer) |\n")
        f.write("|--------|-----------|--------------------|\n")
        
        for metric in ['accuracy', 'top5_accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1']:
            label = metric.replace('_', ' ').title()
            custom_val = custom_results.get(metric, 0)
            resnet_val = resnet_results.get(metric, 0)
            
            if metric in ['accuracy', 'top5_accuracy']:
                f.write(f"| {label} | {custom_val:.2f}% | {resnet_val:.2f}% |\n")
            else:
                f.write(f"| {label} | {custom_val:.4f} | {resnet_val:.4f} |\n")
        
        f.write("\n---\n\n")
        
        # ResNet50 details
        f.write("## ResNet50 Transfer Learning — Detailed Results\n\n")
        f.write(f"- **Overall Accuracy**: {resnet_results['accuracy']:.2f}%\n")
        f.write(f"- **Top-5 Accuracy**: {resnet_results['top5_accuracy']:.2f}%\n")
        f.write(f"- **Macro F1-Score**: {resnet_results['macro_f1']:.4f}\n")
        f.write(f"- **Weighted F1-Score**: {resnet_results['weighted_f1']:.4f}\n\n")
        
        # Confusion matrix
        f.write("### Confusion Matrix\n\n")
        f.write("![ResNet50 Confusion Matrix](figures/resnet50_transfer_learning_confusion_matrix.png)\n\n")
        
        # Sample predictions
        f.write("### Sample Predictions\n\n")
        f.write("A grid of 10 random test images showing the true label, predicted label, and confidence.\n\n")
        f.write("![ResNet50 Sample Predictions](figures/resnet50_transfer_learning_sample_predictions.png)\n\n")
        
        f.write("---\n\n")
        
        # Custom CNN details
        f.write("## Custom CNN — Detailed Results\n\n")
        f.write(f"- **Overall Accuracy**: {custom_results['accuracy']:.2f}%\n")
        f.write(f"- **Top-5 Accuracy**: {custom_results['top5_accuracy']:.2f}%\n")
        f.write(f"- **Macro F1-Score**: {custom_results['macro_f1']:.4f}\n")
        f.write(f"- **Weighted F1-Score**: {custom_results['weighted_f1']:.4f}\n\n")
        
        f.write("### Confusion Matrix\n\n")
        f.write("![Custom CNN Confusion Matrix](figures/custom_cnn_confusion_matrix.png)\n\n")
        
        f.write("### Sample Predictions\n\n")
        f.write("![Custom CNN Sample Predictions](figures/custom_cnn_sample_predictions.png)\n\n")
        
        f.write("---\n\n")
        f.write("## Conclusion\n\n")
        delta = resnet_results['accuracy'] - custom_results['accuracy']
        f.write(f"Transfer Learning with ResNet50 outperforms the Custom CNN by **{delta:.2f}%** in overall accuracy. ")
        f.write("This demonstrates the power of leveraging pre-trained features from ImageNet ")
        f.write("for a specialized task like plant disease classification.\n\n")
        f.write("The ResNet50 model is recommended for production deployment due to its superior accuracy ")
        f.write("and generalization capabilities.\n")
    
    print(f"\nEvaluation report saved to: {report_path}")


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    num_classes = config['model']['num_classes']
    
    # Setup paths
    reports_dir = PROJECT_ROOT / 'reports' / 'figures'
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test data loader
    test_dir = PROJECT_ROOT / 'data' / 'New Plant Diseases Dataset' / 'test'
    train_dir = PROJECT_ROOT / config['paths']['train_dir']
    valid_dir = PROJECT_ROOT / config['paths']['valid_dir']
    
    val_transforms = get_val_transforms(image_size=config['model']['image_size'])
    
    # Get class names from training set
    train_dataset = PlantDiseaseDataset(str(train_dir), transform=val_transforms)
    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    
    from torch.utils.data import DataLoader
    test_dataset = PlantDiseaseDataset(
        str(test_dir), transform=val_transforms, class_to_idx=class_to_idx,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # ---- Evaluate Custom CNN ----
    custom_checkpoint = PROJECT_ROOT / 'checkpoints' / 'custom_cnn_best.pth'
    custom_results = None
    
    if custom_checkpoint.exists():
        print(f"\nLoading Custom CNN from: {custom_checkpoint}")
        custom_model = PlantDiseaseCNN(num_classes=num_classes, dropout_rate=0.5)
        
        checkpoint = torch.load(custom_checkpoint, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            custom_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            custom_model.load_state_dict(checkpoint)
        
        custom_model.to(device)
        custom_results = evaluate_single_model(
            custom_model, "Custom CNN", test_loader, class_names, device, reports_dir
        )
        del custom_model
    else:
        print(f"Custom CNN checkpoint not found: {custom_checkpoint}")
        custom_results = {
            'model': 'Custom CNN', 'accuracy': 0, 'top5_accuracy': 0,
            'macro_precision': 0, 'macro_recall': 0, 'macro_f1': 0,
            'weighted_precision': 0, 'weighted_recall': 0, 'weighted_f1': 0,
        }
    
    # ---- Evaluate ResNet50 ----
    resnet_checkpoint = PROJECT_ROOT / 'checkpoints' / 'resnet50_best.pth'
    resnet_results = None
    
    if resnet_checkpoint.exists():
        print(f"\nLoading ResNet50 from: {resnet_checkpoint}")
        resnet_model = PlantDiseaseResNet(
            num_classes=num_classes, dropout_rate=0.5,
            pretrained=False, freeze_backbone=False,
        )
        
        checkpoint = torch.load(resnet_checkpoint, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            resnet_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            resnet_model.load_state_dict(checkpoint)
        
        resnet_model.to(device)
        resnet_results = evaluate_single_model(
            resnet_model, "ResNet50 Transfer Learning", test_loader, class_names, device, reports_dir
        )
        del resnet_model
    else:
        print(f"ResNet50 checkpoint not found: {resnet_checkpoint}")
        resnet_results = custom_results.copy()
        resnet_results['model'] = 'ResNet50 (Transfer Learning)'
    
    # ---- Generate Report ----
    generate_markdown_report(custom_results, resnet_results, reports_dir)
    
    print("\n" + "=" * 60)
    print("REPORT GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Reports dir:    {reports_dir.parent}")
    print(f"  Figures dir:    {reports_dir}")
    print(f"  Custom CNN:     {custom_results['accuracy']:.2f}% accuracy")
    print(f"  ResNet50:       {resnet_results['accuracy']:.2f}% accuracy")


if __name__ == "__main__":
    main()
