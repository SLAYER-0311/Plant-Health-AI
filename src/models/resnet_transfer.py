"""
PlantHealth AI - Transfer Learning with ResNet50
==================================================
ResNet50-based model for plant disease classification using transfer learning.

This module provides a pre-trained ResNet50 backbone with a custom classifier
head, supporting two-stage fine-tuning:
1. Stage 1: Train only the classifier (backbone frozen)
2. Stage 2: Fine-tune the entire model (backbone unfrozen)

Architecture:
    ResNet50 Backbone (pretrained on ImageNet)
    → AdaptiveAvgPool2d(1, 1)
    → Flatten
    → Linear(2048 → 512) + ReLU + Dropout
    → Linear(512 → num_classes)
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet34_Weights, MobileNet_V2_Weights


class PlantDiseaseResNet(nn.Module):
    """
    ResNet50-based model for plant disease classification.
    
    Uses a pretrained ResNet50 as the backbone with a custom classifier head.
    Supports freezing/unfreezing backbone layers for transfer learning.
    
    Args:
        num_classes: Number of output classes (default: 38)
        dropout_rate: Dropout probability in classifier (default: 0.5)
        pretrained: Whether to use ImageNet pretrained weights (default: True)
        freeze_backbone: Whether to freeze backbone initially (default: True)
        
    Input shape: (batch_size, 3, H, W) where H, W >= 32
    Output shape: (batch_size, num_classes)
    """
    
    def __init__(
        self,
        num_classes: int = 38,
        dropout_rate: float = 0.5,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Load pretrained ResNet50
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.backbone = models.resnet50(weights=weights)
            print("Loaded ResNet50 with ImageNet pretrained weights")
        else:
            self.backbone = models.resnet50(weights=None)
            print("Initialized ResNet50 with random weights")
        
        # Get the number of features from the backbone
        num_features = self.backbone.fc.in_features  # 2048 for ResNet50
        
        # Remove the original fully connected layer
        self.backbone.fc = nn.Identity()
        
        # Create custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )
        
        # Initialize classifier weights
        self._initialize_classifier()
        
        # Optionally freeze backbone
        if freeze_backbone:
            self.freeze_backbone()
    
    def _initialize_classifier(self):
        """Initialize classifier weights using He initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def freeze_backbone(self):
        """Freeze all backbone parameters (for Stage 1 training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen - only classifier will be trained")
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters (for Stage 2 fine-tuning)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen - entire model will be trained")
    
    def unfreeze_last_n_layers(self, n: int = 2):
        """
        Unfreeze only the last N layer groups of the backbone.
        
        ResNet50 layers: conv1, bn1, layer1, layer2, layer3, layer4
        
        Args:
            n: Number of layer groups to unfreeze from the end
        """
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Layer groups in ResNet50 (from early to late)
        layer_groups = ['layer1', 'layer2', 'layer3', 'layer4']
        
        # Unfreeze last n layer groups
        layers_to_unfreeze = layer_groups[-n:] if n > 0 else []
        
        for name, module in self.backbone.named_children():
            if name in layers_to_unfreeze:
                for param in module.parameters():
                    param.requires_grad = True
        
        print(f"Unfroze last {n} layer groups: {layers_to_unfreeze}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Classify
        output = self.classifier(features)
        
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the backbone.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Feature tensor of shape (batch_size, 2048)
        """
        return self.backbone(x)
    
    def count_parameters(self) -> Tuple[int, int]:
        """
        Count trainable and total parameters.
        
        Returns:
            Tuple of (trainable_params, total_params)
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
    
    def get_layer_groups(self) -> List[nn.Module]:
        """
        Get layer groups for differential learning rates.
        
        Returns:
            List of layer groups [early_layers, mid_layers, late_layers, classifier]
        """
        return [
            nn.ModuleList([self.backbone.conv1, self.backbone.bn1, self.backbone.layer1]),
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
            self.classifier,
        ]


class PlantDiseaseResNet34(nn.Module):
    """
    ResNet34-based model (lighter alternative to ResNet50).
    Same interface as PlantDiseaseResNet.
    """
    
    def __init__(
        self,
        num_classes: int = 38,
        dropout_rate: float = 0.5,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        if pretrained:
            weights = ResNet34_Weights.IMAGENET1K_V1
            self.backbone = models.resnet34(weights=weights)
            print("Loaded ResNet34 with ImageNet pretrained weights")
        else:
            self.backbone = models.resnet34(weights=None)
        
        num_features = self.backbone.fc.in_features  # 512 for ResNet34
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )
        
        self._initialize_classifier()
        
        if freeze_backbone:
            self.freeze_backbone()
    
    def _initialize_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)
    
    def count_parameters(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


class PlantDiseaseMobileNet(nn.Module):
    """
    MobileNetV2-based model (optimized for mobile deployment).
    Much smaller and faster than ResNet50.
    """
    
    def __init__(
        self,
        num_classes: int = 38,
        dropout_rate: float = 0.5,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        if pretrained:
            weights = MobileNet_V2_Weights.IMAGENET1K_V2
            self.backbone = models.mobilenet_v2(weights=weights)
            print("Loaded MobileNetV2 with ImageNet pretrained weights")
        else:
            self.backbone = models.mobilenet_v2(weights=None)
        
        num_features = self.backbone.classifier[1].in_features  # 1280 for MobileNetV2
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_features, num_classes),
        )
        
        if freeze_backbone:
            self.freeze_backbone()
    
    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        print("Backbone frozen")
    
    def unfreeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def count_parameters(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def create_resnet_model(
    model_name: str = "resnet50",
    num_classes: int = 38,
    dropout_rate: float = 0.5,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    pretrained_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Factory function to create a transfer learning model.
    
    Args:
        model_name: Model architecture ("resnet50", "resnet34", "mobilenet_v2")
        num_classes: Number of output classes
        dropout_rate: Dropout probability
        pretrained: Whether to use ImageNet pretrained weights
        freeze_backbone: Whether to freeze backbone initially
        pretrained_path: Optional path to custom pretrained weights
        device: Device to load the model on
        
    Returns:
        Model instance
    """
    model_name = model_name.lower()
    
    if model_name == "resnet50":
        model = PlantDiseaseResNet(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    elif model_name == "resnet34":
        model = PlantDiseaseResNet34(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    elif model_name in ["mobilenet", "mobilenet_v2"]:
        model = PlantDiseaseMobileNet(
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: resnet50, resnet34, mobilenet_v2")
    
    # Load custom weights if provided
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from: {pretrained_path}")
    
    # Move to device
    if device is not None:
        model = model.to(device)
    
    # Print summary
    trainable, total = model.count_parameters()
    print(f"\nModel: {model_name.upper()}")
    print(f"  Classes: {num_classes}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Total parameters: {total:,}")
    print(f"  Backbone frozen: {freeze_backbone}")
    
    return model
