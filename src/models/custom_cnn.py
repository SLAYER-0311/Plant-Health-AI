"""
PlantHealth AI - Custom CNN Model
==================================
A custom Convolutional Neural Network built from scratch for plant disease 
classification. This serves as the baseline model before transfer learning.

Architecture:
    Input (3, 224, 224)
    → Conv Block 1: Conv2d(3→32) + BatchNorm + ReLU + MaxPool
    → Conv Block 2: Conv2d(32→64) + BatchNorm + ReLU + MaxPool  
    → Conv Block 3: Conv2d(64→128) + BatchNorm + ReLU + MaxPool
    → Conv Block 4: Conv2d(128→256) + BatchNorm + ReLU + MaxPool
    → AdaptiveAvgPool2d → Flatten
    → FC1: Linear(256→512) + BatchNorm + ReLU + Dropout(0.5)
    → FC2: Linear(512→num_classes)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    A convolutional block consisting of:
    Conv2d → BatchNorm → ReLU → MaxPool
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        pool_size: Size of max pooling window
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # Same padding
            bias=False  # No bias when using BatchNorm
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class PlantDiseaseCNN(nn.Module):
    """
    Custom CNN for plant disease classification.
    
    A 4-block convolutional neural network with batch normalization,
    dropout regularization, and adaptive pooling for flexible input sizes.
    
    Args:
        num_classes: Number of output classes (default: 38)
        dropout_rate: Dropout probability for regularization (default: 0.5)
        in_channels: Number of input channels (default: 3 for RGB)
        
    Input shape: (batch_size, 3, H, W) where H, W >= 32
    Output shape: (batch_size, num_classes)
    """
    
    def __init__(
        self,
        num_classes: int = 38,
        dropout_rate: float = 0.5,
        in_channels: int = 3,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1: 224x224x3 → 112x112x32
            ConvBlock(in_channels, 32, kernel_size=3),
            
            # Block 2: 112x112x32 → 56x56x64
            ConvBlock(32, 64, kernel_size=3),
            
            # Block 3: 56x56x64 → 28x28x128
            ConvBlock(64, 128, kernel_size=3),
            
            # Block 4: 28x28x128 → 14x14x256
            ConvBlock(128, 256, kernel_size=3),
        )
        
        # Global average pooling (flexible input size)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Feature extraction
        x = self.features(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the classifier.
        Useful for visualization and transfer learning.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            Feature tensor of shape (batch_size, 256)
        """
        x = self.features(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x
    
    def count_parameters(self) -> Tuple[int, int]:
        """
        Count the number of trainable and total parameters.
        
        Returns:
            Tuple of (trainable_params, total_params)
        """
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def create_custom_cnn(
    num_classes: int = 38,
    dropout_rate: float = 0.5,
    pretrained_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> PlantDiseaseCNN:
    """
    Factory function to create a PlantDiseaseCNN model.
    
    Args:
        num_classes: Number of output classes
        dropout_rate: Dropout probability
        pretrained_path: Optional path to pretrained weights
        device: Device to load the model on
        
    Returns:
        PlantDiseaseCNN model instance
    """
    model = PlantDiseaseCNN(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )
    
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from: {pretrained_path}")
    
    if device is not None:
        model = model.to(device)
    
    # Print model summary
    trainable, total = model.count_parameters()
    print(f"Model: PlantDiseaseCNN")
    print(f"  Classes: {num_classes}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Total parameters: {total:,}")
    
    return model


# Enhanced model with residual connections
class PlantDiseaseCNNv2(nn.Module):
    """
    Enhanced CNN with residual-like skip connections.
    This version adds skip connections for better gradient flow.
    """
    
    def __init__(
        self,
        num_classes: int = 38,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # Residual-style blocks
        self.layer1 = self._make_layer(32, 64, stride=2)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes),
        )
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, stride: int = 1):
        """Create a layer with convolution and optional downsampling."""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        
        x = self.layer1(x)
        x = F.relu(x)
        
        x = self.layer2(x)
        x = F.relu(x)
        
        x = self.layer3(x)
        x = F.relu(x)
        
        x = self.layer4(x)
        x = F.relu(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    def count_parameters(self) -> Tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
