"""
PlantHealth AI - Training Loop
===============================
Complete training pipeline for plant disease classification models.

Features:
- Training and validation loops
- Mixed precision training (AMP)
- Progress bars with tqdm
- Metric tracking
- Learning rate scheduling
"""

from typing import Dict, Optional, Tuple, List, Callable
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .callbacks import EarlyStopping, ModelCheckpoint, CallbackHandler


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number (for progress bar)
        use_amp: Whether to use automatic mixed precision
        scaler: GradScaler for AMP (created if None and use_amp=True)
        
    Returns:
        Dictionary with 'loss' and 'accuracy'
    """
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create scaler if needed
    if use_amp and scaler is None:
        scaler = GradScaler()
    
    # Progress bar
    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch} [Train]",
        leave=False,
        ncols=100,
    )
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with optional AMP
        if use_amp:
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Validate the model on the validation set.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to run on
        epoch: Current epoch number (for progress bar)
        
    Returns:
        Dictionary with 'loss' and 'accuracy'
    """
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch} [Valid]",
        leave=False,
        ncols=100,
    )
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
    }


class Trainer:
    """
    Complete training pipeline for plant disease classification.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        use_amp: Whether to use automatic mixed precision
        
    Example:
        trainer = Trainer(model, train_loader, val_loader, criterion, optimizer)
        history = trainer.train(epochs=30)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints",
        use_amp: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_amp = use_amp and torch.cuda.is_available()
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        
        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # AMP scaler
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
        }
        
        print(f"Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  AMP enabled: {self.use_amp}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
    
    def train(
        self,
        epochs: int,
        early_stopping_patience: int = 5,
        checkpoint_metric: str = "val_accuracy",
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model for specified number of epochs.
        
        Args:
            epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            checkpoint_metric: Metric to monitor for checkpointing
            verbose: Whether to print epoch summaries
            
        Returns:
            Training history dictionary
        """
        # Setup callbacks
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode='max' if 'accuracy' in checkpoint_metric else 'min',
            verbose=verbose,
        )
        
        checkpoint = ModelCheckpoint(
            save_path=self.checkpoint_dir / "best_model.pth",
            monitor=checkpoint_metric,
            mode='max' if 'accuracy' in checkpoint_metric else 'min',
            verbose=verbose,
        )
        
        callback_handler = CallbackHandler([early_stopping, checkpoint])
        
        print(f"\nStarting training for {epochs} epochs...")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Training
            train_metrics = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                epoch,
                use_amp=self.use_amp,
                scaler=self.scaler,
            )
            
            # Validation
            val_metrics = validate(
                self.model,
                self.val_loader,
                self.criterion,
                self.device,
                epoch,
            )
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            epoch_time = time.time() - epoch_start
            
            # Print epoch summary
            if verbose:
                print(f"\nEpoch {epoch}/{epochs} ({epoch_time:.1f}s)")
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
                print(f"  Valid - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            
            # Learning rate scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Callbacks
            metrics = {
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
            }
            
            if callback_handler.on_epoch_end(
                self.model,
                metrics,
                epoch=epoch,
                optimizer=self.optimizer,
            ):
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        total_time = time.time() - start_time
        print("=" * 60)
        print(f"Training completed in {total_time / 60:.1f} minutes")
        print(f"Best validation accuracy: {max(self.history['val_acc']):.2f}%")
        
        # Load best model
        checkpoint.load_best_model(self.model)
        
        return self.history
    
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: Data loader to evaluate on (default: validation loader)
            
        Returns:
            Dictionary with evaluation metrics
        """
        if dataloader is None:
            dataloader = self.val_loader
        
        return validate(self.model, dataloader, self.criterion, self.device, epoch=0)
    
    def save_model(self, path: str, include_optimizer: bool = False):
        """Save the model to a file."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
        }
        
        if include_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Model saved to: {path}")
    
    def load_model(self, path: str, load_optimizer: bool = False):
        """Load the model from a file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"Model loaded from: {path}")


def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "adamw",
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """
    Create an optimizer for the model.
    
    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer ('adam', 'adamw', 'sgd')
        learning_rate: Learning rate
        weight_decay: Weight decay (L2 regularization)
        momentum: Momentum (for SGD)
        
    Returns:
        Optimizer instance
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "reduce_on_plateau",
    **kwargs,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_name: Name of scheduler
        **kwargs: Additional scheduler arguments
        
    Returns:
        Scheduler instance
    """
    scheduler_name = scheduler_name.lower()
    
    if scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 3),
            min_lr=kwargs.get('min_lr', 1e-6),
        )
    elif scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 10),
            eta_min=kwargs.get('eta_min', 1e-6),
        )
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.1),
        )
    elif scheduler_name == "one_cycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', 0.01),
            epochs=kwargs.get('epochs', 30),
            steps_per_epoch=kwargs.get('steps_per_epoch', 100),
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
