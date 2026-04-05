"""
PlantHealth AI - Training Callbacks
====================================
Callback classes for monitoring and controlling the training process.

Includes:
- EarlyStopping: Stop training when validation metric stops improving
- ModelCheckpoint: Save the best model during training
- CallbackHandler: Manage multiple callbacks
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import torch
import torch.nn as nn


class EarlyStopping:
    """
    Early stopping to terminate training when validation metric stops improving.
    
    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for loss, 'max' for accuracy
        verbose: Whether to print messages
        
    Example:
        early_stop = EarlyStopping(patience=5, mode='min')
        for epoch in range(epochs):
            train(...)
            val_loss = validate(...)
            if early_stop(val_loss):
                print("Early stopping triggered!")
                break
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "min",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        
        # Set comparison function based on mode
        if mode == "min":
            self.is_better = lambda new, best: new < best - min_delta
            self.best_value = float('inf')
        elif mode == "max":
            self.is_better = lambda new, best: new > best + min_delta
            self.best_value = float('-inf')
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
    
    def __call__(self, current_value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_value: Current validation metric value
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.is_better(current_value, self.best_value):
            self.best_value = current_value
            self.counter = 0
            if self.verbose:
                print(f"  EarlyStopping: New best value: {current_value:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  EarlyStopping: Stopping training (best: {self.best_value:.4f})")
        
        return self.early_stop
    
    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.early_stop = False
        if self.mode == "min":
            self.best_value = float('inf')
        else:
            self.best_value = float('-inf')


class ModelCheckpoint:
    """
    Save the best model during training.
    
    Args:
        save_path: Path to save the model
        monitor: Metric to monitor ('val_loss' or 'val_accuracy')
        mode: 'min' for loss, 'max' for accuracy
        save_best_only: Only save when metric improves
        verbose: Whether to print messages
        
    Example:
        checkpoint = ModelCheckpoint('best_model.pth', monitor='val_accuracy', mode='max')
        for epoch in range(epochs):
            train(...)
            val_acc = validate(...)
            checkpoint(model, val_acc)
    """
    
    def __init__(
        self,
        save_path: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        verbose: bool = True,
    ):
        self.save_path = Path(save_path)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        
        self.best_value = None
        
        # Ensure directory exists
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set comparison function
        if mode == "min":
            self.is_better = lambda new, best: new < best
            self.best_value = float('inf')
        elif mode == "max":
            self.is_better = lambda new, best: new > best
            self.best_value = float('-inf')
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
    
    def __call__(
        self,
        model: nn.Module,
        current_value: float,
        epoch: Optional[int] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Save the model if metric improved.
        
        Args:
            model: Model to save
            current_value: Current validation metric value
            epoch: Current epoch number
            optimizer: Optimizer state to save
            extra_info: Additional information to save
            
        Returns:
            True if model was saved, False otherwise
        """
        saved = False
        
        if self.save_best_only:
            if self.is_better(current_value, self.best_value):
                self.best_value = current_value
                self._save_model(model, epoch, optimizer, extra_info)
                saved = True
                if self.verbose:
                    print(f"  Checkpoint: Saved best model ({self.monitor}: {current_value:.4f})")
        else:
            # Save every time
            self._save_model(model, epoch, optimizer, extra_info)
            saved = True
            if current_value is not None and self.is_better(current_value, self.best_value):
                self.best_value = current_value
        
        return saved
    
    def _save_model(
        self,
        model: nn.Module,
        epoch: Optional[int],
        optimizer: Optional[torch.optim.Optimizer],
        extra_info: Optional[Dict[str, Any]],
    ):
        """Save the model checkpoint."""
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'best_value': self.best_value,
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if extra_info is not None:
            checkpoint.update(extra_info)
        
        torch.save(checkpoint, self.save_path)
    
    def load_best_model(self, model: nn.Module) -> nn.Module:
        """
        Load the best saved model.
        
        Args:
            model: Model to load weights into
            
        Returns:
            Model with loaded weights
        """
        if not self.save_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {self.save_path}")
        
        checkpoint = torch.load(self.save_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.verbose:
            best_val = checkpoint.get('best_value', 'N/A')
            epoch = checkpoint.get('epoch', 'N/A')
            print(f"Loaded best model from epoch {epoch} ({self.monitor}: {best_val})")
        
        return model


class LearningRateScheduler:
    """
    Wrapper for learning rate schedulers with logging.
    
    Args:
        scheduler: PyTorch learning rate scheduler
        verbose: Whether to print LR changes
    """
    
    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        verbose: bool = True,
    ):
        self.scheduler = scheduler
        self.verbose = verbose
        self.last_lr = None
    
    def step(self, metrics: Optional[float] = None):
        """Step the scheduler."""
        # Get current LR before stepping
        current_lr = self.scheduler.get_last_lr()[0]
        
        # Step the scheduler
        if metrics is not None and hasattr(self.scheduler, 'step'):
            # For ReduceLROnPlateau
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metrics)
            else:
                self.scheduler.step()
        else:
            self.scheduler.step()
        
        # Get new LR after stepping
        new_lr = self.scheduler.get_last_lr()[0]
        
        # Log if LR changed
        if self.verbose and self.last_lr is not None and new_lr != current_lr:
            print(f"  LR Scheduler: Learning rate changed from {current_lr:.2e} to {new_lr:.2e}")
        
        self.last_lr = new_lr
    
    def get_last_lr(self) -> List[float]:
        """Get the last learning rate."""
        return self.scheduler.get_last_lr()


class CallbackHandler:
    """
    Handler to manage multiple callbacks during training.
    
    Args:
        callbacks: List of callback instances
        
    Example:
        handler = CallbackHandler([
            EarlyStopping(patience=5),
            ModelCheckpoint('model.pth'),
        ])
        
        for epoch in range(epochs):
            val_metrics = train_epoch(...)
            if handler.on_epoch_end(model, val_metrics):
                break  # Early stopping triggered
    """
    
    def __init__(self, callbacks: Optional[List] = None):
        self.callbacks = callbacks or []
    
    def add_callback(self, callback):
        """Add a callback to the handler."""
        self.callbacks.append(callback)
    
    def on_epoch_end(
        self,
        model: nn.Module,
        metrics: Dict[str, float],
        epoch: Optional[int] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> bool:
        """
        Call all callbacks at the end of an epoch.
        
        Args:
            model: Current model
            metrics: Dictionary of validation metrics
            epoch: Current epoch number
            optimizer: Current optimizer
            
        Returns:
            True if training should stop, False otherwise
        """
        should_stop = False
        
        for callback in self.callbacks:
            if isinstance(callback, EarlyStopping):
                # Determine which metric to use
                if 'val_loss' in metrics:
                    value = metrics['val_loss']
                elif 'val_accuracy' in metrics:
                    value = metrics['val_accuracy']
                else:
                    value = list(metrics.values())[0]
                
                if callback(value):
                    should_stop = True
            
            elif isinstance(callback, ModelCheckpoint):
                # Get the monitored metric
                metric_name = callback.monitor
                if metric_name in metrics:
                    value = metrics[metric_name]
                elif metric_name == 'val_loss' and 'loss' in metrics:
                    value = metrics['loss']
                elif metric_name == 'val_accuracy' and 'accuracy' in metrics:
                    value = metrics['accuracy']
                else:
                    value = list(metrics.values())[0]
                
                callback(model, value, epoch=epoch, optimizer=optimizer)
            
            elif isinstance(callback, LearningRateScheduler):
                # Get loss for ReduceLROnPlateau
                loss = metrics.get('val_loss', metrics.get('loss'))
                callback.step(loss)
        
        return should_stop
    
    def reset(self):
        """Reset all callbacks."""
        for callback in self.callbacks:
            if hasattr(callback, 'reset'):
                callback.reset()
