"""
Training pipeline for 3D U-Net soma segmentation.
"""

import os
import time
import logging
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Import custom modules
from models.unet_3d import UNet3D, SomaUNet3D
from data.dataset import SomaDataset, SomaDataModule
from utils.metrics import dice_coefficient, hausdorff_distance_3d, compute_volume_metrics
from utils.visualization import plot_training_curves, visualize_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


class SomaTrainer:
    """
    Trainer class for 3D U-Net soma segmentation.
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: optim.Optimizer,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 criterion: nn.Module = nn.BCEWithLogitsLoss(),
                 checkpoint_dir: str = 'checkpoints',
                 log_dir: str = 'logs'):
        """
        Args:
            model: The neural network model
            device: Device to run training on
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            criterion: Loss function
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
        """
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir)
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.val_dice_scores = []
        self.val_precisions = []
        self.val_recalls = []
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=15)
        
        # Best model tracking
        self.best_val_dice = 0.0
        self.best_model_path = None
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, (volumes, masks) in enumerate(self.train_loader):
            volumes = volumes.to(self.device)
            masks = masks.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(volumes)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log batch progress
            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, '
                          f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        total_precision = 0.0
        total_recall = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for volumes, masks in self.val_loader:
                volumes = volumes.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(volumes)
                loss = self.criterion(outputs, masks)
                
                # Calculate metrics
                dice_scores = []
                precisions = []
                recalls = []
                
                for i in range(outputs.size(0)):
                    pred = (torch.sigmoid(outputs[i]) > 0.5).float()
                    target = masks[i]
                    
                    # Dice coefficient
                    dice = dice_coefficient(pred, target)
                    dice_scores.append(dice)
                    
                    # Precision and recall
                    pred_flat = pred.view(-1).cpu().numpy()
                    target_flat = target.view(-1).cpu().numpy()
                    
                    tp = np.sum(pred_flat * target_flat)
                    fp = np.sum(pred_flat * (1 - target_flat))
                    fn = np.sum((1 - pred_flat) * target_flat)
                    
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    
                    precisions.append(precision)
                    recalls.append(recall)
                
                total_loss += loss.item()
                total_dice += np.mean(dice_scores)
                total_precision += np.mean(precisions)
                total_recall += np.mean(recalls)
        
        metrics = {
            'loss': total_loss / num_batches,
            'dice': total_dice / num_batches,
            'precision': total_precision / num_batches,
            'recall': total_recall / num_batches
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_dice_scores': self.val_dice_scores,
            'val_precisions': self.val_precisions,
            'val_recalls': self.val_recalls
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            logger.info(f"Saved best model at epoch {epoch}")
    
    def train(self, num_epochs: int) -> Dict[str, list]:
        """Full training loop"""
        logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validation
            val_metrics = self.validate(epoch)
            self.val_losses.append(val_metrics['loss'])
            self.val_dice_scores.append(val_metrics['dice'])
            self.val_precisions.append(val_metrics['precision'])
            self.val_recalls.append(val_metrics['recall'])
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_metrics['loss'])
            
            # Logging
            logger.info(f"Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val Dice: {val_metrics['dice']:.4f}, "
                       f"Val Precision: {val_metrics['precision']:.4f}, "
                       f"Val Recall: {val_metrics['recall']:.4f}")
            
            # Tensorboard logging
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
            self.writer.add_scalar('Metrics/Dice', val_metrics['dice'], epoch)
            self.writer.add_scalar('Metrics/Precision', val_metrics['precision'], epoch)
            self.writer.add_scalar('Metrics/Recall', val_metrics['recall'], epoch)
            
            # Save best model
            is_best = val_metrics['dice'] > self.best_val_dice
            if is_best:
                self.best_val_dice = val_metrics['dice']
            
            # Save checkpoint
            if epoch % 10 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Close tensorboard writer
        self.writer.close()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_dice_scores': self.val_dice_scores,
            'val_precisions': self.val_precisions,
            'val_recalls': self.val_recalls
        }
    
    def plot_training_curves(self, save_path: str = None):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Dice score
        axes[0, 1].plot(self.val_dice_scores)
        axes[0, 1].set_title('Validation Dice Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].grid(True)
        
        # Precision and Recall
        axes[1, 0].plot(self.val_precisions, label='Precision')
        axes[1, 0].plot(self.val_recalls, label='Recall')
        axes[1, 0].set_title('Validation Precision and Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Combined metrics
        axes[1, 1].plot(self.val_dice_scores, label='Dice')
        axes[1, 1].plot(self.val_precisions, label='Precision')
        axes[1, 1].plot(self.val_recalls, label='Recall')
        axes[1, 1].set_title('All Validation Metrics')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def train_unet(nrrd_dir: str,
               mask_dir: str,
               model_type: str = 'soma_unet',
               epochs: int = 100,
               batch_size: int = 4,
               learning_rate: float = 1e-4,
               device: Optional[torch.device] = None,
               checkpoint_dir: str = 'results/checkpoints',
               log_dir: str = 'results/logs') -> Dict[str, list]:
    """
    High-level function to train a 3D U-Net for soma segmentation.
    
    Args:
        nrrd_dir: Directory containing NRRD files
        mask_dir: Directory containing mask files
        model_type: Type of model ('unet' or 'soma_unet')
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to run training on
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for logs
    
    Returns:
        Training history dictionary
    """
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    
    # Create data module
    data_module = SomaDataModule(
        nrrd_dir=nrrd_dir,
        mask_dir=mask_dir,
        batch_size=batch_size,
        target_size=(128, 128, 128)
    )
    data_module.setup()
    
    # Create model
    if model_type == 'unet':
        model = UNet3D(n_channels=1, n_classes=1)
    elif model_type == 'soma_unet':
        model = SomaUNet3D(n_channels=1, n_classes=1, dropout_rate=0.1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                    factor=0.5, patience=10, 
                                                    verbose=True)
    
    # Create trainer
    trainer = SomaTrainer(
        model=model,
        device=device,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    
    # Train model
    history = trainer.train(epochs)
    
    # Plot training curves
    trainer.plot_training_curves(save_path=os.path.join(log_dir, 'training_curves.png'))
    
    return history


if __name__ == '__main__':
    # Example usage
    import tempfile
    from data.dataset import create_sample_data
    
    # Create sample data
    with tempfile.TemporaryDirectory() as temp_dir:
        nrrd_dir = os.path.join(temp_dir, 'nrrd')
        mask_dir = os.path.join(temp_dir, 'masks')
        
        create_sample_data(nrrd_dir, mask_dir, num_samples=20)
        
        # Train model
        history = train_unet(
            nrrd_dir=nrrd_dir,
            mask_dir=mask_dir,
            model_type='soma_unet',
            epochs=5,  # Small number for testing
            batch_size=2,
            learning_rate=1e-4
        )
        
        print("Training completed!")
        print(f"Final validation dice: {history['val_dice_scores'][-1]:.4f}")