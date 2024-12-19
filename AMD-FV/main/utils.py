import torch
import numpy as np
from PIL import Image
import logging
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def setup_logging(log_path: str = 'training.log'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def load_checkpoint(model: torch.nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer],
                   checkpoint_path: Path) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], int, float]:
    """Load model checkpoint"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    
    return model, optimizer, epoch, best_acc

def visualize_training(train_losses: list, 
                      val_losses: list, 
                      train_accs: list, 
                      val_accs: list, 
                      save_path: Path):
    """Visualize training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Loss vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Accuracy vs Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_progress.png')
    plt.close()

def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         save_path: Path,
                         num_classes: int = 10):
    """Plot confusion matrix for the first num_classes"""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] > num_classes:
        cm = cm[:num_classes, :num_classes]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path / 'confusion_matrix.png')
    plt.close()

def compute_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Compute accuracy from model output and targets"""
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    total = target.size(0)
    return correct / total

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count