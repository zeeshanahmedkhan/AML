import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

from config import config
from utils import AverageMeter, compute_accuracy, plot_confusion_matrix

def validate_model(model: torch.nn.Module,
                  val_loader: torch.utils.data.DataLoader,
                  criterion: torch.nn.Module,
                  device: torch.device) -> Tuple[float, float]:
    """
    Validate the model and return validation loss and accuracy
    """
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            logits, features = model(inputs)
            loss = criterion(features, labels)

            # Compute accuracy
            acc = compute_accuracy(logits, labels)

            # Update meters
            val_loss.update(loss.item(), inputs.size(0))
            val_acc.update(acc, inputs.size(0))

            # Store predictions and labels for confusion matrix
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({
                'val_loss': f'{val_loss.avg:.4f}',
                'val_acc': f'{val_acc.avg:.4f}'
            })

    # Plot confusion matrix
    plot_confusion_matrix(
        np.array(all_labels),
        np.array(all_preds),
        save_path=config.model_save_path
    )

    return val_loss.avg, val_acc.avg

def evaluate_embeddings(model: torch.nn.Module,
                       val_loader: torch.utils.data.DataLoader,
                       device: torch.device) -> Dict[str, np.ndarray]:
    """
    Extract embeddings for the validation set
    """
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Extracting embeddings')
        for inputs, batch_labels in pbar:
            inputs = inputs.to(device)
            
            # Get embeddings
            _, features = model(inputs)
            embeddings.append(features.cpu().numpy())
            labels.extend(batch_labels.numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.array(labels)

    return {
        'embeddings': embeddings,
        'labels': labels
    }

if __name__ == "__main__":
    # Add code here to load model and run validation independently
    pass