import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from typing import Tuple, List, Dict, Optional
import torch.nn.functional as F

class FaceMetrics:
    """Comprehensive metrics for face recognition evaluation"""
    
    def __init__(self, 
                 thresholds: Optional[List[float]] = None,
                 distance_metric: str = 'cosine'):
        self.thresholds = thresholds or np.arange(0, 1.1, 0.1)
        self.distance_metric = distance_metric

    def compute_distance(self, 
                        feat1: torch.Tensor, 
                        feat2: torch.Tensor) -> torch.Tensor:
        """Compute distance between feature vectors"""
        if self.distance_metric == 'cosine':
            return 1 - F.cosine_similarity(feat1, feat2)
        elif self.distance_metric == 'euclidean':
            return torch.norm(feat1 - feat2, dim=1)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def compute_accuracy(self, 
                        distances: torch.Tensor,
                        labels: torch.Tensor,
                        threshold: float) -> float:
        """Compute verification accuracy at a specific threshold"""
        predictions = (distances <= threshold).float()
        correct = (predictions == labels).float()
        return correct.mean().item()

    def compute_far_frr(self, 
                       distances: torch.Tensor,
                       labels: torch.Tensor,
                       threshold: float) -> Tuple[float, float]:
        """Compute False Accept Rate and False Reject Rate"""
        # Genuine pairs (same identity)
        genuine_mask = labels == 1
        genuine_distances = distances[genuine_mask]
        
        # Impostor pairs (different identity)
        impostor_mask = labels == 0
        impostor_distances = distances[impostor_mask]
        
        # Compute FAR and FRR
        far = (impostor_distances <= threshold).float().mean().item()
        frr = (genuine_distances > threshold).float().mean().item()
        
        return far, frr

    def compute_eer(self, 
                   distances: torch.Tensor,
                   labels: torch.Tensor) -> Tuple[float, float]:
        """Compute Equal Error Rate"""
        fpr, tpr, thresholds = roc_curve(labels.cpu(), -distances.cpu())
        fnr = 1 - tpr
        
        # Find threshold where FAR = FRR
        eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
        eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
        
        return eer, eer_threshold

    def compute_auc(self, 
                   distances: torch.Tensor,
                   labels: torch.Tensor) -> float:
        """Compute Area Under Curve (AUC)"""
        fpr, tpr, _ = roc_curve(labels.cpu(), -distances.cpu())
        return auc(fpr, tpr)

    def evaluate_pairs(self, 
                      features1: torch.Tensor,
                      features2: torch.Tensor,
                      labels: torch.Tensor) -> Dict[str, float]:
        """Evaluate face verification performance on pairs"""
        # Compute distances
        distances = self.compute_distance(features1, features2)
        
        # Initialize results dictionary
        results = {}
        
        # Compute accuracy at different thresholds
        for threshold in self.thresholds:
            acc = self.compute_accuracy(distances, labels, threshold)
            results[f'acc@{threshold:.1f}'] = acc
        
        # Compute EER
        eer, eer_threshold = self.compute_eer(distances, labels)
        results['eer'] = eer
        results['eer_threshold'] = eer_threshold
        
        # Compute AUC
        auc_score = self.compute_auc(distances, labels)
        results['auc'] = auc_score
        
        # Compute FAR/FRR at EER threshold
        far, frr = self.compute_far_frr(distances, labels, eer_threshold)
        results['far'] = far
        results['frr'] = frr
        
        return results

class TripletMetrics:
    """Metrics for triplet-based face recognition"""
    
    def __init__(self, margin: float = 0.5):
        self.margin = margin

    def compute_triplet_accuracy(self, 
                               anchor: torch.Tensor,
                               positive: torch.Tensor,
                               negative: torch.Tensor) -> float:
        """Compute accuracy of triplet constraints"""
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        correct = (pos_dist < neg_dist).float()
        return correct.mean().item()

    def compute_triplet_loss(self,
                           anchor: torch.Tensor,
                           positive: torch.Tensor,
                           negative: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss"""
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        losses = F.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()

    def evaluate_triplets(self,
                         anchor: torch.Tensor,
                         positive: torch.Tensor,
                         negative: torch.Tensor) -> Dict[str, float]:
        """Evaluate model performance on triplets"""
        results = {
            'accuracy': self.compute_triplet_accuracy(anchor, positive, negative),
            'loss': self.compute_triplet_loss(anchor, positive, negative).item()
        }
        return results