import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from typing import List, Optional, Tuple, Dict
import logging
from pathlib import Path

class FaceVisualizer:
    """Visualization tools for face recognition analysis"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        
    def plot_training_curves(self,
                           metrics: Dict[str, List[float]],
                           title: str,
                           filename: str):
        """Plot training metrics over time"""
        plt.figure(figsize=(12, 6))
        
        for metric_name, values in metrics.items():
            plt.plot(values, label=metric_name)
            
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        save_path = self.output_dir / filename
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Saved training curves to {save_path}")

    def plot_embeddings_tsne(self,
                           embeddings: np.ndarray,
                           labels: np.ndarray,
                           title: str = 'T-SNE Visualization of Face Embeddings',
                           filename: str = 'embeddings_tsne.png',
                           perplexity: int = 30,
                           n_components: int = 2):
        """Plot T-SNE visualization of embeddings"""
        # Reduce dimensionality
        tsne = TSNE(n_components=n_components, 
                    perplexity=perplexity, 
                    random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], 
                            embeddings_2d[:, 1], 
                            c=labels, 
                            cmap='tab20')
        
        plt.title(title)
        plt.colorbar(scatter)
        
        save_path = self.output_dir / filename
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Saved T-SNE plot to {save_path}")

    def plot_similarity_matrix(self,
                             features: torch.Tensor,
                             labels: torch.Tensor,
                             title: str = 'Feature Similarity Matrix',
                             filename: str = 'similarity_matrix.png'):
        """Plot similarity matrix between features"""
        # Compute similarity matrix
        similarity = torch.mm(features, features.t())
        similarity = similarity.cpu().numpy()
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity, 
                   cmap='coolwarm', 
                   center=0,
                   xticklabels=labels.cpu().numpy(),
                   yticklabels=labels.cpu().numpy())