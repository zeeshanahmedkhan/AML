from typing import Tuple, List
import torch
from pathlib import Path

@dataclass
class TrainingConfig:
    # Model parameters
    num_classes: int = 85000
    feature_dim: int = 2048
    
    # Training parameters
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    num_workers: int = 4
    
    # Image parameters
    image_size: Tuple[int, int] = (112, 112)
    mean: List[float] = (0.485, 0.456, 0.406)
    std: List[float] = (0.229, 0.224, 0.225)
    
    # Paths
    train_data_path: Path = Path('data/train')
    val_data_path: Path = Path('data/val')
    model_save_path: Path = Path('checkpoints')
    best_model_name: str = 'best_model.pth'
    
    # Device
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __post_init__(self):
        self.model_save_path.mkdir(parents=True, exist_ok=True)

config = TrainingConfig()