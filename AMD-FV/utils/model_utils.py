import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
import logging
from pathlib import Path
import json
import time
from collections import OrderedDict

class ModelCheckpointer:
    """Utility class for model checkpointing"""
    
    def __init__(self,
                 save_dir: Path,
                 model_name: str,
                 max_checkpoints: int = 5):
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.max_checkpoints = max_checkpoints
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load checkpoint history
        self.history_file = self.save_dir / 'checkpoint_history.json'
        self.checkpoint_history = self._load_history()

    def _load_history(self) -> Dict:
        """Load checkpoint history from file"""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return {'checkpoints': [], 'best_metric': float('inf')}

    def _save_history(self):
        """Save checkpoint history to file"""
        with open(self.history_file, 'w') as f:
            json.dump(self.checkpoint_history, f, indent=4)

    def save_checkpoint(self,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       metric: float,
                       is_best: bool) -> str:
        """Save model checkpoint"""
        checkpoint_name = f"{self.model_name}_epoch{epoch}.pth"
        checkpoint_path = self.save_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric': metric,
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Update history
        self.checkpoint_history['checkpoints'].append({
            'name': checkpoint_name,
            'epoch': epoch,
            'metric': metric,
            'timestamp': time.time()
        })
        
        # Save best model separately if needed
        if is_best:
            best_path = self.save_dir / f"{self.model_name}_best.pth"
            torch.save(checkpoint, best_path)
            self.checkpoint_history['best_metric'] = metric
        
        # Remove old checkpoints if needed
        self._cleanup_old_checkpoints()
        
        # Save updated history
        self._save_history()
        
        return str(checkpoint_path)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints if exceeding max_checkpoints"""
        checkpoints = self.checkpoint_history['checkpoints']
        if len(checkpoints) > self.max_checkpoints:
            # Sort by timestamp
            checkpoints.sort(key=lambda x: x['timestamp'])
            
            # Remove oldest checkpoints
            while len(checkpoints) > self.max_checkpoints:
                oldest = checkpoints.pop(0)
                checkpoint_path = self.save_dir / oldest['name']
                if checkpoint_path.exists():
                    checkpoint_path.unlink()

    def load_checkpoint(self,
                       model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       checkpoint_path: Optional[Path] = None) -> Tuple[nn.Module, 
                                                                      Optional[torch.optim.Optimizer],
                                                                      int,
                                                                      float]:
        """Load model checkpoint"""
        if checkpoint_path is None:
            # Load best checkpoint by default
            checkpoint_path = self.save_dir / f"{self.model_name}_best.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return model, optimizer, checkpoint['epoch'], checkpoint['metric']

class ModelProfiler:
    """Utility class for model profiling"""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> Dict[str, int]:
        """Count trainable and non-trainable parameters"""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        return {
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params,
            'total': total_params
        }

    @staticmethod
    def measure_inference_time(model: nn.Module,
                             input_size: Tuple[int, ...],
                             device: torch.device,
                             num_iterations: int = 100) -> Dict[str, float]:
        """Measure model inference time"""
        model.eval()
        dummy_input = torch.randn(input_size).to(device)
        
        # Warm-up
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Measure time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        return {
            'avg_inference_time': avg_time,
            'fps': 1.0 / avg_time
        }

    @staticmethod
    def get_model_summary(model: nn.Module) -> List[Dict[str, str]]:
        """Generate model layer summary"""
        summary = []
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                params = sum(p.numel() for p in module.parameters())
                summary.append({
                    'name': name,
                    'type': module.__class__.__name__,
                    'parameters': str(params)
                })
                
        return summary