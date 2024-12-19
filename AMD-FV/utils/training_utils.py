import torch
import torch.nn as nn
from typing import Dict, Optional, Callable, List, Tuple
import logging
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name: str):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class LRSchedulerWithWarmup:
    """Learning rate scheduler with warmup"""
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_epochs: int,
                 max_epochs: int,
                 warmup_factor: float = 0.1,
                 eta_min: float = 1e-6):
        
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_factor = warmup_factor
        self.eta_min = eta_min
        
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        """Update learning rate"""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Warmup phase
            factor = self.warmup_factor + (1 - self.warmup_factor) * \
                    (self.current_epoch / self.warmup