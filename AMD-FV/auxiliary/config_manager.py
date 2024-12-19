import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field, asdict
import os
from copy import deepcopy

@dataclass
class SystemConfig:
    """System-wide configuration"""
    seed: int = 42
    num_workers: int = 4
    device: str = 'cuda'
    fp16: bool = True
    debug_mode: bool = False

@dataclass
class PathConfig:
    """Path configuration"""
    data_dir: Path = Path('data')
    output_dir: Path = Path('output')
    model_dir: Path = Path('models')
    log_dir: Path = Path('logs')
    cache_dir: Path = Path('cache')

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = 'dpn_plus'
    num_classes: int = 85000
    feature_dim: int = 2048
    pretrained: bool = True
    dropout: float = 0.1

@dataclass
class TrainingConfig:
    """Training configuration"""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    warmup_epochs: int = 5
    early_stopping_patience: int = 10
    gradient_clip_val: float = 1.0

@dataclass
class Config:
    """Main configuration class"""
    system: SystemConfig = field(default_factory=SystemConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def save(self, filepath: Union[str, Path]):
        """Save configuration to file"""
        filepath = Path(filepath)
        
        if filepath.suffix == '.yml' or filepath.suffix == '.yaml':
            with open(filepath, 'w') as f:
                yaml.dump(asdict(self), f, default_flow_style=False)
        elif filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(asdict(self), f, indent=4)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'Config':
        """Load configuration from file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        if filepath.suffix == '.yml' or filepath.suffix == '.yaml':
            with open(filepath) as f:
                config_dict = yaml.safe_load(f)
        elif filepath.suffix == '.json':
            with open(filepath) as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return cls(
            system=SystemConfig(**config_dict['system']),
            paths=PathConfig(**config_dict['paths']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training'])
        )
    
    def update_from_args(self, args: Dict[str, Any]):
        """Update configuration from command line arguments"""
        for section in ['system', 'paths', 'model', 'training']:
            section_config = getattr(self, section)
            for key, value in args.items():
                if hasattr(section_config, key) and value is not None:
                    setattr(section_config, key, value)

class ConfigManager:
    """Manager class for handling configurations"""
    
    def __init__(self, 
                 base_config: Optional[Path] = None,
                 exp_config: Optional[Path] = None):
        
        # Load base configuration
        if base_config is None:
            self.config = Config()
        else:
            self.config = Config.load(base_config)
        
        # Update with experiment-specific configuration
        if exp_config is not None:
            self.update_from_file(exp_config)
        
        # Create necessary directories
        self._create_directories()
    
    def update_from_file(self, config_path: Path):
        """Update configuration from file"""
        new_config = Config.load(config_path)
        
        for section in ['system', 'paths', 'model', 'training']:
            current_section = getattr(self.config, section)
            new_section = getattr(new_config, section)
            
            for key, value in asdict(new_section).items():
                if value is not None:
                    setattr(current_section, key, value)
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section in ['system', 'paths', 'model', 'training']:
            if section in config_dict:
                current_section = getattr(self.config, section)
                for key, value in config_dict[section].items():
                    if hasattr(current_section, key):
                        setattr(current_section, key, value)
    
    def _create_directories(self):
        """Create necessary directories"""
        for path in [self.config.paths.data_dir,
                    self.config.paths.output_dir,
                    self.config.paths.model_dir,
                    self.config.paths.log_dir,
                    self.config.paths.cache_dir]:
            path.mkdir(parents=True, exist_ok=True)
    
    def get_experiment_dir(self, exp_name: str) -> Path:
        """Get experiment directory"""
        exp_dir = self.config.paths.output_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
    
    def save_experiment_config(self, exp_dir: Path):
        """Save experiment configuration"""
        # Save as both YAML and JSON
        self.config.save(exp_dir / 'config.yml')
        self.config.save(exp_dir / 'config.json')