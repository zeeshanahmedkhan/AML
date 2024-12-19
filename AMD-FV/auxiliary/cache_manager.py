import pickle
from pathlib import Path
from typing import Any, Optional, Union, List
import hashlib
import json
import logging
import time
import shutil
from datetime import datetime, timedelta

class CacheManager:
    """Manager class for handling data caching"""
    
    def __init__(self, 
                 cache_dir: Union[str, Path],
                 max_size_gb: float = 10.0,
                 expiration_days: int = 30):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.expiration_delta = timedelta(days=expiration_days)
        
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        self.metadata = self._load_metadata()
        
        # Initial cleanup
        self._cleanup()
    
    def _load_metadata(self) -> dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logging.warning("Corrupted metadata file. Creating new one.")
                return {'files': {}}
        return {'files': {}}
    
    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)
    
    def _compute_key(self, data: Any) -> str:
        """Compute cache key for data"""
        if isinstance(data, (str, bytes, bytearray)):
            content = data if isinstance(data, bytes) else str(data).encode()
        else:
            try:
                content = pickle.dumps(data)
            except (pickle.PickleError, TypeError):
                content = str(data).encode()
        
        return hashlib.sha256(content).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key"""
        return self.cache_dir / f"{key}.pkl"
    
    def put(self, data: Any, key: Optional[str] = None, 
            metadata: Optional[dict] = None) -> str:
        """
        Store data in cache
        
        Args:
            data: Data to cache
            key: Optional cache key (computed from data if not provided)
            metadata: Optional metadata to store with the data
        
        Returns:
            Cache key
        """
        if key is None:
            key = self._compute_key(data)
        
        cache_path = self._get_cache_path(key)
        
        # Store data
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Update metadata
        self.metadata['files'][key] = {
            'created': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'size': cache_path.stat().st_size,
            'metadata': metadata or {}
        }
        
        self._save_metadata()
        self._cleanup()
        
        return key
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve data from cache"""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        if key not in self.metadata['files']:
            # File exists but no metadata - possibly corrupted
            cache_path.unlink()
            return None
        
        # Check expiration
        created = datetime.fromisoformat(self.metadata['files'][key]['created'])
        if datetime.now() - created > self.expiration_delta:
            self._remove_cache_entry(key)
            return None
        
        # Load data
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            # Update last accessed time
            self.metadata['files'][key]['last_accessed'] = datetime.now().isoformat()
            self._save_metadata()
            
            return data
        
        except (pickle.UnpicklingError, EOFError):
            # Corrupted cache file
            self._remove_cache_entry(key)
            return None