import logging
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Optional, Dict
import logging.handlers

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    COLORS = {
        'DEBUG': '\033[0;36m',  # Cyan
        'INFO': '\033[0;32m',   # Green
        'WARNING': '\033[0;33m', # Yellow
        'ERROR': '\033[0;31m',   # Red
        'CRITICAL': '\033[0;35m' # Purple
    }
    RESET = '\033[0m'

    def __init__(self, include_timestamp: bool = True):
        fmt = '%(levelname)s - %(message)s'
        if include_timestamp:
            fmt = '%(asctime)s - ' + fmt
        super().__init__(fmt)

    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)

def setup_logging(
    log_dir: Path,
    log_name: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True,
    file_output: bool = True,
    json_output: bool = False
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory to store log files
        log_name: Name for the logger and log files
        level: Logging level
        console_output: Whether to output logs to console
        file_output: Whether to output logs to file
        json_output: Whether to also save logs in JSON format
    """
    
    # Create log directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate default log name if not provided
    if log_name is None:
        log_name = datetime.now().strftime('log_%Y%m%d_%H%M%S')
    
    # Create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CustomFormatter())
        logger.addHandler(console_handler)
    
    if file_output:
        # Regular file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{log_name}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
        if json_output:
            # JSON file handler
            json_handler = JsonFileHandler(log_dir / f"{log_name}.json")
            json_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(json_handler)
    
    return logger

class JsonFileHandler(logging.FileHandler):
    """Custom handler for JSON format logging"""
    
    def __init__(self, filename):
        super().__init__(filename, mode='a')
        self.records = []
    
    def emit(self, record):
        """Save record as JSON"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        with open(self.baseFilename, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

def log_exception(logger: logging.Logger, exc: Exception, context: str = ''):
    """
    Log exception with traceback and context
    
    Args:
        logger: Logger instance
        exc: Exception to log
        context: Additional context about where/why the exception occurred
    """
    import traceback
    
    if context:
        logger.error(f"Exception occurred in {context}")
    
    logger.error(f"Exception type: {type(exc).__name__}")
    logger.error(f"Exception message: {str(exc)}")
    logger.error("Traceback:")
    for line in traceback.format_exc().split('\n'):
        logger.error(line)