"""
Logging configuration and utilities.
"""
import logging
from pathlib import Path
from src.utils.config import settings

# Create logs directory if it doesn't exist
log_dir = Path(settings.LOG_FILE).parent
log_dir.mkdir(parents=True, exist_ok=True)

# Configure logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Set up root logger
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler(),
    ],
)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: The name of the logger (typically __name__)

    Returns:
        A configured logger instance
    """
    return logging.getLogger(name)
