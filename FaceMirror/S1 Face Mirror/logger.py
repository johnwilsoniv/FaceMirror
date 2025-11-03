"""
S1 Face Mirror - Logging Infrastructure

Provides centralized logging for the application with both console and file output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds color to console output based on log level.
    Only applies colors to console output, not file output.
    """

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[0m',       # Default
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        """Format the log record with color."""
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )
        return super().format(record)


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_color: bool = True
) -> logging.Logger:
    """
    Setup and configure a logger instance.

    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file for persistent logging
        use_color: Whether to use colored output for console (default: True)

    Returns:
        Configured logger instance

    Example:
        >>> from logger import setup_logger
        >>> logger = setup_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.warning("Low confidence detection")
        >>> logger.error("Failed to load model")
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers = []  # Clear any existing handlers

    # Console handler - user-facing messages
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)

    if use_color:
        # Colored format for console
        console_format = ColoredFormatter('%(message)s')
    else:
        # Plain format for console
        console_format = logging.Formatter('%(message)s')

    console.setFormatter(console_format)
    logger.addHandler(console)

    # File handler - detailed logs including debug messages
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Detailed format for file logging
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with default configuration from config.py.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured logger instance

    Example:
        >>> from logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting processing...")
    """
    # Import config here to avoid circular imports
    try:
        import config
        level = config.LOG_LEVEL
        log_file = config.LOG_FILE
    except ImportError:
        # Fallback if config not available
        level = "INFO"
        log_file = None

    return setup_logger(name, level=level, log_file=log_file)


# Convenience function for quick logging without creating logger instance
def log_info(message: str):
    """Quick info log without needing to create logger instance."""
    logger = get_logger("face_mirror")
    logger.info(message)


def log_warning(message: str):
    """Quick warning log without needing to create logger instance."""
    logger = get_logger("face_mirror")
    logger.warning(message)


def log_error(message: str):
    """Quick error log without needing to create logger instance."""
    logger = get_logger("face_mirror")
    logger.error(message)


def log_debug(message: str):
    """Quick debug log without needing to create logger instance."""
    logger = get_logger("face_mirror")
    logger.debug(message)
