"""
Logging configuration for BaKC-plus

This module provides logging setup and utilities for structured logging
throughout the BaKC-plus package.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


# Default log format and date format
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Package-level logger instance
_logger_configured = False


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    enable_file_logging: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    force_reconfigure: bool = False,
) -> None:
    """
    Setup logging configuration for BaKC-plus

    This function configures the Python logging system with console and optional
    file output. It should be called once at the start of the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (default: output/logs/bakc_plus.log)
        enable_file_logging: Whether to enable file logging
        max_bytes: Maximum log file size in bytes before rotation (default: 10MB)
        backup_count: Number of backup log files to keep (default: 5)
        force_reconfigure: Force reconfiguration even if already configured

    Example:
        >>> from bakc_plus import setup_logging
        >>> setup_logging(log_level="DEBUG", enable_file_logging=True)
        >>> # Now all loggers will use this configuration

    Notes:
        - Console handler shows INFO and above
        - File handler (if enabled) shows DEBUG and above
        - Log files are rotated when they reach max_bytes
        - Old backup files are deleted when count exceeds backup_count
    """
    global _logger_configured

    # Skip if already configured (unless forced)
    if _logger_configured and not force_reconfigure:
        return

    # Get root logger for bakc_plus
    root_logger = logging.getLogger("bakc_plus")

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Set level on root logger
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (DEBUG and above, if enabled)
    if enable_file_logging:
        # Determine log file path
        if log_file is None:
            log_file = Path("output/logs/bakc_plus.log")

        # Ensure log directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8',
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Prevent propagation to root logger
    root_logger.propagate = False

    # Mark as configured
    _logger_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module

    This function returns a logger instance with the appropriate name.
    The logger will use the configuration set by setup_logging().

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured logger instance

    Example:
        >>> from bakc_plus.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")
        [2025-11-18 12:00:00,000] [INFO] [bakc_plus.module] This is an info message

    Notes:
        - Always use __name__ as the logger name for proper hierarchy
        - Loggers are cached by Python's logging system
        - Child loggers inherit parent configuration
    """
    # Ensure the logger is under bakc_plus hierarchy
    if not name.startswith("bakc_plus"):
        name = f"bakc_plus.{name}"

    return logging.getLogger(name)


def reset_logging() -> None:
    """
    Reset logging configuration

    This function clears all handlers and resets the configuration state.
    Useful for testing or when reconfiguration is needed.

    Example:
        >>> reset_logging()
        >>> setup_logging(log_level="DEBUG")  # Reconfigure with new settings
    """
    global _logger_configured

    # Get root logger
    root_logger = logging.getLogger("bakc_plus")

    # Clear handlers
    root_logger.handlers.clear()

    # Reset configuration flag
    _logger_configured = False


def set_log_level(level: str) -> None:
    """
    Change the log level at runtime

    Args:
        level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Example:
        >>> set_log_level("DEBUG")  # Enable debug logging
        >>> # ... do some debugging
        >>> set_log_level("INFO")  # Return to normal logging
    """
    root_logger = logging.getLogger("bakc_plus")
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)


# Convenience function for getting a logger without imports
def logger_for(name: str) -> logging.Logger:
    """
    Convenience alias for get_logger()

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return get_logger(name)
