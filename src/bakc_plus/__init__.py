"""
BaKC-plus: Bagging and Kernel-based Conformal Prediction for Anomaly Detection

A modular, production-ready implementation of One-Class SVM with ensemble learning
and conformal prediction for anomaly detection.

This package refactors the original notebook implementation into a well-structured,
tested, and maintainable codebase while preserving the exact methodology to ensure
reproducible results.
"""

__version__ = "0.1.0"
__author__ = "BaKC-plus Development Team"

# Version info
VERSION = __version__

# Package-level imports
from .config import (
    BaKCConfig,
    DataConfig,
    ModelConfig,
    EnsembleConfig,
    ConformalConfig,
    LoggingConfig,
)
from .logger import setup_logging, get_logger

__all__ = [
    "__version__",
    "VERSION",
    "BaKCConfig",
    "DataConfig",
    "ModelConfig",
    "EnsembleConfig",
    "ConformalConfig",
    "LoggingConfig",
    "setup_logging",
    "get_logger",
]
