"""
Conformal prediction module for BaKC-plus

This module provides conformal prediction functionality including calibration
set creation, scoring functions, and prediction with coverage guarantees.
"""

from .scoring import sigmoid_score
from .prediction import (
    compute_threshold,
    predict_anomalies,
    ConformalPredictor
)

__all__ = [
    # Scoring
    'sigmoid_score',
    # Prediction
    'compute_threshold',
    'predict_anomalies',
    'ConformalPredictor',
]
