"""
Pipeline module for BaKC-plus

This module provides end-to-end pipeline orchestration for training and evaluation.
"""

from .training import TrainingPipeline, train_pipeline
from .prediction import PredictionPipeline, predict_pipeline

__all__ = [
    # Training
    'TrainingPipeline',
    'train_pipeline',
    # Prediction
    'PredictionPipeline',
    'predict_pipeline',
]
