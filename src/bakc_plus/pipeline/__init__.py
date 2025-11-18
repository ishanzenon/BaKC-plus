"""
Pipeline module for BaKC-plus

This module provides end-to-end pipeline orchestration for training and evaluation.
"""

from .training import TrainingPipeline, train_pipeline

__all__ = [
    'TrainingPipeline',
    'train_pipeline',
]
