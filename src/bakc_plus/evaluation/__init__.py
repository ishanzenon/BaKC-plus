"""
Evaluation module for BaKC-plus

This module provides metrics computation (Power, FDR) and result aggregation.
"""

from .metrics import (
    compute_metrics,
    compute_power,
    compute_fdr,
    MetricsCalculator
)

__all__ = [
    'compute_metrics',
    'compute_power',
    'compute_fdr',
    'MetricsCalculator',
]
