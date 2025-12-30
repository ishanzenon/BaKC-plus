"""
Model module for BaKC-plus

This module provides OC-SVM implementation, ensemble learning, and bootstrapping.
Preserves the exact methodology from the original notebook to ensure reproducible results.
"""

from .bootstrapping import StratifiedBootstrapper, stratified_bootstrap
from .ocsvm import OCSVMMember, create_ocsvm_member
from .ensemble import EnsembleTrainer, train_ensemble

__all__ = [
    # Bootstrapping
    'StratifiedBootstrapper',
    'stratified_bootstrap',
    # OC-SVM
    'OCSVMMember',
    'create_ocsvm_member',
    # Ensemble
    'EnsembleTrainer',
    'train_ensemble',
]
