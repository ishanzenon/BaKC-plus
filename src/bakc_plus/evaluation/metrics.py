"""
Evaluation metrics for anomaly detection

This module implements Power and FDR (False Discovery Rate) metrics for evaluating
the performance of the conformal anomaly detection system.

CRITICAL PRESERVATION:
- Power = TP / (TP + FN) = True Positive Rate for anomalies
- FDR = FP / (FP + TP) = False Discovery Rate
- Exact counting logic must match notebook
"""

import numpy as np
from typing import Dict, Optional, Tuple

from ..logger import get_logger

logger = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute Power and FDR metrics

    CRITICAL: Counting logic must match notebook exactly!

    Definitions:
    - Power = TP / (TP + FN) - fraction of true anomalies detected
    - FDR = FP / (FP + TP) - fraction of detections that are false

    Assumptions:
    - y_true: 1 = anomaly (positive class), 0 = normal (negative class)
    - y_pred: 1 = predicted anomaly, 0 = predicted normal

    Confusion matrix:
                    Pred=0 (Normal)   Pred=1 (Anomaly)
    True=0 (Normal)      TN                FP
    True=1 (Anomaly)     FN                TP

    Args:
        y_true: Ground truth labels (1 = anomaly, 0 = normal)
        y_pred: Predicted labels (1 = anomaly, 0 = normal)

    Returns:
        Dictionary with keys:
        - 'power': Power (TP / (TP + FN))
        - 'fdr': FDR (FP / (FP + TP)), 0.0 if FP + TP = 0
        - 'tp': True Positives count
        - 'fp': False Positives count
        - 'tn': True Negatives count
        - 'fn': False Negatives count

    Example:
        >>> y_true = np.array([1, 1, 0, 0, 1])
        >>> y_pred = np.array([1, 0, 0, 1, 1])
        >>> metrics = compute_metrics(y_true, y_pred)
        >>> # TP=2, FP=1, TN=1, FN=1
        >>> # Power = 2/(2+1) = 0.667
        >>> # FDR = 1/(1+2) = 0.333
    """
    # Validate inputs
    if y_true is None or len(y_true) == 0:
        raise ValueError("y_true is empty or None")

    if y_pred is None or len(y_pred) == 0:
        raise ValueError("y_pred is empty or None")

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length, "
            f"got {len(y_true)} and {len(y_pred)}"
        )

    # Convert to numpy arrays if needed
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Count confusion matrix elements
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Compute Power = TP / (TP + FN)
    if tp + fn > 0:
        power = tp / (tp + fn)
    else:
        power = 0.0  # No anomalies in ground truth

    # Compute FDR = FP / (FP + TP)
    if fp + tp > 0:
        fdr = fp / (fp + tp)
    else:
        fdr = 0.0  # No predictions made

    logger.debug(
        f"Metrics computed: TP={tp}, FP={fp}, TN={tn}, FN={fn}, "
        f"Power={power:.4f}, FDR={fdr:.4f}"
    )

    return {
        'power': float(power),
        'fdr': float(fdr),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
    }


def compute_power(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Power (True Positive Rate for anomalies)

    Power = TP / (TP + FN)

    Args:
        y_true: Ground truth labels (1 = anomaly, 0 = normal)
        y_pred: Predicted labels (1 = anomaly, 0 = normal)

    Returns:
        Power value (float in [0, 1])
    """
    metrics = compute_metrics(y_true, y_pred)
    return metrics['power']


def compute_fdr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute FDR (False Discovery Rate)

    FDR = FP / (FP + TP)

    Args:
        y_true: Ground truth labels (1 = anomaly, 0 = normal)
        y_pred: Predicted labels (1 = anomaly, 0 = normal)

    Returns:
        FDR value (float in [0, 1])
    """
    metrics = compute_metrics(y_true, y_pred)
    return metrics['fdr']


class MetricsCalculator:
    """
    Metrics calculator for anomaly detection evaluation

    This class provides a convenient interface for computing Power and FDR
    metrics across multiple evaluations.

    Example:
        >>> calculator = MetricsCalculator()
        >>> metrics = calculator.compute(y_true, y_pred)
        >>> print(f"Power: {metrics['power']:.2%}, FDR: {metrics['fdr']:.2%}")
    """

    def __init__(self):
        """Initialize metrics calculator"""
        self.logger = get_logger(__name__)

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute all metrics

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels

        Returns:
            Dictionary with power, fdr, and confusion matrix counts
        """
        return compute_metrics(y_true, y_pred)

    def compute_batch(
        self,
        y_true_list: list,
        y_pred_list: list
    ) -> Dict[str, np.ndarray]:
        """
        Compute metrics for multiple predictions

        Args:
            y_true_list: List of ground truth label arrays
            y_pred_list: List of predicted label arrays

        Returns:
            Dictionary with arrays of metrics:
            - 'power': Array of power values
            - 'fdr': Array of FDR values
            - 'mean_power': Mean power across all
            - 'mean_fdr': Mean FDR across all

        Example:
            >>> y_true_list = [np.array([1,1,0]), np.array([1,0,1])]
            >>> y_pred_list = [np.array([1,0,0]), np.array([1,1,0])]
            >>> results = calculator.compute_batch(y_true_list, y_pred_list)
        """
        if len(y_true_list) != len(y_pred_list):
            raise ValueError(
                f"y_true_list and y_pred_list must have same length, "
                f"got {len(y_true_list)} and {len(y_pred_list)}"
            )

        powers = []
        fdrs = []

        for y_true, y_pred in zip(y_true_list, y_pred_list):
            metrics = self.compute(y_true, y_pred)
            powers.append(metrics['power'])
            fdrs.append(metrics['fdr'])

        powers = np.array(powers)
        fdrs = np.array(fdrs)

        self.logger.info(
            f"Batch metrics: {len(powers)} evaluations, "
            f"mean Power={np.mean(powers):.4f}, mean FDR={np.mean(fdrs):.4f}"
        )

        return {
            'power': powers,
            'fdr': fdrs,
            'mean_power': float(np.mean(powers)),
            'mean_fdr': float(np.mean(fdrs)),
            'std_power': float(np.std(powers)),
            'std_fdr': float(np.std(fdrs)),
        }
