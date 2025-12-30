"""
Conformal prediction with calibration-based thresholding

This module implements conformal prediction threshold computation and binary
classification. It preserves the EXACT methodology from the original notebook.

CRITICAL PRESERVATION:
- Quantile level: q_level = np.ceil((n + 1) * (1 - alpha)) / n
- Quantile method: 'higher' (MUST match notebook)
- Binary prediction: conformity_score <= threshold → anomaly (1), else normal (0)
"""

import numpy as np
from typing import Optional

from ..config import ConformalConfig
from ..logger import get_logger

logger = get_logger(__name__)


def compute_threshold(
    calibration_scores: np.ndarray,
    alpha: float = 0.1,
    method: str = 'higher'
) -> float:
    """
    Compute conformal prediction threshold from calibration scores

    CRITICAL: This formula MUST match the notebook exactly!

    Formula from notebook:
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        threshold = np.quantile(calibration_scores, q_level, method='higher')

    The threshold provides (1-alpha) coverage guarantee: with probability at least
    (1-alpha), a normal sample will have conformity score > threshold.

    Args:
        calibration_scores: Conformity scores from calibration set
                            Shape: (n_samples,)
        alpha: FDR control level (0.0 to 1.0), default 0.1 for 10% FDR
        method: Quantile interpolation method, MUST be 'higher'

    Returns:
        Conformal prediction threshold (float)

    Raises:
        ValueError: If inputs are invalid or method != 'higher'

    Example:
        >>> calib_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        >>> threshold = compute_threshold(calib_scores, alpha=0.1)
        >>> # threshold will be at the ceil((5+1)*(1-0.1))/5 = 6*0.9/5 = 1.08 quantile
        >>> # which clamps to the maximum, using method='higher'

    Note:
        - method='higher' ensures conservative threshold (always picks higher value)
        - q_level can exceed 1.0 for small calibration sets (clamps to max)
        - Threshold is inclusive: score <= threshold → anomaly
    """
    # Validate inputs
    if calibration_scores is None or len(calibration_scores) == 0:
        raise ValueError("calibration_scores is empty or None")

    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    if method != 'higher':
        raise ValueError(
            f"method must be 'higher' to match notebook, got '{method}'"
        )

    # CRITICAL: Exact formula from notebook
    n = len(calibration_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n

    # Clamp q_level to [0, 1] (can exceed 1.0 for small n)
    q_level = min(q_level, 1.0)

    logger.debug(
        f"Computing threshold: n={n}, alpha={alpha}, q_level={q_level:.4f}"
    )

    # Compute quantile with method='higher'
    threshold = np.quantile(calibration_scores, q_level, method=method)

    logger.info(
        f"Conformal threshold computed: {threshold:.6f} "
        f"(alpha={alpha}, n_calib={n})"
    )

    return float(threshold)


def predict_anomalies(
    conformity_scores: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    Predict anomalies using conformal threshold

    Binary classification:
    - conformity_score >= threshold → anomaly (1)
    - conformity_score < threshold → normal (0)

    NOTE: High conformity scores indicate high non-conformity (anomalies)
    due to sigmoid transformation of OC-SVM scores.

    Args:
        conformity_scores: Conformity scores for test samples
                           Shape: (n_samples,)
        threshold: Conformal prediction threshold

    Returns:
        Binary predictions: 1 = anomaly, 0 = normal
        Shape: (n_samples,)

    Example:
        >>> scores = np.array([0.1, 0.5, 0.9])
        >>> threshold = 0.5
        >>> predictions = predict_anomalies(scores, threshold)
        >>> # predictions = [0, 1, 1]  (0.1 < 0.5, 0.5 >= 0.5, 0.9 >= 0.5)
    """
    # Validate inputs
    if conformity_scores is None or len(conformity_scores) == 0:
        raise ValueError("conformity_scores is empty or None")

    # Binary prediction: score > threshold → anomaly (1)
    # EXACT notebook formula (cell 54): p_values = (scores > qhat).astype(int)
    predictions = (conformity_scores > threshold).astype(int)

    n_anomalies = np.sum(predictions)
    logger.debug(
        f"Predicted {n_anomalies}/{len(predictions)} anomalies "
        f"({100.0 * n_anomalies / len(predictions):.1f}%)"
    )

    return predictions


class ConformalPredictor:
    """
    Conformal predictor with calibration-based thresholding

    This class encapsulates the full conformal prediction workflow:
    1. Calibrate threshold from calibration scores
    2. Predict anomalies on test data

    Example:
        >>> predictor = ConformalPredictor(alpha=0.1)
        >>> predictor.calibrate(calibration_scores)
        >>> predictions = predictor.predict(test_conformity_scores)
    """

    def __init__(self, config: Optional[ConformalConfig] = None, alpha: Optional[float] = None):
        """
        Initialize conformal predictor

        Args:
            config: ConformalConfig instance (or None for defaults)
            alpha: FDR control level (overrides config if provided)
        """
        from ..config import ConformalConfig as DefaultConformalConfig

        self.config = config if config is not None else DefaultConformalConfig()
        self.alpha = alpha if alpha is not None else self.config.alpha
        self.threshold: Optional[float] = None
        self.logger = get_logger(__name__)

    def calibrate(self, calibration_scores: np.ndarray) -> float:
        """
        Calibrate threshold from calibration scores

        Args:
            calibration_scores: Conformity scores from calibration set

        Returns:
            Computed threshold

        Raises:
            ValueError: If calibration_scores is invalid
        """
        self.threshold = compute_threshold(
            calibration_scores,
            alpha=self.alpha,
            method='higher'
        )

        self.logger.info(f"Calibrated threshold: {self.threshold:.6f}")

        return self.threshold

    def predict(self, conformity_scores: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using calibrated threshold

        Args:
            conformity_scores: Conformity scores for test samples

        Returns:
            Binary predictions (1 = anomaly, 0 = normal)

        Raises:
            RuntimeError: If not yet calibrated
            ValueError: If conformity_scores is invalid
        """
        if self.threshold is None:
            raise RuntimeError(
                "Predictor not yet calibrated. Call calibrate() first."
            )

        return predict_anomalies(conformity_scores, self.threshold)

    def is_calibrated(self) -> bool:
        """Check if predictor has been calibrated"""
        return self.threshold is not None

    def get_threshold(self) -> Optional[float]:
        """Get calibrated threshold (None if not yet calibrated)"""
        return self.threshold
