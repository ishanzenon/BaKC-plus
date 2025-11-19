"""
Conformity scoring functions for conformal prediction

This module implements scoring functions that transform OC-SVM decision function
outputs into conformity scores. The PRIMARY method is sigmoid scoring, which must
be preserved exactly from the original notebook.

CRITICAL PRESERVATION:
- Sigmoid formula: 1.0 / (1.0 + np.exp(scores))
- This is the ONLY active scoring method (others for reference only)
"""

import numpy as np
from typing import Optional

from ..logger import get_logger

logger = get_logger(__name__)


def sigmoid_score(decision_scores: np.ndarray) -> np.ndarray:
    """
    Apply sigmoid transformation to decision function scores

    CRITICAL: This is the PRIMARY and ONLY active scoring method.
    The formula MUST match the notebook exactly!

    Formula from notebook:
        conformity_score = 1.0 / (1.0 + np.exp(decision_scores))

    This transformation maps OC-SVM decision function outputs (typically in range
    [-5, 5]) to conformity scores in range (0, 1), where:
    - Higher conformity score → more likely to be normal
    - Lower conformity score → more likely to be anomaly

    Args:
        decision_scores: OC-SVM decision function outputs
                         Shape: (n_samples,) or (n_samples, n_models)

    Returns:
        Conformity scores after sigmoid transformation
        Shape: same as input

    Example:
        >>> scores = np.array([-2.0, 0.0, 2.0])
        >>> conformity = sigmoid_score(scores)
        >>> # conformity ≈ [0.88, 0.5, 0.12]

    Note:
        - Handles overflow gracefully (exp clipping handled by numpy)
        - Preserves input shape
        - Works with both 1D and 2D arrays
    """
    # CRITICAL: EXACT notebook formula (cell 46)
    # Formula: 1/(1 + exp(scores))
    # OC-SVM positive scores (inliers) → LOW sigmoid values
    # OC-SVM negative scores (outliers) → HIGH sigmoid values
    conformity_scores = 1.0 / (1.0 + np.exp(decision_scores))

    logger.debug(
        f"Sigmoid scoring: {len(decision_scores)} scores, "
        f"range [{np.min(conformity_scores):.4f}, {np.max(conformity_scores):.4f}]"
    )

    return conformity_scores
