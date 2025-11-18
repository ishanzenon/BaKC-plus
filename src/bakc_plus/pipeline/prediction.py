"""
Prediction pipeline for BaKC-plus

This module implements the end-to-end prediction pipeline that integrates:
- Model scoring (Phase 2: OCSVMMember.decision_function)
- Score aggregation (mean per-fold, median cross-fold)
- Sigmoid scoring (Phase 2: sigmoid_score)
- Conformal prediction (Phase 2: ConformalPredictor.predict)

CRITICAL PRESERVATION:
- Score aggregation order: mean within fold, THEN median across folds
- Sigmoid applied AFTER aggregation
- Binary prediction: conformity_score <= threshold → anomaly (1)
"""

import numpy as np
from typing import List, Optional

from ..logger import get_logger
from ..model import OCSVMMember
from ..conformal import sigmoid_score, ConformalPredictor

logger = get_logger(__name__)


class PredictionPipeline:
    """
    End-to-end prediction pipeline for BaKC-plus

    This class orchestrates the complete prediction workflow:
    1. Score test data with all K×M trained models
    2. Aggregate scores (mean per-fold, median cross-fold)
    3. Apply sigmoid transformation
    4. Apply conformal threshold

    The pipeline preserves EXACT methodology from the notebook to ensure
    reproducible predictions.

    Example:
        >>> from bakc_plus.pipeline import TrainingPipeline, PredictionPipeline
        >>> # Train
        >>> train_pipeline = TrainingPipeline()
        >>> X_train = np.random.randn(1000, 10)
        >>> models, predictor = train_pipeline.train(X_train, len_cal=50)
        >>> # Predict
        >>> pred_pipeline = PredictionPipeline(models, predictor)
        >>> X_test = np.random.randn(100, 10)
        >>> predictions = pred_pipeline.predict(X_test)
        >>> print(f"Detected {np.sum(predictions)} anomalies out of {len(predictions)}")
    """

    def __init__(
        self,
        models: List[List[OCSVMMember]],
        predictor: ConformalPredictor
    ):
        """
        Initialize prediction pipeline

        Args:
            models: Trained ensemble models (K folds × M members)
            predictor: Calibrated conformal predictor

        Raises:
            ValueError: If models is empty or predictor not calibrated
        """
        if not models or len(models) == 0:
            raise ValueError("models cannot be empty")

        if not predictor.is_calibrated():
            raise ValueError("predictor must be calibrated before use")

        self.models = models
        self.predictor = predictor
        self.logger = get_logger(__name__)

        self.n_folds = len(models)
        self.n_members = len(models[0]) if models else 0

        self.logger.debug(
            f"PredictionPipeline initialized: {self.n_folds} folds, "
            f"{self.n_members} members per fold, "
            f"threshold={self.predictor.get_threshold():.6f}"
        )

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict anomalies on test data

        This is the MAIN prediction method. It implements the exact workflow from
        the notebook:
        1. Score X_test with all K×M models
        2. Aggregate per-fold: mean of M scores
        3. Aggregate cross-fold: median of K scores
        4. Apply sigmoid transformation
        5. Apply conformal threshold

        Args:
            X_test: Test data (n_samples, n_features)

        Returns:
            Binary predictions (n_samples,): 1 = anomaly, 0 = normal

        Raises:
            ValueError: If X_test is invalid

        Example:
            >>> pipeline = PredictionPipeline(models, predictor)
            >>> X_test = np.random.randn(100, 10)
            >>> predictions = pipeline.predict(X_test)
            >>> n_anomalies = np.sum(predictions)
            >>> print(f"Detected {n_anomalies} anomalies ({100*n_anomalies/len(predictions):.1f}%)")
        """
        # Validate input
        if X_test is None or len(X_test) == 0:
            raise ValueError("X_test is empty or None")

        self.logger.info(
            f"Starting prediction: X_test shape={X_test.shape}, "
            f"threshold={self.predictor.get_threshold():.6f}"
        )

        # Step 1: Score with all models and aggregate
        self.logger.info("Step 1/3: Scoring test data with ensemble models...")
        aggregated_scores = self._score_and_aggregate(X_test)

        self.logger.debug(
            f"Score aggregation complete: {len(aggregated_scores)} final scores, "
            f"range=[{np.min(aggregated_scores):.4f}, {np.max(aggregated_scores):.4f}]"
        )

        # Step 2: Apply sigmoid transformation
        self.logger.info("Step 2/3: Applying sigmoid transformation...")
        conformity_scores = sigmoid_score(aggregated_scores)

        self.logger.debug(
            f"Sigmoid complete: range=[{np.min(conformity_scores):.4f}, "
            f"{np.max(conformity_scores):.4f}]"
        )

        # Step 3: Apply conformal threshold
        self.logger.info("Step 3/3: Applying conformal threshold...")
        predictions = self.predictor.predict(conformity_scores)

        n_anomalies = np.sum(predictions)
        self.logger.info(
            f"Prediction complete: {n_anomalies}/{len(predictions)} anomalies "
            f"({100.0 * n_anomalies / len(predictions):.1f}%)"
        )

        return predictions

    def _score_and_aggregate(self, X_test: np.ndarray) -> np.ndarray:
        """
        Score test data with all models and aggregate

        CRITICAL: Score aggregation order matters!
        1. Score with all K×M models
        2. Aggregate per-fold: mean of M members → 1 score per sample
        3. Aggregate cross-fold: median of K folds → 1 final score per sample

        Args:
            X_test: Test data (n_samples, n_features)

        Returns:
            Aggregated decision scores (n_samples,)
        """
        n_samples = len(X_test)
        all_fold_scores = []  # Will be shape (K, n_samples)

        # Score with each fold
        for fold_idx, fold_models in enumerate(self.models):
            self.logger.debug(f"Scoring fold {fold_idx + 1}/{self.n_folds}...")

            # Score with all M members in this fold
            member_scores = []  # Will be shape (M, n_samples)
            for member in fold_models:
                scores = member.decision_function(X_test)
                member_scores.append(scores)

            # Aggregate M members → 1 score per sample (MEAN)
            fold_scores_aggregated = np.mean(member_scores, axis=0)
            all_fold_scores.append(fold_scores_aggregated)

        # Aggregate K folds → 1 final score per sample (MEDIAN)
        # Shape: (K, n_samples) → (n_samples,)
        final_scores = np.median(all_fold_scores, axis=0)

        self.logger.debug(
            f"Aggregation: {self.n_folds} folds × {self.n_members} members → "
            f"{len(final_scores)} final scores per sample"
        )

        return final_scores

    def predict_scores(self, X_test: np.ndarray) -> np.ndarray:
        """
        Get conformity scores without applying threshold

        Useful for analysis or custom thresholding.

        Args:
            X_test: Test data (n_samples, n_features)

        Returns:
            Conformity scores (n_samples,) in range (0, 1)

        Example:
            >>> scores = pipeline.predict_scores(X_test)
            >>> # Analyze score distribution
            >>> import matplotlib.pyplot as plt
            >>> plt.hist(scores, bins=50)
        """
        aggregated_scores = self._score_and_aggregate(X_test)
        conformity_scores = sigmoid_score(aggregated_scores)
        return conformity_scores

    def get_threshold(self) -> float:
        """
        Get conformal prediction threshold

        Returns:
            Threshold value (float)
        """
        return self.predictor.get_threshold()


def predict_pipeline(
    models: List[List[OCSVMMember]],
    predictor: ConformalPredictor,
    X_test: np.ndarray
) -> np.ndarray:
    """
    Convenience function for prediction pipeline

    This provides a simpler interface that doesn't require instantiating
    the PredictionPipeline class.

    Args:
        models: Trained ensemble models (K folds × M members)
        predictor: Calibrated conformal predictor
        X_test: Test data (n_samples, n_features)

    Returns:
        Binary predictions (1 = anomaly, 0 = normal)

    Example:
        >>> X_test = np.random.randn(100, 10)
        >>> predictions = predict_pipeline(models, predictor, X_test)
    """
    pipeline = PredictionPipeline(models, predictor)
    return pipeline.predict(X_test)
