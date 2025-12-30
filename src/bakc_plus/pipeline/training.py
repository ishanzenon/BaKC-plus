"""
Training pipeline for BaKC-plus

This module implements the end-to-end training pipeline that integrates:
- Ensemble training (Phase 2: EnsembleTrainer)
- Score accumulation (calibration + OOB)
- Sigmoid scoring (Phase 2: sigmoid_score)
- Conformal calibration (Phase 2: ConformalPredictor)

CRITICAL PRESERVATION:
- Score accumulation order: calibration THEN OOB
- Sigmoid applied to accumulated scores
- Threshold computed from all calibration data
"""

import numpy as np
from typing import Optional, Tuple, List

from ..config import ModelConfig, EnsembleConfig, ConformalConfig
from ..logger import get_logger
from ..model import EnsembleTrainer, OCSVMMember
from ..conformal import sigmoid_score, ConformalPredictor

logger = get_logger(__name__)


class TrainingPipeline:
    """
    End-to-end training pipeline for BaKC-plus

    This class orchestrates the complete training workflow:
    1. Train ensemble of OC-SVM models (K folds × M members)
    2. Accumulate calibration and OOB scores
    3. Apply sigmoid transformation
    4. Calibrate conformal predictor

    The pipeline preserves EXACT methodology from the notebook to ensure
    reproducible baseline results.

    Example:
        >>> from bakc_plus.config import ModelConfig, EnsembleConfig, ConformalConfig
        >>> pipeline = TrainingPipeline(
        ...     model_config=ModelConfig(nu=0.05),
        ...     ensemble_config=EnsembleConfig(num_models=5),
        ...     conformal_config=ConformalConfig(alpha=0.1)
        ... )
        >>> X_train = np.random.randn(1000, 10)
        >>> models, predictor = pipeline.train(X_train, len_cal=50)
        >>> print(f"Threshold: {predictor.get_threshold():.4f}")
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        ensemble_config: Optional[EnsembleConfig] = None,
        conformal_config: Optional[ConformalConfig] = None
    ):
        """
        Initialize training pipeline

        Args:
            model_config: OC-SVM model configuration (or None for defaults)
            ensemble_config: Ensemble training configuration (or None for defaults)
            conformal_config: Conformal prediction configuration (or None for defaults)
        """
        self.model_config = model_config if model_config is not None else ModelConfig()
        self.ensemble_config = ensemble_config if ensemble_config is not None else EnsembleConfig()
        self.conformal_config = conformal_config if conformal_config is not None else ConformalConfig()

        self.logger = get_logger(__name__)

        # State
        self.models: Optional[List[List[OCSVMMember]]] = None
        self.predictor: Optional[ConformalPredictor] = None
        self.is_trained = False

    def train(
        self,
        X_train: np.ndarray,
        len_cal: int,
        random_state: Optional[int] = None
    ) -> Tuple[List[List[OCSVMMember]], ConformalPredictor]:
        """
        Train complete BaKC-plus pipeline

        This is the MAIN training method. It implements the exact workflow from
        the notebook:
        1. Train ensemble (K-fold CV, M members per fold)
        2. Accumulate calibration + OOB scores
        3. Apply sigmoid transformation
        4. Calibrate conformal predictor

        Args:
            X_train: Training data (n_samples, n_features)
            len_cal: Calibration set size per fold
            random_state: Random seed for reproducibility (or None to use config)

        Returns:
            Tuple of:
            - models: Trained ensemble models (K folds × M members)
            - predictor: Calibrated conformal predictor

        Raises:
            ValueError: If X_train is invalid or len_cal is invalid

        Example:
            >>> pipeline = TrainingPipeline()
            >>> X_train = np.random.randn(1000, 10)
            >>> models, predictor = pipeline.train(X_train, len_cal=50, random_state=42)
            >>> print(f"Trained {len(models)} folds with {len(models[0])} members each")
            >>> print(f"Threshold: {predictor.get_threshold():.4f}")
        """
        # Use random_state from config if not provided
        if random_state is None:
            random_state = self.ensemble_config.random_state

        self.logger.info(
            f"Starting training pipeline: "
            f"data shape={X_train.shape}, len_cal={len_cal}, random_state={random_state}"
        )

        # Step 1: Train ensemble and accumulate scores
        self.logger.info("Step 1/3: Training ensemble models...")
        calibration_scores_raw, oob_scores_raw, models = self._train_ensemble(
            X_train, len_cal, random_state
        )

        self.logger.info(
            f"Ensemble training complete: "
            f"{len(calibration_scores_raw)} calib scores, "
            f"{len(oob_scores_raw)} OOB scores, "
            f"{sum(len(fold_models) for fold_models in models)} total models"
        )

        # Step 2: Accumulate all scores (calibration + OOB) and apply sigmoid
        self.logger.info("Step 2/3: Applying sigmoid scoring...")
        all_scores = self._accumulate_and_score(
            calibration_scores_raw, oob_scores_raw
        )

        self.logger.info(
            f"Score accumulation complete: {len(all_scores)} total conformity scores"
        )

        # Step 3: Calibrate conformal predictor
        self.logger.info("Step 3/3: Calibrating conformal predictor...")
        predictor = self._calibrate_predictor(all_scores)

        self.logger.info(
            f"Conformal calibration complete: threshold={predictor.get_threshold():.6f}"
        )

        # Store state
        self.models = models
        self.predictor = predictor
        self.is_trained = True

        self.logger.info("Training pipeline complete")

        return models, predictor

    def _train_ensemble(
        self,
        X_train: np.ndarray,
        len_cal: int,
        random_state: int
    ) -> Tuple[np.ndarray, np.ndarray, List[List[OCSVMMember]]]:
        """
        Train ensemble models and accumulate raw decision scores

        Args:
            X_train: Training data
            len_cal: Calibration set size per fold
            random_state: Random seed

        Returns:
            Tuple of (calibration_scores, oob_scores, models)
        """
        # Create ensemble trainer
        trainer = EnsembleTrainer(
            model_config=self.model_config,
            ensemble_config=self.ensemble_config
        )

        # Train ensemble
        calibration_scores, oob_scores, models = trainer.train_ensemble(
            X_train, len_cal, random_state
        )

        return calibration_scores, oob_scores, models

    def _accumulate_and_score(
        self,
        calibration_scores: np.ndarray,
        oob_scores: np.ndarray
    ) -> np.ndarray:
        """
        Accumulate calibration + OOB scores and apply sigmoid transformation

        CRITICAL: Score accumulation order matters!
        - Calibration scores FIRST
        - OOB scores SECOND
        - Then apply sigmoid to ALL accumulated scores

        Args:
            calibration_scores: Raw decision scores from calibration sets
            oob_scores: Raw decision scores from OOB samples

        Returns:
            Conformity scores after sigmoid transformation
        """
        # CRITICAL: Accumulate in correct order (calibration + OOB)
        all_scores_raw = np.concatenate([calibration_scores, oob_scores])

        self.logger.debug(
            f"Score accumulation: {len(calibration_scores)} calib + "
            f"{len(oob_scores)} OOB = {len(all_scores_raw)} total"
        )

        # Apply sigmoid transformation
        conformity_scores = sigmoid_score(all_scores_raw)

        self.logger.debug(
            f"Sigmoid scoring: range [{np.min(conformity_scores):.4f}, "
            f"{np.max(conformity_scores):.4f}]"
        )

        return conformity_scores

    def _calibrate_predictor(
        self,
        conformity_scores: np.ndarray
    ) -> ConformalPredictor:
        """
        Calibrate conformal predictor with conformity scores

        Args:
            conformity_scores: Conformity scores from calibration data

        Returns:
            Calibrated ConformalPredictor
        """
        # Create conformal predictor
        predictor = ConformalPredictor(config=self.conformal_config)

        # Calibrate with accumulated conformity scores
        predictor.calibrate(conformity_scores)

        return predictor

    def get_models(self) -> Optional[List[List[OCSVMMember]]]:
        """
        Get trained models

        Returns:
            List of lists of trained OCSVMMember instances, or None if not trained
        """
        return self.models

    def get_predictor(self) -> Optional[ConformalPredictor]:
        """
        Get calibrated predictor

        Returns:
            Calibrated ConformalPredictor, or None if not trained
        """
        return self.predictor

    def is_pipeline_trained(self) -> bool:
        """
        Check if pipeline has been trained

        Returns:
            True if trained, False otherwise
        """
        return self.is_trained


def train_pipeline(
    X_train: np.ndarray,
    len_cal: int,
    model_config: Optional[ModelConfig] = None,
    ensemble_config: Optional[EnsembleConfig] = None,
    conformal_config: Optional[ConformalConfig] = None,
    random_state: int = 42
) -> Tuple[List[List[OCSVMMember]], ConformalPredictor]:
    """
    Convenience function for training pipeline

    This provides a simpler interface that doesn't require instantiating
    the TrainingPipeline class.

    Args:
        X_train: Training data (n_samples, n_features)
        len_cal: Calibration set size per fold
        model_config: OC-SVM configuration (optional)
        ensemble_config: Ensemble configuration (optional)
        conformal_config: Conformal configuration (optional)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (models, predictor)

    Example:
        >>> X_train = np.random.randn(1000, 10)
        >>> models, predictor = train_pipeline(X_train, len_cal=50, random_state=42)
    """
    pipeline = TrainingPipeline(
        model_config=model_config,
        ensemble_config=ensemble_config,
        conformal_config=conformal_config
    )
    return pipeline.train(X_train, len_cal, random_state)
