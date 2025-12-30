"""
Ensemble training for OC-SVM models with K-fold cross-validation

This module implements ensemble coordination for training multiple OC-SVM members
across K-fold cross-validation. It preserves the EXACT methodology from the original
notebook to ensure reproducible results.

CRITICAL PRESERVATION:
- Dynamic fold calculation: len_splits = len(train) // len_cal (if < 20000) else 20
- KFold with shuffle=True and random_state for determinism
- Score aggregation: mean per-fold (M→1), concatenation across folds
- Model storage: List[List[OCSVMMember]] structure (K folds x M members)
"""

from typing import List, Optional, Tuple
import numpy as np
from sklearn.model_selection import KFold

from ..config import ModelConfig, EnsembleConfig
from ..logger import get_logger
from .ocsvm import OCSVMMember

logger = get_logger(__name__)


class EnsembleTrainer:
    """
    Ensemble trainer coordinating K-fold CV with M OC-SVM members per fold

    This class implements the EXACT ensemble training logic from the original
    notebook. It coordinates:
    1. Dynamic K-fold calculation based on dataset size
    2. K-fold cross-validation with shuffle=True
    3. M ensemble members per fold with bootstrapped training
    4. Calibration and OOB score accumulation

    CRITICAL: All formulas and logic must match the notebook exactly!

    Example:
        >>> from bakc_plus.config import ModelConfig
        >>> trainer = EnsembleTrainer(ModelConfig(num_models=5))
        >>> X_train = np.random.randn(1000, 10)
        >>> calib_scores, oob_scores, models = trainer.train_ensemble(
        ...     X_train, len_cal=50, random_state=42
        ... )
        >>> # calib_scores.shape = (K * 50,), models.shape = (K, 5)
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        ensemble_config: Optional[EnsembleConfig] = None
    ):
        """
        Initialize ensemble trainer

        Args:
            model_config: ModelConfig for OC-SVM parameters (or None for defaults)
            ensemble_config: EnsembleConfig for ensemble parameters (or None for defaults)

        Example:
            >>> trainer = EnsembleTrainer()  # Uses defaults
            >>> trainer = EnsembleTrainer(
            ...     model_config=ModelConfig(nu=0.01),
            ...     ensemble_config=EnsembleConfig(num_models=10)
            ... )
        """
        self.model_config = model_config if model_config is not None else ModelConfig()
        self.ensemble_config = ensemble_config if ensemble_config is not None else EnsembleConfig()
        self.logger = get_logger(__name__)

        # Model tracking
        self.models: List[List[OCSVMMember]] = []  # K folds x M members
        self.n_folds = 0
        self.n_members = self.ensemble_config.num_models

        # Score accumulation (stored per fold, concatenated at end)
        self.calibration_scores_per_fold: List[np.ndarray] = []
        self.oob_scores_per_fold: List[np.ndarray] = []

        self.logger.debug(
            f"EnsembleTrainer initialized: {self.n_members} members per fold"
        )

    def _calculate_num_folds(
        self,
        n_train: int,
        len_cal: int
    ) -> int:
        """
        Calculate number of CV folds dynamically

        CRITICAL: This formula MUST match the notebook exactly!

        Formula from notebook:
            if len(train) < 20000:
                len_splits = len(train) // len_cal
            else:
                len_splits = 20

        Args:
            n_train: Number of training samples
            len_cal: Calibration set size per fold

        Returns:
            Number of CV folds (integer)

        Example:
            >>> trainer = EnsembleTrainer()
            >>> trainer._calculate_num_folds(1000, 50)
            20
            >>> trainer._calculate_num_folds(500, 25)
            20
            >>> trainer._calculate_num_folds(25000, 100)
            20

        Note:
            - For small datasets (< 20000): fold count = n_train // len_cal
            - For large datasets (>= 20000): cap at 20 folds
            - This affects total model count (K * M) and score counts
        """
        if n_train < 20000:
            len_splits = n_train // len_cal
        else:
            len_splits = 20

        self.logger.debug(
            f"Calculated {len_splits} folds for n_train={n_train}, "
            f"len_cal={len_cal}"
        )

        return len_splits

    def train_ensemble(
        self,
        X_train: np.ndarray,
        len_cal: int,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, List[List[OCSVMMember]]]:
        """
        Train ensemble with K-fold cross-validation

        This is the MAIN training method. It implements the exact logic from
        the notebook for K-fold CV with ensemble member training.

        Algorithm:
        1. Calculate K (num_folds) using dynamic formula
        2. Create KFold splitter with shuffle=True, random_state
        3. For each fold:
           a. Split into train_indices and calib_indices
           b. Train M ensemble members on train_indices (with bootstrapping)
           c. Score calibration set with each member
           d. Aggregate per-member scores (mean across M members)
           e. Accumulate OOB scores from bootstrapping
           f. Store calibration and OOB scores for this fold
        4. Return: (all_calibration_scores, all_oob_scores, models)

        Args:
            X_train: Training feature matrix (n_samples, n_features)
            len_cal: Calibration set size per fold
            random_state: Random seed for reproducibility

        Returns:
            Tuple of:
            - calibration_scores: Concatenated calibration scores from all folds
                                  Shape: (K * len_cal,)
            - oob_scores: Concatenated OOB scores from all folds
                          Shape: (varies, ~K * n_train/num_models)
            - models: List of lists of trained OCSVMMember instances
                      Shape: K folds x M members

        Raises:
            ValueError: If X_train is empty or len_cal invalid

        Example:
            >>> trainer = EnsembleTrainer(ModelConfig(num_models=5))
            >>> X_train = np.random.randn(1000, 10)
            >>> calib_scores, oob_scores, models = trainer.train_ensemble(
            ...     X_train, len_cal=50, random_state=42
            ... )
            >>> len(calib_scores)  # K * 50
            1000
            >>> len(models)  # K folds
            20
            >>> len(models[0])  # M members per fold
            5
        """
        # Input validation
        if X_train is None or len(X_train) == 0:
            raise ValueError("X_train is empty or None")

        if len_cal <= 0 or len_cal >= len(X_train):
            raise ValueError(
                f"len_cal must be in (0, {len(X_train)}), got {len_cal}"
            )

        # Calculate number of folds
        self.n_folds = self._calculate_num_folds(len(X_train), len_cal)

        self.logger.info(
            f"Starting ensemble training: {self.n_folds} folds, "
            f"{self.n_members} members per fold, "
            f"len_cal={len_cal}, random_state={random_state}"
        )

        # Create K-fold splitter
        kfold = KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=random_state
        )

        # Initialize storage
        self.models = []
        all_calibration_scores = []
        all_oob_scores = []

        # Iterate over folds
        for fold_idx, (train_indices, calib_indices) in enumerate(
            kfold.split(X_train)
        ):
            self.logger.info(
                f"Fold {fold_idx + 1}/{self.n_folds}: "
                f"train={len(train_indices)}, calib={len(calib_indices)}"
            )

            # Get fold data
            X_train_fold = X_train[train_indices]
            X_calib_fold = X_train[calib_indices]

            # Train M ensemble members for this fold
            fold_models = []
            fold_calib_scores_per_member = []  # List of M arrays, each (n_calib,)
            fold_oob_scores = []

            for member_idx in range(self.n_members):
                self.logger.debug(
                    f"  Training member {member_idx + 1}/{self.n_members}"
                )

                # Create and fit OC-SVM member with bootstrapping
                member = OCSVMMember(config=self.model_config)
                model, leave_out_indices = member.fit(
                    X_train_fold,
                    member_idx=member_idx,
                    num_members=self.n_members,
                    fold_idx=fold_idx,
                    random_state=random_state
                )

                # Score calibration set
                calib_scores = member.decision_function(X_calib_fold)
                fold_calib_scores_per_member.append(calib_scores)

                # Score OOB samples (leave-out indices from bootstrapping)
                if leave_out_indices is not None and len(leave_out_indices) > 0:
                    X_oob = X_train_fold[leave_out_indices]
                    oob_scores = member.decision_function(X_oob)
                    fold_oob_scores.append(oob_scores)

                fold_models.append(member)

            # Aggregate calibration scores (mean across M members)
            # Shape: (M, n_calib) → (n_calib,)
            fold_calib_scores_aggregated = np.mean(
                fold_calib_scores_per_member,
                axis=0
            )

            # Concatenate OOB scores from all M members
            # Each member has ~n_train/M OOB samples
            if len(fold_oob_scores) > 0:
                fold_oob_scores_concatenated = np.concatenate(fold_oob_scores)
            else:
                # Edge case: no OOB scores (shouldn't happen with bootstrapping)
                fold_oob_scores_concatenated = np.array([])

            # Store
            self.models.append(fold_models)
            all_calibration_scores.append(fold_calib_scores_aggregated)
            all_oob_scores.append(fold_oob_scores_concatenated)

            self.logger.info(
                f"Fold {fold_idx + 1} complete: "
                f"{len(fold_calib_scores_aggregated)} calib scores, "
                f"{len(fold_oob_scores_concatenated)} OOB scores"
            )

        # Concatenate all scores across folds
        calibration_scores = np.concatenate(all_calibration_scores)
        oob_scores = np.concatenate(all_oob_scores)

        self.logger.info(
            f"Ensemble training complete: "
            f"total {len(calibration_scores)} calib scores, "
            f"{len(oob_scores)} OOB scores, "
            f"{self.n_folds * self.n_members} models trained"
        )

        return calibration_scores, oob_scores, self.models


def train_ensemble(
    X_train: np.ndarray,
    len_cal: int,
    model_config: Optional[ModelConfig] = None,
    ensemble_config: Optional[EnsembleConfig] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[List[OCSVMMember]]]:
    """
    Convenience function for training ensemble

    This provides a simpler interface that doesn't require instantiating
    the EnsembleTrainer class.

    Args:
        X_train: Training data (n_samples, n_features)
        len_cal: Calibration set size per fold
        model_config: ModelConfig for OC-SVM (optional, uses defaults if None)
        ensemble_config: EnsembleConfig for ensemble (optional, uses defaults if None)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of:
        - calibration_scores: Shape (K * len_cal,)
        - oob_scores: Shape varies (~K * n_train/M)
        - models: List[List[OCSVMMember]] (K folds x M members)

    Example:
        >>> X_train = np.random.randn(1000, 10)
        >>> calib_scores, oob_scores, models = train_ensemble(
        ...     X_train, len_cal=50, random_state=42
        ... )
    """
    trainer = EnsembleTrainer(
        model_config=model_config,
        ensemble_config=ensemble_config
    )
    return trainer.train_ensemble(X_train, len_cal, random_state)
