"""
One-Class SVM wrapper for ensemble members

This module provides a wrapper around sklearn's OneClassSVM that integrates
with the stratified bootstrapping logic. It preserves the exact training
methodology from the original notebook.
"""

from typing import Optional, Tuple
import numpy as np
from sklearn.svm import OneClassSVM

from ..config import ModelConfig
from ..logger import get_logger
from .bootstrapping import StratifiedBootstrapper

logger = get_logger(__name__)


class OCSVMMember:
    """
    Single OC-SVM ensemble member

    This class wraps sklearn's OneClassSVM and integrates it with the
    stratified bootstrapping logic. Each ensemble member is trained on
    a bootstrapped subset of the training data.

    CRITICAL PRESERVATION:
    - Random state hashing must match notebook
    - Bootstrapping integration must be correct
    - Model parameters must match config

    Attributes:
        nu: Upper bound on fraction of outliers (anomalies) in training data
        kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
        gamma: Kernel coefficient ('scale', 'auto', or float)
        cache_size: Kernel cache size in MB
        model: Fitted sklearn OneClassSVM model (None before fitting)

    Example:
        >>> config = ModelConfig(nu=0.05, kernel='rbf')
        >>> member = OCSVMMember(config)
        >>> X_train = np.random.randn(100, 5)
        >>> model, leave_out = member.fit(
        ...     X_train, member_idx=0, num_members=5,
        ...     fold_idx=0, random_state=42
        ... )
        >>> scores = member.decision_function(X_test)
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        nu: Optional[float] = None,
        kernel: Optional[str] = None,
        gamma: Optional[str] = None,
        cache_size: int = 200
    ):
        """
        Initialize OC-SVM ensemble member

        Args:
            config: ModelConfig with parameters (preferred)
            nu: Upper bound on outlier fraction (override config)
            kernel: Kernel type (override config)
            gamma: Kernel coefficient (override config)
            cache_size: Kernel cache size in MB

        Note:
            If config is provided, its values are used unless overridden
            by explicit parameters.
        """
        # Use config if provided, otherwise use explicit parameters
        if config is not None:
            self.nu = nu if nu is not None else config.nu
            self.kernel = kernel if kernel is not None else config.kernel
            self.gamma = gamma if gamma is not None else config.gamma
            self.cache_size = cache_size if cache_size != 200 else config.cache_size
        else:
            self.nu = nu if nu is not None else 0.05
            self.kernel = kernel if kernel is not None else 'rbf'
            self.gamma = gamma if gamma is not None else 'scale'
            self.cache_size = cache_size

        self.model = None
        self.bootstrapper = StratifiedBootstrapper()
        self.logger = get_logger(__name__)

        self.logger.debug(
            f"OCSVMMember initialized: nu={self.nu}, kernel={self.kernel}, "
            f"gamma={self.gamma}, cache_size={self.cache_size}"
        )

    def fit(
        self,
        X_train: np.ndarray,
        member_idx: int = 0,
        num_members: Optional[int] = None,
        fold_idx: int = 0,
        random_state: int = 42
    ) -> Tuple[OneClassSVM, Optional[np.ndarray]]:
        """
        Fit OC-SVM with optional bootstrapping

        This method trains the One-Class SVM on the training data. If num_members
        is specified, it performs stratified bootstrapping to create diverse
        ensemble members.

        CRITICAL: Random state hashing MUST match original notebook!

        Args:
            X_train: Training feature matrix (n_samples, n_features)
            member_idx: Ensemble member index (0 to num_members-1)
            num_members: Total ensemble members (None = no bootstrapping)
            fold_idx: Cross-validation fold index
            random_state: Base random seed

        Returns:
            Tuple of:
            - model: Fitted OneClassSVM model
            - leave_out_indices: Indices left out during bootstrapping
                                 (None if no bootstrapping)

        Raises:
            ValueError: If X_train is empty or None
            ValueError: If member_idx >= num_members

        Example:
            >>> member = OCSVMMember(nu=0.05, kernel='rbf')
            >>> X_train = np.random.randn(100, 5)
            >>> model, leave_out = member.fit(
            ...     X_train, member_idx=2, num_members=5, fold_idx=0, random_state=42
            ... )
            >>> model.n_support_  # Number of support vectors
            >>> len(leave_out)  # ~20 (100/5)
        """
        # Validate inputs
        if X_train is None or len(X_train) == 0:
            raise ValueError("X_train is empty or None")

        if num_members is not None:
            if member_idx < 0 or member_idx >= num_members:
                raise ValueError(
                    f"member_idx must be in [0, {num_members-1}], got {member_idx}"
                )

        self.logger.info(
            f"Fitting OC-SVM: member {member_idx}, fold {fold_idx}, "
            f"data shape {X_train.shape}, nu={self.nu}"
        )

        # Bootstrap if num_members specified
        if num_members is not None:
            self.logger.debug(
                f"Performing bootstrapping with {num_members} members"
            )

            X_train_bootstrap, leave_out_indices = \
                self.bootstrapper.perform_bootstrapping(
                    X_train, member_idx, num_members, random_state, fold_idx
                )

            self.logger.debug(
                f"Bootstrap: train {len(X_train_bootstrap)}, "
                f"leave-out {len(leave_out_indices)}"
            )
        else:
            self.logger.debug("No bootstrapping (using full training data)")
            X_train_bootstrap = X_train
            leave_out_indices = None

        # Initialize and fit model
        self.model = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma,
            cache_size=self.cache_size
        )

        self.logger.debug(f"Fitting OneClassSVM on {len(X_train_bootstrap)} samples")

        # Fit the model
        self.model.fit(X_train_bootstrap)

        n_sv = int(np.sum(self.model.n_support_))
        self.logger.info(
            f"OC-SVM fitted: {n_sv} support vectors "
            f"({100.0 * n_sv / len(X_train_bootstrap):.1f}%)"
        )

        return self.model, leave_out_indices

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function scores

        This method computes the signed distance from the decision boundary
        for each sample. Positive scores indicate inliers, negative scores
        indicate outliers.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Decision scores (n_samples,)

        Raises:
            ValueError: If model not fitted yet

        Example:
            >>> member = OCSVMMember(nu=0.05)
            >>> member.fit(X_train)
            >>> scores = member.decision_function(X_test)
            >>> predictions = (scores < 0).astype(int)  # 1 = outlier
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        self.logger.debug(f"Computing decision function for {len(X)} samples")

        scores = self.model.decision_function(X)

        self.logger.debug(
            f"Decision scores: min={scores.min():.3f}, "
            f"max={scores.max():.3f}, mean={scores.mean():.3f}"
        )

        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels (1 = inlier, -1 = outlier)

        This is a wrapper around sklearn's predict() method.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predictions: 1 for inliers, -1 for outliers (n_samples,)

        Raises:
            ValueError: If model not fitted yet

        Note:
            For conformal prediction, use decision_function() instead,
            as we need the raw scores for calibration.
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")

        predictions = self.model.predict(X)

        return predictions

    def is_fitted(self) -> bool:
        """
        Check if model is fitted

        Returns:
            True if model is fitted, False otherwise
        """
        return self.model is not None

    def get_n_support(self) -> int:
        """
        Get number of support vectors

        Returns:
            Number of support vectors

        Raises:
            ValueError: If model not fitted yet
        """
        if not self.is_fitted():
            raise ValueError("Model not fitted yet")

        return int(np.sum(self.model.n_support_))

    def __repr__(self) -> str:
        """String representation"""
        fitted = "fitted" if self.is_fitted() else "not fitted"
        n_sv = int(np.sum(self.model.n_support_)) if self.is_fitted() else None
        support = f", {n_sv} SVs" if self.is_fitted() else ""

        return (
            f"OCSVMMember(nu={self.nu}, kernel={self.kernel}, "
            f"gamma={self.gamma}, {fitted}{support})"
        )


def create_ocsvm_member(config: ModelConfig) -> OCSVMMember:
    """
    Factory function to create OC-SVM member from config

    This is a convenience function for creating members from configuration.

    Args:
        config: ModelConfig with parameters

    Returns:
        OCSVMMember instance

    Example:
        >>> from bakc_plus import BaKCConfig
        >>> config = BaKCConfig.from_yaml('configs/cardio.yaml')
        >>> member = create_ocsvm_member(config.model)
    """
    return OCSVMMember(config=config)
