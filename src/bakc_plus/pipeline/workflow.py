"""
End-to-end workflow for BaKC-plus experiments

This module implements the complete experimental workflow that integrates:
- Data loading and splitting (J repetitions × L test splits)
- Training pipeline (ensemble + conformal calibration)
- Prediction pipeline (scoring + conformal prediction)
- Evaluation (Power, FDR metrics)
- Result aggregation (median per-rep, mean/std/p90 across reps)

CRITICAL PRESERVATION:
- Exact loop structure from notebook (J repetitions, L test splits)
- Random state management: base_seed + rep_idx for repetitions, + split_idx for splits
- Aggregation order: median across L splits per repetition, then mean/std/p90 across J reps
- 80/20 train/test split with shuffling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split

from ..logger import get_logger
from ..config import ModelConfig, EnsembleConfig, ConformalConfig
from ..evaluation import MetricsCalculator
from .training import TrainingPipeline
from .prediction import PredictionPipeline

logger = get_logger(__name__)


class BaKCWorkflow:
    """
    End-to-end workflow for BaKC-plus experiments

    This class orchestrates the complete experimental workflow from the notebook:
    1. For each of J repetitions:
       - For each of L test splits:
         - Split data (80% train, 20% test)
         - Train pipeline (ensemble + conformal)
         - Predict on test set
         - Compute metrics (Power, FDR)
       - Aggregate metrics across L splits (median)
    2. Aggregate metrics across J repetitions (mean, std, p90)

    This preserves the EXACT methodology from the notebook to ensure reproducible
    baseline verification.

    Example:
        >>> workflow = BaKCWorkflow(
        ...     num_repetitions=5,
        ...     num_test_splits=20,
        ...     len_cal=50,
        ...     random_state=42
        ... )
        >>> results = workflow.run_experiment(X, y)
        >>> print(f"Power: {results['power_mean']:.2%} ± {results['power_std']:.2%}")
        >>> print(f"FDR: {results['fdr_mean']:.2%} ± {results['fdr_std']:.2%}")
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        ensemble_config: Optional[EnsembleConfig] = None,
        conformal_config: Optional[ConformalConfig] = None,
        num_repetitions: int = 5,
        num_test_splits: int = 20,
        len_cal: int = 50,
        test_size: float = 0.2,
        random_state: Optional[int] = None
    ):
        """
        Initialize BaKC workflow

        Args:
            model_config: OC-SVM configuration (default: nu=0.05, kernel='rbf')
            ensemble_config: Ensemble configuration (default: num_models=5)
            conformal_config: Conformal prediction configuration (default: alpha=0.1)
            num_repetitions: Number of repetitions (J) (default: 5)
            num_test_splits: Number of test splits per repetition (L) (default: 20)
            len_cal: Calibration set size for each fold (default: 50)
            test_size: Fraction of data for test set (default: 0.2 = 80/20 split)
            random_state: Base random state for reproducibility (default: None)

        Example:
            >>> workflow = BaKCWorkflow(
            ...     ensemble_config=EnsembleConfig(num_models=5),
            ...     num_repetitions=5,
            ...     num_test_splits=20,
            ...     random_state=42
            ... )
        """
        self.model_config = model_config if model_config is not None else ModelConfig()
        self.ensemble_config = ensemble_config if ensemble_config is not None else EnsembleConfig()
        self.conformal_config = conformal_config if conformal_config is not None else ConformalConfig()

        self.num_repetitions = num_repetitions
        self.num_test_splits = num_test_splits
        self.len_cal = len_cal
        self.test_size = test_size
        self.random_state = random_state

        self.logger = get_logger(__name__)
        self.metrics_calculator = MetricsCalculator()

        self.logger.debug(
            f"BaKCWorkflow initialized: J={num_repetitions} reps, "
            f"L={num_test_splits} splits, len_cal={len_cal}, "
            f"test_size={test_size}, random_state={random_state}"
        )

    def run_experiment(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Run full experiment with J repetitions × L test splits

        This is the MAIN workflow method that implements the exact loop structure
        from the notebook:
        1. Outer loop: J repetitions with different random seeds
        2. Inner loop: L test splits per repetition
        3. Per-split: train → predict → evaluate
        4. Per-rep aggregation: median of L splits
        5. Cross-rep aggregation: mean, std, p90 of J reps

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) - 1 = anomaly, 0 = normal

        Returns:
            Dictionary with aggregated results:
            - 'power_mean': Mean power across J repetitions
            - 'power_std': Std of power across J repetitions
            - 'power_p90': 90th percentile of power
            - 'fdr_mean': Mean FDR across J repetitions
            - 'fdr_std': Std of FDR across J repetitions
            - 'fdr_p90': 90th percentile of FDR
            - 'power_per_rep': Array of power values (J,)
            - 'fdr_per_rep': Array of FDR values (J,)
            - 'all_metrics': List of per-split metrics for each repetition

        Raises:
            ValueError: If X or y are invalid

        Example:
            >>> workflow = BaKCWorkflow(num_repetitions=5, num_test_splits=20)
            >>> X = np.random.randn(1000, 10)
            >>> y = np.zeros(1000)
            >>> y[-100:] = 1  # 10% anomalies
            >>> results = workflow.run_experiment(X, y)
            >>> print(f"Power: {results['power_mean']:.2%}")
        """
        # Validate inputs
        if X is None or len(X) == 0:
            raise ValueError("X is empty or None")
        if y is None or len(y) == 0:
            raise ValueError("y is empty or None")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length, got {len(X)} and {len(y)}")

        self.logger.info(
            f"Starting experiment: X shape={X.shape}, y shape={y.shape}, "
            f"J={self.num_repetitions} reps, L={self.num_test_splits} splits"
        )

        # Storage for per-repetition aggregated metrics
        power_per_rep = []
        fdr_per_rep = []
        all_metrics_per_rep = []

        # Outer loop: J repetitions
        for rep_idx in range(self.num_repetitions):
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"REPETITION {rep_idx + 1}/{self.num_repetitions}")
            self.logger.info(f"{'='*60}")

            # Compute seed for this repetition
            if self.random_state is not None:
                rep_seed = self.random_state + rep_idx
            else:
                rep_seed = None

            # Run all L test splits for this repetition
            metrics_this_rep = self._run_repetition(X, y, rep_idx, rep_seed)

            # Aggregate across L splits (MEDIAN as per notebook)
            power_splits = [m['power'] for m in metrics_this_rep]
            fdr_splits = [m['fdr'] for m in metrics_this_rep]

            power_rep = np.median(power_splits)
            fdr_rep = np.median(fdr_splits)

            power_per_rep.append(power_rep)
            fdr_per_rep.append(fdr_rep)
            all_metrics_per_rep.append(metrics_this_rep)

            self.logger.info(
                f"Repetition {rep_idx + 1} complete: "
                f"Power={power_rep:.4f} (median of {len(power_splits)} splits), "
                f"FDR={fdr_rep:.4f}"
            )

        # Convert to arrays
        power_per_rep = np.array(power_per_rep)
        fdr_per_rep = np.array(fdr_per_rep)

        # Aggregate across J repetitions (MEAN, STD, P90 as per notebook)
        results = {
            'power_mean': float(np.mean(power_per_rep)),
            'power_std': float(np.std(power_per_rep)),
            'power_p90': float(np.percentile(power_per_rep, 90)),
            'fdr_mean': float(np.mean(fdr_per_rep)),
            'fdr_std': float(np.std(fdr_per_rep)),
            'fdr_p90': float(np.percentile(fdr_per_rep, 90)),
            'power_per_rep': power_per_rep,
            'fdr_per_rep': fdr_per_rep,
            'all_metrics': all_metrics_per_rep,
            'num_repetitions': self.num_repetitions,
            'num_test_splits': self.num_test_splits,
        }

        self.logger.info(f"\n{'='*60}")
        self.logger.info("EXPERIMENT COMPLETE")
        self.logger.info(f"{'='*60}")
        self.logger.info(
            f"Final Results ({self.num_repetitions} repetitions × "
            f"{self.num_test_splits} splits):"
        )
        self.logger.info(
            f"  Power: {results['power_mean']:.4f} ± {results['power_std']:.4f} "
            f"(p90={results['power_p90']:.4f})"
        )
        self.logger.info(
            f"  FDR:   {results['fdr_mean']:.4f} ± {results['fdr_std']:.4f} "
            f"(p90={results['fdr_p90']:.4f})"
        )

        return results

    def _run_repetition(
        self,
        X: np.ndarray,
        y: np.ndarray,
        rep_idx: int,
        rep_seed: Optional[int]
    ) -> List[Dict[str, float]]:
        """
        Run all L test splits for one repetition

        Args:
            X: Feature matrix
            y: Labels
            rep_idx: Repetition index (for logging)
            rep_seed: Random seed for this repetition

        Returns:
            List of L metric dictionaries (one per split)
        """
        metrics_this_rep = []

        # Inner loop: L test splits
        for split_idx in range(self.num_test_splits):
            self.logger.info(
                f"  Split {split_idx + 1}/{self.num_test_splits} "
                f"(rep {rep_idx + 1})"
            )

            # Compute seed for this split
            if rep_seed is not None:
                split_seed = rep_seed + split_idx
            else:
                split_seed = None

            # Run single split: train → predict → evaluate
            metrics = self._run_single_split(X, y, split_seed)
            metrics_this_rep.append(metrics)

            self.logger.debug(
                f"    Split {split_idx + 1}: Power={metrics['power']:.4f}, "
                f"FDR={metrics['fdr']:.4f}"
            )

        return metrics_this_rep

    def _run_single_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_seed: Optional[int]
    ) -> Dict[str, float]:
        """
        Run single train/test split

        This implements the core workflow:
        1. Split data (80% train, 20% test)
        2. Train pipeline (ensemble + conformal)
        3. Predict on test set
        4. Compute metrics (Power, FDR)

        Args:
            X: Feature matrix
            y: Labels
            split_seed: Random seed for this split

        Returns:
            Dictionary with metrics: power, fdr, tp, fp, tn, fn
        """
        # Step 1: Split data (80/20 train/test)
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=split_seed,
            stratify=None  # No stratification as per notebook
        )

        # Extract only inliers for training (y=0)
        # This matches the notebook's approach of training only on normal data
        inlier_mask = (y_train_full == 0)
        X_train = X_train_full[inlier_mask]

        self.logger.debug(
            f"      Data split: train={len(X_train)} inliers "
            f"(from {len(X_train_full)} total), test={len(X_test)}"
        )

        # Step 2: Train pipeline
        train_pipeline = TrainingPipeline(
            model_config=self.model_config,
            ensemble_config=self.ensemble_config,
            conformal_config=self.conformal_config
        )

        models, predictor = train_pipeline.train(
            X_train,
            len_cal=self.len_cal,
            random_state=split_seed
        )

        # Step 3: Predict on test set
        pred_pipeline = PredictionPipeline(models, predictor)
        y_pred = pred_pipeline.predict(X_test)

        # Step 4: Compute metrics
        metrics = self.metrics_calculator.compute(y_test, y_pred)

        return metrics


def run_bakc_experiment(
    X: np.ndarray,
    y: np.ndarray,
    model_config: Optional[ModelConfig] = None,
    ensemble_config: Optional[EnsembleConfig] = None,
    conformal_config: Optional[ConformalConfig] = None,
    num_repetitions: int = 5,
    num_test_splits: int = 20,
    len_cal: int = 50,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to run BaKC experiment

    This provides a simpler interface that doesn't require instantiating
    the BaKCWorkflow class.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,) - 1 = anomaly, 0 = normal
        model_config: OC-SVM configuration
        ensemble_config: Ensemble configuration
        conformal_config: Conformal prediction configuration
        num_repetitions: Number of repetitions (J)
        num_test_splits: Number of test splits per repetition (L)
        len_cal: Calibration set size
        random_state: Base random state

    Returns:
        Dictionary with experiment results

    Example:
        >>> X = np.random.randn(1000, 10)
        >>> y = np.zeros(1000)
        >>> y[-100:] = 1
        >>> results = run_bakc_experiment(X, y, num_repetitions=5, random_state=42)
        >>> print(f"Power: {results['power_mean']:.2%}")
    """
    workflow = BaKCWorkflow(
        model_config=model_config,
        ensemble_config=ensemble_config,
        conformal_config=conformal_config,
        num_repetitions=num_repetitions,
        num_test_splits=num_test_splits,
        len_cal=len_cal,
        random_state=random_state
    )
    return workflow.run_experiment(X, y)
