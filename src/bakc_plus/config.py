"""
Configuration management for BaKC-plus

This module provides configuration dataclasses for managing all hyperparameters
and settings. Configuration can be loaded from YAML files and validated.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataConfig:
    """
    Data loading and preprocessing configuration

    Attributes:
        dataset_name: Name of the dataset (e.g., 'cardio', 'gamma')
        data_dir: Root directory containing dataset folders
        output_dir: Directory for saving outputs (models, predictions, logs)
        train_fraction: Fraction of inliers to use for training (0.0 to 1.0)
        len_cal: Number of calibration samples (None = auto-compute)
        len_test: Number of test samples (None = auto-compute)
    """
    dataset_name: str
    data_dir: Path = Path("./data/input")
    output_dir: Path = Path("./output")
    train_fraction: float = 0.5
    len_cal: Optional[int] = None
    len_test: Optional[int] = None

    def __post_init__(self):
        """Convert string paths to Path objects"""
        if not isinstance(self.data_dir, Path):
            self.data_dir = Path(self.data_dir)
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)


@dataclass
class ModelConfig:
    """
    OC-SVM model configuration

    Attributes:
        nu: Upper bound on fraction of outliers in training data (0.0 to 1.0)
        kernel: Kernel type for SVM ('rbf', 'linear', 'poly', 'sigmoid')
        gamma: Kernel coefficient ('scale', 'auto', or float)
        cache_size: Kernel cache size in MB
        verbose: Enable verbose output during training
    """
    nu: float = 0.05
    kernel: str = "rbf"
    gamma: str = "scale"
    cache_size: int = 200
    verbose: bool = False


@dataclass
class EnsembleConfig:
    """
    Ensemble training configuration

    Attributes:
        num_models: Number of ensemble members per fold (M)
        num_folds: Number of cross-validation folds (K). None = auto-compute
        num_test_splits: Number of test splits per repetition (L)
        num_repetitions: Number of outer loop repetitions (J)
        random_state: Random seed for reproducibility
        use_multiprocessing: Enable parallel processing
        num_workers: Number of worker processes (None = auto-detect CPU count)
    """
    num_models: int = 5
    num_folds: Optional[int] = None
    num_test_splits: int = 20
    num_repetitions: int = 5
    random_state: int = 42
    use_multiprocessing: bool = True
    num_workers: Optional[int] = None


@dataclass
class ConformalConfig:
    """
    Conformal prediction configuration

    Attributes:
        alpha: FDR control level (0.0 to 1.0)
        scoring_method: Conformity scoring function
            - 'sigmoid': 1/(1 + exp(score)) [default, used in baseline]
            - 'normalize': Min-max normalization
            - 'signed_ohe': Binary classification based on sign
        quantile_method: numpy quantile method ('higher', 'lower', 'midpoint', etc.)
        fold_aggregation: How to aggregate scores within a fold ('mean', 'median')
        cross_fold_aggregation: How to aggregate across folds ('mean', 'median')
    """
    alpha: float = 0.05
    scoring_method: str = "sigmoid"
    quantile_method: str = "higher"
    fold_aggregation: str = "mean"
    cross_fold_aggregation: str = "median"


@dataclass
class LoggingConfig:
    """
    Logging configuration

    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_file_logging: Whether to enable file logging
        log_file: Path to log file (None = default: output/logs/bakc_plus.log)
        max_log_size_mb: Maximum log file size in MB before rotation
        backup_count: Number of backup log files to keep
    """
    level: str = "INFO"
    enable_file_logging: bool = True
    log_file: Optional[Path] = None
    max_log_size_mb: int = 10
    backup_count: int = 5

    def __post_init__(self):
        """Convert string paths to Path objects"""
        if self.log_file is not None and not isinstance(self.log_file, Path):
            self.log_file = Path(self.log_file)


@dataclass
class BaKCConfig:
    """
    Main BaKC-plus configuration

    This is the top-level configuration class that contains all sub-configurations.

    Attributes:
        data: Data loading configuration
        model: OC-SVM model configuration
        ensemble: Ensemble training configuration
        conformal: Conformal prediction configuration
        logging: Logging configuration
        save_models: Whether to save trained models to disk
        save_calibration: Whether to save calibration scores
        save_predictions: Whether to save predictions to CSV
    """
    data: DataConfig
    model: ModelConfig
    ensemble: EnsembleConfig
    conformal: ConformalConfig
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    save_models: bool = True
    save_calibration: bool = True
    save_predictions: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> 'BaKCConfig':
        """
        Load configuration from YAML file

        Args:
            path: Path to YAML configuration file

        Returns:
            BaKCConfig instance with loaded configuration

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML file is malformed
            TypeError: If configuration values have wrong types

        Example:
            >>> config = BaKCConfig.from_yaml('configs/cardio.yaml')
            >>> print(config.data.dataset_name)
            'cardio'
        """
        logger.info(f"Loading configuration from {path}")

        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            logger.error(f"Empty or invalid YAML file: {path}")
            raise ValueError(f"Empty or invalid YAML file: {path}")

        logger.debug(f"Loaded YAML with {len(config_dict)} sections")

        # Create sub-configurations with defaults for missing fields
        config = cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            ensemble=EnsembleConfig(**config_dict.get('ensemble', {})),
            conformal=ConformalConfig(**config_dict.get('conformal', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            save_models=config_dict.get('save_models', True),
            save_calibration=config_dict.get('save_calibration', True),
            save_predictions=config_dict.get('save_predictions', True),
        )

        logger.info(f"Configuration loaded successfully for dataset '{config.data.dataset_name}'")
        return config

    def validate(self) -> None:
        """
        Validate configuration values

        Raises:
            ValueError: If any configuration value is invalid

        This method checks:
        - alpha is in range (0, 1)
        - nu is in range (0, 1)
        - num_models is positive
        - num_test_splits is positive
        - num_repetitions is positive
        - train_fraction is in range (0, 1)
        - scoring_method is valid
        - aggregation methods are valid
        - data_dir exists or can be created
        - output_dir can be created
        """
        logger.debug("Validating configuration")

        # Validate conformal config
        if not (0 < self.conformal.alpha < 1):
            raise ValueError(
                f"conformal.alpha must be in (0, 1), got {self.conformal.alpha}"
            )

        allowed_scoring = ['sigmoid', 'normalize', 'signed_ohe']
        if self.conformal.scoring_method not in allowed_scoring:
            raise ValueError(
                f"conformal.scoring_method must be one of {allowed_scoring}, "
                f"got '{self.conformal.scoring_method}'"
            )

        allowed_aggregation = ['mean', 'median']
        if self.conformal.fold_aggregation not in allowed_aggregation:
            raise ValueError(
                f"conformal.fold_aggregation must be one of {allowed_aggregation}, "
                f"got '{self.conformal.fold_aggregation}'"
            )

        if self.conformal.cross_fold_aggregation not in allowed_aggregation:
            raise ValueError(
                f"conformal.cross_fold_aggregation must be one of {allowed_aggregation}, "
                f"got '{self.conformal.cross_fold_aggregation}'"
            )

        # Validate model config
        if not (0 < self.model.nu < 1):
            raise ValueError(
                f"model.nu must be in (0, 1), got {self.model.nu}"
            )

        allowed_kernels = ['rbf', 'linear', 'poly', 'sigmoid']
        if self.model.kernel not in allowed_kernels:
            raise ValueError(
                f"model.kernel must be one of {allowed_kernels}, "
                f"got '{self.model.kernel}'"
            )

        # Validate ensemble config
        if self.ensemble.num_models <= 0:
            raise ValueError(
                f"ensemble.num_models must be positive, got {self.ensemble.num_models}"
            )

        if self.ensemble.num_test_splits <= 0:
            raise ValueError(
                f"ensemble.num_test_splits must be positive, "
                f"got {self.ensemble.num_test_splits}"
            )

        if self.ensemble.num_repetitions <= 0:
            raise ValueError(
                f"ensemble.num_repetitions must be positive, "
                f"got {self.ensemble.num_repetitions}"
            )

        if self.ensemble.num_folds is not None and self.ensemble.num_folds <= 0:
            raise ValueError(
                f"ensemble.num_folds must be positive if specified, "
                f"got {self.ensemble.num_folds}"
            )

        # Validate data config
        if not (0 < self.data.train_fraction < 1):
            raise ValueError(
                f"data.train_fraction must be in (0, 1), "
                f"got {self.data.train_fraction}"
            )

        if self.data.len_cal is not None and self.data.len_cal <= 0:
            raise ValueError(
                f"data.len_cal must be positive if specified, got {self.data.len_cal}"
            )

        if self.data.len_test is not None and self.data.len_test <= 0:
            raise ValueError(
                f"data.len_test must be positive if specified, got {self.data.len_test}"
            )

        # Validate and create directories
        try:
            self.data.output_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Output directory verified/created: {self.data.output_dir}")
        except Exception as e:
            logger.error(f"Cannot create output directory {self.data.output_dir}: {e}")
            raise ValueError(
                f"Cannot create output directory {self.data.output_dir}: {e}"
            )

        # Check data_dir exists (will be created if needed during data loading)
        # We don't create it here to avoid clutter

        # Validate logging config
        allowed_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.logging.level.upper() not in allowed_levels:
            raise ValueError(
                f"logging.level must be one of {allowed_levels}, "
                f"got '{self.logging.level}'"
            )

        if self.logging.max_log_size_mb <= 0:
            raise ValueError(
                f"logging.max_log_size_mb must be positive, "
                f"got {self.logging.max_log_size_mb}"
            )

        if self.logging.backup_count < 0:
            raise ValueError(
                f"logging.backup_count must be non-negative, "
                f"got {self.logging.backup_count}"
            )

        logger.info("Configuration validation passed")

    def to_yaml(self, path: str) -> None:
        """
        Save configuration to YAML file

        Args:
            path: Path to save YAML configuration file

        Example:
            >>> config = BaKCConfig.from_yaml('configs/cardio.yaml')
            >>> config.model.nu = 0.1
            >>> config.to_yaml('configs/cardio_modified.yaml')
        """
        config_dict = {
            'data': {
                'dataset_name': self.data.dataset_name,
                'data_dir': str(self.data.data_dir),
                'output_dir': str(self.data.output_dir),
                'train_fraction': self.data.train_fraction,
                'len_cal': self.data.len_cal,
                'len_test': self.data.len_test,
            },
            'model': {
                'nu': self.model.nu,
                'kernel': self.model.kernel,
                'gamma': self.model.gamma,
                'cache_size': self.model.cache_size,
                'verbose': self.model.verbose,
            },
            'ensemble': {
                'num_models': self.ensemble.num_models,
                'num_folds': self.ensemble.num_folds,
                'num_test_splits': self.ensemble.num_test_splits,
                'num_repetitions': self.ensemble.num_repetitions,
                'random_state': self.ensemble.random_state,
                'use_multiprocessing': self.ensemble.use_multiprocessing,
                'num_workers': self.ensemble.num_workers,
            },
            'conformal': {
                'alpha': self.conformal.alpha,
                'scoring_method': self.conformal.scoring_method,
                'quantile_method': self.conformal.quantile_method,
                'fold_aggregation': self.conformal.fold_aggregation,
                'cross_fold_aggregation': self.conformal.cross_fold_aggregation,
            },
            'logging': {
                'level': self.logging.level,
                'enable_file_logging': self.logging.enable_file_logging,
                'log_file': str(self.logging.log_file) if self.logging.log_file else None,
                'max_log_size_mb': self.logging.max_log_size_mb,
                'backup_count': self.logging.backup_count,
            },
            'save_models': self.save_models,
            'save_calibration': self.save_calibration,
            'save_predictions': self.save_predictions,
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        """String representation of configuration"""
        return (
            f"BaKCConfig(\n"
            f"  dataset={self.data.dataset_name},\n"
            f"  nu={self.model.nu},\n"
            f"  num_models={self.ensemble.num_models},\n"
            f"  alpha={self.conformal.alpha},\n"
            f"  random_state={self.ensemble.random_state}\n"
            f")"
        )
