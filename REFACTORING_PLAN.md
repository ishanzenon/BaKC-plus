# BaKC-plus Refactoring Plan

## Executive Summary

This document outlines a comprehensive plan to refactor the monolithic Jupyter notebook (`oc-svm-x-cv-x-bagging (1).ipynb`) into a production-ready, modular Python package while **preserving the exact methodology** to ensure reproducible results.

### Primary Objectives

1. **Modularity**: Transform monolithic notebook into well-organized Python modules
2. **Reproducibility**: Ensure refactored code produces identical results (Power: 90.29%, FDR: 8.47% on CARDIO)
3. **Maintainability**: Add type hints, documentation, logging, and error handling
4. **Testability**: Implement comprehensive unit and integration tests
5. **Scalability**: Support multiple datasets with configuration-based approach
6. **Methodology Preservation**: DO NOT alter the core algorithm, even if inefficiencies exist

---

## Current State Assessment

### Baseline Performance (CARDIO Dataset)
- **Average Power**: 90.29% (90th %ile: 100%, Min: 76.47%, Std: 8.61%)
- **Average FDR**: 8.47% (90th %ile: 12.42%, Min: 4.82%, Std: 3.39%)
- **Dataset**: 1,831 samples, 21 features, 9.61% anomalies
- **Total Predictions**: 1,003 (229 anomalies predicted, 176 actual)

### Algorithm Architecture

```
┌─────────────────────────────────────────────────────┐
│ 1. Data Loading & Splitting                        │
│    - Separate inliers (90.4%) / outliers (9.6%)    │
│    - Train/Test split of inliers                   │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│ 2. K-Fold Cross-Validation Setup                   │
│    - K folds (dynamic: len/len_cal or max 20)      │
│    - Each fold: train_index, calib_index           │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│ 3. Ensemble Training (Per Fold)                    │
│    - M=5 OC-SVM models per fold                    │
│    - Stratified bootstrapping (leave-one-out)      │
│    - Total models: K × M (e.g., 3 × 5 = 15)        │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│ 4. Scoring & Calibration                           │
│    - Sigmoid transform: 1/(1 + exp(decision_fn))   │
│    - Per-fold aggregation: mean across M models    │
│    - Accumulate calibration + OOB scores           │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│ 5. Conformal Prediction Threshold                  │
│    - q_level = ceil((n+1)*(1-α))/n                 │
│    - qhat = quantile(calib_scores, q_level)        │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│ 6. Test Evaluation (L=20 splits)                   │
│    - Median aggregation across folds               │
│    - Binary prediction: (score > qhat)             │
│    - Compute Power & FDR per split                 │
└─────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│ 7. Aggregate Results (J=5 repetitions)             │
│    - Collect all powers & FDRs                     │
│    - Report mean, std, 90th percentile             │
└─────────────────────────────────────────────────────┘
```

### Critical Algorithm Details (MUST PRESERVE)

1. **Stratified Bootstrapping**:
   ```python
   # Shuffle indices, split into M groups
   # Member m gets all groups EXCEPT group m
   indices = np.arange(len(X_train))
   rnd_state.shuffle(indices)
   index_sets = np.array_split(indices, num_members)
   leave_out_indices = index_sets[member_idx]
   mask = np.ones_like(indices, dtype=bool)
   mask[leave_out_indices] = False
   X_train_bootstrap = X_train[mask]
   ```

2. **Random State Hashing**:
   ```python
   rnd = hash((member_idx, fold_idx, random_state)) % 4294967296
   rnd = rnd ^ 0x7FFFFFFF
   ```

3. **Sigmoid Scoring**:
   ```python
   def sigmoid_scored_sample(calib):
       return 1/(1 + np.exp(calib))
   ```

4. **Quantile Threshold**:
   ```python
   q_level = np.ceil((n+1)*(1-alpha))/n
   qhat = np.quantile(calibration_scores, q_level, method='higher')
   ```

5. **Score Aggregation**:
   - **Per-fold**: Mean of M model scores (`np.mean(calibration_scores_i, axis=1)`)
   - **Cross-fold**: Median (`np.median(scores, axis=1)`)

6. **Dynamic K-Fold Splits**:
   ```python
   len_splits = len(train) // len_cal if len(train) < 20000 else 20
   ```

7. **OOB + Calibration Scores**:
   ```python
   calibration_scores = np.append(calibration_scores, calibration_scores_i)
   calibration_scores = np.append(calibration_scores, calibration_scores_i_leave_out)
   ```

### Identified Issues

| Category | Issue | Impact | Action |
|----------|-------|--------|--------|
| **Critical** | `visited_i` undefined | ✅ Fixed | Already resolved |
| **High** | Hardcoded Kaggle paths | Breaks portability | Replace with config |
| **High** | Global variable pollution | Hard to test | Encapsulate in classes |
| **High** | No input validation | Silent failures | Add validation layer |
| **Medium** | No type hints | Poor IDE support | Add typing throughout |
| **Medium** | No logging | Hard to debug | Add structured logging |
| **Medium** | No error handling | Crashes on edge cases | Add try/except blocks |
| **Low** | Mixed scoring functions | Code bloat | Document active one |

---

## Target Architecture

### Package Structure

```
bakc_plus/
├── README.md                          # Usage documentation
├── REFACTORING_PLAN.md                # This document
├── requirements.txt                   # Dependencies (existing)
├── setup.py                           # Package installation
├── pytest.ini                         # Pytest configuration
├── .gitignore                         # Git ignore rules
│
├── data/                              # Dataset storage (existing)
│   └── input/
│       ├── cardio/
│       ├── gamma/
│       └── ...
│
├── output/                            # Results output (existing)
│   ├── models/                        # Trained models
│   ├── calibration/                   # Calibration artifacts
│   ├── predictions/                   # Prediction CSVs
│   └── metrics/                       # Evaluation metrics
│
├── configs/                           # Configuration files
│   ├── default.yaml                   # Default configuration
│   ├── cardio.yaml                    # CARDIO-specific config
│   └── gamma.yaml                     # GAMMA-specific config
│
├── src/                               # Source code
│   └── bakc_plus/
│       ├── __init__.py                # Package initialization
│       ├── config.py                  # Configuration management
│       ├── logger.py                  # Logging setup
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py              # Data loading utilities
│       │   ├── splitter.py            # Train/test splitting
│       │   └── validator.py           # Data validation
│       ├── model/
│       │   ├── __init__.py
│       │   ├── ocsvm.py               # OC-SVM wrapper
│       │   ├── ensemble.py            # Ensemble logic
│       │   └── bootstrapping.py       # Stratified bootstrapping
│       ├── conformal/
│       │   ├── __init__.py
│       │   ├── calibration.py         # Calibration set creation
│       │   ├── scoring.py             # Scoring functions
│       │   └── prediction.py          # Conformal prediction
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py             # Power, FDR computation
│       │   └── aggregation.py         # Result aggregation
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── trainer.py             # Training pipeline
│       │   ├── evaluator.py           # Evaluation pipeline
│       │   └── orchestrator.py        # Full orchestration
│       └── utils/
│           ├── __init__.py
│           ├── paths.py               # Path management
│           └── serialization.py       # Model save/load
│
├── scripts/                           # Executable scripts
│   ├── train.py                       # Training entry point
│   ├── evaluate.py                    # Evaluation entry point
│   └── verify_baseline.py             # Baseline verification
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures
│   ├── test_data/                     # Test data
│   │   └── synthetic_dataset.csv
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_data_loader.py
│   │   ├── test_bootstrapping.py
│   │   ├── test_scoring.py
│   │   ├── test_metrics.py
│   │   └── test_calibration.py
│   └── integration/
│       ├── test_pipeline.py
│       └── test_baseline_reproduction.py
│
└── notebooks/                         # Analysis notebooks
    ├── exploratory_analysis.ipynb     # EDA
    ├── baseline_comparison.ipynb      # Compare old vs new
    └── results_visualization.ipynb    # Visualize results
```

---

## Detailed Module Design

### 1. Configuration Module (`src/bakc_plus/config.py`)

**Purpose**: Centralized configuration management with validation

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import yaml

@dataclass
class DataConfig:
    """Data loading and preprocessing configuration"""
    dataset_name: str
    data_dir: Path = Path("./data/input")
    output_dir: Path = Path("./output")

    # Split ratios
    train_fraction: float = 0.5
    len_cal: Optional[int] = None  # Auto-computed if None
    len_test: Optional[int] = None  # Auto-computed if None

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class ModelConfig:
    """OC-SVM model configuration"""
    nu: float = 0.05
    kernel: str = "rbf"
    gamma: str = "scale"
    cache_size: int = 200
    verbose: bool = False

@dataclass
class EnsembleConfig:
    """Ensemble training configuration"""
    num_models: int = 5  # M: ensemble members per fold
    num_folds: Optional[int] = None  # K: dynamic if None
    num_test_splits: int = 20  # L: test splits per repetition
    num_repetitions: int = 5  # J: outer loop repetitions

    # Random state
    random_state: int = 42

    # Parallel processing
    use_multiprocessing: bool = True
    num_workers: Optional[int] = None  # Auto-detect if None

@dataclass
class ConformalConfig:
    """Conformal prediction configuration"""
    alpha: float = 0.05  # FDR control level
    scoring_method: str = "sigmoid"  # "sigmoid", "normalize", "signed_ohe"
    quantile_method: str = "higher"  # numpy quantile method

    # Score aggregation
    fold_aggregation: str = "mean"  # "mean" or "median"
    cross_fold_aggregation: str = "median"  # "mean" or "median"

@dataclass
class BaKCConfig:
    """Main BaKC+ configuration"""
    data: DataConfig
    model: ModelConfig
    ensemble: EnsembleConfig
    conformal: ConformalConfig

    # Execution settings
    save_models: bool = True
    save_calibration: bool = True
    save_predictions: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> 'BaKCConfig':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict.get('model', {})),
            ensemble=EnsembleConfig(**config_dict.get('ensemble', {})),
            conformal=ConformalConfig(**config_dict.get('conformal', {}))
        )

    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        # Implementation
        pass

    def validate(self):
        """Validate configuration values"""
        assert 0 < self.conformal.alpha < 1, "Alpha must be in (0, 1)"
        assert self.ensemble.num_models > 0, "num_models must be positive"
        assert self.data.train_fraction > 0, "train_fraction must be positive"
        # More validations...
```

### 2. Data Module (`src/bakc_plus/data/`)

#### `loader.py` - Data Loading

```python
from typing import Tuple
import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    """Load and preprocess datasets"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Load dataset from CSV file

        Args:
            dataset_name: Name of dataset (e.g., 'cardio', 'gamma')

        Returns:
            DataFrame with features and 'y' column
        """
        dataset_path = self.data_dir / dataset_name / f"{dataset_name}.csv"
        df = pd.read_csv(dataset_path)

        # Rename 'Class' to 'y' if needed
        if 'Class' in df.columns:
            df.rename(columns={'Class': 'y'}, inplace=True)

        return df

    def split_inliers_outliers(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separate inliers (y=0) and outliers (y=1)

        Args:
            df: Input dataframe

        Returns:
            (inliers_df, outliers_df)
        """
        inliers_df = df.loc[df['y'] == 0].copy()
        outliers_df = df.loc[df['y'] == 1].copy()
        return inliers_df, outliers_df
```

#### `validator.py` - Data Validation

```python
import pandas as pd
import numpy as np

class DataValidator:
    """Validate input data"""

    @staticmethod
    def validate_dataframe(df: pd.DataFrame, expected_cols: int = None):
        """Validate dataframe structure and content"""
        assert df is not None, "DataFrame is None"
        assert len(df) > 0, "DataFrame is empty"
        assert 'y' in df.columns, "Target column 'y' not found"

        if expected_cols:
            assert df.shape[1] == expected_cols, \
                f"Expected {expected_cols} columns, got {df.shape[1]}"

        # Check for missing values
        if df.isnull().any().any():
            raise ValueError("Data contains NaN values")

        # Check target is binary
        if not df['y'].isin([0, 1]).all():
            raise ValueError("Target must be binary (0, 1)")

        return True
```

### 3. Model Module (`src/bakc_plus/model/`)

#### `bootstrapping.py` - Stratified Bootstrapping

```python
import numpy as np
from typing import Tuple

class StratifiedBootstrapper:
    """Stratified bootstrapping for ensemble members"""

    @staticmethod
    def perform_bootstrapping(
        X_train: np.ndarray,
        member_idx: int,
        num_members: int,
        random_state: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform stratified bootstrapping (leave-one-out style)

        CRITICAL: This method MUST match the original implementation exactly!

        Args:
            X_train: Training feature matrix
            member_idx: Index of current ensemble member (0-indexed)
            num_members: Total number of ensemble members
            random_state: Random seed for shuffling

        Returns:
            (X_train_bootstrap, leave_out_indices)
            - X_train_bootstrap: Training data for this member
            - leave_out_indices: Indices left out (OOB samples)
        """
        rnd_state = np.random.RandomState(random_state)
        indices = np.arange(len(X_train))
        rnd_state.shuffle(indices)

        # Split into num_members groups
        index_sets = np.array_split(indices, num_members)

        # Leave out group corresponding to member_idx
        leave_out_indices = index_sets[member_idx]

        # Create mask: True for indices to keep
        mask = np.ones_like(indices, dtype=bool)
        mask[leave_out_indices] = False

        X_train_bootstrap = X_train[mask]

        return X_train_bootstrap, leave_out_indices
```

#### `ocsvm.py` - OC-SVM Wrapper

```python
from sklearn.svm import OneClassSVM
import numpy as np
from typing import Optional, Tuple
from .bootstrapping import StratifiedBootstrapper

class OCSVMMember:
    """Single OC-SVM ensemble member"""

    def __init__(self, nu: float = 0.05, kernel: str = "rbf"):
        self.nu = nu
        self.kernel = kernel
        self.model = None

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

        CRITICAL: Random state hashing MUST match original!

        Args:
            X_train: Training data
            member_idx: Ensemble member index
            num_members: Total ensemble members (None = no bootstrapping)
            fold_idx: Cross-validation fold index
            random_state: Base random state

        Returns:
            (model, leave_out_indices)
        """
        if X_train is None or len(X_train) == 0:
            raise ValueError("X_train is empty or None")

        # Initialize model
        self.model = OneClassSVM(nu=self.nu, kernel=self.kernel)

        # Hash random state (CRITICAL: preserve original logic)
        rnd = hash((member_idx, fold_idx, random_state)) % 4294967296
        rnd = rnd ^ 0x7FFFFFFF

        # Bootstrap if num_members specified
        if num_members is not None:
            bootstrapper = StratifiedBootstrapper()
            X_train_bootstrap, leave_out_indices = \
                bootstrapper.perform_bootstrapping(
                    X_train, member_idx, num_members, rnd
                )
        else:
            X_train_bootstrap = X_train
            leave_out_indices = None

        # Fit model
        self.model.fit(X_train_bootstrap)

        return self.model, leave_out_indices

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function scores"""
        if self.model is None:
            raise ValueError("Model not fitted yet")
        return self.model.decision_function(X)
```

### 4. Conformal Module (`src/bakc_plus/conformal/`)

#### `scoring.py` - Scoring Functions

```python
import numpy as np

class ScoringFunctions:
    """Conformity scoring functions"""

    @staticmethod
    def sigmoid_scored_sample(scores: np.ndarray) -> np.ndarray:
        """
        Sigmoid transformation of decision function scores

        CRITICAL: This is the active scoring method used in baseline!

        Args:
            scores: Raw decision function scores from OC-SVM

        Returns:
            Sigmoid-transformed scores in [0, 1]
        """
        return 1.0 / (1.0 + np.exp(scores))

    @staticmethod
    def normalize_scored_sample(scores: np.ndarray) -> np.ndarray:
        """Min-max normalization (alternative method)"""
        minval, maxval = scores.min(), scores.max()
        if maxval == minval:
            return np.zeros_like(scores)
        return (scores - minval) / (maxval - minval)

    @staticmethod
    def signed_ohe_scored_sample(scores: np.ndarray) -> np.ndarray:
        """Binary classification based on sign"""
        return np.where(scores < 0, 1, 0)

    @classmethod
    def get_scoring_function(cls, method: str):
        """Get scoring function by name"""
        methods = {
            'sigmoid': cls.sigmoid_scored_sample,
            'normalize': cls.normalize_scored_sample,
            'signed_ohe': cls.signed_ohe_scored_sample
        }
        if method not in methods:
            raise ValueError(f"Unknown scoring method: {method}")
        return methods[method]
```

#### `calibration.py` - Calibration Set Creation

```python
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
from ..model.ocsvm import OCSVMMember
from .scoring import ScoringFunctions

class CalibrationSetCreator:
    """Create calibration sets using K-fold CV"""

    def __init__(
        self,
        num_models: int,
        num_folds: Optional[int],
        len_cal: int,
        scoring_method: str = "sigmoid",
        random_state: int = 42
    ):
        self.num_models = num_models
        self.num_folds = num_folds
        self.len_cal = len_cal
        self.scoring_method = scoring_method
        self.random_state = random_state
        self.scoring_fn = ScoringFunctions.get_scoring_function(scoring_method)

    def create_calibration_sets(
        self,
        train: pd.DataFrame
    ) -> Tuple[List[OneClassSVM], np.ndarray, np.ndarray]:
        """
        Create calibration sets via K-fold CV with ensemble bagging

        CRITICAL: This logic MUST match original exactly!

        Args:
            train: Training dataframe with 'y' column

        Returns:
            (models, calibration_scores, calibration_scores_std)
        """
        # Initialize storage
        calibration_scores = np.array([], dtype=np.float32)
        calibration_scores_std = np.array([], dtype=np.float32)
        models = []

        # Determine number of folds (CRITICAL: preserve original logic)
        if self.num_folds is None:
            len_splits = len(train) // self.len_cal if len(train) < 20000 else 20
        else:
            len_splits = self.num_folds

        # K-Fold CV
        kf = KFold(n_splits=len_splits, shuffle=True, random_state=self.random_state)

        X_train = train.drop('y', axis=1).to_numpy()
        ground_truth = train['y'].to_numpy()

        # Track visited indices (for debugging/validation)
        visited = {}

        # Iterate over folds
        for i, (train_index, calib_index) in tqdm(
            enumerate(kf.split(train)),
            total=len_splits,
            desc="Creating calibration sets"
        ):
            # Initialize fold-level storage
            calibration_scores_i = np.array([], dtype=np.float32)
            calibration_scores_i_leave_out = np.array([], dtype=np.float32)
            visited_i = {}  # CRITICAL: Initialize per-fold!

            # Train ensemble members for this fold
            for j in range(self.num_models):
                # Fit OC-SVM member
                ocsvm = OCSVMMember(nu=0.05)
                model, leave_out_indices = ocsvm.fit(
                    X_train[train_index],
                    member_idx=j,
                    num_members=self.num_models,
                    fold_idx=i,
                    random_state=self.random_state
                )

                # Track visited indices
                for idx in calib_index:
                    visited[idx] = visited.get(idx, 0) + 1
                for idx in leave_out_indices:
                    visited_i[idx] = visited_i.get(idx, 0) + 1

                # Score calibration set
                calib_scores_j = self.scoring_fn(
                    model.decision_function(X_train[calib_index])
                )

                # Score OOB samples
                oob_scores_j = self.scoring_fn(
                    model.decision_function(X_train[leave_out_indices])
                )

                # Accumulate
                if calibration_scores_i.shape[0] > 0:
                    calibration_scores_i = np.vstack([
                        calibration_scores_i, calib_scores_j
                    ])
                else:
                    calibration_scores_i = calib_scores_j

                calibration_scores_i_leave_out = np.append(
                    calibration_scores_i_leave_out, oob_scores_j
                )

                models.append(model)

            # Aggregate fold scores (CRITICAL: mean per sample)
            calibration_scores_i_std = np.std(calibration_scores_i, axis=0)
            calibration_scores_i = np.mean(calibration_scores_i, axis=0)

            # Append to global calibration scores
            calibration_scores = np.append(calibration_scores, calibration_scores_i)
            calibration_scores = np.append(
                calibration_scores, calibration_scores_i_leave_out
            )
            calibration_scores_std = np.append(
                calibration_scores_std, calibration_scores_i_std
            )

        return models, calibration_scores, calibration_scores_std
```

### 5. Evaluation Module (`src/bakc_plus/evaluation/`)

#### `metrics.py` - Metrics Computation

```python
import numpy as np

class Metrics:
    """Compute evaluation metrics"""

    @staticmethod
    def compute_power(
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ) -> float:
        """
        Compute statistical power (True Positive Rate for anomalies)

        Power = TP / (TP + FN)

        Args:
            predictions: Binary predictions (1=anomaly, 0=normal)
            ground_truth: True labels (1=anomaly, 0=normal)

        Returns:
            Power value in [0, 1]
        """
        if predictions is None or len(predictions) == 0:
            return None

        true_positives = np.sum(predictions == ground_truth)
        false_negatives = np.sum((1 - predictions) == ground_truth)

        if true_positives + false_negatives == 0:
            return 0.0

        power = true_positives / (true_positives + false_negatives)
        return power

    @staticmethod
    def compute_fdr(
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ) -> float:
        """
        Compute False Discovery Rate (for inliers)

        FDR = FP / (TP + FP)

        Args:
            predictions: Binary predictions
            ground_truth: True labels

        Returns:
            FDR value in [0, 1]
        """
        if predictions is None or len(predictions) == 0:
            return None

        true_positives = np.sum(predictions == ground_truth)
        false_positives = np.sum(predictions == (1 - ground_truth))

        if true_positives + false_positives == 0:
            return 0.0

        fdr = false_positives / (true_positives + false_positives)
        return fdr
```

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

#### Step 1.1: Project Setup
- [ ] Create directory structure
- [ ] Set up `setup.py` for package installation
- [ ] Configure `pytest.ini` for testing
- [ ] Update `.gitignore`

#### Step 1.2: Configuration System
- [ ] Implement `config.py` with dataclasses
- [ ] Create default YAML configs
- [ ] Add config validation

#### Step 1.3: Logging System
- [ ] Implement `logger.py`
- [ ] Add structured logging throughout

#### Step 1.4: Data Module
- [ ] Implement `data/loader.py`
- [ ] Implement `data/validator.py`
- [ ] Implement `data/splitter.py`
- [ ] Write unit tests for data module

### Phase 2: Core Algorithm (Week 2)

#### Step 2.1: Model Module
- [ ] Implement `model/bootstrapping.py`
- [ ] Implement `model/ocsvm.py`
- [ ] Implement `model/ensemble.py`
- [ ] Write unit tests for model module

#### Step 2.2: Conformal Module
- [ ] Implement `conformal/scoring.py`
- [ ] Implement `conformal/calibration.py`
- [ ] Implement `conformal/prediction.py`
- [ ] Write unit tests for conformal module

#### Step 2.3: Evaluation Module
- [ ] Implement `evaluation/metrics.py`
- [ ] Implement `evaluation/aggregation.py`
- [ ] Write unit tests for evaluation module

### Phase 3: Pipeline Integration (Week 3)

#### Step 3.1: Pipeline Module
- [ ] Implement `pipeline/trainer.py`
- [ ] Implement `pipeline/evaluator.py`
- [ ] Implement `pipeline/orchestrator.py`
- [ ] Write integration tests

#### Step 3.2: Command-Line Scripts
- [ ] Implement `scripts/train.py`
- [ ] Implement `scripts/evaluate.py`
- [ ] Implement `scripts/verify_baseline.py`
- [ ] Add CLI argument parsing

### Phase 4: Verification & Testing (Week 4)

#### Step 4.1: Baseline Verification
- [ ] Run refactored code on CARDIO dataset
- [ ] Compare results with baseline (Power: 90.29%, FDR: 8.47%)
- [ ] Debug any discrepancies
- [ ] Document differences (if any)

#### Step 4.2: Comprehensive Testing
- [ ] Complete unit test suite (>80% coverage)
- [ ] Integration tests for full pipeline
- [ ] Edge case testing
- [ ] Performance benchmarking

#### Step 4.3: Documentation
- [ ] API documentation (docstrings)
- [ ] Usage guide (README.md)
- [ ] Configuration guide
- [ ] Troubleshooting guide

---

## Verification Strategy

### Baseline Reproduction Test

**Objective**: Ensure refactored code produces identical results to baseline

**Method**:
1. Run refactored pipeline on CARDIO dataset
2. Use same random seeds (random_state=42)
3. Use same hyperparameters (nu=0.05, K=3, M=5, L=20, J=5)
4. Compare:
   - Average Power: Target = 90.29% (tolerance: ±0.5%)
   - Average FDR: Target = 8.47% (tolerance: ±0.5%)
   - Total predictions: Target = 1,003
   - Predicted anomalies: Target = 229

**Acceptance Criteria**:
- Power within ±0.5% of baseline
- FDR within ±0.5% of baseline
- Same total prediction count

**Debug Plan** (if results differ):
1. Check random state propagation
2. Verify bootstrapping logic
3. Verify scoring function
4. Verify quantile computation
5. Verify aggregation methods

---

## Risk Mitigation

### High-Risk Areas

1. **Random State Management**
   - **Risk**: Different random sequences → different results
   - **Mitigation**: Unit tests for random state hashing; deterministic tests

2. **Numpy Aggregation Functions**
   - **Risk**: Mean/median behavior with axis parameter
   - **Mitigation**: Extensive unit tests; shape assertions

3. **Bootstrapping Logic**
   - **Risk**: Off-by-one errors in index partitioning
   - **Mitigation**: Test with small synthetic datasets; visualize partitions

4. **Quantile Computation**
   - **Risk**: Method parameter ('higher') affects threshold
   - **Mitigation**: Verify quantile method matches original

5. **Data Type Precision**
   - **Risk**: float16 → float32 conversion affects results
   - **Mitigation**: Use float32 throughout; document precision choices

---

## Configuration Examples

### `configs/cardio.yaml`

```yaml
data:
  dataset_name: cardio
  data_dir: ./data/input
  output_dir: ./output/cardio
  train_fraction: 0.5
  len_cal: null  # Auto-compute
  len_test: null  # Auto-compute

model:
  nu: 0.05
  kernel: rbf
  gamma: scale

ensemble:
  num_models: 5
  num_folds: null  # Dynamic: len(train)//len_cal or 20
  num_test_splits: 20
  num_repetitions: 5
  random_state: 42
  use_multiprocessing: true

conformal:
  alpha: 0.05
  scoring_method: sigmoid
  quantile_method: higher
  fold_aggregation: mean
  cross_fold_aggregation: median

# Execution
save_models: true
save_calibration: true
save_predictions: true
```

---

## Success Criteria

### Functional Requirements
- ✅ Produces same results as baseline (Power: 90.29%, FDR: 8.47% on CARDIO)
- ✅ Supports multiple datasets via configuration
- ✅ Modular codebase with clear separation of concerns
- ✅ Comprehensive error handling and validation

### Non-Functional Requirements
- ✅ Unit test coverage >80%
- ✅ Integration tests for full pipeline
- ✅ Type hints throughout
- ✅ Comprehensive docstrings
- ✅ Structured logging
- ✅ Clear documentation

### Performance Requirements
- ✅ Runtime comparable to original notebook (2-5 minutes for CARDIO)
- ✅ Memory usage within acceptable limits
- ✅ Supports multiprocessing for parallelization

---

## Timeline

### Week 1: Infrastructure
- Days 1-2: Project setup, directory structure
- Days 3-4: Configuration and logging systems
- Days 5-7: Data module + unit tests

### Week 2: Core Algorithm
- Days 8-10: Model module (bootstrapping, OC-SVM) + unit tests
- Days 11-13: Conformal module (scoring, calibration) + unit tests
- Day 14: Evaluation module + unit tests

### Week 3: Pipeline & Scripts
- Days 15-17: Pipeline integration (trainer, evaluator, orchestrator)
- Days 18-19: Command-line scripts
- Days 20-21: Integration tests

### Week 4: Verification & Documentation
- Days 22-24: Baseline verification and debugging
- Days 25-26: Comprehensive testing and coverage
- Days 27-28: Documentation and cleanup

**Total Duration**: 4 weeks

---

## Next Steps

1. **Review and Approval**: Review this plan with stakeholders
2. **Branch Creation**: Create feature branch for development
3. **Setup**: Initialize project structure
4. **Implementation**: Follow phase-by-phase implementation
5. **Verification**: Run baseline comparison tests
6. **Documentation**: Complete API and usage documentation
7. **Merge**: Merge to main after approval

---

## Notes

- **Methodology Preservation**: The refactoring preserves the exact algorithm logic. No optimizations or improvements to the core methodology are made.
- **Testing First**: Write unit tests alongside implementation to catch errors early.
- **Incremental Verification**: Verify each module against notebook behavior before proceeding.
- **Documentation**: Document all assumptions and deviations (if any) from original.

---

**Document Version**: 1.0
**Date**: 2025-11-18
**Author**: Claude Code Assistant
**Status**: Ready for Implementation
