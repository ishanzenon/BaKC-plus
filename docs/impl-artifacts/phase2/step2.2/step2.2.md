# Step 2.2: Ensemble Training Module

**Parent**: Phase 2: Core Algorithm Implementation
**Status**: In Progress
**Priority**: High (blocking Step 2.3)

---

## Overview

### Purpose

Step 2.2 implements the **Ensemble Training Module**, which coordinates K-fold cross-validation with ensemble member training. This module integrates the OC-SVM members from Step 2.1 into a coordinated training framework that:

1. Dynamically calculates the number of CV folds based on dataset size
2. Splits training data using stratified K-fold cross-validation
3. Trains M ensemble members per fold with bootstrapped data
4. Accumulates calibration and out-of-bag (OOB) scores per fold
5. Maintains model references for later prediction

This step is CRITICAL for preserving the exact methodology from the original notebook, particularly the dynamic fold calculation, score accumulation logic, and cross-validation structure.

### Context

**Dependencies**:
- Step 2.1: OC-SVM Model Module (StratifiedBootstrapper, OCSVMMember)
- Step 1.2: Configuration System (ModelConfig)
- Step 1.3: Logging System (structured logging)
- Step 1.4: Data Module (DataSplit)

**Used By**:
- Step 2.3: Conformal Prediction Module (uses accumulated calibration scores)
- Pipeline integration (full training workflow)

**Critical Algorithm Preservation**:
The ensemble training logic contains EXACT formulas and patterns from the notebook:
- Dynamic fold calculation: `len_splits = len(train) // len_cal if len(train) < 20000 else 20`
- KFold with shuffle=True, random_state determinism
- Per-fold ensemble training with bootstrapping
- OOB score accumulation AFTER calibration scores per fold
- Score aggregation: mean per-fold (M→1), median cross-fold (K→1)

### Success Criteria

**Acceptance Criteria (AC)**:
1. AC2.2.1: EnsembleTrainer class implemented with correct initialization
2. AC2.2.2: Dynamic fold calculation matches notebook formula
3. AC2.2.3: train_ensemble() method implements K-fold CV correctly
4. AC2.2.4: Per-fold ensemble training with M members
5. AC2.2.5: Score accumulation (calibration + OOB) per fold
6. AC2.2.6: Module exports and convenience functions
7. AC2.2.7: Unit tests with >85% coverage

**Definition of Done (DoD)**:
1. All acceptance criteria met
2. EnsembleTrainer initialization validated
3. Dynamic fold calculation verified against notebook
4. K-fold split determinism verified (same seed → same splits)
5. Ensemble member count correct (K * M models total)
6. Calibration score shapes correct (n_cal per fold)
7. OOB score shapes correct (~n_train/M per fold)
8. Unit tests pass with >85% coverage
9. No inappropriate hardcoded values (all configurable)
10. Documentation complete with examples
11. Issue log clean (zero issues)
12. All deliverables exist and validated

---

## Tasks

### Task 2.2.1: Create ensemble.py with EnsembleTrainer class

**Objective**: Implement core EnsembleTrainer class with initialization and configuration

**Implementation Details**:

**File**: `src/bakc_plus/model/ensemble.py`

**Class**: `EnsembleTrainer`

**Attributes**:
```python
class EnsembleTrainer:
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize ensemble trainer

        Args:
            config: ModelConfig instance (or None for defaults)
        """
        self.config = config if config is not None else ModelConfig()
        self.logger = get_logger(__name__)

        # Model tracking
        self.models: List[List[OCSVMMember]] = []  # K folds x M members
        self.n_folds = 0
        self.n_members = self.config.num_models

        # Score accumulation
        self.calibration_scores_per_fold: List[np.ndarray] = []
        self.oob_scores_per_fold: List[np.ndarray] = []
```

**Key Points**:
- Initialize from ModelConfig (num_models, nu, kernel, etc.)
- Track all models in 2D structure: models[fold_idx][member_idx]
- Separate score lists for calibration and OOB per fold
- Use structured logging from Phase 1

**Validation**:
- EnsembleTrainer can be created with/without config
- Attributes initialized correctly
- Logging works

### Task 2.2.2: Implement dynamic fold calculation

**Objective**: Implement EXACT formula from notebook for calculating number of folds

**Critical Formula** (MUST PRESERVE EXACTLY):
```python
def _calculate_num_folds(
    self,
    n_train: int,
    len_cal: int
) -> int:
    """
    Calculate number of CV folds dynamically

    CRITICAL: This formula MUST match the notebook exactly!

    Formula:
        if len(train) < 20000:
            len_splits = len(train) // len_cal
        else:
            len_splits = 20

    Args:
        n_train: Number of training samples
        len_cal: Calibration set size per fold

    Returns:
        Number of CV folds (integer)
    """
    if n_train < 20000:
        len_splits = n_train // len_cal
    else:
        len_splits = 20

    return len_splits
```

**Key Points**:
- Threshold is 20000 (hardcoded, from notebook)
- For small datasets: fold count = n_train // len_cal
- For large datasets: cap at 20 folds
- This MUST match the notebook to ensure reproducible results

**Example**:
- n_train=1000, len_cal=50 → len_splits = 1000 // 50 = 20 folds
- n_train=500, len_cal=25 → len_splits = 500 // 25 = 20 folds
- n_train=25000, len_cal=100 → len_splits = 20 folds (capped)

**Validation**:
- Test with various dataset sizes
- Verify formula matches notebook
- Check edge cases (n_train < len_cal, etc.)

### Task 2.2.3: Implement train_ensemble() with K-fold CV

**Objective**: Implement main training method with K-fold cross-validation

**Method Signature**:
```python
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
    """
    # Input validation
    if X_train is None or len(X_train) == 0:
        raise ValueError("X_train is empty or None")

    if len_cal <= 0 or len_cal >= len(X_train):
        raise ValueError(f"len_cal must be in (0, {len(X_train)}), got {len_cal}")

    # Calculate number of folds
    self.n_folds = self._calculate_num_folds(len(X_train), len_cal)

    self.logger.info(
        f"Starting ensemble training: {self.n_folds} folds, "
        f"{self.n_members} members per fold, "
        f"len_cal={len_cal}, random_state={random_state}"
    )

    # Create K-fold splitter
    from sklearn.model_selection import KFold
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
    for fold_idx, (train_indices, calib_indices) in enumerate(kfold.split(X_train)):
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
            member = OCSVMMember(config=self.config)
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
        fold_oob_scores_concatenated = np.concatenate(fold_oob_scores)

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
```

**Key Points**:
- Use sklearn.model_selection.KFold with shuffle=True
- Random state passed to both KFold and OCSVMMember.fit()
- Score aggregation: mean per-fold (across M members)
- OOB scores concatenated (not aggregated)
- Return both scores and models for later use

**Validation**:
- Test with various dataset sizes
- Verify fold counts match dynamic formula
- Check score shapes: calibration = K * len_cal, OOB varies
- Verify determinism (same seed → same splits, same scores)
- Check model count = K * M

### Task 2.2.4: Update model/__init__.py exports

**Objective**: Export EnsembleTrainer and convenience function

**File**: `src/bakc_plus/model/__init__.py`

**Update**:
```python
from .bootstrapping import StratifiedBootstrapper, stratified_bootstrap
from .ocsvm import OCSVMMember, create_ocsvm_member
from .ensemble import EnsembleTrainer, train_ensemble

__all__ = [
    # Bootstrapping
    'StratifiedBootstrapper',
    'stratified_bootstrap',
    # OC-SVM
    'OCSVMMember',
    'create_ocsvm_member',
    # Ensemble
    'EnsembleTrainer',
    'train_ensemble',
]
```

**Convenience Function**:
```python
# In ensemble.py
def train_ensemble(
    X_train: np.ndarray,
    len_cal: int,
    config: Optional[ModelConfig] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[List[OCSVMMember]]]:
    """
    Convenience function for training ensemble

    Args:
        X_train: Training data
        len_cal: Calibration set size per fold
        config: ModelConfig (optional)
        random_state: Random seed

    Returns:
        (calibration_scores, oob_scores, models)
    """
    trainer = EnsembleTrainer(config=config)
    return trainer.train_ensemble(X_train, len_cal, random_state)
```

**Validation**:
- Imports work correctly
- Convenience function provides same results as class method
- __all__ list complete

### Task 2.2.5: Write comprehensive unit tests

**Objective**: Create test_ensemble.py with >85% coverage

**File**: `tests/unit/test_ensemble.py`

**Test Categories** (minimum 40 tests):

1. **Initialization Tests** (~5 tests)
   - Test init with config
   - Test init without config (defaults)
   - Test init with explicit parameters
   - Test attribute initialization
   - Test logger setup

2. **Dynamic Fold Calculation Tests** (~10 tests)
   - Test formula for small datasets (< 20000)
   - Test formula for large datasets (>= 20000)
   - Test edge cases (n_train = 20000, n_train = 19999, n_train = 20001)
   - Test various len_cal values
   - Parametrize over dataset sizes: [500, 1000, 5000, 10000, 20000, 25000, 50000]

3. **train_ensemble() Tests** (~15 tests)
   - Test basic training (small dataset, 2 folds, 2 members)
   - Test determinism (same seed → same results)
   - Test different seeds produce different results
   - Test score shapes (calibration = K * len_cal)
   - Test OOB score shapes (varies, but >0)
   - Test model count (K * M models)
   - Test with various num_models: [1, 2, 5, 10]
   - Test with various len_cal: [25, 50, 100]
   - Test invalid inputs (empty X_train, len_cal=0, len_cal >= n_train)
   - Test score ranges (reasonable decision function values)
   - Test model list structure (K folds x M members)

4. **Integration Tests** (~5 tests)
   - Test full workflow with real-ish data (100 samples, 5 features)
   - Test with different dataset sizes
   - Test score accumulation correctness
   - Test convenience function equivalence
   - Test reproducibility across multiple runs

5. **Edge Cases** (~5 tests)
   - Test minimum dataset size
   - Test single fold
   - Test single member
   - Test maximum members (10)
   - Test with high-dimensional data

**Example Test**:
```python
def test_train_ensemble_basic(self):
    """Test basic ensemble training workflow"""
    trainer = EnsembleTrainer(config=ModelConfig(num_models=2))

    X_train = np.random.randn(200, 5)
    len_cal = 20

    calib_scores, oob_scores, models = trainer.train_ensemble(
        X_train, len_cal, random_state=42
    )

    # Check score shapes
    expected_n_folds = 200 // 20  # = 10 folds
    assert len(calib_scores) == expected_n_folds * len_cal
    assert len(oob_scores) > 0

    # Check model count
    assert len(models) == expected_n_folds
    assert all(len(fold_models) == 2 for fold_models in models)

    # Check all models are fitted
    assert all(
        member.is_fitted()
        for fold_models in models
        for member in fold_models
    )
```

**Validation**:
- At least 40 tests total
- >85% code coverage for ensemble.py
- All tests pass
- Parametrized tests for broad coverage

### Task 2.2.6: Create validation script

**Objective**: Create validate_step2_2.py for automated validation

**File**: `scripts/validate_step2_2.py`

**Validation Checks**:

1. **AC2.2.1: EnsembleTrainer Class** (8 checks)
   - Class exists
   - Initialize with config
   - Initialize without config
   - Attributes initialized (models, n_folds, n_members, score lists)
   - Logger setup
   - Config propagation
   - Type hints present

2. **AC2.2.2: Dynamic Fold Calculation** (6 checks)
   - Method exists
   - Formula correct for small datasets (n < 20000)
   - Formula correct for large datasets (n >= 20000)
   - Edge case: n = 20000
   - Edge case: n = 19999
   - Edge case: n = 20001

3. **AC2.2.3: train_ensemble() Method** (10 checks)
   - Method exists
   - Returns correct types (tuple of 3 elements)
   - Calibration scores shape correct
   - OOB scores shape correct (> 0)
   - Model count correct (K * M)
   - Determinism verified (3 runs)
   - KFold integration works
   - Invalid input handling
   - Score aggregation correct (mean per-fold)
   - All models fitted

4. **AC2.2.4: Per-Fold Ensemble Training** (6 checks)
   - M members trained per fold
   - Each member uses bootstrapping
   - Calibration scores per member
   - OOB scores per member
   - Score aggregation per fold
   - Model storage per fold

5. **AC2.2.5: Score Accumulation** (6 checks)
   - Calibration scores concatenated across folds
   - OOB scores concatenated across folds
   - Score shapes match expectations
   - Score ranges reasonable
   - No NaN or inf values
   - Score ordering preserved

6. **AC2.2.6: Module Exports** (4 checks)
   - EnsembleTrainer exported
   - train_ensemble() convenience function exported
   - __all__ list includes new exports
   - Convenience function works

7. **AC2.2.7: Unit Tests** (5 checks)
   - test_ensemble.py exists
   - At least 40 tests
   - All tests pass
   - Coverage > 85%
   - Test categories present

8. **DoD Validation** (12 checks)
   - All AC met
   - Dynamic fold calculation verified
   - Determinism verified
   - Model count verified
   - Calibration score shapes verified
   - OOB score shapes verified
   - Unit tests pass with >85% coverage
   - No hardcoded values (all configurable)
   - Documentation complete
   - Issue log clean
   - All deliverables exist
   - Integration with Step 2.1 works

**Output Format**: Same as validate_step2_1.py with pass/fail summary

---

## Acceptance Criteria

### AC2.2.1: EnsembleTrainer Class Implementation ✓

**Criteria**:
- EnsembleTrainer class exists in src/bakc_plus/model/ensemble.py
- Initializes with ModelConfig or defaults
- Attributes: config, logger, models, n_folds, n_members, score lists
- Logger uses structured logging from Phase 1
- Type hints throughout
- Comprehensive docstrings

**Validation**:
- Create EnsembleTrainer with/without config
- Check all attributes initialized correctly
- Verify logger works (debug, info messages)
- Check type hints with mypy

**Success Metrics**:
- Class instantiation works
- All attributes present and correct types
- Logging produces expected output
- No type errors

### AC2.2.2: Dynamic Fold Calculation ✓

**Criteria**:
- _calculate_num_folds() method exists
- Implements EXACT formula from notebook
- Handles small datasets (n < 20000): len_splits = n // len_cal
- Handles large datasets (n >= 20000): len_splits = 20
- Edge cases handled correctly (n = 20000, etc.)

**Validation**:
- Test with n_train = [500, 1000, 5000, 10000, 19999, 20000, 20001, 50000]
- Verify formula outputs match expectations
- Compare with notebook outputs (if available)

**Success Metrics**:
- Formula exactly matches notebook
- Edge cases produce correct results
- Deterministic (same inputs → same output)

### AC2.2.3: train_ensemble() Method ✓

**Criteria**:
- train_ensemble() method exists
- Implements K-fold CV with sklearn.model_selection.KFold
- Uses shuffle=True and random_state for reproducibility
- Returns (calibration_scores, oob_scores, models) tuple
- Handles invalid inputs gracefully

**Validation**:
- Test basic training workflow
- Verify determinism (same seed → same splits, same scores)
- Check return types and shapes
- Test error handling

**Success Metrics**:
- Method works end-to-end
- Deterministic across runs
- Correct return types
- Proper error handling

### AC2.2.4: Per-Fold Ensemble Training ✓

**Criteria**:
- For each fold, trains M ensemble members
- Each member uses OCSVMMember.fit() with bootstrapping
- Per-member calibration scores collected
- Per-member OOB scores collected
- Models stored in fold structure

**Validation**:
- Verify M members trained per fold
- Check bootstrapping integration (leave_out_indices returned)
- Verify score collection per member
- Check model storage structure

**Success Metrics**:
- Exactly M members per fold
- All members use bootstrapping
- Score shapes correct per member
- Model structure is List[List[OCSVMMember]]

### AC2.2.5: Score Accumulation ✓

**Criteria**:
- Calibration scores: mean aggregation per-fold (M→1), concatenation across folds
- OOB scores: concatenation per-fold, concatenation across folds
- Total calibration scores = K * len_cal
- Total OOB scores varies (~K * n_train / M)
- No NaN or inf values

**Validation**:
- Check calibration score shape = K * len_cal
- Check OOB score shape > 0
- Verify aggregation logic (mean per-fold)
- Check for invalid values

**Success Metrics**:
- Score shapes match expectations
- Aggregation correct (mean for calib, concat for OOB)
- All scores finite and reasonable
- Scores concatenated in correct order

### AC2.2.6: Module Exports ✓

**Criteria**:
- EnsembleTrainer exported in model/__init__.py
- train_ensemble() convenience function exported
- __all__ list updated
- Convenience function equivalent to class method

**Validation**:
- Import EnsembleTrainer, train_ensemble from bakc_plus.model
- Check __all__ list
- Test convenience function

**Success Metrics**:
- All exports work
- Convenience function produces same results as class method
- __all__ list complete

### AC2.2.7: Unit Tests ✓

**Criteria**:
- tests/unit/test_ensemble.py exists
- At least 40 test cases
- Coverage >85% for ensemble.py
- All tests pass
- Test categories: init, fold calculation, train_ensemble, integration, edge cases

**Validation**:
- Run pytest on test_ensemble.py
- Check coverage report
- Verify test categories present

**Success Metrics**:
- >=40 tests total
- 100% pass rate
- >85% coverage
- All categories covered

---

## Definition of Done

### Code Quality
1. ✓ All acceptance criteria met (AC2.2.1 - AC2.2.7)
2. ✓ EnsembleTrainer initialization validated
3. ✓ Dynamic fold calculation verified against notebook
4. ✓ K-fold split determinism verified (same seed → same splits)

### Functionality
5. ✓ Ensemble member count correct (K * M models total)
6. ✓ Calibration score shapes correct (K * len_cal per dataset)
7. ✓ OOB score shapes correct (~K * n_train/M, varies per dataset)

### Testing
8. ✓ Unit tests pass with >85% coverage
9. ✓ No inappropriate hardcoded values (all configurable via ModelConfig)

### Documentation
10. ✓ Documentation complete with examples (docstrings, this spec)

### Process
11. ✓ Issue log clean (zero issues encountered or all resolved)
12. ✓ All deliverables exist and validated

---

## Critical Algorithm Preservation

### Dynamic Fold Calculation (MUST PRESERVE EXACTLY)

```python
# From notebook (EXACT FORMULA):
if len(train) < 20000:
    len_splits = len(train) // len_cal
else:
    len_splits = 20

# Our implementation MUST match this exactly!
```

**Why Critical**: This formula determines the number of CV folds, which affects:
- Total number of models trained (K * M)
- Calibration score count (K * len_cal)
- OOB score count (~K * n_train / M)
- Overall computational cost
- Result reproducibility

**Validation**: Test with multiple dataset sizes, compare outputs with notebook

### K-Fold Cross-Validation (MUST PRESERVE EXACTLY)

```python
# From notebook:
from sklearn.model_selection import KFold
kfold = KFold(n_splits=len_splits, shuffle=True, random_state=random_state)

# Our implementation MUST use:
# - KFold (not StratifiedKFold or other variants)
# - shuffle=True
# - random_state for reproducibility
```

**Why Critical**: The splitting strategy determines which samples go into calibration vs. training for each fold, affecting all downstream results.

**Validation**: Verify determinism (same seed → same splits across runs)

### Score Aggregation (MUST PRESERVE EXACTLY)

```python
# Per-fold aggregation (M members → 1 score per sample):
calibration_scores_fold = np.mean(calibration_scores_per_member, axis=0)

# Cross-fold accumulation:
all_calibration_scores = np.concatenate([fold1_scores, fold2_scores, ...])

# OOB accumulation (no aggregation, just concatenation):
all_oob_scores = np.concatenate([fold1_oob, fold2_oob, ...])
```

**Why Critical**: Score aggregation affects the final calibration set used for conformal prediction threshold calculation.

**Validation**:
- Check calibration shape = K * len_cal (mean reduces M→1 per fold)
- Check OOB shape varies (~K * n_train / M, concatenated)
- Verify mean aggregation (not median, max, or other)

### Model Storage

```python
# Models stored as List[List[OCSVMMember]]
# Structure: models[fold_idx][member_idx]
# Total count: K folds * M members

# This structure allows later prediction:
# - Per-fold prediction (average M members)
# - Cross-fold prediction (median K folds)
```

**Why Critical**: Model structure must match notebook to enable correct prediction logic in later steps.

**Validation**: Check models structure, count, all fitted

---

## Implementation Guidance

### Step-by-Step Implementation Order

1. **Create ensemble.py skeleton** (Task 2.2.1)
   - Class definition
   - __init__ method
   - Attributes
   - Logger setup

2. **Implement dynamic fold calculation** (Task 2.2.2)
   - _calculate_num_folds() method
   - Test with various dataset sizes
   - Verify formula matches notebook

3. **Implement train_ensemble() scaffold** (Task 2.2.3)
   - Method signature
   - Input validation
   - KFold setup
   - Return structure

4. **Implement per-fold training loop** (Task 2.2.3 continued)
   - Iterate over folds
   - Train M members per fold
   - Collect scores
   - Store models

5. **Implement score aggregation** (Task 2.2.3 continued)
   - Mean aggregation per-fold for calibration
   - Concatenation for OOB
   - Final concatenation across folds

6. **Update exports** (Task 2.2.4)
   - Add to __init__.py
   - Create convenience function

7. **Write tests** (Task 2.2.5)
   - Start with basic tests (init, fold calc)
   - Add train_ensemble tests
   - Add edge cases
   - Verify coverage >85%

8. **Create validation script** (Task 2.2.6)
   - Implement all AC checks
   - Implement DoD checks
   - Run and verify zero issues

### Testing Strategy

**Unit Tests** (test_ensemble.py):
- Focus on individual methods (init, fold calc, train)
- Use small datasets for speed (50-200 samples)
- Parametrize over key variables (num_models, len_cal)
- Mock external dependencies if needed (though OCSVMMember is fast)

**Integration Tests** (within test_ensemble.py):
- Test full workflow end-to-end
- Use realistic dataset sizes (500-1000 samples)
- Verify determinism across multiple runs
- Check integration with Step 2.1 (OCSVMMember, StratifiedBootstrapper)

**Validation Script** (validate_step2_2.py):
- Automate all AC/DoD checks
- Use deterministic data for reproducibility
- Generate detailed pass/fail report
- Target: 6/6 AC pass, 12/12 DoD pass = zero issues

### Common Pitfalls to Avoid

1. **Wrong fold calculation formula**: Must use `//` integer division, not `/`
2. **Wrong KFold configuration**: Must use shuffle=True, NOT StratifiedKFold
3. **Wrong score aggregation**: Mean for calibration, concat for OOB (not median/max)
4. **Missing random_state**: Must pass to both KFold and OCSVMMember.fit()
5. **Wrong model structure**: Must be List[List[OCSVMMember]], not flattened list
6. **Inconsistent score shapes**: Calibration must be exactly K * len_cal
7. **Missing OOB scores**: Each member should produce ~n_train/M OOB scores
8. **Not all models fitted**: Check is_fitted() for all members

### Code Quality Checklist

- [ ] Type hints on all public methods
- [ ] Comprehensive docstrings with examples
- [ ] Structured logging at appropriate levels (debug, info, warning)
- [ ] Input validation with clear error messages
- [ ] No hardcoded magic numbers (use config or constants)
- [ ] Preserve exact formulas from notebook (comments with "CRITICAL", "MUST PRESERVE")
- [ ] Determinism verified (same seed → same results)
- [ ] Memory efficient (don't duplicate large arrays unnecessarily)
- [ ] Integration with Step 2.1 seamless (use OCSVMMember, StratifiedBootstrapper)

---

## Deliverables

### Source Code
1. `src/bakc_plus/model/ensemble.py` (~300-400 lines)
   - EnsembleTrainer class
   - _calculate_num_folds() method
   - train_ensemble() method
   - train_ensemble() convenience function

2. `src/bakc_plus/model/__init__.py` (updated)
   - Add EnsembleTrainer, train_ensemble to exports
   - Update __all__ list

### Tests
3. `tests/unit/test_ensemble.py` (~800-1000 lines)
   - 40+ tests covering all functionality
   - >85% coverage for ensemble.py
   - All tests passing

### Validation
4. `scripts/validate_step2_2.py` (~600-800 lines)
   - All AC checks (7 sections)
   - All DoD checks (12 items)
   - Detailed pass/fail reporting

### Documentation
5. `docs/impl-artifacts/phase2/step2.2/step2.2.md` (this document)
6. `docs/impl-artifacts/phase2/step2.2/FINAL-STATUS.md` (created after validation passes)

---

## Dependencies

**Required from Previous Steps**:
- Step 2.1: OCSVMMember, StratifiedBootstrapper
- Step 1.2: ModelConfig
- Step 1.3: get_logger
- Step 1.4: (for later integration, not directly used in Step 2.2)

**External Libraries**:
- numpy: arrays, aggregation
- sklearn.model_selection.KFold: CV splitting
- typing: type hints
- logging: via Phase 1 logger

**Python Version**: 3.8+

---

## Estimated Metrics

**Code**:
- Source: ~350 lines (ensemble.py)
- Tests: ~900 lines (test_ensemble.py)
- Validation: ~700 lines (validate_step2_2.py)
- Total: ~1,950 lines

**Tests**:
- Test count: 40-50 tests
- Coverage target: >85%
- Expected runtime: <5 seconds

**Effort**:
- Implementation: ~3-4 hours
- Testing: ~2-3 hours
- Validation: ~1-2 hours
- Total: ~6-9 hours

---

## Success Criteria Summary

**Step 2.2 is complete when**:
1. ✓ All 7 acceptance criteria pass
2. ✓ All 12 definition of done items met
3. ✓ Validation script shows 6/6 AC pass
4. ✓ Zero issues in issue log
5. ✓ FINAL-STATUS.md created with pass summary
6. ✓ Code committed and pushed

**Ready to proceed to Step 2.3** when Step 2.2 has zero issues.

---

*Document version: 1.0*
*Created: 2025-11-18*
*Status: Ready for implementation*
