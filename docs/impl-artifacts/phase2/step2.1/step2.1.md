# Step 2.1: OC-SVM Model Module

**Parent**: Phase 2 - Core Algorithm  
**Timeline**: Day 8-9  
**Status**: Ready for Implementation  
**Dependencies**: Step 1 (Core Infrastructure) ✅ Complete  

---

## Overview

Step 2.1 implements the OC-SVM Model Module: the foundational abstraction for One-Class SVM ensemble members with stratified bootstrapping. This step is critical because it preserves the **exact random state hashing and leave-one-out bootstrapping logic** from the original notebook, ensuring 100% deterministic reproducibility.

### Context from Phase 1 Completion

Phase 1 delivered:
- Configuration system with DataConfig and ModelConfig for OC-SVM parameters
- Logging infrastructure for tracking model training
- Data loading and validation (handles inlier/outlier separation)
- Unit test framework with >85% coverage

This step builds on these foundations to implement the core algorithm components that will be used by the ensemble trainer (Step 2.2), conformal predictor (Step 2.3), and evaluator (Step 2.4).

### Critical Preservation Requirements

From **phase2.md** (Critical Algorithm Details):

**Random State Hashing Formula** (EXACT):
```python
rnd = hash((member_idx, fold_idx, random_state)) % 4294967296
rnd = rnd ^ 0x7FFFFFFF
```

**Stratified Bootstrapping** (EXACT):
```python
indices = np.arange(len(X_train))
rnd_state.shuffle(indices)
index_sets = np.array_split(indices, num_members)
leave_out_indices = index_sets[member_idx]
mask = np.ones_like(indices, dtype=bool)
mask[leave_out_indices] = False
X_train_bootstrap = X_train[mask]
```

These exact implementations are **non-negotiable** because they directly affect the random number sequences used in training, which determine model weights and ultimately the baseline performance (Power: 90.29%, FDR: 8.47%).

---

## Goals and Objectives

### Primary Goals

1. **Implement StratifiedBootstrapper Class**
   - Create leave-one-out style stratified bootstrapping
   - Preserve exact random state hashing formula
   - Ensure index partitioning matches notebook logic exactly
   - Success Metric: Deterministic tests verify identical indices across multiple runs

2. **Implement OCSVMMember Class**
   - Wrap sklearn OneClassSVM with configuration integration
   - Implement `fit()` method using bootstrapped samples
   - Implement `decision_function()` wrapper
   - Coordinate with StratifiedBootstrapper for random state management
   - Success Metric: Models train without errors and produce decision scores

3. **Ensure 100% Deterministic Reproducibility**
   - Random state must be hashed identically on every run
   - Bootstrapping indices must be identical for same (member_idx, fold_idx, random_state)
   - No reliance on external randomness (system time, thread order, etc.)
   - Success Metric: Three independent runs produce identical results to machine precision

4. **Achieve >85% Unit Test Coverage**
   - Test random state hashing with multiple (member_idx, fold_idx, random_state) combinations
   - Test bootstrapping logic with various data shapes and num_members
   - Test OC-SVM fitting and decision function
   - Test edge cases (single sample, single member, etc.)
   - Success Metric: `pytest --cov=src/bakc_plus/model --cov-report=term-missing` shows >85%

### Success Metrics

- ✅ Random state hashing formula implemented and tested exactly
- ✅ Bootstrapping indices deterministically computed
- ✅ OCSVMMember integrates sklearn OneClassSVM with config
- ✅ Three independent runs produce identical results
- ✅ Unit tests pass with >85% coverage

---

## Detailed Requirements

### StratifiedBootstrapper Design

From the original notebook analysis and phase2.md, this class must:

1. **Accept parameters:**
   - `random_state: int` - Base random seed (e.g., 42)
   - `num_members: int` - Number of bootstrap samples (M, typically 5)
   - Additional context: `member_idx: int`, `fold_idx: int` for per-call hashing

2. **Implement random state hashing:**
   - Hash tuple `(member_idx, fold_idx, random_state)` using Python's `hash()`
   - Apply modulo: `% 4294967296` (ensure fits in uint32)
   - Apply XOR: `^ 0x7FFFFFFF` (flip sign bit)
   - Create `np.random.RandomState(rnd)` with result

3. **Implement bootstrapping:**
   - Shuffle indices using hashed random state
   - Split shuffled indices into `num_members` groups
   - Return leave-one-out indices for given member_idx
   - Also return training mask for convenience

4. **Return values:**
   - Tuple: `(X_bootstrap, leave_out_indices, rnd_state)` or similar

### OCSVMMember Design

From phase2.md and sklearn documentation, this class must:

1. **Store configuration:**
   - `nu: float` - Expected outlier fraction (from ModelConfig, typically 0.05)
   - `kernel: str` - Kernel type (from ModelConfig, typically 'rbf')
   - `gamma: str or float` - Kernel coefficient (from ModelConfig, typically 'scale')
   - `random_state: int` - For reproducibility

2. **Implement fit() method:**
   - Accept training data: `X_train, y_train` (y unused for OC-SVM)
   - Accept bootstrap context: `member_idx, fold_idx, bootstrapper` or directly: `X_bootstrap`
   - Create sklearn OneClassSVM with config parameters
   - Call `model.fit(X_bootstrap)`
   - Return: `(self, leave_out_indices)` for later reference, or just fitted self

3. **Implement decision_function() method:**
   - Wrapper around sklearn model's `decision_function(X)`
   - Returns float array of shape (n_samples,)
   - Can be positive (inlier-like) or negative (outlier-like)

4. **Store metadata:**
   - `leave_out_indices_` - Indices used for bootstrapping (for diagnostics)
   - `sklearn_model_` - Underlying OneClassSVM instance

### Random State Hashing Specification

**Formula**:
```python
def compute_random_state(member_idx: int, fold_idx: int, base_random_state: int) -> int:
    """
    Compute deterministic random state for given member and fold.
    
    CRITICAL: This formula must be preserved exactly to maintain reproducibility.
    The hash() function is deterministic in Python 3 (unless PYTHONHASHSEED is randomized).
    """
    rnd = hash((member_idx, fold_idx, base_random_state)) % 4294967296
    rnd = rnd ^ 0x7FFFFFFF
    return rnd
```

**Example values** (with PYTHONHASHSEED=0 or without hash randomization):
- `compute_random_state(0, 0, 42)` → deterministic integer
- `compute_random_state(1, 0, 42)` → different from (0,0,42)
- `compute_random_state(0, 0, 42)` (2nd run) → MUST be identical to 1st run

**Testing**: Create test that calls function 100 times and verifies all results are identical

### Bootstrapping Specification

**Algorithm**:
```python
def perform_bootstrapping(X_train, num_members, member_idx, random_state):
    """
    Perform leave-one-out style stratified bootstrapping.
    
    CRITICAL: This logic must preserve the exact partitioning scheme from notebook.
    """
    indices = np.arange(len(X_train))  # [0, 1, 2, ..., n-1]
    rnd_state = np.random.RandomState(random_state)
    rnd_state.shuffle(indices)  # In-place shuffle
    
    # Split into num_members groups
    index_sets = np.array_split(indices, num_members)
    
    # Member member_idx leaves out group member_idx
    leave_out_indices = index_sets[member_idx]
    
    # Create mask: True for samples to KEEP, False for leave-out
    mask = np.ones(len(indices), dtype=bool)
    mask[leave_out_indices] = False
    
    # Bootstrap sample is all EXCEPT leave-out indices
    X_bootstrap = X_train[mask]
    
    return X_bootstrap, leave_out_indices, rnd_state
```

**Key observations**:
- Shuffling order is deterministic only if RandomState is seeded identically
- `np.array_split()` divides into `num_members` parts as evenly as possible
- Masking must be applied to ORIGINAL indices array, not leave_out_indices directly
- Leave-out indices may not be contiguous after shuffle (correct behavior)

---

## Task Breakdown

### Task 2.1.1: Create bootstrapping.py

**Objective**: Implement StratifiedBootstrapper class with exact notebook logic

**File**: `src/bakc_plus/model/bootstrapping.py`

**Implementation**:
1. Create `StratifiedBootstrapper` class
2. Implement `__init__(self, random_state: int, num_members: int)`
3. Implement `_compute_random_state(self, member_idx: int, fold_idx: int) -> int`
4. Implement `perform_bootstrapping(self, X_train: np.ndarray, member_idx: int, fold_idx: int) -> tuple`
5. Return: `(X_bootstrap, leave_out_indices)` from perform_bootstrapping
6. Add docstrings with Args, Returns, Raises sections
7. Add type hints on all functions

**Validation**:
- Instantiation without errors: `bootstrapper = StratifiedBootstrapper(random_state=42, num_members=5)`
- Deterministic output: `perform_bootstrapping()` called twice returns identical arrays
- Correct partitioning: `len(X_bootstrap) + len(leave_out_indices) == len(X_train)`
- Index coverage: `set(leave_out_indices) ∪ set(keep_indices) == set(range(len(X_train)))`

### Task 2.1.2: Create ocsvm.py

**Objective**: Implement OCSVMMember wrapper around sklearn OneClassSVM

**File**: `src/bakc_plus/model/ocsvm.py`

**Implementation**:
1. Create `OCSVMMember` class
2. Implement `__init__(self, nu: float, kernel: str = 'rbf', gamma: str = 'scale', random_state: int = 42)`
3. Implement `fit(self, X_train: np.ndarray, bootstrapper: StratifiedBootstrapper, member_idx: int, fold_idx: int) -> 'OCSVMMember'`
4. Implement `decision_function(self, X: np.ndarray) -> np.ndarray`
5. Store `sklearn_model_` (OneClassSVM instance) and `leave_out_indices_`
6. Add docstrings and type hints

**Critical Details**:
- Initialize sklearn OneClassSVM: `OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)`
- Fit with bootstrapped data: `self.sklearn_model_.fit(X_bootstrap)`
- Store leave_out_indices for later reference (used in ensemble aggregation)
- decision_function delegates to `self.sklearn_model_.decision_function(X)`

**Validation**:
- OCSVMMember instantiation: `member = OCSVMMember(nu=0.05, kernel='rbf', gamma='scale')`
- Fitting works: `member.fit(X_train, bootstrapper, member_idx=0, fold_idx=0)` returns self
- Decision function shape: `scores = member.decision_function(X_test)` has shape `(n_test,)`
- Decision scores are numeric: `np.isfinite(scores).all()`

### Task 2.1.3: Update model/__init__.py Exports

**Objective**: Make bootstrapping and ocsvm modules importable

**File**: `src/bakc_plus/model/__init__.py`

**Implementation**:
1. Add imports: `from .bootstrapping import StratifiedBootstrapper`
2. Add imports: `from .ocsvm import OCSVMMember`
3. Create `__all__` list: `__all__ = ['StratifiedBootstrapper', 'OCSVMMember']`
4. Update docstring with module overview

**Validation**:
- Direct import works: `from bakc_plus.model import StratifiedBootstrapper, OCSVMMember`
- Classes are accessible: `bakc_plus.model.StratifiedBootstrapper`
- Importing bakc_plus.model doesn't raise errors

### Task 2.1.4: Write Comprehensive Unit Tests

**Objective**: Achieve >85% coverage with determinism and edge case testing

**File**: `tests/unit/test_ocsvm.py` and `tests/unit/test_bootstrapping.py`

**Test Cases** (minimum 20 total):

**test_bootstrapping.py**:
1. `test_bootstrapper_initialization()` - Instantiate with valid params
2. `test_random_state_hashing_deterministic()` - Same inputs → same output (3 calls)
3. `test_random_state_hashing_different_members()` - Different member_idx → different rnd
4. `test_random_state_hashing_different_folds()` - Different fold_idx → different rnd
5. `test_bootstrapping_index_partitioning()` - Correct length: keep + leave-out = total
6. `test_bootstrapping_no_duplicates()` - No index appears twice
7. `test_bootstrapping_full_coverage()` - All indices accounted for
8. `test_bootstrapping_different_sizes()` - Test with num_members in [1, 2, 3, 5, 10]
9. `test_bootstrapping_single_sample()` - Edge case: n=1
10. `test_bootstrapping_deterministic()` - Same call → identical arrays (3 runs)

**test_ocsvm.py**:
1. `test_ocsvm_initialization()` - Valid nu, kernel, gamma
2. `test_ocsvm_initialization_defaults()` - nu=0.05, kernel='rbf', gamma='scale'
3. `test_ocsvm_fit_basic()` - Fit on simple data, returns self
4. `test_ocsvm_fit_with_bootstrapper()` - Fit with StratifiedBootstrapper context
5. `test_ocsvm_fit_stores_indices()` - leave_out_indices_ stored correctly
6. `test_ocsvm_decision_function_shape()` - Output shape matches input
7. `test_ocsvm_decision_function_numeric()` - All values are finite floats
8. `test_ocsvm_decision_function_sign()` - Inliers tend positive, outliers negative
9. `test_ocsvm_deterministic_fitting()` - Same data + random_state → identical model
10. `test_ocsvm_different_nu_values()` - nu in [0.01, 0.05, 0.1, 0.2]

**Example test structure**:
```python
def test_random_state_hashing_deterministic():
    """Verify random state hashing is deterministic (CRITICAL)"""
    bootstrapper = StratifiedBootstrapper(random_state=42, num_members=5)
    
    # Call three times
    rnd1 = bootstrapper._compute_random_state(member_idx=0, fold_idx=0)
    rnd2 = bootstrapper._compute_random_state(member_idx=0, fold_idx=0)
    rnd3 = bootstrapper._compute_random_state(member_idx=0, fold_idx=0)
    
    # All must be identical
    assert rnd1 == rnd2, f"Non-deterministic: {rnd1} != {rnd2}"
    assert rnd2 == rnd3, f"Non-deterministic: {rnd2} != {rnd3}"
```

**Coverage target**: `pytest --cov=src/bakc_plus/model --cov-report=term-missing` shows:
- `bootstrapping.py`: >90% coverage
- `ocsvm.py`: >85% coverage
- Overall: >85% coverage

### Task 2.1.5: Create Validation Script

**Objective**: Standalone script to validate all critical properties

**File**: `scripts/validate_step2_1.py`

**Content**:
1. Load CARDIO data (requires Phase 1 data module)
2. Create StratifiedBootstrapper and verify:
   - Random state hashing is deterministic
   - Bootstrapping indices are correct
   - Three runs produce identical results
3. Create OCSVMMember and verify:
   - Fitting works with bootstrapped data
   - Decision function produces valid scores
   - Three runs produce identical decision scores (determinism)
4. Print summary: "Step 2.1 Validation: PASS" or detailed failures

**Execution**:
```bash
python scripts/validate_step2_1.py --dataset cardio --seed 42 --runs 3
```

**Output**:
```
Step 2.1 Validation Report
==========================
Test 1: Random State Hashing (Determinism)
  Run 1: rnd_state(0,0,42) = 1234567890
  Run 2: rnd_state(0,0,42) = 1234567890
  Run 3: rnd_state(0,0,42) = 1234567890
  Status: PASS (all identical)

Test 2: Bootstrapping Partitioning
  Data shape: (1400, 21)
  num_members: 5
  Member 0 leave-out: 280 samples, keep: 1120 samples
  Coverage check: 280 + 1120 = 1400 ✓
  Status: PASS

Test 3: OCSVMMember Fitting
  nu=0.05, kernel=rbf, gamma=scale
  Training samples: 1120
  Fitted model type: OneClassSVM
  Status: PASS

Test 4: Decision Function Determinism
  Run 1: 1000 test samples, mean score = 0.342
  Run 2: 1000 test samples, mean score = 0.342
  Run 3: 1000 test samples, mean score = 0.342
  Max difference: 0.0 (exact match)
  Status: PASS

Overall: ALL TESTS PASS ✓
```

---

## Acceptance Criteria

### AC2.1.1: StratifiedBootstrapper Implementation
- [ ] Class created in `src/bakc_plus/model/bootstrapping.py`
- [ ] `__init__` accepts `random_state` and `num_members`
- [ ] `_compute_random_state()` implements formula: `hash((member_idx, fold_idx, random_state)) % 4294967296 ^ 0x7FFFFFFF`
- [ ] `perform_bootstrapping()` returns `(X_bootstrap, leave_out_indices)`
- [ ] All methods have type hints and docstrings
- [ ] Deterministic output: identical results for identical inputs (tested 3 times)

### AC2.1.2: OCSVMMember Implementation
- [ ] Class created in `src/bakc_plus/model/ocsvm.py`
- [ ] `__init__` accepts `nu, kernel, gamma, random_state`
- [ ] `fit()` method trains sklearn OneClassSVM on bootstrapped data
- [ ] `decision_function()` returns float array of shape `(n_samples,)`
- [ ] Stores `sklearn_model_` and `leave_out_indices_` attributes
- [ ] All methods have type hints and docstrings

### AC2.1.3: Module Exports
- [ ] `src/bakc_plus/model/__init__.py` imports both classes
- [ ] `__all__` list includes both classes
- [ ] Import works: `from bakc_plus.model import StratifiedBootstrapper, OCSVMMember`
- [ ] No import errors when importing parent package

### AC2.1.4: Unit Tests
- [ ] Test file `tests/unit/test_bootstrapping.py` created with ≥10 test cases
- [ ] Test file `tests/unit/test_ocsvm.py` created with ≥10 test cases
- [ ] All tests pass: `pytest tests/unit/test_bootstrapping.py tests/unit/test_ocsvm.py -v`
- [ ] Coverage >85%: `pytest --cov=src/bakc_plus/model --cov-report=term-missing`
- [ ] Tests verify determinism (multiple runs produce identical results)
- [ ] Tests cover edge cases (1 sample, 1 member, etc.)

### AC2.1.5: Validation Script
- [ ] Script created at `scripts/validate_step2_1.py`
- [ ] Loads CARDIO data and validates all critical properties
- [ ] Produces clear pass/fail output
- [ ] Execution completes in <10 seconds
- [ ] Script is executable: `python scripts/validate_step2_1.py --dataset cardio`

---

## Definition of Done

Step 2.1 is considered **DONE** when:

1. ✅ **All Acceptance Criteria Met** - Every item in AC2.1.1 through AC2.1.5 is checked and validated

2. ✅ **Random State Hashing Verified**
   ```bash
   python -c "
   from bakc_plus.model import StratifiedBootstrapper
   b = StratifiedBootstrapper(42, 5)
   r1 = b._compute_random_state(0, 0)
   r2 = b._compute_random_state(0, 0)
   print(f'Deterministic: {r1 == r2}')
   "
   # Output: "Deterministic: True"
   ```

3. ✅ **Bootstrapping Index Correctness**
   ```bash
   python -c "
   import numpy as np
   from bakc_plus.model import StratifiedBootstrapper
   X = np.random.randn(100, 5)
   b = StratifiedBootstrapper(42, 5)
   X_boot, leave_out = b.perform_bootstrapping(X, 0, 0)
   print(f'Coverage: {len(X_boot) + len(leave_out)} == {len(X)}: {len(X_boot) + len(leave_out) == len(X)}')
   "
   # Output: "Coverage: True"
   ```

4. ✅ **OCSVMMember Fitting Works**
   ```bash
   python -c "
   import numpy as np
   from bakc_plus.model import OCSVMMember, StratifiedBootstrapper
   X_train = np.random.randn(100, 5)
   member = OCSVMMember(nu=0.05)
   bootstrapper = StratifiedBootstrapper(42, 5)
   member.fit(X_train, bootstrapper, member_idx=0, fold_idx=0)
   X_test = np.random.randn(20, 5)
   scores = member.decision_function(X_test)
   print(f'Scores shape: {scores.shape}, finite: {np.isfinite(scores).all()}')
   "
   # Output: "Scores shape: (20,), finite: True"
   ```

5. ✅ **Unit Tests Pass with >85% Coverage**
   ```bash
   pytest tests/unit/test_bootstrapping.py tests/unit/test_ocsvm.py -v --cov=src/bakc_plus/model --cov-report=term-missing
   # All tests pass, coverage >85%
   ```

6. ✅ **Determinism Verified (Three Independent Runs)**
   ```bash
   python scripts/validate_step2_1.py --dataset cardio --seed 42 --runs 3
   # Output: "Overall: ALL TESTS PASS ✓" with identical results across 3 runs
   ```

7. ✅ **No Hardcoded Values**
   - No magic numbers in bootstrapping.py or ocsvm.py
   - All parameters passed via config and function arguments
   - No file I/O operations (except in validation script)

8. ✅ **Documentation Complete**
   - All functions have comprehensive docstrings (description, Args, Returns, Raises)
   - Type hints on all public functions
   - Critical algorithm sections documented inline
   - This step2.1.md document reflects actual implementation

9. ✅ **No Issues in Issue Log** - All issues discovered during implementation are resolved

10. ✅ **Git Commit Ready**
    ```bash
    git add src/bakc_plus/model/bootstrapping.py src/bakc_plus/model/ocsvm.py tests/unit/test_bootstrapping.py tests/unit/test_ocsvm.py scripts/validate_step2_1.py src/bakc_plus/model/__init__.py
    git commit -m "Step 2.1: OC-SVM Model Module - StratifiedBootstrapper and OCSVMMember implementation"
    ```

---

## Implementation Details

### Random State Hashing Deep Dive

**Why hashing is critical:**
- Ensures different (member_idx, fold_idx) pairs get different random sequences
- Ensures deterministic behavior (same inputs always produce same hash)
- Hash must be within uint32 range for numpy compatibility
- XOR operation (`^ 0x7FFFFFFF`) ensures proper sign bit handling

**Python hash() function:**
- Deterministic within same Python session (hash randomization disabled)
- In Python 3.3+, hash randomization is ON by default for security
- SOLUTION: Tests must set `PYTHONHASHSEED=0` or disable hash randomization
- In production, may need to explicitly handle this (or use custom hash function)

**Test approach:**
```python
import os
os.environ['PYTHONHASHSEED'] = '0'
# Then import and test
```

### Bootstrapping Deep Dive

**Why leave-one-out style matters:**
- Each of M ensemble members trains on slightly different data
- Member i trains on all data EXCEPT group i
- Provides implicit cross-validation within ensemble
- Ensures we have OOB (out-of-bag) predictions for member i

**Index manipulation details:**
```python
indices = np.arange(len(X_train))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] for n=10
rnd_state.shuffle(indices)  # e.g. [5, 2, 8, 0, 9, 1, 3, 7, 4, 6]
splits = np.array_split(indices, 5)  # m=5
# splits[0] = [5, 2]
# splits[1] = [8, 0]
# splits[2] = [9, 1]
# splits[3] = [3, 7]
# splits[4] = [4, 6]

# Member 0 leaves out group 0: [5, 2]
# Train on: [8, 0, 9, 1, 3, 7, 4, 6]
mask = [F, T, F, T, T, F, T, T, T, F]  # False at positions 0,2,5,9 (original indices 0,2,5,9)
X_bootstrap = X_train[[1,3,4,6,7,8]]  # 8 samples (10 - 2)
```

**Critical point:** The mask indices refer to positions in the ORIGINAL `indices` array, not the leave-out_indices directly.

### OC-SVM Wrapper Rationale

**Why wrap sklearn OneClassSVM:**
- Decouples our code from sklearn internals
- Enables easier mock testing
- Allows future algorithm changes without affecting callers
- Integrates with our bootstrapping system

**Parameter mapping:**
- `nu` from ModelConfig: Expected fraction of training errors
- `kernel` from ModelConfig: Kernel type ('rbf' is typical)
- `gamma` from ModelConfig: Kernel coefficient (sklearn interprets 'scale' specially)
- `random_state`: Internal sklearn seed (may not affect OneClassSVM much, but good practice)

---

## Issue Log

| ID | Date | Issue Description | Resolution | Status |
|----|------|-------------------|------------|--------|
| - | - | - | - | - |

---

## Integration Points

### Phase 1 Integration

**Uses from Phase 1:**
- `bakc_plus.config.ModelConfig` - OC-SVM parameters (nu, kernel, gamma)
- `bakc_plus.config.EnsembleConfig` - Ensemble parameters (num_models, random_state)
- `bakc_plus.logger` - Logging for debugging
- `bakc_plus.data` - Input data (will be used in validation script)

**Example usage**:
```python
from bakc_plus.config import BaKCConfig
from bakc_plus.model import StratifiedBootstrapper, OCSVMMember

config = BaKCConfig.from_yaml('configs/cardio.yaml')
bootstrapper = StratifiedBootstrapper(
    random_state=config.ensemble.random_state,
    num_members=config.ensemble.num_models
)

member = OCSVMMember(
    nu=config.model.nu,
    kernel=config.model.kernel,
    gamma=config.model.gamma,
    random_state=config.ensemble.random_state
)

member.fit(X_train, bootstrapper, member_idx=0, fold_idx=0)
scores = member.decision_function(X_test)
```

### Phase 2 Usage (Future Steps)

**Used by Step 2.2 (Ensemble Training):**
- EnsembleTrainer will instantiate OCSVMMember and StratifiedBootstrapper
- Will call `member.fit()` for each fold and member
- Will collect `decision_function()` scores for calibration

**Used by Step 2.3 (Conformal Prediction):**
- Will receive OOB indices from bootstrapper
- Will score OOB samples for threshold calibration

**Used by Step 2.4 (Evaluation):**
- Will use member's decision scores for metric computation

---

## Dependencies

### Internal (Phase 1)
- `bakc_plus.config` (ModelConfig, EnsembleConfig)
- `bakc_plus.logger` (optional, for debugging)

### External
- `numpy>=1.20.0` - Array operations
- `scikit-learn>=1.0.0` - OneClassSVM
- `pytest>=6.2.0` - Testing (dev only)
- `pytest-cov>=2.12.0` - Coverage (dev only)

---

## Testing Strategy

### Unit Tests (Integration Free)
- Use synthetic numpy arrays (no data loading)
- Test with small data (n=100, d=5)
- Verify shapes, dtypes, and value ranges
- NO sklearn model object introspection (black box approach)

### Determinism Tests (CRITICAL)
- Run same operation 3+ times with identical inputs
- Assert results are identical (use `np.array_equal()`)
- This validates random state hashing

### Edge Cases
- Single sample: `n=1`
- Single member: `num_members=1` (entire dataset is leave-out)
- Large features: `d=100`
- Very small nu: `nu=0.01`

### Integration (with Phase 1)
- Create validation script that loads real CARDIO data
- Test end-to-end: config → data → bootstrapper → ocsvm

---

## Next Steps

After Step 2.1 is DONE:
1. Validate against all AC (checklist above)
2. Run validation script: `python scripts/validate_step2_1.py`
3. Ensure zero issues in issue log
4. Commit changes with clear message
5. Move to Step 2.2 (Ensemble Training Module)

---

**Document Version**: 1.0  
**Created**: 2025-11-18  
**Last Updated**: 2025-11-18  
**Status**: Ready for Implementation  
