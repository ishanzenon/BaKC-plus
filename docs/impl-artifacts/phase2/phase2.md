# Phase 2: Core Machine Learning Algorithm

**Timeline**: Week 2 (Days 8-14)
**Status**: Ready for Implementation
**Parent**: BaKC-plus Refactoring Project
**Depends On**: Phase 1 (Core Infrastructure) - COMPLETE

---

## Overview

Phase 2 implements the core machine learning algorithm: OC-SVM ensemble training, conformal prediction calibration, and evaluation metrics. This phase transforms Phase 1's infrastructure into a functional anomaly detection system while preserving the **exact methodology** from the original notebook to ensure reproducible baseline results (Power: 90.29%, FDR: 8.47% on CARDIO).

### Context from Phase 1 Completion

Phase 1 successfully delivered:
- Package structure with `pip install -e .` support
- Configuration management via YAML with validation
- Structured logging across all modules
- Data loading and validation layer
- Comprehensive unit test coverage (>85%)

Phase 1 also documented all critical algorithm details to preserve in Phase 2.

### Current State (Baseline)

**From REFACTORING_PLAN.md - Algorithm Architecture**:
The original notebook implements a 7-step pipeline:
1. Data loading and inlier/outlier separation
2. K-fold cross-validation setup (dynamic: len/len_cal or max 20)
3. Ensemble training (M=5 OC-SVM models per fold)
4. Scoring and calibration (sigmoid transform)
5. Conformal prediction threshold (quantile-based)
6. Test evaluation (L=20 splits with median aggregation)
7. Result aggregation (J=5 repetitions)

**Baseline Performance (Must Preserve)**:
- Power: 90.29% ± 8.61% (90th %ile: 100%, Min: 76.47%)
- FDR: 8.47% ± 3.39% (90th %ile: 12.42%, Min: 4.82%)
- Dataset: CARDIO (1,831 samples, 21 features, 9.61% anomalies)
- Total predictions: 1,003 (229 anomalies, 176 actual)

---

## Goals and Objectives

### Primary Goals

1. **Implement OC-SVM Model Module**
   - Create abstraction for One-Class SVM ensemble members
   - Implement stratified bootstrapping (leave-one-out style)
   - Preserve random state hashing: `rnd = hash((member_idx, fold_idx, random_state)) % 4294967296; rnd = rnd ^ 0x7FFFFFFF`
   - Success Metric: Unit tests pass with 100% deterministic reproducibility

2. **Implement Ensemble Training Module**
   - Create K-fold cross-validation wrapper
   - Coordinate M ensemble members per fold
   - Implement dynamic fold calculation: `len_splits = len(train) // len_cal if len(train) < 20000 else 20`
   - Success Metric: Training produces K × M models with correct split structure

3. **Implement Conformal Prediction Module**
   - Create scoring functions (sigmoid: `1/(1 + np.exp(scores))`)
   - Implement calibration score accumulation (OOB + calibration)
   - Implement quantile threshold calculation: `q_level = ceil((n+1)*(1-α))/n` with 'higher' method
   - Success Metric: Thresholds computed correctly with proper score aggregation

4. **Implement Evaluation Module**
   - Create Power and FDR metric computation
   - Implement fold aggregation (mean) and cross-fold aggregation (median)
   - Create result aggregation and reporting
   - Success Metric: Results match or exceed baseline within ±0.5%

### Success Metrics

- ✅ All acceptance criteria (AC1-AC6) met without exceptions
- ✅ Unit test coverage >85% for Phase 2 modules
- ✅ Baseline verification: Power 90.29% ± 0.5%, FDR 8.47% ± 0.5% on CARDIO
- ✅ All four modules integrated and functioning together
- ✅ Critical algorithm details preserved (see section 6)

---

## Detailed Scope

### Step 2.1: OC-SVM Model Module

**Objective**: Implement One-Class SVM ensemble member abstraction

**Deliverables**:
1. `src/bakc_plus/model/ocsvm.py`
   - `OCSVMMember` class with `fit()` and `decision_function()` methods
   - Random state hashing implementation
   - Integration with bootstrapping module
   
2. `src/bakc_plus/model/bootstrapping.py`
   - `StratifiedBootstrapper` class
   - `perform_bootstrapping()` method with leave-one-out logic
   - Index partitioning and masking
   
3. Unit tests: `tests/unit/test_ocsvm.py`, `tests/unit/test_bootstrapping.py`

**Critical Algorithm Details**:
```python
# Random state hashing
rnd = hash((member_idx, fold_idx, random_state)) % 4294967296
rnd = rnd ^ 0x7FFFFFFF

# Stratified bootstrapping
indices = np.arange(len(X_train))
rnd_state.shuffle(indices)
index_sets = np.array_split(indices, num_members)
leave_out_indices = index_sets[member_idx]
mask = np.ones_like(indices, dtype=bool)
mask[leave_out_indices] = False
X_train_bootstrap = X_train[mask]
```

**Key Decisions**:
- Use Python's built-in `hash()` for deterministic random state
- Preserve integer modulo and XOR operations exactly
- Test with multiple (member_idx, fold_idx) combinations

### Step 2.2: Ensemble Training Module

**Objective**: Implement K-fold cross-validation with ensemble coordination

**Deliverables**:
1. `src/bakc_plus/model/ensemble.py`
   - `EnsembleTrainer` class
   - `train_ensemble()` method for K-fold CV
   - Fold splitting and per-fold model management
   - OOB score accumulation
   
2. Unit tests: `tests/unit/test_ensemble.py`

**Critical Algorithm Details**:
```python
# Dynamic K-fold calculation
len_splits = len(train) // len_cal if len(train) < 20000 else 20

# Per-fold ensemble training
for fold_idx, (train_index, calib_index):
    for member_idx in range(num_models):
        # Fit OC-SVM with bootstrapping
        model, leave_out_indices = fit_ocsvm(...)
        # Accumulate scores
```

**Key Decisions**:
- Use `sklearn.model_selection.KFold` with shuffle=True
- Maintain model list for later use
- Accumulate calibration and OOB scores per fold

### Step 2.3: Conformal Prediction Module

**Objective**: Implement conformal prediction with calibration and thresholding

**Deliverables**:
1. `src/bakc_plus/conformal/scoring.py`
   - `ScoringFunctions` class with sigmoid method (primary)
   - Alternative methods (normalize, signed_ohe) for reference
   
2. `src/bakc_plus/conformal/calibration.py`
   - `CalibrationSetCreator` class
   - `create_calibration_sets()` method
   - Score aggregation (mean per-fold, median cross-fold)
   
3. `src/bakc_plus/conformal/prediction.py`
   - `ConformalPredictor` class
   - `compute_threshold()` with quantile method='higher'
   - `predict()` for binary classification
   
4. Unit tests: `tests/unit/test_scoring.py`, `tests/unit/test_calibration.py`, `tests/unit/test_prediction.py`

**Critical Algorithm Details**:
```python
# Sigmoid scoring (PRIMARY METHOD)
def sigmoid_scored_sample(scores):
    return 1.0 / (1.0 + np.exp(scores))

# Quantile threshold
q_level = np.ceil((n + 1) * (1 - alpha)) / n
qhat = np.quantile(calibration_scores, q_level, method='higher')

# Score aggregation
calibration_scores_fold = np.mean(calibration_scores_i, axis=0)  # M→1 per sample
final_scores = np.median(scores_per_fold, axis=1)  # K→1 per sample

# OOB + Calibration accumulation
calibration_scores = np.append(calibration_scores, fold_calib_scores)
calibration_scores = np.append(calibration_scores, fold_oob_scores)
```

**Key Decisions**:
- Sigmoid is the ONLY active scoring method (preserve exactly)
- Use numpy's quantile with method='higher' (must match)
- Accumulate OOB scores AFTER fold calibration scores
- Mean aggregation per-fold, median cross-fold (not configurable)

### Step 2.4: Evaluation Module

**Objective**: Implement Power/FDR metrics and result aggregation

**Deliverables**:
1. `src/bakc_plus/evaluation/metrics.py`
   - `Metrics` class with Power and FDR methods
   - Proper TP/FP/FN counting
   
2. `src/bakc_plus/evaluation/aggregation.py`
   - `ResultAggregator` class
   - Per-split aggregation (L=20 test splits)
   - Per-repetition aggregation (J=5 repetitions)
   - Reporting: mean, std, 90th percentile
   
3. Unit tests: `tests/unit/test_metrics.py`, `tests/unit/test_aggregation.py`

**Critical Algorithm Details**:
```python
# Power: True Positive Rate for anomalies
Power = TP / (TP + FN)

# FDR: False Discovery Rate (inliers misclassified)
FDR = FP / (TP + FP)

# Per-split aggregation: Median across folds
predicted_scores = np.median(fold_predictions, axis=1)
binary_pred = (predicted_scores > qhat)

# Per-repetition aggregation: Mean and std
final_powers = np.mean(powers_per_split)
final_fdr = np.mean(fdrs_per_split)
```

**Key Decisions**:
- Power and FDR are defined as per-split metrics
- Aggregate with mean (per-repetition) and median (per-fold)
- Report mean, std, and 90th percentile across all runs

---

## Acceptance Criteria

### AC1: OC-SVM Model Module
- [ ] `OCSVMMember` class implemented with `fit()` returning (model, leave_out_indices)
- [ ] Random state hashing produces correct integer values (tested with specific (member_idx, fold_idx, random_state) tuples)
- [ ] `StratifiedBootstrapper.perform_bootstrapping()` correctly partitions indices
- [ ] Bootstrapping masks are correct (sum to n-len(leave_out_indices))
- [ ] Deterministic tests verify identical results with same random state
- [ ] Unit tests: `test_ocsvm.py`, `test_bootstrapping.py` (>90% coverage)

### AC2: Ensemble Training Module
- [ ] `EnsembleTrainer` accepts num_models, num_folds, random_state parameters
- [ ] Dynamic K-fold calculation: `len_splits = len(train) // len_cal if len(train) < 20000 else 20`
- [ ] Returns K × M models (e.g., 3 folds × 5 models = 15 total)
- [ ] Accumulates calibration and OOB scores correctly
- [ ] Per-fold aggregation: mean of M model scores per sample
- [ ] Unit tests: `test_ensemble.py` with >90% coverage

### AC3: Conformal Prediction Module
- [ ] Sigmoid scoring function: `1 / (1 + exp(x))` implemented and tested
- [ ] Quantile threshold computed with: `np.quantile(scores, q_level, method='higher')`
- [ ] OOB scores accumulated AFTER calibration scores (order matters)
- [ ] Score accumulation tested with known inputs/outputs
- [ ] Prediction method returns binary labels correctly
- [ ] Unit tests: >90% coverage for all three modules

### AC4: Evaluation Module
- [ ] Power metric: TP / (TP + FN) computed correctly
- [ ] FDR metric: FP / (TP + FP) computed correctly
- [ ] Handles edge cases (no anomalies, no predictions, etc.)
- [ ] Aggregation: median per-fold, mean per-repetition
- [ ] Result reporting includes mean, std, 90th percentile
- [ ] Unit tests: >90% coverage

### AC5: Testing
- [ ] All unit tests pass: `pytest tests/unit/ -v`
- [ ] Code coverage >85% for Phase 2 modules (target: >90%)
- [ ] Tests run in <5 seconds (no external I/O)
- [ ] Coverage report: `pytest --cov=src/bakc_plus --cov-report=term-missing`
- [ ] Edge cases tested (empty arrays, single sample, etc.)

### AC6: Documentation and Integration
- [ ] All modules have comprehensive docstrings
- [ ] All public functions have type hints (inputs and return types)
- [ ] Phase 2 implementation document complete
- [ ] Step 2.1-2.4 documents with detailed validation steps
- [ ] README.md updated with Phase 2 information
- [ ] Integration with Phase 1 verified (config → data → model pipeline works)

---

## Definition of Done

Phase 2 is considered **DONE** when:

1. ✅ **All Acceptance Criteria Met** - Every AC1-AC6 item is checked and validated

2. ✅ **Unit Tests Pass and Coverage >85%**
   ```bash
   pytest tests/unit/test_ocsvm.py tests/unit/test_bootstrapping.py -v --cov
   pytest tests/unit/test_ensemble.py -v --cov
   pytest tests/unit/test_scoring.py tests/unit/test_calibration.py tests/unit/test_prediction.py -v --cov
   pytest tests/unit/test_metrics.py tests/unit/test_aggregation.py -v --cov
   ```

3. ✅ **Deterministic Reproducibility Verified**
   - Same random_state produces identical results across multiple runs
   - Tested with 3+ independent runs on CARDIO subset
   - No floating-point errors >1e-10

4. ✅ **Phase 1 Integration Complete**
   - Config loading works: `BaKCConfig.from_yaml('configs/cardio.yaml')`
   - Data loading works: `DataLoader.load_dataset('cardio')`
   - Logging integrated: All Phase 2 modules use logger
   - No import errors or missing dependencies

5. ✅ **Critical Algorithm Details Preserved**
   - Random state hashing: `rnd = hash(...) % 4294967296; rnd = rnd ^ 0x7FFFFFFF`
   - Stratified bootstrapping: Leave-one-out partitioning with shuffled indices
   - Sigmoid scoring: `1 / (1 + exp(x))`
   - Quantile threshold: `ceil((n+1)*(1-alpha))/n` with method='higher'
   - Score aggregation: mean per-fold, median cross-fold
   - OOB + calibration: accumulation order preserved

6. ✅ **Issue Log Complete** - All issues found during AC validation are resolved

7. ✅ **Code Quality**
   - PEP 8 compliance (use `black` or `flake8` to verify)
   - Type hints on all public functions
   - Docstrings with Args, Returns, Raises sections
   - No hardcoded values (all via config)

8. ✅ **Git History Clean**
   - Step 2.1, 2.2, 2.3, 2.4 commits with descriptive messages
   - Commit pattern: `Step 2.X: [Task] - validation report`
   - All commits on feature branch ready for PR

9. ✅ **Baseline Verification Ready**
   - Test data prepared (CARDIO dataset loaded)
   - Baseline config: `configs/cardio.yaml` with correct parameters
   - Expected results documented: Power 90.29%, FDR 8.47%
   - Verification script framework created

10. ✅ **Documentation Complete**
    - phase2.md: This document, all sections complete
    - step2.1/step2.1.md through step2.4/step2.4.md: Detailed specs
    - README.md: Updated with Phase 2 modules and usage
    - Inline code comments: Critical algorithm sections documented

---

## Critical Algorithm Details to Preserve

### Random State Management
```python
# CRITICAL: This exact hashing produces deterministic sequences
rnd = hash((member_idx, fold_idx, random_state)) % 4294967296
rnd = rnd ^ 0x7FFFFFFF
# Example: hash((0, 0, 42)) % 4294967296 must produce same value every run
```

### Stratified Bootstrapping (Leave-One-Out Style)
```python
# CRITICAL: Order and masking logic must be preserved exactly
indices = np.arange(len(X_train))
rnd_state.shuffle(indices)  # Shuffle with hashed random state
index_sets = np.array_split(indices, num_members)  # Split into M groups
leave_out_indices = index_sets[member_idx]  # Member m leaves out group m
mask = np.ones_like(indices, dtype=bool)
mask[leave_out_indices] = False  # Mark leave-out indices as False
X_train_bootstrap = X_train[mask]  # Keep only True indices
```

### Sigmoid Scoring Function (PRIMARY)
```python
# CRITICAL: This is the ONLY active method, alternative methods for reference only
def sigmoid_scored_sample(scores):
    return 1.0 / (1.0 + np.exp(scores))
# Input: decision_function scores (can be negative)
# Output: [0, 1] probability-like scores
# Note: exp() can overflow, but original code uses this directly
```

### Quantile Threshold Calculation
```python
# CRITICAL: q_level calculation and method='higher' are essential
n = len(calibration_scores)
alpha = 0.05  # FDR control level
q_level = np.ceil((n + 1) * (1 - alpha)) / n
qhat = np.quantile(calibration_scores, q_level, method='higher')
# method='higher' rounds UP, essential for conformal guarantees
# Example: n=100, alpha=0.05 → q_level=0.95 → 95th percentile using 'higher'
```

### Score Aggregation (Per-Fold and Cross-Fold)
```python
# CRITICAL: Two-level aggregation with different functions
# Level 1: Per-fold aggregation (M models → 1 score per sample)
calibration_scores_fold = np.mean(calibration_scores_per_model, axis=0)  # shape: (n_samples,)

# Level 2: Cross-fold aggregation (K folds → 1 score per sample)
final_scores = np.median(scores_per_fold, axis=0)  # shape: (n_test,)
# Mean per-fold, median cross-fold (NOT configurable)
```

### OOB + Calibration Score Accumulation
```python
# CRITICAL: Order matters - calibration THEN OOB per fold
calibration_scores = np.array([], dtype=np.float32)
for fold_idx in folds:
    for member_idx in members:
        # Score calibration set
        calib_scores = scoring_fn(model.decision_function(X_calib))
        # Score OOB samples
        oob_scores = scoring_fn(model.decision_function(X_oob))
    
    # Aggregate per-fold
    calib_fold_agg = np.mean(all_calib_scores, axis=0)
    oob_fold_agg = np.mean(all_oob_scores, axis=0)  # Also aggregated!
    
    # Append in order: CALIBRATION FIRST, then OOB
    calibration_scores = np.append(calibration_scores, calib_fold_agg)
    calibration_scores = np.append(calibration_scores, oob_fold_agg)
# Result: alternating calib/OOB pairs, used for threshold
```

### Dynamic K-Fold Splitting
```python
# CRITICAL: len_cal is input parameter, splits are dynamic
len_cal = len(test_inliers)  # ~250 for CARDIO
len_train = len(train_inliers)  # ~1400 for CARDIO
if len_train < 20000:
    len_splits = len_train // len_cal  # e.g., 1400 // 250 = 5
else:
    len_splits = 20  # Cap at 20 for large datasets
# Result: K=5 for CARDIO (note: different from baseline documentation which said K=3)
```

---

## Dependencies

### From Phase 1
- `bakc_plus.config` - Configuration management
- `bakc_plus.logger` - Logging infrastructure
- `bakc_plus.data.loader` - Data loading
- `bakc_plus.data.validator` - Data validation

### External Libraries
- numpy>=1.20.0 (array operations)
- pandas>=1.3.0 (data frames)
- scikit-learn>=1.0.0 (OneClassSVM, KFold)
- scipy>=1.7.0 (special functions)
- tqdm>=4.60.0 (progress bars)
- pytest>=6.2.0 (testing)
- pytest-cov>=2.12.0 (coverage)

---

## Issue Log

*Issues discovered during Phase 2 implementation will be tracked here*

| ID | Date | Issue Description | Component | Resolution | Status |
|----|------|-------------------|-----------|------------|--------|
| - | - | - | - | - | - |

---

## Implementation Notes

### Random State Propagation
- Random state must be passed from config through all layers
- Each fold and ensemble member gets a **hashed** random state
- Hashing ensures: (1) deterministic, (2) different for each (member, fold) pair, (3) within int32 range
- Test: Same config + dataset must produce identical results across runs

### Numpy Array Operations
- Use `np.append()` carefully: creates new array each time (O(n) operation)
- For calibration scores: acceptable since n is small (~1000)
- Use `dtype=np.float32` consistently (matches original)
- Test shape assertions: `assert calibration_scores.shape[0] == expected_count`

### Precision and Numerical Stability
- OC-SVM decision function can produce large negative numbers
- Sigmoid of large negative: `1 / (1 + exp(-100))` = very small but positive
- Sigmoid of large positive: `1 / (1 + exp(100))` = very close to 1
- No clipping needed if using float32 (numpy handles gracefully)

### Testing Strategy
- Unit test each module in isolation with synthetic data
- Integration tests verify K-fold structure (correct train/calib splits)
- Determinism tests: run 3 times with same config, assert identical results
- Edge cases: 1 sample, 1 fold, 1 member, all same class

### Debugging Helpers
- Add validation assertions in critical sections:
  ```python
  assert calibration_scores.shape[0] > 0, "No calibration scores"
  assert not np.any(np.isnan(calibration_scores)), "NaN in scores"
  assert 0 <= qhat <= 1, f"Threshold outside [0,1]: {qhat}"
  ```
- Log intermediate shapes: `logger.info(f"Fold {i}: scores shape {scores.shape}")`

---

## Next Steps

1. **Create Step 2.1 Document** - Detailed OC-SVM module specification with unit test plan
2. **Implement Step 2.1** - OCSVMMember and StratifiedBootstrapper classes
3. **Validate Step 2.1** - Unit tests pass, determinism verified
4. **Repeat for Steps 2.2, 2.3, 2.4** - Ensemble, Conformal, Evaluation modules
5. **Integration Testing** - Verify all modules work together
6. **Baseline Verification** - Run on CARDIO, compare with original
7. **Code Review and Documentation** - Final cleanup and documentation
8. **Merge to Main** - After all validations pass

---

**Document Version**: 1.0
**Created**: 2025-11-18
**Status**: Ready for Implementation
**Next Review**: After Step 2.1 completion

