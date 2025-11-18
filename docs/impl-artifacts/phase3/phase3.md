# Phase 3: Pipeline Integration

**Status**: In Progress
**Priority**: High (blocking baseline verification)
**Dependencies**: Phase 1 (Infrastructure), Phase 2 (Core Algorithms)

---

## Overview

### Purpose

Phase 3 integrates all components from Phases 1 and 2 into complete end-to-end pipelines for:
1. **Training**: Load data → Train ensemble → Calibrate conformal predictor
2. **Prediction**: Load test data → Score with models → Apply conformal prediction
3. **Evaluation**: Run full workflow → Compute metrics → Aggregate results

This phase is CRITICAL for baseline verification (target: Power 90.29%, FDR 8.47% on CARDIO dataset).

### Context

**Phase 1 Outputs** (Infrastructure):
- Config system (BaKCConfig, ModelConfig, etc.)
- Logging system (structured logging)
- Data module (DataLoader, DataSplitter, DataValidator)

**Phase 2 Outputs** (Core Algorithms):
- Model module (StratifiedBootstrapper, OCSVMMember, EnsembleTrainer)
- Conformal module (sigmoid_score, ConformalPredictor)
- Evaluation module (compute_metrics, MetricsCalculator)

**Phase 3 Goal**:
Integrate all components into cohesive pipelines that can:
- Train models on any dataset
- Make predictions with calibrated thresholds
- Evaluate performance with Power/FDR metrics
- Match baseline results exactly

### Success Criteria

**Acceptance Criteria (AC)**:
1. AC1: Training pipeline implemented and tested
2. AC2: Prediction pipeline implemented and tested
3. AC3: End-to-end integration verified
4. AC4: Baseline verification on CARDIO dataset
5. AC5: All integration tests pass
6. AC6: Documentation complete

**Definition of Done (DoD)**:
1. All acceptance criteria met
2. Integration tests pass (>90% coverage)
3. Baseline results match notebook (within tolerance)
4. Full workflow documented
5. Issue log clean
6. Ready for production use

---

## Phase 3 Steps

### Step 3.1: Training Pipeline

**Objective**: Implement end-to-end training pipeline

**Deliverables**:
1. `src/bakc_plus/pipeline/training.py`
   - `TrainingPipeline` class
   - `train()` method: data → ensemble → calibration
   - State management (save/load trained models)

2. `tests/unit/test_training_pipeline.py`
   - Unit tests for training pipeline
   - Mock data tests
   - Integration with Phase 1/2 modules

**Algorithm**:
```python
def train(config, X_train, y_train):
    # 1. Split data (train/calibration)
    # 2. Train ensemble (K folds, M members per fold)
    # 3. Accumulate calibration + OOB scores
    # 4. Apply sigmoid scoring
    # 5. Calibrate conformal predictor
    # 6. Return: (models, predictor, threshold)
```

**Critical Details**:
- Use exact data splitting from notebook
- Preserve score accumulation order (calibration + OOB)
- Deterministic: same seed → same threshold

### Step 3.2: Prediction Pipeline

**Objective**: Implement end-to-end prediction pipeline

**Deliverables**:
1. `src/bakc_plus/pipeline/prediction.py`
   - `PredictionPipeline` class
   - `predict()` method: data → scores → binary predictions
   - Support for batch prediction

2. `tests/unit/test_prediction_pipeline.py`
   - Unit tests for prediction pipeline
   - Score aggregation tests
   - Conformal prediction tests

**Algorithm**:
```python
def predict(models, predictor, X_test):
    # 1. Score X_test with each model
    # 2. Aggregate scores (mean per-fold, median cross-fold)
    # 3. Apply sigmoid transformation
    # 4. Apply conformal threshold
    # 5. Return: binary predictions (1=anomaly, 0=normal)
```

**Critical Details**:
- Score aggregation: mean within fold, median across folds
- Sigmoid applied AFTER aggregation
- Threshold comparison: score <= threshold → anomaly

### Step 3.3: End-to-End Integration

**Objective**: Integrate training + prediction + evaluation

**Deliverables**:
1. `src/bakc_plus/pipeline/workflow.py`
   - `BaKCWorkflow` class
   - `run_full_experiment()` method
   - Multi-repetition support (J repetitions, L test splits)

2. `tests/integration/test_full_workflow.py`
   - Full workflow tests
   - CARDIO baseline test
   - Determinism verification

3. `scripts/run_baseline.py`
   - Command-line script for baseline verification
   - Outputs: Power, FDR, saved results

**Algorithm**:
```python
def run_full_experiment(config, dataset_name):
    # For each repetition j in 1..J:
    #   Load and split data (L test splits)
    #   For each test split l in 1..L:
    #     Train on train_l
    #     Predict on test_l
    #     Compute metrics (Power, FDR)
    #   Aggregate metrics across L splits
    # Aggregate metrics across J repetitions
    # Report: mean, std, 90th percentile
```

**Critical Details**:
- Outer loop: J repetitions (default 5)
- Inner loop: L test splits (default 20)
- Result aggregation: median per-rep, mean across reps
- Match notebook's exact loop structure

---

## Acceptance Criteria

### AC1: Training Pipeline ✓

**Criteria**:
- [ ] `TrainingPipeline` class implemented
- [ ] Accepts: config, X_train, y_train (optional)
- [ ] Trains: K×M models via EnsembleTrainer
- [ ] Accumulates: calibration + OOB scores
- [ ] Calibrates: ConformalPredictor with accumulated scores
- [ ] Returns: trained models, calibrated predictor
- [ ] Deterministic: same seed → same threshold
- [ ] Unit tests: >90% coverage

**Validation**:
- Train on synthetic data (1000 samples, 10 features)
- Verify: K folds created, M models per fold
- Verify: threshold computed from calibration scores
- Verify: 3 runs with same seed produce identical results

### AC2: Prediction Pipeline ✓

**Criteria**:
- [ ] `PredictionPipeline` class implemented
- [ ] Accepts: models, predictor, X_test
- [ ] Scores: X_test with all K×M models
- [ ] Aggregates: mean per-fold (M→1), median across-folds (K→1)
- [ ] Applies: sigmoid transformation to aggregated scores
- [ ] Predicts: binary labels using conformal threshold
- [ ] Returns: predictions (1=anomaly, 0=normal)
- [ ] Unit tests: >90% coverage

**Validation**:
- Predict on synthetic test data (100 samples)
- Verify: score aggregation correct (mean then median)
- Verify: sigmoid applied correctly
- Verify: predictions binary (0 or 1)

### AC3: End-to-End Integration ✓

**Criteria**:
- [ ] `BaKCWorkflow` class implemented
- [ ] Accepts: config, dataset_name
- [ ] Runs: J repetitions × L test splits
- [ ] Loads: data via DataLoader
- [ ] Splits: data via DataSplitter
- [ ] Trains: via TrainingPipeline
- [ ] Predicts: via PredictionPipeline
- [ ] Evaluates: via MetricsCalculator
- [ ] Aggregates: results across splits and repetitions
- [ ] Returns: final metrics (Power, FDR) with mean/std/p90
- [ ] Integration tests: >85% coverage

**Validation**:
- Run on CARDIO dataset (1 repetition, 2 test splits)
- Verify: all components integrate correctly
- Verify: no errors or exceptions
- Verify: results in expected ranges

### AC4: Baseline Verification ✓

**Criteria**:
- [ ] Baseline script: `scripts/run_baseline.py`
- [ ] Runs: CARDIO dataset with exact notebook config
- [ ] Reports: Power and FDR
- [ ] Target: Power ≈ 90.29% (within ±2%)
- [ ] Target: FDR ≈ 8.47% (within ±2%)
- [ ] Outputs: results to CSV/JSON
- [ ] Logs: detailed execution info

**Validation**:
- Run: `python scripts/run_baseline.py --dataset cardio --config configs/cardio.yaml`
- Compare: results to notebook baseline
- Accept: if Power in [88.29%, 92.29%] and FDR in [6.47%, 10.47%]

### AC5: Integration Tests ✓

**Criteria**:
- [ ] Integration test suite: `tests/integration/test_full_workflow.py`
- [ ] Tests: full workflow end-to-end
- [ ] Tests: CARDIO dataset (small subset)
- [ ] Tests: determinism verification
- [ ] Tests: error handling and edge cases
- [ ] Coverage: >85% for pipeline module
- [ ] All tests pass: 100% pass rate

**Validation**:
- Run: `pytest tests/integration/ -v`
- Verify: all tests pass
- Verify: no errors or warnings

### AC6: Documentation ✓

**Criteria**:
- [ ] Phase 3 specification: `phase3.md` (this document)
- [ ] Step 3.1 docs: training pipeline
- [ ] Step 3.2 docs: prediction pipeline
- [ ] Step 3.3 docs: workflow integration
- [ ] Usage guide: How to run baseline
- [ ] API docs: All pipeline classes documented
- [ ] README updated: Phase 3 completion

**Validation**:
- All docs exist and are comprehensive
- Code examples provided
- User can follow docs to run baseline

---

## Definition of Done

Phase 3 is considered **DONE** when:

1. ✅ **All Acceptance Criteria Met** - Every AC1-AC6 item checked

2. ✅ **Integration Tests Pass and Coverage >85%**
   ```bash
   pytest tests/unit/test_training_pipeline.py -v --cov
   pytest tests/unit/test_prediction_pipeline.py -v --cov
   pytest tests/integration/test_full_workflow.py -v --cov
   ```

3. ✅ **Baseline Verification Complete**
   - CARDIO results within tolerance: Power ≈ 90.29%, FDR ≈ 8.47%
   - Results reproducible across runs
   - Saved to: `results/cardio_baseline.json`

4. ✅ **Full Integration Verified**
   - Phase 1 → Phase 2 → Phase 3 integration seamless
   - Config → Data → Model → Conformal → Evaluation pipeline works
   - No import errors, no missing dependencies

5. ✅ **Deterministic Reproducibility**
   - Same config + same seed → identical results
   - Verified with 3+ independent runs
   - Max difference <1e-10

6. ✅ **Issue Log Complete**
   - All issues found during validation resolved
   - Zero critical issues remaining

7. ✅ **Code Quality**
   - PEP 8 compliance
   - Type hints on all public functions
   - Comprehensive docstrings
   - No hardcoded values

8. ✅ **Git History Clean**
   - Step 3.1, 3.2, 3.3 commits with descriptive messages
   - All commits on feature branch
   - Ready for PR

9. ✅ **Production Ready**
   - Command-line interface works
   - Error messages clear and actionable
   - Logging provides debugging info
   - Memory efficient for large datasets

---

## Critical Implementation Details

### Score Aggregation (MUST PRESERVE EXACTLY)

From notebook:
```python
# Step 1: Score test data with all K×M models
for fold_idx in range(K):
    fold_scores = []  # M scores per sample
    for member_idx in range(M):
        model = models[fold_idx][member_idx]
        scores = model.decision_function(X_test)
        fold_scores.append(scores)

    # Aggregate M models → 1 score per sample (mean)
    fold_scores_agg = np.mean(fold_scores, axis=0)
    all_fold_scores.append(fold_scores_agg)

# Step 2: Aggregate K folds → 1 score per sample (median)
final_scores = np.median(all_fold_scores, axis=0)

# Step 3: Apply sigmoid
conformity_scores = 1.0 / (1.0 + np.exp(final_scores))

# Step 4: Predict
predictions = (conformity_scores <= threshold).astype(int)
```

**Why Critical**: Score aggregation order affects final predictions. Must match notebook exactly.

### Data Splitting (MUST PRESERVE EXACTLY)

From notebook:
```python
# Outer loop: J repetitions with different random seeds
for rep_idx in range(J):
    seed = base_seed + rep_idx

    # Inner loop: L test splits
    for split_idx in range(L):
        # Split: 80% train, 20% test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed + split_idx
        )

        # Train, predict, evaluate
        ...
```

**Why Critical**: Data splitting determines which samples are in train vs test, affecting all downstream results.

### Result Aggregation (MUST PRESERVE EXACTLY)

From notebook:
```python
# Aggregate across L test splits (per repetition)
power_per_rep = np.median(power_per_split)
fdr_per_rep = np.median(fdr_per_split)

# Aggregate across J repetitions
final_power_mean = np.mean(power_per_rep_all)
final_power_std = np.std(power_per_rep_all)
final_power_p90 = np.percentile(power_per_rep_all, 90)

# Same for FDR
...
```

**Why Critical**: Aggregation method (median vs mean) affects reported metrics.

---

## Implementation Plan

### Step-by-Step Execution

1. **Step 3.1: Training Pipeline** (Estimated: 3-4 hours)
   - Create `src/bakc_plus/pipeline/training.py`
   - Implement `TrainingPipeline` class
   - Write unit tests: `tests/unit/test_training_pipeline.py`
   - Validate: determinism, integration with Phase 2

2. **Step 3.2: Prediction Pipeline** (Estimated: 2-3 hours)
   - Create `src/bakc_plus/pipeline/prediction.py`
   - Implement `PredictionPipeline` class
   - Write unit tests: `tests/unit/test_prediction_pipeline.py`
   - Validate: score aggregation, conformal prediction

3. **Step 3.3: End-to-End Integration** (Estimated: 4-5 hours)
   - Create `src/bakc_plus/pipeline/workflow.py`
   - Implement `BaKCWorkflow` class
   - Write integration tests: `tests/integration/test_full_workflow.py`
   - Create baseline script: `scripts/run_baseline.py`
   - Run baseline on CARDIO dataset
   - Validate: results match notebook

4. **Phase 3 Validation** (Estimated: 1-2 hours)
   - Run all tests
   - Verify coverage >85%
   - Verify baseline results
   - Create FINAL-STATUS.md
   - Commit and push

**Total Estimated Time**: 10-14 hours

---

## Baseline Verification Target

**Dataset**: CARDIO (1831 samples, 21 features, 176 anomalies, 9.6% anomaly rate)

**Configuration**:
- num_models (M) = 5
- num_folds (K) = auto-computed (len_train // len_cal)
- len_cal = 50
- alpha = 0.1
- num_repetitions (J) = 5
- num_test_splits (L) = 20
- random_state = 42

**Expected Results** (from notebook):
- **Power**: 90.29% (±2%)
- **FDR**: 8.47% (±2%)

**Acceptance**:
- Power ∈ [88.29%, 92.29%]
- FDR ∈ [6.47%, 10.47%]

---

## Deliverables Summary

### Source Code
1. `src/bakc_plus/pipeline/training.py` (TrainingPipeline)
2. `src/bakc_plus/pipeline/prediction.py` (PredictionPipeline)
3. `src/bakc_plus/pipeline/workflow.py` (BaKCWorkflow)
4. `src/bakc_plus/pipeline/__init__.py` (exports)

### Tests
5. `tests/unit/test_training_pipeline.py` (unit tests)
6. `tests/unit/test_prediction_pipeline.py` (unit tests)
7. `tests/integration/test_full_workflow.py` (integration tests)

### Scripts
8. `scripts/run_baseline.py` (baseline verification)

### Documentation
9. `docs/impl-artifacts/phase3/phase3.md` (this document)
10. `docs/impl-artifacts/phase3/FINAL-STATUS.md` (completion report)

---

## Success Metrics

**Code Metrics**:
- Source: ~600-800 lines (pipeline module)
- Tests: ~800-1000 lines
- Total: ~1,400-1,800 lines

**Quality Metrics**:
- Test coverage: >85% (target: >90%)
- Pass rate: 100%
- Baseline accuracy: within ±2% of target

**Performance Metrics**:
- Training time: <5 minutes (CARDIO, single repetition)
- Prediction time: <10 seconds (CARDIO test set)
- Memory usage: <2GB (CARDIO dataset)

---

## Risk Management

**Potential Risks**:

1. **Baseline Mismatch**: Results don't match notebook
   - Mitigation: Rigorous formula preservation in Phase 2
   - Contingency: Debug with notebook side-by-side comparison

2. **Integration Errors**: Components don't integrate smoothly
   - Mitigation: Unit test all interfaces
   - Contingency: Add adapter layers if needed

3. **Performance Issues**: Pipeline too slow for large datasets
   - Mitigation: Use efficient numpy operations
   - Contingency: Add multiprocessing if needed

4. **Memory Issues**: Large models exceed memory
   - Mitigation: Streaming data processing where possible
   - Contingency: Batch processing for large datasets

---

## Conclusion

Phase 3 integrates all BaKC-plus components into production-ready pipelines. Upon completion, the system will be ready for:
- Baseline verification on CARDIO dataset
- Application to other datasets
- Production deployment
- Further research and development

**Next Steps After Phase 3**:
- Phase 4: Additional datasets (if needed)
- Phase 5: Performance optimization (if needed)
- Phase 6: Documentation and deployment

---

*Document version: 1.0*
*Created: 2025-11-18*
*Status: Ready for implementation*
