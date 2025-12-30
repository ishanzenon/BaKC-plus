# Phase 4: Baseline Verification and Validation

**Status**: In Progress
**Priority**: High (Critical for notebook replication validation)
**Dependencies**: Phase 3 (Pipeline Integration)

---

## Overview

### Purpose

Phase 4 focuses on **validating the complete BaKC-plus implementation** against the original notebook baseline results. This is the critical verification step that confirms:

1. **Exact Methodology Preservation**: All algorithms match the notebook exactly
2. **Result Replication**: Baseline metrics match within acceptable tolerance
3. **Production Readiness**: System performs reliably on real datasets
4. **Documentation Completeness**: All results documented for reproducibility

This phase is **ESSENTIAL** for confirming that the refactored implementation preserves the exact behavior of the original notebook.

### Context

**Completed Phases**:
- Phase 1: Core Infrastructure (Config, Logger, Data) ✓
- Phase 2: Core Algorithms (Model, Conformal, Evaluation) ✓
- Phase 3: Pipeline Integration (Training, Prediction, Workflow) ✓

**What Phase 3 Delivered**:
- Baseline script (`scripts/run_baseline.py`) created and functional
- Smoke tests passed (minimal configuration: 1 rep, 2 splits)
- Full baseline NOT YET RUN due to ~15 minute runtime

**Phase 4 Goal**:
Run full baseline verification with exact notebook configuration and validate results match the expected baseline within ±2% tolerance.

### Success Criteria

**Acceptance Criteria (AC)**:
1. AC1: CARDIO baseline verification complete (5 reps × 20 splits)
2. AC2: Results within tolerance (Power ≈ 90.29% ± 2%, FDR ≈ 8.47% ± 2%)
3. AC3: Results reproducible across multiple independent runs
4. AC4: Baseline results documented with analysis
5. AC5: Performance metrics documented (runtime, memory)
6. AC6: Any discrepancies investigated and explained

**Definition of Done (DoD)**:
1. All acceptance criteria met
2. Full baseline run completed successfully
3. Results match notebook baseline OR discrepancies explained
4. Reproducibility verified (3+ independent runs)
5. Documentation complete (results, analysis, recommendations)
6. Performance acceptable (<30 minutes for full baseline)
7. Memory usage acceptable (<5GB peak)
8. Issue log complete
9. Production validation complete

---

## Phase 4 Steps

### Step 4.1: Full Baseline Execution

**Objective**: Run complete baseline verification on CARDIO dataset

**Deliverables**:
1. Execute full baseline with exact notebook configuration
2. Capture all output (logs, results, performance metrics)
3. Save results to structured format (JSON)

**Configuration** (exact match to notebook):
- Dataset: CARDIO (1831 samples, 21 features, 176 anomalies = 9.6%)
- num_repetitions (J) = 5
- num_test_splits (L) = 20
- num_models (M) = 5
- len_cal = 50
- alpha = 0.1
- random_state = 42
- nu = 0.05
- kernel = 'rbf'

**Expected Output**:
```json
{
  "dataset": "cardio",
  "results": {
    "power_mean": ~0.9029,
    "power_std": <value>,
    "power_p90": <value>,
    "fdr_mean": ~0.0847,
    "fdr_std": <value>,
    "fdr_p90": <value>
  },
  "baseline": {
    "target_power": 0.9029,
    "target_fdr": 0.0847,
    "tolerance": 0.02,
    "passed": true/false
  }
}
```

**Execution**:
```bash
python scripts/run_baseline.py \
    --dataset cardio \
    --data-dir data/input \
    --reps 5 \
    --splits 20 \
    --len-cal 50 \
    --num-models 5 \
    --alpha 0.1 \
    --seed 42 \
    --output results/cardio_baseline.json
```

---

### Step 4.2: Result Validation

**Objective**: Validate results against notebook baseline

**Deliverables**:
1. Compare actual vs expected results
2. Calculate deviations (absolute and relative)
3. Determine pass/fail for each metric
4. Document any discrepancies

**Validation Checks**:

1. **Power Validation**:
   - Target: 90.29%
   - Tolerance: ±2% (absolute)
   - Pass range: [88.29%, 92.29%]
   - Check: Is `power_mean` in pass range?

2. **FDR Validation**:
   - Target: 8.47%
   - Tolerance: ±2% (absolute)
   - Pass range: [6.47%, 10.47%]
   - Check: Is `fdr_mean` in pass range?

3. **Statistical Validation**:
   - Check standard deviations reasonable
   - Check p90 values reasonable
   - Check per-repetition distributions

**Expected Outcome**:
- ✅ Power in [88.29%, 92.29%]
- ✅ FDR in [6.47%, 10.47%]
- ✅ Standard deviations < 10% (absolute)
- ✅ No anomalous per-repetition values

---

### Step 4.3: Reproducibility Verification

**Objective**: Verify results are reproducible across runs

**Deliverables**:
1. Run baseline 3 independent times with same configuration
2. Compare results across runs
3. Calculate variance between runs
4. Document reproducibility metrics

**Test Protocol**:
```bash
# Run 1
python scripts/run_baseline.py --seed 42 --output results/cardio_run1.json

# Run 2 (same seed)
python scripts/run_baseline.py --seed 42 --output results/cardio_run2.json

# Run 3 (same seed)
python scripts/run_baseline.py --seed 42 --output results/cardio_run3.json
```

**Validation**:
- Compare `power_mean` across 3 runs
- Compare `fdr_mean` across 3 runs
- Maximum difference should be < 1e-10 (machine precision)
- Verify: Run1 == Run2 == Run3 (bitwise identical)

**Expected Outcome**:
- ✅ All runs produce identical results
- ✅ Determinism verified
- ✅ Reproducibility confirmed

---

### Step 4.4: Performance Analysis

**Objective**: Document performance characteristics

**Deliverables**:
1. Runtime measurements
2. Memory usage profiling
3. Computational efficiency analysis
4. Scalability assessment

**Metrics to Collect**:

1. **Runtime**:
   - Total experiment time
   - Per-repetition time
   - Per-split time
   - Training time vs prediction time

2. **Memory**:
   - Peak memory usage
   - Average memory usage
   - Memory per component (training, prediction, evaluation)

3. **Computational**:
   - CPU utilization
   - Number of models trained (K × M × J × L)
   - Number of predictions made

**Performance Targets**:
- Total runtime: < 30 minutes (acceptable for research code)
- Peak memory: < 5GB (fits on standard laptops)
- CPU utilization: > 80% (efficient use of resources)

**Measurement Tools**:
```python
import time
import psutil
import tracemalloc

# Runtime
start = time.time()
# ... run baseline ...
runtime = time.time() - start

# Memory
tracemalloc.start()
# ... run baseline ...
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

# CPU
process = psutil.Process()
cpu_percent = process.cpu_percent(interval=1.0)
```

---

### Step 4.5: Discrepancy Investigation (if needed)

**Objective**: Investigate and explain any discrepancies from baseline

**Triggers**:
- Power outside [88.29%, 92.29%]
- FDR outside [6.47%, 10.47%]
- Results not reproducible
- Unexpected errors or warnings

**Investigation Protocol**:

1. **Data Verification**:
   - Verify CARDIO dataset matches notebook
   - Check number of samples (1831)
   - Check number of features (21)
   - Check anomaly count (176)
   - Check anomaly rate (9.6%)

2. **Configuration Verification**:
   - Verify all hyperparameters match
   - Check random seed handling
   - Verify K-fold calculation
   - Check calibration set size

3. **Algorithm Verification**:
   - Review bootstrapping implementation
   - Check score aggregation order
   - Verify sigmoid transformation
   - Confirm conformal threshold calculation

4. **Numerical Verification**:
   - Check for numerical precision issues
   - Verify random number generation
   - Check for platform differences

**Documentation**:
- Document root cause of discrepancy
- Explain why discrepancy occurred
- Assess impact on results
- Recommend mitigation if needed

---

### Step 4.6: Documentation and Reporting

**Objective**: Create comprehensive documentation of baseline verification

**Deliverables**:
1. `BASELINE-RESULTS.md`: Detailed results documentation
2. `PERFORMANCE-REPORT.md`: Performance analysis
3. `VALIDATION-SUMMARY.md`: Executive summary
4. Updated `FINAL-STATUS.md`: Phase 4 completion

**BASELINE-RESULTS.md Contents**:
- Configuration used
- Results summary (Power, FDR, std, p90)
- Comparison to notebook baseline
- Pass/fail determination
- Reproducibility verification
- Statistical analysis
- Per-repetition breakdown
- Visualizations (if applicable)

**PERFORMANCE-REPORT.md Contents**:
- Runtime analysis
- Memory profiling
- Computational efficiency
- Scalability assessment
- Bottleneck identification
- Optimization recommendations

**VALIDATION-SUMMARY.md Contents**:
- Executive summary (1 page)
- Key findings
- Pass/fail status
- Recommendations
- Next steps

---

## Acceptance Criteria

### AC1: CARDIO Baseline Verification Complete ✓

**Criteria**:
- [ ] Full baseline executed (5 reps × 20 splits × 5 models)
- [ ] All repetitions completed successfully
- [ ] No errors or warnings during execution
- [ ] Results saved to JSON file
- [ ] Logs captured and available

**Validation**:
- Run baseline script with full configuration
- Verify output file exists: `results/cardio_baseline.json`
- Verify JSON structure is correct
- Verify all expected fields present
- Check logs for errors/warnings

---

### AC2: Results Within Tolerance ✓

**Criteria**:
- [ ] Power in range [88.29%, 92.29%]
- [ ] FDR in range [6.47%, 10.47%]
- [ ] Standard deviations reasonable (< 10%)
- [ ] P90 values reasonable
- [ ] Per-repetition values consistent

**Validation**:
- Extract `power_mean` from results
- Check: 0.8829 <= power_mean <= 0.9229
- Extract `fdr_mean` from results
- Check: 0.0647 <= fdr_mean <= 0.1047
- Review statistical distributions

**Acceptance**:
- PASS if both Power AND FDR in range
- CONDITIONAL PASS if one metric slightly outside but explainable
- FAIL if both outside range or large discrepancy

---

### AC3: Results Reproducible ✓

**Criteria**:
- [ ] 3 independent runs with same seed
- [ ] All runs produce identical results
- [ ] Difference < 1e-10 (machine precision)
- [ ] Determinism verified
- [ ] Documentation complete

**Validation**:
```python
import json
import numpy as np

# Load 3 runs
run1 = json.load(open('results/cardio_run1.json'))
run2 = json.load(open('results/cardio_run2.json'))
run3 = json.load(open('results/cardio_run3.json'))

# Compare
power1 = run1['results']['power_mean']
power2 = run2['results']['power_mean']
power3 = run3['results']['power_mean']

assert abs(power1 - power2) < 1e-10
assert abs(power2 - power3) < 1e-10
assert abs(power1 - power3) < 1e-10

# Same for FDR
```

---

### AC4: Baseline Results Documented ✓

**Criteria**:
- [ ] BASELINE-RESULTS.md created
- [ ] All results documented
- [ ] Comparison to notebook included
- [ ] Statistical analysis included
- [ ] Visualizations included (if applicable)

**Validation**:
- File exists: `docs/impl-artifacts/phase4/BASELINE-RESULTS.md`
- File contains all required sections
- Results are accurate and complete
- Analysis is thorough

---

### AC5: Performance Metrics Documented ✓

**Criteria**:
- [ ] Runtime measured and documented
- [ ] Memory usage measured and documented
- [ ] CPU utilization documented
- [ ] Performance analysis complete
- [ ] PERFORMANCE-REPORT.md created

**Validation**:
- File exists: `docs/impl-artifacts/phase4/PERFORMANCE-REPORT.md`
- All metrics documented
- Analysis is thorough
- Recommendations included

---

### AC6: Discrepancies Investigated ✓

**Criteria**:
- [ ] If results outside tolerance: investigation complete
- [ ] Root cause identified
- [ ] Explanation documented
- [ ] Impact assessed
- [ ] Mitigation recommended

**Validation**:
- If discrepancy exists: investigation document present
- Root cause clearly explained
- Impact quantified
- Recommendations actionable

**Note**: This criterion is only applicable if discrepancies are found.

---

## Definition of Done

Phase 4 is considered **DONE** when:

1. ✅ **All Acceptance Criteria Met** - Every AC1-AC6 item checked

2. ✅ **Full Baseline Run Completed**
   - 5 repetitions × 20 splits executed
   - All runs successful
   - Results saved and validated

3. ✅ **Results Match Baseline OR Explained**
   - Power within ±2% of 90.29% OR discrepancy explained
   - FDR within ±2% of 8.47% OR discrepancy explained
   - Statistical distributions reasonable

4. ✅ **Reproducibility Verified**
   - 3 independent runs with same seed
   - All produce identical results
   - Determinism confirmed

5. ✅ **Documentation Complete**
   - BASELINE-RESULTS.md comprehensive
   - PERFORMANCE-REPORT.md thorough
   - VALIDATION-SUMMARY.md clear
   - FINAL-STATUS.md updated

6. ✅ **Performance Acceptable**
   - Runtime < 30 minutes
   - Memory < 5GB peak
   - No performance bottlenecks identified

7. ✅ **Memory Usage Acceptable**
   - Peak < 5GB
   - Average reasonable
   - No memory leaks

8. ✅ **Issue Log Complete**
   - All issues documented
   - All issues resolved or explained
   - Zero critical issues

9. ✅ **Production Validation Complete**
   - System validated on real dataset
   - Results publishable
   - Ready for research use

---

## Risk Management

**Potential Risks**:

1. **Results Don't Match Baseline**
   - Risk: High impact (questions implementation correctness)
   - Mitigation: Thorough investigation, step-by-step verification
   - Contingency: Document discrepancy, identify root cause

2. **Runtime Too Long**
   - Risk: Medium impact (usability concern)
   - Mitigation: Profile code, identify bottlenecks
   - Contingency: Optimize critical paths, add parallelization

3. **Memory Overflow**
   - Risk: Medium impact (limits dataset size)
   - Mitigation: Profile memory usage, optimize data structures
   - Contingency: Implement batch processing, reduce memory footprint

4. **Non-Reproducible Results**
   - Risk: High impact (questions determinism)
   - Mitigation: Verify random seed handling, check for race conditions
   - Contingency: Identify non-deterministic components, fix or document

---

## Expected Timeline

**Step-by-Step Estimates**:

1. **Step 4.1**: Full Baseline Execution (15-20 minutes)
   - Script execution: 15 minutes
   - Output verification: 2 minutes
   - Log review: 3 minutes

2. **Step 4.2**: Result Validation (15 minutes)
   - Load and parse results: 2 minutes
   - Compare to baseline: 5 minutes
   - Statistical analysis: 8 minutes

3. **Step 4.3**: Reproducibility Verification (45-60 minutes)
   - 3 runs × 15 minutes: 45 minutes
   - Comparison and validation: 10 minutes
   - Documentation: 5 minutes

4. **Step 4.4**: Performance Analysis (20 minutes)
   - Collect metrics: 5 minutes
   - Analysis: 10 minutes
   - Documentation: 5 minutes

5. **Step 4.5**: Discrepancy Investigation (0-60 minutes)
   - If no discrepancy: 0 minutes
   - If discrepancy found: 30-60 minutes investigation

6. **Step 4.6**: Documentation and Reporting (30 minutes)
   - Write BASELINE-RESULTS.md: 15 minutes
   - Write PERFORMANCE-REPORT.md: 10 minutes
   - Write VALIDATION-SUMMARY.md: 5 minutes

**Total Estimated Time**: 2-3 hours (plus ~1 hour compute time)

---

## Baseline Targets (from Notebook)

**Dataset**: CARDIO
- Samples: 1831
- Features: 21
- Anomalies: 176 (9.6%)

**Configuration**:
- num_repetitions (J) = 5
- num_test_splits (L) = 20
- num_models (M) = 5
- len_cal = 50
- alpha = 0.1
- random_state = 42

**Expected Results** (from notebook):
- **Power**: 90.29% ± 2%
- **FDR**: 8.47% ± 2%

**Acceptance Ranges**:
- Power: [88.29%, 92.29%]
- FDR: [6.47%, 10.47%]

---

## Success Metrics

**Primary Metrics**:
- Baseline pass/fail: PASS
- Power deviation: < 2% (absolute)
- FDR deviation: < 2% (absolute)
- Reproducibility: 100% (identical results)

**Secondary Metrics**:
- Runtime: < 30 minutes
- Memory: < 5GB peak
- CPU utilization: > 80%

**Quality Metrics**:
- Documentation: Comprehensive
- Analysis: Thorough
- Recommendations: Actionable

---

## Deliverables Summary

### Results Files
1. `results/cardio_baseline.json` - Full baseline results
2. `results/cardio_run1.json` - Reproducibility run 1
3. `results/cardio_run2.json` - Reproducibility run 2
4. `results/cardio_run3.json` - Reproducibility run 3

### Documentation
5. `docs/impl-artifacts/phase4/phase4.md` - This specification
6. `docs/impl-artifacts/phase4/BASELINE-RESULTS.md` - Detailed results
7. `docs/impl-artifacts/phase4/PERFORMANCE-REPORT.md` - Performance analysis
8. `docs/impl-artifacts/phase4/VALIDATION-SUMMARY.md` - Executive summary
9. `docs/impl-artifacts/phase4/FINAL-STATUS.md` - Phase completion report

---

## Conclusion

Phase 4 is the **critical validation phase** that confirms the BaKC-plus implementation correctly replicates the original notebook methodology. Upon completion, we will have:

1. **Verified Results**: Baseline metrics confirmed within tolerance
2. **Proven Reproducibility**: Deterministic results verified
3. **Documented Performance**: Runtime and memory characteristics known
4. **Research-Ready System**: Validated and ready for use

**Next Steps After Phase 4**:
- If baseline passes: System ready for additional datasets, publications
- If discrepancies found: Investigate, fix, re-validate
- Future work: Additional datasets, performance optimization, productionization

---

*Document version: 1.0*
*Created: 2025-11-18*
*Status: Ready for execution*
