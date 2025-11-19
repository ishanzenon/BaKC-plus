# CRITICAL BUG REPORT: Inverted Prediction Logic

**Date**: 2025-11-18
**Severity**: CRITICAL
**Status**: Root Cause Identified
**Impact**: Complete baseline failure (Power 5% instead of 90%)

---

## Summary

The BaKC-plus implementation has a **critical bug** in the conformal prediction logic that **inverts the predictions**, causing:
- Normal samples to be predicted as anomalies
- Anomalous samples to be predicted as normal

This results in:
- **Power**: 5.01% (should be ~90%) - Only detecting 5% of anomalies
- **FDR**: 99.39% (should be ~8.5%) - 99% false positive rate

---

## Root Cause Analysis

### The Problem

The current prediction logic in `src/bakc_plus/conformal/prediction.py` line 127:

```python
predictions = (conformity_scores <= threshold).astype(int)
# score <= threshold → anomaly (1)
# score > threshold → normal (0)
```

### Why It's Wrong

**Score Flow Analysis**:

1. **OC-SVM Decision Function**:
   - **Inliers (normal)**: Positive scores (e.g., +1.5, +2.0)
   - **Outliers (anomalies)**: Negative scores (e.g., -1.5, -2.0)

2. **Sigmoid Transformation** `sigmoid(x) = 1/(1 + exp(x))`:
   - **Positive scores**: `sigmoid(+2.0) ≈ 0.12` (LOW values)
   - **Negative scores**: `sigmoid(-2.0) ≈ 0.88` (HIGH values)

3. **Current Conformal Logic** `score <= threshold`:
   - **Normal samples**: LOW sigmoid (0.12) <= threshold (0.49) → **Predicted ANOMALY** ❌
   - **Anomaly samples**: HIGH sigmoid (0.88) > threshold (0.49) → **Predicted NORMAL** ❌

**Result**: Predictions are completely inverted!

---

## Evidence

### Empirical Evidence (Single Split Debug):

```
Conformal threshold: 0.493057

Test set scores:
  Min: 0.096, Max: 0.924, Mean: 0.364, Std: 0.174

Predictions:
  Predicted anomalies (1): 301
  Predicted normal (0): 66

Metrics:
  TP: 0    ← Not detecting ANY actual anomalies!
  FP: 301  ← All predictions are false positives!
  FN: 40   ← Missing ALL 40 actual anomalies!
  TN: 26

  Power: 0.0000 (should be ~0.90)
  FDR: 1.0000 (should be ~0.08)
```

### Logical Proof:

**Normal Sample** (actual label = 0):
1. OC-SVM score: +1.8 (inside boundary, inlier)
2. Sigmoid: 1/(1+exp(1.8)) ≈ 0.14
3. Comparison: 0.14 <= 0.49 → TRUE
4. Prediction: ANOMALY (1)
5. **Result: FALSE POSITIVE** ❌

**Anomaly Sample** (actual label = 1):
1. OC-SVM score: -1.8 (outside boundary, outlier)
2. Sigmoid: 1/(1+exp(-1.8)) ≈ 0.86
3. Comparison: 0.86 <= 0.49 → FALSE
4. Prediction: NORMAL (0)
5. **Result: FALSE NEGATIVE** ❌

---

## Solution Options

### Option 1: Reverse the Comparison (RECOMMENDED)

**Change** `src/bakc_plus/conformal/prediction.py` line 127:

```python
# BEFORE (WRONG):
predictions = (conformity_scores <= threshold).astype(int)

# AFTER (CORRECT):
predictions = (conformity_scores >= threshold).astype(int)
```

**Rationale**:
- High sigmoid scores (from negative OC-SVM) indicate anomalies
- score >= threshold → anomaly
- score < threshold → normal

### Option 2: Invert the Sigmoid

**Change** `src/bakc_plus/conformal/scoring.py`:

```python
# BEFORE:
def sigmoid_score(scores: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(scores))

# AFTER:
def sigmoid_score(scores: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-scores))  # Note the minus sign
```

**Rationale**:
- Inverts the sigmoid to preserve OC-SVM score direction
- Positive OC-SVM → high sigmoid → high conformity
- Negative OC-SVM → low sigmoid → low conformity
- Keep current logic: score <= threshold → anomaly

### Option 3: Use 1 - Sigmoid

**Change** `src/bakc_plus/conformal/scoring.py`:

```python
# BEFORE:
def sigmoid_score(scores: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(scores))

# AFTER:
def sigmoid_score(scores: np.ndarray) -> np.ndarray:
    return 1.0 - 1.0 / (1.0 + np.exp(scores))
```

**Rationale**:
- Converts non-conformity to conformity
- High values = high conformity (normal)
- Low values = low conformity (anomaly)
- Keep current logic: score <= threshold → anomaly

---

## Recommendation

**RECOMMENDED**: **Option 1** - Reverse the comparison

**Reasons**:
1. **Minimal change**: Single line modification
2. **Clearest semantics**: High non-conformity scores → anomalies
3. **Easiest to verify**: Simple logic reversal
4. **Fastest to implement**: No formula changes

**Implementation**:
1. Modify `src/bakc_plus/conformal/prediction.py` line 127
2. Update docstring to reflect: `score >= threshold → anomaly`
3. Update tests to match new logic
4. Re-run baseline verification

---

## Impact Assessment

**Before Fix**:
- Power: 5.01% (detecting almost no anomalies)
- FDR: 99.39% (almost all predictions wrong)
- **Status**: System completely non-functional

**After Fix** (Expected):
- Power: ~90% (detecting most anomalies)
- FDR: ~8.5% (low false positive rate)
- **Status**: System should match notebook baseline

---

## Verification Plan

After implementing the fix:

1. **Unit Test**: Verify prediction logic with known scores
2. **Integration Test**: Run single split and check metrics
3. **Full Baseline**: Re-run 5 reps × 20 splits
4. **Validate**: Confirm Power ∈ [88%, 92%], FDR ∈ [6.5%, 10.5%]

---

## Timeline

- **Identified**: 2025-11-18 (Phase 4 baseline execution)
- **Root Cause**: 2025-11-18 (systematic investigation)
- **Fix**: PENDING
- **Verification**: PENDING

---

## Lessons Learned

1. **Direction Matters**: Score transformations can invert semantics
2. **Test Early**: Should have caught this with simple unit tests
3. **Validate Incrementally**: Each transformation step should be validated
4. **Document Semantics**: Clearly document what high/low scores mean

---

**Status**: Root cause identified, fix ready to implement
**Next Step**: Apply Option 1 fix and re-run baseline
