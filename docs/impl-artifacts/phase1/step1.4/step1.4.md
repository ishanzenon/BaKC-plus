# Step 1.4: Data Module Implementation

**Parent**: Phase 1 - Core Infrastructure
**Timeline**: Days 7-9
**Status**: Ready for Implementation
**Dependencies**: Step 1.1 (Project Setup) ✅, Step 1.2 (Configuration) ✅, Step 1.3 (Logging) ✅

---

## Overview

Step 1.4 implements a modular data loading and preprocessing system that replaces the hardcoded data handling in the notebook. The Data Module provides CSV loading, data validation, and splitting functionality while preserving the exact train/calibration/test split methodology from the original notebook.

### Context from Existing Analysis

From **NOTEBOOK_ANALYSIS.md** (Section 2: Data Flow Analysis):
> **Dataset**: CARDIO (Cardiovascular anomalies)
> - **Total samples**: 1,831
> - **Features**: 21 (V1-V21) - normalized and standardized
> - **Target**: Binary (0=inlier, 1=anomaly)
> - **Class imbalance**: 90.39% inliers, 9.61% anomalies

From **NOTEBOOK_ANALYSIS.md** (Section 6.3: Error Handling & Validation):
> **ADD**: Data validation function that checks:
> - DataFrame is not None
> - DataFrame is not empty
> - Correct number of features
> - Target column exists
> - No NaN values
> - Target values are binary (0, 1)

From **ocsvm_x_cv_x_bagging.py** (Lines 7-25):
```python
# Data loading (hardcoded paths - to be refactored)
dataset_folder_path = '/kaggle/input/adbench/input/gamma'
inliers_df = ...
outliers_df = ...

# Data splitting logic (to be preserved)
len_train = len(inliers_df) // 2
train = inliers_df[:len_train]
test = inliers_df[len_train:]
```

### Current Data Handling Issues

1. **Hardcoded Paths**: `/kaggle/input/` only works on Kaggle platform
2. **No Validation**: Assumes data is clean; no error checking
3. **Monolithic Logic**: Data processing scattered throughout notebook
4. **Unclear Splitting**: Train/test split logic unclear from code
5. **No Type Hints**: Data types and return values not explicit

### Data Module Goals

- Load CSV datasets with pandas
- Validate data integrity and schema
- Implement train/calibration/test splits with exact notebook methodology
- Provide type-safe API with comprehensive error handling
- Enable reproducibility via configuration

---

## Goals and Objectives

### Primary Goals

1. **CSV Data Loading**
   - Load datasets from configurable paths
   - Support multiple dataset formats
   - Handle missing/invalid files gracefully
   - Preserve data types and precision

2. **Data Validation**
   - Verify dataset schema (features, target column)
   - Check for missing values and anomalies
   - Validate feature ranges and data types
   - Provide detailed error messages for debugging

3. **Data Splitting**
   - Implement train/calibration/test split (preserving notebook methodology)
   - Separate inliers and outliers
   - Apply `train_fraction` to create calibration sets
   - Return data in correct format for downstream modules

4. **Cross-Platform Support**
   - Work with configurable data paths
   - Handle path separators correctly (Path objects)
   - Support Linux, Windows, macOS environments

### Success Metrics

- ✅ Load CARDIO dataset without errors
- ✅ Data validation catches schema mismatches
- ✅ Train/calib/test split matches notebook (quantitatively verified)
- ✅ All paths use configuration (no hardcoded paths)
- ✅ Unit tests pass (>85% coverage for data module)
- ✅ Reproducible results with seed control

---

## Detailed Requirements

### Data Module Architecture

```
src/bakc_plus/data/
├── __init__.py           # Package exports
├── loader.py             # CSV loading functionality
├── validator.py          # Data validation logic
├── splitter.py           # Train/calib/test splitting
├── schema.py             # Data schema definitions
└── utils.py              # Helper utilities
```

### Expected Data Flow

```
Raw CSV File (cardio.csv)
        ↓
Load CSV via pandas
        ↓
Validate Schema & Values
        ↓
Separate Inliers (y=0) & Outliers (y=1)
        ↓
Split Inliers by train_fraction
        ├─ Training Inliers: len(inliers) * train_fraction
        ├─ Test Inliers: len(inliers) * (1-train_fraction)
        └─ Held-out Outliers: all anomalies
        ↓
Return DataSplit object with:
        ├─ train: pd.DataFrame (training inliers only)
        ├─ test: pd.DataFrame (test inliers + all outliers)
        ├─ calibration: None (computed in conformal module)
        └─ metadata: dict with statistics
```

### CARDIO Dataset Specification

**File**: `data/input/cardio.csv`

**Schema**:
- Columns: `V1, V2, ..., V21, y`
- V1-V21: Features (float, normalized)
- y: Target (int, 0 or 1)
- Total rows: 1,831
- No header row (CSV format)

**Statistics**:
- Inliers (y=0): 1,655 samples (90.39%)
- Outliers (y=1): 176 samples (9.61%)
- Features: 21 numeric features
- Value ranges: Normalized (typically -3 to +3)

**Baseline Split** (train_fraction=0.5):
- Training inliers: 827 samples
- Test inliers: 828 samples
- Test outliers: 176 samples
- Total test: 1,004 samples

---

## Task Breakdown

### Task 1.4.1: Create loader.py Module

**Objective**: Implement CSV dataset loading functionality

**Implementation**:
1. Create `src/bakc_plus/data/loader.py`
2. Implement `DataLoader` class with methods:
   - `__init__(config: DataConfig, logger)`
   - `load_dataset(dataset_name: str) -> pd.DataFrame`
   - `load_cardio() -> pd.DataFrame` (specific loader for CARDIO)
3. Add file existence checking
4. Handle missing values and type conversion
5. Add logging at INFO/DEBUG levels
6. Implement comprehensive error messages

**Key Functions**:

```python
class DataLoader:
    """Load datasets from CSV files"""
    
    def __init__(self, config: DataConfig):
        """Initialize loader with data directory from config"""
        self.data_dir = config.data_dir
        self.logger = get_logger(__name__)
    
    def load_dataset(
        self,
        dataset_name: str,
        filename: Optional[str] = None,
        header: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load dataset from CSV file
        
        Args:
            dataset_name: Dataset identifier (e.g., 'cardio')
            filename: CSV filename (default: {dataset_name}.csv)
            header: CSV header row (default: None for no header)
        
        Returns:
            DataFrame with all columns and rows
        
        Raises:
            FileNotFoundError: If CSV file not found
            ValueError: If CSV is empty or malformed
        """
        # Implementation
    
    def load_cardio(self) -> pd.DataFrame:
        """Load CARDIO dataset with specific configuration"""
        # Load from cardio.csv with no header
        # Expects 22 columns (V1-V21 + y)
```

**Validation**:
- Loads CARDIO dataset successfully
- Returns correct number of rows (1,831)
- Returns correct number of columns (22)
- Handles missing file gracefully
- Logging captures file loading

### Task 1.4.2: Implement Data Validation

**Objective**: Add validation logic to check data integrity

**Implementation**:
1. Create `src/bakc_plus/data/validator.py`
2. Implement `DataValidator` class with methods:
   - `validate_schema(df, expected_features, target_col)`
   - `validate_values(df, target_col)`
   - `validate_complete(df, expected_features, target_col)`
3. Check for:
   - Correct number of columns
   - Presence of target column
   - No NaN values
   - Target is binary (0 or 1)
   - Feature value ranges (optional)
4. Raise descriptive `ValueError` on failures
5. Log validation results

**Key Functions**:

```python
class DataValidator:
    """Validate dataset schema and values"""
    
    def validate_schema(
        self,
        df: pd.DataFrame,
        expected_features: int,
        target_col: str = 'y'
    ) -> bool:
        """
        Validate dataset schema
        
        Checks:
        - DataFrame not None and not empty
        - Correct number of columns
        - Target column exists
        
        Raises:
            ValueError: If schema validation fails
        
        Returns:
            True if valid
        """
        # Implementation
    
    def validate_values(
        self,
        df: pd.DataFrame,
        target_col: str = 'y'
    ) -> bool:
        """
        Validate dataset values
        
        Checks:
        - No NaN values
        - Target column binary (0 or 1)
        - Feature values are numeric
        
        Raises:
            ValueError: If value validation fails
        
        Returns:
            True if valid
        """
        # Implementation
    
    def validate_complete(
        self,
        df: pd.DataFrame,
        expected_features: int = 21,
        target_col: str = 'y'
    ) -> bool:
        """Run all validation checks"""
```

**Validation**:
- Validates correct CARDIO dataset
- Rejects dataset with wrong column count
- Rejects dataset with missing target
- Rejects dataset with NaN values
- Rejects invalid target values
- Error messages are clear

### Task 1.4.3: Implement Splitting Functions

**Objective**: Create train/calibration/test split logic preserving notebook methodology

**Implementation**:
1. Create `src/bakc_plus/data/splitter.py`
2. Implement `DataSplitter` class with methods:
   - `split_inliers_outliers(df, target_col)`
   - `split_train_test(inliers, train_fraction)`
   - `create_data_split(df, train_fraction, target_col)`
3. Preserve exact notebook split logic:
   ```python
   # From notebook
   inliers = df[df['y'] == 0]
   outliers = df[df['y'] == 1]
   len_train = len(inliers) // 2
   train = inliers[:len_train]
   test = inliers[len_train:]
   ```
4. Return `DataSplit` namedtuple with all splits
5. Include metadata (row counts, feature info)
6. Log split statistics

**Key Classes**:

```python
from typing import NamedTuple

class DataSplit(NamedTuple):
    """Result of train/test split"""
    train: pd.DataFrame       # Training inliers only
    test: pd.DataFrame        # Test inliers + outliers
    inliers: pd.DataFrame     # All inliers (before split)
    outliers: pd.DataFrame    # All outliers
    metadata: dict            # Statistics

class DataSplitter:
    """Split data into train/test sets"""
    
    def split_inliers_outliers(
        self,
        df: pd.DataFrame,
        target_col: str = 'y'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separate inliers (y=0) and outliers (y=1)
        
        Returns:
            (inliers_df, outliers_df)
        """
        # Implementation
    
    def split_train_test(
        self,
        inliers: pd.DataFrame,
        train_fraction: float = 0.5
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split inliers into train and test
        
        Args:
            inliers: DataFrame with only inliers (y=0)
            train_fraction: Fraction for training (0.0-1.0)
        
        Returns:
            (train_inliers, test_inliers)
        
        Example (CARDIO):
            Input: 1,655 inliers, train_fraction=0.5
            Output: (827 train, 828 test)
        """
        # Implementation
    
    def create_data_split(
        self,
        df: pd.DataFrame,
        train_fraction: float = 0.5,
        target_col: str = 'y',
        random_state: int = 42
    ) -> DataSplit:
        """
        Create train/test split from full dataset
        
        Returns:
            DataSplit with train, test, metadata
        """
        # Implementation
```

**Validation**:
- CARDIO dataset splits to correct sizes (827, 828, 176)
- Train contains only inliers
- Test contains inliers + outliers
- No data leakage between splits
- Reproducible with seed
- Metadata calculated correctly

### Task 1.4.4: Write Unit Tests

**Objective**: Comprehensive tests for data module

**Test File**: `tests/unit/test_data.py`

**Test Structure**:
```
tests/unit/
├── test_data_loader.py      # Tests for loader.py
├── test_data_validator.py   # Tests for validator.py
├── test_data_splitter.py    # Tests for splitter.py
└── conftest.py              # Shared fixtures
```

**Test Cases** (minimum 25 tests):

**Loader Tests** (7 tests):
1. `test_load_cardio_dataset()` - Load CARDIO dataset
2. `test_load_dataset_file_not_found()` - Error on missing file
3. `test_load_dataset_empty_file()` - Error on empty CSV
4. `test_load_dataset_correct_shape()` - Verify shape (1831, 22)
5. `test_load_dataset_data_types()` - Features are numeric, target is int
6. `test_load_dataset_no_header()` - CSV parsed without header
7. `test_load_dataset_logging()` - Logs file loading

**Validator Tests** (8 tests):
8. `test_validate_schema_valid()` - Valid CARDIO data passes
9. `test_validate_schema_wrong_columns()` - Reject wrong column count
10. `test_validate_schema_missing_target()` - Reject missing 'y'
11. `test_validate_values_no_nans()` - Accept data without NaN
12. `test_validate_values_with_nans()` - Reject data with NaN
13. `test_validate_values_valid_target()` - Accept binary target
14. `test_validate_values_invalid_target()` - Reject non-binary target
15. `test_validate_complete()` - Full validation passes

**Splitter Tests** (10 tests):
16. `test_split_inliers_outliers_cardio()` - Correct class separation (1655, 176)
17. `test_split_inliers_outliers_all_inliers()` - Edge case: all inliers
18. `test_split_inliers_outliers_all_outliers()` - Edge case: all outliers
19. `test_split_train_test_fraction_50()` - 50% split (827, 828)
20. `test_split_train_test_fraction_25()` - 25% split
21. `test_split_train_test_no_overlap()` - Train and test don't overlap
22. `test_create_data_split_structure()` - DataSplit has all fields
23. `test_create_data_split_metadata()` - Metadata contains statistics
24. `test_create_data_split_reproducible()` - Same seed = same split
25. `test_data_split_train_only_inliers()` - Train contains no outliers

**Example Tests**:

```python
def test_load_cardio_dataset(data_loader):
    """Test loading CARDIO dataset"""
    df = data_loader.load_cardio()
    
    # Check shape
    assert df.shape[0] == 1831, "Expected 1,831 rows"
    assert df.shape[1] == 22, "Expected 22 columns (V1-V21 + y)"
    
    # Check column names
    expected_cols = [f'V{i}' for i in range(1, 22)] + ['y']
    assert list(df.columns) == expected_cols
    
    # Check data types
    for col in df.columns[:-1]:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} should be numeric"
    assert pd.api.types.is_integer_dtype(df['y']), "Target column should be integer"

def test_split_inliers_outliers_cardio(data_splitter):
    """Test inlier/outlier separation on CARDIO data"""
    inliers, outliers = data_splitter.split_inliers_outliers(df)
    
    # Check counts
    assert len(inliers) == 1655, "Expected 1,655 inliers"
    assert len(outliers) == 176, "Expected 176 outliers"
    
    # Check purity
    assert (inliers['y'] == 0).all(), "Inliers should have y=0"
    assert (outliers['y'] == 1).all(), "Outliers should have y=1"

def test_split_train_test_fraction_50(data_splitter, cardio_inliers):
    """Test 50% train/test split matches notebook"""
    train, test = data_splitter.split_train_test(cardio_inliers, train_fraction=0.5)
    
    # Verify notebook-exact split
    assert len(train) == 827, "Expected 827 training samples"
    assert len(test) == 828, "Expected 828 test samples"
    assert len(train) + len(test) == 1655, "Should account for all inliers"

def test_create_data_split_structure(data_splitter, cardio_df):
    """Test DataSplit return structure"""
    split = data_splitter.create_data_split(cardio_df)
    
    # Check all fields present
    assert hasattr(split, 'train'), "Should have train field"
    assert hasattr(split, 'test'), "Should have test field"
    assert hasattr(split, 'inliers'), "Should have inliers field"
    assert hasattr(split, 'outliers'), "Should have outliers field"
    assert hasattr(split, 'metadata'), "Should have metadata field"
    
    # Check metadata contains key statistics
    assert 'n_train' in split.metadata
    assert 'n_test' in split.metadata
    assert 'n_inliers' in split.metadata
    assert 'n_outliers' in split.metadata
    assert 'n_features' in split.metadata
```

**Validation**:
- All tests pass
- Coverage >85% for data module
- Tests are isolated and reproducible
- Tests cover success and failure paths

---

## Acceptance Criteria

### AC1.4.1: Data Loader Implementation
- [ ] `src/bakc_plus/data/loader.py` created
- [ ] `DataLoader` class implemented with type hints
- [ ] `load_dataset()` method works with configurable paths
- [ ] `load_cardio()` method loads CARDIO dataset (1,831 x 22)
- [ ] Handles missing files with clear error messages
- [ ] Logging tracks file loading and data info
- [ ] No hardcoded paths (uses config)
- [ ] Docstrings present for all methods

### AC1.4.2: Data Validation Implementation
- [ ] `src/bakc_plus/data/validator.py` created
- [ ] `DataValidator` class validates schema
- [ ] Validates schema (columns, target column)
- [ ] Validates values (no NaN, binary target)
- [ ] Catches schema mismatches with clear errors
- [ ] Catches value errors with actionable messages
- [ ] Logging records validation results
- [ ] Unit tests verify all validation paths

### AC1.4.3: Data Splitting Implementation
- [ ] `src/bakc_plus/data/splitter.py` created
- [ ] `DataSplitter` class separates inliers/outliers correctly
- [ ] `split_train_test()` preserves notebook methodology
- [ ] Produces correct split sizes for CARDIO (827, 828, 176)
- [ ] `DataSplit` namedtuple has all required fields
- [ ] Metadata includes row counts and feature info
- [ ] Reproducible with seed parameter
- [ ] No data leakage between splits

### AC1.4.4: Unit Tests
- [ ] Test file `tests/unit/test_data.py` created
- [ ] Minimum 25 test cases implemented
- [ ] All tests pass
- [ ] Code coverage >85% for data module
- [ ] Tests cover loader, validator, splitter
- [ ] Tests include success and error paths
- [ ] Fixtures for CARDIO data provided

### AC1.4.5: Package Integration
- [ ] `src/bakc_plus/data/__init__.py` created
- [ ] Exports `DataLoader`, `DataValidator`, `DataSplitter`, `DataSplit`
- [ ] Can import: `from bakc_plus.data import DataLoader`
- [ ] Can import: `from bakc_plus.data import DataSplit`
- [ ] Module is properly documented in __init__

### AC1.4.6: Configuration Integration
- [ ] Config module provides `DataConfig` (from step 1.2)
- [ ] Data module uses config for paths
- [ ] `train_fraction` configurable via config
- [ ] Test results reproducible with same config

### AC1.4.7: Schema Definition
- [ ] `src/bakc_plus/data/schema.py` defines CARDIO schema
- [ ] Specifies expected columns (V1-V21, y)
- [ ] Specifies column types (float, int)
- [ ] Specifies column ranges/constraints
- [ ] Provides schema validation helper

---

## Definition of Done

Step 1.4 is considered **DONE** when:

1. ✅ **All Acceptance Criteria Met** - Every item AC1.4.1 through AC1.4.7 verified

2. ✅ **Data Loading Works**
   ```bash
   python -c "
   import sys; sys.path.insert(0, 'src')
   from bakc_plus.data import DataLoader
   from bakc_plus.config import BaKCConfig
   cfg = BaKCConfig.from_yaml('configs/cardio.yaml')
   loader = DataLoader(cfg.data)
   df = loader.load_cardio()
   print(f'Loaded {df.shape[0]} rows, {df.shape[1]} columns')
   "
   # Output: "Loaded 1831 rows, 22 columns"
   ```

3. ✅ **Data Validation Works**
   ```bash
   python -c "
   import sys; sys.path.insert(0, 'src')
   from bakc_plus.data import DataValidator
   validator = DataValidator()
   df = ...  # load data
   validator.validate_complete(df, expected_features=21)
   print('Validation passed')
   "
   ```

4. ✅ **Data Splitting Works**
   ```bash
   python -c "
   import sys; sys.path.insert(0, 'src')
   from bakc_plus.data import DataSplitter
   splitter = DataSplitter()
   split = splitter.create_data_split(df, train_fraction=0.5)
   print(f'Train: {len(split.train)}, Test: {len(split.test)}')
   "
   # Output: "Train: 827, Test: 1004"
   ```

5. ✅ **CARDIO Dataset Quantitatively Verified**
   - Loaded dataset: 1,831 rows, 22 columns
   - Inliers: 1,655 (90.39%)
   - Outliers: 176 (9.61%)
   - Train split (50%): 827 inliers
   - Test split: 828 inliers + 176 outliers = 1,004 total

6. ✅ **Unit Tests Pass**
   ```bash
   PYTHONPATH=src pytest tests/unit/test_data.py -v --cov=src/bakc_plus/data --cov-report=term-missing
   # All tests pass, coverage >85%
   ```

7. ✅ **No Hardcoded Paths**
   - No '/kaggle/' paths in data module
   - All paths use config objects
   - Paths work on Linux, Windows, macOS

8. ✅ **Logging Integration**
   - Data loading events logged at INFO level
   - Validation results logged at DEBUG level
   - Errors logged at ERROR level
   - Log file contains data module messages

9. ✅ **No Issues in Issue Log** - All discovered issues resolved

10. ✅ **Code Committed** - All changes committed with clear messages

---

## Issue Log

| ID | Date | Issue Description | Resolution | Status |
|----|------|-------------------|------------|--------|
| - | - | - | - | - |

---

## Implementation Notes

### Design Decisions

1. **Separate Concerns**: Loader, Validator, Splitter as distinct classes
2. **Named Tuples**: Use `DataSplit` for immutable, typed return values
3. **Logging Integration**: All operations logged via get_logger()
4. **Config-Driven**: All paths and parameters from config object
5. **Type Hints**: Full type hints for IDE support and documentation
6. **Preserve Methodology**: Exact notebook split logic preserved (no randomization of train/test)

### Critical Preservation Requirements

From notebook analysis, MUST preserve:
1. Inlier/outlier separation by y column (0=inlier, 1=outlier)
2. Train fraction split: `len_train = len(inliers) // 2`
3. No random shuffling of train/test split (deterministic indexing)
4. All 21 features preserved (V1-V21)
5. Target column preserved as-is (y)

### Performance Considerations

- pandas for efficient CSV loading (C-optimized)
- In-memory processing (dataset is small: 1.8K rows)
- No unnecessary copies of large DataFrames
- Validation on load (fail-fast principle)

### Integration Points

1. **Config Module**: Uses DataConfig from step 1.2
2. **Logger Module**: Uses get_logger() from step 1.3
3. **Conformal Module** (future): Will consume DataSplit object
4. **CLI Scripts** (future): Will use DataLoader for data ingestion

### References

- **NOTEBOOK_ANALYSIS.md**: Section 2 (Data Flow), Section 4.3 (Hardcoded Paths), Section 6.3 (Validation)
- **REFACTORING_PLAN.md**: Data module design
- **ocsvm_x_cv_x_bagging.py**: Lines 7-25 (data loading), lines 127-140 (splitting logic)

---

## Next Steps

After Step 1.4 is DONE:

1. Validate all acceptance criteria
2. Run full test suite with coverage
3. Verify CARDIO dataset quantitatively
4. Ensure zero issues in issue log
5. Commit changes to branch
6. Update Phase 1 progress tracking
7. Proceed to Step 1.5 (Conformal Prediction Module)

---

**Document Version**: 1.0
**Created**: 2025-11-18
**Last Updated**: 2025-11-18
**Status**: Ready for Implementation
