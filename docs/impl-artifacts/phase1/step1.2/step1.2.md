# Step 1.2: Configuration System

**Parent**: Phase 1 - Core Infrastructure
**Timeline**: Days 3-4
**Status**: In Progress
**Dependencies**: Step 1.1 (Project Setup) ✅ Complete

---

## Overview

Step 1.2 implements a YAML-based configuration system using Python dataclasses to replace hardcoded values and enable flexible, multi-dataset support. This is critical for transitioning from the notebook's hardcoded Kaggle paths to a production-ready configuration approach.

### Context from Existing Analysis

From **EXECUTION_SUMMARY.md** (Issues Identified):
> **High Priority Issue #2: Hardcoded Kaggle Paths**
> - Paths like `/kaggle/input/` only work on Kaggle platform
> - **FIX**: Use configuration-based path management

From **NOTEBOOK_ANALYSIS.md** (Section 6.2):
> **CURRENT**:
> ```python
> num_models = 5
> len_train = len(inliers_df) // 2
> WORKDIR_PATH = '/kaggle/working'
> ```
> **Problem**: Global variable pollution, hard to test, not configurable

From **ocsvm_x_cv_x_bagging.py** (lines 43-44, 127):
```python
dataset_folder_path = '/kaggle/input/adbench/input/gamma'  # Hardcoded!
WORKDIR_PATH = '/kaggle/working'  # Hardcoded!
num_models = 5  # Global constant
```

### Current Hardcoded Values to Replace

From the notebook analysis:
- `dataset_folder_path = '/kaggle/input/...'` → `config.data.data_dir`
- `num_models = 5` → `config.ensemble.num_models`
- `alpha = 0.05` → `config.conformal.alpha`
- `nu = 0.05` → `config.model.nu`
- `random_state = 42` → `config.ensemble.random_state`
- `J = 5` (repetitions) → `config.ensemble.num_repetitions`
- `L = 20` (test splits) → `config.ensemble.num_test_splits`

---

## Goals and Objectives

### Primary Goals

1. **Eliminate Hardcoded Paths**
   - Replace all `/kaggle/input/` and `/kaggle/working/` paths
   - Support configurable data directories via YAML
   - Enable cross-platform operation (Linux, Windows, macOS)

2. **Centralize Configuration**
   - All hyperparameters in one place
   - Type-safe configuration with dataclasses
   - Validation on load (fail fast if invalid)

3. **Multi-Dataset Support**
   - Default configuration for baseline
   - Dataset-specific overrides (cardio.yaml, gamma.yaml)
   - Easy to add new datasets

4. **Reproducibility**
   - Configuration files are version-controlled
   - Exact parameters documented
   - Random seeds configurable

### Success Metrics

- ✅ No hardcoded paths in code
- ✅ Configuration loads from YAML without errors
- ✅ Validation catches invalid values
- ✅ CARDIO dataset config matches baseline parameters
- ✅ Unit tests pass (>80% coverage for config module)

---

## Detailed Requirements

### Configuration Structure

From **REFACTORING_PLAN.md** (Configuration Module Design):

```python
@dataclass
class DataConfig:
    """Data loading and preprocessing configuration"""
    dataset_name: str
    data_dir: Path = Path("./data/input")
    output_dir: Path = Path("./output")
    train_fraction: float = 0.5
    len_cal: Optional[int] = None  # Auto-computed if None
    len_test: Optional[int] = None  # Auto-computed if None

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
    num_models: int = 5  # M
    num_folds: Optional[int] = None  # K: dynamic if None
    num_test_splits: int = 20  # L
    num_repetitions: int = 5  # J
    random_state: int = 42
    use_multiprocessing: bool = True
    num_workers: Optional[int] = None

@dataclass
class ConformalConfig:
    """Conformal prediction configuration"""
    alpha: float = 0.05
    scoring_method: str = "sigmoid"
    quantile_method: str = "higher"
    fold_aggregation: str = "mean"
    cross_fold_aggregation: str = "median"

@dataclass
class BaKCConfig:
    """Main configuration"""
    data: DataConfig
    model: ModelConfig
    ensemble: EnsembleConfig
    conformal: ConformalConfig
    save_models: bool = True
    save_calibration: bool = True
    save_predictions: bool = True
```

### Critical Values (MUST PRESERVE for CARDIO)

From **EXECUTION_SUMMARY.md** baseline:
- `nu = 0.05` (OC-SVM parameter)
- `num_models = 5` (ensemble size M)
- `alpha = 0.05` (FDR control level)
- `random_state = 42` (for reproducibility)
- `num_test_splits = 20` (L)
- `num_repetitions = 5` (J)
- `scoring_method = "sigmoid"` (active scoring function)
- `quantile_method = "higher"` (numpy quantile method)

---

## Task Breakdown

### Task 1.2.1: Create config.py with Dataclasses

**Objective**: Implement configuration dataclasses with type hints

**Implementation**:
1. Create `src/bakc_plus/config.py`
2. Import required types: `dataclass`, `Optional`, `Path`
3. Implement `DataConfig`, `ModelConfig`, `EnsembleConfig`, `ConformalConfig`
4. Implement `BaKCConfig` as main config container
5. Add `__post_init__` for Path conversion in DataConfig
6. Add docstrings for all classes and fields

**Validation**:
- All dataclasses instantiate without errors
- Default values match baseline parameters
- Type hints are correct and comprehensive

### Task 1.2.2: Implement YAML Loading

**Objective**: Add `from_yaml()` class method to load configuration

**Implementation**:
1. Add PyYAML import: `import yaml`
2. Implement `BaKCConfig.from_yaml(cls, path: str) -> 'BaKCConfig'`
3. Load YAML file and parse to dict
4. Create nested dataclass instances from dict
5. Handle missing optional fields with defaults
6. Add clear error messages for parsing failures

**Critical Logic**:
```python
@classmethod
def from_yaml(cls, path: str) -> 'BaKCConfig':
    """Load configuration from YAML file"""
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return cls(
        data=DataConfig(**config_dict.get('data', {})),
        model=ModelConfig(**config_dict.get('model', {})),
        ensemble=EnsembleConfig(**config_dict.get('ensemble', {})),
        conformal=ConformalConfig(**config_dict.get('conformal', {})),
        save_models=config_dict.get('save_models', True),
        save_calibration=config_dict.get('save_calibration', True),
        save_predictions=config_dict.get('save_predictions', True),
    )
```

**Validation**:
- Loads valid YAML without errors
- Missing fields use defaults
- Invalid YAML raises clear exception

### Task 1.2.3: Implement Configuration Validation

**Objective**: Add `validate()` method to catch invalid values

**Implementation**:
1. Add `validate()` method to `BaKCConfig`
2. Check `alpha` in range (0, 1)
3. Check `nu` in range (0, 1)
4. Check `num_models > 0`
5. Check `num_test_splits > 0`
6. Check `num_repetitions > 0`
7. Check `train_fraction` in range (0, 1)
8. Check `scoring_method` in allowed values
9. Check `data_dir` and `output_dir` are valid paths
10. Raise `ValueError` with descriptive message if invalid

**Validation Logic**:
```python
def validate(self) -> None:
    """Validate configuration values"""
    if not (0 < self.conformal.alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {self.conformal.alpha}")

    if not (0 < self.model.nu < 1):
        raise ValueError(f"nu must be in (0, 1), got {self.model.nu}")

    if self.ensemble.num_models <= 0:
        raise ValueError(f"num_models must be positive, got {self.ensemble.num_models}")

    # ... more validations

    allowed_scoring = ['sigmoid', 'normalize', 'signed_ohe']
    if self.conformal.scoring_method not in allowed_scoring:
        raise ValueError(f"scoring_method must be one of {allowed_scoring}")

    # Validate paths exist or can be created
    self.data.data_dir.mkdir(parents=True, exist_ok=True)
    self.data.output_dir.mkdir(parents=True, exist_ok=True)
```

**Validation**:
- Catches all invalid value combinations
- Error messages are clear and actionable
- Valid configs pass without errors

### Task 1.2.4: Create Default YAML Configuration

**Objective**: Create `configs/default.yaml` with baseline parameters

**Content** (`configs/default.yaml`):
```yaml
# Default BaKC-plus Configuration
# This configuration uses baseline parameters from the original notebook

data:
  dataset_name: "cardio"
  data_dir: "./data/input"
  output_dir: "./output"
  train_fraction: 0.5
  # len_cal and len_test are auto-computed if null
  len_cal: null
  len_test: null

model:
  nu: 0.05
  kernel: "rbf"
  gamma: "scale"
  cache_size: 200
  verbose: false

ensemble:
  num_models: 5  # M: ensemble members per fold
  num_folds: null  # K: computed dynamically as len(train)//len_cal or 20
  num_test_splits: 20  # L: test splits per repetition
  num_repetitions: 5  # J: outer loop repetitions
  random_state: 42
  use_multiprocessing: true
  num_workers: null  # Auto-detect CPU count

conformal:
  alpha: 0.05  # FDR control level
  scoring_method: "sigmoid"  # Options: sigmoid, normalize, signed_ohe
  quantile_method: "higher"  # numpy quantile method
  fold_aggregation: "mean"  # Per-fold aggregation
  cross_fold_aggregation: "median"  # Cross-fold aggregation

# Save options
save_models: true
save_calibration: true
save_predictions: true
```

**Validation**:
- YAML parses without errors
- Loaded config matches baseline parameters
- All fields present and correctly typed

### Task 1.2.5: Create CARDIO Dataset Configuration

**Objective**: Create `configs/cardio.yaml` for CARDIO dataset

**Content** (`configs/cardio.yaml`):
```yaml
# BaKC-plus Configuration for CARDIO Dataset
# Dataset: 1,831 samples, 21 features, 9.61% anomalies
# Baseline Performance: Power=90.29%, FDR=8.47%

data:
  dataset_name: "cardio"
  data_dir: "./data/input"
  output_dir: "./output/cardio"
  train_fraction: 0.5  # 50% of inliers for training

model:
  nu: 0.05  # Expected outlier fraction in training data

ensemble:
  num_models: 5
  num_folds: null  # Will be computed: len(train)//len_cal
  num_test_splits: 20
  num_repetitions: 5
  random_state: 42

conformal:
  alpha: 0.05
  scoring_method: "sigmoid"

save_models: true
save_calibration: true
save_predictions: true
```

**Validation**:
- Loads successfully
- Output directory is dataset-specific
- All baseline parameters preserved

### Task 1.2.6: Write Unit Tests for Config Module

**Objective**: Comprehensive unit tests for configuration

**Test File**: `tests/unit/test_config.py`

**Test Cases**:
1. `test_data_config_defaults()` - Test DataConfig default values
2. `test_model_config_defaults()` - Test ModelConfig default values
3. `test_ensemble_config_defaults()` - Test EnsembleConfig defaults
4. `test_conformal_config_defaults()` - Test ConformalConfig defaults
5. `test_bakc_config_creation()` - Test main config creation
6. `test_load_default_yaml()` - Test loading default.yaml
7. `test_load_cardio_yaml()` - Test loading cardio.yaml
8. `test_missing_yaml_file()` - Test error on missing file
9. `test_invalid_yaml_format()` - Test error on malformed YAML
10. `test_validation_invalid_alpha()` - Test alpha validation
11. `test_validation_invalid_nu()` - Test nu validation
12. `test_validation_invalid_num_models()` - Test num_models validation
13. `test_validation_invalid_scoring_method()` - Test scoring method validation
14. `test_path_conversion()` - Test Path conversion in DataConfig
15. `test_config_round_trip()` - Test save and load cycle

**Example Test**:
```python
def test_load_cardio_yaml():
    """Test loading CARDIO configuration from YAML"""
    config = BaKCConfig.from_yaml('configs/cardio.yaml')

    # Check data config
    assert config.data.dataset_name == "cardio"
    assert config.data.train_fraction == 0.5

    # Check model config
    assert config.model.nu == 0.05

    # Check ensemble config
    assert config.ensemble.num_models == 5
    assert config.ensemble.random_state == 42

    # Check conformal config
    assert config.conformal.alpha == 0.05
    assert config.conformal.scoring_method == "sigmoid"
```

**Validation**:
- All tests pass
- Coverage >80% for config.py
- Tests are isolated and reproducible

---

## Acceptance Criteria

### AC1.2.1: Dataclass Implementation
- [ ] All 5 dataclasses implemented (DataConfig, ModelConfig, EnsembleConfig, ConformalConfig, BaKCConfig)
- [ ] All fields have type hints
- [ ] Default values match baseline parameters
- [ ] Docstrings present for all classes
- [ ] Can instantiate all dataclasses without errors

### AC1.2.2: YAML Loading
- [ ] `from_yaml()` method implemented on BaKCConfig
- [ ] Loads `configs/default.yaml` successfully
- [ ] Loads `configs/cardio.yaml` successfully
- [ ] Missing fields use default values
- [ ] Invalid YAML raises clear exception with helpful message

### AC1.2.3: Configuration Validation
- [ ] `validate()` method implemented
- [ ] Catches alpha not in (0, 1)
- [ ] Catches nu not in (0, 1)
- [ ] Catches negative num_models
- [ ] Catches invalid scoring_method
- [ ] Creates output directories if they don't exist
- [ ] Validation error messages are clear and actionable

### AC1.2.4: Configuration Files
- [ ] `configs/default.yaml` created with all required fields
- [ ] `configs/cardio.yaml` created with CARDIO-specific settings
- [ ] Both YAML files parse without syntax errors
- [ ] Loaded configs have correct values
- [ ] CARDIO config preserves baseline parameters (nu=0.05, num_models=5, alpha=0.05, etc.)

### AC1.2.5: Unit Tests
- [ ] Test file `tests/unit/test_config.py` created
- [ ] At least 15 test cases implemented
- [ ] All tests pass
- [ ] Code coverage >80% for config.py
- [ ] Tests cover success and failure paths

### AC1.2.6: Integration
- [ ] Config module is importable: `from bakc_plus.config import BaKCConfig`
- [ ] Can load and validate config in one line: `config = BaKCConfig.from_yaml('configs/cardio.yaml'); config.validate()`
- [ ] No hardcoded paths remain in config code
- [ ] Path objects work cross-platform (Path handles separators)

---

## Definition of Done

Step 1.2 is considered **DONE** when:

1. ✅ **All Acceptance Criteria Met** - Every item in AC1.2.1 through AC1.2.6 is checked and validated

2. ✅ **Config Loading Verified**
   ```bash
   python -c "from bakc_plus.config import BaKCConfig; cfg = BaKCConfig.from_yaml('configs/cardio.yaml'); print(f'Dataset: {cfg.data.dataset_name}, nu={cfg.model.nu}')"
   # Output: "Dataset: cardio, nu=0.05"
   ```

3. ✅ **Validation Works**
   ```bash
   python -c "from bakc_plus.config import BaKCConfig, ConformalConfig; cfg = BaKCConfig(data=..., model=..., ensemble=..., conformal=ConformalConfig(alpha=1.5)); cfg.validate()"
   # Should raise ValueError: "alpha must be in (0, 1)"
   ```

4. ✅ **Unit Tests Pass**
   ```bash
   pytest tests/unit/test_config.py -v --cov=src/bakc_plus/config --cov-report=term-missing
   # All tests pass, coverage >80%
   ```

5. ✅ **Baseline Parameters Preserved**
   - CARDIO config has nu=0.05, num_models=5, alpha=0.05, random_state=42
   - All critical values match EXECUTION_SUMMARY.md baseline

6. ✅ **No Hardcoded Paths**
   - No '/kaggle/' paths in config.py
   - All paths use Path objects
   - Paths are configurable via YAML

7. ✅ **Documentation Complete**
   - All dataclasses have docstrings
   - All methods have docstrings
   - YAML files have inline comments explaining parameters

8. ✅ **No Issues in Issue Log** - All issues discovered during AC validation are resolved

9. ✅ **Code Committed and Pushed** - All changes committed with clear messages

10. ✅ **Step 1.2 Document Updated** - This document reflects actual implementation

---

## Issue Log

| ID | Date | Issue Description | Resolution | Status |
|----|------|-------------------|------------|--------|
| - | - | - | - | - |

---

## Implementation Notes

### Design Decisions

1. **Use Dataclasses**: Python 3.7+ dataclasses for clean, type-safe config (no external deps)
2. **YAML Format**: Human-readable, widely supported, good for version control
3. **Fail Fast**: Validate on load, raise exceptions immediately if invalid
4. **Path Objects**: Use `pathlib.Path` for cross-platform compatibility
5. **Optional Fields**: Use `Optional[T]` for auto-computed values (num_folds, len_cal)

### Critical Considerations

1. **Preserve Baseline Parameters**: CARDIO config must exactly match baseline
2. **Type Safety**: All parameters must have type hints
3. **Validation**: Catch common mistakes (alpha>1, negative counts, typos in method names)
4. **Documentation**: Each parameter needs clear explanation
5. **Testing**: Cover both success and failure paths

### References

- **REFACTORING_PLAN.md**: Configuration module design (lines 195-309)
- **EXECUTION_SUMMARY.md**: Baseline parameters that must be preserved
- **ocsvm_x_cv_x_bagging.py**: Current hardcoded values to replace (lines 43-44, 112, 127, 178-184)

---

## Next Steps

After Step 1.2 is DONE:
1. Validate against all AC
2. Run validation script (to be created)
3. Ensure zero issues in issue log
4. Commit changes
5. Update Phase 1 progress
6. Move to Step 1.3 (Logging System)

---

**Document Version**: 1.0
**Created**: 2025-11-18
**Last Updated**: 2025-11-18
**Status**: Ready for Implementation
