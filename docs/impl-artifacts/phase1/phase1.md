# Phase 1: Core Infrastructure

**Timeline**: Week 1 (Days 1-7)
**Status**: In Progress
**Parent**: BaKC-plus Refactoring Project

---

## Overview

Phase 1 establishes the foundational infrastructure for the refactored BaKC-plus package. This phase transforms the monolithic notebook into a well-structured Python package with configuration management, logging, and data handling capabilities.

### Context from Existing Analysis

From **NOTEBOOK_ANALYSIS.md** (Section 6.1):
> "Current: Single .ipynb with 97 cells, mixed concerns"
> "Problem: Hard to test, maintain, extend"

From **EXECUTION_SUMMARY.md** (Section: Issues Identified):
> **Critical Issues:**
> 1. Hardcoded Kaggle paths - Only works on Kaggle platform
> 2. No Input Validation - Assumes data is clean, properly formatted
> 3. Global Variable Pollution - Variables scattered across cells

From **REFACTORING_PLAN.md** (Phase 1 Objectives):
> Phase 1 creates the foundation: project structure, configuration system, logging, and data module with proper validation.

### Current State (Baseline)

**Existing Files**:
- `oc-svm-x-cv-x-bagging (1).ipynb` - 97 cells, monolithic structure
- `ocsvm_x_cv_x_bagging.py` - 714 lines, converted script (still has `visited_i` bug in line 284)
- `requirements.txt` - Dependencies defined
- `data/input/` - Multiple datasets (cardio, gamma, etc.)
- `output/` - Results directory

**Current Issues**:
1. Hardcoded paths: `/kaggle/input/`, `/kaggle/working/`
2. No configuration management
3. No logging (only tqdm progress bars)
4. No input validation
5. Global constants scattered throughout
6. No package structure

**Baseline Performance (Must Preserve)**:
- Power: 90.29% ± 8.61%
- FDR: 8.47% ± 3.39%
- Dataset: CARDIO (1,831 samples, 21 features)

---

## Goals and Objectives

### Primary Goals

1. **Establish Package Structure**
   - Create proper Python package layout
   - Enable `pip install -e .` for development
   - Support imports like `from bakc_plus.config import BaKCConfig`

2. **Configuration Management**
   - Replace hardcoded paths with configuration files
   - Support multiple datasets via YAML configs
   - Enable easy hyperparameter tuning

3. **Logging Infrastructure**
   - Replace print statements with structured logging
   - Enable debugging and monitoring
   - Support different log levels (DEBUG, INFO, WARNING, ERROR)

4. **Data Handling Layer**
   - Abstract data loading from specific paths
   - Add input validation to catch errors early
   - Support train/test splitting with proper error handling

### Success Metrics

- ✅ Package installable via `pip install -e .`
- ✅ Configuration loads from YAML without errors
- ✅ Logging outputs structured messages
- ✅ Data module handles CARDIO dataset correctly
- ✅ All unit tests pass (>80% coverage for Phase 1 modules)

---

## Detailed Scope

### Step 1.1: Project Setup

**Objective**: Create the foundational project structure

**Tasks**:
1. Create directory structure (`src/bakc_plus/`, `tests/`, `configs/`, `scripts/`)
2. Create `setup.py` for package installation
3. Create `pytest.ini` for test configuration
4. Update `.gitignore` with Python-specific patterns
5. Create `__init__.py` files for package initialization

**Key Decisions**:
- Use `src/` layout (not flat layout) for better import isolation
- Package name: `bakc_plus` (underscore, not hyphen)
- Minimum Python version: 3.8 (for dataclasses, type hints)

### Step 1.2: Configuration System

**Objective**: Implement YAML-based configuration with dataclasses

**Tasks**:
1. Create `src/bakc_plus/config.py` with dataclasses
2. Implement `DataConfig`, `ModelConfig`, `EnsembleConfig`, `ConformalConfig`
3. Implement `BaKCConfig` with `from_yaml()` and `validate()` methods
4. Create `configs/default.yaml` with default values
5. Create `configs/cardio.yaml` for CARDIO dataset
6. Write unit tests for config loading and validation

**Key Decisions**:
- Use Python 3.7+ `dataclasses` (built-in, no external deps)
- Use `PyYAML` for YAML parsing
- Validate on load (fail fast if config is invalid)

**Critical Values to Preserve**:
- `nu = 0.05` (OC-SVM parameter)
- `num_models = 5` (ensemble size)
- `alpha = 0.05` (FDR control level)
- `random_state = 42` (for reproducibility)

### Step 1.3: Logging System

**Objective**: Replace print statements with structured logging

**Tasks**:
1. Create `src/bakc_plus/logger.py` with logging setup
2. Configure log levels, formatters, and handlers
3. Support file logging to `output/logs/`
4. Add convenience methods for common log patterns
5. Write unit tests for logger configuration

**Key Decisions**:
- Use Python's built-in `logging` module
- Format: `[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s`
- Default level: INFO (DEBUG in development mode)
- Log file rotation: Keep last 5 files, 10MB each

### Step 1.4: Data Module

**Objective**: Create abstraction layer for data loading and validation

**Tasks**:
1. Create `src/bakc_plus/data/loader.py` - Data loading from CSV
2. Create `src/bakc_plus/data/validator.py` - Input validation
3. Create `src/bakc_plus/data/splitter.py` - Train/test splitting
4. Create unit tests for each module
5. Test with CARDIO dataset to ensure compatibility

**Key Decisions**:
- Support CSV format (current datasets)
- Handle 'Class' → 'y' column renaming automatically
- Validate: no NaN, binary target, expected feature count
- Use pandas DataFrames for data representation

**Critical Behavior to Preserve**:
From `ocsvm_x_cv_x_bagging.py` lines 43-46:
```python
df = pd.read_csv(os.path.join(dataset_folder_path, dataset_file_name))
df.rename(columns={'Class': 'y'}, inplace=True)
```

From lines 84-86:
```python
inliers_df = df.loc[df['y'] == 0]
outliers_df = df.loc[df['y'] == 1]
```

---

## Dependencies

### From Existing `requirements.txt`:
- numpy>=1.20.0
- pandas>=1.3.0
- scikit-learn>=1.0.0
- scipy>=1.7.0
- tqdm>=4.60.0
- matplotlib>=3.4.0

### New Dependencies (Phase 1):
- PyYAML>=5.4.0 (for configuration)
- pytest>=6.2.0 (for testing)
- pytest-cov>=2.12.0 (for coverage)

---

## Acceptance Criteria

### AC1: Project Structure
- [ ] Directory structure matches REFACTORING_PLAN.md target architecture
- [ ] All `__init__.py` files are present and correct
- [ ] `setup.py` is complete with all metadata and dependencies
- [ ] Package installs successfully with `pip install -e .`
- [ ] Can import `bakc_plus` modules without errors

### AC2: Configuration System
- [ ] `BaKCConfig` dataclass is fully implemented
- [ ] YAML config files parse without errors
- [ ] Config validation catches invalid values (negative nu, alpha > 1, etc.)
- [ ] `configs/cardio.yaml` contains correct baseline parameters
- [ ] Configuration prints readable summary when loaded

### AC3: Logging System
- [ ] Logger initializes without errors
- [ ] Log messages write to both console and file
- [ ] Log levels (DEBUG, INFO, WARNING, ERROR) work correctly
- [ ] Log format is readable and includes timestamps
- [ ] Log files rotate properly when size limit reached

### AC4: Data Module
- [ ] `DataLoader` successfully loads CARDIO dataset
- [ ] Column 'Class' is renamed to 'y' automatically
- [ ] `DataValidator` catches invalid data (NaN, wrong types, non-binary y)
- [ ] Inliers and outliers are split correctly (expect 1,655 inliers, 176 outliers for CARDIO)
- [ ] Data shapes match expected values

### AC5: Testing
- [ ] All unit tests pass (`pytest tests/unit/`)
- [ ] Code coverage >80% for Phase 1 modules
- [ ] Tests run in isolation (no interdependencies)
- [ ] Test data fixtures are created and reusable

### AC6: Documentation
- [ ] All modules have docstrings
- [ ] All public functions have type hints
- [ ] README.md updated with installation instructions
- [ ] Phase 1 implementation matches this specification document

---

## Definition of Done

Phase 1 is considered **DONE** when:

1. ✅ **All Acceptance Criteria are met** - Every item in AC1-AC6 is checked and validated

2. ✅ **Package Installation Verified**
   ```bash
   pip install -e .
   python -c "from bakc_plus.config import BaKCConfig; print('Success')"
   ```

3. ✅ **Configuration Loading Verified**
   ```bash
   python -c "from bakc_plus.config import BaKCConfig; cfg = BaKCConfig.from_yaml('configs/cardio.yaml'); print(cfg.data.dataset_name)"
   # Output: "cardio"
   ```

4. ✅ **Data Loading Verified**
   ```bash
   python -c "from bakc_plus.data.loader import DataLoader; loader = DataLoader('./data/input'); df = loader.load_dataset('cardio'); print(len(df))"
   # Output: "1831"
   ```

5. ✅ **Unit Tests Pass**
   ```bash
   pytest tests/unit/ -v --cov=src/bakc_plus --cov-report=term-missing
   # All tests pass, coverage >80%
   ```

6. ✅ **Logging Verified**
   ```bash
   python -c "from bakc_plus.logger import get_logger; logger = get_logger('test'); logger.info('Test message'); logger.warning('Test warning')"
   # Logs appear in console with proper formatting
   ```

7. ✅ **No Gaps in Issue Log** - All issues identified during AC validation are resolved

8. ✅ **Code Review** - Code is clean, follows PEP 8, and matches design specifications

9. ✅ **Git Commit** - All changes committed with clear commit messages

10. ✅ **Documentation Complete** - This document, all step documents, and README.md are up-to-date

---

## Issue Log

*Issues discovered during AC validation will be tracked here*

| ID | Date | Issue Description | Resolution | Status |
|----|------|-------------------|------------|--------|
| - | - | - | - | - |

---

## Implementation Notes

### Critical Considerations

1. **Path Handling**
   - Use `pathlib.Path` for cross-platform compatibility
   - All paths configurable via YAML
   - Default paths relative to project root

2. **Reproducibility**
   - Configuration must support `random_state` parameter
   - Must be passed through to all random operations
   - Document all sources of randomness

3. **Error Handling**
   - Fail fast with clear error messages
   - Validate inputs at boundaries (config load, data load)
   - Use custom exceptions where appropriate

4. **Testing Strategy**
   - Unit tests for each module in isolation
   - Use fixtures for common test data
   - Mock file I/O where appropriate
   - Test both success and failure paths

### References

- **REFACTORING_PLAN.md**: Overall architecture and design
- **EXECUTION_SUMMARY.md**: Baseline performance targets
- **NOTEBOOK_ANALYSIS.md**: Current code analysis and issues
- **requirements.txt**: Existing dependencies

---

## Next Steps

1. Create Step 1.1 documentation with detailed AC and DoD
2. Implement Step 1.1 tasks
3. Validate Step 1.1 against AC and DoD
4. Repeat for Steps 1.2, 1.3, 1.4
5. Validate entire Phase 1 against this document's AC and DoD
6. Commit and push Phase 1 completion

---

**Document Version**: 1.0
**Created**: 2025-11-18
**Last Updated**: 2025-11-18
**Status**: Initial Draft - Ready for Implementation
