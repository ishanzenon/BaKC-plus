# Step 1.3: Logging System

**Parent**: Phase 1 - Core Infrastructure
**Timeline**: Days 5-6
**Status**: In Progress
**Dependencies**: Step 1.1 (Project Setup) ✅, Step 1.2 (Configuration System) ✅

---

## Overview

Step 1.3 implements a structured logging system using Python's built-in `logging` module to replace print statements and tqdm progress bars with proper logging. This enables debugging, monitoring, and provides configurable log levels.

### Context from Existing Analysis

From **NOTEBOOK_ANALYSIS.md** (Section 6.6):
> **Weak Logging/Monitoring**
> - No progress tracking beyond tqdm
> - Hard to debug if something fails
> - **FIX**: Add logging module

From **ocsvm_x_cv_x_bagging.py**:
- No logging imports or setup
- Uses `print()` for debugging (lines 82-83)
- Uses `tqdm` for progress bars only
- No structured logging for errors or warnings

### Current State

**Logging in Notebook**:
- Print statements scattered throughout
- tqdm progress bars for loops
- No log files
- No log levels (DEBUG, INFO, WARNING, ERROR)
- No timestamps or structured format

---

## Goals and Objectives

### Primary Goals

1. **Replace Print Statements**
   - Convert all print() to logger.info() or logger.debug()
   - Maintain same information output
   - Add log levels for filtering

2. **Structured Logging**
   - Consistent format: `[timestamp] [level] [module] message`
   - Support multiple log levels
   - Enable/disable debug logging via config

3. **File Logging**
   - Write logs to `output/logs/bakc_plus.log`
   - Rotate log files (keep last 5 files, 10MB each)
   - Separate console and file output

4. **Integration with Config**
   - Log level configurable
   - Log file path configurable
   - Enable/disable file logging

### Success Metrics

- ✅ No print() statements in production code
- ✅ Logging configuration works
- ✅ Logs written to file and console
- ✅ Log rotation works
- ✅ Unit tests pass (>80% coverage for logger module)

---

## Detailed Requirements

### Logging Configuration

**Log Format**:
```
[2025-11-18 12:00:00,123] [INFO] [bakc_plus.config] Configuration loaded from configs/cardio.yaml
[2025-11-18 12:00:00,456] [DEBUG] [bakc_plus.model.ocsvm] Fitting OC-SVM member 0 for fold 1
[2025-11-18 12:00:01,789] [WARNING] [bakc_plus.conformal] Calibration scores have high variance: std=0.15
[2025-11-18 12:00:02,012] [ERROR] [bakc_plus.data.loader] Failed to load dataset 'missing.csv': FileNotFoundError
```

**Log Levels**:
- **DEBUG**: Detailed information for debugging (e.g., individual model training)
- **INFO**: General informational messages (e.g., config loaded, training started)
- **WARNING**: Warning messages (e.g., high variance, non-critical issues)
- **ERROR**: Error messages (e.g., file not found, validation failed)
- **CRITICAL**: Critical errors (e.g., system failures)

**Log Destinations**:
1. **Console**: INFO and above (colored output if possible)
2. **File**: DEBUG and above (all messages)

**Log File Management**:
- Path: `output/logs/bakc_plus.log`
- Rotation: When file reaches 10MB
- Backup count: Keep last 5 files
- Naming: `bakc_plus.log`, `bakc_plus.log.1`, `bakc_plus.log.2`, etc.

---

## Task Breakdown

### Task 1.3.1: Create logger.py Module

**Objective**: Implement logging setup and configuration

**Implementation**:
1. Create `src/bakc_plus/logger.py`
2. Implement `setup_logging(log_level, log_file, enable_file_logging)`
3. Implement `get_logger(name)` function
4. Configure formatters for console and file
5. Set up rotating file handler
6. Set up console handler with colors (optional)

**Logger Module Structure**:
```python
import logging
import logging.handlers
from pathlib import Path
from typing import Optional

# Default format
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    enable_file_logging: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Setup logging configuration for BaKC-plus

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (default: output/logs/bakc_plus.log)
        enable_file_logging: Whether to enable file logging
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
    """
    # ... implementation

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
```

**Validation**:
- Logger can be initialized with different levels
- Logs appear in console and file
- Log format is correct
- File rotation works when size exceeds limit

### Task 1.3.2: Update Package __init__ to Export Logger

**Objective**: Make logger accessible from package level

**Implementation**:
1. Update `src/bakc_plus/__init__.py`
2. Import `setup_logging` and `get_logger`
3. Add to `__all__` exports

**Code**:
```python
from .logger import setup_logging, get_logger

__all__ = [
    # ... existing exports
    "setup_logging",
    "get_logger",
]
```

**Validation**:
- Can import: `from bakc_plus import get_logger`
- Can import: `from bakc_plus.logger import setup_logging`

### Task 1.3.3: Add Logging to Config Module

**Objective**: Demonstrate logging usage in existing module

**Implementation**:
1. Add logger to `src/bakc_plus/config.py`
2. Log configuration loading
3. Log validation results
4. Log any warnings during validation

**Example Usage**:
```python
from .logger import get_logger

logger = get_logger(__name__)

class BaKCConfig:
    @classmethod
    def from_yaml(cls, path: str) -> 'BaKCConfig':
        logger.info(f"Loading configuration from {path}")
        # ... load config
        logger.debug(f"Loaded config with {len(config_dict)} sections")
        return config

    def validate(self) -> None:
        logger.debug("Validating configuration")
        # ... validation checks
        logger.info("Configuration validation passed")
```

**Validation**:
- Config loading logs appear
- Validation logs appear
- No print statements remain

### Task 1.3.4: Create Example Usage Script

**Objective**: Demonstrate logging in action

**Implementation**:
1. Create `scripts/demo_logging.py`
2. Show different log levels
3. Show file and console output
4. Show log rotation

**Script Content**:
```python
#!/usr/bin/env python3
"""Demonstrate logging functionality"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bakc_plus import setup_logging, get_logger
from bakc_plus.config import BaKCConfig

# Setup logging
setup_logging(log_level="DEBUG", enable_file_logging=True)

# Get logger
logger = get_logger(__name__)

# Demonstrate different log levels
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")

# Load config (will generate logs)
config = BaKCConfig.from_yaml('configs/cardio.yaml')
config.validate()

logger.info("Logging demonstration complete")
```

**Validation**:
- Script runs without errors
- Logs appear in console
- Logs appear in file
- Different log levels visible

### Task 1.3.5: Write Unit Tests

**Objective**: Comprehensive tests for logging module

**Test File**: `tests/unit/test_logger.py`

**Test Cases**:
1. `test_setup_logging_default()` - Test default logging setup
2. `test_setup_logging_custom_level()` - Test custom log level
3. `test_setup_logging_with_file()` - Test file logging
4. `test_setup_logging_without_file()` - Test console-only logging
5. `test_get_logger()` - Test logger creation
6. `test_log_levels()` - Test different log levels work
7. `test_logger_hierarchy()` - Test module logger hierarchy
8. `test_log_file_creation()` - Test log file is created
9. `test_log_file_rotation()` - Test log rotation when size exceeded
10. `test_log_format()` - Test log message format
11. `test_multiple_loggers()` - Test multiple module loggers
12. `test_logger_configuration_persistence()` - Test config persists across calls

**Example Test**:
```python
def test_setup_logging_with_file(temp_output_dir):
    """Test logging setup with file output"""
    log_file = temp_output_dir / "test.log"

    setup_logging(
        log_level="DEBUG",
        log_file=log_file,
        enable_file_logging=True
    )

    logger = get_logger("test")
    logger.info("Test message")

    # Check log file created
    assert log_file.exists()

    # Check log file contains message
    content = log_file.read_text()
    assert "Test message" in content
    assert "[INFO]" in content
```

**Validation**:
- All tests pass
- Coverage >80% for logger.py
- Tests are isolated and reproducible

### Task 1.3.6: Update Configuration to Support Logging Settings

**Objective**: Add logging configuration options

**Implementation**:
1. Add `LoggingConfig` dataclass to config.py
2. Add logging settings to YAML files
3. Integrate with `setup_logging()`

**New Dataclass**:
```python
@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    enable_file_logging: bool = True
    log_file: Optional[Path] = None
    max_log_size_mb: int = 10
    backup_count: int = 5

    def __post_init__(self):
        if self.log_file is not None and not isinstance(self.log_file, Path):
            self.log_file = Path(self.log_file)
```

**YAML Addition**:
```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  enable_file_logging: true
  log_file: null  # null = output/logs/bakc_plus.log
  max_log_size_mb: 10
  backup_count: 5
```

**Validation**:
- Config loads with logging settings
- Logging settings validate correctly
- Can configure log level and file location

---

## Acceptance Criteria

### AC1.3.1: Logger Module Implementation
- [ ] `src/bakc_plus/logger.py` created
- [ ] `setup_logging()` function implemented with parameters
- [ ] `get_logger()` function implemented
- [ ] Log format includes timestamp, level, module name, message
- [ ] Console handler configured
- [ ] File handler with rotation configured
- [ ] All functions have docstrings and type hints

### AC1.3.2: Package Integration
- [ ] Logger exported from `bakc_plus.__init__`
- [ ] Can import: `from bakc_plus import setup_logging, get_logger`
- [ ] Config module uses logger instead of print
- [ ] No print() statements in production code (except CLI scripts)

### AC1.3.3: File Logging
- [ ] Logs written to `output/logs/bakc_plus.log` by default
- [ ] Log directory created if doesn't exist
- [ ] Log rotation works (creates .1, .2, etc. files)
- [ ] Keeps only specified number of backup files
- [ ] File logging can be disabled via config

### AC1.3.4: Console Logging
- [ ] Logs appear in console with proper formatting
- [ ] Log level is configurable (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- [ ] Console shows INFO and above by default
- [ ] Console output is readable and well-formatted

### AC1.3.5: Configuration Integration
- [ ] `LoggingConfig` dataclass added to config.py
- [ ] Logging settings in default.yaml and cardio.yaml
- [ ] Logging config validates correctly
- [ ] Can configure log level, file path, rotation settings

### AC1.3.6: Unit Tests
- [ ] Test file `tests/unit/test_logger.py` created
- [ ] At least 12 test cases implemented
- [ ] All tests pass
- [ ] Code coverage >80% for logger.py
- [ ] Tests cover file logging, console logging, rotation

### AC1.3.7: Example Usage
- [ ] Demo script `scripts/demo_logging.py` created
- [ ] Script demonstrates all log levels
- [ ] Script shows file and console output
- [ ] Script runs without errors

---

## Definition of Done

Step 1.3 is considered **DONE** when:

1. ✅ **All Acceptance Criteria Met** - Every item in AC1.3.1 through AC1.3.7 validated

2. ✅ **Logging Works**
   ```bash
   python scripts/demo_logging.py
   # Logs appear in console
   # Logs written to output/logs/bakc_plus.log
   ```

3. ✅ **Config Module Uses Logger**
   ```bash
   python -c "import sys; sys.path.insert(0, 'src'); from bakc_plus import setup_logging, BaKCConfig; setup_logging('DEBUG'); cfg = BaKCConfig.from_yaml('configs/cardio.yaml')"
   # Should show: [timestamp] [INFO] [bakc_plus.config] Loading configuration from...
   ```

4. ✅ **Unit Tests Pass**
   ```bash
   PYTHONPATH=src pytest tests/unit/test_logger.py -v
   # All tests pass, coverage >80%
   ```

5. ✅ **Log File Created and Rotates**
   - Log file exists at `output/logs/bakc_plus.log`
   - When size exceeds limit, rotation creates backup files
   - Old backup files are deleted when exceeding backup_count

6. ✅ **No Print Statements**
   - No print() in src/bakc_plus/*.py (except __init__ version display)
   - All information output via logger

7. ✅ **Documentation Complete**
   - Logger module has comprehensive docstrings
   - Example usage documented
   - Log format documented

8. ✅ **No Issues in Issue Log** - All issues resolved

9. ✅ **Code Committed and Pushed** - All changes committed

10. ✅ **Step Document Updated** - This document reflects actual implementation

---

## Issue Log

| ID | Date | Issue Description | Resolution | Status |
|----|------|-------------------|------------|--------|
| - | - | - | - | - |

---

## Implementation Notes

### Design Decisions

1. **Use Built-in logging**: Python's logging module is robust and standard
2. **Rotating File Handler**: Prevents log files from growing unbounded
3. **Two Handlers**: Separate console and file for different levels/formats
4. **Module-level Loggers**: Each module gets its own logger via `get_logger(__name__)`
5. **Configurable**: All settings available via config for flexibility

### Log Level Guidelines

- **DEBUG**: Detailed diagnostic information (e.g., "Fitting model 3/5 for fold 2/3")
- **INFO**: Confirmation that things are working (e.g., "Configuration loaded", "Training complete")
- **WARNING**: Indication of potential issues (e.g., "High FDR detected: 12%")
- **ERROR**: A more serious problem (e.g., "Failed to load dataset")
- **CRITICAL**: A very serious error (e.g., "Out of memory, cannot continue")

### Integration Points

- Config module: Log configuration loading and validation
- Future modules (data, model, conformal): Log operations and progress
- CLI scripts: Setup logging at entry point

### Performance Considerations

- File I/O for logging is minimal overhead
- Rotation only happens when threshold reached
- DEBUG level can be disabled in production for performance

---

## Next Steps

After Step 1.3 is DONE:
1. Validate against all AC
2. Run validation script
3. Ensure zero issues in issue log
4. Commit changes
5. Update Phase 1 progress
6. Move to Step 1.4 (Data Module)

---

**Document Version**: 1.0
**Created**: 2025-11-18
**Status**: Ready for Implementation
