#!/usr/bin/env python3
"""
Demonstrate logging functionality

This script demonstrates the BaKC-plus logging system including:
- Different log levels (DEBUG, INFO, WARNING, ERROR)
- Console and file output
- Integration with config module
- Structured log format
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bakc_plus import setup_logging, get_logger
from bakc_plus.config import BaKCConfig


def main():
    """Main demo function"""
    print("=" * 60)
    print("BaKC-plus Logging Demonstration")
    print("=" * 60)
    print()

    # Setup logging with DEBUG level to see all messages
    print("1. Setting up logging (DEBUG level, file logging enabled)")
    print("   Log file: output/logs/bakc_plus.log")
    print()
    setup_logging(log_level="DEBUG", enable_file_logging=True)

    # Get logger for this module
    logger = get_logger(__name__)

    # Demonstrate different log levels
    print("2. Demonstrating different log levels:")
    print()

    logger.debug("This is a DEBUG message - detailed diagnostic information")
    logger.info("This is an INFO message - general informational message")
    logger.warning("This is a WARNING message - indication of potential issues")
    logger.error("This is an ERROR message - a more serious problem occurred")

    print()
    print("   Note: Console shows INFO and above, file shows DEBUG and above")
    print()

    # Load and validate config (generates logs)
    print("3. Loading configuration (generates logs):")
    print()

    config = BaKCConfig.from_yaml('configs/cardio.yaml')
    config.validate()

    print()
    print("4. Configuration loaded successfully!")
    print(f"   Dataset: {config.data.dataset_name}")
    print(f"   nu: {config.model.nu}")
    print(f"   num_models: {config.ensemble.num_models}")
    print(f"   alpha: {config.conformal.alpha}")
    print()

    # More log examples
    print("5. Additional logging examples:")
    print()

    logger.info("Starting model training simulation...")
    logger.debug("Training model 1/5 with 100 samples")
    logger.debug("Training model 2/5 with 100 samples")
    logger.warning("Model 3/5 took longer than expected (12.5s)")
    logger.debug("Training model 4/5 with 100 samples")
    logger.debug("Training model 5/5 with 100 samples")
    logger.info("Model training complete")

    print()
    print("6. Checking log file:")
    log_file = Path("output/logs/bakc_plus.log")
    if log_file.exists():
        print(f"   ✅ Log file created: {log_file}")
        print(f"   File size: {log_file.stat().st_size} bytes")
        print()
        print("   Last 10 lines of log file:")
        print("   " + "-" * 56)
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(f"   {line.rstrip()}")
        print("   " + "-" * 56)
    else:
        print(f"   ❌ Log file not found: {log_file}")

    print()
    print("=" * 60)
    print("Logging demonstration complete!")
    print()
    print("Key takeaways:")
    print("  - Logs appear in both console (INFO+) and file (DEBUG+)")
    print("  - Structured format: [timestamp] [level] [module] message")
    print("  - Log rotation happens at 10MB (keeps last 5 files)")
    print("  - All modules use same logger via get_logger(__name__)")
    print("=" * 60)


if __name__ == "__main__":
    main()
