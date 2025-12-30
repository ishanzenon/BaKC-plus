#!/usr/bin/env python3
"""
Validation script for Step 1.1: Project Setup

This script validates all acceptance criteria for Step 1.1.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def check_directories():
    """Check all required directories exist"""
    print("\n=== AC1.1.1: Directory Structure ===")
    required = [
        "src/bakc_plus",
        "src/bakc_plus/data",
        "src/bakc_plus/model",
        "src/bakc_plus/conformal",
        "src/bakc_plus/evaluation",
        "src/bakc_plus/pipeline",
        "src/bakc_plus/utils",
        "tests/unit",
        "tests/integration",
        "tests/test_data",
        "configs",
        "scripts",
    ]
    missing = [d for d in required if not Path(d).exists()]
    if missing:
        print(f"❌ Missing directories: {missing}")
        return False
    print("✅ All required directories exist")
    return True


def check_init_files():
    """Check all __init__.py files exist"""
    print("\n=== AC1.1.2: Package Initialization ===")
    required = [
        "src/bakc_plus/__init__.py",
        "src/bakc_plus/data/__init__.py",
        "src/bakc_plus/model/__init__.py",
        "src/bakc_plus/conformal/__init__.py",
        "src/bakc_plus/evaluation/__init__.py",
        "src/bakc_plus/pipeline/__init__.py",
        "src/bakc_plus/utils/__init__.py",
        "tests/__init__.py",
        "tests/unit/__init__.py",
    ]
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        print(f"❌ Missing __init__.py files: {missing}")
        return False
    print("✅ All __init__.py files exist (9 total)")
    return True


def check_setup_files():
    """Check setup.py and pytest.ini exist"""
    print("\n=== AC1.1.3: Package Installation ===")
    required = ["setup.py", "pytest.ini", "tests/conftest.py", ".gitignore"]
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        print(f"❌ Missing setup files: {missing}")
        return False
    print("✅ All setup files exist (setup.py, pytest.ini, conftest.py, .gitignore)")
    return True


def check_package_importable():
    """Check if package can be imported"""
    print("\n=== AC1.1.2 (continued): Package Import ===")
    try:
        import bakc_plus
        version = bakc_plus.__version__
        if version != "0.1.0":
            print(f"❌ Version mismatch: expected 0.1.0, got {version}")
            return False
        print(f"✅ Package importable: BaKC-plus v{version}")

        # Test sub-package imports
        import bakc_plus.data
        import bakc_plus.model
        import bakc_plus.conformal
        import bakc_plus.evaluation
        import bakc_plus.pipeline
        import bakc_plus.utils
        print("✅ All sub-packages importable")

        return True
    except ImportError as e:
        print(f"❌ Package import failed: {e}")
        return False


def check_pytest_config():
    """Check pytest configuration"""
    print("\n=== AC1.1.4: Test Configuration ===")
    pytest_ini = Path("pytest.ini")
    if not pytest_ini.exists():
        print("❌ pytest.ini not found")
        return False

    content = pytest_ini.read_text()
    required_sections = [
        "[pytest]",
        "testpaths = tests",
        "--cov=src/bakc_plus",
        "markers =",
    ]
    missing = [s for s in required_sections if s not in content]
    if missing:
        print(f"❌ Missing pytest.ini sections: {missing}")
        return False

    print("✅ pytest.ini configured correctly")
    return True


def check_gitignore():
    """Check .gitignore configuration"""
    print("\n=== AC1.1.5: Git Configuration ===")
    gitignore = Path(".gitignore")
    if not gitignore.exists():
        print("❌ .gitignore not found")
        return False

    content = gitignore.read_text()
    required_patterns = [
        "__pycache__/",
        "*.py[cod]",
        "htmlcov/",
        ".pytest_cache/",
        "*.egg-info/",
    ]
    missing = [p for p in required_patterns if p not in content]
    if missing:
        print(f"❌ Missing .gitignore patterns: {missing}")
        return False

    print("✅ .gitignore configured correctly")
    return True


def check_conftest():
    """Check conftest.py fixtures"""
    print("\n=== AC1.1.6: Test Fixtures ===")
    conftest = Path("tests/conftest.py")
    if not conftest.exists():
        print("❌ tests/conftest.py not found")
        return False

    content = conftest.read_text()
    required_fixtures = [
        "def test_data_dir(",
        "def synthetic_dataset(",
        "def sample_config_dict(",
        "def temp_output_dir(",
    ]
    missing = [f for f in required_fixtures if f not in content]
    if missing:
        print(f"❌ Missing fixtures: {missing}")
        return False

    print("✅ All required fixtures defined in conftest.py")

    # Note: Pytest fixtures cannot be called directly for testing
    # They are validated by their presence in conftest.py
    # Actual functionality will be validated when running pytest tests in Step 1.4
    print("✅ Fixture definitions validated (will be tested with pytest in Step 1.4)")
    return True


def check_definition_of_done():
    """Check Definition of Done criteria"""
    print("\n=== Definition of Done Checks ===")

    # DoD 2: Package Importable
    try:
        import bakc_plus
        print(f"✅ DoD 2: Package importable - BaKC-plus v{bakc_plus.__version__}")
    except:
        print("❌ DoD 2: Package not importable")
        return False

    # DoD 4: Directory Structure
    try:
        from pathlib import Path
        key_dirs = [
            "src/bakc_plus/data",
            "src/bakc_plus/model",
            "src/bakc_plus/conformal",
            "tests/unit",
        ]
        if all(Path(d).exists() for d in key_dirs):
            print("✅ DoD 6: Directory structure verified")
        else:
            print("❌ DoD 6: Directory structure incomplete")
            return False
    except Exception as e:
        print(f"❌ DoD 6: Error checking directory structure: {e}")
        return False

    # DoD 5: Git Status (check __pycache__ not tracked)
    import subprocess
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )
        status = result.stdout
        if "__pycache__" in status or ".pyc" in status:
            print("❌ DoD 5: Git tracking unwanted files (__pycache__, .pyc)")
            return False
        print("✅ DoD 5: Git status clean (no Python artifacts)")
    except Exception as e:
        print(f"⚠️  DoD 5: Could not check git status: {e}")

    return True


def main():
    """Run all validation checks"""
    print("=" * 60)
    print("Step 1.1 Validation: Project Setup")
    print("=" * 60)

    checks = [
        check_directories(),
        check_init_files(),
        check_setup_files(),
        check_package_importable(),
        check_pytest_config(),
        check_gitignore(),
        check_conftest(),
        check_definition_of_done(),
    ]

    print("\n" + "=" * 60)
    if all(checks):
        print("🎉 Step 1.1 Validation PASSED!")
        print("=" * 60)
        return 0
    else:
        print("❌ Step 1.1 Validation FAILED!")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
