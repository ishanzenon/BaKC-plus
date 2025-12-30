"""
Pytest configuration and shared fixtures for BaKC-plus tests
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def test_data_dir():
    """Return path to test data directory"""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def synthetic_dataset():
    """
    Create a small synthetic dataset for testing

    Returns:
        pd.DataFrame with features and binary target 'y'
    """
    np.random.seed(42)

    # Generate 100 inliers
    inliers = np.random.randn(100, 5)

    # Generate 10 outliers (shifted distribution)
    outliers = np.random.randn(10, 5) + 3

    # Combine
    X = np.vstack([inliers, outliers])
    y = np.array([0] * 100 + [1] * 10)

    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'V{i+1}' for i in range(5)])
    df['y'] = y

    return df


@pytest.fixture
def sample_config_dict():
    """
    Return a sample configuration dictionary

    Returns:
        dict with configuration parameters
    """
    return {
        'data': {
            'dataset_name': 'test',
            'data_dir': './data/input',
            'output_dir': './output',
            'train_fraction': 0.5,
        },
        'model': {
            'nu': 0.05,
            'kernel': 'rbf',
        },
        'ensemble': {
            'num_models': 5,
            'num_test_splits': 20,
            'num_repetitions': 5,
            'random_state': 42,
        },
        'conformal': {
            'alpha': 0.05,
            'scoring_method': 'sigmoid',
        },
    }


@pytest.fixture(scope="session")
def temp_output_dir(tmp_path_factory):
    """
    Create a temporary output directory for tests

    Returns:
        Path to temporary directory
    """
    return tmp_path_factory.mktemp("output")
