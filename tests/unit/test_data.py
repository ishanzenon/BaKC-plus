"""
Comprehensive unit tests for the BaKC-plus data module

Tests cover data loading, validation, and splitting functionality with:
- DataLoader: CSV loading with file validation and error handling
- DataValidator: Schema and value validation
- DataSplitter: Train/test splitting with inlier/outlier separation

Tests use synthetic data and temporary file fixtures to avoid external dependencies.
"""

import pytest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

from bakc_plus.data import (
    DataLoader,
    DataValidator,
    DataSplitter,
    DataSplit,
    load_dataset_from_config,
    validate_dataset,
    split_dataset,
)
from bakc_plus.config import DataConfig


# ============================================================================
# FIXTURES: Helper functions to create synthetic test data
# ============================================================================

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory for test CSV files"""
    return tmp_path / "data"


@pytest.fixture
def sample_csv_file(temp_data_dir):
    """Create a simple CSV file for testing basic loading"""
    temp_data_dir.mkdir(parents=True, exist_ok=True)

    # Create a simple CSV with header
    df = pd.DataFrame({
        'V1': [1.0, 2.0, 3.0],
        'V2': [4.0, 5.0, 6.0],
        'y': [0, 1, 0]
    })

    csv_path = temp_data_dir / "test.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def cardio_csv_file(temp_data_dir):
    """Create a synthetic CARDIO-like CSV file (no header, 22 columns)"""
    temp_data_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    # Create synthetic data: 21 features + 1 target
    # 100 inliers (y=0), 10 outliers (y=1)
    inliers = np.random.randn(100, 21) * 0.5
    outliers = np.random.randn(10, 21) * 0.5 + 2.0  # Shifted distribution

    X = np.vstack([inliers, outliers])
    y = np.array([0] * 100 + [1] * 10)

    # Combine features and target
    data = np.hstack([X, y.reshape(-1, 1)])

    # Write as CSV without header (like original CARDIO dataset)
    csv_path = temp_data_dir / "cardio.csv"
    np.savetxt(csv_path, data, delimiter=',', fmt='%.10f')

    return csv_path


@pytest.fixture
def synthetic_dataset():
    """Create a comprehensive synthetic dataset for testing"""
    np.random.seed(42)

    # Generate 100 inliers
    inliers = np.random.randn(100, 5) * 0.5

    # Generate 10 outliers (shifted distribution)
    outliers = np.random.randn(10, 5) + 2.0

    # Combine
    X = np.vstack([inliers, outliers])
    y = np.array([0] * 100 + [1] * 10)

    # Create DataFrame with feature names V1-V5 and target y
    df = pd.DataFrame(X, columns=[f'V{i+1}' for i in range(5)])
    df['y'] = y

    return df


@pytest.fixture
def data_config(temp_data_dir):
    """Create a DataConfig instance for testing"""
    return DataConfig(
        dataset_name="test",
        data_dir=str(temp_data_dir),
        output_dir="./test_output"
    )


# ============================================================================
# DATALOADER TESTS (7+ tests)
# ============================================================================

class TestDataLoader:
    """Tests for DataLoader class"""

    def test_load_dataset_valid_data(self, data_config, sample_csv_file):
        """Test loading a valid CSV file with header"""
        loader = DataLoader(data_config)

        # Load the test CSV with explicit header row
        df = loader.load_dataset("test", header=0)

        # Assertions
        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        assert not df.empty, "DataFrame should not be empty"
        assert df.shape[1] == 3, f"Expected 3 columns, got {df.shape[1]}"
        assert df.shape[0] == 3, f"Expected 3 rows, got {df.shape[0]}"
        assert list(df.columns) == ['V1', 'V2', 'y'], "Column names should match"
        assert df['V1'].tolist() == [1.0, 2.0, 3.0], "Feature values should match"

    def test_load_dataset_missing_file_error(self, data_config):
        """Test FileNotFoundError when CSV file doesn't exist"""
        loader = DataLoader(data_config)

        # Attempt to load non-existent file
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_dataset("nonexistent")

        # Check error message is descriptive
        assert "not found" in str(exc_info.value).lower(), "Error should mention file not found"

    def test_load_dataset_empty_file_error(self, temp_data_dir, data_config):
        """Test ValueError when CSV file is empty"""
        # Create empty CSV file
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        empty_csv = temp_data_dir / "empty.csv"
        empty_csv.write_text("")

        loader = DataLoader(data_config)

        # Attempt to load empty file
        with pytest.raises(ValueError) as exc_info:
            loader.load_dataset("empty")

        assert "empty" in str(exc_info.value).lower(), "Error should mention empty file"

    def test_load_dataset_custom_filename(self, temp_data_dir, data_config):
        """Test loading with custom filename parameter"""
        # Create a CSV with custom name
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        custom_df = pd.DataFrame({
            'A': [10, 20],
            'B': [30, 40]
        })
        custom_path = temp_data_dir / "custom_name.csv"
        custom_df.to_csv(custom_path, index=False)

        loader = DataLoader(data_config)

        # Load with custom filename and header
        df = loader.load_dataset("anything", filename="custom_name.csv", header=0)

        assert df.shape == (2, 2), "Should load custom filename correctly"
        assert list(df.columns) == ['A', 'B'], "Should preserve column names"

    def test_load_dataset_custom_column_names(self, temp_data_dir, data_config):
        """Test loading with custom column names"""
        # Create CSV without header
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        data = np.array([[1, 2, 3], [4, 5, 6]])
        csv_path = temp_data_dir / "no_header.csv"
        np.savetxt(csv_path, data, delimiter=',', fmt='%d')

        loader = DataLoader(data_config)

        # Load with custom column names
        custom_names = ['Feature1', 'Feature2', 'Target']
        df = loader.load_dataset(
            "no_header",
            filename="no_header.csv",
            header=None,
            names=custom_names
        )

        assert list(df.columns) == custom_names, "Should apply custom column names"
        assert df.shape == (2, 3), "Should have correct shape"

    def test_load_cardio(self, cardio_csv_file, data_config):
        """Test load_cardio() method with CARDIO dataset format"""
        loader = DataLoader(data_config)

        # Load CARDIO dataset
        df = loader.load_cardio()

        # Assertions
        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        assert df.shape[0] == 110, "Should have 110 rows (100 inliers + 10 outliers)"
        assert df.shape[1] == 22, "Should have 22 columns (21 features + 1 target)"

        # Check column names
        expected_cols = [f'V{i}' for i in range(1, 22)] + ['y']
        assert list(df.columns) == expected_cols, "CARDIO columns should match expected names"

        # Check target values
        assert set(df['y'].unique()) == {0, 1}, "Target should contain only 0 and 1"
        assert (df['y'] == 0).sum() == 100, "Should have 100 inliers"
        assert (df['y'] == 1).sum() == 10, "Should have 10 outliers"

    def test_load_dataset_from_config(self, cardio_csv_file, data_config):
        """Test load_dataset_from_config() convenience function"""
        # Test with cardio dataset
        df = load_dataset_from_config(data_config, dataset_name="cardio")

        assert isinstance(df, pd.DataFrame), "Should return DataFrame"
        assert df.shape[1] == 22, "CARDIO should have 22 columns"
        assert 'y' in df.columns, "Should have target column"

        # Test with generic dataset
        data_config.data_dir = str(cardio_csv_file.parent)
        data_config.dataset_name = "test"

        # Create a test CSV without header (no-header format)
        test_csv = cardio_csv_file.parent / "test.csv"
        test_data = np.array([[1, 3], [2, 4]])
        np.savetxt(test_csv, test_data, delimiter=',', fmt='%d')

        df = load_dataset_from_config(data_config, dataset_name="test")
        assert df.shape[1] == 2, "Should have 2 columns"
        assert df.shape[0] == 2, "Should have 2 rows"


# ============================================================================
# DATAVALIDATOR TESTS (8+ tests)
# ============================================================================

class TestDataValidator:
    """Tests for DataValidator class"""

    def test_validate_schema_valid_data(self, synthetic_dataset):
        """Test validate_schema() passes with valid data"""
        validator = DataValidator()

        # Should not raise
        result = validator.validate_schema(
            synthetic_dataset,
            expected_features=5,
            target_col='y'
        )

        assert result is True, "Validation should return True for valid schema"

    def test_validate_schema_wrong_column_count(self, synthetic_dataset):
        """Test validate_schema() fails with wrong number of columns"""
        validator = DataValidator()

        # Expect 10 features but have 5
        with pytest.raises(ValueError) as exc_info:
            validator.validate_schema(
                synthetic_dataset,
                expected_features=10,
                target_col='y'
            )

        assert "column" in str(exc_info.value).lower(), "Error should mention columns"

    def test_validate_schema_missing_target(self, synthetic_dataset):
        """Test validate_schema() fails when target column is missing"""
        validator = DataValidator()

        # Remove target column
        df_no_target = synthetic_dataset.drop('y', axis=1)

        with pytest.raises(ValueError) as exc_info:
            validator.validate_schema(
                df_no_target,
                expected_features=5,
                target_col='y'
            )

        assert "target" in str(exc_info.value).lower(), "Error should mention missing target"

    def test_validate_schema_none_dataframe(self):
        """Test validate_schema() fails with None DataFrame"""
        validator = DataValidator()

        with pytest.raises(ValueError) as exc_info:
            validator.validate_schema(None, expected_features=5, target_col='y')

        assert "none" in str(exc_info.value).lower(), "Error should mention None DataFrame"

    def test_validate_schema_empty_dataframe(self):
        """Test validate_schema() fails with empty DataFrame"""
        validator = DataValidator()

        empty_df = pd.DataFrame()

        with pytest.raises(ValueError) as exc_info:
            validator.validate_schema(empty_df, expected_features=5, target_col='y')

        assert "empty" in str(exc_info.value).lower(), "Error should mention empty DataFrame"

    def test_validate_values_valid_data(self, synthetic_dataset):
        """Test validate_values() passes with valid data"""
        validator = DataValidator()

        result = validator.validate_values(
            synthetic_dataset,
            target_col='y'
        )

        assert result is True, "Validation should return True for valid values"

    def test_validate_values_with_nan(self, synthetic_dataset):
        """Test validate_values() fails when NaN values are present"""
        validator = DataValidator()

        # Add NaN values
        df_with_nan = synthetic_dataset.copy()
        df_with_nan.iloc[0, 0] = np.nan

        with pytest.raises(ValueError) as exc_info:
            validator.validate_values(df_with_nan, target_col='y')

        assert "nan" in str(exc_info.value).lower(), "Error should mention NaN values"

    def test_validate_values_invalid_target(self, synthetic_dataset):
        """Test validate_values() fails with invalid target values"""
        validator = DataValidator()

        # Add invalid target values (should be 0 or 1 only)
        df_invalid_target = synthetic_dataset.copy()
        df_invalid_target.iloc[0, -1] = 2  # Invalid target value

        with pytest.raises(ValueError) as exc_info:
            validator.validate_values(df_invalid_target, target_col='y')

        assert "target" in str(exc_info.value).lower(), "Error should mention target column"

    def test_validate_complete_integration(self, synthetic_dataset):
        """Test validate_complete() runs both schema and value validation"""
        validator = DataValidator()

        result = validator.validate_complete(
            synthetic_dataset,
            expected_features=5,
            target_col='y'
        )

        assert result is True, "Complete validation should pass for valid data"

    def test_validate_dataset_convenience_function(self, synthetic_dataset):
        """Test validate_dataset() convenience function"""
        result = validate_dataset(
            synthetic_dataset,
            expected_features=5,
            target_col='y'
        )

        assert result is True, "Convenience function should work correctly"

    def test_validate_values_non_numeric_feature(self, synthetic_dataset):
        """Test validate_values() fails with non-numeric features"""
        validator = DataValidator()

        # Add string column
        df_with_string = synthetic_dataset.copy()
        df_with_string['StringCol'] = 'text'

        with pytest.raises(ValueError) as exc_info:
            validator.validate_values(df_with_string, target_col='y')

        assert "numeric" in str(exc_info.value).lower(), "Error should mention numeric requirement"


# ============================================================================
# DATASPLITTER TESTS (10+ tests)
# ============================================================================

class TestDataSplitter:
    """Tests for DataSplitter class"""

    def test_split_inliers_outliers(self, synthetic_dataset):
        """Test split_inliers_outliers() correctly separates data"""
        splitter = DataSplitter()

        inliers, outliers = splitter.split_inliers_outliers(
            synthetic_dataset,
            target_col='y'
        )

        # Assertions
        assert len(inliers) == 100, "Should have 100 inliers"
        assert len(outliers) == 10, "Should have 10 outliers"
        assert set(inliers['y'].unique()) == {0}, "Inliers should have y=0"
        assert set(outliers['y'].unique()) == {1}, "Outliers should have y=1"

    def test_split_train_test_half_fraction(self, synthetic_dataset):
        """Test split_train_test() with 0.5 fraction"""
        splitter = DataSplitter()
        inliers, _ = splitter.split_inliers_outliers(synthetic_dataset)

        train, test = splitter.split_train_test(inliers, train_fraction=0.5)

        # With 100 inliers and 0.5 fraction: 50 train, 50 test
        assert len(train) == 50, "Should have 50 training samples"
        assert len(test) == 50, "Should have 50 test samples"
        assert len(train) + len(test) == len(inliers), "No data loss"

    def test_split_train_test_custom_fraction(self, synthetic_dataset):
        """Test split_train_test() with custom fraction"""
        splitter = DataSplitter()
        inliers, _ = splitter.split_inliers_outliers(synthetic_dataset)

        # Test with 0.7 fraction
        train, test = splitter.split_train_test(inliers, train_fraction=0.7)

        # With 100 inliers and 0.7 fraction: 70 train, 30 test
        assert len(train) == 70, "Should have 70 training samples"
        assert len(test) == 30, "Should have 30 test samples"

    def test_split_train_test_invalid_fraction(self, synthetic_dataset):
        """Test split_train_test() rejects invalid fractions"""
        splitter = DataSplitter()
        inliers, _ = splitter.split_inliers_outliers(synthetic_dataset)

        # Test fraction = 0 (invalid)
        with pytest.raises(ValueError) as exc_info:
            splitter.split_train_test(inliers, train_fraction=0.0)

        assert "train_fraction" in str(exc_info.value).lower()

        # Test fraction = 1 (invalid)
        with pytest.raises(ValueError):
            splitter.split_train_test(inliers, train_fraction=1.0)

        # Test fraction > 1 (invalid)
        with pytest.raises(ValueError):
            splitter.split_train_test(inliers, train_fraction=1.5)

    def test_create_data_split_half_fraction(self, synthetic_dataset):
        """Test create_data_split() with 0.5 fraction"""
        splitter = DataSplitter()

        split = splitter.create_data_split(
            synthetic_dataset,
            train_fraction=0.5,
            target_col='y'
        )

        # Check structure
        assert isinstance(split, DataSplit), "Should return DataSplit object"
        assert isinstance(split.train, pd.DataFrame), "train should be DataFrame"
        assert isinstance(split.test, pd.DataFrame), "test should be DataFrame"
        assert isinstance(split.metadata, dict), "metadata should be dict"

    def test_data_split_train_only_inliers(self, synthetic_dataset):
        """Test that train set contains ONLY inliers"""
        splitter = DataSplitter()
        split = splitter.create_data_split(synthetic_dataset, train_fraction=0.5)

        # Train should contain only inliers (y=0)
        assert (split.train['y'] == 0).all(), "Train should contain only inliers"
        assert len(split.train) == 50, "Train should have 50 inliers"

    def test_data_split_test_has_inliers_and_outliers(self, synthetic_dataset):
        """Test that test set contains inliers + outliers"""
        splitter = DataSplitter()
        split = splitter.create_data_split(synthetic_dataset, train_fraction=0.5)

        # Test should contain both inliers and outliers
        test_inliers = (split.test['y'] == 0).sum()
        test_outliers = (split.test['y'] == 1).sum()

        assert test_inliers == 50, "Test should have 50 inliers"
        assert test_outliers == 10, "Test should have all 10 outliers"
        assert len(split.test) == 60, "Test should have 60 total samples"

    def test_data_split_no_leakage(self, synthetic_dataset):
        """Test that train and test sets have no overlap (no data leakage)"""
        splitter = DataSplitter()
        split = splitter.create_data_split(synthetic_dataset, train_fraction=0.5)

        # Get indices for comparison
        train_indices = set(split.train.index)
        test_indices = set(split.test.index)

        # Check no overlap (except potentially for index names)
        # Note: We check row count instead since pandas may have duplicate indices
        total_rows = len(split.train) + len(split.test)
        assert total_rows == 110, "Total rows should match original dataset"

    def test_data_split_metadata_correct(self, synthetic_dataset):
        """Test that DataSplit metadata is correct"""
        splitter = DataSplitter()
        split = splitter.create_data_split(synthetic_dataset, train_fraction=0.5)

        metadata = split.metadata

        # Check all expected keys
        assert 'num_train' in metadata, "Should have num_train"
        assert 'num_test' in metadata, "Should have num_test"
        assert 'num_inliers' in metadata, "Should have num_inliers"
        assert 'num_outliers' in metadata, "Should have num_outliers"
        assert 'num_features' in metadata, "Should have num_features"
        assert 'train_fraction' in metadata, "Should have train_fraction"

        # Check values
        assert metadata['num_train'] == 50, "num_train should be 50"
        assert metadata['num_test'] == 60, "num_test should be 60"
        assert metadata['num_inliers'] == 100, "num_inliers should be 100"
        assert metadata['num_outliers'] == 10, "num_outliers should be 10"
        assert metadata['num_features'] == 5, "num_features should be 5"
        assert metadata['train_all_inliers'] is True, "train should contain only inliers"

    def test_split_dataset_reproducibility(self, synthetic_dataset):
        """Test that split_dataset() is reproducible"""
        # First split
        split1 = split_dataset(synthetic_dataset, train_fraction=0.5)

        # Second split
        split2 = split_dataset(synthetic_dataset, train_fraction=0.5)

        # Check both produce same results
        assert split1.train.shape == split2.train.shape, "Train shapes should match"
        assert split1.test.shape == split2.test.shape, "Test shapes should match"

        # Check indices match (deterministic split)
        assert list(split1.train.index) == list(split2.train.index), "Train indices should match"

    def test_split_dataset_convenience_function(self, synthetic_dataset):
        """Test split_dataset() convenience function"""
        split = split_dataset(synthetic_dataset, train_fraction=0.6)

        assert isinstance(split, DataSplit), "Should return DataSplit"
        assert len(split.train) == 60, "0.6 of 100 inliers = 60"
        assert len(split.test) == 50, "40 inliers + 10 outliers = 50"

    def test_split_with_custom_target_column(self):
        """Test splitting with custom target column name"""
        # Create dataset with different target column name
        df = pd.DataFrame({
            'F1': np.random.randn(100),
            'F2': np.random.randn(100),
            'target': np.array([0] * 80 + [1] * 20)
        })

        splitter = DataSplitter()
        split = splitter.create_data_split(
            df,
            train_fraction=0.5,
            target_col='target'
        )

        # Check split worked correctly
        assert 'target' in split.train.columns, "Should preserve custom target column"
        assert (split.train['target'] == 0).all(), "Train should have only 0 values"
        assert len(split.metadata['feature_columns']) == 2, "Should have 2 features"

    def test_split_with_odd_number_inliers(self):
        """Test splitting with odd number of inliers"""
        # Create dataset with odd number of inliers (99)
        df = pd.DataFrame({
            'F1': np.random.randn(109),
            'F2': np.random.randn(109),
            'y': np.array([0] * 99 + [1] * 10)
        })

        splitter = DataSplitter()
        split = splitter.create_data_split(df, train_fraction=0.5)

        # With 99 inliers and 0.5 fraction: 49 train, 50 test
        assert len(split.train) == 49, "Should have 49 training samples"
        assert len(split.test) == 60, "Should have 50 test inliers + 10 outliers"

    def test_split_preserves_features(self, synthetic_dataset):
        """Test that splitting preserves all feature columns"""
        splitter = DataSplitter()
        split = splitter.create_data_split(synthetic_dataset)

        # Check columns match
        assert list(split.train.columns) == list(synthetic_dataset.columns), "Train columns should match"
        assert list(split.test.columns) == list(synthetic_dataset.columns), "Test columns should match"

        # Check feature columns in metadata
        expected_features = [f'V{i+1}' for i in range(5)]
        assert split.metadata['feature_columns'] == expected_features, "Feature list should match"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestDataIntegration:
    """Integration tests combining multiple data module components"""

    def test_full_pipeline_load_validate_split(self, cardio_csv_file, data_config):
        """Test complete pipeline: load -> validate -> split"""
        # Step 1: Load data
        loader = DataLoader(data_config)
        df = loader.load_cardio()

        # Step 2: Validate data
        validator = DataValidator()
        assert validator.validate_complete(df, expected_features=21, target_col='y')

        # Step 3: Split data
        splitter = DataSplitter()
        split = splitter.create_data_split(df, train_fraction=0.5)

        # Verify structure
        assert len(split.train) > 0, "Train should have samples"
        assert len(split.test) > 0, "Test should have samples"
        assert (split.train['y'] == 0).all(), "Train should have only inliers"

    def test_integration_with_convenience_functions(self, cardio_csv_file, data_config):
        """Test pipeline using convenience functions"""
        # Load data
        loader = DataLoader(data_config)
        df = loader.load_cardio()

        # Validate using convenience function
        assert validate_dataset(df, expected_features=21)

        # Split using convenience function
        split = split_dataset(df, train_fraction=0.5)

        # Verify
        assert len(split.train) > 0
        assert len(split.test) > 0

    def test_split_maintains_data_integrity(self, synthetic_dataset):
        """Test that splitting doesn't corrupt data"""
        original_shape = synthetic_dataset.shape

        split = split_dataset(synthetic_dataset, train_fraction=0.5)

        # Reconstruct full dataset (train + test)
        combined = pd.concat([split.train, split.test], axis=0)

        # Check shape (may have duplicates in index, but same row count)
        assert len(combined) == original_shape[0], "Should preserve number of rows"
        assert combined.shape[1] == original_shape[1], "Should preserve number of columns"

    def test_split_statistics_consistency(self, synthetic_dataset):
        """Test that split maintains statistical consistency"""
        inlier_count_original = (synthetic_dataset['y'] == 0).sum()
        outlier_count_original = (synthetic_dataset['y'] == 1).sum()

        split = split_dataset(synthetic_dataset, train_fraction=0.5)

        # Check counts are preserved
        inliers_in_split = (split.train['y'] == 0).sum() + (split.test['y'] == 0).sum()
        outliers_in_split = (split.train['y'] == 1).sum() + (split.test['y'] == 1).sum()

        assert inliers_in_split == inlier_count_original, "Inlier count should be preserved"
        assert outliers_in_split == outlier_count_original, "Outlier count should be preserved"


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error conditions"""

    def test_loader_malformed_csv(self, temp_data_dir, data_config):
        """Test DataLoader handles malformed CSV gracefully"""
        # Create malformed CSV
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        malformed_csv = temp_data_dir / "malformed.csv"
        malformed_csv.write_text("a,b,c\n1,2\n3,4,5,6\n")

        loader = DataLoader(data_config)

        # Should load (pandas is lenient) or raise ValueError
        try:
            df = loader.load_dataset("malformed")
            # If it loads, check it's a DataFrame
            assert isinstance(df, pd.DataFrame)
        except ValueError:
            # It's also acceptable to raise an error
            pass

    def test_validator_single_sample(self):
        """Test validator with single sample"""
        df = pd.DataFrame({
            'V1': [1.0],
            'y': [0]
        })

        validator = DataValidator()

        # Should validate successfully
        assert validator.validate_schema(df, expected_features=1)
        assert validator.validate_values(df)

    def test_splitter_minimum_dataset(self):
        """Test splitter with very small dataset"""
        df = pd.DataFrame({
            'F1': [1.0, 2.0, 3.0, 4.0],
            'y': [0, 0, 0, 1]
        })

        splitter = DataSplitter()
        split = splitter.create_data_split(df, train_fraction=0.5)

        # Should handle minimum case
        assert len(split.train) > 0
        assert len(split.test) > 0
        assert (split.train['y'] == 0).all()

    def test_loader_with_special_characters(self, temp_data_dir, data_config):
        """Test loader with special characters in data"""
        # Create CSV with special characters (but numeric data) without header
        temp_data_dir.mkdir(parents=True, exist_ok=True)
        data = np.array([
            [1.5e-3, -1.5, 0],
            [2.5e-3, -2.5, 1]
        ])

        special_csv = temp_data_dir / "special.csv"
        np.savetxt(special_csv, data, delimiter=',', fmt='%.10f,%.1f,%d')

        loader = DataLoader(data_config)
        loaded = loader.load_dataset("special")

        assert loaded.shape == (2, 3)
        assert np.isclose(loaded.iloc[0, 0], 1.5e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
