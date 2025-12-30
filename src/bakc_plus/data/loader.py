"""
Data loading functionality for BaKC-plus

This module provides classes and functions for loading datasets from CSV files.
Supports configuration-based path handling and robust error checking.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

from ..config import DataConfig
from ..logger import get_logger


class DataLoader:
    """
    Load datasets from CSV files

    This class handles loading CSV datasets with configurable paths,
    automatic file discovery, and comprehensive error handling.

    Attributes:
        data_dir: Path to directory containing dataset files
        logger: Logger instance for structured logging

    Example:
        >>> from bakc_plus import DataConfig
        >>> config = DataConfig(dataset_name="cardio")
        >>> loader = DataLoader(config)
        >>> df = loader.load_dataset("cardio")
        >>> print(df.shape)
        (1831, 22)
    """

    def __init__(self, config: DataConfig):
        """
        Initialize loader with configuration

        Args:
            config: DataConfig instance with data_dir setting
        """
        self.data_dir = Path(config.data_dir)
        self.logger = get_logger(__name__)

        self.logger.debug(f"DataLoader initialized with data_dir={self.data_dir}")

    def load_dataset(
        self,
        dataset_name: str,
        filename: Optional[str] = None,
        header: Optional[int] = None,
        names: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Load dataset from CSV file

        This method loads a CSV file from the configured data directory.
        It performs file existence checking and basic validation.

        Args:
            dataset_name: Dataset identifier (e.g., 'cardio', 'gamma')
            filename: CSV filename (default: {dataset_name}.csv)
            header: CSV header row number (default: None for no header)
            names: List of column names to assign (default: None)

        Returns:
            DataFrame with all columns and rows from the CSV

        Raises:
            FileNotFoundError: If CSV file doesn't exist at expected path
            ValueError: If CSV is empty or cannot be parsed
            pd.errors.ParserError: If CSV format is invalid

        Example:
            >>> loader = DataLoader(config)
            >>> df = loader.load_dataset("cardio")  # Loads cardio.csv
            >>> df = loader.load_dataset("my_data", filename="custom.csv")
        """
        # Determine filename
        if filename is None:
            filename = f"{dataset_name}.csv"

        # Construct full path
        file_path = self.data_dir / filename

        self.logger.info(f"Loading dataset '{dataset_name}' from {file_path}")

        # Check file exists
        if not file_path.exists():
            self.logger.error(f"Dataset file not found: {file_path}")
            raise FileNotFoundError(
                f"Dataset file not found: {file_path}\n"
                f"Expected location: {self.data_dir}\n"
                f"Please ensure the CSV file exists at the specified path."
            )

        # Check file is readable
        if not file_path.is_file():
            self.logger.error(f"Path exists but is not a file: {file_path}")
            raise ValueError(f"Path is not a file: {file_path}")

        # Load CSV
        try:
            self.logger.debug(f"Reading CSV with header={header}, names={names}")
            df = pd.read_csv(file_path, header=header, names=names)

            # Validate non-empty
            if df.empty:
                self.logger.error(f"Loaded DataFrame is empty: {file_path}")
                raise ValueError(f"CSV file is empty: {file_path}")

            self.logger.info(
                f"Successfully loaded dataset '{dataset_name}': "
                f"{df.shape[0]} rows, {df.shape[1]} columns"
            )
            self.logger.debug(f"Columns: {list(df.columns)}")
            self.logger.debug(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

            return df

        except pd.errors.EmptyDataError as e:
            self.logger.error(f"CSV file is empty: {file_path}")
            raise ValueError(f"CSV file contains no data: {file_path}") from e

        except pd.errors.ParserError as e:
            self.logger.error(f"Failed to parse CSV file: {file_path}, error: {e}")
            raise ValueError(
                f"Invalid CSV format in {file_path}: {e}\n"
                f"Please check the file format and encoding."
            ) from e

        except Exception as e:
            self.logger.error(f"Unexpected error loading dataset: {e}")
            raise ValueError(
                f"Failed to load dataset '{dataset_name}' from {file_path}: {e}"
            ) from e

    def load_cardio(self) -> pd.DataFrame:
        """
        Load CARDIO dataset with specific configuration

        The CARDIO dataset is a cardiac arrhythmia dataset with 21 features
        (V1-V21) and a binary target (y) indicating outliers.

        File format:
        - No header row
        - 22 columns: V1, V2, ..., V21, y
        - 1,831 total rows
        - y values: 0 (inlier) or 1 (outlier)

        Returns:
            DataFrame with columns ['V1', ..., 'V21', 'y']

        Raises:
            FileNotFoundError: If cardio.csv not found
            ValueError: If data doesn't match expected format

        Example:
            >>> loader = DataLoader(config)
            >>> df = loader.load_cardio()
            >>> print(df.shape)
            (1831, 22)
            >>> print(df['y'].value_counts())
            0    1655
            1     176
        """
        # Define column names for CARDIO dataset
        feature_cols = [f'V{i}' for i in range(1, 22)]  # V1 through V21
        column_names = feature_cols + ['y']

        self.logger.info("Loading CARDIO dataset")

        # Load with specific configuration
        df = self.load_dataset(
            dataset_name="cardio",
            filename="cardio.csv",
            header=None,  # No header in CARDIO CSV
            names=column_names
        )

        # Validate expected shape
        expected_rows = 1831
        expected_cols = 22

        if df.shape != (expected_rows, expected_cols):
            self.logger.warning(
                f"CARDIO dataset shape mismatch: expected ({expected_rows}, {expected_cols}), "
                f"got {df.shape}"
            )

        # Log dataset statistics
        num_inliers = (df['y'] == 0).sum()
        num_outliers = (df['y'] == 1).sum()

        self.logger.info(
            f"CARDIO dataset loaded: {num_inliers} inliers, {num_outliers} outliers "
            f"({100.0 * num_outliers / len(df):.2f}% outlier rate)"
        )

        return df


def load_dataset_from_config(config: DataConfig, dataset_name: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to load dataset using configuration

    This is a simpler interface for loading datasets when you have a config object.

    Args:
        config: DataConfig with data_dir and dataset_name
        dataset_name: Optional override for config.dataset_name

    Returns:
        Loaded DataFrame

    Example:
        >>> config = BaKCConfig.from_yaml('configs/cardio.yaml')
        >>> df = load_dataset_from_config(config.data)
    """
    loader = DataLoader(config)
    name = dataset_name or config.dataset_name

    # Use specific loader for known datasets
    if name.lower() == 'cardio':
        return loader.load_cardio()
    else:
        return loader.load_dataset(name)
