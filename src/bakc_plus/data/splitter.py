"""
Data splitting functionality for BaKC-plus

This module provides classes and functions for splitting datasets into
train/calibration/test sets while preserving the exact methodology from
the original notebook.
"""

from typing import NamedTuple, Tuple, Dict, Any
import pandas as pd

from ..logger import get_logger


class DataSplit(NamedTuple):
    """
    Result of train/test split

    This named tuple encapsulates the results of splitting a dataset
    into training and test sets, along with metadata about the split.

    Attributes:
        train: Training inliers only (y=0)
        test: Test inliers + all outliers
        inliers: All inliers before splitting (for reference)
        outliers: All outliers (included in test set)
        metadata: Dictionary with split statistics

    Example:
        >>> split = DataSplit(
        ...     train=train_df,
        ...     test=test_df,
        ...     inliers=all_inliers,
        ...     outliers=all_outliers,
        ...     metadata={'num_train': 827, 'num_test': 1004}
        ... )
        >>> print(f"Training samples: {len(split.train)}")
    """
    train: pd.DataFrame
    test: pd.DataFrame
    inliers: pd.DataFrame
    outliers: pd.DataFrame
    metadata: Dict[str, Any]


class DataSplitter:
    """
    Split data into train/test sets

    This class implements the data splitting logic from the original notebook,
    preserving the exact methodology to ensure reproducible results.

    The splitting process:
    1. Separate inliers (y=0) and outliers (y=1)
    2. Split inliers by train_fraction (default: 0.5)
    3. Create train set (training inliers only)
    4. Create test set (test inliers + all outliers)

    Attributes:
        logger: Logger instance for structured logging

    Example:
        >>> splitter = DataSplitter()
        >>> split = splitter.create_data_split(df, train_fraction=0.5)
        >>> print(f"Train: {len(split.train)}, Test: {len(split.test)}")
    """

    def __init__(self):
        """Initialize splitter with logger"""
        self.logger = get_logger(__name__)

    def split_inliers_outliers(
        self,
        df: pd.DataFrame,
        target_col: str = 'y'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separate inliers (y=0) and outliers (y=1)

        This method separates the dataset into normal samples (inliers) and
        anomalous samples (outliers) based on the target column.

        Args:
            df: DataFrame with features and target column
            target_col: Name of target column (default: 'y')

        Returns:
            Tuple of (inliers_df, outliers_df)

        Example:
            >>> splitter = DataSplitter()
            >>> inliers, outliers = splitter.split_inliers_outliers(df)
            >>> print(f"Inliers: {len(inliers)}, Outliers: {len(outliers)}")
        """
        self.logger.debug(f"Splitting inliers/outliers by target column '{target_col}'")

        # Separate based on target value
        inliers = df[df[target_col] == 0].copy()
        outliers = df[df[target_col] == 1].copy()

        self.logger.info(
            f"Split into {len(inliers)} inliers ({100.0 * len(inliers) / len(df):.2f}%) "
            f"and {len(outliers)} outliers ({100.0 * len(outliers) / len(df):.2f}%)"
        )

        return inliers, outliers

    def split_train_test(
        self,
        inliers: pd.DataFrame,
        train_fraction: float = 0.5
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split inliers into train and test

        This method splits the inlier samples into training and testing subsets.
        The split is deterministic (no shuffling) to preserve notebook behavior.

        CRITICAL: This preserves the exact methodology from the notebook:
        - len_train = len(inliers) // 2  (integer division)
        - train = inliers[:len_train]  (first half)
        - test = inliers[len_train:]  (second half)

        Args:
            inliers: DataFrame with only inliers (y=0)
            train_fraction: Fraction for training (0.0-1.0, default: 0.5)

        Returns:
            Tuple of (train_inliers, test_inliers)

        Example (CARDIO):
            >>> # Input: 1,655 inliers, train_fraction=0.5
            >>> train, test = splitter.split_train_test(inliers, 0.5)
            >>> # Output: (827 train, 828 test)

        Note:
            With train_fraction=0.5 and odd number of inliers,
            test will have one more sample than train.
        """
        if not (0.0 < train_fraction < 1.0):
            raise ValueError(
                f"train_fraction must be in (0, 1), got {train_fraction}"
            )

        self.logger.debug(
            f"Splitting {len(inliers)} inliers with train_fraction={train_fraction}"
        )

        # Calculate split point (integer division as in notebook)
        len_train = int(len(inliers) * train_fraction)

        # Split without shuffling (preserves notebook behavior)
        train_inliers = inliers[:len_train].copy()
        test_inliers = inliers[len_train:].copy()

        self.logger.info(
            f"Split inliers: {len(train_inliers)} train, {len(test_inliers)} test "
            f"({100.0 * len(train_inliers) / len(inliers):.2f}% train)"
        )

        return train_inliers, test_inliers

    def create_data_split(
        self,
        df: pd.DataFrame,
        train_fraction: float = 0.5,
        target_col: str = 'y'
    ) -> DataSplit:
        """
        Create train/test split from full dataset

        This is the main splitting function that orchestrates the complete
        splitting process:
        1. Separate inliers and outliers
        2. Split inliers into train/test
        3. Combine test inliers with all outliers
        4. Generate metadata about the split

        Args:
            df: Full dataset with features and target
            train_fraction: Fraction of inliers for training (default: 0.5)
            target_col: Name of target column (default: 'y')

        Returns:
            DataSplit object with train, test, and metadata

        Raises:
            ValueError: If train_fraction is invalid

        Example:
            >>> splitter = DataSplitter()
            >>> split = splitter.create_data_split(df, train_fraction=0.5)
            >>> print(f"Train shape: {split.train.shape}")
            >>> print(f"Test shape: {split.test.shape}")
            >>> print(f"Metadata: {split.metadata}")

        Notes:
            - Train set contains ONLY inliers (for OC-SVM training)
            - Test set contains inliers AND outliers (for evaluation)
            - No shuffling is performed (preserves notebook behavior)
        """
        self.logger.info(
            f"Creating data split: {len(df)} samples, "
            f"train_fraction={train_fraction}, target_col='{target_col}'"
        )

        # Step 1: Separate inliers and outliers
        inliers, outliers = self.split_inliers_outliers(df, target_col)

        # Step 2: Split inliers into train and test
        train_inliers, test_inliers = self.split_train_test(inliers, train_fraction)

        # Step 3: Combine test inliers with all outliers
        # This matches notebook behavior: test set = test inliers + all outliers
        test = pd.concat([test_inliers, outliers], axis=0, ignore_index=False)

        self.logger.info(
            f"Final split: train={len(train_inliers)} inliers, "
            f"test={len(test)} ({len(test_inliers)} inliers + {len(outliers)} outliers)"
        )

        # Step 4: Generate metadata
        num_features = len(df.columns) - 1  # Exclude target column
        feature_cols = [col for col in df.columns if col != target_col]

        metadata = {
            'total_samples': len(df),
            'num_features': num_features,
            'feature_columns': feature_cols,
            'target_column': target_col,
            'train_fraction': train_fraction,
            'num_train': len(train_inliers),
            'num_test': len(test),
            'num_inliers': len(inliers),
            'num_outliers': len(outliers),
            'num_test_inliers': len(test_inliers),
            'num_test_outliers': len(outliers),
            'train_all_inliers': True,  # Train contains only inliers
            'test_outlier_rate': 100.0 * len(outliers) / len(test),
        }

        self.logger.debug(f"Split metadata: {metadata}")

        return DataSplit(
            train=train_inliers,
            test=test,
            inliers=inliers,
            outliers=outliers,
            metadata=metadata
        )


def split_dataset(
    df: pd.DataFrame,
    train_fraction: float = 0.5,
    target_col: str = 'y'
) -> DataSplit:
    """
    Convenience function to split a dataset

    This is a simpler interface for splitting when you don't need to reuse
    the splitter instance.

    Args:
        df: DataFrame to split
        train_fraction: Fraction of inliers for training
        target_col: Name of target column

    Returns:
        DataSplit object

    Example:
        >>> df = load_cardio()
        >>> split = split_dataset(df, train_fraction=0.5)
        >>> print(split.metadata)
    """
    splitter = DataSplitter()
    return splitter.create_data_split(df, train_fraction, target_col)
