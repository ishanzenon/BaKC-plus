"""
Data validation functionality for BaKC-plus

This module provides classes and functions for validating dataset schema and values.
Ensures data integrity before training and evaluation.
"""

from typing import List, Optional
import pandas as pd
import numpy as np

from ..logger import get_logger


class DataValidator:
    """
    Validate dataset schema and values

    This class performs comprehensive validation of datasets including
    schema checks (columns, types) and value checks (NaN, ranges, target values).

    Attributes:
        logger: Logger instance for structured logging

    Example:
        >>> validator = DataValidator()
        >>> validator.validate_complete(df, expected_features=21, target_col='y')
        True
    """

    def __init__(self):
        """Initialize validator with logger"""
        self.logger = get_logger(__name__)

    def validate_schema(
        self,
        df: pd.DataFrame,
        expected_features: int,
        target_col: str = 'y'
    ) -> bool:
        """
        Validate dataset schema

        Checks:
        - DataFrame is not None and not empty
        - Correct number of columns (features + target)
        - Target column exists
        - All feature columns exist

        Args:
            df: DataFrame to validate
            expected_features: Expected number of feature columns
            target_col: Name of target column (default: 'y')

        Returns:
            True if validation passes

        Raises:
            ValueError: If schema validation fails with descriptive message

        Example:
            >>> validator = DataValidator()
            >>> df = pd.DataFrame({'V1': [1, 2], 'y': [0, 1]})
            >>> validator.validate_schema(df, expected_features=1, target_col='y')
            True
        """
        self.logger.debug(f"Validating schema: expected_features={expected_features}, target_col='{target_col}'")

        # Check DataFrame is not None
        if df is None:
            self.logger.error("DataFrame is None")
            raise ValueError("DataFrame cannot be None")

        # Check DataFrame is not empty
        if df.empty:
            self.logger.error("DataFrame is empty")
            raise ValueError("DataFrame is empty (no rows)")

        # Check correct number of columns
        expected_cols = expected_features + 1  # features + target
        actual_cols = len(df.columns)

        if actual_cols != expected_cols:
            self.logger.error(
                f"Column count mismatch: expected {expected_cols} ({expected_features} features + 1 target), "
                f"got {actual_cols}"
            )
            raise ValueError(
                f"Invalid number of columns: expected {expected_cols} "
                f"({expected_features} features + 1 target), got {actual_cols}\n"
                f"Columns present: {list(df.columns)}"
            )

        # Check target column exists
        if target_col not in df.columns:
            self.logger.error(f"Target column '{target_col}' not found in DataFrame")
            raise ValueError(
                f"Target column '{target_col}' not found in dataset\n"
                f"Available columns: {list(df.columns)}"
            )

        # Check feature columns
        feature_cols = [col for col in df.columns if col != target_col]
        if len(feature_cols) != expected_features:
            self.logger.error(
                f"Feature count mismatch: expected {expected_features}, "
                f"got {len(feature_cols)} after excluding target"
            )
            raise ValueError(
                f"Invalid number of feature columns: expected {expected_features}, "
                f"got {len(feature_cols)}\n"
                f"Feature columns: {feature_cols}"
            )

        self.logger.debug(f"Schema validation passed: {actual_cols} columns, {len(df)} rows")
        return True

    def validate_values(
        self,
        df: pd.DataFrame,
        target_col: str = 'y',
        check_feature_ranges: bool = False,
        feature_min: float = -10.0,
        feature_max: float = 10.0
    ) -> bool:
        """
        Validate dataset values

        Checks:
        - No NaN values in any column
        - Target column is binary (0 or 1 only)
        - Feature values are numeric
        - Feature values within acceptable range (optional)

        Args:
            df: DataFrame to validate
            target_col: Name of target column (default: 'y')
            check_feature_ranges: Whether to check feature value ranges
            feature_min: Minimum acceptable feature value (if checking ranges)
            feature_max: Maximum acceptable feature value (if checking ranges)

        Returns:
            True if validation passes

        Raises:
            ValueError: If value validation fails with descriptive message

        Example:
            >>> validator = DataValidator()
            >>> df = pd.DataFrame({'V1': [0.5, -0.5], 'y': [0, 1]})
            >>> validator.validate_values(df, target_col='y')
            True
        """
        self.logger.debug(f"Validating values: target_col='{target_col}', check_ranges={check_feature_ranges}")

        # Check for NaN values
        nan_counts = df.isna().sum()
        total_nans = nan_counts.sum()

        if total_nans > 0:
            nan_cols = nan_counts[nan_counts > 0].to_dict()
            self.logger.error(f"Found {total_nans} NaN values in dataset")
            raise ValueError(
                f"Dataset contains {total_nans} NaN values\n"
                f"NaN counts by column: {nan_cols}\n"
                f"Please clean the data before proceeding."
            )

        # Check target column is binary
        unique_targets = df[target_col].unique()
        expected_targets = {0, 1}

        if not set(unique_targets).issubset(expected_targets):
            self.logger.error(f"Invalid target values: {unique_targets}")
            raise ValueError(
                f"Target column '{target_col}' must contain only 0 or 1\n"
                f"Found unique values: {sorted(unique_targets)}"
            )

        # Log target distribution
        target_counts = df[target_col].value_counts().sort_index()
        self.logger.debug(f"Target distribution: {target_counts.to_dict()}")

        # Check feature columns are numeric
        feature_cols = [col for col in df.columns if col != target_col]

        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                self.logger.error(f"Feature column '{col}' is not numeric: {df[col].dtype}")
                raise ValueError(
                    f"Feature column '{col}' must be numeric, got dtype: {df[col].dtype}"
                )

        # Optional: Check feature value ranges
        if check_feature_ranges:
            for col in feature_cols:
                col_min = df[col].min()
                col_max = df[col].max()

                if col_min < feature_min or col_max > feature_max:
                    self.logger.warning(
                        f"Feature '{col}' has values outside expected range "
                        f"[{feature_min}, {feature_max}]: min={col_min:.2f}, max={col_max:.2f}"
                    )
                    # Note: This is a warning, not an error, as normalized features
                    # may legitimately exceed typical ranges

        self.logger.debug(f"Value validation passed: {len(feature_cols)} features validated")
        return True

    def validate_complete(
        self,
        df: pd.DataFrame,
        expected_features: int = 21,
        target_col: str = 'y',
        check_feature_ranges: bool = False
    ) -> bool:
        """
        Run all validation checks

        This is a convenience method that runs both schema and value validation.

        Args:
            df: DataFrame to validate
            expected_features: Expected number of feature columns (default: 21 for CARDIO)
            target_col: Name of target column (default: 'y')
            check_feature_ranges: Whether to check feature value ranges

        Returns:
            True if all validation passes

        Raises:
            ValueError: If any validation check fails

        Example:
            >>> validator = DataValidator()
            >>> df = load_cardio()
            >>> validator.validate_complete(df, expected_features=21)
            True
        """
        self.logger.info(f"Running complete validation: {expected_features} features, target='{target_col}'")

        # Run schema validation
        self.validate_schema(df, expected_features, target_col)

        # Run value validation
        self.validate_values(df, target_col, check_feature_ranges)

        self.logger.info("Complete validation passed")
        return True


def validate_dataset(
    df: pd.DataFrame,
    expected_features: int = 21,
    target_col: str = 'y'
) -> bool:
    """
    Convenience function to validate a dataset

    This is a simpler interface for validation when you don't need
    to reuse the validator instance.

    Args:
        df: DataFrame to validate
        expected_features: Expected number of feature columns
        target_col: Name of target column

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails

    Example:
        >>> df = load_cardio()
        >>> validate_dataset(df, expected_features=21)
        True
    """
    validator = DataValidator()
    return validator.validate_complete(df, expected_features, target_col)
