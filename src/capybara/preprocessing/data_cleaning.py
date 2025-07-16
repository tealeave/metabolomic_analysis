"""
Data cleaning utilities for metabolomic analysis.

This module provides functions for cleaning and preprocessing metabolomic intensity data
using PySpark for distributed processing.
"""

import logging
from typing import List, Optional, Tuple

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, isnan, isnull, when, count, mean, stddev, abs as spark_abs
from pyspark.sql.types import DoubleType

logger = logging.getLogger(__name__)


def clean_intensity_data(
    dataframe: DataFrame,
    intensity_columns: List[str],
    missing_threshold: float = 0.5,
    zero_threshold: float = 0.3,
    outlier_method: str = "iqr",
    outlier_threshold: float = 3.0
) -> DataFrame:
    """
    Clean intensity data by removing low-quality features and samples.
    
    Args:
        dataframe: Input DataFrame with intensity data
        intensity_columns: List of column names containing intensity values
        missing_threshold: Maximum fraction of missing values allowed per feature
        zero_threshold: Maximum fraction of zero values allowed per feature
        outlier_method: Method for outlier detection ("iqr" or "zscore")
        outlier_threshold: Threshold for outlier detection
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Starting data cleaning process")
    
    # Remove features with too many missing values
    cleaned_df = remove_low_quality_features(
        dataframe, intensity_columns, missing_threshold, zero_threshold
    )
    
    # Handle missing values in remaining features
    cleaned_df = handle_missing_values(cleaned_df, intensity_columns)
    
    # Remove outliers
    if outlier_method:
        cleaned_df = remove_outliers(
            cleaned_df, intensity_columns, method=outlier_method, threshold=outlier_threshold
        )
    
    logger.info("Data cleaning completed")
    return cleaned_df


def remove_low_quality_features(
    dataframe: DataFrame,
    intensity_columns: List[str],
    missing_threshold: float = 0.5,
    zero_threshold: float = 0.3
) -> Tuple[DataFrame, List[str]]:
    """
    Remove features with too many missing or zero values.
    
    Args:
        dataframe: Input DataFrame
        intensity_columns: List of intensity column names
        missing_threshold: Maximum fraction of missing values allowed
        zero_threshold: Maximum fraction of zero values allowed
        
    Returns:
        Tuple of (cleaned DataFrame, list of removed columns)
    """
    logger.info("Removing low-quality features")
    
    total_rows = dataframe.count()
    removed_columns = []
    
    for col_name in intensity_columns:
        # Count missing values
        missing_count = dataframe.filter(
            col(col_name).isNull() | isnan(col(col_name))
        ).count()
        
        # Count zero values
        zero_count = dataframe.filter(col(col_name) == 0.0).count()
        
        missing_fraction = missing_count / total_rows
        zero_fraction = zero_count / total_rows
        
        if missing_fraction > missing_threshold or zero_fraction > zero_threshold:
            removed_columns.append(col_name)
            logger.debug(f"Removing {col_name}: {missing_fraction:.2%} missing, {zero_fraction:.2%} zero")
    
    # Remove columns
    if removed_columns:
        remaining_columns = [c for c in dataframe.columns if c not in removed_columns]
        cleaned_df = dataframe.select(*remaining_columns)
        logger.info(f"Removed {len(removed_columns)} low-quality features")
    else:
        cleaned_df = dataframe
        logger.info("No features removed")
    
    return cleaned_df


def handle_missing_values(
    dataframe: DataFrame,
    intensity_columns: List[str],
    method: str = "median",
    group_by_cols: Optional[List[str]] = None
) -> DataFrame:
    """
    Handle missing values in intensity data.
    
    Args:
        dataframe: Input DataFrame
        intensity_columns: List of intensity column names
        method: Method for handling missing values ("median", "mean", "zero", "drop")
        group_by_cols: Optional columns to group by for imputation
        
    Returns:
        DataFrame with missing values handled
    """
    logger.info(f"Handling missing values using {method} method")
    
    if method == "drop":
        # Drop rows with any missing values in intensity columns
        result_df = dataframe.dropna(subset=intensity_columns)
        
    elif method == "zero":
        # Replace missing values with zero
        result_df = dataframe.fillna(0.0, subset=intensity_columns)
        
    elif method in ["mean", "median"]:
        # Calculate statistics for imputation
        if group_by_cols:
            # Group-wise imputation
            result_df = _group_wise_imputation(dataframe, intensity_columns, method, group_by_cols)
        else:
            # Global imputation
            result_df = _global_imputation(dataframe, intensity_columns, method)
    
    else:
        logger.warning(f"Unknown missing value method: {method}. Using median.")
        result_df = _global_imputation(dataframe, intensity_columns, "median")
    
    logger.info("Missing value handling completed")
    return result_df


def _global_imputation(dataframe: DataFrame, intensity_columns: List[str], method: str) -> DataFrame:
    """Perform global imputation of missing values."""
    result_df = dataframe
    
    for col_name in intensity_columns:
        if method == "mean":
            fill_value = dataframe.agg(mean(col(col_name))).collect()[0][0]
        else:  # median
            fill_value = dataframe.approxQuantile(col_name, [0.5], 0.01)[0]
        
        if fill_value is not None:
            result_df = result_df.fillna(fill_value, subset=[col_name])
    
    return result_df


def _group_wise_imputation(
    dataframe: DataFrame, 
    intensity_columns: List[str], 
    method: str, 
    group_by_cols: List[str]
) -> DataFrame:
    """Perform group-wise imputation of missing values."""
    from pyspark.sql.window import Window
    from pyspark.sql.functions import avg, expr
    
    # Create window specification for group-wise statistics
    window_spec = Window.partitionBy(*group_by_cols)
    
    result_df = dataframe
    
    for col_name in intensity_columns:
        if method == "mean":
            group_stat = avg(col(col_name)).over(window_spec)
        else:  # median approximation using percentile
            group_stat = expr(f"percentile_approx({col_name}, 0.5)").over(window_spec)
        
        # Fill missing values with group statistics
        result_df = result_df.withColumn(
            col_name,
            when(col(col_name).isNull() | isnan(col(col_name)), group_stat).otherwise(col(col_name))
        )
    
    return result_df


def remove_outliers(
    dataframe: DataFrame,
    intensity_columns: List[str],
    method: str = "iqr",
    threshold: float = 3.0,
    action: str = "cap"
) -> DataFrame:
    """
    Remove or cap outliers in intensity data.
    
    Args:
        dataframe: Input DataFrame
        intensity_columns: List of intensity column names
        method: Outlier detection method ("iqr", "zscore")
        threshold: Threshold for outlier detection
        action: Action for outliers ("remove", "cap", "flag")
        
    Returns:
        DataFrame with outliers handled
    """
    logger.info(f"Removing outliers using {method} method with threshold {threshold}")
    
    result_df = dataframe
    
    for col_name in intensity_columns:
        if method == "iqr":
            result_df = _remove_outliers_iqr(result_df, col_name, threshold, action)
        elif method == "zscore":
            result_df = _remove_outliers_zscore(result_df, col_name, threshold, action)
        else:
            logger.warning(f"Unknown outlier method: {method}")
    
    logger.info("Outlier removal completed")
    return result_df


def _remove_outliers_iqr(dataframe: DataFrame, col_name: str, threshold: float, action: str) -> DataFrame:
    """Remove outliers using IQR method."""
    # Calculate quartiles
    quantiles = dataframe.approxQuantile(col_name, [0.25, 0.75], 0.01)
    q1, q3 = quantiles[0], quantiles[1]
    iqr = q3 - q1
    
    # Calculate bounds
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    if action == "remove":
        # Remove outliers
        result_df = dataframe.filter(
            (col(col_name) >= lower_bound) & (col(col_name) <= upper_bound)
        )
    elif action == "cap":
        # Cap outliers
        result_df = dataframe.withColumn(
            col_name,
            when(col(col_name) < lower_bound, lower_bound)
            .when(col(col_name) > upper_bound, upper_bound)
            .otherwise(col(col_name))
        )
    elif action == "flag":
        # Flag outliers
        outlier_flag_col = f"{col_name}_outlier"
        result_df = dataframe.withColumn(
            outlier_flag_col,
            when((col(col_name) < lower_bound) | (col(col_name) > upper_bound), True)
            .otherwise(False)
        )
    else:
        result_df = dataframe
    
    return result_df


def _remove_outliers_zscore(dataframe: DataFrame, col_name: str, threshold: float, action: str) -> DataFrame:
    """Remove outliers using Z-score method."""
    # Calculate mean and standard deviation
    stats = dataframe.agg(mean(col(col_name)), stddev(col(col_name))).collect()[0]
    col_mean, col_std = stats[0], stats[1]
    
    if col_std is None or col_std == 0:
        logger.warning(f"Cannot calculate Z-score for {col_name} (std = {col_std})")
        return dataframe
    
    # Calculate Z-score
    z_score_col = f"zscore_{col_name}"
    df_with_zscore = dataframe.withColumn(
        z_score_col,
        spark_abs((col(col_name) - col_mean) / col_std)
    )
    
    if action == "remove":
        # Remove outliers
        result_df = df_with_zscore.filter(col(z_score_col) <= threshold).drop(z_score_col)
    elif action == "cap":
        # Cap outliers (preserve original scale)
        lower_bound = col_mean - threshold * col_std
        upper_bound = col_mean + threshold * col_std
        
        result_df = df_with_zscore.withColumn(
            col_name,
            when(col(col_name) < lower_bound, lower_bound)
            .when(col(col_name) > upper_bound, upper_bound)
            .otherwise(col(col_name))
        ).drop(z_score_col)
    elif action == "flag":
        # Flag outliers
        outlier_flag_col = f"{col_name}_outlier"
        result_df = df_with_zscore.withColumn(
            outlier_flag_col,
            when(col(z_score_col) > threshold, True).otherwise(False)
        ).drop(z_score_col)
    else:
        result_df = df_with_zscore.drop(z_score_col)
    
    return result_df


def validate_data_quality(
    dataframe: DataFrame,
    intensity_columns: List[str],
    min_samples: int = 10,
    min_features: int = 5
) -> Tuple[bool, List[str]]:
    """
    Validate data quality after cleaning.
    
    Args:
        dataframe: Cleaned DataFrame
        intensity_columns: List of intensity column names
        min_samples: Minimum number of samples required
        min_features: Minimum number of features required
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    logger.info("Validating data quality")
    
    issues = []
    
    # Check sample count
    sample_count = dataframe.count()
    if sample_count < min_samples:
        issues.append(f"Insufficient samples: {sample_count} < {min_samples}")
    
    # Check feature count
    available_features = [col for col in intensity_columns if col in dataframe.columns]
    feature_count = len(available_features)
    if feature_count < min_features:
        issues.append(f"Insufficient features: {feature_count} < {min_features}")
    
    # Check for remaining missing values
    for col_name in available_features:
        missing_count = dataframe.filter(col(col_name).isNull() | isnan(col(col_name))).count()
        if missing_count > 0:
            missing_fraction = missing_count / sample_count
            if missing_fraction > 0.1:  # More than 10% missing
                issues.append(f"High missing values in {col_name}: {missing_fraction:.2%}")
    
    is_valid = len(issues) == 0
    
    if is_valid:
        logger.info("Data quality validation passed")
    else:
        logger.warning(f"Data quality issues found: {'; '.join(issues)}")
    
    return is_valid, issues