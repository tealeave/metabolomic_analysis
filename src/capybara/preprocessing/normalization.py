"""
Data normalization utilities for metabolomic analysis.

This module provides functions for normalizing and transforming metabolomic intensity data
using PySpark for distributed processing.
"""

import logging
from typing import List, Optional, Dict, Any

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, log, exp, mean, stddev, broadcast, lit
from pyspark.sql.types import DoubleType

logger = logging.getLogger(__name__)


def normalize_intensity_data(
    dataframe: DataFrame,
    intensity_columns: List[str],
    method: str = "log_median_centering",
    group_by_cols: Optional[List[str]] = None,
    reference_stats: Optional[Dict[str, Any]] = None
) -> DataFrame:
    """
    Normalize intensity data using specified method.
    
    Args:
        dataframe: Input DataFrame with intensity data
        intensity_columns: List of intensity column names
        method: Normalization method ("log_median_centering", "log_mean_centering", 
                "zscore", "quantile", "log_transform")
        group_by_cols: Optional columns to group by for normalization (e.g., ["site"])
        reference_stats: Optional pre-computed statistics for consistent normalization
        
    Returns:
        Normalized DataFrame
    """
    logger.info(f"Normalizing intensity data using {method}")
    
    if method == "log_median_centering":
        return log_median_centering(dataframe, intensity_columns, group_by_cols, reference_stats)
    elif method == "log_mean_centering":
        return log_mean_centering(dataframe, intensity_columns, group_by_cols, reference_stats)
    elif method == "zscore":
        return zscore_normalization(dataframe, intensity_columns, group_by_cols, reference_stats)
    elif method == "quantile":
        return quantile_normalization(dataframe, intensity_columns)
    elif method == "log_transform":
        return log_transform(dataframe, intensity_columns)
    else:
        logger.warning(f"Unknown normalization method: {method}. Applying log transform only.")
        return log_transform(dataframe, intensity_columns)


def log_median_centering(
    dataframe: DataFrame,
    intensity_columns: List[str],
    group_by_cols: Optional[List[str]] = None,
    reference_stats: Optional[Dict[str, Any]] = None
) -> DataFrame:
    """
    Apply log transformation followed by median centering.
    
    This method: log2(intensity) - median(log2(intensity)) + global_median
    
    Args:
        dataframe: Input DataFrame
        intensity_columns: List of intensity column names
        group_by_cols: Optional grouping columns (e.g., site for batch correction)
        reference_stats: Optional pre-computed statistics
        
    Returns:
        Normalized DataFrame
    """
    logger.info("Applying log median centering normalization")
    
    # Apply log transformation first
    log_df = log_transform(dataframe, intensity_columns)
    
    if group_by_cols:
        # Group-wise median centering (e.g., for batch correction)
        return _group_wise_median_centering(log_df, intensity_columns, group_by_cols, reference_stats)
    else:
        # Global median centering
        return _global_median_centering(log_df, intensity_columns, reference_stats)


def log_mean_centering(
    dataframe: DataFrame,
    intensity_columns: List[str],
    group_by_cols: Optional[List[str]] = None,
    reference_stats: Optional[Dict[str, Any]] = None
) -> DataFrame:
    """
    Apply log transformation followed by mean centering.
    
    Args:
        dataframe: Input DataFrame
        intensity_columns: List of intensity column names
        group_by_cols: Optional grouping columns
        reference_stats: Optional pre-computed statistics
        
    Returns:
        Normalized DataFrame
    """
    logger.info("Applying log mean centering normalization")
    
    # Apply log transformation first
    log_df = log_transform(dataframe, intensity_columns)
    
    if group_by_cols:
        return _group_wise_mean_centering(log_df, intensity_columns, group_by_cols, reference_stats)
    else:
        return _global_mean_centering(log_df, intensity_columns, reference_stats)


def zscore_normalization(
    dataframe: DataFrame,
    intensity_columns: List[str],
    group_by_cols: Optional[List[str]] = None,
    reference_stats: Optional[Dict[str, Any]] = None
) -> DataFrame:
    """
    Apply Z-score normalization: (x - mean) / std.
    
    Args:
        dataframe: Input DataFrame
        intensity_columns: List of intensity column names
        group_by_cols: Optional grouping columns
        reference_stats: Optional pre-computed statistics
        
    Returns:
        Z-score normalized DataFrame
    """
    logger.info("Applying Z-score normalization")
    
    # Apply log transformation first
    log_df = log_transform(dataframe, intensity_columns)
    
    result_df = log_df
    
    for col_name in intensity_columns:
        log_col_name = f"log2_{col_name}"
        
        if reference_stats and col_name in reference_stats:
            # Use pre-computed statistics
            col_mean = reference_stats[col_name]["mean"]
            col_std = reference_stats[col_name]["std"]
        else:
            # Calculate statistics
            stats = result_df.agg(mean(col(log_col_name)), stddev(col(log_col_name))).collect()[0]
            col_mean, col_std = stats[0], stats[1]
            
            if col_std is None or col_std == 0:
                logger.warning(f"Cannot normalize {col_name}: std = {col_std}")
                continue
        
        # Apply Z-score normalization
        result_df = result_df.withColumn(
            log_col_name,
            (col(log_col_name) - col_mean) / col_std
        )
    
    return result_df


def quantile_normalization(
    dataframe: DataFrame,
    intensity_columns: List[str]
) -> DataFrame:
    """
    Apply quantile normalization across samples.
    
    Args:
        dataframe: Input DataFrame
        intensity_columns: List of intensity column names
        
    Returns:
        Quantile normalized DataFrame
    """
    logger.info("Applying quantile normalization")
    
    # Apply log transformation first
    log_df = log_transform(dataframe, intensity_columns)
    
    # This is a simplified quantile normalization
    # For full quantile normalization, we'd need to rank within samples
    # and replace with mean of ranks across samples
    
    result_df = log_df
    
    for col_name in intensity_columns:
        log_col_name = f"log2_{col_name}"
        
        # Calculate percentiles
        percentiles = result_df.approxQuantile(log_col_name, [0.25, 0.5, 0.75], 0.01)
        
        if len(percentiles) == 3:
            q25, q50, q75 = percentiles
            iqr = q75 - q25
            
            # Robust scaling: (x - median) / IQR
            if iqr > 0:
                result_df = result_df.withColumn(
                    log_col_name,
                    (col(log_col_name) - q50) / iqr
                )
    
    return result_df


def log_transform(
    dataframe: DataFrame,
    intensity_columns: List[str],
    base: str = "log2",
    offset: float = 1.0
) -> DataFrame:
    """
    Apply log transformation to intensity data.
    
    Args:
        dataframe: Input DataFrame
        intensity_columns: List of intensity column names
        base: Logarithm base ("log2", "log10", "ln")
        offset: Value to add before log transformation (to handle zeros)
        
    Returns:
        Log-transformed DataFrame
    """
    logger.info(f"Applying {base} transformation with offset {offset}")
    
    result_df = dataframe
    
    for col_name in intensity_columns:
        # Add offset to handle zeros
        offset_col = col(col_name) + offset
        
        if base == "log2":
            log_col = log(offset_col) / log(lit(2.0))
        elif base == "log10":
            log_col = log(offset_col) / log(lit(10.0))
        elif base == "ln":
            log_col = log(offset_col)
        else:
            logger.warning(f"Unknown log base: {base}. Using log2.")
            log_col = log(offset_col) / log(lit(2.0))
        
        # Create new column with log-transformed values
        log_col_name = f"log2_{col_name}"
        result_df = result_df.withColumn(log_col_name, log_col)
    
    return result_df


def _global_median_centering(
    dataframe: DataFrame,
    intensity_columns: List[str],
    reference_stats: Optional[Dict[str, Any]] = None
) -> DataFrame:
    """Apply global median centering."""
    result_df = dataframe
    
    for col_name in intensity_columns:
        log_col_name = f"log2_{col_name}"
        
        if reference_stats and col_name in reference_stats:
            col_median = reference_stats[col_name]["median"]
            global_median = reference_stats["global_median"]
        else:
            col_median = result_df.approxQuantile(log_col_name, [0.5], 0.01)[0]
            global_median = 0.0  # Default global median
        
        # Apply median centering
        result_df = result_df.withColumn(
            log_col_name,
            col(log_col_name) - col_median + global_median
        )
    
    return result_df


def _group_wise_median_centering(
    dataframe: DataFrame,
    intensity_columns: List[str],
    group_by_cols: List[str],
    reference_stats: Optional[Dict[str, Any]] = None
) -> DataFrame:
    """Apply group-wise median centering (e.g., site-specific batch correction)."""
    from pyspark.sql.window import Window
    from pyspark.sql.functions import expr
    
    result_df = dataframe
    
    # Create window specification for group-wise statistics
    window_spec = Window.partitionBy(*group_by_cols)
    
    for col_name in intensity_columns:
        log_col_name = f"log2_{col_name}"
        
        if reference_stats and col_name in reference_stats:
            # Use pre-computed group-specific statistics
            # This would require more complex logic to apply group-specific stats
            logger.warning("Group-wise reference stats not fully implemented")
            group_median = expr(f"percentile_approx({log_col_name}, 0.5)").over(window_spec)
        else:
            # Calculate group-wise median
            group_median = expr(f"percentile_approx({log_col_name}, 0.5)").over(window_spec)
        
        # Calculate global median
        global_median = dataframe.approxQuantile(log_col_name, [0.5], 0.01)[0]
        
        # Apply group-wise median centering: log_intensity - group_median + global_median
        result_df = result_df.withColumn(
            log_col_name,
            col(log_col_name) - group_median + global_median
        )
    
    return result_df


def _global_mean_centering(
    dataframe: DataFrame,
    intensity_columns: List[str],
    reference_stats: Optional[Dict[str, Any]] = None
) -> DataFrame:
    """Apply global mean centering."""
    result_df = dataframe
    
    for col_name in intensity_columns:
        log_col_name = f"log2_{col_name}"
        
        if reference_stats and col_name in reference_stats:
            col_mean = reference_stats[col_name]["mean"]
            global_mean = reference_stats["global_mean"]
        else:
            col_mean = result_df.agg(mean(col(log_col_name))).collect()[0][0]
            global_mean = 0.0  # Default global mean
        
        # Apply mean centering
        result_df = result_df.withColumn(
            log_col_name,
            col(log_col_name) - col_mean + global_mean
        )
    
    return result_df


def _group_wise_mean_centering(
    dataframe: DataFrame,
    intensity_columns: List[str],
    group_by_cols: List[str],
    reference_stats: Optional[Dict[str, Any]] = None
) -> DataFrame:
    """Apply group-wise mean centering."""
    from pyspark.sql.window import Window
    from pyspark.sql.functions import avg
    
    result_df = dataframe
    
    # Create window specification for group-wise statistics
    window_spec = Window.partitionBy(*group_by_cols)
    
    for col_name in intensity_columns:
        log_col_name = f"log2_{col_name}"
        
        # Calculate group-wise mean
        group_mean = avg(col(log_col_name)).over(window_spec)
        
        # Calculate global mean
        global_mean = dataframe.agg(mean(col(log_col_name))).collect()[0][0]
        
        # Apply group-wise mean centering
        result_df = result_df.withColumn(
            log_col_name,
            col(log_col_name) - group_mean + global_mean
        )
    
    return result_df


def compute_normalization_stats(
    dataframe: DataFrame,
    intensity_columns: List[str],
    group_by_cols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute normalization statistics for consistent normalization across datasets.
    
    Args:
        dataframe: Training DataFrame
        intensity_columns: List of intensity column names
        group_by_cols: Optional grouping columns
        
    Returns:
        Dictionary containing normalization statistics
    """
    logger.info("Computing normalization statistics")
    
    # Apply log transformation first
    log_df = log_transform(dataframe, intensity_columns)
    
    stats = {}
    
    for col_name in intensity_columns:
        log_col_name = f"log2_{col_name}"
        
        # Calculate basic statistics
        basic_stats = log_df.agg(
            mean(col(log_col_name)),
            stddev(col(log_col_name))
        ).collect()[0]
        
        median = log_df.approxQuantile(log_col_name, [0.5], 0.01)[0]
        
        stats[col_name] = {
            "mean": basic_stats[0],
            "std": basic_stats[1],
            "median": median
        }
    
    # Calculate global statistics
    all_log_cols = [f"log2_{col}" for col in intensity_columns]
    if all_log_cols:
        # This is a simplified global median calculation
        stats["global_median"] = 0.0
        stats["global_mean"] = 0.0
    
    logger.info(f"Computed statistics for {len(intensity_columns)} features")
    return stats