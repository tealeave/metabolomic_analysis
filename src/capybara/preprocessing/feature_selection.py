"""
Feature selection utilities for metabolomic analysis.

This module provides functions for selecting relevant features from metabolomic data
using various statistical and machine learning approaches.
"""

import logging
from typing import List, Optional, Tuple, Dict, Any

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, variance, stddev, corr, abs as spark_abs, count, isnan, isnull

logger = logging.getLogger(__name__)


def select_features_by_variance(
    dataframe: DataFrame,
    intensity_columns: List[str],
    variance_threshold: float = 0.0,
    return_stats: bool = False
) -> Tuple[List[str], Optional[Dict[str, float]]]:
    """
    Select features based on variance threshold.
    
    Args:
        dataframe: Input DataFrame with intensity data
        intensity_columns: List of intensity column names
        variance_threshold: Minimum variance threshold
        return_stats: Whether to return variance statistics
        
    Returns:
        Tuple of (selected feature names, optional variance stats)
    """
    logger.info(f"Selecting features with variance > {variance_threshold}")
    
    selected_features = []
    variance_stats = {} if return_stats else None
    
    for col_name in intensity_columns:
        # Calculate variance
        var_result = dataframe.agg(variance(col(col_name))).collect()[0][0]
        
        if var_result is not None and var_result > variance_threshold:
            selected_features.append(col_name)
            
        if return_stats:
            variance_stats[col_name] = var_result if var_result is not None else 0.0
    
    logger.info(f"Selected {len(selected_features)} out of {len(intensity_columns)} features")
    return selected_features, variance_stats


def select_features_by_correlation(
    dataframe: DataFrame,
    intensity_columns: List[str],
    target_column: str,
    correlation_threshold: float = 0.1,
    method: str = "pearson",
    return_stats: bool = False
) -> Tuple[List[str], Optional[Dict[str, float]]]:
    """
    Select features based on correlation with target variable.
    
    Args:
        dataframe: Input DataFrame
        intensity_columns: List of intensity column names
        target_column: Target variable column name
        correlation_threshold: Minimum absolute correlation threshold
        method: Correlation method ("pearson" is default, "spearman" requires additional work)
        return_stats: Whether to return correlation statistics
        
    Returns:
        Tuple of (selected feature names, optional correlation stats)
    """
    logger.info(f"Selecting features with |correlation| > {correlation_threshold} with {target_column}")
    
    selected_features = []
    correlation_stats = {} if return_stats else None
    
    for col_name in intensity_columns:
        try:
            # Calculate correlation
            corr_result = dataframe.stat.corr(col_name, target_column)
            
            if corr_result is not None and abs(corr_result) > correlation_threshold:
                selected_features.append(col_name)
                
            if return_stats:
                correlation_stats[col_name] = corr_result if corr_result is not None else 0.0
                
        except Exception as e:
            logger.warning(f"Could not calculate correlation for {col_name}: {str(e)}")
            if return_stats:
                correlation_stats[col_name] = 0.0
    
    logger.info(f"Selected {len(selected_features)} out of {len(intensity_columns)} features")
    return selected_features, correlation_stats


def select_features_by_missing_values(
    dataframe: DataFrame,
    intensity_columns: List[str],
    missing_threshold: float = 0.1,
    return_stats: bool = False
) -> Tuple[List[str], Optional[Dict[str, float]]]:
    """
    Select features based on missing value threshold.
    
    Args:
        dataframe: Input DataFrame
        intensity_columns: List of intensity column names
        missing_threshold: Maximum fraction of missing values allowed
        return_stats: Whether to return missing value statistics
        
    Returns:
        Tuple of (selected feature names, optional missing value stats)
    """
    logger.info(f"Selecting features with missing values < {missing_threshold:.1%}")
    
    total_rows = dataframe.count()
    selected_features = []
    missing_stats = {} if return_stats else None
    
    for col_name in intensity_columns:
        # Count missing values
        missing_count = dataframe.filter(
            col(col_name).isNull() | isnan(col(col_name))
        ).count()
        
        missing_fraction = missing_count / total_rows
        
        if missing_fraction <= missing_threshold:
            selected_features.append(col_name)
            
        if return_stats:
            missing_stats[col_name] = missing_fraction
    
    logger.info(f"Selected {len(selected_features)} out of {len(intensity_columns)} features")
    return selected_features, missing_stats


def select_features_by_significance(
    dataframe: DataFrame,
    intensity_columns: List[str],
    significance_results: DataFrame,
    p_value_threshold: float = 0.05,
    p_value_column: str = "p_value_corrected",
    feature_id_column: str = "unique_feature_label",
    return_stats: bool = False
) -> Tuple[List[str], Optional[Dict[str, float]]]:
    """
    Select features based on statistical significance results.
    
    Args:
        dataframe: Input DataFrame (not directly used but kept for consistency)
        intensity_columns: List of intensity column names
        significance_results: DataFrame with significance test results
        p_value_threshold: Maximum p-value threshold
        p_value_column: Column name for p-values
        feature_id_column: Column name for feature identifiers
        return_stats: Whether to return p-value statistics
        
    Returns:
        Tuple of (selected feature names, optional p-value stats)
    """
    logger.info(f"Selecting features with p-value < {p_value_threshold}")
    
    # Get significant features
    significant_features_df = significance_results.filter(
        col(p_value_column) < p_value_threshold
    ).select(feature_id_column, p_value_column)
    
    # Collect significant feature names
    significant_features_data = significant_features_df.collect()
    significant_feature_names = [row[feature_id_column] for row in significant_features_data]
    
    # Filter to only include features that are in intensity_columns
    selected_features = [f for f in significant_feature_names if f in intensity_columns]
    
    p_value_stats = None
    if return_stats:
        p_value_stats = {row[feature_id_column]: row[p_value_column] 
                        for row in significant_features_data
                        if row[feature_id_column] in intensity_columns}
    
    logger.info(f"Selected {len(selected_features)} significant features")
    return selected_features, p_value_stats


def select_top_n_features(
    dataframe: DataFrame,
    intensity_columns: List[str],
    n_features: int,
    criterion: str = "variance",
    target_column: Optional[str] = None,
    ascending: bool = False
) -> List[str]:
    """
    Select top N features based on specified criterion.
    
    Args:
        dataframe: Input DataFrame
        intensity_columns: List of intensity column names
        n_features: Number of features to select
        criterion: Selection criterion ("variance", "correlation", "std")
        target_column: Target column for correlation (required if criterion="correlation")
        ascending: Whether to sort in ascending order
        
    Returns:
        List of selected feature names
    """
    logger.info(f"Selecting top {n_features} features by {criterion}")
    
    if n_features >= len(intensity_columns):
        logger.warning(f"Requested {n_features} features but only {len(intensity_columns)} available")
        return intensity_columns
    
    feature_scores = []
    
    for col_name in intensity_columns:
        try:
            if criterion == "variance":
                score = dataframe.agg(variance(col(col_name))).collect()[0][0]
            elif criterion == "std":
                score = dataframe.agg(stddev(col(col_name))).collect()[0][0]
            elif criterion == "correlation":
                if target_column is None:
                    raise ValueError("target_column is required for correlation criterion")
                score = abs(dataframe.stat.corr(col_name, target_column))
            else:
                raise ValueError(f"Unknown criterion: {criterion}")
            
            if score is not None:
                feature_scores.append((col_name, score))
        except Exception as e:
            logger.warning(f"Could not calculate {criterion} for {col_name}: {str(e)}")
    
    # Sort by score
    feature_scores.sort(key=lambda x: x[1], reverse=not ascending)
    
    # Select top N features
    selected_features = [f[0] for f in feature_scores[:n_features]]
    
    logger.info(f"Selected top {len(selected_features)} features")
    return selected_features


def comprehensive_feature_selection(
    dataframe: DataFrame,
    intensity_columns: List[str],
    target_column: Optional[str] = None,
    significance_results: Optional[DataFrame] = None,
    selection_criteria: Dict[str, Any] = None
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Perform comprehensive feature selection using multiple criteria.
    
    Args:
        dataframe: Input DataFrame
        intensity_columns: List of intensity column names
        target_column: Optional target variable column
        significance_results: Optional significance test results
        selection_criteria: Dictionary of selection criteria and thresholds
        
    Returns:
        Tuple of (selected features, selection statistics)
    """
    logger.info("Performing comprehensive feature selection")
    
    # Default selection criteria
    if selection_criteria is None:
        selection_criteria = {
            "missing_threshold": 0.1,
            "variance_threshold": 0.01,
            "correlation_threshold": 0.05,
            "p_value_threshold": 0.05,
            "max_features": None
        }
    
    current_features = intensity_columns.copy()
    selection_stats = {}
    
    # Step 1: Remove features with too many missing values
    if "missing_threshold" in selection_criteria:
        current_features, missing_stats = select_features_by_missing_values(
            dataframe, current_features, 
            selection_criteria["missing_threshold"], return_stats=True
        )
        selection_stats["missing_values"] = missing_stats
        logger.info(f"After missing value filter: {len(current_features)} features")
    
    # Step 2: Remove low-variance features
    if "variance_threshold" in selection_criteria:
        current_features, variance_stats = select_features_by_variance(
            dataframe, current_features,
            selection_criteria["variance_threshold"], return_stats=True
        )
        selection_stats["variance"] = variance_stats
        logger.info(f"After variance filter: {len(current_features)} features")
    
    # Step 3: Filter by correlation with target (if available)
    if target_column and "correlation_threshold" in selection_criteria:
        current_features, correlation_stats = select_features_by_correlation(
            dataframe, current_features, target_column,
            selection_criteria["correlation_threshold"], return_stats=True
        )
        selection_stats["correlation"] = correlation_stats
        logger.info(f"After correlation filter: {len(current_features)} features")
    
    # Step 4: Filter by statistical significance (if available)
    if significance_results and "p_value_threshold" in selection_criteria:
        current_features, p_value_stats = select_features_by_significance(
            dataframe, current_features, significance_results,
            selection_criteria["p_value_threshold"], return_stats=True
        )
        selection_stats["p_values"] = p_value_stats
        logger.info(f"After significance filter: {len(current_features)} features")
    
    # Step 5: Limit to maximum number of features (if specified)
    if selection_criteria.get("max_features") and len(current_features) > selection_criteria["max_features"]:
        # Use variance as default ranking criterion
        ranking_criterion = selection_criteria.get("ranking_criterion", "variance")
        current_features = select_top_n_features(
            dataframe, current_features, selection_criteria["max_features"],
            criterion=ranking_criterion, target_column=target_column
        )
        logger.info(f"After max features limit: {len(current_features)} features")
    
    selection_stats["final_feature_count"] = len(current_features)
    selection_stats["original_feature_count"] = len(intensity_columns)
    selection_stats["selection_ratio"] = len(current_features) / len(intensity_columns)
    
    logger.info(f"Comprehensive feature selection complete: {len(current_features)} / {len(intensity_columns)} features selected")
    
    return current_features, selection_stats


def get_feature_importance_from_model(
    model,
    feature_names: List[str],
    importance_type: str = "gain"
) -> Dict[str, float]:
    """
    Extract feature importance from trained model.
    
    Args:
        model: Trained model with feature importance
        feature_names: List of feature names
        importance_type: Type of importance to extract
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    logger.info(f"Extracting {importance_type} feature importance from model")
    
    try:
        if hasattr(model, 'feature_importances_'):
            # Sklearn-style models
            importances = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            # XGBoost/CatBoost style
            importances = model.get_feature_importance(type=importance_type)
        else:
            logger.warning("Model does not have recognizable feature importance method")
            return {}
        
        # Create feature importance dictionary
        if len(importances) == len(feature_names):
            importance_dict = dict(zip(feature_names, importances))
        else:
            logger.warning(f"Feature importance length ({len(importances)}) does not match feature names ({len(feature_names)})")
            return {}
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        logger.info(f"Extracted importance for {len(sorted_importance)} features")
        return sorted_importance
        
    except Exception as e:
        logger.error(f"Error extracting feature importance: {str(e)}")
        return {}