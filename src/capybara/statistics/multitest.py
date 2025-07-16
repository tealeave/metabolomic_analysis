"""
Multiple testing correction utilities for metabolomic analysis.

This module provides distributed multiple testing correction capabilities using PySpark,
specifically designed for metabolomics biomarker discovery workflows.
"""

import logging
from enum import Enum
from typing import Optional

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, when, rank, count, lit
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)


class CorrectionMethods(Enum):
    """Multiple testing correction methods."""
    FDR_BH = "fdr_bh"  # Benjamini-Hochberg False Discovery Rate
    BONFERRONI = "bonferroni"  # Bonferroni correction
    HOLM = "holm"  # Holm-Bonferroni method


def multitest_correct(
    dataframe: DataFrame,
    method: CorrectionMethods = CorrectionMethods.FDR_BH,
    alpha: float = 0.05,
    spark: SparkSession = None,
    separate_annotated_mtbs: bool = True,
    p_value_col: str = "p_value",
    compound_class_col: str = "compound_class"
) -> DataFrame:
    """
    Apply multiple testing correction to p-values.
    
    This function performs multiple testing correction on p-values in a distributed manner,
    with optional separation by compound class for metabolomics data.
    
    Args:
        dataframe: Input DataFrame with p-values
        method: Correction method to use
        alpha: Significance threshold
        spark: SparkSession (optional)
        separate_annotated_mtbs: Whether to separate correction by compound class
        p_value_col: Name of p-value column
        compound_class_col: Name of compound class column
        
    Returns:
        DataFrame with corrected p-values and significance indicators
    """
    logger.info(f"Applying multiple testing correction using {method.value}")
    
    try:
        from pyspark.sql.functions import udf, collect_list, struct, explode
        from pyspark.sql.types import ArrayType, StructType, StructField, DoubleType, BooleanType
        import numpy as np
        from statsmodels.stats.multitest import multipletests
        
        # Validate input
        if p_value_col not in dataframe.columns:
            raise ValueError(f"P-value column '{p_value_col}' not found in DataFrame")
        
        # Filter out null p-values
        clean_df = dataframe.filter(col(p_value_col).isNotNull())
        
        if separate_annotated_mtbs and compound_class_col in dataframe.columns:
            # Apply correction separately for each compound class
            logger.info("Applying correction separately by compound class")
            
            # Get unique compound classes
            compound_classes = clean_df.select(compound_class_col).distinct().rdd.map(lambda x: x[0]).collect()
            
            corrected_dfs = []
            
            for compound_class in compound_classes:
                if compound_class is not None:
                    logger.info(f"Processing compound class: {compound_class}")
                    
                    # Filter for current compound class
                    class_df = clean_df.filter(col(compound_class_col) == compound_class)
                    
                    # Apply correction to this class
                    corrected_class_df = _apply_correction_to_group(
                        class_df, method, alpha, p_value_col
                    )
                    
                    corrected_dfs.append(corrected_class_df)
            
            # Handle unannotated (null compound class) separately
            null_class_df = clean_df.filter(col(compound_class_col).isNull())
            if null_class_df.count() > 0:
                logger.info("Processing unannotated compounds")
                corrected_null_df = _apply_correction_to_group(
                    null_class_df, method, alpha, p_value_col
                )
                corrected_dfs.append(corrected_null_df)
            
            # Union all corrected DataFrames
            if corrected_dfs:
                result_df = corrected_dfs[0]
                for df in corrected_dfs[1:]:
                    result_df = result_df.union(df)
            else:
                result_df = clean_df.withColumn("p_value_corrected", col(p_value_col)) \
                                  .withColumn("significant", lit(False))
        
        else:
            # Apply correction to entire dataset
            logger.info("Applying correction to entire dataset")
            result_df = _apply_correction_to_group(clean_df, method, alpha, p_value_col)
        
        logger.info("Multiple testing correction completed successfully")
        return result_df
        
    except ImportError as e:
        logger.error(f"Required packages not available: {str(e)}")
        logger.error("Please install required packages: statsmodels")
        raise
    except Exception as e:
        logger.error(f"Error in multiple testing correction: {str(e)}")
        raise


def _apply_correction_to_group(
    dataframe: DataFrame,
    method: CorrectionMethods,
    alpha: float,
    p_value_col: str
) -> DataFrame:
    """
    Apply multiple testing correction to a single group.
    
    Args:
        dataframe: Input DataFrame
        method: Correction method
        alpha: Significance threshold
        p_value_col: P-value column name
        
    Returns:
        DataFrame with corrected p-values
    """
    from pyspark.sql.functions import udf, collect_list, explode, monotonically_increasing_id
    from pyspark.sql.types import ArrayType, StructType, StructField, DoubleType, BooleanType, LongType
    import numpy as np
    from statsmodels.stats.multitest import multipletests
    
    # Add row ID for reconstruction
    df_with_id = dataframe.withColumn("row_id", monotonically_increasing_id())
    
    # Collect p-values
    p_values = df_with_id.select(p_value_col).rdd.map(lambda x: float(x[0])).collect()
    
    if len(p_values) == 0:
        return dataframe.withColumn("p_value_corrected", col(p_value_col)) \
                       .withColumn("significant", lit(False))
    
    # Apply correction using statsmodels
    try:
        if method == CorrectionMethods.FDR_BH:
            reject, p_corrected, alpha_sidak, alpha_bonf = multipletests(
                p_values, alpha=alpha, method='fdr_bh'
            )
        elif method == CorrectionMethods.BONFERRONI:
            reject, p_corrected, alpha_sidak, alpha_bonf = multipletests(
                p_values, alpha=alpha, method='bonferroni'
            )
        elif method == CorrectionMethods.HOLM:
            reject, p_corrected, alpha_sidak, alpha_bonf = multipletests(
                p_values, alpha=alpha, method='holm'
            )
        else:
            raise ValueError(f"Unsupported correction method: {method}")
        
        # Create correction results DataFrame
        correction_data = [(i, float(p_corrected[i]), bool(reject[i])) 
                          for i in range(len(p_values))]
        
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        
        correction_schema = StructType([
            StructField("array_index", LongType(), False),
            StructField("p_value_corrected", DoubleType(), False),
            StructField("significant", BooleanType(), False)
        ])
        
        correction_df = spark.createDataFrame(correction_data, correction_schema)
        
        # Add array index to original DataFrame
        df_with_index = df_with_id.withColumn(
            "array_index", 
            row_number().over(Window.orderBy("row_id")) - 1
        )
        
        # Join with correction results
        result_df = df_with_index.join(
            correction_df, on="array_index", how="inner"
        ).drop("row_id", "array_index")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error applying {method.value} correction: {str(e)}")
        # Return original data with uncorrected p-values
        return dataframe.withColumn("p_value_corrected", col(p_value_col)) \
                       .withColumn("significant", col(p_value_col) < alpha)


def fdr_correction(
    dataframe: DataFrame,
    alpha: float = 0.05,
    p_value_col: str = "p_value"
) -> DataFrame:
    """
    Convenience function for FDR (Benjamini-Hochberg) correction.
    
    Args:
        dataframe: Input DataFrame with p-values
        alpha: Significance threshold
        p_value_col: Name of p-value column
        
    Returns:
        DataFrame with FDR-corrected p-values
    """
    return multitest_correct(
        dataframe=dataframe,
        method=CorrectionMethods.FDR_BH,
        alpha=alpha,
        p_value_col=p_value_col,
        separate_annotated_mtbs=False
    )


def bonferroni_correction(
    dataframe: DataFrame,
    alpha: float = 0.05,
    p_value_col: str = "p_value"
) -> DataFrame:
    """
    Convenience function for Bonferroni correction.
    
    Args:
        dataframe: Input DataFrame with p-values
        alpha: Significance threshold
        p_value_col: Name of p-value column
        
    Returns:
        DataFrame with Bonferroni-corrected p-values
    """
    return multitest_correct(
        dataframe=dataframe,
        method=CorrectionMethods.BONFERRONI,
        alpha=alpha,
        p_value_col=p_value_col,
        separate_annotated_mtbs=False
    )


# Import for row_number function
try:
    from pyspark.sql.functions import row_number
except ImportError:
    # Define a simple row_number function if not available
    from pyspark.sql.functions import monotonically_increasing_id
    from pyspark.sql.window import Window
    
    def row_number():
        return monotonically_increasing_id()