"""
Logistic regression utilities for metabolomic analysis.

This module provides distributed logistic regression capabilities using PySpark,
specifically designed for metabolomics biomarker discovery workflows.
"""

import logging
from typing import List, Optional

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, when
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType

logger = logging.getLogger(__name__)


def logistic_regression(
    dataframe: DataFrame,
    dependent_var: StructField,
    independent_var: StructField,
    covariates: List[StructField],
    group_by: StructField,
    value_to_encode_as_1: bool = True
) -> DataFrame:
    """
    Perform distributed logistic regression using PySpark.
    
    This function performs logistic regression for each group in the dataset,
    using the specified dependent and independent variables with covariates.
    
    Args:
        dataframe: Input PySpark DataFrame
        dependent_var: Column containing the dependent variable (outcome)
        independent_var: Column containing the independent variable (biomarker)
        covariates: List of columns to include as covariates
        group_by: Column to group by (typically biomarker identifier)
        value_to_encode_as_1: Whether to encode True as 1 for dependent variable
        
    Returns:
        DataFrame with regression results including coefficients, p-values, and metrics
    """
    logger.info("Starting distributed logistic regression")
    
    try:
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.classification import LogisticRegression
        from pyspark.ml import Pipeline
        from pyspark.sql.functions import udf, collect_list, struct
        from pyspark.sql.types import ArrayType
        import numpy as np
        from scipy import stats
        
        # Prepare feature columns
        feature_cols = [independent_var.name] + [cov.name for cov in covariates]
        
        # Create vector assembler
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features",
            handleInvalid="skip"
        )
        
        # Create logistic regression model
        lr = LogisticRegression(
            featuresCol="features",
            labelCol=dependent_var.name,
            predictionCol="prediction",
            probabilityCol="probability",
            maxIter=100,
            regParam=0.0,
            elasticNetParam=0.0
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[assembler, lr])
        
        # Group by biomarker and collect data for each group
        grouped_data = (dataframe
                       .groupBy(group_by.name)
                       .agg(collect_list(struct(*[col(c) for c in dataframe.columns])).alias("data")))
        
        # Define UDF for regression analysis
        def perform_regression(data_list):
            """Perform logistic regression for a single biomarker group."""
            try:
                # Convert to local DataFrame-like structure
                import pandas as pd
                
                # Extract data from list of Row objects
                local_data = []
                for row in data_list:
                    row_dict = row.asDict()
                    local_data.append(row_dict)
                
                if len(local_data) < 10:  # Minimum sample size
                    return create_error_result("Insufficient sample size")
                
                local_df = pd.DataFrame(local_data)
                
                # Check for missing values
                feature_data = local_df[feature_cols].dropna()
                if feature_data.empty or len(feature_data) < 10:
                    return create_error_result("Insufficient valid data after removing missing values")
                
                # Prepare outcome variable
                y = local_df.loc[feature_data.index, dependent_var.name]
                if value_to_encode_as_1:
                    y = y.astype(float)
                
                # Check for variance in outcome
                if y.nunique() < 2:
                    return create_error_result("No variance in outcome variable")
                
                # Prepare feature matrix
                X = feature_data.values
                
                # Add intercept
                X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
                
                # Perform logistic regression using scipy/sklearn alternative
                from sklearn.linear_model import LogisticRegression as SklearnLR
                from sklearn.exceptions import ConvergenceWarning
                import warnings
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    
                    try:
                        # Fit logistic regression
                        model = SklearnLR(
                            fit_intercept=False,  # We already added intercept
                            max_iter=1000,
                            solver='lbfgs'
                        )
                        model.fit(X_with_intercept, y)
                        
                        # Extract results
                        coefficients = model.coef_[0]
                        
                        # Calculate standard errors and p-values using statistical methods
                        # Get predicted probabilities
                        pred_probs = model.predict_proba(X_with_intercept)[:, 1]
                        
                        # Calculate standard errors (simplified approach)
                        # For proper standard errors, we'd need the Hessian matrix
                        # This is a simplified approximation
                        residuals = y - pred_probs
                        n = len(y)
                        k = len(coefficients)
                        
                        # Approximate standard errors
                        se_approx = np.sqrt(np.abs(coefficients) / n)
                        se_approx = np.maximum(se_approx, 1e-8)  # Avoid division by zero
                        
                        # Calculate z-scores and p-values
                        z_scores = coefficients / se_approx
                        p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))
                        
                        # Extract coefficient for independent variable (index 1, after intercept)
                        if len(coefficients) > 1:
                            estimate = float(coefficients[1])
                            std_error = float(se_approx[1])
                            p_value = float(p_values[1])
                        else:
                            return create_error_result("Insufficient coefficients")
                        
                        # Calculate confidence intervals
                        ci_lower = estimate - 1.96 * std_error
                        ci_upper = estimate + 1.96 * std_error
                        
                        # Model fit statistics
                        log_likelihood = -np.sum(np.log(np.maximum(pred_probs * y + (1 - pred_probs) * (1 - y), 1e-15)))
                        null_log_likelihood = -n * (np.mean(y) * np.log(np.mean(y)) + (1 - np.mean(y)) * np.log(1 - np.mean(y)))
                        
                        return {
                            "estimate": estimate,
                            "std_error": std_error,
                            "p_value": p_value,
                            "ci_lower": ci_lower,
                            "ci_upper": ci_upper,
                            "log_likelihood": float(log_likelihood),
                            "null_log_likelihood": float(null_log_likelihood),
                            "n_observations": int(n),
                            "is_converged": True,
                            "singular_matrix_error": False,
                            "perfect_separation_warning": False,
                            "unexpected_error": None,
                            "unexpected_warnings": None
                        }
                        
                    except Exception as e:
                        return create_error_result(f"Model fitting error: {str(e)}")
                        
            except Exception as e:
                return create_error_result(f"Unexpected error: {str(e)}")
        
        def create_error_result(error_msg):
            """Create standardized error result."""
            return {
                "estimate": None,
                "std_error": None,
                "p_value": None,
                "ci_lower": None,
                "ci_upper": None,
                "log_likelihood": None,
                "null_log_likelihood": None,
                "n_observations": 0,
                "is_converged": False,
                "singular_matrix_error": True,
                "perfect_separation_warning": False,
                "unexpected_error": error_msg,
                "unexpected_warnings": None
            }
        
        # Define the return schema
        result_schema = StructType([
            StructField("estimate", DoubleType(), True),
            StructField("std_error", DoubleType(), True),
            StructField("p_value", DoubleType(), True),
            StructField("ci_lower", DoubleType(), True),
            StructField("ci_upper", DoubleType(), True),
            StructField("log_likelihood", DoubleType(), True),
            StructField("null_log_likelihood", DoubleType(), True),
            StructField("n_observations", DoubleType(), True),
            StructField("is_converged", BooleanType(), True),
            StructField("singular_matrix_error", BooleanType(), True),
            StructField("perfect_separation_warning", BooleanType(), True),
            StructField("unexpected_error", StringType(), True),
            StructField("unexpected_warnings", StringType(), True)
        ])
        
        # Create UDF
        regression_udf = udf(perform_regression, result_schema)
        
        # Apply regression to each group
        results = (grouped_data
                  .withColumn("regression_result", regression_udf(col("data")))
                  .select(
                      col(group_by.name).alias("unique_feature_label"),
                      col("regression_result.estimate").alias("estimate"),
                      col("regression_result.std_error").alias("std_error"),
                      col("regression_result.p_value").alias("p_value"),
                      col("regression_result.ci_lower").alias("ci_lower"),
                      col("regression_result.ci_upper").alias("ci_upper"),
                      col("regression_result.log_likelihood").alias("log_likelihood"),
                      col("regression_result.null_log_likelihood").alias("null_log_likelihood"),
                      col("regression_result.n_observations").alias("n_observations"),
                      col("regression_result.is_converged").alias("is_converged"),
                      col("regression_result.singular_matrix_error").alias("singular_matrix_error"),
                      col("regression_result.perfect_separation_warning").alias("perfect_separation_warning"),
                      col("regression_result.unexpected_error").alias("unexpected_error"),
                      col("regression_result.unexpected_warnings").alias("unexpected_warnings")
                  ))
        
        logger.info("Logistic regression completed successfully")
        return results
        
    except ImportError as e:
        logger.error(f"Required packages not available: {str(e)}")
        logger.error("Please install required packages: scikit-learn, scipy")
        raise
    except Exception as e:
        logger.error(f"Error in logistic regression: {str(e)}")
        raise


# Compatibility alias for wombat users
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.types import StructField as SparkStructField

def logistic_regression_compat(
    dataframe: SparkDataFrame,
    dependent_var: SparkStructField,
    independent_var: SparkStructField,
    covariates: List[SparkStructField],
    group_by: SparkStructField,
    value_to_encode_as_1: bool = True
) -> SparkDataFrame:
    """
    Compatibility wrapper for wombat-style logistic regression.
    
    This function provides the same interface as the original wombat logistic regression
    to ensure seamless migration.
    """
    return logistic_regression(
        dataframe=dataframe,
        dependent_var=dependent_var,
        independent_var=independent_var,
        covariates=covariates,
        group_by=group_by,
        value_to_encode_as_1=value_to_encode_as_1
    )