"""
Regression Analysis Module

This module performs logistic regression analysis for metabolomics biomarker discovery.
It runs regression for different outcome groups and stratified populations.

Based on the reference implementation from {outcome}-regression.py
"""

import itertools
import logging
import warnings
from typing import Dict, List, Tuple

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast, col, when

from ...utils.config_utils import ConfigManager
from ...utils.spark_utils import create_spark_session

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)


class RegressionAnalyzer:
    """
    Handles logistic regression analysis for metabolomics biomarker discovery.
    
    This class performs the following steps:
    1. Load regression input data
    2. Run logistic regression for different strata and group combinations
    3. Apply multiple testing correction
    4. Save regression results
    """
    
    def __init__(self, config_manager: ConfigManager, spark: SparkSession):
        self.config = config_manager.config
        self.spark = spark
        
        # Outcome mapping for column selection
        self.outcome_mapping = {
            "eope": "eope",
            "lope": "lope", 
            "sb": "sb_new",
            "ptb": "ptb_new",
            "sga": "sga_10"
        }
        
        # Analysis strata and group types
        self.strata = ["lt20", "gte20"]
        self.group_types = ["control", "noncase"]
    
    def load_regression_input(self, outcome: str):
        """
        Load regression input data and prepare for analysis.
        
        Args:
            outcome: The outcome of interest
            
        Returns:
            Prepared DataFrame for regression analysis
        """
        logger.info(f"Loading regression input data for outcome: {outcome}")
        
        if outcome not in self.outcome_mapping:
            raise KeyError(f"Outcome '{outcome}' not found in outcome_mapping.")
        
        outcome_selector = self.outcome_mapping[outcome]
        input_path = self.config.paths.analysis.regression_analysis.pooled_analysis.input_delta_tables.path
        
        # Load and prepare data
        combined_data = (
            self.spark.read.format("delta")
            .load(input_path)
            # Create eope and lope columns when needed
            .withColumn("eope", when((col("pe_new") == 1) & (col("pe_cat") == "EOPE"), 1).otherwise(0))
            .withColumn("lope", when((col("pe_new") == 1) & (col("pe_cat") == "LOPE"), 1).otherwise(0))
            .select(
                "group",
                "unique_feature_label", 
                "intensity_fill_log_scaled",
                "pw_age",
                "ga_weeks",
                "site",
                outcome_selector
            )
            .withColumnRenamed(outcome_selector, outcome)
            .repartition("unique_feature_label")
            .cache()
        )
        
        # Materialize cache
        record_count = combined_data.count()
        logger.info(f"Loaded {record_count} records for regression analysis")
        
        return combined_data
    
    def load_global_id_mapping(self):
        """
        Load the deduplicated global ID mapping for multiple testing correction.
        
        Returns:
            Global ID mapping DataFrame
        """
        logger.info("Loading global ID mapping")
        
        global_id_path = self.config.paired_alignment.alignment.deduplicated_global_id_file
        
        global_id_df = (
            self.spark.read.csv(global_id_path, header=True, inferSchema=True)
            .withColumnRenamed("ID", "compound_id")
            .withColumnRenamed("Compound_Class", "compound_class")
        )
        
        return global_id_df
    
    def run_logistic_regression(
        self, 
        data, 
        outcome: str, 
        strata_val: str, 
        group_type: str
    ) -> pd.DataFrame:
        """
        Run logistic regression for a specific stratum and group combination.
        
        Args:
            data: Input DataFrame
            outcome: The outcome of interest
            strata_val: Stratum value ("lt20" or "gte20")
            group_type: Group type ("control" or "noncase")
            
        Returns:
            Regression results as pandas DataFrame
        """
        logger.info(f"Running logistic regression for {outcome}_{group_type}_{strata_val}")
        
        # Construct group filter conditions
        if group_type == "control":
            control_group_condition = f"{outcome}_control_{strata_val}"
        elif group_type == "noncase":
            control_group_condition = f"{outcome}_noncase_{strata_val}"
        else:
            raise ValueError(f"Unknown group type: {group_type}")
        
        # Filter data for current analysis
        filtered_df = data.filter(
            (col("group") == f"{outcome}_{strata_val}") | 
            (col("group") == control_group_condition)
        ).cache()
        
        # Materialize filtered data
        filtered_count = filtered_df.count()
        logger.info(f"Filtered to {filtered_count} records")
        
        # Define covariates for regression
        covariates = [
            filtered_df.schema["pw_age"],
            filtered_df.schema["ga_weeks"], 
            filtered_df.schema["site"]
        ]
        
        try:
            # Import wombat regression function (will be replaced with capybara)
            from wombat.regression.logistic import logistic_regression
            
            # Perform logistic regression
            regression_results = logistic_regression(
                dataframe=filtered_df,
                dependent_var=filtered_df.schema[outcome],
                independent_var=filtered_df.schema["intensity_fill_log_scaled"],
                covariates=covariates,
                group_by=filtered_df.schema["unique_feature_label"],
                value_to_encode_as_1=True
            ).cache()
            
            # Get regression metrics
            total_results = regression_results.count()
            convergence_rate = regression_results.filter(col('is_converged')).count() / total_results * 100
            valid_pvals = regression_results.filter(col('p_value').isNotNull()).count() / total_results * 100
            singular_errors = regression_results.filter(col('singular_matrix_error')).count() / total_results * 100
            
            logger.info(f"Regression metrics for {outcome}_{group_type}_{strata_val}:")
            logger.info(f"  Convergence rate: {convergence_rate:.2f}%")
            logger.info(f"  Valid p-values: {valid_pvals:.2f}%")
            logger.info(f"  Singular matrix errors: {singular_errors:.2f}%")
            
            # Convert to pandas for multiple testing correction
            results_df = regression_results.toPandas()
            
            # Cleanup
            filtered_df.unpersist()
            regression_results.unpersist()
            
            return results_df
            
        except ImportError:
            logger.error("wombat package not available. Please implement capybara regression module.")
            raise
    
    def apply_multiple_testing_correction(
        self, 
        regression_results: pd.DataFrame,
        global_id_df,
        outcome: str,
        group_type: str, 
        strata_val: str
    ) -> pd.DataFrame:
        """
        Apply multiple testing correction to regression results.
        
        Args:
            regression_results: Raw regression results
            global_id_df: Global ID mapping
            outcome: The outcome of interest
            group_type: Group type
            strata_val: Stratum value
            
        Returns:
            Results with multiple testing correction
        """
        logger.info(f"Applying multiple testing correction for {outcome}_{group_type}_{strata_val}")
        
        # Convert regression results to Spark DataFrame for joining
        regression_spark = self.spark.createDataFrame(regression_results)
        
        # Join with global ID mapping
        results_with_id = (regression_spark
                          .join(broadcast(global_id_df), on="unique_feature_label", how="left")
                          .cache())
        
        try:
            # Import wombat statistics function (will be replaced with capybara)
            from wombat.statistics.multitest import CorrectionMethods, multitest_correct
            
            # Filter for valid results and apply correction
            corrected_results = multitest_correct(
                dataframe=results_with_id.filter(
                    (col("p_value").isNotNull()) &
                    (col("is_converged")) &
                    (~col("singular_matrix_error")) &
                    (col("unexpected_error").isNull())
                ),
                method=CorrectionMethods.FDR_BH,
                alpha=self.config.paths.statistical_analysis_cutoffs.meta_analysis.fdr,
                spark=self.spark,
                separate_annotated_mtbs=True
            ).toPandas()
            
            # Add group identifier
            corrected_results["group"] = f"{outcome}_{group_type}_{strata_val}"
            
            logger.info(f"Multiple testing correction completed. Shape: {corrected_results.shape}")
            
            # Cleanup
            results_with_id.unpersist()
            
            return corrected_results
            
        except ImportError:
            logger.error("wombat package not available. Please implement capybara statistics module.")
            raise
    
    def save_regression_results(
        self, 
        results: pd.DataFrame, 
        outcome: str, 
        group_type: str, 
        strata_val: str
    ) -> str:
        """
        Save regression results to CSV.
        
        Args:
            results: Regression results DataFrame
            outcome: The outcome of interest
            group_type: Group type
            strata_val: Stratum value
            
        Returns:
            Path where results were saved
        """
        # Get output path from config
        output_path = getattr(
            self.config.paths.analysis.regression_analysis.pooled_analysis.pooled_regression_results,
            f"pooled_regression_{group_type}_{strata_val}"
        )
        
        logger.info(f"Saving regression results to: {output_path}")
        
        # Save results
        results.to_csv(output_path, index=False)
        
        logger.info(f"Successfully saved {len(results)} regression results")
        return output_path
    
    def run_all_regressions(self, outcome: str) -> Dict[str, str]:
        """
        Run regression analysis for all strata and group combinations.
        
        Args:
            outcome: The outcome of interest
            
        Returns:
            Dictionary mapping analysis names to output paths
        """
        logger.info(f"Running all regression analyses for outcome: {outcome}")
        
        # Load input data
        regression_data = self.load_regression_input(outcome)
        global_id_df = self.load_global_id_mapping()
        
        output_paths = {}
        
        # Run regression for each combination
        for strata_val, group_type in itertools.product(self.strata, self.group_types):
            analysis_name = f"{outcome}_{group_type}_{strata_val}"
            logger.info(f"Processing analysis: {analysis_name}")
            
            try:
                # Run logistic regression
                raw_results = self.run_logistic_regression(
                    regression_data, outcome, strata_val, group_type
                )
                
                # Apply multiple testing correction  
                corrected_results = self.apply_multiple_testing_correction(
                    raw_results, global_id_df, outcome, group_type, strata_val
                )
                
                # Save results
                output_path = self.save_regression_results(
                    corrected_results, outcome, group_type, strata_val
                )
                
                output_paths[analysis_name] = output_path
                
            except Exception as e:
                logger.error(f"Error in analysis {analysis_name}: {str(e)}")
                raise
        
        # Cleanup
        regression_data.unpersist()
        
        logger.info(f"Completed all regression analyses for outcome: {outcome}")
        return output_paths
    
    def run(self, outcome: str) -> Dict[str, str]:
        """
        Run the complete regression analysis pipeline.
        
        Args:
            outcome: The outcome of interest
            
        Returns:
            Dictionary mapping analysis names to output paths
        """
        logger.info(f"Starting regression analysis for outcome: {outcome}")
        
        result_paths = self.run_all_regressions(outcome)
        
        logger.info(f"Completed regression analysis for outcome: {outcome}")
        return result_paths


def run_regression_analysis(outcome: str, config_path: str = None, spark_session: SparkSession = None) -> Dict[str, str]:
    """
    Convenience function to run regression analysis.
    
    Args:
        outcome: The outcome of interest
        config_path: Optional path to config file
        spark_session: Optional Spark session
        
    Returns:
        Dictionary mapping analysis names to output paths
    """
    # Initialize Spark session if not provided
    if spark_session is None:
        spark_session = create_spark_session("regression-analysis", enable_delta=True)
    
    # Initialize configuration
    config_manager = ConfigManager()
    config_manager.load_config(outcome=outcome, config_path=config_path)
    
    # Run regression analysis
    analyzer = RegressionAnalyzer(config_manager, spark_session)
    return analyzer.run(outcome)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run regression analysis")
    parser.add_argument("outcome", help="Outcome of interest (sb, eope, lope, ptb, sga)")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    result_paths = run_regression_analysis(args.outcome, args.config)
    
    print(f"Regression analysis completed for outcome: {args.outcome}")
    for analysis_name, path in result_paths.items():
        print(f"  {analysis_name}: {path}")