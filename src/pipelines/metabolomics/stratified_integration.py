"""
Stratified Integration Module

This module performs stratified metadata integration and merges with intensity data
for regression analysis. It creates combined datasets stratified by gestational age.

Based on the reference implementation from {outcome}-strata-mtb-input.py
"""

import logging
import warnings
from functools import reduce
from typing import Dict, List, Tuple

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as spark_fx

from ...utils.config_utils import ConfigManager
from ...utils.spark_utils import create_spark_session

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)


class StratifiedIntegrator:
    """
    Handles stratified integration of metadata and intensity data.
    
    This class performs the following steps:
    1. Read case, non-case, and control cohorts
    2. Create stratified metadata based on gestational age
    3. Merge with intensity data
    4. Prepare regression-ready datasets
    """
    
    def __init__(self, config_manager: ConfigManager, spark: SparkSession):
        self.config = config_manager.config
        self.spark = spark
        
        # Define columns needed for regression analysis
        self.regression_columns = [
            "sample_id", "orig_id", "site", "pe_new", "pe_priority", "pe_cat",
            "sb_new", "ptb_new", "severe_ptb", "sga_3", "sga_10", "mat_height",
            "mat_weight", "bwt_centile_new", "chron_htn", "diabetes", "pw_age", "bmi"
        ]
        
        # Site name mappings
        self.site_mappings = {
            "AMANHIB": "amanhi_bangladesh",
            "GAPPSB": "gapps_bangladesh", 
            "ZAPPS": "gapps_zambia",
            "AMANHIP": "amanhi_pakistan",
            "AMANHIT": "amanhi_tanzania"
        }
    
    def load_cohort_data(self, outcome: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load case, non-case, and control cohorts from CSV files.
        
        Args:
            outcome: The outcome of interest (e.g., 'sb', 'eope', 'lope', 'ptb', 'sga')
            
        Returns:
            Tuple of (cases, non_cases, controls) DataFrames
        """
        logger.info(f"Loading cohort data for outcome: {outcome}")
        
        # Get file paths from config
        case_path = self.config.paths.combined_manifest.flowchart_counts.case_df
        non_case_path = self.config.paths.combined_manifest.flowchart_counts.non_case_df
        control_path = self.config.paths.combined_manifest.flowchart_counts.control_df
        
        # Load the cohort dataframes
        cases = pd.read_csv(case_path)
        non_cases = pd.read_csv(non_case_path)
        controls = pd.read_csv(control_path)
        
        logger.info(f"Loaded {len(cases)} cases, {len(non_cases)} non-cases, {len(controls)} controls")
        
        return cases, non_cases, controls
    
    def create_stratified_metadata(
        self, 
        cases: pd.DataFrame, 
        non_cases: pd.DataFrame, 
        controls: pd.DataFrame,
        outcome: str
    ) -> pd.DataFrame:
        """
        Create stratified metadata based on gestational age (<20 weeks vs >=20 weeks).
        
        Args:
            cases: Case cohort DataFrame
            non_cases: Non-case cohort DataFrame  
            controls: Control cohort DataFrame
            outcome: The outcome of interest
            
        Returns:
            Stratified metadata DataFrame
        """
        logger.info("Creating stratified metadata")
        
        # Define GA stratification condition
        base_condition = ("ga_weeks < 20", "ga_weeks >= 20")
        suffixes = ["", "_noncase", "_control"]
        conditions = [base_condition] * len(suffixes)
        group_names = [(f"{outcome}{suffix}_lt20", f"{outcome}{suffix}_gte20") for suffix in suffixes]
        
        # Ordered DataFrames
        ordered_dfs = [cases, non_cases, controls]
        
        # Create combined stratified metadata
        stratified_metadata = self._create_combined_data(
            dfs=ordered_dfs,
            conditions=conditions,
            group_names=group_names,
            additional_columns=self.regression_columns
        )
        
        return stratified_metadata
    
    def _create_combined_data(
        self,
        dfs: List[pd.DataFrame],
        conditions: List[Tuple[str, str]], 
        group_names: List[Tuple[str, str]],
        additional_columns: List[str]
    ) -> pd.DataFrame:
        """
        Create combined data with stratification.
        
        Args:
            dfs: List of DataFrames to combine
            conditions: List of condition tuples for stratification
            group_names: List of group name tuples  
            additional_columns: Additional columns to include
            
        Returns:
            Combined DataFrame with group labels
        """
        combined_data = []
        
        for df, condition_tuple, group_tuple in zip(dfs, conditions, group_names):
            for condition, group_name in zip(condition_tuple, group_tuple):
                # Filter based on condition
                filtered_df = df.query(condition).copy()
                filtered_df['group'] = group_name
                
                # Select required columns
                cols_to_select = ['group'] + additional_columns
                available_cols = [col for col in cols_to_select if col in filtered_df.columns]
                filtered_df = filtered_df[available_cols]
                
                combined_data.append(filtered_df)
        
        return pd.concat(combined_data, ignore_index=True)
    
    def check_sample_uniqueness(self, metadata: pd.DataFrame, outcome: str) -> None:
        """
        Check sample ID uniqueness within each stratum.
        
        Args:
            metadata: Stratified metadata DataFrame
            outcome: The outcome of interest
        """
        logger.info("Checking sample ID uniqueness")
        
        for group in metadata['group'].unique():
            group_data = metadata[metadata['group'] == group]
            duplicates = group_data['sample_id'].duplicated().sum()
            
            if duplicates > 0:
                logger.warning(f"Found {duplicates} duplicate sample IDs in group {group}")
            else:
                logger.info(f"No duplicate sample IDs found in group {group}")
    
    def load_intensity_data(self) -> DataFrame:
        """
        Load and combine intensity data from all sites.
        
        Returns:
            Combined intensity data as Spark DataFrame
        """
        logger.info("Loading intensity data from all sites")
        
        # Create list of intensity DataFrames for each site
        intensity_dfs = []
        for site_code, site_name in self.site_mappings.items():
            logger.info(f"Loading intensity data for site: {site_code}")
            
            # Get intensity data path from config
            intensity_path = getattr(self.config.sites, site_name).intensity_data
            
            # Load intensity data and add site column
            site_df = (self.spark.read.format("delta")
                      .load(intensity_path)
                      .withColumn("site", spark_fx.lit(site_code)))
            
            intensity_dfs.append(site_df)
        
        # Combine all site DataFrames
        combined_intensities = reduce(DataFrame.union, intensity_dfs)
        
        logger.info("Successfully loaded and combined intensity data from all sites")
        return combined_intensities
    
    def load_feature_alignment(self) -> DataFrame:
        """
        Load the melted feature alignment data.
        
        Returns:
            Feature alignment DataFrame
        """
        logger.info("Loading feature alignment data")
        
        alignment_path = self.config.paired_alignment.alignment.melted_paired_alignment_uniq_to_site_long_format
        
        melted_feature_df = (
            self.spark.read.format("delta")
            .load(alignment_path)
            .withColumn(
                "feature_label",
                spark_fx.when(
                    spark_fx.col("site") == "AMANHIP",
                    spark_fx.element_at(spark_fx.split(spark_fx.col("feature_label"), "_"), -1)
                ).otherwise(spark_fx.col("feature_label"))
            )
            .select("site", "feature_label", "unique_feature_label")
        )
        
        return melted_feature_df
    
    def merge_intensity_with_alignment(
        self, 
        intensity_data: DataFrame, 
        alignment_data: DataFrame
    ) -> DataFrame:
        """
        Merge intensity data with feature alignment.
        
        Args:
            intensity_data: Combined intensity data
            alignment_data: Feature alignment data
            
        Returns:
            Merged DataFrame with aligned features
        """
        logger.info("Merging intensity data with feature alignment")
        
        merged_data = (intensity_data
                      .join(alignment_data, on=["site", "feature_label"], how="inner")
                      .cache())
        
        # Force materialization and get counts
        total_records = merged_data.count()
        unique_features = merged_data.select("unique_feature_label").distinct().count()
        
        logger.info(f"Merged data contains {total_records} records with {unique_features} unique features")
        
        return merged_data
    
    def create_regression_input(
        self, 
        stratified_metadata: pd.DataFrame, 
        intensity_data: DataFrame,
        outcome: str
    ) -> DataFrame:
        """
        Create final regression input by merging metadata with intensity data.
        
        Args:
            stratified_metadata: Stratified metadata DataFrame
            intensity_data: Intensity data with aligned features
            outcome: The outcome of interest
            
        Returns:
            Final regression-ready DataFrame
        """
        logger.info("Creating regression input data")
        
        # Convert metadata to Spark DataFrame
        metadata_spark = self.spark.createDataFrame(stratified_metadata)
        
        # Merge metadata with intensity data
        regression_input = (metadata_spark
                           .join(intensity_data.drop("site"), on="sample_id", how="inner"))
        
        # Cache the result
        regression_input = regression_input.cache()
        
        # Get final counts
        total_samples = regression_input.count()
        logger.info(f"Created regression input with {total_samples} samples")
        
        return regression_input
    
    def save_regression_input(self, regression_data: DataFrame, outcome: str) -> str:
        """
        Save regression input data to Delta format.
        
        Args:
            regression_data: Regression-ready DataFrame
            outcome: The outcome of interest
            
        Returns:
            Path where data was saved
        """
        output_path = self.config.paths.analysis.regression_analysis.pooled_analysis.input_delta_tables.path
        
        logger.info(f"Saving regression input to: {output_path}")
        
        (regression_data.write
         .format("delta")
         .mode("overwrite")
         .option("overwriteSchema", "true")
         .save(output_path))
        
        logger.info("Successfully saved regression input data")
        return output_path
    
    def run(self, outcome: str) -> str:
        """
        Run the complete stratified integration pipeline.
        
        Args:
            outcome: The outcome of interest
            
        Returns:
            Path to saved regression input data
        """
        logger.info(f"Starting stratified integration for outcome: {outcome}")
        
        # Step 1: Load cohort data
        cases, non_cases, controls = self.load_cohort_data(outcome)
        
        # Step 2: Create stratified metadata
        stratified_metadata = self.create_stratified_metadata(cases, non_cases, controls, outcome)
        
        # Step 3: Check sample uniqueness
        self.check_sample_uniqueness(stratified_metadata, outcome)
        
        # Step 4: Load intensity data
        intensity_data = self.load_intensity_data()
        
        # Step 5: Load feature alignment
        alignment_data = self.load_feature_alignment()
        
        # Step 6: Merge intensity with alignment
        merged_intensity = self.merge_intensity_with_alignment(intensity_data, alignment_data)
        
        # Step 7: Create regression input
        regression_input = self.create_regression_input(stratified_metadata, merged_intensity, outcome)
        
        # Step 8: Save regression input
        output_path = self.save_regression_input(regression_input, outcome)
        
        # Cleanup
        merged_intensity.unpersist()
        regression_input.unpersist()
        
        logger.info(f"Completed stratified integration for outcome: {outcome}")
        return output_path


def run_stratified_integration(outcome: str, config_path: str = None, spark_session: SparkSession = None) -> str:
    """
    Convenience function to run stratified integration.
    
    Args:
        outcome: The outcome of interest
        config_path: Optional path to config file
        spark_session: Optional Spark session
        
    Returns:
        Path to saved regression input data
    """
    # Initialize Spark session if not provided
    if spark_session is None:
        spark_session = create_spark_session("stratified-integration", enable_delta=True)
    
    # Initialize configuration
    config_manager = ConfigManager()
    config_manager.load_config(outcome=outcome, config_path=config_path)
    
    # Run stratified integration
    integrator = StratifiedIntegrator(config_manager, spark_session)
    return integrator.run(outcome)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run stratified integration")
    parser.add_argument("outcome", help="Outcome of interest (sb, eope, lope, ptb, sga)")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    result_path = run_stratified_integration(args.outcome, args.config)
    print(f"Stratified integration completed. Results saved to: {result_path}")