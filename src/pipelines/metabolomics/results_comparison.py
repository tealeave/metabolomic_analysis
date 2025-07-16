"""
Results Comparison Module

This module compares new regression results with previous analysis results
to validate findings and track changes in biomarker discovery.

Based on the reference implementation from {outcome}-comparing-to-old-reg-results.py
"""

import itertools
import logging
import warnings
from typing import Dict, List, Tuple

import pandas as pd
from pyspark.sql import SparkSession

from ...utils.config_utils import ConfigManager
from ...utils.spark_utils import create_spark_session

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)


class ResultsComparator:
    """
    Handles comparison of new regression results with previous analyses.
    
    This class performs the following steps:
    1. Load new regression results from current analysis
    2. Load previous regression results for comparison
    3. Merge and compare significant biomarkers
    4. Generate comparison reports
    """
    
    def __init__(self, config_manager: ConfigManager, spark: SparkSession = None):
        self.config = config_manager.config
        self.spark = spark
        
        # Analysis strata and group types  
        self.strata = ["lt20", "gte20"]
        self.group_types = ["control", "noncase"]
    
    def load_new_results(self, outcome: str) -> pd.DataFrame:
        """
        Load new regression results from current analysis.
        
        Args:
            outcome: The outcome of interest
            
        Returns:
            Combined DataFrame of significant new results
        """
        logger.info(f"Loading new regression results for outcome: {outcome}")
        
        new_results = []
        
        for strata_val, group_type in itertools.product(self.strata, self.group_types):
            logger.info(f"Loading results for {outcome}_{group_type}_{strata_val}")
            
            # Get file path for this analysis
            result_path = getattr(
                self.config.paths.analysis.regression_analysis.pooled_analysis.pooled_regression_results,
                f"pooled_regression_{group_type}_{strata_val}"
            )
            
            try:
                # Load and filter for significant results
                df = (
                    pd.read_csv(result_path)
                    .query("significant == True")
                    [["unique_feature_label", "group", "compound_id", "compound_class", 
                      "estimate", "p_value", "p_value_corrected"]]
                    .rename(columns={
                        "estimate": "estimate_new",
                        "p_value": "p_value_new", 
                        "p_value_corrected": "p_value_corrected_new"
                    })
                )
                
                new_results.append(df)
                logger.info(f"Loaded {len(df)} significant results from {group_type}_{strata_val}")
                
            except FileNotFoundError:
                logger.warning(f"Results file not found: {result_path}")
            except Exception as e:
                logger.error(f"Error loading results from {result_path}: {str(e)}")
        
        if new_results:
            combined_new = pd.concat(new_results, axis=0, ignore_index=True)
            logger.info(f"Combined {len(combined_new)} total significant new results")
            return combined_new
        else:
            logger.warning("No new results loaded")
            return pd.DataFrame()
    
    def load_previous_results(self, outcome: str) -> pd.DataFrame:
        """
        Load previous regression results for comparison.
        
        Args:
            outcome: The outcome of interest
            
        Returns:
            DataFrame of previous significant results
        """
        logger.info(f"Loading previous regression results for outcome: {outcome}")
        
        previous_path = self.config.paths.analysis.regression_analysis.previous_results.sig_mtbs_with_alignment_info
        
        try:
            previous_results = (
                pd.read_csv(previous_path)
                .assign(
                    # Create matching group format
                    group=lambda x: x["group"].apply(lambda y: "_".join(y.split("_")[2:]).lower())
                )
                .query('analysis == "pooled"')  # Only compare pooled analysis
                [["unique_feature_label", "group", "Coefficient_Intensity", "p_value", "FDR"]]
                .rename(columns={
                    "Coefficient_Intensity": "estimate_old",
                    "p_value": "p_value_old", 
                    "FDR": "p_value_corrected_old"
                })
            )
            
            logger.info(f"Loaded {len(previous_results)} previous significant results")
            return previous_results
            
        except FileNotFoundError:
            logger.warning(f"Previous results file not found: {previous_path}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading previous results: {str(e)}")
            return pd.DataFrame()
    
    def merge_and_compare_results(
        self, 
        new_results: pd.DataFrame, 
        previous_results: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge new and previous results and create comparison indicators.
        
        Args:
            new_results: New significant results
            previous_results: Previous significant results
            
        Returns:
            Merged DataFrame with comparison indicators
        """
        logger.info("Merging and comparing new vs previous results")
        
        if new_results.empty and previous_results.empty:
            logger.warning("Both new and previous results are empty")
            return pd.DataFrame()
        
        # Merge on unique_feature_label and group
        merged_results = (
            new_results.merge(
                previous_results, 
                on=["unique_feature_label", "group"], 
                how="outer", 
                indicator=True
            )
            .assign(
                sig_only_in_new=lambda x: x["_merge"] == "left_only",
                sig_only_in_old=lambda x: x["_merge"] == "right_only", 
                sig_in_both=lambda x: x["_merge"] == "both"
            )
            .drop(columns=["_merge"])
        )
        
        # Log comparison statistics
        only_new = merged_results["sig_only_in_new"].sum()
        only_old = merged_results["sig_only_in_old"].sum()
        in_both = merged_results["sig_in_both"].sum()
        
        logger.info(f"Comparison results:")
        logger.info(f"  Significant only in new: {only_new}")
        logger.info(f"  Significant only in previous: {only_old}")
        logger.info(f"  Significant in both: {in_both}")
        
        return merged_results
    
    def generate_comparison_summary(self, merged_results: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for the comparison.
        
        Args:
            merged_results: Merged comparison results
            
        Returns:
            Summary DataFrame with counts by group and comparison type
        """
        logger.info("Generating comparison summary")
        
        if merged_results.empty:
            return pd.DataFrame()
        
        # Group by comparison indicators and count
        summary = (
            merged_results.groupby(["sig_only_in_new", "sig_only_in_old", "sig_in_both"])
            .size()
            .reset_index(name="count")
        )
        
        # Also create summary by group
        group_summary = (
            merged_results.groupby(["group", "sig_only_in_new", "sig_only_in_old", "sig_in_both"])
            .size()
            .reset_index(name="count")
        )
        
        logger.info("Comparison summary generated")
        return summary, group_summary
    
    def find_notable_biomarkers(self, merged_results: pd.DataFrame, outcome: str) -> pd.DataFrame:
        """
        Find notable biomarkers that appear in both analyses.
        
        Args:
            merged_results: Merged comparison results
            outcome: The outcome of interest
            
        Returns:
            DataFrame of notable biomarkers with annotations
        """
        logger.info("Finding notable biomarkers")
        
        if merged_results.empty:
            return pd.DataFrame()
        
        # Find biomarkers significant in both analyses
        both_significant = merged_results.query("sig_in_both == True").copy()
        
        if both_significant.empty:
            logger.info("No biomarkers found significant in both analyses")
            return pd.DataFrame()
        
        # Filter for biomarkers with compound IDs (annotated)
        annotated_biomarkers = both_significant.query("compound_id.notna()")
        
        # Look for specific biomarkers of interest (example: pregnanolone for stillbirth)
        notable_patterns = {
            "sb": ["preg", "Preg"],  # Pregnanolone-related compounds
            "eope": ["lipid", "Lipid"],  # Lipid-related compounds
            "lope": ["amino", "Amino"],  # Amino acid-related compounds
            "ptb": ["fatty", "Fatty"],  # Fatty acid-related compounds
            "sga": ["glucose", "Glucose"]  # Glucose-related compounds
        }
        
        if outcome in notable_patterns:
            patterns = notable_patterns[outcome]
            pattern_matches = annotated_biomarkers[
                annotated_biomarkers["compound_id"].str.contains(
                    "|".join(patterns), case=False, na=False
                )
            ]
            
            if not pattern_matches.empty:
                logger.info(f"Found {len(pattern_matches)} notable biomarkers matching patterns for {outcome}")
                return pattern_matches
        
        # Return top annotated biomarkers if no pattern matches
        if not annotated_biomarkers.empty:
            top_biomarkers = annotated_biomarkers.nsmallest(10, "p_value_corrected_new")
            logger.info(f"Returning top {len(top_biomarkers)} annotated biomarkers")
            return top_biomarkers
        
        logger.info("No notable annotated biomarkers found")
        return pd.DataFrame()
    
    def save_comparison_results(self, merged_results: pd.DataFrame, outcome: str) -> str:
        """
        Save comparison results to file.
        
        Args:
            merged_results: Merged comparison results
            outcome: The outcome of interest
            
        Returns:
            Path where results were saved
        """
        output_path = self.config.paths.analysis.regression_analysis.comparing_to_old_results.pooled_sig_mtbs_comparison
        
        logger.info(f"Saving comparison results to: {output_path}")
        
        merged_results.to_csv(output_path, index=False)
        
        logger.info(f"Successfully saved {len(merged_results)} comparison results")
        return output_path
    
    def run(self, outcome: str) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
        """
        Run the complete results comparison pipeline.
        
        Args:
            outcome: The outcome of interest
            
        Returns:
            Tuple of (output_path, summary_stats, notable_biomarkers)
        """
        logger.info(f"Starting results comparison for outcome: {outcome}")
        
        # Load new and previous results
        new_results = self.load_new_results(outcome)
        previous_results = self.load_previous_results(outcome)
        
        # Merge and compare
        merged_results = self.merge_and_compare_results(new_results, previous_results)
        
        # Generate summary statistics
        if not merged_results.empty:
            summary_stats, group_summary = self.generate_comparison_summary(merged_results)
            
            # Find notable biomarkers
            notable_biomarkers = self.find_notable_biomarkers(merged_results, outcome)
            
            # Save results
            output_path = self.save_comparison_results(merged_results, outcome)
            
            logger.info(f"Results comparison completed for outcome: {outcome}")
            return output_path, summary_stats, notable_biomarkers
        else:
            logger.warning("No results to compare")
            return None, pd.DataFrame(), pd.DataFrame()


def run_results_comparison(
    outcome: str, 
    config_path: str = None, 
    spark_session: SparkSession = None
) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to run results comparison.
    
    Args:
        outcome: The outcome of interest
        config_path: Optional path to config file
        spark_session: Optional Spark session
        
    Returns:
        Tuple of (output_path, summary_stats, notable_biomarkers)
    """
    # Initialize Spark session if not provided (optional for this module)
    if spark_session is None:
        spark_session = create_spark_session("results-comparison", enable_delta=True)
    
    # Initialize configuration
    config_manager = ConfigManager()
    config_manager.load_config(outcome=outcome, config_path=config_path)
    
    # Run results comparison
    comparator = ResultsComparator(config_manager, spark_session)
    return comparator.run(outcome)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare regression results")
    parser.add_argument("outcome", help="Outcome of interest (sb, eope, lope, ptb, sga)")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    output_path, summary, notable = run_results_comparison(args.outcome, args.config)
    
    if output_path:
        print(f"Results comparison completed for outcome: {args.outcome}")
        print(f"Results saved to: {output_path}")
        
        if not summary.empty:
            print("\nComparison Summary:")
            print(summary.to_string(index=False))
        
        if not notable.empty:
            print(f"\nNotable biomarkers ({len(notable)} found):")
            print(notable[["compound_id", "group", "estimate_new", "p_value_corrected_new"]].to_string(index=False))
    else:
        print(f"No results to compare for outcome: {args.outcome}")