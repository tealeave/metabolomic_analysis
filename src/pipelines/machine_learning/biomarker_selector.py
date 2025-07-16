"""
Biomarker Selection Module

This module handles the selection of biomarkers for machine learning analysis.
It supports multiple selection modes including user-specified lists, top-N significant
biomarkers, and loading from metabolomics pipeline results.

Key Features:
- User-specified biomarker lists per outcome
- Top-N significant biomarker selection
- Loading biomarkers from metabolomics results
- Biomarker validation and availability checking
- Cross-outcome biomarker comparison
"""

import logging
import warnings
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from pyspark.sql import SparkSession

from ...utils.config_utils import ConfigManager

warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = logging.getLogger(__name__)


class BiomarkerSelector:
    """
    Handles biomarker selection for machine learning analysis.
    
    This class supports multiple selection modes:
    1. User-specified biomarker lists
    2. Top-N significant biomarkers from metabolomics results
    3. Loading from metabolomics pipeline output files
    4. Biomarker validation and filtering
    """
    
    def __init__(self, config_manager: ConfigManager, spark_session: SparkSession = None):
        self.config = config_manager.config
        self.spark = spark_session
        
        # Supported outcomes
        self.supported_outcomes = ["eope", "lope", "sb", "ptb", "sga"]
        
        # Selection modes
        self.selection_modes = ["user_specified", "top_n", "from_results", "intersection", "union"]
    
    def validate_outcome(self, outcome: str) -> None:
        """
        Validate that the outcome is supported.
        
        Args:
            outcome: The outcome to validate
            
        Raises:
            ValueError: If outcome is not supported
        """
        if outcome not in self.supported_outcomes:
            raise ValueError(f"Outcome '{outcome}' not supported. Supported outcomes: {self.supported_outcomes}")
    
    def get_user_specified_biomarkers(self, outcome: str) -> List[str]:
        """
        Get user-specified biomarkers for an outcome from configuration.
        
        Args:
            outcome: The outcome of interest
            
        Returns:
            List of user-specified biomarkers
        """
        self.validate_outcome(outcome)
        
        try:
            ml_config = self.config.get("ml", {})
            biomarker_config = ml_config.get("biomarker_selection", {})
            user_biomarkers = biomarker_config.get("user_biomarkers", {})
            
            biomarkers = user_biomarkers.get(outcome, [])
            
            if biomarkers:
                logger.info(f"Found {len(biomarkers)} user-specified biomarkers for {outcome}")
                return biomarkers
            else:
                logger.warning(f"No user-specified biomarkers found for {outcome}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting user-specified biomarkers for {outcome}: {str(e)}")
            return []
    
    def load_metabolomics_results(self, outcome: str) -> pd.DataFrame:
        """
        Load significant biomarkers from metabolomics pipeline results.
        
        Args:
            outcome: The outcome of interest
            
        Returns:
            DataFrame with metabolomics results
        """
        self.validate_outcome(outcome)
        logger.info(f"Loading metabolomics results for outcome: {outcome}")
        
        try:
            # Load regression results from all strata and group combinations
            strata = ["lt20", "gte20"]
            group_types = ["control", "noncase"]
            
            all_results = []
            
            for strata_val in strata:
                for group_type in group_types:
                    result_path = getattr(
                        self.config.paths.analysis.regression_analysis.pooled_analysis.pooled_regression_results,
                        f"pooled_regression_{group_type}_{strata_val}"
                    )
                    
                    try:
                        df = pd.read_csv(result_path)
                        df['strata'] = strata_val
                        df['group_type'] = group_type
                        all_results.append(df)
                        logger.info(f"Loaded {len(df)} results from {group_type}_{strata_val}")
                    except FileNotFoundError:
                        logger.warning(f"Results file not found: {result_path}")
                    except Exception as e:
                        logger.error(f"Error loading {result_path}: {str(e)}")
            
            if all_results:
                combined_results = pd.concat(all_results, ignore_index=True)
                logger.info(f"Combined {len(combined_results)} total metabolomics results")
                return combined_results
            else:
                logger.warning(f"No metabolomics results found for {outcome}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading metabolomics results for {outcome}: {str(e)}")
            return pd.DataFrame()
    
    def get_top_n_biomarkers(
        self, 
        outcome: str, 
        n: int = 100, 
        sort_by: str = "p_value_corrected",
        filter_significant: bool = True
    ) -> List[str]:
        """
        Get top N significant biomarkers from metabolomics results.
        
        Args:
            outcome: The outcome of interest
            n: Number of top biomarkers to select
            sort_by: Column to sort by (default: p_value_corrected)
            filter_significant: Whether to filter for significant results only
            
        Returns:
            List of top N biomarkers
        """
        results_df = self.load_metabolomics_results(outcome)
        
        if results_df.empty:
            logger.warning(f"No metabolomics results available for {outcome}")
            return []
        
        # Filter for significant results if requested
        if filter_significant and "significant" in results_df.columns:
            results_df = results_df[results_df["significant"] == True]
            logger.info(f"Filtered to {len(results_df)} significant results")
        
        # Check if sort column exists
        if sort_by not in results_df.columns:
            logger.warning(f"Sort column '{sort_by}' not found. Available columns: {list(results_df.columns)}")
            sort_by = "p_value" if "p_value" in results_df.columns else results_df.columns[0]
        
        # Sort by specified column (ascending for p-values)
        ascending = True if "p_value" in sort_by.lower() else False
        results_df = results_df.sort_values(by=sort_by, ascending=ascending)
        
        # Get unique biomarkers (remove duplicates across strata/groups)
        unique_biomarkers = results_df["unique_feature_label"].drop_duplicates().head(n).tolist()
        
        logger.info(f"Selected top {len(unique_biomarkers)} biomarkers for {outcome}")
        return unique_biomarkers
    
    def get_biomarkers_by_criteria(
        self,
        outcome: str,
        fdr_threshold: float = 0.05,
        min_effect_size: float = None,
        annotated_only: bool = False
    ) -> List[str]:
        """
        Get biomarkers based on specific statistical criteria.
        
        Args:
            outcome: The outcome of interest
            fdr_threshold: FDR threshold for significance
            min_effect_size: Minimum absolute effect size (coefficient)
            annotated_only: Whether to include only annotated biomarkers
            
        Returns:
            List of biomarkers meeting criteria
        """
        results_df = self.load_metabolomics_results(outcome)
        
        if results_df.empty:
            return []
        
        # Apply filters
        filtered_df = results_df.copy()
        
        # FDR threshold
        if "p_value_corrected" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["p_value_corrected"] <= fdr_threshold]
            logger.info(f"Filtered to {len(filtered_df)} biomarkers with FDR <= {fdr_threshold}")
        
        # Effect size threshold
        if min_effect_size is not None and "estimate" in filtered_df.columns:
            filtered_df = filtered_df[abs(filtered_df["estimate"]) >= min_effect_size]
            logger.info(f"Filtered to {len(filtered_df)} biomarkers with |effect| >= {min_effect_size}")
        
        # Annotated only
        if annotated_only and "compound_id" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["compound_id"].notna()]
            logger.info(f"Filtered to {len(filtered_df)} annotated biomarkers")
        
        # Get unique biomarkers
        unique_biomarkers = filtered_df["unique_feature_label"].drop_duplicates().tolist()
        
        logger.info(f"Selected {len(unique_biomarkers)} biomarkers meeting criteria for {outcome}")
        return unique_biomarkers
    
    def validate_biomarker_availability(
        self, 
        biomarkers: List[str], 
        outcome: str,
        dataset_type: str = "biosample"
    ) -> Tuple[List[str], List[str]]:
        """
        Validate that biomarkers are available in the dataset.
        
        Args:
            biomarkers: List of biomarker names to validate
            outcome: The outcome of interest
            dataset_type: Type of dataset (biosample, subject, window)
            
        Returns:
            Tuple of (available_biomarkers, missing_biomarkers)
        """
        logger.info(f"Validating availability of {len(biomarkers)} biomarkers")
        
        try:
            # Load intensity data to check availability
            ml_config = self.config.get("ml", {})
            data_path = f"data/ml_datasets/{outcome}/intensity_{dataset_type}.csv"
            
            # Try to load a sample of the intensity data
            intensity_df = pd.read_csv(data_path, nrows=1)  # Just read header
            available_features = set(intensity_df.columns)
            
            # Check biomarker availability
            biomarker_set = set(biomarkers)
            available_biomarkers = list(biomarker_set.intersection(available_features))
            missing_biomarkers = list(biomarker_set - available_features)
            
            logger.info(f"Found {len(available_biomarkers)} available, {len(missing_biomarkers)} missing biomarkers")
            
            if missing_biomarkers:
                logger.warning(f"Missing biomarkers: {missing_biomarkers[:10]}...")  # Show first 10
            
            return available_biomarkers, missing_biomarkers
            
        except Exception as e:
            logger.error(f"Error validating biomarker availability: {str(e)}")
            # Return all as missing if we can't validate
            return [], biomarkers
    
    def get_cross_outcome_biomarkers(
        self, 
        outcomes: List[str], 
        operation: str = "intersection",
        **kwargs
    ) -> Dict[str, List[str]]:
        """
        Get biomarkers that are common across multiple outcomes.
        
        Args:
            outcomes: List of outcomes to analyze
            operation: Operation to perform ("intersection", "union")
            **kwargs: Additional arguments for biomarker selection
            
        Returns:
            Dictionary mapping operation results and individual outcome results
        """
        logger.info(f"Getting cross-outcome biomarkers for {outcomes} using {operation}")
        
        outcome_biomarkers = {}
        
        # Get biomarkers for each outcome
        for outcome in outcomes:
            try:
                self.validate_outcome(outcome)
                biomarkers = self.get_top_n_biomarkers(outcome, **kwargs)
                outcome_biomarkers[outcome] = set(biomarkers)
                logger.info(f"Found {len(biomarkers)} biomarkers for {outcome}")
            except Exception as e:
                logger.error(f"Error getting biomarkers for {outcome}: {str(e)}")
                outcome_biomarkers[outcome] = set()
        
        # Perform cross-outcome operation
        if operation == "intersection":
            if outcome_biomarkers:
                common_biomarkers = set.intersection(*outcome_biomarkers.values())
                logger.info(f"Found {len(common_biomarkers)} common biomarkers across all outcomes")
            else:
                common_biomarkers = set()
        elif operation == "union":
            if outcome_biomarkers:
                common_biomarkers = set.union(*outcome_biomarkers.values())
                logger.info(f"Found {len(common_biomarkers)} total unique biomarkers across all outcomes")
            else:
                common_biomarkers = set()
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Convert back to lists
        result = {
            f"{operation}_biomarkers": list(common_biomarkers),
            "individual_outcomes": {outcome: list(biomarkers) for outcome, biomarkers in outcome_biomarkers.items()}
        }
        
        return result
    
    def select_biomarkers(
        self, 
        outcome: str, 
        mode: str = "auto", 
        **kwargs
    ) -> List[str]:
        """
        Main biomarker selection function that chooses the appropriate method.
        
        Args:
            outcome: The outcome of interest
            mode: Selection mode ("user_specified", "top_n", "from_results", "auto")
            **kwargs: Additional arguments for specific selection methods
            
        Returns:
            List of selected biomarkers
        """
        self.validate_outcome(outcome)
        logger.info(f"Selecting biomarkers for {outcome} using mode: {mode}")
        
        if mode == "user_specified":
            biomarkers = self.get_user_specified_biomarkers(outcome)
        elif mode == "top_n":
            n = kwargs.get("n", 100)
            biomarkers = self.get_top_n_biomarkers(outcome, n=n)
        elif mode == "criteria":
            biomarkers = self.get_biomarkers_by_criteria(outcome, **kwargs)
        elif mode == "auto":
            # Auto mode: try user-specified first, then top-N
            biomarkers = self.get_user_specified_biomarkers(outcome)
            if not biomarkers:
                logger.info("No user-specified biomarkers found, falling back to top-N")
                biomarkers = self.get_top_n_biomarkers(outcome, n=kwargs.get("n", 100))
        else:
            raise ValueError(f"Unknown selection mode: {mode}")
        
        # Validate availability if requested
        if kwargs.get("validate_availability", True):
            available, missing = self.validate_biomarker_availability(
                biomarkers, outcome, kwargs.get("dataset_type", "biosample")
            )
            if missing:
                logger.warning(f"Some biomarkers not available in dataset: {len(missing)} missing")
            biomarkers = available
        
        logger.info(f"Final selection: {len(biomarkers)} biomarkers for {outcome}")
        return biomarkers
    
    def save_biomarker_selection(
        self, 
        biomarkers: List[str], 
        outcome: str, 
        output_path: str = None
    ) -> str:
        """
        Save biomarker selection to file.
        
        Args:
            biomarkers: List of selected biomarkers
            outcome: The outcome of interest
            output_path: Optional output path (default: auto-generated)
            
        Returns:
            Path where biomarkers were saved
        """
        if output_path is None:
            output_path = f"data/ml_datasets/{outcome}/selected_biomarkers.txt"
        
        logger.info(f"Saving {len(biomarkers)} biomarkers to: {output_path}")
        
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save biomarkers
        with open(output_path, 'w') as f:
            for biomarker in biomarkers:
                f.write(f"{biomarker}\n")
        
        logger.info(f"Successfully saved biomarker selection")
        return output_path


def select_biomarkers_for_outcome(
    outcome: str,
    config_manager: ConfigManager = None,
    mode: str = "auto",
    **kwargs
) -> List[str]:
    """
    Convenience function to select biomarkers for an outcome.
    
    Args:
        outcome: The outcome of interest
        config_manager: Optional configuration manager
        mode: Selection mode
        **kwargs: Additional arguments
        
    Returns:
        List of selected biomarkers
    """
    if config_manager is None:
        config_manager = ConfigManager()
        config_manager.load_config(outcome=outcome)
    
    selector = BiomarkerSelector(config_manager)
    return selector.select_biomarkers(outcome, mode, **kwargs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Select biomarkers for ML analysis")
    parser.add_argument("outcome", help="Outcome of interest (sb, eope, lope, ptb, sga)")
    parser.add_argument("--mode", default="auto", 
                       choices=["user_specified", "top_n", "criteria", "auto"],
                       help="Selection mode")
    parser.add_argument("--n", type=int, default=100, help="Number of top biomarkers to select")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config_manager = ConfigManager()
    config_manager.load_config(outcome=args.outcome, config_path=args.config)
    
    # Select biomarkers
    selector = BiomarkerSelector(config_manager)
    biomarkers = selector.select_biomarkers(args.outcome, mode=args.mode, n=args.n)
    
    # Save if output specified
    if args.output:
        selector.save_biomarker_selection(biomarkers, args.outcome, args.output)
    
    print(f"Selected {len(biomarkers)} biomarkers for {args.outcome}:")
    for i, biomarker in enumerate(biomarkers[:10], 1):  # Show first 10
        print(f"  {i}. {biomarker}")
    
    if len(biomarkers) > 10:
        print(f"  ... and {len(biomarkers) - 10} more")