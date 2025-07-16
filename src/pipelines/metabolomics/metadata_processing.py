"""
Metadata Processing Module for Metabolomic Analysis Pipeline.
Equivalent to {outcome}-process-metadata-new.py from the reference pipeline.

This module processes metadata following Kim's flowchart decision tree,
including binary column imputation and outcome-specific filtering.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.config_utils import ConfigManager, create_output_directory


class MetadataProcessor:
    """
    Processes metadata for metabolomic analysis pipeline.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize metadata processor.
        
        Args:
            config_manager: Configured ConfigManager instance
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
    def load_raw_metadata(self) -> pd.DataFrame:
        """
        Load and preprocess raw metadata.
        
        Returns:
            Preprocessed metadata DataFrame
        """
        self.logger.info("Loading raw metadata...")
        
        # Get metadata path from configuration
        metadata_path = self.config_manager.get_path("paths.metadata.dec_2024_metadata")
        
        # Load and preprocess metadata
        raw_metadata = (
            pd.read_csv(metadata_path, low_memory=False)
            .rename(columns=str.lower)
            .assign(
                orig_id=lambda x: x["orig_id"].astype(str).str.strip(),
                severe_ptb=lambda x: x["very_ptb"].astype(bool) | x["ext_ptb"].astype(bool),
                visitdt=lambda x: pd.to_datetime(x["visitdt"], format="%d%b%Y"),
                site=lambda x: x["participant_id"].str.split("-").str[0],
                eope=lambda x: x["pe_cat"] == "EOPE",
                lope=lambda x: x["pe_cat"] == "LOPE",
            )
            .query('site != "THSTI"')
        )
        
        self.logger.info(f"Loaded metadata with shape: {raw_metadata.shape}")
        return raw_metadata
    
    def clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean numeric columns by replacing negative values with NaN.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info("Cleaning numeric columns...")
        
        # Replace negative numeric values with NaN
        numeric_cols = df.select_dtypes(include=["number"]).columns
        df[numeric_cols] = df[numeric_cols].mask(df[numeric_cols] < 0, np.nan)
        
        return df
    
    def determine_fill_value(self, values: pd.Series) -> float:
        """
        Determine fill value for binary imputation.
        
        Args:
            values: Series of values to analyze
            
        Returns:
            Fill value or NaN if ambiguous
        """
        unique_vals = values.dropna().unique()
        if len(unique_vals) == 1 and unique_vals[0] in [0, 1]:
            return unique_vals[0]
        else:
            return np.nan
    
    def impute_binary_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute binary columns using subject-level logic.
        
        Args:
            df: DataFrame with binary columns to impute
            
        Returns:
            DataFrame with imputed binary columns
        """
        self.logger.info("Imputing binary columns...")
        
        # Get binary columns to impute from configuration
        binary_cols_to_impute = self.config_manager.get_binary_columns_to_impute()
        
        # Pre-imputation check
        self.logger.info("Pre-imputation missing value counts:")
        for col in binary_cols_to_impute:
            if col in df.columns:
                na_count = df[col].isna().sum()
                if na_count == 0:
                    self.logger.info(f"  {col}: No missing values")
                else:
                    self.logger.info(f"  {col}: {na_count} missing values")
        
        # Calculate derived fields if needed
        df = self.calculate_derived_fields(df)
        
        # Determine fill values for binary columns
        self.logger.info("Determining fill values for binary columns:")
        fill_values = {}
        for col in binary_cols_to_impute:
            if col in df.columns:
                self.logger.info(f"  Processing column: {col}")
                col_fill_values = df.groupby(["site", "orig_id"])[col].apply(self.determine_fill_value)
                fill_values[col] = col_fill_values
        
        # Build fill DataFrame
        if fill_values:
            self.logger.info("Building fill DataFrame...")
            fill_df = pd.DataFrame(index=list(fill_values.values())[0].index)
            for col in binary_cols_to_impute:
                if col in fill_values:
                    fill_df[col] = fill_values[col]
            
            # Convert index to columns for merging
            fill_df = fill_df.reset_index()
            
            # Merge fill values back to main DataFrame
            self.logger.info("Merging fill values back to metadata...")
            df = df.merge(
                fill_df,
                on=["site", "orig_id"],
                suffixes=("", "_fill"),
            )
            
            # Impute missing binary columns
            self.logger.info("Imputing missing binary columns...")
            for col in binary_cols_to_impute:
                if col in df.columns:
                    fill_col = col + "_fill"
                    if fill_col in df.columns:
                        # Fill missing only where we have a valid fill value (0 or 1)
                        mask = df[fill_col].notna() & df[col].isna()
                        before_count = df[col].isna().sum()
                        
                        df.loc[mask, col] = df.loc[mask, fill_col]
                        df.drop(columns=fill_col, inplace=True)
                        
                        after_count = df[col].isna().sum()
                        filled = before_count - after_count
                        self.logger.info(f"  Column '{col}': {filled} values imputed. Remaining NaN: {after_count}")
        
        self.logger.info("Binary column imputation complete.")
        return df
    
    def calculate_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived fields from metadata.
        
        Args:
            df: DataFrame to add derived fields to
            
        Returns:
            DataFrame with derived fields
        """
        return df.assign(
            ga_weeks=lambda x: (x["ga_hdlk_new"] + 6) // 7,
            gagebrth_weeks=lambda x: (x["gagebrth_new"] + 6) // 7,
            bmi=lambda x: x["mat_weight"] / (x["mat_height"] / 100) ** 2,
        )
    
    def load_and_combine_manifests(self) -> pd.DataFrame:
        """
        Load and combine manifests from all sites.
        
        Returns:
            Combined manifests DataFrame
        """
        self.logger.info("Loading and combining manifests...")
        
        # Get manifest paths from configuration
        manifest_paths = [
            {
                "name": "GAPPSB",
                "path": self.config_manager.get_path("paths.manifest_kim.gapps_bangladesh"),
            },
            {
                "name": "AMANHIB",
                "path": self.config_manager.get_path("paths.manifest_kim.amanhi_bangladesh"),
            },
            {
                "name": "AMANHIT",
                "path": self.config_manager.get_path("paths.manifest_kim.amanhi_tanzania"),
            },
            {
                "name": "AMANHIP",
                "path": self.config_manager.get_path("paths.manifest_kim.amanhi_pakistan"),
            },
            {
                "name": "ZAPPS",
                "path": self.config_manager.get_path("paths.manifest_kim.gapps_zambia_05_07_2024"),
            },
        ]
        
        # Load and filter manifests
        site_manifest_dfs = []
        for manifest_info in manifest_paths:
            site_name = manifest_info["name"]
            manifest_path = manifest_info["path"]
            
            self.logger.info(f"Reading manifest for site: {site_name}")
            
            # Read manifest based on file extension
            if manifest_path.endswith(".xlsx"):
                manifest_df = pd.read_excel(manifest_path)
            else:
                manifest_df = pd.read_csv(manifest_path)
            
            # Filter and clean manifest
            manifest_df = (
                manifest_df
                .rename(columns={"VISIT_DT": "VISITDT"} if "VISIT_DT" in manifest_df.columns else {})
                .rename(columns=str.lower)
                .assign(
                    visitdt=lambda x: pd.to_datetime(x["visitdt"]),
                    orig_id=lambda x: x["orig_id"].astype(str).str.strip(),
                    site=site_name,
                )
                .query('sample_type == "Maternal Plasma" and note != "Duplicate-remove"')
                .loc[:, ~manifest_df.columns.str.contains("^Unnamed")]
            )
            
            site_manifest_dfs.append(manifest_df)
        
        # Combine all manifests
        combined_manifests = pd.concat(site_manifest_dfs, ignore_index=True)
        
        self.logger.info(f"Combined manifests shape: {combined_manifests.shape}")
        return combined_manifests
    
    def filter_by_manifest(self, metadata_df: pd.DataFrame, manifest_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter metadata to only include samples present in manifest.
        
        Args:
            metadata_df: Metadata DataFrame
            manifest_df: Manifest DataFrame
            
        Returns:
            Filtered metadata DataFrame
        """
        self.logger.info("Filtering metadata by manifest...")
        
        # Merge with manifest to filter samples
        filtered_df = metadata_df.merge(
            manifest_df[["site", "orig_id", "visitdt", "sample_id"]],
            on=["site", "orig_id", "visitdt"],
            how="inner",
        )
        
        self.logger.info(f"Filtered metadata shape: {filtered_df.shape}")
        return filtered_df
    
    def count_unique_subjects(self, df: pd.DataFrame) -> int:
        """
        Count unique combinations of site and orig_id.
        
        Args:
            df: DataFrame to count subjects in
            
        Returns:
            Number of unique subjects
        """
        return df.groupby(["site", "orig_id"]).size().reset_index(name="count").shape[0]
    
    def apply_stillbirth_flowchart(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Apply stillbirth-specific flowchart logic.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (cases, non_cases, controls) DataFrames
        """
        self.logger.info("Applying stillbirth flowchart logic...")
        
        # Initial cohort count
        base_subjects = self.count_unique_subjects(df)
        self.logger.info(f"Initial total subjects: {base_subjects}")
        
        # Apply filters step by step
        current_df = df.copy()
        
        # 1. Single births filter
        current_df = current_df.query("single_twin == 1")
        single_counts = self.count_unique_subjects(current_df)
        self.logger.info(f"After excluding multiple pregnancies: {single_counts}")
        
        # 2. GA filter
        current_df = current_df.query("~ga_hdlk_new.isna()")
        ga_counts = self.count_unique_subjects(current_df)
        self.logger.info(f"After excluding missing GA: {ga_counts}")
        
        # 3. Maternal age filter
        current_df = current_df.query("~pw_age.isna() and ~pw_age.isin([-77, -88, -99])")
        pw_age_counts = self.count_unique_subjects(current_df)
        self.logger.info(f"After excluding missing maternal age: {pw_age_counts}")
        
        # 4. Birth outcome filter
        current_df = current_df.query("birth_outcome.notna() and ~birth_outcome.isin([-77, -88])")
        birth_outcome_counts = self.count_unique_subjects(current_df)
        self.logger.info(f"After excluding missing/invalid birth outcome: {birth_outcome_counts}")
        
        # 5. SB priority filter
        current_df = current_df.query("sb_priority != 3")
        sb_priority_counts = self.count_unique_subjects(current_df)
        self.logger.info(f"After excluding sb_priority 3: {sb_priority_counts}")
        
        # 6. Define cases
        cases = current_df.query("sb_new == 1 and sb_priority in [1,2]")
        case_counts = self.count_unique_subjects(cases)
        self.logger.info(f"Stillbirth cases (priority 1 & 2): {case_counts}")
        
        # 7. Define non-cases
        non_cases = current_df.query("birth_outcome == 1 and sb_new == 0")
        non_cases = non_cases.drop_duplicates(subset=["site", "orig_id", "visitdt"], keep="first")
        non_case_counts = self.count_unique_subjects(non_cases)
        self.logger.info(f"Non-cases (livebirth): {non_case_counts}")
        
        # 8. Define controls with exclusions
        controls = non_cases.copy()
        
        # Apply control exclusions
        exclusions = [
            ("gagebrth_new > 273 and gagebrth_new <= 287", "Gestational age 39-41 weeks"),
            ("birth_weight >= 2500", "Birth weight ≥2,500g"),
            ("chron_htn != 1", "No chronic hypertension"),
            ("diabetes != 1", "No diabetes"),
            ("fetal_anomalies != 1 and cong_anomalies != 1", "No anomalies"),
            ("syphilis != 1 and hiv != 1 and malaria != 1 and tb != 1", "No infectious diseases"),
        ]
        
        for query, description in exclusions:
            before_count = self.count_unique_subjects(controls)
            controls = controls.query(query)
            after_count = self.count_unique_subjects(controls)
            excluded = before_count - after_count
            self.logger.info(f"  {description}: {excluded} subjects excluded")
        
        control_counts = self.count_unique_subjects(controls)
        self.logger.info(f"Final controls: {control_counts}")
        
        # Final summary
        self.logger.info("=== Final Counts ===")
        self.logger.info(f"Cases (Stillbirth priority 1 & 2): {case_counts}")
        self.logger.info(f"Non-cases (All livebirths): {non_case_counts}")
        self.logger.info(f"Controls (Healthy livebirths): {control_counts}")
        
        return cases, non_cases, controls
    
    def save_cohort_dataframes(self, cases: pd.DataFrame, non_cases: pd.DataFrame, controls: pd.DataFrame) -> None:
        """
        Save case, non-case, and control DataFrames.
        
        Args:
            cases: Cases DataFrame
            non_cases: Non-cases DataFrame
            controls: Controls DataFrame
        """
        self.logger.info("Saving cohort DataFrames...")
        
        # Get output paths from configuration
        case_path = self.config_manager.get_path("paths.combined_manifest.flowchart_counts.case_df")
        non_case_path = self.config_manager.get_path("paths.combined_manifest.flowchart_counts.non_case_df")
        control_path = self.config_manager.get_path("paths.combined_manifest.flowchart_counts.control_df")
        
        # Create output directories
        for path in [case_path, non_case_path, control_path]:
            create_output_directory(path)
        
        # Save DataFrames
        cases.to_csv(case_path, index=False)
        non_cases.to_csv(non_case_path, index=False)
        controls.to_csv(control_path, index=False)
        
        self.logger.info(f"Saved cases to: {case_path}")
        self.logger.info(f"Saved non-cases to: {non_case_path}")
        self.logger.info(f"Saved controls to: {control_path}")
    
    def run(self, outcome: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Run the complete metadata processing pipeline.
        
        Args:
            outcome: Outcome name (sb, ptb, sga, etc.)
            
        Returns:
            Tuple of (cases, non_cases, controls) DataFrames
        """
        self.logger.info(f"Starting metadata processing for outcome: {outcome}")
        
        # Load and preprocess metadata
        metadata_df = self.load_raw_metadata()
        metadata_df = self.clean_numeric_columns(metadata_df)
        metadata_df = self.impute_binary_columns(metadata_df)
        
        # Load and combine manifests
        manifest_df = self.load_and_combine_manifests()
        
        # Filter metadata by manifest
        filtered_df = self.filter_by_manifest(metadata_df, manifest_df)
        
        # Apply outcome-specific logic
        if outcome == "sb":
            cases, non_cases, controls = self.apply_stillbirth_flowchart(filtered_df)
        else:
            # For other outcomes, implement specific logic as needed
            raise NotImplementedError(f"Outcome {outcome} not yet implemented")
        
        # Save cohort DataFrames
        self.save_cohort_dataframes(cases, non_cases, controls)
        
        self.logger.info("Metadata processing complete")
        return cases, non_cases, controls


def process_metadata(outcome: str, config_dir: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to process metadata.
    
    Args:
        outcome: Outcome name (sb, ptb, sga, etc.)
        config_dir: Configuration directory path
        
    Returns:
        Tuple of (cases, non_cases, controls) DataFrames
    """
    # Initialize configuration manager
    config_manager = ConfigManager(config_dir)
    config_manager.load_config(outcome=outcome)
    
    # Create metadata processor
    processor = MetadataProcessor(config_manager)
    
    # Process metadata
    return processor.run(outcome)


def main():
    """
    Main function for running metadata processing.
    """
    import argparse
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Process metadata for metabolomic analysis")
    parser.add_argument("outcome", help="Outcome name (sb, ptb, sga, etc.)")
    parser.add_argument("--config-dir", help="Configuration directory path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Process metadata
    try:
        cases, non_cases, controls = process_metadata(args.outcome, args.config_dir)
        
        print(f"Metadata processing completed for outcome: {args.outcome}")
        print(f"Cases: {len(cases)}")
        print(f"Non-cases: {len(non_cases)}")
        print(f"Controls: {len(controls)}")
        
    except Exception as e:
        logging.error(f"Failed to process metadata: {e}")
        raise


if __name__ == "__main__":
    main()