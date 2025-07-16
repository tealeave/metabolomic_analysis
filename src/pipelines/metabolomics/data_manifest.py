"""
Data Manifest Module for Metabolomic Analysis Pipeline.
Equivalent to {outcome}-data-manifest.py from the reference pipeline.

This module creates outcome-specific data manifests by extracting and categorizing
all relevant file paths from the configuration system.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
from omegaconf import OmegaConf

from ..utils.config_utils import ConfigManager, create_output_directory


class DataManifestGenerator:
    """
    Generates data manifests for metabolomic analysis pipeline.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize data manifest generator.
        
        Args:
            config_manager: Configured ConfigManager instance
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
    def extract_data_paths(self, config_dict: Dict[str, Any], prefix_keys: List[str] = None) -> List[Tuple[str, str]]:
        """
        Recursively traverse configuration dict and extract data paths (S3 or local).
        
        Args:
            config_dict: Configuration dictionary to traverse
            prefix_keys: Current hierarchy prefix keys
            
        Returns:
            List of tuples: (description_hierarchy, file_path)
        """
        if prefix_keys is None:
            prefix_keys = []
        
        results = []
        
        if isinstance(config_dict, dict):
            for key, value in config_dict.items():
                new_prefix = prefix_keys + [key]
                
                if isinstance(value, dict):
                    results.extend(self.extract_data_paths(value, new_prefix))
                elif isinstance(value, str) and (value.startswith("s3://") or self._is_local_data_path(value)):
                    description = " > ".join(new_prefix)
                    results.append((description, value))
                    
        return results
    
    def _is_local_data_path(self, path: str) -> bool:
        """
        Check if a path is a local data path.
        
        Args:
            path: File path to check
            
        Returns:
            True if it's a local data path, False otherwise
        """
        # Check if it's a relative path starting with 'data/' or contains common data extensions
        if path.startswith("data/"):
            return True
        
        # Check for common data file extensions
        data_extensions = ['.csv', '.xlsx', '.delta', '.parquet', '.json']
        if any(path.endswith(ext) for ext in data_extensions):
            # Check if it's not an absolute system path
            if not path.startswith('/') and not path.startswith('C:'):
                return True
        
        return False
    
    def infer_data_type(self, description: str) -> str:
        """
        Infer data type from description hierarchy.
        
        Args:
            description: Description hierarchy string
            
        Returns:
            Inferred data type category
        """
        desc_lower = description.lower()
        
        if "meta_data" in desc_lower or "metadata" in desc_lower or "manifest" in desc_lower:
            return "metadata"
        elif "preprocessed_maternal_data" in desc_lower or "intensities" in desc_lower or "intensity_data" in desc_lower:
            return "intensity_data"
        elif "regression_analysis" in desc_lower or "analysis_results" in desc_lower:
            return "analysis_result"
        elif "deliverables" in desc_lower:
            return "deliverable"
        elif "alignment" in desc_lower:
            return "alignment_data"
        else:
            return "other"
    
    def infer_sub_data_type(self, description: str) -> str:
        """
        Infer sub-data type from description hierarchy.
        
        Args:
            description: Description hierarchy string
            
        Returns:
            Inferred sub-data type
        """
        desc_lower = description.lower()
        
        if "manifest_kim" in desc_lower:
            return "manifest_kim (original manifests)"
        elif "manifest_cleaned" in desc_lower:
            return "manifest_cleaned (cleaned manifests)"
        elif "combined_imputed_metadata" in desc_lower:
            return "combined_imputed_metadata"
        elif "stratified_metadata" in desc_lower:
            return "stratified_metadata"
        elif "feature_metadata" in desc_lower:
            return "feature_metadata"
        elif "sample_metadata" in desc_lower:
            return "sample_metadata"
        elif "maternal_intensities_preprocessed_long" in desc_lower or "intensity_data" in desc_lower:
            return "maternal_preprocessed_intensities"
        elif "project_deliverables" in desc_lower:
            return "project_deliverables"
        elif "data_analysis" in desc_lower:
            return "data_analysis_intermediate"
        elif "alignment" in desc_lower:
            return "alignment_data"
        else:
            return "general"
    
    def infer_purpose(self, description: str) -> str:
        """
        Infer purpose from description hierarchy.
        
        Args:
            description: Description hierarchy string
            
        Returns:
            Inferred purpose description
        """
        desc_lower = description.lower()
        
        # Master alignment file
        if "momi_melted_paired_alignment_uniq_to_site_long_format" in desc_lower:
            return (
                "Master alignment file that aligns metabolites across sites/cohorts, "
                "facilitating integrative analysis across multiple datasets. "
                "Essential for cross-site metabolomics analysis."
            )
        
        # Metadata purposes
        if "meta_data" in desc_lower or "metadata" in desc_lower:
            return "Provides metadata for subjects, samples, or derived variables."
        elif "manifest_kim" in desc_lower:
            return "Original sample manifest before cleaning, used for initial QC."
        elif "manifest_cleaned" in desc_lower:
            return "Cleaned manifest after QC for accurate sample-level metadata."
        elif "combined_imputed_metadata" in desc_lower:
            return "Aggregate metadata used for downstream analyses with imputed covariates."
        elif "stratified_metadata" in desc_lower:
            return "Stratified metadata for grouping samples by outcome and gestational age."
        
        # Intensity data purposes
        elif "preprocessed_maternal_data" in desc_lower or "intensity_data" in desc_lower:
            return "Preprocessed maternal metabolite intensity data for downstream analysis."
        elif "feature_metadata" in desc_lower:
            return "Metadata describing metabolite features and their properties."
        elif "sample_metadata" in desc_lower:
            return "Sample-level metadata for intensity data integration."
        
        # Analysis purposes
        elif "regression_analysis" in desc_lower:
            return "Input or results for regression comparing outcomes and controls."
        elif "pooled_regression_results" in desc_lower:
            return "Final pooled regression results for outcome comparison."
        elif "comparing_to_old_results" in desc_lower:
            return "Comparison results between new and previous analysis."
        
        # Other purposes
        elif "deliverables" in desc_lower:
            return "Final deliverables for the project."
        elif "alignment" in desc_lower:
            return "Data aligning metabolites across sites/cohorts for meta-analysis."
        else:
            return "General input or result file used in the MOMI pipeline."
    
    def generate_manifest(self, outcome: str, sample_type: str = "maternal") -> pd.DataFrame:
        """
        Generate data manifest for specified outcome.
        
        Args:
            outcome: Outcome name (sb, ptb, sga, etc.)
            sample_type: Sample type (default: "maternal")
            
        Returns:
            DataFrame containing the data manifest
        """
        self.logger.info(f"Creating data manifest for outcome: {outcome}")
        
        # Get configuration
        config = self.config_manager.get_config()
        
        # Convert to container for path extraction
        config_dict = OmegaConf.to_container(config, resolve=True)
        
        # Extract all data paths (S3 or local)
        all_paths = self.extract_data_paths(config_dict)
        
        # Create DataFrame
        df = pd.DataFrame(all_paths, columns=["description_hierarchy", "file_path"])
        
        # Add metadata columns
        df = df.assign(
            outcome=outcome,
            sample_type=sample_type,
            data_type=df["description_hierarchy"].apply(self.infer_data_type),
            sub_data_type=df["description_hierarchy"].apply(self.infer_sub_data_type),
            purpose=df["description_hierarchy"].apply(self.infer_purpose),
            data_source=df["file_path"].apply(lambda x: "s3" if x.startswith("s3://") else "local"),
        )
        
        # Reorder columns
        df = df[["outcome", "sample_type", "data_type", "sub_data_type", "purpose", "data_source", "file_path", "description_hierarchy"]]
        
        self.logger.info(f"Generated manifest with {len(df)} entries")
        return df
    
    def save_manifest(self, manifest_df: pd.DataFrame, output_path: str) -> None:
        """
        Save data manifest to CSV file.
        
        Args:
            manifest_df: DataFrame containing the manifest
            output_path: Path to save the manifest
        """
        # Create output directory if needed
        create_output_directory(output_path)
        
        # Save to CSV
        manifest_df.to_csv(output_path, index=False)
        
        self.logger.info(f"Data manifest saved to: {output_path}")
    
    def run(self, outcome: str, sample_type: str = "maternal") -> pd.DataFrame:
        """
        Run the complete data manifest generation process.
        
        Args:
            outcome: Outcome name (sb, ptb, sga, etc.)
            sample_type: Sample type (default: "maternal")
            
        Returns:
            Generated manifest DataFrame
        """
        # Generate manifest
        manifest_df = self.generate_manifest(outcome, sample_type)
        
        # Get output path from configuration
        output_path = self.config_manager.get_path("paths.file_manifest.outcome_manifest")
        
        # Save manifest
        self.save_manifest(manifest_df, output_path)
        
        return manifest_df


def create_data_manifest(outcome: str, config_dir: str = None) -> pd.DataFrame:
    """
    Convenience function to create data manifest.
    
    Args:
        outcome: Outcome name (sb, ptb, sga, etc.)
        config_dir: Configuration directory path
        
    Returns:
        Generated manifest DataFrame
    """
    # Initialize configuration manager
    config_manager = ConfigManager(config_dir)
    config_manager.load_config(outcome=outcome)
    
    # Create manifest generator
    generator = DataManifestGenerator(config_manager)
    
    # Generate and return manifest
    return generator.run(outcome)


def main():
    """
    Main function for running data manifest generation.
    """
    import argparse
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Generate data manifest for metabolomic analysis")
    parser.add_argument("outcome", help="Outcome name (sb, ptb, sga, etc.)")
    parser.add_argument("--config-dir", help="Configuration directory path")
    parser.add_argument("--sample-type", default="maternal", help="Sample type (default: maternal)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create data manifest
    try:
        manifest_df = create_data_manifest(args.outcome, args.config_dir)
        
        print(f"Data manifest created successfully for outcome: {args.outcome}")
        print(f"Total entries: {len(manifest_df)}")
        print("\nFirst 10 entries:")
        print(manifest_df.head(10))
        
    except Exception as e:
        logging.error(f"Failed to create data manifest: {e}")
        raise


if __name__ == "__main__":
    main()