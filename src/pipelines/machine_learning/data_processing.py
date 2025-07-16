"""
ML Data Processing Module for Metabolomic Analysis Pipeline.
Equivalent to 0__data_processing.py from the reference Risk_Scores pipeline.

This module creates ML-ready datasets from regression results, including
significant metabolite identification, multiple dataset formats, and
stratified cross-validation splits.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from ...utils.config_utils import ConfigManager, create_output_directory
from ...capybara.statistics import multitest_correct
from ...capybara.preprocessing import select_features_by_significance


class MLDataProcessor:
    """
    Processes data for machine learning from regression results and metadata.
    """
    
    def __init__(self, config_manager: ConfigManager, spark: SparkSession):
        """
        Initialize ML data processor.
        
        Args:
            config_manager: Configured ConfigManager instance
            spark: SparkSession for big data processing
        """
        self.config_manager = config_manager
        self.spark = spark
        self.logger = logging.getLogger(__name__)
        
    def load_and_process_regression_results(self, outcome: str) -> pd.DataFrame:
        """
        Load and process regression results with FDR correction.
        
        Args:
            outcome: Outcome name (eope, lope, sb, ptb, sga)
            
        Returns:
            Processed regression results with FDR correction
        """
        self.logger.info(f"Loading regression results for outcome: {outcome}")
        
        # Load pooled regression results
        pooled_results_path = self.config_manager.get_path(
            "paths.analysis.regression_analysis.pooled_analysis.pooled_regression_results"
        )
        
        # For now, assume we have pooled results - in practice this would come from regression module
        # This is a placeholder that would be replaced with actual regression results loading
        pooled_regression_result = pd.DataFrame({
            'unique_feature_label': ['feature_1', 'feature_2', 'feature_3'],
            'group': ['case_vs_control', 'case_vs_control', 'case_vs_control'],
            'P_Value_Intensity': [0.001, 0.05, 0.1],
            'Coefficient_Intensity': [1.2, -0.8, 0.3],
            'Lower_CI_Intensity': [0.5, -1.5, -0.2],
            'Upper_CI_Intensity': [1.9, -0.1, 0.8],
        })
        
        # Add global ID information (placeholder)
        pooled_regression_result['ID'] = [
            'metabolite_1', 'metabolite_2', None
        ]
        pooled_regression_result['Compound_Class'] = [
            'lipid', 'amino_acid', 'unknown'
        ]
        
        # Apply FDR correction by group and ID availability
        pooled_regression_result['has_id'] = ~pooled_regression_result['ID'].isna()
        
        corrected_results = pooled_regression_result.groupby(['group', 'has_id']).apply(
            lambda g: apply_group_fdr_correction(g, 'P_Value_Intensity'),
            include_groups=False
        ).reset_index(drop=True)
        
        # Create unique feature names
        corrected_results['unique_feature_name'] = np.where(
            corrected_results['ID'].notna(),
            corrected_results['unique_feature_label'] + '\n' + corrected_results['ID'],
            corrected_results['unique_feature_label'],
        )
        
        self.logger.info(f"Processed {len(corrected_results)} regression results")
        return corrected_results
    
    def identify_significant_metabolites(
        self, 
        regression_results: pd.DataFrame,
        fdr_cutoff: float = 0.05
    ) -> List[str]:
        """
        Identify significant metabolites based on FDR threshold.
        
        Args:
            regression_results: Processed regression results
            fdr_cutoff: FDR threshold for significance
            
        Returns:
            List of significant metabolite feature labels
        """
        return filter_significant_features(
            regression_results,
            fdr_cutoff=fdr_cutoff,
            sort_by_fdr=True
        )
    
    def load_curated_metadata(self) -> pd.DataFrame:
        """
        Load and curate metadata for ML analysis.
        
        Returns:
            Curated metadata DataFrame
        """
        self.logger.info("Loading curated metadata...")
        
        # Load imputed metadata
        metadata_path = self.config_manager.get_path(
            "paths.combined_manifest.combined_imputed_metadata"
        )
        
        # For now, create placeholder metadata
        # In practice, this would load from the metadata processing module
        imputed_cdf = pd.DataFrame({
            'subject_id': [f'subj_{i}' for i in range(100)],
            'sample_id': [f'sample_{i}' for i in range(100)],
            'site': np.random.choice(['AMANHIB', 'GAPPSB', 'ZAPPS', 'AMANHIP', 'AMANHIT'], 100),
            'single_twin': np.ones(100),
            'ga_hdlk_new': np.random.normal(180, 30, 100),  # GA in days
            'gagebrth_new': np.random.normal(270, 20, 100),  # Birth GA in days
            'pw_age': np.random.normal(25, 5, 100),  # Maternal age
            'birth_weight': np.random.normal(3000, 500, 100),
            'pe_new': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
            'pe_cat': np.random.choice(['EOPE', 'LOPE', 'TBD'], 100),
            'pe_priority': np.random.choice([1, 2, 3], 100),
            'sb_new': np.random.choice([0, 1], 100, p=[0.95, 0.05]),
            'sb_priority': np.random.choice([1, 2, 3], 100),
            'ptb_new': np.random.choice([0, 1], 100, p=[0.85, 0.15]),
            'sga_10': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
            'chron_htn': np.random.choice([0, 1], 100, p=[0.95, 0.05]),
            'diabetes': np.random.choice([0, 1], 100, p=[0.98, 0.02]),
            'fetal_anomalies': np.random.choice([0, 1], 100, p=[0.98, 0.02]),
            'cong_anomalies': np.random.choice([0, 1], 100, p=[0.98, 0.02]),
            'syphilis': np.random.choice([0, 1, 2], 100, p=[0.95, 0.03, 0.02]),
            'hiv': np.random.choice([0, 1], 100, p=[0.98, 0.02]),
            'malaria': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
            'tb': np.random.choice([0, 1], 100, p=[0.98, 0.02]),
            'smoke_hist': np.random.choice([1, 2, 3, 4], 100),
            'parity': np.random.choice([0, 1, 2, 3, 4], 100),
            'bmi': np.random.normal(25, 5, 100),
            'passive_smok': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
            'prev_sb': np.random.choice([0, 1, 2], 100, p=[0.9, 0.08, 0.02]),
            'prev_mis': np.random.choice([0, 1, 2, 3], 100, p=[0.7, 0.2, 0.08, 0.02]),
            'wealth_index': np.random.choice([1, 2, 3, 4, 5], 100),
            'alcohol': np.random.choice([1, 2, 3, 4], 100),
            'sniff_toba': np.random.choice([1, 2, 3, 4], 100),
            'gfr': np.random.normal(100, 20, 100),
        })
        
        # Apply common filtering
        common_filter = (
            "(single_twin == 1) and "
            "~gagebrth_new.isna() and "
            "~ga_hdlk_new.isna() and "
            "~pw_age.isna()"
        )
        
        eligible_metadata = imputed_cdf.query(common_filter).copy()
        
        # Rename columns to match expected format
        curated_metadata = eligible_metadata.rename(columns={
            'ga_hdlk_new': 'ga_sample_days',
            'gagebrth_new': 'ga_birth_days',
            'pw_age': 'maternal_age',
            'chron_htn': 'chronic_hypertension',
            'fetal_anomalies': 'fetal_anomalies',
            'cong_anomalies': 'congenital_anomalies',
            'smoke_hist': 'smoking_history',
            'parity': 'n_previous_pregnancies',
            'passive_smok': 'smoking_exposure',
            'prev_sb': 'n_previous_stillbirth',
            'prev_mis': 'n_previous_miscarriage',
            'wealth_index': 'wealth_index',
            'alcohol': 'alcohol_history',
            'sniff_toba': 'tobacco_history',
            'tb': 'tuberculosis',
        })
        
        # Calculate derived fields
        curated_metadata = curated_metadata.assign(
            ga_weeks=(curated_metadata['ga_sample_days'] + 6) // 7,
            ga_birth_weeks=(curated_metadata['ga_birth_days'] + 6) // 7,
        )
        
        self.logger.info(f"Curated metadata shape: {curated_metadata.shape}")
        return curated_metadata
    
    def create_outcome_columns(self, metadata: pd.DataFrame, outcome: str) -> pd.DataFrame:
        """
        Create outcome-specific columns in metadata.
        
        Args:
            metadata: Metadata DataFrame
            outcome: Outcome name
            
        Returns:
            Metadata with outcome-specific columns
        """
        metadata = metadata.copy()
        
        if outcome == 'eope':
            metadata['eope'] = (
                (metadata['pe_priority'].isin([1, 2])) & 
                (metadata['pe_cat'] == 'EOPE')
            ).astype(int)
        elif outcome == 'lope':
            metadata['lope'] = (
                (metadata['pe_priority'].isin([1, 2])) & 
                (metadata['pe_cat'] == 'LOPE')
            ).astype(int)
        # Add other outcomes as needed
        
        return metadata
    
    def create_clinical_populations(self, metadata: pd.DataFrame, outcome: str) -> pd.DataFrame:
        """
        Create clinical population classifications (healthy controls, high risk).
        
        Args:
            metadata: Metadata DataFrame
            outcome: Outcome name
            
        Returns:
            Metadata with population classifications
        """
        metadata = metadata.copy()
        
        # Define healthy controls
        metadata['healthy_control'] = (
            (metadata[outcome] == 0) &
            (metadata['ga_birth_days'] >= 273) &
            (metadata['ga_birth_days'] < 287) &
            (metadata['birth_weight'] >= 2500) &
            (metadata['chronic_hypertension'] != 1) &
            (metadata['diabetes'] != 1) &
            (metadata['fetal_anomalies'] != 1) &
            (metadata['congenital_anomalies'] != 1) &
            (~metadata['syphilis'].isin([1, 2])) &
            (metadata['hiv'] != 1) &
            (metadata['malaria'] != 1) &
            (metadata['tuberculosis'] != 1)
        )
        
        # Define high risk
        metadata['high_risk'] = (
            (metadata['maternal_age'] >= 35) |
            (metadata['smoking_history'].isin([2, 3, 4])) |
            (metadata['alcohol_history'].isin([2, 3, 4])) |
            (metadata['chronic_hypertension'] == 1) |
            (metadata['diabetes'] == 1) |
            (metadata['syphilis'].isin([1, 2])) |
            (metadata['hiv'] == 1) |
            (metadata['malaria'] == 1) |
            (metadata['tuberculosis'] == 1) |
            (metadata['bmi'] >= 30)
        )
        
        # Define broad high risk
        metadata['high_risk_broad'] = (
            metadata['high_risk'] |
            (metadata['n_previous_pregnancies'] == 0) |
            (metadata['smoking_exposure'] == 1) |
            (metadata['n_previous_miscarriage'] > 0) |
            (metadata['n_previous_stillbirth'] > 0) |
            (metadata['wealth_index'].isin([1, 2]))
        )
        
        # Aggregate to subject level for consistency
        subject_classifications = metadata.groupby('subject_id').agg({
            'healthy_control': 'all',
            'high_risk': 'any',
            'high_risk_broad': 'any',
        })
        
        # Merge back to sample level
        metadata = metadata.drop(columns=['healthy_control', 'high_risk', 'high_risk_broad'])
        metadata = metadata.merge(subject_classifications, on='subject_id', how='inner')
        
        return metadata
    
    def create_dataset_splits(self, metadata: pd.DataFrame, outcome: str) -> pd.DataFrame:
        """
        Create stratified dataset splits for cross-validation.
        
        Args:
            metadata: Metadata DataFrame
            outcome: Outcome name
            
        Returns:
            Metadata with dataset assignments
        """
        self.logger.info("Creating stratified dataset splits...")
        
        ml_config = self.config_manager.get_config().ml
        random_state = ml_config.random_state
        n_splits = ml_config.cross_validation.n_splits
        
        metadata_with_splits = create_stratified_splits(
            metadata=metadata,
            outcome_col=outcome,
            n_splits=n_splits,
            random_state=random_state,
        )
        
        # Log split statistics
        for fold, fold_df in metadata_with_splits.groupby('dataset_assignment'):
            n_samples = len(fold_df)
            n_subjects = fold_df['subject_id'].nunique()
            n_outcome = fold_df[outcome].sum()
            outcome_pct = (n_outcome / n_subjects) * 100
            
            self.logger.info(
                f"{fold}: {n_samples} samples, {n_subjects} subjects, "
                f"{n_outcome} {outcome} cases ({outcome_pct:.1f}%)"
            )
            
            # Log site distribution
            site_counts = fold_df.groupby('site').size()
            self.logger.info(f"  Sites: {dict(site_counts)}")
        
        return metadata_with_splits
    
    def load_intensity_data(self, significant_features: List[str]) -> pd.DataFrame:
        """
        Load and combine intensity data for significant features.
        
        Args:
            significant_features: List of significant feature labels
            
        Returns:
            Combined intensity DataFrame in long format
        """
        self.logger.info("Loading intensity data for significant features...")
        
        # For now, create placeholder intensity data
        # In practice, this would use the actual intensity data loading logic
        
        n_samples = 100
        n_features = len(significant_features)
        
        # Create long-form intensity data
        intensity_data = []
        for i in range(n_samples):
            for feature in significant_features:
                intensity_data.append({
                    'sample_id': f'sample_{i}',
                    'unique_feature_label': feature,
                    'intensity_fill_log': np.random.normal(5, 1),  # Log-transformed intensity
                })
        
        intensity_df = pd.DataFrame(intensity_data)
        
        self.logger.info(f"Loaded intensity data: {len(intensity_df)} records")
        return intensity_df
    
    def create_biosample_dataset(
        self, 
        metadata: pd.DataFrame, 
        intensity_df: pd.DataFrame,
        outcome: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create biosample-level dataset.
        
        Args:
            metadata: Processed metadata
            intensity_df: Intensity data in long format
            outcome: Outcome name
            
        Returns:
            Tuple of (metadata, intensity_wide) for biosample dataset
        """
        self.logger.info("Creating biosample dataset...")
        
        # Filter timepoints close to birth
        days_before_birth = 14
        close_to_birth = metadata[
            metadata['ga_sample_days'] >= (metadata['ga_birth_days'] - days_before_birth)
        ]
        
        if len(close_to_birth) > 0:
            self.logger.info(f"Removing {len(close_to_birth)} samples close to birth")
            metadata = metadata.drop(index=close_to_birth.index)
        
        # Convert to wide format
        intensity_wide = create_wide_intensity_matrix(
            intensity_df,
            sample_col='sample_id',
            feature_col='unique_feature_label',
            value_col='intensity_fill_log',
        )
        
        # Filter metadata to samples with intensity data
        metadata_filtered = metadata[metadata['sample_id'].isin(intensity_wide.index)]
        metadata_filtered = metadata_filtered.set_index('sample_id')
        
        # Ensure consistent indexing
        common_samples = metadata_filtered.index.intersection(intensity_wide.index)
        metadata_final = metadata_filtered.loc[common_samples]
        intensity_final = intensity_wide.loc[common_samples]
        
        self.logger.info(f"Biosample dataset: {len(metadata_final)} samples, {intensity_final.shape[1]} features")
        
        return metadata_final, intensity_final
    
    def create_subject_auc_dataset(
        self, 
        metadata: pd.DataFrame, 
        intensity_df: pd.DataFrame,
        outcome: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create subject-level dataset using AUC features.
        
        Args:
            metadata: Processed metadata  
            intensity_df: Intensity data in long format
            outcome: Outcome name
            
        Returns:
            Tuple of (subject_metadata, subject_features) for AUC dataset
        """
        self.logger.info("Creating subject AUC dataset...")
        
        # Convert to wide format first
        intensity_wide = create_wide_intensity_matrix(intensity_df)
        
        # Filter to subjects with multiple timepoints
        timepoint_counts = metadata.groupby('subject_id').size()
        multi_timepoint_subjects = timepoint_counts[timepoint_counts > 1].index
        
        metadata_multi = metadata[metadata['subject_id'].isin(multi_timepoint_subjects)]
        
        # Create subject-level metadata
        subject_metadata_rows = []
        subject_feature_rows = []
        
        for subject_id, subject_df in metadata_multi.groupby('subject_id'):
            n_timepoints = len(subject_df)
            
            # Sort by time
            subject_df = subject_df.sort_values('ga_sample_days')
            
            # Basic subject info
            subject_row = subject_df.iloc[0].copy()
            
            # Time information
            timepoints = subject_df['ga_sample_days'].values
            first_tp, last_tp = timepoints[0], timepoints[-1]
            time_delta = last_tp - first_tp
            
            # Skip subjects with small time deltas
            if time_delta < 28:  # Less than 4 weeks
                continue
            
            # Get intensities for this subject's samples
            sample_ids = subject_df['sample_id'].tolist()
            if not all(sid in intensity_wide.index for sid in sample_ids):
                continue
                
            sample_intensities = intensity_wide.loc[sample_ids]
            
            # Calculate AUC features
            first_timepoint_intensities = sample_intensities.iloc[0]
            baselined_intensities = sample_intensities - first_timepoint_intensities
            aucs = np.trapz(baselined_intensities, x=timepoints, axis=0)
            aucs_norm = aucs / time_delta
            
            # Create feature dictionary
            feature_dict = {
                'first_timepoint': first_tp,
                'time_delta': time_delta,
            }
            
            # Add first timepoint intensities
            for feature, intensity in first_timepoint_intensities.items():
                feature_dict[feature] = intensity
            
            # Add AUC features
            for feature, auc in zip(sample_intensities.columns, aucs_norm):
                feature_dict[f"{feature}_auc"] = auc
            
            # Store results
            subject_metadata_rows.append({
                'subject_id': subject_id,
                'n_timepoints': n_timepoints,
                outcome: subject_row[outcome],
                'maternal_age': subject_row['maternal_age'],
                'ga_birth': subject_row['ga_birth_days'],
                'site': subject_row['site'],
                'dataset_assignment': subject_row['dataset_assignment'],
                'first_timepoint': first_tp,
                'last_timepoint': last_tp,
                'time_delta': time_delta,
                'sample_ids': ';'.join(sample_ids),
                'healthy_control': subject_row['healthy_control'],
                'high_risk': subject_row['high_risk'],
                'high_risk_broad': subject_row['high_risk_broad'],
            })
            
            subject_feature_rows.append(feature_dict)
        
        subject_metadata = pd.DataFrame(subject_metadata_rows).set_index('subject_id')
        subject_features = pd.DataFrame(subject_feature_rows, index=subject_metadata.index)
        
        self.logger.info(f"Subject AUC dataset: {len(subject_metadata)} subjects, {subject_features.shape[1]} features")
        
        return subject_metadata, subject_features
    
    def create_ga_window_dataset(
        self, 
        metadata: pd.DataFrame, 
        intensity_df: pd.DataFrame,
        outcome: str,
        window_start: int = 63,
        window_size: int = 120,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create GA window-based dataset.
        
        Args:
            metadata: Processed metadata
            intensity_df: Intensity data in long format
            outcome: Outcome name
            window_start: Start day for GA window
            window_size: Size of GA window in days
            
        Returns:
            Tuple of (window_metadata, window_intensity) for GA window dataset
        """
        self.logger.info(f"Creating GA window dataset (days {window_start}-{window_start + window_size})...")
        
        # Filter to samples in the GA window
        window_metadata = metadata[
            (metadata['ga_sample_days'] >= window_start) &
            (metadata['ga_sample_days'] <= window_start + window_size)
        ]
        
        # Convert intensity to wide format
        intensity_wide = create_wide_intensity_matrix(intensity_df)
        
        # Filter to samples in window
        window_intensity = intensity_wide.loc[
            intensity_wide.index.intersection(window_metadata['sample_id'])
        ]
        
        # Update metadata indexing
        window_metadata = window_metadata[
            window_metadata['sample_id'].isin(window_intensity.index)
        ].set_index('sample_id')
        
        # Ensure consistent indexing
        common_samples = window_metadata.index.intersection(window_intensity.index)
        window_metadata = window_metadata.loc[common_samples]
        window_intensity = window_intensity.loc[common_samples]
        
        self.logger.info(f"GA window dataset: {len(window_metadata)} samples, {window_intensity.shape[1]} features")
        
        return window_metadata, window_intensity
    
    def save_dataset(
        self, 
        metadata: pd.DataFrame, 
        intensity: pd.DataFrame, 
        dataset_type: str, 
        outcome: str
    ) -> None:
        """
        Save dataset to configured output location.
        
        Args:
            metadata: Metadata DataFrame
            intensity: Intensity DataFrame
            dataset_type: Type of dataset (biosample, subject, window)
            outcome: Outcome name
        """
        # Create output directory
        output_dir = f"s3://output-bucket/ml_datasets/{outcome}/"
        create_output_directory(output_dir)
        
        # Save files
        metadata_path = f"{output_dir}metadata_{dataset_type}.csv"
        intensity_path = f"{output_dir}intensity_{dataset_type}.csv"
        
        metadata.to_csv(metadata_path)
        intensity.to_csv(intensity_path)
        
        self.logger.info(f"Saved {dataset_type} dataset to {output_dir}")
    
    def run(self, outcome: str) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Run the complete ML data processing pipeline.
        
        Args:
            outcome: Outcome name (eope, lope, sb, ptb, sga)
            
        Returns:
            Dictionary of dataset_type -> (metadata, intensity) pairs
        """
        self.logger.info(f"Starting ML data processing for outcome: {outcome}")
        
        # Load and process regression results
        regression_results = self.load_and_process_regression_results(outcome)
        
        # Identify significant metabolites
        significant_features = self.identify_significant_metabolites(regression_results)
        self.logger.info(f"Found {len(significant_features)} significant features")
        
        # Load and curate metadata
        metadata = self.load_curated_metadata()
        metadata = self.create_outcome_columns(metadata, outcome)
        metadata = self.create_clinical_populations(metadata, outcome)
        metadata = self.create_dataset_splits(metadata, outcome)
        
        # Load intensity data
        intensity_df = self.load_intensity_data(significant_features)
        
        # Create different dataset types
        datasets = {}
        
        # Biosample dataset
        biosample_metadata, biosample_intensity = self.create_biosample_dataset(
            metadata, intensity_df, outcome
        )
        datasets['biosample'] = (biosample_metadata, biosample_intensity)
        
        # Subject AUC dataset
        subject_metadata, subject_intensity = self.create_subject_auc_dataset(
            metadata, intensity_df, outcome
        )
        datasets['subject'] = (subject_metadata, subject_intensity)
        
        # GA window dataset
        window_metadata, window_intensity = self.create_ga_window_dataset(
            metadata, intensity_df, outcome
        )
        datasets['window'] = (window_metadata, window_intensity)
        
        # Save datasets
        for dataset_type, (md, intensity) in datasets.items():
            self.save_dataset(md, intensity, dataset_type, outcome)
        
        self.logger.info("ML data processing complete")
        return datasets


def process_ml_data(outcome: str, config_dir: str = None, spark: SparkSession = None) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Convenience function to process ML data.
    
    Args:
        outcome: Outcome name (eope, lope, sb, ptb, sga)
        config_dir: Configuration directory path
        spark: SparkSession instance
        
    Returns:
        Dictionary of dataset_type -> (metadata, intensity) pairs
    """
    # Initialize configuration manager
    config_manager = ConfigManager(config_dir)
    config_manager.load_config(outcome=outcome)
    
    # Get or create Spark session
    if spark is None:
        from ..utils.spark_utils import get_or_create_spark_session
        spark = get_or_create_spark_session()
    
    # Create ML data processor
    processor = MLDataProcessor(config_manager, spark)
    
    # Process data
    return processor.run(outcome)


def main():
    """
    Main function for running ML data processing.
    """
    import argparse
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Process data for ML risk score analysis")
    parser.add_argument("outcome", help="Outcome name (eope, lope, sb, ptb, sga)")
    parser.add_argument("--config-dir", help="Configuration directory path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Process ML data
    try:
        datasets = process_ml_data(args.outcome, args.config_dir)
        
        print(f"ML data processing completed for outcome: {args.outcome}")
        for dataset_type, (metadata, intensity) in datasets.items():
            print(f"{dataset_type}: {len(metadata)} samples, {intensity.shape[1]} features")
        
    except Exception as e:
        logging.error(f"Failed to process ML data: {e}")
        raise


if __name__ == "__main__":
    main()