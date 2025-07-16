"""
ML Exploratory Data Analysis Module for Metabolomic Analysis Pipeline.
Equivalent to 1__eda.py from the reference Risk_Scores pipeline.

This module provides comprehensive EDA capabilities for ML datasets including
PCA analysis, feature correlation assessment, batch correction validation,
and population stratification analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ...utils.config_utils import ConfigManager, create_output_directory
from ...capybara.preprocessing import log_median_centering
from ...capybara.visualization import create_pca_plot, set_plot_style


class MLExploratoryAnalysis:
    """
    Performs exploratory data analysis for ML datasets.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize ML EDA analyzer.
        
        Args:
            config_manager: Configured ConfigManager instance
        """
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Set up plotting
        set_plot_font()
        self.custom_cmap = make_brand_bwr_colormap()
        
    def load_datasets(self, outcome: str) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Load ML datasets created by MLDataProcessor.
        
        Args:
            outcome: Outcome name (eope, lope, sb, ptb, sga)
            
        Returns:
            Dictionary of dataset_type -> (metadata, intensity) pairs
        """
        self.logger.info(f"Loading ML datasets for outcome: {outcome}")
        
        datasets = {}
        output_dir = f"s3://output-bucket/ml_datasets/{outcome}/"
        
        for dataset_type in ['biosample', 'subject', 'window']:
            try:
                metadata_path = f"{output_dir}metadata_{dataset_type}.csv"
                intensity_path = f"{output_dir}intensity_{dataset_type}.csv"
                
                metadata = pd.read_csv(metadata_path, index_col=0)
                intensity = pd.read_csv(intensity_path, index_col=0)
                
                datasets[dataset_type] = (metadata, intensity)
                self.logger.info(f"Loaded {dataset_type} dataset: {len(metadata)} samples, {intensity.shape[1]} features")
                
            except Exception as e:
                self.logger.warning(f"Could not load {dataset_type} dataset: {e}")
                
        return datasets
    
    def analyze_dataset_characteristics(
        self, 
        metadata: pd.DataFrame, 
        intensity: pd.DataFrame,
        dataset_type: str,
        outcome: str
    ) -> Dict[str, any]:
        """
        Analyze basic characteristics of a dataset.
        
        Args:
            metadata: Metadata DataFrame
            intensity: Intensity DataFrame
            dataset_type: Type of dataset
            outcome: Outcome name
            
        Returns:
            Dictionary of dataset characteristics
        """
        self.logger.info(f"Analyzing {dataset_type} dataset characteristics...")
        
        characteristics = {
            'dataset_type': dataset_type,
            'n_samples': len(metadata),
            'n_features': intensity.shape[1],
            'n_subjects': metadata['subject_id'].nunique() if 'subject_id' in metadata.columns else len(metadata),
            'outcome_prevalence': metadata[outcome].mean() if outcome in metadata.columns else None,
        }
        
        # Site distribution
        if 'site' in metadata.columns:
            site_counts = metadata['site'].value_counts()
            characteristics['sites'] = dict(site_counts)
            characteristics['n_sites'] = len(site_counts)
        
        # Dataset assignment distribution
        if 'dataset_assignment' in metadata.columns:
            assignment_counts = metadata['dataset_assignment'].value_counts()
            characteristics['dataset_assignments'] = dict(assignment_counts)
        
        # Population stratification
        for pop_type in ['healthy_control', 'high_risk', 'high_risk_broad']:
            if pop_type in metadata.columns:
                characteristics[f'{pop_type}_count'] = metadata[pop_type].sum()
                characteristics[f'{pop_type}_proportion'] = metadata[pop_type].mean()
        
        # Feature intensity statistics
        characteristics['feature_stats'] = {
            'mean_intensity': intensity.mean().mean(),
            'std_intensity': intensity.std().mean(),
            'min_intensity': intensity.min().min(),
            'max_intensity': intensity.max().max(),
            'missing_values': intensity.isna().sum().sum(),
        }
        
        self.logger.info(f"Dataset characteristics computed for {dataset_type}")
        
        return characteristics
    
    def perform_pca_analysis(
        self,
        intensity: pd.DataFrame,
        metadata: pd.DataFrame,
        dataset_type: str,
        outcome: str,
        apply_batch_correction: bool = True,
        n_components: int = 10
    ) -> Dict[str, any]:
        """
        Perform Principal Component Analysis on intensity data.
        
        Args:
            intensity: Intensity DataFrame
            metadata: Metadata DataFrame
            dataset_type: Type of dataset
            outcome: Outcome name
            apply_batch_correction: Whether to apply batch correction
            n_components: Number of PCA components to compute
            
        Returns:
            Dictionary containing PCA results
        """
        self.logger.info(f"Performing PCA analysis for {dataset_type} dataset...")
        
        # Prepare data
        X = intensity.copy()
        
        # Apply batch correction if requested
        if apply_batch_correction and 'site' in metadata.columns:
            X = batch_correction(X, metadata)
            self.logger.info("Applied batch correction for PCA")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            index=X.index,
            columns=X.columns
        )
        
        # Perform PCA
        pca = PCA(n_components=min(n_components, min(X_scaled.shape)))
        pca_result = pca.fit_transform(X_scaled)
        
        # Create PCA DataFrame
        pca_columns = [f'PC{i+1}' for i in range(pca_result.shape[1])]
        pca_df = pd.DataFrame(
            pca_result,
            index=X.index,
            columns=pca_columns
        )
        
        # Merge with metadata
        pca_with_metadata = pca_df.merge(metadata, left_index=True, right_index=True, how='inner')
        
        pca_analysis = {
            'pca_data': pca_with_metadata,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'n_components': pca.n_components_,
            'feature_loadings': pd.DataFrame(
                pca.components_.T,
                index=X.columns,
                columns=pca_columns
            )
        }
        
        self.logger.info(f"PCA completed. Explained variance by first 3 PCs: {pca.explained_variance_ratio_[:3]}")
        
        return pca_analysis
    
    def analyze_batch_effects(
        self,
        intensity: pd.DataFrame,
        metadata: pd.DataFrame,
        dataset_type: str
    ) -> Dict[str, any]:
        """
        Analyze batch effects across sites before and after correction.
        
        Args:
            intensity: Intensity DataFrame
            metadata: Metadata DataFrame
            dataset_type: Type of dataset
            
        Returns:
            Dictionary containing batch effect analysis
        """
        if 'site' not in metadata.columns:
            self.logger.warning("No site information available for batch effect analysis")
            return {}
        
        self.logger.info(f"Analyzing batch effects for {dataset_type} dataset...")
        
        # Original data PCA
        pca_original = self.perform_pca_analysis(
            intensity, metadata, dataset_type, None, 
            apply_batch_correction=False, n_components=5
        )
        
        # Batch-corrected data PCA
        pca_corrected = self.perform_pca_analysis(
            intensity, metadata, dataset_type, None,
            apply_batch_correction=True, n_components=5
        )
        
        # Calculate site-specific statistics
        site_stats = {}
        for site, site_data in metadata.groupby('site'):
            site_intensity = intensity.loc[site_data.index]
            site_stats[site] = {
                'n_samples': len(site_data),
                'mean_intensity': site_intensity.mean().mean(),
                'std_intensity': site_intensity.std().mean(),
            }
        
        batch_analysis = {
            'site_statistics': site_stats,
            'pca_original': pca_original,
            'pca_corrected': pca_corrected,
            'n_sites': metadata['site'].nunique(),
        }
        
        self.logger.info("Batch effect analysis completed")
        
        return batch_analysis
    
    def analyze_feature_correlations(
        self,
        intensity: pd.DataFrame,
        metadata: pd.DataFrame,
        dataset_type: str,
        max_features: int = 50
    ) -> Dict[str, any]:
        """
        Analyze correlations between features and with covariates.
        
        Args:
            intensity: Intensity DataFrame
            metadata: Metadata DataFrame
            dataset_type: Type of dataset
            max_features: Maximum number of features to include in correlation analysis
            
        Returns:
            Dictionary containing correlation analysis
        """
        self.logger.info(f"Analyzing feature correlations for {dataset_type} dataset...")
        
        # Select subset of features if too many
        if intensity.shape[1] > max_features:
            # Select features with highest variance
            feature_vars = intensity.var()
            top_features = feature_vars.nlargest(max_features).index
            intensity_subset = intensity[top_features]
            self.logger.info(f"Selected top {max_features} features by variance for correlation analysis")
        else:
            intensity_subset = intensity
        
        # Feature-feature correlations
        feature_corr = intensity_subset.corr()
        
        # Feature-covariate correlations
        ml_config = self.config_manager.get_config().ml
        if dataset_type in ml_config.dataset_types:
            base_covariates = ml_config.dataset_types[dataset_type].covariates.base
            available_covariates = [cov for cov in base_covariates if cov in metadata.columns]
            
            if available_covariates:
                covariate_data = metadata[available_covariates]
                
                # Calculate correlations between features and covariates
                feature_covariate_corr = pd.DataFrame(
                    index=intensity_subset.columns,
                    columns=available_covariates
                )
                
                for feature in intensity_subset.columns:
                    for covariate in available_covariates:
                        corr = intensity_subset[feature].corr(covariate_data[covariate])
                        feature_covariate_corr.loc[feature, covariate] = corr
                
                feature_covariate_corr = feature_covariate_corr.astype(float)
            else:
                feature_covariate_corr = pd.DataFrame()
        else:
            feature_covariate_corr = pd.DataFrame()
        
        correlation_analysis = {
            'feature_correlations': feature_corr,
            'feature_covariate_correlations': feature_covariate_corr,
            'high_corr_pairs': self._find_high_correlation_pairs(feature_corr),
            'correlation_summary': {
                'mean_abs_correlation': feature_corr.abs().mean().mean(),
                'max_correlation': feature_corr.abs().max().max(),
                'n_high_corr_pairs': (feature_corr.abs() > 0.8).sum().sum() // 2,  # Divide by 2 for symmetry
            }
        }
        
        self.logger.info("Feature correlation analysis completed")
        
        return correlation_analysis
    
    def _find_high_correlation_pairs(
        self, 
        corr_matrix: pd.DataFrame, 
        threshold: float = 0.8
    ) -> List[Tuple[str, str, float]]:
        """
        Find pairs of features with high correlation.
        
        Args:
            corr_matrix: Correlation matrix
            threshold: Correlation threshold
            
        Returns:
            List of (feature1, feature2, correlation) tuples
        """
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    feature1 = corr_matrix.columns[i]
                    feature2 = corr_matrix.columns[j]
                    high_corr_pairs.append((feature1, feature2, corr_val))
        
        # Sort by absolute correlation value
        high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return high_corr_pairs
    
    def create_eda_visualizations(
        self,
        datasets: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]],
        analyses: Dict[str, Dict],
        outcome: str,
        output_dir: str
    ) -> None:
        """
        Create comprehensive EDA visualizations.
        
        Args:
            datasets: Dictionary of datasets
            analyses: Dictionary of analysis results
            outcome: Outcome name
            output_dir: Output directory for plots
        """
        self.logger.info("Creating EDA visualizations...")
        
        create_output_directory(output_dir)
        
        for dataset_type, (metadata, intensity) in datasets.items():
            if dataset_type not in analyses:
                continue
                
            analysis = analyses[dataset_type]
            
            # Create dataset-specific subdirectory
            dataset_output_dir = Path(output_dir) / dataset_type
            create_output_directory(str(dataset_output_dir))
            
            # 1. Dataset overview plot
            self._plot_dataset_overview(
                metadata, intensity, dataset_type, outcome,
                str(dataset_output_dir / "dataset_overview.png")
            )
            
            # 2. PCA plots
            if 'pca_analysis' in analysis:
                self._plot_pca_analysis(
                    analysis['pca_analysis'], outcome,
                    str(dataset_output_dir / "pca_analysis.png")
                )
            
            # 3. Batch effect plots
            if 'batch_analysis' in analysis:
                self._plot_batch_effects(
                    analysis['batch_analysis'],
                    str(dataset_output_dir / "batch_effects.png")
                )
            
            # 4. Correlation heatmaps
            if 'correlation_analysis' in analysis:
                self._plot_correlation_analysis(
                    analysis['correlation_analysis'],
                    str(dataset_output_dir / "correlations.png")
                )
        
        self.logger.info(f"EDA visualizations saved to {output_dir}")
    
    def _plot_dataset_overview(
        self,
        metadata: pd.DataFrame,
        intensity: pd.DataFrame,
        dataset_type: str,
        outcome: str,
        output_path: str
    ) -> None:
        """Create dataset overview plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Dataset Overview: {dataset_type.title()}', fontsize=16)
        
        # Sample distribution by site
        if 'site' in metadata.columns:
            metadata['site'].value_counts().plot(kind='bar', ax=axes[0, 0])
            axes[0, 0].set_title('Samples by Site')
            axes[0, 0].set_xlabel('Site')
            axes[0, 0].set_ylabel('Number of Samples')
        
        # Outcome distribution
        if outcome in metadata.columns:
            metadata[outcome].value_counts().plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title(f'{outcome.upper()} Distribution')
            axes[0, 1].set_xlabel(f'{outcome.upper()}')
            axes[0, 1].set_ylabel('Number of Samples')
        
        # Feature intensity distribution
        intensity_means = intensity.mean(axis=1)
        axes[1, 0].hist(intensity_means, bins=30, alpha=0.7)
        axes[1, 0].set_title('Mean Feature Intensity Distribution')
        axes[1, 0].set_xlabel('Mean Intensity')
        axes[1, 0].set_ylabel('Frequency')
        
        # Missing values heatmap
        missing_data = intensity.isna().sum()
        if missing_data.sum() > 0:
            missing_data.plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Missing Values by Feature')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center')
            axes[1, 1].set_title('Missing Values by Feature')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_pca_analysis(
        self,
        pca_analysis: Dict,
        outcome: str,
        output_path: str
    ) -> None:
        """Create PCA analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Principal Component Analysis', fontsize=16)
        
        pca_data = pca_analysis['pca_data']
        
        # Scree plot
        explained_var = pca_analysis['explained_variance_ratio']
        cumulative_var = pca_analysis['cumulative_variance']
        
        x_range = range(1, len(explained_var) + 1)
        axes[0, 0].bar(x_range, explained_var, alpha=0.7, label='Individual')
        axes[0, 0].plot(x_range, cumulative_var, 'ro-', label='Cumulative')
        axes[0, 0].set_title('PCA Scree Plot')
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Explained Variance Ratio')
        axes[0, 0].legend()
        
        # PC1 vs PC2 colored by outcome
        if outcome in pca_data.columns:
            for outcome_val in pca_data[outcome].unique():
                subset = pca_data[pca_data[outcome] == outcome_val]
                axes[0, 1].scatter(subset['PC1'], subset['PC2'], 
                                 label=f'{outcome}={outcome_val}', alpha=0.6)
            axes[0, 1].set_title(f'PC1 vs PC2 (colored by {outcome})')
            axes[0, 1].set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
            axes[0, 1].set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
            axes[0, 1].legend()
        
        # PC1 vs PC2 colored by site
        if 'site' in pca_data.columns:
            sites = pca_data['site'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(sites)))
            for site, color in zip(sites, colors):
                subset = pca_data[pca_data['site'] == site]
                axes[1, 0].scatter(subset['PC1'], subset['PC2'], 
                                 c=[color], label=site, alpha=0.6)
            axes[1, 0].set_title('PC1 vs PC2 (colored by site)')
            axes[1, 0].set_xlabel(f'PC1 ({explained_var[0]:.1%} variance)')
            axes[1, 0].set_ylabel(f'PC2 ({explained_var[1]:.1%} variance)')
            axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # PC3 vs PC4
        if pca_analysis['n_components'] >= 4:
            axes[1, 1].scatter(pca_data['PC3'], pca_data['PC4'], alpha=0.6)
            axes[1, 1].set_title('PC3 vs PC4')
            axes[1, 1].set_xlabel(f'PC3 ({explained_var[2]:.1%} variance)')
            axes[1, 1].set_ylabel(f'PC4 ({explained_var[3]:.1%} variance)')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_batch_effects(
        self,
        batch_analysis: Dict,
        output_path: str
    ) -> None:
        """Create batch effect visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Batch Effect Analysis', fontsize=16)
        
        # Site statistics
        site_stats = batch_analysis['site_statistics']
        sites = list(site_stats.keys())
        n_samples = [site_stats[site]['n_samples'] for site in sites]
        mean_intensities = [site_stats[site]['mean_intensity'] for site in sites]
        
        axes[0, 0].bar(sites, n_samples)
        axes[0, 0].set_title('Sample Count by Site')
        axes[0, 0].set_ylabel('Number of Samples')
        
        axes[0, 1].bar(sites, mean_intensities)
        axes[0, 1].set_title('Mean Intensity by Site')
        axes[0, 1].set_ylabel('Mean Intensity')
        
        # PCA before and after batch correction
        if 'pca_original' in batch_analysis and 'pca_corrected' in batch_analysis:
            pca_orig = batch_analysis['pca_original']['pca_data']
            pca_corr = batch_analysis['pca_corrected']['pca_data']
            
            # Before correction
            if 'site' in pca_orig.columns:
                for site in pca_orig['site'].unique():
                    subset = pca_orig[pca_orig['site'] == site]
                    axes[1, 0].scatter(subset['PC1'], subset['PC2'], 
                                     label=site, alpha=0.6)
                axes[1, 0].set_title('Before Batch Correction')
                axes[1, 0].set_xlabel('PC1')
                axes[1, 0].set_ylabel('PC2')
                axes[1, 0].legend()
            
            # After correction
            if 'site' in pca_corr.columns:
                for site in pca_corr['site'].unique():
                    subset = pca_corr[pca_corr['site'] == site]
                    axes[1, 1].scatter(subset['PC1'], subset['PC2'], 
                                     label=site, alpha=0.6)
                axes[1, 1].set_title('After Batch Correction')
                axes[1, 1].set_xlabel('PC1')
                axes[1, 1].set_ylabel('PC2')
                axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_analysis(
        self,
        correlation_analysis: Dict,
        output_path: str
    ) -> None:
        """Create correlation analysis plots."""
        fig = plt.figure(figsize=(15, 10))
        
        # Feature correlation heatmap
        feature_corr = correlation_analysis['feature_correlations']
        
        # Select subset for visualization if too large
        if feature_corr.shape[0] > 30:
            # Select features with highest average absolute correlation
            avg_corr = feature_corr.abs().mean()
            top_features = avg_corr.nlargest(30).index
            feature_corr_subset = feature_corr.loc[top_features, top_features]
        else:
            feature_corr_subset = feature_corr
        
        ax1 = plt.subplot(2, 2, (1, 2))
        sns.heatmap(feature_corr_subset, cmap=self.custom_cmap, center=0, 
                   square=True, cbar_kws={'label': 'Correlation'})
        ax1.set_title('Feature-Feature Correlations')
        
        # Feature-covariate correlations
        if not correlation_analysis['feature_covariate_correlations'].empty:
            ax2 = plt.subplot(2, 2, 3)
            feature_cov_corr = correlation_analysis['feature_covariate_correlations']
            sns.heatmap(feature_cov_corr.T, cmap=self.custom_cmap, center=0,
                       cbar_kws={'label': 'Correlation'})
            ax2.set_title('Feature-Covariate Correlations')
            ax2.set_xlabel('Features')
        
        # Correlation distribution
        ax3 = plt.subplot(2, 2, 4)
        corr_values = feature_corr.values[np.triu_indices_from(feature_corr.values, k=1)]
        ax3.hist(corr_values, bins=30, alpha=0.7)
        ax3.set_title('Distribution of Feature Correlations')
        ax3.set_xlabel('Correlation')
        ax3.set_ylabel('Frequency')
        
        plt.suptitle('Correlation Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_comprehensive_eda(
        self,
        outcome: str,
        datasets: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = None
    ) -> Dict[str, Dict]:
        """
        Run comprehensive EDA analysis for all datasets.
        
        Args:
            outcome: Outcome name
            datasets: Optional pre-loaded datasets
            
        Returns:
            Dictionary containing all EDA analyses
        """
        self.logger.info(f"Starting comprehensive EDA for outcome: {outcome}")
        
        # Load datasets if not provided
        if datasets is None:
            datasets = self.load_datasets(outcome)
        
        if not datasets:
            self.logger.error("No datasets available for EDA")
            return {}
        
        # Perform analyses for each dataset
        eda_results = {}
        
        for dataset_type, (metadata, intensity) in datasets.items():
            self.logger.info(f"Analyzing {dataset_type} dataset...")
            
            dataset_analyses = {}
            
            # Basic characteristics
            dataset_analyses['characteristics'] = self.analyze_dataset_characteristics(
                metadata, intensity, dataset_type, outcome
            )
            
            # PCA analysis
            dataset_analyses['pca_analysis'] = self.perform_pca_analysis(
                intensity, metadata, dataset_type, outcome
            )
            
            # Batch effect analysis
            dataset_analyses['batch_analysis'] = self.analyze_batch_effects(
                intensity, metadata, dataset_type
            )
            
            # Correlation analysis
            dataset_analyses['correlation_analysis'] = self.analyze_feature_correlations(
                intensity, metadata, dataset_type
            )
            
            eda_results[dataset_type] = dataset_analyses
        
        # Create visualizations
        output_dir = f"s3://output-bucket/ml_eda/{outcome}/"
        self.create_eda_visualizations(datasets, eda_results, outcome, output_dir)
        
        self.logger.info("Comprehensive EDA completed")
        
        return eda_results


def run_ml_eda(outcome: str, config_dir: str = None) -> Dict[str, Dict]:
    """
    Convenience function to run ML EDA.
    
    Args:
        outcome: Outcome name (eope, lope, sb, ptb, sga)
        config_dir: Configuration directory path
        
    Returns:
        Dictionary containing EDA results
    """
    # Initialize configuration manager
    config_manager = ConfigManager(config_dir)
    config_manager.load_config(outcome=outcome)
    
    # Create EDA analyzer
    eda_analyzer = MLExploratoryAnalysis(config_manager)
    
    # Run comprehensive EDA
    return eda_analyzer.run_comprehensive_eda(outcome)


def main():
    """
    Main function for running ML EDA.
    """
    import argparse
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run ML exploratory data analysis")
    parser.add_argument("outcome", help="Outcome name (eope, lope, sb, ptb, sga)")
    parser.add_argument("--config-dir", help="Configuration directory path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run ML EDA
    try:
        eda_results = run_ml_eda(args.outcome, args.config_dir)
        
        print(f"ML EDA completed for outcome: {args.outcome}")
        for dataset_type, analysis in eda_results.items():
            characteristics = analysis.get('characteristics', {})
            print(f"{dataset_type}: {characteristics.get('n_samples', 0)} samples, "
                  f"{characteristics.get('n_features', 0)} features")
        
    except Exception as e:
        logging.error(f"Failed to run ML EDA: {e}")
        raise


if __name__ == "__main__":
    main()