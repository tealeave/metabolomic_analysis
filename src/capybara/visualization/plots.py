"""
Basic plotting utilities for metabolomic analysis.

This module provides essential plotting functions for biomarker discovery
and model evaluation workflows.
"""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def create_volcano_plot(
    results_df: pd.DataFrame,
    p_value_col: str = "p_value_corrected",
    effect_col: str = "estimate",
    significance_threshold: float = 0.05,
    effect_threshold: float = 0.5,
    title: str = "Volcano Plot",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a volcano plot for biomarker discovery results.
    
    Args:
        results_df: DataFrame with regression results
        p_value_col: Column name for p-values
        effect_col: Column name for effect sizes
        significance_threshold: P-value threshold for significance
        effect_threshold: Minimum effect size threshold
        title: Plot title
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    logger.info("Creating volcano plot")
    
    # Prepare data
    df = results_df.copy()
    df = df.dropna(subset=[p_value_col, effect_col])
    
    if df.empty:
        logger.warning("No valid data for volcano plot")
        return plt.figure()
    
    # Calculate -log10(p-value)
    df['neg_log10_pval'] = -np.log10(df[p_value_col].clip(lower=1e-300))
    
    # Determine significance and effect size
    df['significant'] = (df[p_value_col] < significance_threshold) & (np.abs(df[effect_col]) > effect_threshold)
    df['color'] = df['significant'].map({True: 'red', False: 'gray'})
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    scatter = ax.scatter(
        df[effect_col], 
        df['neg_log10_pval'],
        c=df['color'],
        alpha=0.6,
        s=20
    )
    
    # Add threshold lines
    ax.axhline(y=-np.log10(significance_threshold), color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=effect_threshold, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=-effect_threshold, color='black', linestyle='--', alpha=0.5)
    
    # Labels and title
    ax.set_xlabel(f'Effect Size ({effect_col})')
    ax.set_ylabel(f'-log10({p_value_col})')
    ax.set_title(title)
    
    # Add significance count
    n_significant = df['significant'].sum()
    ax.text(0.02, 0.98, f'Significant: {n_significant}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Volcano plot saved to {save_path}")
    
    return fig


def create_pca_plot(
    intensity_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    color_by: str = "outcome",
    title: str = "PCA Plot",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a PCA plot for exploratory data analysis.
    
    Args:
        intensity_df: DataFrame with intensity data (samples x features)
        metadata_df: DataFrame with sample metadata
        color_by: Column in metadata to color points by
        title: Plot title
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    logger.info("Creating PCA plot")
    
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data
        common_samples = intensity_df.index.intersection(metadata_df.index)
        if len(common_samples) == 0:
            logger.error("No common samples between intensity and metadata")
            return plt.figure()
        
        X = intensity_df.loc[common_samples].fillna(0)
        meta = metadata_df.loc[common_samples]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if color_by in meta.columns:
            # Color by specified column
            unique_values = meta[color_by].unique()
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_values)))
            
            for i, value in enumerate(unique_values):
                mask = meta[color_by] == value
                ax.scatter(
                    X_pca[mask, 0], 
                    X_pca[mask, 1],
                    c=[colors[i]], 
                    label=str(value),
                    alpha=0.7,
                    s=50
                )
            ax.legend()
        else:
            # Single color if column not found
            ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=50)
        
        # Labels
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PCA plot saved to {save_path}")
        
        return fig
        
    except ImportError:
        logger.error("scikit-learn not available for PCA")
        return plt.figure()


def create_feature_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: Optional[str] = None,
    plot_type: str = "scatter",
    title: str = "Feature Plot",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a feature plot for exploratory data analysis.
    
    Args:
        data: DataFrame with data to plot
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        hue_col: Optional column name for color grouping
        plot_type: Type of plot ("scatter", "box", "violin")
        title: Plot title
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    logger.info(f"Creating {plot_type} plot")
    
    # Clean data
    df = data.dropna(subset=[x_col, y_col])
    if hue_col and hue_col in df.columns:
        df = df.dropna(subset=[hue_col])
    
    if df.empty:
        logger.warning("No valid data for feature plot")
        return plt.figure()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        if plot_type == "scatter":
            if hue_col:
                sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
            else:
                sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
                
        elif plot_type == "box":
            if hue_col:
                sns.boxplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
            else:
                sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
                
        elif plot_type == "violin":
            if hue_col:
                sns.violinplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
            else:
                sns.violinplot(data=df, x=x_col, y=y_col, ax=ax)
        
        ax.set_title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature plot saved to {save_path}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating feature plot: {str(e)}")
        return plt.figure()


def create_performance_plot(
    performance_df: pd.DataFrame,
    metric_col: str = "AUROC",
    group_col: str = "dataset_type",
    title: str = "Model Performance Comparison",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a performance comparison plot for ML models.
    
    Args:
        performance_df: DataFrame with model performance metrics
        metric_col: Column name for performance metric
        group_col: Column name for grouping (e.g., dataset type)
        title: Plot title
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    logger.info("Creating performance comparison plot")
    
    # Clean data
    df = performance_df.dropna(subset=[metric_col, group_col])
    
    if df.empty:
        logger.warning("No valid data for performance plot")
        return plt.figure()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        sns.boxplot(data=df, x=group_col, y=metric_col, ax=ax)
        sns.swarmplot(data=df, x=group_col, y=metric_col, color='red', size=4, ax=ax)
        
        ax.set_title(title)
        ax.set_ylabel(metric_col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance plot saved to {save_path}")
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating performance plot: {str(e)}")
        return plt.figure()


def set_plot_style(style: str = "whitegrid"):
    """
    Set the plotting style for consistent visualizations.
    
    Args:
        style: Seaborn style name
    """
    try:
        sns.set_style(style)
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
    except Exception as e:
        logger.warning(f"Could not set plot style: {str(e)}")


# Set default style
set_plot_style()