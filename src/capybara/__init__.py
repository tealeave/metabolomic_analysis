"""
Capybara: Personal utilities package for metabolomic analysis.

This package provides statistical, regression, and visualization utilities
adapted from the wombat package for personal use.

Modules:
- regression: Logistic and linear regression utilities
- statistics: Hypothesis testing and multiple testing correction
- visualization: Plotting and charting utilities  
- preprocessing: Data preprocessing and transformation
"""

__version__ = "0.1.0"
__author__ = "Personal Development"

# Core modules
from . import regression
from . import statistics 
from . import preprocessing
from . import visualization

# Key functions for easy access
from .regression.logistic import logistic_regression
from .statistics.multitest import multitest_correct, CorrectionMethods
from .preprocessing.data_cleaning import clean_intensity_data
from .preprocessing.normalization import normalize_intensity_data, log_median_centering
from .preprocessing.feature_selection import comprehensive_feature_selection
from .visualization.plots import create_volcano_plot, create_pca_plot