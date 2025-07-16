"""
Machine learning pipeline for risk prediction.

This pipeline performs the following steps:
1. Biomarker selection (user-specified or from metabolomics results)
2. Data processing and EDA
3. AutoML model training and evaluation

Supports multiple outcomes: EOPE, LOPE, SB, PTB, SGA
"""