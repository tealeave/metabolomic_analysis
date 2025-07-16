"""
ML AutoML Module for Metabolomic Analysis Pipeline.
Equivalent to 2__automl.py from the reference Risk_Scores pipeline.

This module provides automated machine learning capabilities including
hyperparameter optimization with Optuna, cross-validation, model training,
and comprehensive evaluation with SHAP explainability.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from ...utils.config_utils import ConfigManager, create_output_directory
from ...capybara.preprocessing import log_median_centering
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class MLAutoMLPipeline:
    """
    Automated machine learning pipeline with hyperparameter optimization.
    """
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize AutoML pipeline.
        
        Args:
            config_manager: Configured ConfigManager instance
        """
        self.config_manager = config_manager
        self.ml_config = config_manager.get_config().ml
        self.logger = logging.getLogger(__name__)
        
        # Initialize optimization tracking
        self.best_trial = None
        self.study = None
        self.cv_results = []
        
    def load_ml_dataset(
        self, 
        outcome: str, 
        dataset_type: str = 'biosample',
        population: str = 'all'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load ML dataset for training.
        
        Args:
            outcome: Outcome name (eope, lope, sb, ptb, sga)
            dataset_type: Type of dataset (biosample, subject, window)
            population: Population filter (all, healthy_control, high_risk, high_risk_broad)
            
        Returns:
            Tuple of (metadata, intensity) DataFrames
        """
        self.logger.info(f"Loading {dataset_type} dataset for {outcome} with {population} population")
        
        # Load from saved ML datasets
        output_dir = f"s3://output-bucket/ml_datasets/{outcome}/"
        metadata_path = f"{output_dir}metadata_{dataset_type}.csv"
        intensity_path = f"{output_dir}intensity_{dataset_type}.csv"
        
        metadata = pd.read_csv(metadata_path, index_col=0)
        intensity = pd.read_csv(intensity_path, index_col=0)
        
        # Apply population filter
        if population != 'all':
            if population in metadata.columns:
                population_filter = metadata[population] == True
                metadata = metadata[population_filter]
                intensity = intensity.loc[metadata.index]
                self.logger.info(f"Filtered to {population} population: {len(metadata)} samples")
            else:
                self.logger.warning(f"Population filter {population} not available")
        
        # Remove lockbox from training data
        if 'dataset_assignment' in metadata.columns:
            non_lockbox = metadata['dataset_assignment'] != 'lockbox'
            metadata_train = metadata[non_lockbox]
            intensity_train = intensity.loc[metadata_train.index]
            
            self.logger.info(f"Removed lockbox: {len(metadata_train)} training samples")
            return metadata_train, intensity_train
        
        return metadata, intensity
    
    def prepare_features_and_targets(
        self,
        metadata: pd.DataFrame,
        intensity: pd.DataFrame,
        outcome: str,
        dataset_type: str = 'biosample'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target variables for ML.
        
        Args:
            metadata: Metadata DataFrame
            intensity: Intensity DataFrame  
            outcome: Outcome name
            dataset_type: Type of dataset
            
        Returns:
            Tuple of (features, targets)
        """
        self.logger.info("Preparing features and targets...")
        
        # Get target variable
        if outcome in metadata.columns:
            y = metadata[outcome]
        else:
            raise ValueError(f"Outcome {outcome} not found in metadata")
        
        # Prepare feature matrix (just intensity for now)
        X = intensity.copy()
        
        # Handle missing values by dropping features with any NaN
        if X.isna().any().any():
            missing_features = X.columns[X.isna().any()].tolist()
            self.logger.warning(f"Dropping {len(missing_features)} features with missing values")
            X = X.dropna(axis=1)
        
        self.logger.info(f"Prepared features: {X.shape[1]} features, {len(y)} samples")
        self.logger.info(f"Outcome distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def apply_undersampling(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        method: str = 'random',
        trial: optuna.Trial = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply undersampling to address class imbalance.
        
        Args:
            X_train: Training features
            y_train: Training targets
            method: Undersampling method
            trial: Optuna trial for hyperparameter suggestion
            
        Returns:
            Resampled (X_train, y_train)
        """
        if y_train.value_counts().min() / len(y_train) > 0.4:
            # Skip undersampling if classes are reasonably balanced
            self.logger.info("Classes are reasonably balanced, skipping undersampling")
            return X_train, y_train
        
        self.logger.info(f"Applying {method} undersampling...")
        
        preprocessing_config = self.ml_config.preprocessing.undersampling
        
        if method == 'random':
            strategy = preprocessing_config.strategies.random.sampling_strategy
            if trial is not None and strategy == 'auto':
                # Optimize sampling strategy
                ratio = trial.suggest_float('sampling_ratio', 0.3, 1.0)
                strategy = {0: int(y_train.sum() / ratio)}
            
            sampler = RandomUnderSampler(
                sampling_strategy=strategy,
                random_state=self.ml_config.random_state
            )
            
        elif method == 'near_miss':
            version = trial.suggest_categorical('near_miss_version', [1, 2, 3]) if trial else 1
            sampler = NearMiss(
                version=version,
                n_jobs=preprocessing_config.strategies.near_miss.n_jobs
            )
            
        elif method == 'tomek':
            sampler = TomekLinks(
                n_jobs=preprocessing_config.strategies.tomek.n_jobs
            )
            
        else:
            raise ValueError(f"Unknown undersampling method: {method}")
        
        # Apply undersampling
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        # Convert back to DataFrame/Series with proper indexing
        X_resampled = pd.DataFrame(
            X_resampled, 
            columns=X_train.columns,
            index=range(len(X_resampled))
        )
        y_resampled = pd.Series(y_resampled, index=X_resampled.index)
        
        self.logger.info(f"After undersampling: {len(X_resampled)} samples, "
                        f"class distribution: {y_resampled.value_counts().to_dict()}")
        
        return X_resampled, y_resampled
    
    def create_xgboost_model(self, trial: optuna.Trial, class_ratio: float) -> xgb.XGBClassifier:
        """
        Create XGBoost model with trial-suggested hyperparameters.
        
        Args:
            trial: Optuna trial for hyperparameter suggestion
            class_ratio: Ratio of negative to positive classes
            
        Returns:
            Configured XGBoost classifier
        """
        model_config = self.ml_config.models.xgboost
        param_ranges = model_config.hyperparameter_ranges
        base_params = model_config.base_params
        
        # Suggest hyperparameters
        params = {}
        
        # Class balancing
        scale_pos_weight_max = min(param_ranges.scale_pos_weight.high, class_ratio * 2)
        params['scale_pos_weight'] = trial.suggest_float(
            'scale_pos_weight',
            param_ranges.scale_pos_weight.low,
            scale_pos_weight_max
        )
        
        # Core hyperparameters
        params['max_depth'] = trial.suggest_int(
            'max_depth',
            param_ranges.max_depth.low,
            param_ranges.max_depth.high
        )
        
        params['learning_rate'] = trial.suggest_float(
            'learning_rate',
            param_ranges.learning_rate.low,
            param_ranges.learning_rate.high,
            log=param_ranges.learning_rate.log
        )
        
        params['n_estimators'] = trial.suggest_int(
            'n_estimators',
            param_ranges.n_estimators.low,
            param_ranges.n_estimators.high
        )
        
        # Regularization
        params['gamma'] = trial.suggest_float(
            'gamma',
            param_ranges.gamma.low,
            param_ranges.gamma.high
        )
        
        params['reg_lambda'] = trial.suggest_float(
            'reg_lambda',
            param_ranges.reg_lambda.low,
            param_ranges.reg_lambda.high,
            log=param_ranges.reg_lambda.log
        )
        
        params['reg_alpha'] = trial.suggest_float(
            'reg_alpha',
            param_ranges.reg_alpha.low,
            param_ranges.reg_alpha.high,
            log=param_ranges.reg_alpha.log
        )
        
        # Additional parameters
        params['min_child_weight'] = trial.suggest_int(
            'min_child_weight',
            param_ranges.min_child_weight.low,
            param_ranges.min_child_weight.high
        )
        
        params['subsample'] = trial.suggest_float(
            'subsample',
            param_ranges.subsample.low,
            param_ranges.subsample.high
        )
        
        params['colsample_bytree'] = trial.suggest_float(
            'colsample_bytree',
            param_ranges.colsample_bytree.low,
            param_ranges.colsample_bytree.high
        )
        
        # Early stopping
        if model_config.early_stopping.enabled:
            early_stopping_rounds = trial.suggest_int(
                'early_stopping_rounds',
                model_config.early_stopping.rounds_range.low,
                model_config.early_stopping.rounds_range.high
            )
            params['early_stopping_rounds'] = early_stopping_rounds
        
        # Combine with base parameters
        final_params = {**base_params, **params}
        
        return xgb.XGBClassifier(**final_params)
    
    def objective_function(
        self,
        trial: optuna.Trial,
        X: pd.DataFrame,
        y: pd.Series,
        metadata: pd.DataFrame,
        outcome: str,
        dataset_type: str
    ) -> float:
        """
        Optuna objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial
            X: Feature matrix
            y: Target vector
            metadata: Metadata DataFrame
            outcome: Outcome name
            dataset_type: Dataset type
            
        Returns:
            Objective value to maximize
        """
        try:
            # Suggest undersampling method
            undersampling_methods = self.ml_config.preprocessing.undersampling.methods
            undersampling_method = trial.suggest_categorical('undersampling_method', undersampling_methods)
            
            # Calculate class ratio for model configuration
            class_counts = y.value_counts()
            class_ratio = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0
            
            # Create model
            model = self.create_xgboost_model(trial, class_ratio)
            
            # Get covariates for this dataset type
            covariates = []
            if dataset_type in self.ml_config.dataset_types:
                dataset_config = self.ml_config.dataset_types[dataset_type]
                covariates = dataset_config.covariates.base + dataset_config.covariates.temporal
                # Filter to available covariates
                covariates = [cov for cov in covariates if cov in metadata.columns]
            
            # Cross-validation
            fold_metrics = []
            folds = make_folds(metadata)
            
            for fold_name, (train_idx, val_idx) in folds:
                # Split data
                X_fold_train, X_fold_val = X.loc[train_idx], X.loc[val_idx]
                y_fold_train, y_fold_val = y.loc[train_idx], y.loc[val_idx]
                metadata_fold = metadata.loc[train_idx]
                
                # Apply undersampling to training fold
                X_fold_train_resampled, y_fold_train_resampled = self.apply_undersampling(
                    X_fold_train, y_fold_train, undersampling_method, trial
                )
                
                # Update metadata for resampled data
                metadata_fold_resampled = metadata_fold.loc[X_fold_train_resampled.index]
                
                # Apply batch correction and add covariates
                X_fold_train_processed, X_fold_val_processed = batch_correct_and_add_covariates(
                    X_fold_train_resampled, metadata_fold_resampled, 
                    covariates=covariates, X_val=X_fold_val
                )
                
                # Apply scaling
                scaler = StandardScaler()
                X_fold_train_scaled = pd.DataFrame(
                    scaler.fit_transform(X_fold_train_processed),
                    index=X_fold_train_processed.index,
                    columns=X_fold_train_processed.columns
                )
                X_fold_val_scaled = pd.DataFrame(
                    scaler.transform(X_fold_val_processed),
                    index=X_fold_val_processed.index,
                    columns=X_fold_val_processed.columns
                )
                
                # Train and evaluate model
                use_early_stopping = self.ml_config.models.xgboost.early_stopping.enabled
                metrics = run_model(
                    model, X_fold_train_scaled, y_fold_train_resampled,
                    X_fold_val_scaled, y_fold_val, early_stopping=use_early_stopping
                )
                
                fold_metrics.append(metrics)
            
            # Aggregate metrics across folds
            aggregated_metrics = self._aggregate_cv_metrics(fold_metrics)
            
            # Store trial results
            trial_result = {
                'trial_number': trial.number,
                'params': trial.params,
                'metrics': aggregated_metrics,
                'fold_metrics': fold_metrics
            }
            self.cv_results.append(trial_result)
            
            # Return objective value
            optimization_config = self.ml_config.optimization
            primary_metric = optimization_config.objective_function.primary_metric
            
            if primary_metric in aggregated_metrics:
                objective_value = aggregated_metrics[primary_metric]
            else:
                # Fallback to PPV_val if configured metric not available
                objective_value = aggregated_metrics.get('PPV_val', 0.0)
            
            self.logger.info(f"Trial {trial.number}: {primary_metric} = {objective_value:.4f}")
            
            return objective_value
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0  # Return poor score for failed trials
    
    def _aggregate_cv_metrics(self, fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate metrics across cross-validation folds.
        
        Args:
            fold_metrics: List of metric dictionaries from each fold
            
        Returns:
            Dictionary of aggregated metrics
        """
        if not fold_metrics:
            return {}
        
        # Get all metric names
        all_metrics = set()
        for metrics in fold_metrics:
            all_metrics.update(metrics.keys())
        
        # Calculate mean and std for each metric
        aggregated = {}
        for metric in all_metrics:
            values = [metrics.get(metric, 0) for metrics in fold_metrics]
            aggregated[metric] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
        
        return aggregated
    
    def run_hyperparameter_optimization(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metadata: pd.DataFrame,
        outcome: str,
        dataset_type: str
    ) -> optuna.Study:
        """
        Run hyperparameter optimization with Optuna.
        
        Args:
            X: Feature matrix
            y: Target vector
            metadata: Metadata DataFrame
            outcome: Outcome name
            dataset_type: Dataset type
            
        Returns:
            Completed Optuna study
        """
        self.logger.info("Starting hyperparameter optimization...")
        
        optimization_config = self.ml_config.optimization
        
        # Create study
        study_name = f"{outcome}_{dataset_type}_optimization"
        self.study = optuna.create_study(
            direction=optimization_config.study_settings.direction,
            study_name=study_name
        )
        
        # Define objective with fixed parameters
        def objective(trial):
            return self.objective_function(trial, X, y, metadata, outcome, dataset_type)
        
        # Run optimization
        n_trials = optimization_config.study_settings.n_trials
        timeout = optimization_config.study_settings.timeout
        
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[self._optuna_callback] if optimization_config.pruning.enabled else None
        )
        
        self.best_trial = self.study.best_trial
        self.logger.info(f"Optimization completed. Best {optimization_config.objective_function.primary_metric}: "
                        f"{self.best_trial.value:.4f}")
        
        return self.study
    
    def _optuna_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """
        Callback function for Optuna optimization.
        
        Args:
            study: Optuna study
            trial: Current trial
        """
        if trial.number % 10 == 0:
            self.logger.info(f"Completed {trial.number} trials. "
                           f"Best value so far: {study.best_value:.4f}")
    
    def train_final_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metadata: pd.DataFrame,
        outcome: str,
        dataset_type: str,
        use_best_params: bool = True
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train final model using best hyperparameters.
        
        Args:
            X: Feature matrix
            y: Target vector
            metadata: Metadata DataFrame
            outcome: Outcome name
            dataset_type: Dataset type
            use_best_params: Whether to use best parameters from optimization
            
        Returns:
            Tuple of (trained_model, final_metrics)
        """
        self.logger.info("Training final model...")
        
        if use_best_params and self.best_trial is not None:
            # Use best hyperparameters
            best_params = self.best_trial.params.copy()
            undersampling_method = best_params.pop('undersampling_method', 'random')
            
            # Calculate class ratio
            class_counts = y.value_counts()
            class_ratio = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0
            
            # Create model with best parameters
            model_config = self.ml_config.models.xgboost
            base_params = model_config.base_params
            
            # Remove trial-specific parameters and combine with base
            model_params = {k: v for k, v in best_params.items() 
                          if not k.startswith(('sampling_', 'near_miss_', 'early_stopping_'))}
            
            final_params = {**base_params, **model_params}
            model = xgb.XGBClassifier(**final_params)
            
        else:
            # Use default parameters
            class_counts = y.value_counts()
            class_ratio = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0
            
            # Create trial with default suggestions for model creation
            import optuna
            default_study = optuna.create_study()
            default_trial = default_study.ask()
            model = self.create_xgboost_model(default_trial, class_ratio)
            undersampling_method = 'random'
        
        # Get covariates
        covariates = []
        if dataset_type in self.ml_config.dataset_types:
            dataset_config = self.ml_config.dataset_types[dataset_type]
            covariates = dataset_config.covariates.base + dataset_config.covariates.temporal
            covariates = [cov for cov in covariates if cov in metadata.columns]
        
        # Apply undersampling
        X_resampled, y_resampled = self.apply_undersampling(X, y, undersampling_method)
        metadata_resampled = metadata.loc[X_resampled.index]
        
        # Apply preprocessing
        X_processed = batch_correct_and_add_covariates(
            X_resampled, metadata_resampled, covariates=covariates
        )
        
        # Apply scaling
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_processed),
            index=X_processed.index,
            columns=X_processed.columns
        )
        
        # Train model
        use_early_stopping = self.ml_config.models.xgboost.early_stopping.enabled
        if use_early_stopping:
            # For final model, we'll split a small validation set for early stopping
            from sklearn.model_selection import train_test_split
            X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
                X_scaled, y_resampled, test_size=0.2, 
                random_state=self.ml_config.random_state, stratify=y_resampled
            )
            
            model.fit(
                X_train_final, y_train_final,
                eval_set=[(X_train_final, y_train_final), (X_val_final, y_val_final)],
                verbose=0
            )
        else:
            model.fit(X_scaled, y_resampled)
        
        # Calculate final metrics on full resampled dataset
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]
        final_metrics = calculate_metrics(y_resampled, y_pred, y_proba, 'final')
        
        self.logger.info("Final model training completed")
        
        return model, final_metrics, scaler, X_processed.columns.tolist()
    
    def evaluate_on_lockbox(
        self,
        model: Any,
        scaler: StandardScaler,
        feature_columns: List[str],
        outcome: str,
        dataset_type: str,
        population: str = 'all'
    ) -> Dict[str, float]:
        """
        Evaluate model on lockbox holdout set.
        
        Args:
            model: Trained model
            scaler: Fitted scaler
            feature_columns: List of feature column names
            outcome: Outcome name
            dataset_type: Dataset type
            population: Population filter
            
        Returns:
            Dictionary of lockbox evaluation metrics
        """
        self.logger.info("Evaluating on lockbox holdout set...")
        
        # Load full dataset including lockbox
        output_dir = f"s3://output-bucket/ml_datasets/{outcome}/"
        metadata_path = f"{output_dir}metadata_{dataset_type}.csv"
        intensity_path = f"{output_dir}intensity_{dataset_type}.csv"
        
        metadata_full = pd.read_csv(metadata_path, index_col=0)
        intensity_full = pd.read_csv(intensity_path, index_col=0)
        
        # Filter to lockbox set
        lockbox_metadata = metadata_full[metadata_full['dataset_assignment'] == 'lockbox']
        lockbox_intensity = intensity_full.loc[lockbox_metadata.index]
        
        if len(lockbox_metadata) == 0:
            self.logger.warning("No lockbox data available")
            return {}
        
        # Apply population filter
        if population != 'all' and population in lockbox_metadata.columns:
            population_filter = lockbox_metadata[population] == True
            lockbox_metadata = lockbox_metadata[population_filter]
            lockbox_intensity = lockbox_intensity.loc[lockbox_metadata.index]
        
        # Prepare features
        X_lockbox, y_lockbox = self.prepare_features_and_targets(
            lockbox_metadata, lockbox_intensity, outcome, dataset_type
        )
        
        # Get covariates
        covariates = []
        if dataset_type in self.ml_config.dataset_types:
            dataset_config = self.ml_config.dataset_types[dataset_type]
            covariates = dataset_config.covariates.base + dataset_config.covariates.temporal
            covariates = [cov for cov in covariates if cov in lockbox_metadata.columns]
        
        # Apply same preprocessing as training (without undersampling)
        X_lockbox_processed = batch_correct_and_add_covariates(
            X_lockbox, lockbox_metadata, covariates=covariates
        )
        
        # Ensure same feature columns as training
        X_lockbox_processed = X_lockbox_processed.reindex(columns=feature_columns, fill_value=0)
        
        # Apply scaling
        X_lockbox_scaled = pd.DataFrame(
            scaler.transform(X_lockbox_processed),
            index=X_lockbox_processed.index,
            columns=X_lockbox_processed.columns
        )
        
        # Make predictions
        y_pred_lockbox = model.predict(X_lockbox_scaled)
        y_proba_lockbox = model.predict_proba(X_lockbox_scaled)[:, 1]
        
        # Calculate metrics
        lockbox_metrics = calculate_metrics(y_lockbox, y_pred_lockbox, y_proba_lockbox, 'lockbox')
        
        self.logger.info(f"Lockbox evaluation completed on {len(y_lockbox)} samples")
        
        return lockbox_metrics
    
    def generate_shap_explanations(
        self,
        model: Any,
        X_sample: pd.DataFrame,
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for model predictions.
        
        Args:
            model: Trained model
            X_sample: Sample of features for explanation
            max_samples: Maximum number of samples to explain
            
        Returns:
            Dictionary containing SHAP values and explanations
        """
        self.logger.info("Generating SHAP explanations...")
        
        # Limit sample size for computational efficiency
        if len(X_sample) > max_samples:
            X_sample = X_sample.sample(n=max_samples, random_state=self.ml_config.random_state)
        
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Calculate feature importance
            feature_importance = pd.DataFrame({
                'feature': X_sample.columns,
                'importance': np.abs(shap_values).mean(0)
            }).sort_values('importance', ascending=False)
            
            shap_explanations = {
                'shap_values': shap_values,
                'feature_importance': feature_importance,
                'expected_value': explainer.expected_value,
                'feature_names': X_sample.columns.tolist()
            }
            
            self.logger.info("SHAP explanations generated successfully")
            
            return shap_explanations
            
        except Exception as e:
            self.logger.error(f"Failed to generate SHAP explanations: {e}")
            return {}
    
    def save_results(
        self,
        model: Any,
        scaler: StandardScaler,
        feature_columns: List[str],
        final_metrics: Dict[str, float],
        lockbox_metrics: Dict[str, float],
        shap_explanations: Dict[str, Any],
        outcome: str,
        dataset_type: str,
        population: str
    ) -> None:
        """
        Save all model results and artifacts.
        
        Args:
            model: Trained model
            scaler: Fitted scaler
            feature_columns: Feature column names
            final_metrics: Final training metrics
            lockbox_metrics: Lockbox evaluation metrics
            shap_explanations: SHAP explanations
            outcome: Outcome name
            dataset_type: Dataset type
            population: Population filter
        """
        self.logger.info("Saving model results...")
        
        # Create output directory
        output_dir = f"s3://output-bucket/ml_results/{outcome}/{dataset_type}/{population}/"
        create_output_directory(output_dir)
        
        # Save model
        model_path = f"{output_dir}model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save scaler
        scaler_path = f"{output_dir}scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save feature columns
        feature_path = f"{output_dir}feature_columns.txt"
        with open(feature_path, 'w') as f:
            f.write('\n'.join(feature_columns))
        
        # Save metrics
        all_metrics = {**final_metrics, **lockbox_metrics}
        metrics_df = pd.DataFrame([all_metrics])
        metrics_df.to_csv(f"{output_dir}metrics.csv", index=False)
        
        # Save CV results
        if self.cv_results:
            cv_results_df = pd.DataFrame(self.cv_results)
            cv_results_df.to_csv(f"{output_dir}cv_results.csv", index=False)
        
        # Save best trial parameters
        if self.best_trial:
            best_params_df = pd.DataFrame([self.best_trial.params])
            best_params_df.to_csv(f"{output_dir}best_parameters.csv", index=False)
        
        # Save SHAP explanations
        if shap_explanations:
            feature_importance = shap_explanations['feature_importance']
            feature_importance.to_csv(f"{output_dir}feature_importance.csv", index=False)
        
        # Save study results if available
        if self.study:
            study_df = self.study.trials_dataframe()
            study_df.to_csv(f"{output_dir}optuna_study.csv", index=False)
        
        self.logger.info(f"Results saved to {output_dir}")
    
    def run_complete_pipeline(
        self,
        outcome: str,
        dataset_type: str = 'biosample',
        population: str = 'all'
    ) -> Dict[str, Any]:
        """
        Run the complete AutoML pipeline.
        
        Args:
            outcome: Outcome name (eope, lope, sb, ptb, sga)
            dataset_type: Dataset type (biosample, subject, window)
            population: Population filter (all, healthy_control, high_risk, high_risk_broad)
            
        Returns:
            Dictionary containing all pipeline results
        """
        self.logger.info(f"Starting complete AutoML pipeline for {outcome} - {dataset_type} - {population}")
        
        # Load dataset
        metadata, intensity = self.load_ml_dataset(outcome, dataset_type, population)
        
        # Prepare features and targets
        X, y = self.prepare_features_and_targets(metadata, intensity, outcome, dataset_type)
        
        # Run hyperparameter optimization
        study = self.run_hyperparameter_optimization(X, y, metadata, outcome, dataset_type)
        
        # Train final model
        model, final_metrics, scaler, feature_columns = self.train_final_model(
            X, y, metadata, outcome, dataset_type
        )
        
        # Evaluate on lockbox
        lockbox_metrics = self.evaluate_on_lockbox(
            model, scaler, feature_columns, outcome, dataset_type, population
        )
        
        # Generate SHAP explanations
        shap_explanations = self.generate_shap_explanations(model, X)
        
        # Save all results
        self.save_results(
            model, scaler, feature_columns, final_metrics, lockbox_metrics,
            shap_explanations, outcome, dataset_type, population
        )
        
        # Compile results
        pipeline_results = {
            'outcome': outcome,
            'dataset_type': dataset_type,
            'population': population,
            'final_metrics': final_metrics,
            'lockbox_metrics': lockbox_metrics,
            'best_parameters': self.best_trial.params if self.best_trial else {},
            'feature_importance': shap_explanations.get('feature_importance'),
            'n_trials': len(study.trials) if study else 0,
            'best_objective_value': self.best_trial.value if self.best_trial else None
        }
        
        self.logger.info("Complete AutoML pipeline finished successfully")
        
        return pipeline_results


def run_automl_pipeline(
    outcome: str,
    dataset_type: str = 'biosample',
    population: str = 'all',
    config_dir: str = None
) -> Dict[str, Any]:
    """
    Convenience function to run AutoML pipeline.
    
    Args:
        outcome: Outcome name (eope, lope, sb, ptb, sga)
        dataset_type: Dataset type (biosample, subject, window)
        population: Population filter
        config_dir: Configuration directory path
        
    Returns:
        Dictionary containing pipeline results
    """
    # Initialize configuration manager
    config_manager = ConfigManager(config_dir)
    config_manager.load_config(outcome=outcome)
    
    # Create AutoML pipeline
    automl = MLAutoMLPipeline(config_manager)
    
    # Run complete pipeline
    return automl.run_complete_pipeline(outcome, dataset_type, population)


def main():
    """
    Main function for running AutoML pipeline.
    """
    import argparse
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Run AutoML pipeline for risk score prediction")
    parser.add_argument("outcome", help="Outcome name (eope, lope, sb, ptb, sga)")
    parser.add_argument("--dataset-type", default="biosample", 
                       choices=["biosample", "subject", "window"],
                       help="Dataset type to use")
    parser.add_argument("--population", default="all",
                       choices=["all", "healthy_control", "high_risk", "high_risk_broad"],
                       help="Population filter to apply")
    parser.add_argument("--config-dir", help="Configuration directory path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run AutoML pipeline
    try:
        results = run_automl_pipeline(
            args.outcome, args.dataset_type, args.population, args.config_dir
        )
        
        print(f"AutoML pipeline completed for {args.outcome}")
        print(f"Dataset: {args.dataset_type}, Population: {args.population}")
        print(f"Best objective value: {results.get('best_objective_value', 'N/A'):.4f}")
        print(f"Number of trials: {results.get('n_trials', 0)}")
        
        if 'lockbox_metrics' in results and results['lockbox_metrics']:
            lockbox_mcc = results['lockbox_metrics'].get('MCC_lockbox', 'N/A')
            print(f"Lockbox MCC: {lockbox_mcc}")
        
    except Exception as e:
        logging.error(f"Failed to run AutoML pipeline: {e}")
        raise


if __name__ == "__main__":
    main()