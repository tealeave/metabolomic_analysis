"""
Master Orchestrator for Metabolomic Analysis Pipeline

This module coordinates both the metabolomics biomarker discovery pipeline
and the machine learning risk prediction pipeline, enabling end-to-end analysis
from raw data to predictive models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import os

from src.utils.config_utils import ConfigManager
from src.utils.spark_utils import create_spark_session
from src.pipelines.metabolomics.orchestrator import MetabolomicsOrchestrator
from src.pipelines.machine_learning.orchestrator import MachineLearningOrchestrator

logger = logging.getLogger(__name__)


class MasterOrchestrator:
    """
    Master orchestrator that coordinates metabolomics and ML pipelines.
    
    This class provides a unified interface for running either individual pipelines
    or the complete end-to-end workflow from biomarker discovery to risk prediction.
    """
    
    def __init__(self, 
                 metabolomics_config_path: str = "config/metabolomics_config.yaml",
                 ml_config_path: str = "config/ml_config.yaml"):
        """
        Initialize the master orchestrator.
        
        Args:
            metabolomics_config_path: Path to metabolomics pipeline configuration
            ml_config_path: Path to ML pipeline configuration
        """
        self.metabolomics_config_path = metabolomics_config_path
        self.ml_config_path = ml_config_path
        
        # Initialize configuration managers
        self.metabolomics_config_manager = ConfigManager()
        self.ml_config_manager = ConfigManager()
        
        # Pipeline orchestrators (initialized when needed)
        self.metabolomics_orchestrator = None
        self.ml_orchestrator = None
        
        # Shared Spark session
        self.spark = None
        
        logger.info("Master orchestrator initialized")
    
    def run_full_pipeline(self, 
                         outcome: str,
                         biomarker_selection_mode: str = "auto",
                         biomarker_list: Optional[List[str]] = None,
                         run_metabolomics: bool = True,
                         run_ml: bool = True,
                         **kwargs) -> Dict[str, Any]:
        """
        Run the complete end-to-end pipeline.
        
        Args:
            outcome: Target outcome (sb, ptb, sga, eope, lope)
            biomarker_selection_mode: How to select biomarkers for ML ("auto", "user_specified", "top_n", "criteria")
            biomarker_list: List of biomarkers to use (required if mode="user_specified")
            run_metabolomics: Whether to run metabolomics pipeline
            run_ml: Whether to run ML pipeline
            **kwargs: Additional configuration overrides
            
        Returns:
            Dictionary containing results from both pipelines
        """
        logger.info(f"Starting full pipeline for outcome: {outcome}")
        
        results = {
            "outcome": outcome,
            "biomarker_selection_mode": biomarker_selection_mode,
            "metabolomics_results": None,
            "ml_results": None,
            "selected_biomarkers": None,
            "pipeline_summary": {}
        }
        
        try:
            # Step 1: Run metabolomics biomarker discovery pipeline
            if run_metabolomics:
                logger.info("=== Starting Metabolomics Biomarker Discovery Pipeline ===")
                metabolomics_results = self.run_metabolomics_pipeline(outcome, **kwargs)
                results["metabolomics_results"] = metabolomics_results
                logger.info("Metabolomics pipeline completed successfully")
            
            # Step 2: Run ML risk prediction pipeline
            if run_ml:
                logger.info("=== Starting Machine Learning Risk Prediction Pipeline ===")
                
                # Configure biomarker selection
                ml_kwargs = kwargs.copy()
                ml_kwargs["biomarker_selection_mode"] = biomarker_selection_mode
                if biomarker_list:
                    ml_kwargs["user_biomarkers"] = biomarker_list
                
                ml_results = self.run_ml_pipeline(outcome, **ml_kwargs)
                results["ml_results"] = ml_results
                results["selected_biomarkers"] = ml_results.get("selected_biomarkers", [])
                logger.info("Machine learning pipeline completed successfully")
            
            # Step 3: Generate pipeline summary
            results["pipeline_summary"] = self._generate_pipeline_summary(results)
            
            logger.info("Full pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in full pipeline: {str(e)}")
            raise
        finally:
            self._cleanup()
    
    def run_metabolomics_pipeline(self, 
                                 outcome: str,
                                 **kwargs) -> Dict[str, Any]:
        """
        Run only the metabolomics biomarker discovery pipeline.
        
        Args:
            outcome: Target outcome
            **kwargs: Configuration overrides
            
        Returns:
            Dictionary containing metabolomics pipeline results
        """
        logger.info(f"Running metabolomics pipeline for outcome: {outcome}")
        
        try:
            # Load metabolomics configuration
            overrides = [f"outcome={outcome}"] + [f"{k}={v}" for k, v in kwargs.items()]
            self.metabolomics_config_manager.load_config(
                config_name=self.metabolomics_config_path,
                overrides=overrides
            )
            
            # Create Spark session
            self._initialize_spark("metabolomics")
            
            # Initialize and run metabolomics orchestrator
            self.metabolomics_orchestrator = MetabolomicsOrchestrator(
                config_manager=self.metabolomics_config_manager,
                spark=self.spark
            )
            
            results = self.metabolomics_orchestrator.run_full_pipeline(outcome)
            
            logger.info("Metabolomics pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in metabolomics pipeline: {str(e)}")
            raise
    
    def run_ml_pipeline(self, 
                       outcome: str,
                       **kwargs) -> Dict[str, Any]:
        """
        Run only the machine learning risk prediction pipeline.
        
        Args:
            outcome: Target outcome
            **kwargs: Configuration overrides (including biomarker selection parameters)
            
        Returns:
            Dictionary containing ML pipeline results
        """
        logger.info(f"Running ML pipeline for outcome: {outcome}")
        
        try:
            # Load ML configuration
            overrides = [f"outcome={outcome}"] + [f"{k}={v}" for k, v in kwargs.items()]
            self.ml_config_manager.load_config(
                config_name=self.ml_config_path,
                overrides=overrides
            )
            
            # Create Spark session if not already created
            if not self.spark:
                self._initialize_spark("ml")
            
            # Initialize and run ML orchestrator
            self.ml_orchestrator = MachineLearningOrchestrator(
                config_manager=self.ml_config_manager,
                spark=self.spark
            )
            
            results = self.ml_orchestrator.run_full_pipeline(outcome)
            
            logger.info("ML pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in ML pipeline: {str(e)}")
            raise
    
    def run_biomarker_discovery_only(self, 
                                   outcome: str,
                                   **kwargs) -> Dict[str, Any]:
        """
        Run only biomarker discovery (metabolomics pipeline steps 1-4).
        
        Args:
            outcome: Target outcome
            **kwargs: Configuration overrides
            
        Returns:
            Dictionary containing biomarker discovery results
        """
        logger.info(f"Running biomarker discovery for outcome: {outcome}")
        
        # This runs metabolomics pipeline but stops before results comparison
        metabolomics_results = self.run_metabolomics_pipeline(outcome, **kwargs)
        
        # Extract just the biomarker discovery results
        discovery_results = {
            "outcome": outcome,
            "significant_biomarkers": metabolomics_results.get("regression_results", {}),
            "data_manifest": metabolomics_results.get("data_manifest", {}),
            "metadata_processing": metabolomics_results.get("metadata_processing", {}),
            "stratified_integration": metabolomics_results.get("stratified_integration", {}),
        }
        
        logger.info("Biomarker discovery completed successfully")
        return discovery_results
    
    def list_available_biomarkers(self, outcome: str) -> List[str]:
        """
        Get list of available biomarkers for an outcome from previous analyses.
        
        Args:
            outcome: Target outcome
            
        Returns:
            List of available biomarker identifiers
        """
        logger.info(f"Listing available biomarkers for outcome: {outcome}")
        
        try:
            # Load ML configuration to get biomarker results path
            self.ml_config_manager.load_config(
                config_name=self.ml_config_path,
                overrides=[f"outcome={outcome}"]
            )
            
            # Create minimal Spark session
            if not self.spark:
                self._initialize_spark("ml")
            
            # Initialize ML orchestrator to use biomarker selector
            self.ml_orchestrator = MachineLearningOrchestrator(
                config_manager=self.ml_config_manager,
                spark=self.spark
            )
            
            # Get available biomarkers
            available_biomarkers = self.ml_orchestrator.biomarker_selector.get_available_biomarkers(outcome)
            
            logger.info(f"Found {len(available_biomarkers)} available biomarkers")
            return available_biomarkers
            
        except Exception as e:
            logger.error(f"Error listing available biomarkers: {str(e)}")
            return []
    
    def validate_biomarker_list(self, outcome: str, biomarker_list: List[str]) -> Dict[str, Any]:
        """
        Validate a user-provided biomarker list.
        
        Args:
            outcome: Target outcome
            biomarker_list: List of biomarker identifiers to validate
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating biomarker list for outcome: {outcome}")
        
        available_biomarkers = self.list_available_biomarkers(outcome)
        
        valid_biomarkers = [b for b in biomarker_list if b in available_biomarkers]
        invalid_biomarkers = [b for b in biomarker_list if b not in available_biomarkers]
        
        validation_results = {
            "total_requested": len(biomarker_list),
            "valid_count": len(valid_biomarkers),
            "invalid_count": len(invalid_biomarkers),
            "valid_biomarkers": valid_biomarkers,
            "invalid_biomarkers": invalid_biomarkers,
            "validation_passed": len(invalid_biomarkers) == 0
        }
        
        if invalid_biomarkers:
            logger.warning(f"Found {len(invalid_biomarkers)} invalid biomarkers: {invalid_biomarkers[:5]}...")
        else:
            logger.info("All biomarkers validated successfully")
        
        return validation_results
    
    def _initialize_spark(self, pipeline_type: str = "master"):
        """Initialize Spark session for the pipeline."""
        if not self.spark:
            self.spark = create_spark_session(
                app_name=f"metabolomic-analysis-{pipeline_type}",
                enable_delta=True
            )
            logger.info(f"Spark session initialized for {pipeline_type} pipeline")
    
    def _generate_pipeline_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of pipeline execution."""
        summary = {
            "outcome": results["outcome"],
            "pipelines_run": [],
            "biomarker_selection": {
                "mode": results["biomarker_selection_mode"],
                "selected_count": len(results.get("selected_biomarkers", []))
            }
        }
        
        if results["metabolomics_results"]:
            summary["pipelines_run"].append("metabolomics")
            summary["metabolomics_summary"] = {
                "steps_completed": len([k for k, v in results["metabolomics_results"].items() if v]),
                "data_manifest_files": len(results["metabolomics_results"].get("data_manifest", {}).get("files", [])),
                "cases_processed": results["metabolomics_results"].get("metadata_processing", {}).get("case_count", 0),
                "significant_biomarkers": len(results["metabolomics_results"].get("regression_results", {}).get("significant_features", []))
            }
        
        if results["ml_results"]:
            summary["pipelines_run"].append("machine_learning")
            summary["ml_summary"] = {
                "model_performance": results["ml_results"].get("model_performance", {}),
                "features_used": len(results["ml_results"].get("selected_biomarkers", [])),
                "best_model": results["ml_results"].get("best_model_type", "unknown")
            }
        
        return summary
    
    def _cleanup(self):
        """Clean up resources."""
        if self.spark:
            try:
                self.spark.stop()
                logger.info("Spark session stopped")
            except Exception as e:
                logger.warning(f"Error stopping Spark session: {str(e)}")
            finally:
                self.spark = None


def main():
    """
    Main function for command-line usage.
    
    Example usage:
        python -m src.master_orchestrator --outcome sb --mode full --biomarker-selection auto
        python -m src.master_orchestrator --outcome ptb --mode metabolomics --verbose
        python -m src.master_orchestrator --outcome sga --mode ml --biomarker-selection user_specified --biomarkers "MTB001,MTB002,MTB003"
    """
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Master Orchestrator for Metabolomic Analysis Pipeline")
    parser.add_argument("--outcome", required=True, choices=["sb", "ptb", "sga", "eope", "lope"],
                       help="Target outcome for analysis")
    parser.add_argument("--mode", choices=["full", "metabolomics", "ml", "discovery"], default="full",
                       help="Pipeline mode to run")
    parser.add_argument("--biomarker-selection", choices=["auto", "user_specified", "top_n", "criteria"], default="auto",
                       help="Biomarker selection mode for ML pipeline")
    parser.add_argument("--biomarkers", type=str,
                       help="Comma-separated list of biomarkers (for user_specified mode)")
    parser.add_argument("--config-overrides", type=str,
                       help="JSON string of configuration overrides")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Parse configuration overrides
    config_overrides = {}
    if args.config_overrides:
        try:
            config_overrides = json.loads(args.config_overrides)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in config-overrides")
            return 1
    
    # Parse biomarker list
    biomarker_list = None
    if args.biomarkers:
        biomarker_list = [b.strip() for b in args.biomarkers.split(",")]
    
    try:
        # Initialize orchestrator
        orchestrator = MasterOrchestrator()
        
        # Run requested pipeline mode
        if args.mode == "full":
            results = orchestrator.run_full_pipeline(
                outcome=args.outcome,
                biomarker_selection_mode=args.biomarker_selection,
                biomarker_list=biomarker_list,
                **config_overrides
            )
        elif args.mode == "metabolomics":
            results = orchestrator.run_metabolomics_pipeline(
                outcome=args.outcome,
                **config_overrides
            )
        elif args.mode == "ml":
            results = orchestrator.run_ml_pipeline(
                outcome=args.outcome,
                biomarker_selection_mode=args.biomarker_selection,
                user_biomarkers=biomarker_list,
                **config_overrides
            )
        elif args.mode == "discovery":
            results = orchestrator.run_biomarker_discovery_only(
                outcome=args.outcome,
                **config_overrides
            )
        
        # Print summary
        if "pipeline_summary" in results:
            print("\n=== Pipeline Summary ===")
            summary = results["pipeline_summary"]
            print(f"Outcome: {summary['outcome']}")
            print(f"Pipelines run: {', '.join(summary['pipelines_run'])}")
            
            if "metabolomics_summary" in summary:
                ms = summary["metabolomics_summary"]
                print(f"Metabolomics: {ms['steps_completed']} steps, {ms['significant_biomarkers']} significant biomarkers")
            
            if "ml_summary" in summary:
                mls = summary["ml_summary"]
                print(f"ML: {mls['features_used']} features, best model: {mls['best_model']}")
        
        logger.info("Pipeline execution completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())