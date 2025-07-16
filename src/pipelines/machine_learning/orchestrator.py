"""
ML Pipeline Orchestrator for Metabolomic Analysis Pipeline.

This module orchestrates the complete machine learning pipeline from biomarker selection
through model training and evaluation, providing a unified interface for running
the entire ML workflow.

Pipeline Steps:
1. Biomarker selection (user-specified or from metabolomics results)
2. Data processing and feature engineering
3. Exploratory data analysis
4. AutoML model training and evaluation
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pyspark.sql import SparkSession

from ...utils.config_utils import ConfigManager
from ...utils.spark_utils import create_spark_session
from .biomarker_selector import BiomarkerSelector
from .data_processing import MLDataProcessor
from .eda import MLExploratoryAnalysis
from .automl import MLAutoMLPipeline


class MLPipelineOrchestrator:
    """
    Orchestrates the complete ML pipeline for metabolomic analysis.
    
    This class coordinates all ML pipeline components and manages the workflow
    from biomarker selection to trained model deployment.
    """
    
    def __init__(self, config_manager: ConfigManager, spark_session: SparkSession = None):
        """
        Initialize ML pipeline orchestrator.
        
        Args:
            config_manager: Configuration manager instance
            spark_session: Optional Spark session
        """
        self.config_manager = config_manager
        self.config = config_manager.config
        
        # Initialize Spark session if not provided
        if spark_session is None:
            self.spark = create_spark_session("ml-pipeline", enable_delta=True)
        else:
            self.spark = spark_session
        
        # Initialize pipeline components
        self.biomarker_selector = BiomarkerSelector(config_manager, self.spark)
        self.data_processor = MLDataProcessor(config_manager, self.spark)
        self.eda_analyzer = MLExploratoryAnalysis(config_manager)
        self.automl_pipeline = MLAutoMLPipeline(config_manager)
        
        # Track pipeline state
        self.pipeline_state = {
            "biomarker_selection": {"completed": False, "output": None, "duration": None},
            "data_processing": {"completed": False, "output": None, "duration": None},
            "eda": {"completed": False, "output": None, "duration": None},
            "automl": {"completed": False, "output": None, "duration": None}
        }
    
    def run_biomarker_selection(
        self, 
        outcome: str, 
        mode: str = "auto",
        **kwargs
    ) -> List[str]:
        """
        Run biomarker selection step.
        
        Args:
            outcome: The outcome of interest
            mode: Selection mode ("user_specified", "top_n", "auto")
            **kwargs: Additional arguments for biomarker selection
            
        Returns:
            List of selected biomarkers
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Step 1/4: Running biomarker selection for outcome: {outcome}")
        start_time = time.time()
        
        try:
            selected_biomarkers = self.biomarker_selector.select_biomarkers(
                outcome, mode=mode, **kwargs
            )
            
            duration = time.time() - start_time
            self.pipeline_state["biomarker_selection"] = {
                "completed": True,
                "output": selected_biomarkers,
                "duration": duration
            }
            
            logger.info(f"Biomarker selection completed in {duration:.2f} seconds")
            logger.info(f"Selected {len(selected_biomarkers)} biomarkers")
            
            return selected_biomarkers
            
        except Exception as e:
            logger.error(f"Biomarker selection failed: {str(e)}")
            raise
    
    def run_data_processing(
        self, 
        outcome: str, 
        biomarkers: List[str],
        dataset_type: str = "biosample"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run data processing step.
        
        Args:
            outcome: The outcome of interest
            biomarkers: Selected biomarkers
            dataset_type: Type of dataset ("biosample", "subject", "window")
            
        Returns:
            Tuple of (metadata, intensity_data)
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Step 2/4: Running data processing for outcome: {outcome}")
        start_time = time.time()
        
        try:
            processed_data = self.data_processor.run(
                outcome, biomarkers=biomarkers, dataset_type=dataset_type
            )
            
            duration = time.time() - start_time
            self.pipeline_state["data_processing"] = {
                "completed": True,
                "output": processed_data,
                "duration": duration
            }
            
            logger.info(f"Data processing completed in {duration:.2f} seconds")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Data processing failed: {str(e)}")
            raise
    
    def run_eda(
        self, 
        outcome: str, 
        metadata: pd.DataFrame, 
        intensity_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Run exploratory data analysis step.
        
        Args:
            outcome: The outcome of interest
            metadata: Processed metadata
            intensity_data: Processed intensity data
            
        Returns:
            Dictionary with EDA results
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Step 3/4: Running EDA for outcome: {outcome}")
        start_time = time.time()
        
        try:
            eda_results = self.eda_analyzer.run(outcome, metadata, intensity_data)
            
            duration = time.time() - start_time
            self.pipeline_state["eda"] = {
                "completed": True,
                "output": eda_results,
                "duration": duration
            }
            
            logger.info(f"EDA completed in {duration:.2f} seconds")
            
            return eda_results
            
        except Exception as e:
            logger.error(f"EDA failed: {str(e)}")
            raise
    
    def run_automl(
        self, 
        outcome: str, 
        metadata: pd.DataFrame, 
        intensity_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Run AutoML training step.
        
        Args:
            outcome: The outcome of interest
            metadata: Processed metadata
            intensity_data: Processed intensity data
            
        Returns:
            Dictionary with AutoML results
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Step 4/4: Running AutoML for outcome: {outcome}")
        start_time = time.time()
        
        try:
            automl_results = self.automl_pipeline.run(outcome, metadata, intensity_data)
            
            duration = time.time() - start_time
            self.pipeline_state["automl"] = {
                "completed": True,
                "output": automl_results,
                "duration": duration
            }
            
            logger.info(f"AutoML completed in {duration:.2f} seconds")
            
            return automl_results
            
        except Exception as e:
            logger.error(f"AutoML failed: {str(e)}")
            raise
    
    def run_pipeline(
        self, 
        outcome: str, 
        biomarker_mode: str = "auto",
        dataset_type: str = "biosample",
        steps: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the complete ML pipeline or specified steps.
        
        Args:
            outcome: The outcome of interest
            biomarker_mode: Biomarker selection mode
            dataset_type: Type of dataset to use
            steps: Optional list of steps to run
            **kwargs: Additional arguments for pipeline steps
            
        Returns:
            Dictionary with pipeline results and state
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Starting ML pipeline for outcome: {outcome}")
        
        if steps is None:
            steps = ["biomarker_selection", "data_processing", "eda", "automl"]
        
        pipeline_start_time = time.time()
        
        try:
            # Step 1: Biomarker selection
            if "biomarker_selection" in steps:
                selected_biomarkers = self.run_biomarker_selection(
                    outcome, mode=biomarker_mode, **kwargs
                )
            else:
                # Use previously selected biomarkers or default
                selected_biomarkers = kwargs.get("biomarkers", [])
            
            # Step 2: Data processing
            if "data_processing" in steps:
                metadata, intensity_data = self.run_data_processing(
                    outcome, selected_biomarkers, dataset_type
                )
            else:
                # Load previously processed data
                metadata, intensity_data = kwargs.get("processed_data", (None, None))
            
            # Step 3: EDA
            eda_results = None
            if "eda" in steps and metadata is not None and intensity_data is not None:
                eda_results = self.run_eda(outcome, metadata, intensity_data)
            
            # Step 4: AutoML
            automl_results = None
            if "automl" in steps and metadata is not None and intensity_data is not None:
                automl_results = self.run_automl(outcome, metadata, intensity_data)
            
            total_duration = time.time() - pipeline_start_time
            
            logger.info(f"ML pipeline completed successfully in {total_duration:.2f} seconds")
            
            # Return pipeline summary
            return {
                "outcome": outcome,
                "status": "completed",
                "total_duration": total_duration,
                "steps_run": steps,
                "selected_biomarkers": selected_biomarkers,
                "eda_results": eda_results,
                "automl_results": automl_results,
                "pipeline_state": self.pipeline_state
            }
            
        except Exception as e:
            total_duration = time.time() - pipeline_start_time
            logger.error(f"ML pipeline failed after {total_duration:.2f} seconds: {str(e)}")
            
            return {
                "outcome": outcome,
                "status": "failed",
                "error": str(e),
                "total_duration": total_duration,
                "steps_run": steps,
                "pipeline_state": self.pipeline_state
            }
        
        finally:
            # Cleanup Spark resources
            if hasattr(self, 'spark') and self.spark is not None:
                self.spark.catalog.clearCache()
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the pipeline execution state.
        
        Returns:
            Dictionary with pipeline execution summary
        """
        completed_steps = [step for step, state in self.pipeline_state.items() if state["completed"]]
        total_duration = sum(state["duration"] or 0 for state in self.pipeline_state.values())
        
        return {
            "completed_steps": completed_steps,
            "total_steps": len(self.pipeline_state),
            "completion_rate": len(completed_steps) / len(self.pipeline_state),
            "total_duration": total_duration,
            "pipeline_state": self.pipeline_state
        }


def run_ml_pipeline(
    outcome: str,
    config_path: str = None,
    biomarker_mode: str = "auto",
    dataset_type: str = "biosample",
    steps: List[str] = None,
    spark_session: SparkSession = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run the ML pipeline.
    
    Args:
        outcome: The outcome of interest (sb, eope, lope, ptb, sga)
        config_path: Optional path to config file
        biomarker_mode: Biomarker selection mode
        dataset_type: Type of dataset to use
        steps: Optional list of pipeline steps to run
        spark_session: Optional Spark session
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with pipeline results and state
    """
    # Initialize configuration
    config_manager = ConfigManager()
    config_manager.load_config(outcome=outcome, config_path=config_path)
    
    # Create and run orchestrator
    orchestrator = MLPipelineOrchestrator(config_manager, spark_session)
    return orchestrator.run_pipeline(outcome, biomarker_mode, dataset_type, steps, **kwargs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ML pipeline for risk prediction")
    parser.add_argument("outcome", help="Outcome of interest (sb, eope, lope, ptb, sga)")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--biomarker-mode", default="auto",
                       choices=["user_specified", "top_n", "auto"],
                       help="Biomarker selection mode")
    parser.add_argument("--dataset-type", default="biosample",
                       choices=["biosample", "subject", "window"],
                       help="Type of dataset to use")
    parser.add_argument("--steps", nargs="+",
                       choices=["biomarker_selection", "data_processing", "eda", "automl"],
                       help="Specific pipeline steps to run (default: all)")
    parser.add_argument("--n-biomarkers", type=int, default=100,
                       help="Number of biomarkers to select (for top_n mode)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Run pipeline
    result = run_ml_pipeline(
        args.outcome,
        config_path=args.config,
        biomarker_mode=args.biomarker_mode,
        dataset_type=args.dataset_type,
        steps=args.steps,
        n=args.n_biomarkers
    )
    
    # Print results
    print(f"\nML Pipeline Results for {args.outcome}:")
    print(f"Status: {result['status']}")
    print(f"Total Duration: {result['total_duration']:.2f} seconds")
    
    if result['status'] == 'completed':
        print(f"Steps Completed: {result['steps_run']}")
        print(f"Selected Biomarkers: {len(result.get('selected_biomarkers', []))}")
        
        # Print step-specific results
        for step, state in result['pipeline_state'].items():
            if state['completed']:
                print(f"  {step}: ✓ ({state['duration']:.2f}s)")
            else:
                print(f"  {step}: ✗")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")