"""
Metabolomics Pipeline Orchestrator

This module coordinates the execution of the complete metabolomics biomarker discovery pipeline.
It runs all pipeline steps in sequence and manages data flow between components.

Pipeline Steps:
1. Data manifest generation
2. Metadata processing
3. Stratified integration  
4. Regression analysis
5. Results comparison
"""

import logging
import time
from typing import Dict, Optional, Tuple

from pyspark.sql import SparkSession

from .data_manifest import DataManifestGenerator
from .metadata_processing import MetadataProcessor  
from .stratified_integration import StratifiedIntegrator
from .regression_analysis import RegressionAnalyzer
from .results_comparison import ResultsComparator
from ...utils.config_utils import ConfigManager
from ...utils.spark_utils import create_spark_session

logger = logging.getLogger(__name__)


class MetabolomicsOrchestrator:
    """
    Orchestrates the complete metabolomics biomarker discovery pipeline.
    
    This class coordinates all pipeline components and manages the workflow
    from raw data to biomarker discovery results.
    """
    
    def __init__(self, config_manager: ConfigManager, spark_session: SparkSession = None):
        self.config_manager = config_manager
        
        # Initialize Spark session if not provided
        if spark_session is None:
            self.spark = create_spark_session("metabolomics-pipeline", enable_delta=True)
        else:
            self.spark = spark_session
        
        # Initialize pipeline components
        self.data_manifest_generator = DataManifestGenerator(config_manager)
        self.metadata_processor = MetadataProcessor(config_manager)
        self.stratified_integrator = StratifiedIntegrator(config_manager, self.spark)
        self.regression_analyzer = RegressionAnalyzer(config_manager, self.spark)
        self.results_comparator = ResultsComparator(config_manager, self.spark)
        
        # Track pipeline state
        self.pipeline_state = {
            "data_manifest": {"completed": False, "output": None, "duration": None},
            "metadata_processing": {"completed": False, "output": None, "duration": None},
            "stratified_integration": {"completed": False, "output": None, "duration": None},
            "regression_analysis": {"completed": False, "output": None, "duration": None},
            "results_comparison": {"completed": False, "output": None, "duration": None}
        }
    
    def run_data_manifest(self, outcome: str) -> str:
        """
        Run data manifest generation step.
        
        Args:
            outcome: The outcome of interest
            
        Returns:
            Path to generated manifest
        """
        logger.info(f"Step 1/5: Running data manifest generation for outcome: {outcome}")
        start_time = time.time()
        
        try:
            manifest_path = self.data_manifest_generator.run(outcome)
            
            duration = time.time() - start_time
            self.pipeline_state["data_manifest"] = {
                "completed": True,
                "output": manifest_path,
                "duration": duration
            }
            
            logger.info(f"Data manifest generation completed in {duration:.2f} seconds")
            return manifest_path
            
        except Exception as e:
            logger.error(f"Data manifest generation failed: {str(e)}")
            raise
    
    def run_metadata_processing(self, outcome: str) -> Tuple[int, int, int]:
        """
        Run metadata processing step.
        
        Args:
            outcome: The outcome of interest
            
        Returns:
            Tuple of (cases_count, non_cases_count, controls_count)
        """
        logger.info(f"Step 2/5: Running metadata processing for outcome: {outcome}")
        start_time = time.time()
        
        try:
            cases, non_cases, controls = self.metadata_processor.run(outcome)
            
            duration = time.time() - start_time
            result_counts = (len(cases), len(non_cases), len(controls))
            
            self.pipeline_state["metadata_processing"] = {
                "completed": True,
                "output": result_counts,
                "duration": duration
            }
            
            logger.info(f"Metadata processing completed in {duration:.2f} seconds")
            logger.info(f"Generated {result_counts[0]} cases, {result_counts[1]} non-cases, {result_counts[2]} controls")
            
            return result_counts
            
        except Exception as e:
            logger.error(f"Metadata processing failed: {str(e)}")
            raise
    
    def run_stratified_integration(self, outcome: str) -> str:
        """
        Run stratified integration step.
        
        Args:
            outcome: The outcome of interest
            
        Returns:
            Path to regression input data
        """
        logger.info(f"Step 3/5: Running stratified integration for outcome: {outcome}")
        start_time = time.time()
        
        try:
            regression_input_path = self.stratified_integrator.run(outcome)
            
            duration = time.time() - start_time
            self.pipeline_state["stratified_integration"] = {
                "completed": True,
                "output": regression_input_path,
                "duration": duration
            }
            
            logger.info(f"Stratified integration completed in {duration:.2f} seconds")
            return regression_input_path
            
        except Exception as e:
            logger.error(f"Stratified integration failed: {str(e)}")
            raise
    
    def run_regression_analysis(self, outcome: str) -> Dict[str, str]:
        """
        Run regression analysis step.
        
        Args:
            outcome: The outcome of interest
            
        Returns:
            Dictionary mapping analysis names to output paths
        """
        logger.info(f"Step 4/5: Running regression analysis for outcome: {outcome}")
        start_time = time.time()
        
        try:
            result_paths = self.regression_analyzer.run(outcome)
            
            duration = time.time() - start_time
            self.pipeline_state["regression_analysis"] = {
                "completed": True,
                "output": result_paths,
                "duration": duration
            }
            
            logger.info(f"Regression analysis completed in {duration:.2f} seconds")
            logger.info(f"Generated {len(result_paths)} regression result files")
            
            return result_paths
            
        except Exception as e:
            logger.error(f"Regression analysis failed: {str(e)}")
            raise
    
    def run_results_comparison(self, outcome: str) -> Tuple[Optional[str], int, int]:
        """
        Run results comparison step.
        
        Args:
            outcome: The outcome of interest
            
        Returns:
            Tuple of (comparison_path, summary_count, notable_count)
        """
        logger.info(f"Step 5/5: Running results comparison for outcome: {outcome}")
        start_time = time.time()
        
        try:
            output_path, summary_stats, notable_biomarkers = self.results_comparator.run(outcome)
            
            duration = time.time() - start_time
            result_counts = (
                len(summary_stats) if not summary_stats.empty else 0,
                len(notable_biomarkers) if not notable_biomarkers.empty else 0
            )
            
            self.pipeline_state["results_comparison"] = {
                "completed": True,
                "output": (output_path, result_counts),
                "duration": duration
            }
            
            logger.info(f"Results comparison completed in {duration:.2f} seconds")
            if output_path:
                logger.info(f"Found {result_counts[1]} notable biomarkers")
            
            return output_path, result_counts[0], result_counts[1]
            
        except Exception as e:
            logger.error(f"Results comparison failed: {str(e)}")
            raise
    
    def run_pipeline(self, outcome: str, steps: Optional[list] = None) -> Dict:
        """
        Run the complete metabolomics pipeline or specified steps.
        
        Args:
            outcome: The outcome of interest
            steps: Optional list of steps to run. If None, runs all steps.
                  Valid steps: ['data_manifest', 'metadata_processing', 'stratified_integration', 
                               'regression_analysis', 'results_comparison']
            
        Returns:
            Dictionary with pipeline results and state
        """
        logger.info(f"Starting metabolomics pipeline for outcome: {outcome}")
        
        if steps is None:
            steps = [
                'data_manifest',
                'metadata_processing', 
                'stratified_integration',
                'regression_analysis',
                'results_comparison'
            ]
        
        pipeline_start_time = time.time()
        
        try:
            # Step 1: Data manifest generation
            if 'data_manifest' in steps:
                self.run_data_manifest(outcome)
            
            # Step 2: Metadata processing
            if 'metadata_processing' in steps:
                self.run_metadata_processing(outcome)
            
            # Step 3: Stratified integration  
            if 'stratified_integration' in steps:
                self.run_stratified_integration(outcome)
            
            # Step 4: Regression analysis
            if 'regression_analysis' in steps:
                self.run_regression_analysis(outcome)
            
            # Step 5: Results comparison
            if 'results_comparison' in steps:
                self.run_results_comparison(outcome)
            
            total_duration = time.time() - pipeline_start_time
            
            logger.info(f"Metabolomics pipeline completed successfully in {total_duration:.2f} seconds")
            
            # Return pipeline summary
            return {
                "outcome": outcome,
                "status": "completed",
                "total_duration": total_duration,
                "steps_run": steps,
                "pipeline_state": self.pipeline_state
            }
            
        except Exception as e:
            total_duration = time.time() - pipeline_start_time
            logger.error(f"Pipeline failed after {total_duration:.2f} seconds: {str(e)}")
            
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
    
    def get_pipeline_summary(self) -> Dict:
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


def run_metabolomics_pipeline(
    outcome: str, 
    config_path: str = None,
    steps: list = None,
    spark_session: SparkSession = None
) -> Dict:
    """
    Convenience function to run the metabolomics pipeline.
    
    Args:
        outcome: The outcome of interest (sb, eope, lope, ptb, sga)
        config_path: Optional path to config file
        steps: Optional list of pipeline steps to run
        spark_session: Optional Spark session
        
    Returns:
        Dictionary with pipeline results and state
    """
    # Initialize configuration
    config_manager = ConfigManager()
    config_manager.load_config(outcome=outcome, config_path=config_path)
    
    # Create and run orchestrator
    orchestrator = MetabolomicsOrchestrator(config_manager, spark_session)
    return orchestrator.run_pipeline(outcome, steps)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run metabolomics biomarker discovery pipeline")
    parser.add_argument("outcome", help="Outcome of interest (sb, eope, lope, ptb, sga)")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--steps", nargs="+", 
                       choices=['data_manifest', 'metadata_processing', 'stratified_integration', 
                               'regression_analysis', 'results_comparison'],
                       help="Specific pipeline steps to run (default: all)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Run pipeline
    result = run_metabolomics_pipeline(args.outcome, args.config, args.steps)
    
    # Print results
    print(f"\nMetabolomics Pipeline Results for {args.outcome}:")
    print(f"Status: {result['status']}")
    print(f"Total Duration: {result['total_duration']:.2f} seconds")
    
    if result['status'] == 'completed':
        print(f"Steps Completed: {result['steps_run']}")
        
        # Print step-specific results
        for step, state in result['pipeline_state'].items():
            if state['completed']:
                print(f"  {step}: ✓ ({state['duration']:.2f}s)")
            else:
                print(f"  {step}: ✗")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")