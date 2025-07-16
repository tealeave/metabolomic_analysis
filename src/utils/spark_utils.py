"""
Spark utilities for metabolomic analysis pipeline.
Provides PySpark session management and common utilities.
"""

import logging
import os
from typing import Dict, Optional, Any

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from delta import configure_spark_with_delta_pip


def create_spark_session(
    app_name: str = "metabolomic-analysis-pipeline",
    master: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    enable_delta: bool = True,
    enable_hive: bool = False,
) -> SparkSession:
    """
    Create a Spark session optimized for metabolomic analysis.
    
    Args:
        app_name: Name of the Spark application
        master: Spark master URL (None for auto-detection)
        config: Additional Spark configuration
        enable_delta: Whether to enable Delta Lake
        enable_hive: Whether to enable Hive support
        
    Returns:
        Configured SparkSession
    """
    logger = logging.getLogger(__name__)
    
    # Base configuration
    spark_config = {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.adaptive.skewJoin.enabled": "true",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.sql.adaptive.advisoryPartitionSizeInBytes": "256MB",
        "spark.sql.adaptive.localShuffleReader.enabled": "true",
    }
    
    # Delta Lake configuration
    if enable_delta:
        spark_config.update({
            "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
            "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        })
    
    # Merge with user-provided config
    if config:
        spark_config.update(config)
    
    # Create Spark configuration
    conf = SparkConf().setAppName(app_name)
    
    # Set master if provided
    if master:
        conf.setMaster(master)
    elif not os.getenv("SPARK_MASTER"):
        # Default to local mode if not in cluster
        conf.setMaster("local[*]")
    
    # Apply configuration
    for key, value in spark_config.items():
        conf.set(key, value)
    
    # Create session builder
    builder = SparkSession.builder.config(conf=conf)
    
    # Enable Hive if requested
    if enable_hive:
        builder = builder.enableHiveSupport()
    
    # Configure Delta Lake if enabled
    if enable_delta:
        builder = configure_spark_with_delta_pip(builder)
    
    # Build session
    spark = builder.getOrCreate()
    
    # Set log level
    spark.sparkContext.setLogLevel("WARN")
    
    logger.info(f"Created Spark session: {app_name}")
    logger.info(f"Spark version: {spark.version}")
    logger.info(f"Spark master: {spark.sparkContext.master}")
    
    return spark


def get_or_create_spark_session(
    app_name: str = "metabolomic-analysis-pipeline",
    config: Optional[Dict[str, Any]] = None,
) -> SparkSession:
    """
    Get existing Spark session or create a new one.
    
    Args:
        app_name: Name of the Spark application
        config: Additional Spark configuration
        
    Returns:
        SparkSession
    """
    try:
        # Try to get existing session
        spark = SparkSession.getActiveSession()
        if spark is not None:
            return spark
    except Exception:
        pass
    
    # Create new session
    return create_spark_session(app_name=app_name, config=config)


def stop_spark_session(spark: SparkSession) -> None:
    """
    Stop Spark session and clean up resources.
    
    Args:
        spark: SparkSession to stop
    """
    logger = logging.getLogger(__name__)
    
    try:
        spark.stop()
        logger.info("Spark session stopped")
    except Exception as e:
        logger.warning(f"Error stopping Spark session: {e}")


def configure_logging(level: str = "INFO") -> None:
    """
    Configure logging for the pipeline.
    
    Args:
        level: Logging level (DEBUG, INFO, WARN, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/pipeline.log", mode="a"),
        ],
    )


def optimize_spark_for_metabolomics(spark: SparkSession) -> None:
    """
    Apply metabolomics-specific optimizations to Spark session.
    
    Args:
        spark: SparkSession to optimize
    """
    # Set configuration for large datasets
    spark.conf.set("spark.sql.adaptive.maxShuffledHashJoinLocalMapThreshold", "256MB")
    spark.conf.set("spark.sql.adaptive.advisoryPartitionSizeInBytes", "256MB")
    spark.conf.set("spark.sql.adaptive.nonEmptyPartitionRatioForBroadcastJoin", "0.2")
    
    # Optimize for wide transformations common in metabolomics
    spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
    spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB")
    
    # Cache optimization
    spark.conf.set("spark.sql.adaptive.coalescePartitions.parallelismFirst", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.minPartitionNum", "1")


def validate_spark_environment() -> bool:
    """
    Validate that Spark environment is properly configured.
    
    Returns:
        True if environment is valid, False otherwise
    """
    try:
        spark = get_or_create_spark_session()
        
        # Test basic functionality
        test_df = spark.range(10)
        count = test_df.count()
        
        # Test Delta Lake if available
        try:
            spark.sql("CREATE TABLE IF NOT EXISTS test_delta (id INT) USING DELTA")
            spark.sql("DROP TABLE IF EXISTS test_delta")
        except Exception:
            logging.warning("Delta Lake not available")
        
        return count == 10
    except Exception as e:
        logging.error(f"Spark environment validation failed: {e}")
        return False