"""
Configuration utilities for metabolomic analysis pipeline.
Handles Hydra configuration management and path resolution.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf


class ConfigManager:
    """
    Manages configuration for the metabolomic analysis pipeline.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.logger = logging.getLogger(__name__)
        self.config_dir = config_dir or self._get_default_config_dir()
        self.config: Optional[DictConfig] = None
        
    def _get_default_config_dir(self) -> str:
        """Get default configuration directory."""
        # Assume config directory is relative to project root
        project_root = Path(__file__).parent.parent.parent
        return str(project_root / "config")
    
    def load_config(
        self,
        config_name: str = "metabolomics_config",
        outcome: Optional[str] = None,
        overrides: Optional[list] = None,
    ) -> DictConfig:
        """
        Load configuration with optional overrides.
        
        Args:
            config_name: Name of the configuration file
            outcome: Outcome to override (sb, ptb, sga, etc.)
            overrides: List of configuration overrides
            
        Returns:
            Loaded configuration
        """
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()
        
        # Prepare overrides
        override_list = overrides or []
        if outcome:
            override_list.append(f"outcome={outcome}")
        
        try:
            # Initialize Hydra with config directory
            initialize_config_dir(config_dir=self.config_dir, version_base="1.1")
            
            # Compose configuration
            self.config = compose(
                config_name=config_name,
                overrides=override_list,
            )
            
            self.logger.info(f"Loaded configuration: {config_name}")
            if outcome:
                self.logger.info(f"Outcome: {outcome}")
                
            return self.config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def get_config(self) -> DictConfig:
        """
        Get current configuration.
        
        Returns:
            Current configuration
            
        Raises:
            RuntimeError: If configuration not loaded
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self.config
    
    def get_path(self, path_key: str) -> str:
        """
        Get resolved path from configuration.
        
        Args:
            path_key: Dot-separated path key (e.g., "paths.metadata.dec_2024_metadata")
            
        Returns:
            Resolved path string
        """
        config = self.get_config()
        try:
            return OmegaConf.select(config, path_key)
        except Exception as e:
            self.logger.error(f"Failed to get path {path_key}: {e}")
            raise
    
    def get_site_config(self, site_name: str) -> DictConfig:
        """
        Get configuration for a specific site.
        
        Args:
            site_name: Name of the site (e.g., "amanhi_bangladesh")
            
        Returns:
            Site configuration
        """
        config = self.get_config()
        try:
            return config.sites[site_name]
        except KeyError:
            self.logger.error(f"Site {site_name} not found in configuration")
            raise
    
    def get_manifest_paths(self) -> Dict[str, str]:
        """
        Get all cleaned manifest paths.
        
        Returns:
            Dictionary mapping site names to manifest paths
        """
        config = self.get_config()
        return OmegaConf.to_container(config.paths.manifest_cleaned, resolve=True)
    
    def get_intensity_paths(self) -> Dict[str, str]:
        """
        Get all site intensity data paths.
        
        Returns:
            Dictionary mapping site names to intensity data paths
        """
        config = self.get_config()
        intensity_paths = {}
        
        for site_name, site_config in config.sites.items():
            intensity_paths[site_name.upper()] = site_config.intensity_data
            
        return intensity_paths
    
    def get_regression_columns(self) -> list:
        """
        Get list of columns needed for regression analysis.
        
        Returns:
            List of column names
        """
        config = self.get_config()
        return config.processing.regression_columns
    
    def get_binary_columns_to_impute(self) -> list:
        """
        Get list of binary columns to impute.
        
        Returns:
            List of column names
        """
        config = self.get_config()
        return config.processing.binary_columns_to_impute
    
    def get_site_mappings(self) -> Dict[str, str]:
        """
        Get site name mappings.
        
        Returns:
            Dictionary mapping short site names to full names
        """
        config = self.get_config()
        return OmegaConf.to_container(config.processing.site_mappings, resolve=True)
    
    def get_spark_config(self) -> Dict[str, Any]:
        """
        Get Spark configuration.
        
        Returns:
            Spark configuration dictionary
        """
        config = self.get_config()
        return OmegaConf.to_container(config.spark.config, resolve=True)
    
    def save_config(self, path: str) -> None:
        """
        Save current configuration to file.
        
        Args:
            path: Path to save configuration
        """
        config = self.get_config()
        with open(path, 'w') as f:
            OmegaConf.save(config, f)
        self.logger.info(f"Configuration saved to {path}")
    
    def print_config(self) -> None:
        """Print current configuration."""
        config = self.get_config()
        print(OmegaConf.to_yaml(config))


def get_project_root() -> Path:
    """
    Get project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent.parent


def create_output_directory(path: str) -> None:
    """
    Create output directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def validate_paths(config: DictConfig) -> bool:
    """
    Validate that required paths exist or can be created.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if paths are valid, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    # Check that output directories can be created
    try:
        output_paths = [
            config.paths.file_manifest.base,
            config.paths.combined_manifest.base,
            config.paths.analysis.regression_analysis.pooled_analysis.base,
        ]
        
        for path in output_paths:
            if path.startswith("s3://"):
                # Skip S3 path validation for now
                continue
            create_output_directory(path)
            
        return True
        
    except Exception as e:
        logger.error(f"Path validation failed: {e}")
        return False