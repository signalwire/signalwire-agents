"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import os
import re
import json
from typing import Any, Dict, List, Optional, Union
from signalwire_agents.core.logging_config import get_logger

logger = get_logger("config_loader")


class ConfigLoader:
    """
    Configuration loader with environment variable substitution.
    
    Supports ${VAR|default} syntax for referencing environment variables
    within JSON configuration files. This provides a clean pattern for
    configuration across all SignalWire services.
    """
    
    def __init__(self, config_paths: Optional[List[str]] = None):
        """
        Initialize config loader.
        
        Args:
            config_paths: Optional list of config file paths to check.
                         If not provided, uses default search paths.
        """
        self.config_paths = config_paths or self._get_default_paths()
        self._config = None
        self._config_file = None
        self._load_config()
    
    def _get_default_paths(self) -> List[str]:
        """Get default configuration file search paths."""
        return [
            "config.json",
            "agent_config.json",
            "swml_config.json",
            ".swml/config.json",
            os.path.expanduser("~/.swml/config.json"),
            "/etc/swml/config.json"
        ]
    
    def _load_config(self) -> None:
        """Load configuration from the first available config file."""
        for path in self.config_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        self._config = json.load(f)
                        self._config_file = path
                        logger.info("config_loaded", path=path)
                        break
                except Exception as e:
                    logger.error("config_load_error", path=path, error=str(e))
    
    def has_config(self) -> bool:
        """Check if a configuration was loaded."""
        return self._config is not None
    
    def get_config_file(self) -> Optional[str]:
        """Get the path of the loaded config file."""
        return self._config_file
    
    def get_config(self) -> Dict[str, Any]:
        """Get the raw configuration (before substitution)."""
        return self._config or {}
    
    def substitute_vars(self, value: Any, max_depth: int = 10) -> Any:
        """
        Recursively substitute environment variables in configuration values.

        Supports ${VAR|default} syntax where:
        - VAR is the environment variable name
        - default is the fallback value if VAR is not set

        Args:
            value: The value to process (can be string, dict, list, etc.)
            max_depth: Maximum recursion depth to prevent infinite loops

        Returns:
            The value with all environment variables substituted
        """
        if max_depth <= 0:
            raise ValueError("Maximum variable substitution depth exceeded")

        if isinstance(value, str):
            # Pattern to match ${VAR} or ${VAR|default}
            pattern = r'\$\{([^}|]+)(?:\|([^}]*))?\}'
            
            def replacer(match):
                var_name = match.group(1)
                default = match.group(2) if match.group(2) is not None else ''
                return os.environ.get(var_name, default)
            
            # Substitute all variables
            result = re.sub(pattern, replacer, value)
            
            # Try to parse as JSON to get proper types
            if result.lower() in ('true', 'false'):
                return result.lower() == 'true'
            elif result.isdigit():
                return int(result)
            elif result.replace('.', '', 1).isdigit():
                return float(result)
            else:
                return result
                
        elif isinstance(value, dict):
            # Recursively process dictionary
            return {k: self.substitute_vars(v, max_depth - 1) for k, v in value.items()}

        elif isinstance(value, list):
            # Recursively process list
            return [self.substitute_vars(item, max_depth - 1) for item in value]
            
        else:
            # Return other types as-is
            return value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation path.
        
        Args:
            key_path: Dot-separated path (e.g., "security.ssl_enabled")
            default: Default value if path not found
            
        Returns:
            The configuration value with variables substituted
        """
        if not self._config:
            return default
            
        # Navigate through the config using the dot path
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        # Substitute variables before returning
        return self.substitute_vars(value)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.
        
        Args:
            section: The section name (e.g., "security", "server")
            
        Returns:
            The configuration section with all variables substituted
        """
        if not self._config or section not in self._config:
            return {}
            
        return self.substitute_vars(self._config[section])
    
    def merge_with_env(self, env_prefix: str = "SWML_") -> Dict[str, Any]:
        """
        Merge configuration with environment variables.
        
        Config file takes precedence over environment variables,
        but config can reference env vars via substitution.
        
        Args:
            env_prefix: Prefix for environment variables to consider
            
        Returns:
            Merged configuration dictionary
        """
        # Start with substituted config
        result = self.substitute_vars(self._config) if self._config else {}
        
        # Only add env vars that aren't already in config
        # This preserves config file precedence
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                # Convert SWML_SSL_ENABLED to ssl_enabled
                config_key = key[len(env_prefix):].lower()
                
                # Only set if not already in config
                if not self._has_nested_key(result, config_key):
                    self._set_nested_key(result, config_key, value)
        
        return result
    
    def _has_nested_key(self, data: Dict, key_path: str) -> bool:
        """Check if a nested key exists in dictionary."""
        keys = key_path.split('_')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return False
        return True
    
    def _set_nested_key(self, data: Dict, key_path: str, value: Any) -> None:
        """Set a value in dictionary using underscore-separated path."""
        keys = key_path.split('_')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
            
        current[keys[-1]] = value
    
    @staticmethod
    def find_config_file(service_name: Optional[str] = None, 
                        additional_paths: Optional[List[str]] = None) -> Optional[str]:
        """
        Static method to find a config file for a service.
        
        Args:
            service_name: Optional service name for service-specific config
            additional_paths: Additional paths to check
            
        Returns:
            Path to the first config file found, or None
        """
        paths = []
        
        # Service-specific config
        if service_name:
            paths.extend([
                f"{service_name}_config.json",
                f".swml/{service_name}_config.json"
            ])
        
        # Additional paths
        if additional_paths:
            paths.extend(additional_paths)
        
        # Default paths
        paths.extend([
            "config.json",
            "agent_config.json",
            ".swml/config.json",
            os.path.expanduser("~/.swml/config.json"),
            "/etc/swml/config.json"
        ])
        
        for path in paths:
            if os.path.exists(path):
                return path
                
        return None