#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

# -*- coding: utf-8 -*-
"""
Schema utilities for SWML validation and verb extraction.

Uses jsonschema-rs for full JSON Schema validation with type checking.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Tuple

import requests
import jsonschema_rs

try:
    import structlog
    # Ensure structlog is configured
    if not structlog.is_configured():
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
except ImportError:
    raise ImportError(
        "structlog is required. Install it with: pip install structlog"
    )

# Create a logger
logger = structlog.get_logger("schema_utils")

class SchemaUtils:
    """
    Utility class for loading and working with SWML schemas
    """

    # Remote schema URL (raw GitHub content)
    REMOTE_SCHEMA_URL = (
        "https://raw.githubusercontent.com/signalwire/docs/main/"
        "specs/swml/tsp-output/%40typespec/json-schema/SWMLObject.json"
    )

    # Cache directory and file names
    CACHE_DIR = os.path.expanduser("~/.swml")
    CACHE_SCHEMA_FILE = "schema_cache.json"
    CACHE_META_FILE = "schema_cache_meta.json"

    # HTTP fetch timeout in seconds
    REMOTE_FETCH_TIMEOUT = 5

    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize the schema utilities.

        Args:
            schema_path: Path to the schema file. If not provided, fetches
                        from remote URL with ETag caching, falling back to
                        disk cache, then bundled schema.
        """
        self.log = logger.bind(component="schema_utils")

        self.schema_path = schema_path

        if self.schema_path:
            # Explicit path provided — use it directly
            self.log.debug("using_explicit_schema_path", path=self.schema_path)
            self.schema = self.load_schema()
        else:
            # Try remote fetch with cache, then fall back to bundled
            self.schema = self._load_schema_with_remote_fallback()

        self.verbs = self._extract_verb_definitions()
        self.log.debug("schema_initialized", verb_count=len(self.verbs))
        if self.verbs:
            self.log.debug("first_verbs_extracted", verbs=list(self.verbs.keys())[:5])

        # Initialize full schema validator
        self._full_validator = None
        if self.schema:
            self._init_full_validator()

    def _init_full_validator(self) -> None:
        """Initialize the jsonschema-rs validator for full schema validation."""
        self._full_validator = jsonschema_rs.Draft202012Validator(self.schema)
        self.log.info("schema_validator_initialized",
                    validator="jsonschema-rs",
                    verbs=len(self.verbs))

    # ── Remote fetch + ETag caching ──────────────────────────────────────

    def _get_cache_schema_path(self) -> str:
        """Return the full path to the cached schema file."""
        return os.path.join(self.CACHE_DIR, self.CACHE_SCHEMA_FILE)

    def _get_cache_meta_path(self) -> str:
        """Return the full path to the cached metadata file."""
        return os.path.join(self.CACHE_DIR, self.CACHE_META_FILE)

    def _read_cache_meta(self) -> Optional[Dict[str, str]]:
        """Read cached metadata (ETag) from disk."""
        meta_path = self._get_cache_meta_path()
        try:
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    return json.load(f)
        except (IOError, OSError, json.JSONDecodeError) as e:
            self.log.debug("schema_cache_meta_read_error", error=str(e))
        return None

    def _load_cached_schema(self) -> Optional[Dict[str, Any]]:
        """Load and parse the cached schema from disk."""
        cache_path = self._get_cache_schema_path()
        try:
            if os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    schema = json.load(f)
                if isinstance(schema, dict):
                    self.log.debug("schema_cache_loaded", path=cache_path)
                    return schema
        except (IOError, OSError, json.JSONDecodeError) as e:
            self.log.debug("schema_cache_read_error", path=cache_path, error=str(e))
        return None

    def _write_cache(self, schema: Dict[str, Any], etag: Optional[str]) -> None:
        """
        Write schema and metadata to disk cache.

        Creates ~/.swml/ if it doesn't exist. Silently handles write errors
        (caching is best-effort).
        """
        try:
            os.makedirs(self.CACHE_DIR, exist_ok=True)

            with open(self._get_cache_schema_path(), "w") as f:
                json.dump(schema, f)

            meta = {"etag": etag, "cached_at": time.time()}
            with open(self._get_cache_meta_path(), "w") as f:
                json.dump(meta, f)

            self.log.debug("schema_cache_written", has_etag=bool(etag))
        except (IOError, OSError) as e:
            self.log.warning("schema_cache_write_error", error=str(e))

    def _fetch_remote_schema(self) -> Optional[Dict[str, Any]]:
        """
        Fetch schema from remote URL using HTTP conditional requests (ETag).

        Returns the schema dict on success, or None on failure (caller
        should fall through to the next source in the fallback chain).
        """
        url = os.environ.get("SWML_SCHEMA_URL", self.REMOTE_SCHEMA_URL)

        headers = {}
        meta = self._read_cache_meta()
        if meta and meta.get("etag"):
            headers["If-None-Match"] = meta["etag"]

        try:
            self.log.debug("schema_remote_fetch_start", url=url, has_etag=bool(headers))

            response = requests.get(url, headers=headers, timeout=self.REMOTE_FETCH_TIMEOUT)

            if response.status_code == 304:
                self.log.info("schema_remote_not_modified")
                cached = self._load_cached_schema()
                if cached:
                    return cached
                self.log.warning("schema_cache_missing_despite_etag")
                return None

            response.raise_for_status()

            schema = response.json()
            if not isinstance(schema, dict):
                self.log.warning("schema_remote_invalid_structure")
                return None

            new_etag = response.headers.get("ETag")
            self._write_cache(schema, new_etag)

            self.log.info("schema_loaded_from_remote", url=url, has_etag=bool(new_etag))
            return schema

        except requests.exceptions.Timeout:
            self.log.warning("schema_remote_timeout", timeout=self.REMOTE_FETCH_TIMEOUT)
            return None
        except requests.exceptions.ConnectionError:
            self.log.warning("schema_remote_connection_error")
            return None
        except requests.exceptions.RequestException as e:
            self.log.warning("schema_remote_request_error", error=str(e))
            return None
        except (json.JSONDecodeError, ValueError) as e:
            self.log.warning("schema_remote_parse_error", error=str(e))
            return None

    def _load_schema_with_remote_fallback(self) -> Dict[str, Any]:
        """
        Load schema using the fallback chain: remote → cache → bundled.
        """
        # 1. Try remote fetch (conditional GET with ETag)
        schema = self._fetch_remote_schema()
        if schema:
            return schema

        # 2. Try disk cache (from a previous successful fetch)
        schema = self._load_cached_schema()
        if schema:
            self.log.info("schema_using_disk_cache")
            return schema

        # 3. Fall back to bundled schema.json
        self.log.info("schema_falling_back_to_bundled")
        self.schema_path = self._get_default_schema_path()
        if self.schema_path:
            self.log.debug("using_default_schema_path", path=self.schema_path)
        return self.load_schema()

    # ── End remote fetch ─────────────────────────────────────────────────

    @property
    def full_validation_available(self) -> bool:
        """Check if full JSON Schema validation is available."""
        return self._full_validator is not None
        
    def _get_default_schema_path(self) -> Optional[str]:
        """
        Get the default path to the schema file using the same robust logic as SWMLService
        
        Returns:
            Path to the schema file or None if not found
        """
        # Try package resources first (most reliable after pip install)
        try:
            import importlib.resources
            try:
                # Python 3.9+
                try:
                    # Python 3.13+
                    path = importlib.resources.files("signalwire_agents").joinpath("schema.json")
                    return str(path)
                except Exception:
                    # Python 3.9-3.12
                    with importlib.resources.files("signalwire_agents").joinpath("schema.json") as path:
                        return str(path)
            except AttributeError:
                # Python 3.7-3.8
                with importlib.resources.path("signalwire_agents", "schema.json") as path:
                    return str(path)
        except (ImportError, ModuleNotFoundError):
            pass
            
        # Fall back to pkg_resources for older Python or alternative lookup
        try:
            import pkg_resources
            return pkg_resources.resource_filename("signalwire_agents", "schema.json")
        except (ImportError, ModuleNotFoundError, pkg_resources.DistributionNotFound):
            pass

        # Fall back to manual search in various locations
        import sys
        
        # Get package directory relative to this file
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Potential locations for schema.json
        potential_paths = [
            os.path.join(os.getcwd(), "schema.json"),  # Current working directory
            os.path.join(package_dir, "schema.json"),  # Package directory
            os.path.join(os.path.dirname(package_dir), "schema.json"),  # Parent of package directory
            os.path.join(sys.prefix, "schema.json"),  # Python installation directory
            os.path.join(package_dir, "data", "schema.json"),  # Data subdirectory
            os.path.join(os.path.dirname(package_dir), "data", "schema.json"),  # Parent's data subdirectory
        ]
        
        # Try to find the schema file
        for path in potential_paths:
            self.log.debug("checking_schema_path", path=path, exists=os.path.exists(path))
            if os.path.exists(path):
                self.log.debug("schema_found_at", path=path)
                return path
        
        self.log.warning("schema_not_found_in_any_location")
        return None
        
    def load_schema(self) -> Dict[str, Any]:
        """
        Load the JSON schema from the specified path
        
        Returns:
            The schema as a dictionary
        """
        if not self.schema_path:
            self.log.warning("no_schema_path_provided")
            return {}
            
        try:
            self.log.debug("loading_schema", path=self.schema_path, exists=os.path.exists(self.schema_path))
            
            if os.path.exists(self.schema_path):
                with open(self.schema_path, "r") as f:
                    schema = json.load(f)
                self.log.debug("schema_loaded_successfully", 
                              path=self.schema_path,
                              top_level_keys=len(schema.keys()) if schema else 0)
                if "$defs" in schema:
                    self.log.debug("schema_definitions_found", count=len(schema['$defs']))
                return schema
            else:
                self.log.error("schema_file_not_found", path=self.schema_path)
                return {}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.log.error("schema_loading_error", error=str(e), path=self.schema_path)
            return {}
    
    def _extract_verb_definitions(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract verb definitions from the schema
        
        Returns:
            A dictionary mapping verb names to their definitions
        """
        verbs = {}
        
        # Extract from SWMLMethod anyOf
        if "$defs" in self.schema and "SWMLMethod" in self.schema["$defs"]:
            swml_method = self.schema["$defs"]["SWMLMethod"]
            self.log.debug("swml_method_found", keys=list(swml_method.keys()))
            
            if "anyOf" in swml_method:
                self.log.debug("anyof_found", count=len(swml_method['anyOf']))
                
                for ref in swml_method["anyOf"]:
                    if "$ref" in ref:
                        # Extract the verb name from the reference
                        verb_ref = ref["$ref"]
                        verb_name = verb_ref.split("/")[-1]
                        self.log.debug("processing_verb_reference", ref=verb_ref, name=verb_name)
                        
                        # Look up the verb definition
                        if verb_name in self.schema["$defs"]:
                            verb_def = self.schema["$defs"][verb_name]
                            
                            # Extract the actual verb name (lowercase)
                            if "properties" in verb_def:
                                prop_names = list(verb_def["properties"].keys())
                                if prop_names:
                                    actual_verb = prop_names[0]
                                    verbs[actual_verb] = {
                                        "name": actual_verb,
                                        "schema_name": verb_name,
                                        "definition": verb_def
                                    }
                                    self.log.debug("verb_added", verb=actual_verb)
        else:
            self.log.warning("missing_swml_method_or_defs")
            if "$defs" in self.schema:
                self.log.debug("available_definitions", defs=list(self.schema['$defs'].keys()))
        
        return verbs
    
    def get_verb_properties(self, verb_name: str) -> Dict[str, Any]:
        """
        Get the properties for a specific verb
        
        Args:
            verb_name: The name of the verb (e.g., "ai", "answer", etc.)
            
        Returns:
            The properties for the verb or an empty dict if not found
        """
        if verb_name in self.verbs:
            verb_def = self.verbs[verb_name]["definition"]
            if "properties" in verb_def and verb_name in verb_def["properties"]:
                return verb_def["properties"][verb_name]
        return {}
    
    def get_verb_required_properties(self, verb_name: str) -> List[str]:
        """
        Get the required properties for a specific verb
        
        Args:
            verb_name: The name of the verb (e.g., "ai", "answer", etc.)
            
        Returns:
            List of required property names for the verb or an empty list if not found
        """
        if verb_name in self.verbs:
            verb_def = self.verbs[verb_name]["definition"]
            if "properties" in verb_def and verb_name in verb_def["properties"]:
                verb_props = verb_def["properties"][verb_name]
                return verb_props.get("required", [])
        return []
    
    def validate_verb(self, verb_name: str, verb_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a verb configuration against the schema.

        Performs full JSON Schema validation including type checking and nested
        object validation using jsonschema-rs.

        Args:
            verb_name: The name of the verb (e.g., "ai", "answer", etc.)
            verb_config: The configuration for the verb

        Returns:
            (is_valid, error_messages) tuple
        """
        errors = []

        # Check if the verb exists in the schema
        if verb_name not in self.verbs:
            errors.append(f"Unknown verb: {verb_name}")
            return False, errors

        # Use full validation if validator is available
        # getattr handles cases where __init__ wasn't called (e.g., tests with mocked schemas)
        if getattr(self, '_full_validator', None):
            return self._validate_verb_full(verb_name, verb_config)

        # Lightweight validation for mocked/test schemas without full document structure
        return self._validate_verb_lightweight(verb_name, verb_config)

    def _validate_verb_full(self, verb_name: str, verb_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Perform full JSON Schema validation using jsonschema-rs.

        Wraps the verb in a minimal SWML document structure for validation
        since the schema requires the full document context.

        Args:
            verb_name: The name of the verb
            verb_config: The configuration for the verb

        Returns:
            (is_valid, error_messages) tuple
        """
        # Check if schema supports document structure (has 'sections' in properties)
        # Use lightweight for partial/test schemas that don't have full document structure
        schema_props = self.schema.get('properties', {})
        if 'sections' not in schema_props:
            return self._validate_verb_lightweight(verb_name, verb_config)

        # Wrap the verb in a minimal SWML document structure
        minimal_doc = {
            "version": "1.0.0",
            "sections": {
                "main": [
                    {verb_name: verb_config}
                ]
            }
        }

        try:
            self._full_validator.validate(minimal_doc)
            return True, []
        except jsonschema_rs.ValidationError as e:
            error_msg = str(e)
            if len(error_msg) > 500:
                error_msg = error_msg[:500] + "..."
            return False, [f"Schema validation error for '{verb_name}': {error_msg}"]

    def _validate_verb_lightweight(self, verb_name: str, verb_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Perform lightweight validation (verb existence + required fields only).

        This is the fallback when jsonschema-rs is not available.

        Args:
            verb_name: The name of the verb
            verb_config: The configuration for the verb

        Returns:
            (is_valid, error_messages) tuple
        """
        errors = []

        # Get the required properties for this verb
        required_props = self.get_verb_required_properties(verb_name)

        # Check if all required properties are present
        for prop in required_props:
            if prop not in verb_config:
                errors.append(f"Missing required property '{prop}' for verb '{verb_name}'")

        return len(errors) == 0, errors
    
    def validate_document(self, document: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a complete SWML document against the schema.

        Args:
            document: The complete SWML document to validate

        Returns:
            (is_valid, error_messages) tuple
        """
        if not self._full_validator:
            return False, ["Schema validator not initialized"]

        try:
            self._full_validator.validate(document)
            return True, []
        except jsonschema_rs.ValidationError as e:
            error_msg = str(e)
            if len(error_msg) > 500:
                error_msg = error_msg[:500] + "..."
            return False, [f"Document validation error: {error_msg}"]

    def get_all_verb_names(self) -> List[str]:
        """
        Get all verb names defined in the schema

        Returns:
            List of verb names
        """
        return list(self.verbs.keys())
        
    def get_verb_parameters(self, verb_name: str) -> Dict[str, Any]:
        """
        Get the parameter definitions for a specific verb
        
        Args:
            verb_name: The name of the verb (e.g., "ai", "answer", etc.)
            
        Returns:
            Dictionary mapping parameter names to their definitions
        """
        properties = self.get_verb_properties(verb_name)
        if "properties" in properties:
            return properties["properties"]
        return {}
        
    def generate_method_signature(self, verb_name: str) -> str:
        """
        Generate a Python method signature for a verb
        
        Args:
            verb_name: The name of the verb
            
        Returns:
            A Python method signature string
        """
        # Get the verb properties
        verb_props = self.get_verb_properties(verb_name)
        
        # Get verb parameters
        verb_params = self.get_verb_parameters(verb_name)
        
        # Get required parameters
        required_params = self.get_verb_required_properties(verb_name)
        
        # Initialize method parameters
        param_list = ["self"]
        
        # Add the parameters
        for param_name, param_def in verb_params.items():
            # Check if this is a required parameter
            is_required = param_name in required_params
            
            # Determine parameter type annotation
            param_type = self._get_type_annotation(param_def)
            
            # Add default value if not required
            if is_required:
                param_list.append(f"{param_name}: {param_type}")
            else:
                param_list.append(f"{param_name}: Optional[{param_type}] = None")
        
        # Add **kwargs at the end
        param_list.append("**kwargs")
        
        # Generate method docstring
        docstring = f'"""\n        Add the {verb_name} verb to the current document\n        \n'
        
        # Add parameter documentation
        for param_name, param_def in verb_params.items():
            description = param_def.get("description", "")
            # Clean up the description for docstring
            description = description.replace('\n', ' ').strip()
            docstring += f"        Args:\n            {param_name}: {description}\n"
            
        # Add return documentation
        docstring += f'        \n        Returns:\n            True if the verb was added successfully, False otherwise\n        """\n'
        
        # Create the full method signature with docstring
        method_signature = f"def {verb_name}({', '.join(param_list)}) -> bool:\n{docstring}"
        
        return method_signature
        
    def generate_method_body(self, verb_name: str) -> str:
        """
        Generate the method body implementation for a verb
        
        Args:
            verb_name: The name of the verb
            
        Returns:
            The method body as a string
        """
        # Get verb parameters
        verb_params = self.get_verb_parameters(verb_name)
        
        body = []
        body.append("        # Prepare the configuration")
        body.append("        config = {}")
        
        # Add handling for each parameter
        for param_name in verb_params.keys():
            body.append(f"        if {param_name} is not None:")
            body.append(f"            config['{param_name}'] = {param_name}")
            
        # Add handling for kwargs
        body.append("        # Add any additional parameters from kwargs")
        body.append("        for key, value in kwargs.items():")
        body.append("            if value is not None:")
        body.append("                config[key] = value")
        
        # Add the call to add_verb
        body.append("")
        body.append(f"        # Add the {verb_name} verb")
        body.append(f"        return self.add_verb('{verb_name}', config)")
        
        return "\n".join(body)
    
    def _get_type_annotation(self, param_def: Dict[str, Any]) -> str:
        """
        Get the Python type annotation for a parameter
        
        Args:
            param_def: Parameter definition from the schema
            
        Returns:
            Python type annotation as a string
        """
        schema_type = param_def.get("type")
        
        if schema_type == "string":
            return "str"
        elif schema_type == "integer":
            return "int"
        elif schema_type == "number":
            return "float"
        elif schema_type == "boolean":
            return "bool"
        elif schema_type == "array":
            item_type = "Any"
            if "items" in param_def:
                item_def = param_def["items"]
                item_type = self._get_type_annotation(item_def)
            return f"List[{item_type}]"
        elif schema_type == "object":
            return "Dict[str, Any]"
        else:
            # Handle complex types or oneOf/anyOf
            if "anyOf" in param_def or "oneOf" in param_def:
                return "Any"
            if "$ref" in param_def:
                return "Any"  # Could be enhanced to resolve references
            return "Any" 