#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

# -*- coding: utf-8 -*-
"""
Schema utilities for SWML validation and verb extraction
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

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
    
    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize the schema utilities
        
        Args:
            schema_path: Path to the schema file
        """
        self.log = logger.bind(component="schema_utils")
        
        self.schema_path = schema_path
        if not self.schema_path:
            self.schema_path = self._get_default_schema_path()
            self.log.debug("using_default_schema_path", path=self.schema_path)
        
        self.schema = self.load_schema()
        self.verbs = self._extract_verb_definitions()
        self.log.debug("schema_initialized", verb_count=len(self.verbs))
        if self.verbs:
            self.log.debug("first_verbs_extracted", verbs=list(self.verbs.keys())[:5])
        
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

        if "$defs" not in self.schema or "SWMLMethod" not in self.schema["$defs"]:
            self.log.warning("missing_swml_method_or_defs")
            if "$defs" in self.schema:
                self.log.debug("available_definitions", defs=list(self.schema['$defs'].keys()))
            return verbs

        swml_method = self.schema["$defs"]["SWMLMethod"]
        self.log.debug("swml_method_found", keys=list(swml_method.keys()))

        # Extract verb references from anyOf, oneOf, or allOf
        verb_refs = []
        for keyword in ("anyOf", "oneOf", "allOf"):
            if keyword in swml_method:
                self.log.debug(f"{keyword}_found", count=len(swml_method[keyword]))
                verb_refs.extend(swml_method[keyword])
                break  # Only one composition keyword should be present at this level

        for ref in verb_refs:
            if "$ref" not in ref:
                continue

            # Extract the verb name from the reference
            verb_ref = ref["$ref"]
            verb_name = verb_ref.split("/")[-1]
            self.log.debug("processing_verb_reference", ref=verb_ref, name=verb_name)

            # Look up the verb definition
            if verb_name not in self.schema["$defs"]:
                continue

            verb_def = self.schema["$defs"][verb_name]

            # Extract the actual verb name (lowercase) from properties
            if "properties" not in verb_def:
                continue

            prop_names = list(verb_def["properties"].keys())
            if not prop_names:
                continue

            actual_verb = prop_names[0]
            verbs[actual_verb] = {
                "name": actual_verb,
                "schema_name": verb_name,
                "definition": verb_def
            }
            self.log.debug("verb_added", verb=actual_verb)

        return verbs
    
    def get_verb_properties(self, verb_name: str) -> Dict[str, Any]:
        """
        Get the properties for a specific verb

        Args:
            verb_name: The name of the verb (e.g., "ai", "answer", etc.)

        Returns:
            The properties for the verb or an empty dict if not found.
            For verbs using oneOf/anyOf/allOf, returns merged properties from all branches.
        """
        if verb_name not in self.verbs:
            return {}

        verb_def = self.verbs[verb_name]["definition"]
        if "properties" not in verb_def or verb_name not in verb_def["properties"]:
            return {}

        verb_props = verb_def["properties"][verb_name]

        # Handle composition keywords - merge properties from referenced schemas
        for keyword in ("allOf", "anyOf", "oneOf"):
            if keyword in verb_props:
                merged = self._merge_properties_from_composition(verb_props[keyword], keyword)
                # Preserve description and other metadata from original
                result = {k: v for k, v in verb_props.items() if k != keyword}
                result.update(merged)
                return result

        return verb_props

    def get_verb_required_properties(self, verb_name: str) -> List[str]:
        """
        Get the required properties for a specific verb

        Args:
            verb_name: The name of the verb (e.g., "ai", "answer", etc.)

        Returns:
            List of required property names for the verb or an empty list if not found
        """
        if verb_name not in self.verbs:
            return []

        verb_def = self.verbs[verb_name]["definition"]
        if "properties" not in verb_def or verb_name not in verb_def["properties"]:
            return []

        verb_props = verb_def["properties"][verb_name]

        # Check for direct required field
        if "required" in verb_props:
            return verb_props["required"]

        # Handle composition keywords - for allOf, merge required from all branches
        # For oneOf/anyOf, only return commonly required fields (intersection)
        for keyword in ("allOf", "anyOf", "oneOf"):
            if keyword in verb_props:
                merged = self._merge_properties_from_composition(verb_props[keyword], keyword)
                return merged.get("required", [])

        return []
    
    def validate_verb(self, verb_name: str, verb_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a verb configuration against the schema
        
        Args:
            verb_name: The name of the verb (e.g., "ai", "answer", etc.)
            verb_config: The configuration for the verb
            
        Returns:
            (is_valid, error_messages) tuple
        """
        # Simple validation for now - can be enhanced with more complete JSON Schema validation
        errors = []
        
        # Check if the verb exists in the schema
        if verb_name not in self.verbs:
            errors.append(f"Unknown verb: {verb_name}")
            return False, errors
            
        # Get the required properties for this verb
        required_props = self.get_verb_required_properties(verb_name)
        
        # Check if all required properties are present
        for prop in required_props:
            if prop not in verb_config:
                errors.append(f"Missing required property '{prop}' for verb '{verb_name}'")
                
        # Return validation result
        return len(errors) == 0, errors
    
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
    
    def _resolve_ref(self, ref: str) -> Dict[str, Any]:
        """
        Resolve a $ref reference to its definition

        Args:
            ref: The $ref string (e.g., "#/$defs/SomeDef")

        Returns:
            The resolved definition or empty dict if not found
        """
        if not ref.startswith("#/$defs/"):
            self.log.debug("unsupported_ref_format", ref=ref)
            return {}

        def_name = ref.split("/")[-1]
        if "$defs" in self.schema and def_name in self.schema["$defs"]:
            return self.schema["$defs"][def_name]

        self.log.debug("ref_not_found", ref=ref, def_name=def_name)
        return {}

    def _merge_properties_from_composition(self, composition: List[Dict[str, Any]], keyword: str) -> Dict[str, Any]:
        """
        Merge properties from a composition keyword (allOf, anyOf, oneOf)

        Args:
            composition: List of schema references/definitions
            keyword: The composition keyword ("allOf", "anyOf", "oneOf")

        Returns:
            Merged properties dictionary. For oneOf/anyOf:
            - Properties are merged for discovery (all possible properties)
            - If same property has different types, marked with _conflicting_types=True
            - Required fields are the INTERSECTION (required in ALL branches)
        """
        merged_properties = {}
        all_required_sets = []  # Track required fields from each branch

        for item in composition:
            resolved = item
            if "$ref" in item:
                resolved = self._resolve_ref(item["$ref"])

            # Get properties from the resolved schema
            if "properties" in resolved:
                for prop_name, prop_def in resolved["properties"].items():
                    if prop_name in merged_properties:
                        # Property already exists - check for type conflict
                        existing = merged_properties[prop_name]
                        existing_type = existing.get("type")
                        new_type = prop_def.get("type")

                        # If types differ (and both are defined), mark as conflicting
                        if existing_type and new_type and existing_type != new_type:
                            merged_properties[prop_name] = {
                                **existing,
                                "_conflicting_types": True
                            }
                        # Otherwise keep existing (first definition wins for type)
                    else:
                        merged_properties[prop_name] = prop_def

            # Collect required fields from this branch
            if "required" in resolved:
                all_required_sets.append(set(resolved["required"]))

        # Compute final required fields based on keyword
        final_required = []
        if keyword == "allOf":
            # For allOf: UNION of all required fields (all must be satisfied)
            for req_set in all_required_sets:
                final_required.extend(req_set)
            final_required = list(set(final_required))  # deduplicate
        elif all_required_sets:
            # For oneOf/anyOf: INTERSECTION of required fields
            # (only fields required in ALL branches are truly required)
            common_required = all_required_sets[0]
            for req_set in all_required_sets[1:]:
                common_required = common_required & req_set
            final_required = list(common_required)

        result = {}
        if merged_properties:
            result["properties"] = merged_properties
        if final_required:
            result["required"] = final_required

        return result

    def _get_type_annotation(self, param_def: Dict[str, Any]) -> str:
        """
        Get the Python type annotation for a parameter

        Args:
            param_def: Parameter definition from the schema

        Returns:
            Python type annotation as a string
        """
        # If this property has conflicting types from oneOf/anyOf merge, use Any
        if param_def.get("_conflicting_types"):
            return "Any"

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
            # Handle complex types or allOf/oneOf/anyOf
            if "allOf" in param_def or "anyOf" in param_def or "oneOf" in param_def:
                return "Any"
            if "$ref" in param_def:
                return "Any"  # Could be enhanced to resolve references
            return "Any" 