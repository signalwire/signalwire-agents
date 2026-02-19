"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, TYPE_CHECKING, Optional

from signalwire_agents.core.logging_config import get_logger

if TYPE_CHECKING:
    from signalwire_agents.core.agent_base import AgentBase
    from signalwire_agents.core.function_result import SwaigFunctionResult

class SkillBase(ABC):
    """Abstract base class for all agent skills"""
    
    # Subclasses must define these
    SKILL_NAME: str = None           # Required: unique identifier
    SKILL_DESCRIPTION: str = None    # Required: human-readable description
    SKILL_VERSION: str = "1.0.0"     # Semantic version
    REQUIRED_PACKAGES: List[str] = [] # Python packages needed
    REQUIRED_ENV_VARS: List[str] = [] # Environment variables needed
    
    # Multiple instance support
    SUPPORTS_MULTIPLE_INSTANCES: bool = False  # Set to True to allow multiple instances
    
    def __init__(self, agent: 'AgentBase', params: Optional[Dict[str, Any]] = None):
        if self.SKILL_NAME is None:
            raise ValueError(f"{self.__class__.__name__} must define SKILL_NAME")
        if self.SKILL_DESCRIPTION is None:
            raise ValueError(f"{self.__class__.__name__} must define SKILL_DESCRIPTION")
            
        self.agent = agent
        self.params = params or {}
        self.logger = get_logger(f"signalwire_agents.skills.{self.SKILL_NAME}")
        
        # Extract swaig_fields from params for merging into tool definitions
        self.swaig_fields = self.params.pop('swaig_fields', {})
        
    @abstractmethod
    def setup(self) -> bool:
        """
        Setup the skill (validate env vars, initialize APIs, etc.)
        Returns True if setup successful, False otherwise
        """
        pass
        
    @abstractmethod
    def register_tools(self) -> None:
        """Register SWAIG tools with the agent"""
        pass
        
    def define_tool(self, **kwargs) -> None:
        """
        Wrapper method that automatically includes swaig_fields when defining tools.
        
        This method delegates to self.agent.define_tool() but automatically merges
        any swaig_fields configured for this skill. Skills should use this method
        instead of calling self.agent.define_tool() directly.
        
        Args:
            **kwargs: All arguments supported by agent.define_tool()
                     (name, description, parameters, handler, etc.)
        """
        # Merge swaig_fields with any explicitly passed fields
        # Explicit fields take precedence over swaig_fields
        merged_kwargs = dict(self.swaig_fields)
        merged_kwargs.update(kwargs)
        
        # Call the agent's define_tool with merged arguments
        return self.agent.define_tool(**merged_kwargs)
        

        
    def get_hints(self) -> List[str]:
        """Return speech recognition hints for this skill"""
        return []
        
    def get_global_data(self) -> Dict[str, Any]:
        """Return data to add to agent's global context"""
        return {}
        
    def get_prompt_sections(self) -> List[Dict[str, Any]]:
        """Return prompt sections to add to agent.
        Returns empty list if skip_prompt is set to True in params."""
        if self.params.get("skip_prompt", False):
            return []
        return self._get_prompt_sections()

    def _get_prompt_sections(self) -> List[Dict[str, Any]]:
        """Override this in subclasses to provide prompt sections."""
        return []
        
    def cleanup(self) -> None:
        """Cleanup when skill is removed or agent shuts down"""
        pass
        
    def validate_env_vars(self) -> bool:
        """Check if all required environment variables are set"""
        import os
        missing = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing:
            self.logger.error(f"Missing required environment variables: {missing}")
            return False
        return True
        
    def validate_packages(self) -> bool:
        """Check if all required packages are available"""
        import importlib
        missing = []
        for package in self.REQUIRED_PACKAGES:
            try:
                importlib.import_module(package)
            except ImportError:
                missing.append(package)
        if missing:
            self.logger.error(f"Missing required packages: {missing}")
            return False
        return True
        
    def get_instance_key(self) -> str:
        """
        Get the key used to track this skill instance
        
        For skills that support multiple instances (SUPPORTS_MULTIPLE_INSTANCES = True),
        this method can be overridden to provide a unique key for each instance.
        
        Default implementation:
        - If SUPPORTS_MULTIPLE_INSTANCES is False: returns SKILL_NAME
        - If SUPPORTS_MULTIPLE_INSTANCES is True: returns SKILL_NAME + "_" + tool_name
          (where tool_name comes from params['tool_name'] or defaults to the skill name)
        
        Returns:
            str: Unique key for this skill instance
        """
        if not self.SUPPORTS_MULTIPLE_INSTANCES:
            return self.SKILL_NAME
            
        # For multi-instance skills, create key from skill name + tool name
        tool_name = self.params.get('tool_name', self.SKILL_NAME)
        return f"{self.SKILL_NAME}_{tool_name}"

    def _get_skill_namespace(self) -> str:
        """
        Get the namespaced key for this skill instance's global_data.

        Uses the 'prefix' param if available, otherwise falls back to
        the instance key. This ensures multiple skill instances can
        store state in global_data without collisions.

        Returns:
            str: Namespace key like "skill:<prefix>" or "skill:<instance_key>"
        """
        prefix = self.params.get('prefix')
        if prefix:
            return f"skill:{prefix}"
        return f"skill:{self.get_instance_key()}"

    def get_skill_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Read this skill instance's namespaced data from raw_data global_data.

        Args:
            raw_data: The raw_data dict passed to SWAIG function handlers,
                     expected to contain a 'global_data' key.

        Returns:
            dict: The skill's namespaced state, or empty dict if not found.
        """
        namespace = self._get_skill_namespace()
        global_data = raw_data.get("global_data", {})
        return global_data.get(namespace, {})

    def update_skill_data(self, result: 'SwaigFunctionResult', data: Dict[str, Any]) -> 'SwaigFunctionResult':
        """
        Write this skill instance's namespaced data into a SwaigFunctionResult.

        Wraps the data under the skill's namespace key and calls
        result.update_global_data().

        Args:
            result: The SwaigFunctionResult to add the global_data update to.
            data: The skill state dict to store under the namespace.

        Returns:
            SwaigFunctionResult: The result, for method chaining.
        """
        namespace = self._get_skill_namespace()
        result.update_global_data({namespace: data})
        return result

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get the parameter schema for this skill
        
        This method returns metadata about all parameters the skill accepts,
        including their types, descriptions, default values, and whether they
        are required or should be hidden (e.g., API keys).
        
        The base implementation provides common parameters available to all skills.
        Subclasses should override this method and merge their specific parameters
        with the base schema.
        
        Returns:
            Dict[str, Dict[str, Any]]: Parameter schema where keys are parameter names
            and values are dictionaries containing:
                - type: Parameter type ("string", "integer", "number", "boolean", "object", "array")
                - description: Human-readable description
                - default: Default value if not provided (optional)
                - required: Whether the parameter is required (default: False)
                - hidden: Whether to hide this field in UIs (for secrets/keys)
                - env_var: Environment variable that can provide this value (optional)
                - enum: List of allowed values (optional)
                - min/max: Minimum/maximum values for numeric types (optional)
        
        Example:
            {
                "tool_name": {
                    "type": "string",
                    "description": "Name for the tool when using multiple instances",
                    "default": "my_skill",
                    "required": False
                },
                "api_key": {
                    "type": "string",
                    "description": "API key for the service",
                    "required": True,
                    "hidden": True,
                    "env_var": "MY_API_KEY"
                }
            }
        """
        schema = {}
        
        # Add swaig_fields parameter (available to all skills)
        schema["swaig_fields"] = {
            "type": "object",
            "description": "Additional SWAIG function metadata to merge into tool definitions",
            "default": {},
            "required": False
        }

        # Add skip_prompt flag (available to all skills)
        schema["skip_prompt"] = {
            "type": "boolean",
            "description": "If true, the skill will not inject its default prompt section into the POM",
            "default": False,
            "required": False
        }
        
        # Add tool_name for multi-instance skills
        if cls.SUPPORTS_MULTIPLE_INSTANCES:
            schema["tool_name"] = {
                "type": "string",
                "description": "Custom name for this skill instance (for multiple instances)",
                "default": cls.SKILL_NAME,
                "required": False
            }
        
        return schema 