"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, TYPE_CHECKING, Optional
import logging

if TYPE_CHECKING:
    from signalwire_agents.core.agent_base import AgentBase

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
        self.logger = logging.getLogger(f"skill.{self.SKILL_NAME}")
        
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
        

        
    def get_hints(self) -> List[str]:
        """Return speech recognition hints for this skill"""
        return []
        
    def get_global_data(self) -> Dict[str, Any]:
        """Return data to add to agent's global context"""
        return {}
        
    def get_prompt_sections(self) -> List[Dict[str, Any]]:
        """Return prompt sections to add to agent"""
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