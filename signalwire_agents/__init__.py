"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
SignalWire AI Agents SDK
=======================

A package for building AI agents using SignalWire's AI and SWML capabilities.
"""

# Configure logging before any other imports to ensure early initialization
from .core.logging_config import configure_logging
configure_logging()

__version__ = "0.1.42"

# Import core classes for easier access
from .core.agent_base import AgentBase
from .core.contexts import ContextBuilder, Context, Step, create_simple_context
from .core.data_map import DataMap, create_simple_api_tool, create_expression_tool
from signalwire_agents.agent_server import AgentServer
from signalwire_agents.core.swml_service import SWMLService
from signalwire_agents.core.swml_builder import SWMLBuilder
from signalwire_agents.core.function_result import SwaigFunctionResult
from signalwire_agents.core.swaig_function import SWAIGFunction
from signalwire_agents.agents.bedrock import BedrockAgent

# Lazy import skills to avoid slow startup for CLI tools
# Skills are now loaded on-demand when requested
def _get_skill_registry():
    """Lazy import and return skill registry"""
    import signalwire_agents.skills
    return signalwire_agents.skills.skill_registry

# Lazy import convenience functions from the CLI (if available)
def start_agent(*args, **kwargs):
    """Start an agent (lazy import)"""
    try:
        from signalwire_agents.cli.helpers import start_agent as _start_agent
        return _start_agent(*args, **kwargs)
    except ImportError:
        raise NotImplementedError("CLI helpers not available")

def run_agent(*args, **kwargs):
    """Run an agent (lazy import)"""
    try:
        from signalwire_agents.cli.helpers import run_agent as _run_agent
        return _run_agent(*args, **kwargs)
    except ImportError:
        raise NotImplementedError("CLI helpers not available")

def list_skills(*args, **kwargs):
    """List available skills (lazy import)"""
    try:
        from signalwire_agents.cli.helpers import list_skills as _list_skills
        return _list_skills(*args, **kwargs)
    except ImportError:
        raise NotImplementedError("CLI helpers not available")

def list_skills_with_params():
    """
    Get complete schema for all available skills including parameter metadata
    
    This function returns a comprehensive schema for all available skills,
    including their metadata and parameter definitions. This is useful for
    GUI configuration tools, API documentation, or programmatic skill discovery.
    
    Returns:
        Dict[str, Dict[str, Any]]: Complete skill schema where keys are skill names
        
    Example:
        >>> schema = list_skills_with_params()
        >>> print(schema['web_search']['parameters']['api_key'])
        {
            'type': 'string',
            'description': 'Google Custom Search API key',
            'required': True,
            'hidden': True,
            'env_var': 'GOOGLE_SEARCH_API_KEY'
        }
    """
    from signalwire_agents.skills.registry import skill_registry
    return skill_registry.get_all_skills_schema()

def register_skill(skill_class):
    """
    Register a custom skill class
    
    This allows third-party code to register skill classes directly without
    requiring them to be in a specific directory structure.
    
    Args:
        skill_class: A class that inherits from SkillBase
        
    Example:
        >>> from my_custom_skills import MyWeatherSkill
        >>> register_skill(MyWeatherSkill)
        >>> # Now you can use it in agents:
        >>> agent.add_skill('my_weather')
    """
    from signalwire_agents.skills.registry import skill_registry
    return skill_registry.register_skill(skill_class)

def add_skill_directory(path):
    """
    Add a directory to search for skills
    
    This allows third-party skill collections to be registered by path.
    Skills in these directories should follow the same structure as built-in skills.
    
    Args:
        path: Path to directory containing skill subdirectories
        
    Example:
        >>> add_skill_directory('/opt/custom_skills')
        >>> # Now agent.add_skill('my_custom_skill') will search in this directory
    """
    from signalwire_agents.skills.registry import skill_registry
    return skill_registry.add_skill_directory(path)

__all__ = [
    "AgentBase",
    "AgentServer", 
    "SWMLService",
    "SWMLBuilder",
    "SwaigFunctionResult",
    "SWAIGFunction",
    "DataMap",
    "create_simple_api_tool", 
    "create_expression_tool",
    "ContextBuilder",
    "Context", 
    "Step",
    "create_simple_context",
    "start_agent",
    "run_agent",
    "list_skills",
    "list_skills_with_params",
    "register_skill",
    "add_skill_directory",
    "BedrockAgent"
]
