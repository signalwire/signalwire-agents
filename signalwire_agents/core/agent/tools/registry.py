"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""Tool registration and management."""

from typing import Dict, Any, Optional, List, Callable, Union
import inspect
import logging

from signalwire_agents.core.swaig_function import SWAIGFunction

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Manages SWAIG function registration."""
    
    def __init__(self, agent):
        """
        Initialize ToolRegistry with reference to parent agent.
        
        Args:
            agent: Parent AgentBase instance
        """
        self.agent = agent
        self._swaig_functions = {}
        self._class_decorated_tools = []
    
    def define_tool(
        self, 
        name: str, 
        description: str, 
        parameters: Dict[str, Any], 
        handler: Callable,
        secure: bool = True,
        fillers: Optional[Dict[str, List[str]]] = None,
        wait_file: Optional[str] = None,
        wait_file_loops: Optional[int] = None,
        webhook_url: Optional[str] = None,
        required: Optional[List[str]] = None,
        **swaig_fields
    ) -> None:
        """
        Define a SWAIG function that the AI can call.
        
        Args:
            name: Function name (must be unique)
            description: Function description for the AI
            parameters: JSON Schema of parameters
            handler: Function to call when invoked
            secure: Whether to require token validation
            fillers: Optional dict mapping language codes to arrays of filler phrases (deprecated)
            wait_file: Optional URL to audio file to play while function executes
            wait_file_loops: Optional number of times to loop the wait_file
            webhook_url: Optional external webhook URL to use instead of local handling
            required: Optional list of required parameter names
            **swaig_fields: Additional SWAIG fields to include in function definition
            
        Raises:
            ValueError: If tool name already exists
        """
        if name in self._swaig_functions:
            raise ValueError(f"Tool with name '{name}' already exists")
            
        self._swaig_functions[name] = SWAIGFunction(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            secure=secure,
            fillers=fillers,
            wait_file=wait_file,
            wait_file_loops=wait_file_loops,
            webhook_url=webhook_url,
            required=required,
            **swaig_fields
        )
        
        logger.debug(f"Defined tool: {name}")
    
    def register_swaig_function(self, function_dict: Dict[str, Any]) -> None:
        """
        Register a raw SWAIG function dictionary (e.g., from DataMap.to_swaig_function()).
        
        Args:
            function_dict: Complete SWAIG function definition dictionary
            
        Raises:
            ValueError: If function name missing or already exists
        """
        function_name = function_dict.get('function')
        if not function_name:
            raise ValueError("Function dictionary must contain 'function' field with the function name")
            
        if function_name in self._swaig_functions:
            raise ValueError(f"Tool with name '{function_name}' already exists")
        
        # Store the raw function dictionary for data_map tools
        # These don't have handlers since they execute on SignalWire's server
        self._swaig_functions[function_name] = function_dict
        
        # Debug logging using the module logger with proper format
        logger.debug(f"Registered SWAIG function in registry: {function_name} (registry_id={id(self)}, agent_id={id(self.agent) if hasattr(self, 'agent') else None}, total_functions={len(self._swaig_functions)})")
        
        logger.debug(f"Registered SWAIG function: {function_name}")
    
    def register_class_decorated_tools(self) -> None:
        """
        Register tools defined with @AgentBase.tool class decorator.
        
        This method scans the class for methods decorated with @AgentBase.tool
        and registers them automatically.
        """
        # Get the class of this instance
        cls = self.agent.__class__
        
        # Loop through all attributes in the class
        for name in dir(cls):
            # Get the attribute
            attr = getattr(cls, name)
            
            # Check if it's a method decorated with @AgentBase.tool
            if inspect.ismethod(attr) or inspect.isfunction(attr):
                if hasattr(attr, "_is_tool") and getattr(attr, "_is_tool", False):
                    # Extract tool information
                    tool_name = getattr(attr, "_tool_name", name)
                    tool_params = getattr(attr, "_tool_params", {})
                    
                    # Extract known parameters and pass through the rest as swaig_fields
                    tool_params_copy = tool_params.copy()
                    description = tool_params_copy.pop("description", attr.__doc__ or f"Function {tool_name}")
                    parameters = tool_params_copy.pop("parameters", {})
                    secure = tool_params_copy.pop("secure", True)
                    fillers = tool_params_copy.pop("fillers", None)
                    wait_file = tool_params_copy.pop("wait_file", None)
                    wait_file_loops = tool_params_copy.pop("wait_file_loops", None)
                    webhook_url = tool_params_copy.pop("webhook_url", None)
                    required = tool_params_copy.pop("required", None)
                    
                    # Register the tool with any remaining params as swaig_fields
                    self.define_tool(
                        name=tool_name,
                        description=description,
                        parameters=parameters,
                        handler=attr.__get__(self.agent, cls),  # Bind the method to this instance
                        secure=secure,
                        fillers=fillers,
                        wait_file=wait_file,
                        wait_file_loops=wait_file_loops,
                        webhook_url=webhook_url,
                        required=required,
                        **tool_params_copy  # Pass through any additional swaig_fields
                    )
                    
                    logger.debug(f"Registered class-decorated tool: {tool_name}")
    
    def get_function(self, name: str) -> Optional[Union[SWAIGFunction, Dict[str, Any]]]:
        """
        Get a registered function by name.
        
        Args:
            name: Function name
            
        Returns:
            SWAIGFunction instance or raw function dict, or None if not found
        """
        return self._swaig_functions.get(name)
    
    def get_all_functions(self) -> Dict[str, Union[SWAIGFunction, Dict[str, Any]]]:
        """
        Get all registered functions.
        
        Returns:
            Dictionary of function name to function object/dict
        """
        return self._swaig_functions.copy()
    
    def has_function(self, name: str) -> bool:
        """
        Check if a function is registered.
        
        Args:
            name: Function name
            
        Returns:
            True if function exists, False otherwise
        """
        return name in self._swaig_functions
    
    def remove_function(self, name: str) -> bool:
        """
        Remove a registered function.
        
        Args:
            name: Function name
            
        Returns:
            True if removed, False if not found
        """
        if name in self._swaig_functions:
            del self._swaig_functions[name]
            logger.debug(f"Removed function: {name}")
            return True
        return False