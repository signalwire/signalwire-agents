"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""Tool decorator functionality."""

from functools import wraps
from typing import Callable, Optional, Dict, Any

from signalwire_agents.core.logging_config import get_logger

logger = get_logger(__name__)


class ToolDecorator:
    """Handles tool decoration logic."""
    
    @staticmethod
    def create_instance_decorator(registry):
        """
        Create instance tool decorator.

        Args:
            registry: ToolRegistry instance to register with

        Returns:
            Decorator function
        """
        def decorator(name=None, **kwargs):
            """
            Decorator for defining SWAIG tools in a class.

            Used as:

            @agent.tool(name="example_function", parameters={...})
            def example_function(self, param1):
                # ...
            """
            def inner_decorator(func):
                nonlocal name
                if name is None:
                    name = func.__name__

                parameters = kwargs.pop("parameters", {})
                description = kwargs.pop("description", None)
                secure = kwargs.pop("secure", True)
                fillers = kwargs.pop("fillers", None)
                webhook_url = kwargs.pop("webhook_url", None)
                required = kwargs.pop("required", None)

                handler = func
                is_typed = False

                # If parameters not explicitly provided, try type inference
                if not parameters:
                    from signalwire_agents.core.agent.tools.type_inference import (
                        infer_schema,
                        create_typed_handler_wrapper,
                    )
                    inferred_params, inferred_required, inferred_desc, is_typed, has_raw_data = infer_schema(func)
                    if is_typed:
                        parameters = inferred_params
                        if inferred_required and required is None:
                            required = inferred_required
                        if inferred_desc and description is None:
                            description = inferred_desc
                        handler = create_typed_handler_wrapper(func, has_raw_data)

                # Fall back to docstring or default description
                if description is None:
                    description = func.__doc__ or f"Function {name}"

                registry.define_tool(
                    name=name,
                    description=description,
                    parameters=parameters,
                    handler=handler,
                    secure=secure,
                    fillers=fillers,
                    webhook_url=webhook_url,
                    required=required,
                    is_typed_handler=is_typed,
                    **kwargs  # Pass through any additional swaig_fields
                )
                return func
            return inner_decorator
        return decorator
    
    @classmethod
    def create_class_decorator(cls):
        """
        Create class tool decorator.
        
        Returns:
            Decorator function
        """
        def tool(name=None, **kwargs):
            """
            Class method decorator for defining SWAIG tools.
            
            Used as:
            
            @AgentBase.tool(name="example_function", parameters={...})
            def example_function(self, param1):
                # ...
            """
            def decorator(func):
                # Mark the function as a tool
                func._is_tool = True
                func._tool_name = name if name else func.__name__
                func._tool_params = kwargs
                
                # Return the original function
                return func
            return decorator
        return tool