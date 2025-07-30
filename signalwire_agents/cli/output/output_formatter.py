#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Display agent/tools and format results
"""

import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from signalwire_agents.core.agent_base import AgentBase
    from signalwire_agents.core.function_result import SwaigFunctionResult


def display_agent_tools(agent: 'AgentBase', verbose: bool = False) -> None:
    """
    Display the available SWAIG functions for an agent
    
    Args:
        agent: The agent instance
        verbose: Whether to show verbose details
    """
    print("\nAvailable SWAIG functions:")
    # Try to access functions from the tool registry
    functions = {}
    if hasattr(agent, '_tool_registry') and hasattr(agent._tool_registry, '_swaig_functions'):
        functions = agent._tool_registry._swaig_functions
    elif hasattr(agent, '_swaig_functions'):
        functions = agent._swaig_functions
    
    if functions:
        for name, func in functions.items():
            if isinstance(func, dict):
                # DataMap function
                description = func.get('description', 'DataMap function (serverless)')
                print(f"  {name} - {description}")
                
                # Show parameters for DataMap functions
                if 'parameters' in func and func['parameters']:
                    params = func['parameters']
                    # Handle both formats: direct properties dict or full schema
                    if 'properties' in params:
                        properties = params['properties']
                        required_fields = params.get('required', [])
                    else:
                        properties = params
                        required_fields = []
                    
                    if properties:
                        print(f"    Parameters:")
                        for param_name, param_def in properties.items():
                            param_type = param_def.get('type', 'unknown')
                            param_desc = param_def.get('description', 'No description')
                            is_required = param_name in required_fields
                            required_marker = " (required)" if is_required else ""
                            
                            # Build constraint details
                            constraints = []
                            
                            # Enum values
                            if 'enum' in param_def:
                                constraints.append(f"options: {', '.join(map(str, param_def['enum']))}")
                            
                            # Numeric constraints
                            if 'minimum' in param_def:
                                constraints.append(f"min: {param_def['minimum']}")
                            if 'maximum' in param_def:
                                constraints.append(f"max: {param_def['maximum']}")
                            if 'exclusiveMinimum' in param_def:
                                constraints.append(f"min (exclusive): {param_def['exclusiveMinimum']}")
                            if 'exclusiveMaximum' in param_def:
                                constraints.append(f"max (exclusive): {param_def['exclusiveMaximum']}")
                            if 'multipleOf' in param_def:
                                constraints.append(f"multiple of: {param_def['multipleOf']}")
                            
                            # String constraints
                            if 'minLength' in param_def:
                                constraints.append(f"min length: {param_def['minLength']}")
                            if 'maxLength' in param_def:
                                constraints.append(f"max length: {param_def['maxLength']}")
                            if 'pattern' in param_def:
                                constraints.append(f"pattern: {param_def['pattern']}")
                            if 'format' in param_def:
                                constraints.append(f"format: {param_def['format']}")
                            
                            # Array constraints
                            if param_type == 'array':
                                if 'minItems' in param_def:
                                    constraints.append(f"min items: {param_def['minItems']}")
                                if 'maxItems' in param_def:
                                    constraints.append(f"max items: {param_def['maxItems']}")
                                if 'uniqueItems' in param_def and param_def['uniqueItems']:
                                    constraints.append("unique items")
                                if 'items' in param_def and 'type' in param_def['items']:
                                    constraints.append(f"item type: {param_def['items']['type']}")
                            
                            # Default value
                            if 'default' in param_def:
                                constraints.append(f"default: {param_def['default']}")
                            
                            # Format the type with constraints
                            if constraints:
                                param_type_full = f"{param_type} [{', '.join(constraints)}]"
                            else:
                                param_type_full = param_type
                            
                            print(f"      {param_name} ({param_type_full}){required_marker}: {param_desc}")
                    else:
                        print(f"    Parameters: None")
                else:
                    print(f"    Parameters: None")
                    
                if verbose:
                    print(f"    Config: {json.dumps(func, indent=6)}")
            else:
                # Regular SWAIG function
                func_type = ""
                if hasattr(func, 'webhook_url') and func.webhook_url and func.is_external:
                    func_type = " (EXTERNAL webhook)"
                elif hasattr(func, 'webhook_url') and func.webhook_url:
                    func_type = " (webhook)"
                else:
                    func_type = " (LOCAL webhook)"
                
                print(f"  {name} - {func.description}{func_type}")
                
                # Show external URL if applicable
                if hasattr(func, 'webhook_url') and func.webhook_url and func.is_external:
                    print(f"    External URL: {func.webhook_url}")
                
                # Show parameters
                if hasattr(func, 'parameters') and func.parameters:
                    params = func.parameters
                    # Handle both formats: direct properties dict or full schema
                    if 'properties' in params:
                        properties = params['properties']
                        required_fields = params.get('required', [])
                    else:
                        properties = params
                        required_fields = []
                    
                    if properties:
                        print(f"    Parameters:")
                        for param_name, param_def in properties.items():
                            param_type = param_def.get('type', 'unknown')
                            param_desc = param_def.get('description', 'No description')
                            is_required = param_name in required_fields
                            required_marker = " (required)" if is_required else ""
                            
                            # Build constraint details
                            constraints = []
                            
                            # Enum values
                            if 'enum' in param_def:
                                constraints.append(f"options: {', '.join(map(str, param_def['enum']))}")
                            
                            # Numeric constraints
                            if 'minimum' in param_def:
                                constraints.append(f"min: {param_def['minimum']}")
                            if 'maximum' in param_def:
                                constraints.append(f"max: {param_def['maximum']}")
                            if 'exclusiveMinimum' in param_def:
                                constraints.append(f"min (exclusive): {param_def['exclusiveMinimum']}")
                            if 'exclusiveMaximum' in param_def:
                                constraints.append(f"max (exclusive): {param_def['exclusiveMaximum']}")
                            if 'multipleOf' in param_def:
                                constraints.append(f"multiple of: {param_def['multipleOf']}")
                            
                            # String constraints
                            if 'minLength' in param_def:
                                constraints.append(f"min length: {param_def['minLength']}")
                            if 'maxLength' in param_def:
                                constraints.append(f"max length: {param_def['maxLength']}")
                            if 'pattern' in param_def:
                                constraints.append(f"pattern: {param_def['pattern']}")
                            if 'format' in param_def:
                                constraints.append(f"format: {param_def['format']}")
                            
                            # Array constraints
                            if param_type == 'array':
                                if 'minItems' in param_def:
                                    constraints.append(f"min items: {param_def['minItems']}")
                                if 'maxItems' in param_def:
                                    constraints.append(f"max items: {param_def['maxItems']}")
                                if 'uniqueItems' in param_def and param_def['uniqueItems']:
                                    constraints.append("unique items")
                                if 'items' in param_def and 'type' in param_def['items']:
                                    constraints.append(f"item type: {param_def['items']['type']}")
                            
                            # Default value
                            if 'default' in param_def:
                                constraints.append(f"default: {param_def['default']}")
                            
                            # Format the type with constraints
                            if constraints:
                                param_type_full = f"{param_type} [{', '.join(constraints)}]"
                            else:
                                param_type_full = param_type
                            
                            print(f"      {param_name} ({param_type_full}){required_marker}: {param_desc}")
                    else:
                        print(f"    Parameters: None")
                else:
                    print(f"    Parameters: None")
                    
                if verbose:
                    print(f"    Function object: {func}")
    else:
        print("  No SWAIG functions registered")


def format_result(result: Any) -> str:
    """
    Format the result of a SWAIG function call for display
    
    Args:
        result: The result from the SWAIG function
        
    Returns:
        Formatted string representation
    """
    # Import here to avoid circular imports
    from signalwire_agents.core.function_result import SwaigFunctionResult
    
    if isinstance(result, SwaigFunctionResult):
        output = [f"SwaigFunctionResult: {result.response}"]
        
        # Show actions if present
        if hasattr(result, 'action') and result.action:
            output.append("\nActions:")
            for action in result.action:
                output.append(json.dumps(action, indent=2))
        
        # Show post_process flag if set
        if hasattr(result, 'post_process') and result.post_process:
            output.append(f"\nPost-process: {result.post_process}")
            
        return "\n".join(output)
    elif isinstance(result, dict):
        if 'response' in result:
            return f"Response: {result['response']}"
        else:
            return f"Dict: {json.dumps(result, indent=2)}"
    elif isinstance(result, str):
        return f"String: {result}"
    else:
        return f"Other ({type(result).__name__}): {result}"