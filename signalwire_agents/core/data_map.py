"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
DataMap class for building SWAIG data_map configurations
"""

from typing import Dict, List, Any, Optional, Union, Pattern, Tuple
import re
from .function_result import SwaigFunctionResult


class DataMap:
    """
    Builder class for creating SWAIG data_map configurations.
    
    This provides a fluent interface for building data_map tools that execute
    on the SignalWire server without requiring webhook endpoints. Works similar
    to SwaigFunctionResult but for building data_map structures.
    
    Example usage:
        # Simple API call - output goes inside webhook
        data_map = (DataMap('get_weather')
            .purpose('Get current weather information')
            .parameter('location', 'string', 'City name', required=True)
            .webhook('GET', 'https://api.weather.com/v1/current?key=API_KEY&q=${location}')
            .output(SwaigFunctionResult('Weather in ${location}: ${response.current.condition.text}, ${response.current.temp_f}°F'))
        )
        
        # Multiple webhooks with fallback
        data_map = (DataMap('search_multi')
            .purpose('Search with fallback APIs')
            .parameter('query', 'string', 'Search query', required=True)
            .webhook('GET', 'https://api.primary.com/search?q=${query}')
            .output(SwaigFunctionResult('Primary result: ${response.title}'))
            .webhook('GET', 'https://api.fallback.com/search?q=${query}')
            .output(SwaigFunctionResult('Fallback result: ${response.title}'))
            .fallback_output(SwaigFunctionResult('Sorry, all search APIs are unavailable'))
        )
        
        # Expression-based responses (no API calls)
        data_map = (DataMap('file_control')
            .purpose('Control file playback')
            .parameter('command', 'string', 'Playback command')
            .parameter('filename', 'string', 'File to control', required=False)
            .expression('${args.command}', r'start.*', SwaigFunctionResult().add_action('start_playbook', {'file': '${args.filename}'}))
            .expression('${args.command}', r'stop.*', SwaigFunctionResult().add_action('stop_playback', True))
        )
        
        # API with array processing
        data_map = (DataMap('search_docs')
            .purpose('Search documentation')
            .parameter('query', 'string', 'Search query', required=True)
            .webhook('POST', 'https://api.docs.com/search', headers={'Authorization': 'Bearer TOKEN'})
            .body({'query': '${query}', 'limit': 3})
            .output(SwaigFunctionResult('Found: ${response.results[0].title} - ${response.results[0].summary}'))
            .foreach('${response.results}')
        )
    """
    
    def __init__(self, function_name: str):
        """
        Initialize a new DataMap builder
        
        Args:
            function_name: Name of the SWAIG function this data_map will create
        """
        self.function_name = function_name
        self._purpose = ""
        self._parameters = {}
        self._expressions = []
        self._webhooks = []
        self._output = None
        self._error_keys = []
        
    def purpose(self, description: str) -> 'DataMap':
        """
        Set the function description/purpose
        
        Args:
            description: Human-readable description of what this function does
            
        Returns:
            Self for method chaining
        """
        self._purpose = description
        return self
    
    def description(self, description: str) -> 'DataMap':
        """
        Set the function description (alias for purpose)
        
        Args:
            description: Human-readable description of what this function does
            
        Returns:
            Self for method chaining
        """
        return self.purpose(description)
    
    def parameter(self, name: str, param_type: str, description: str, 
                 required: bool = False, enum: Optional[List[str]] = None) -> 'DataMap':
        """
        Add a function parameter
        
        Args:
            name: Parameter name
            param_type: JSON schema type (string, number, boolean, array, object)
            description: Parameter description
            required: Whether parameter is required
            enum: Optional list of allowed values
            
        Returns:
            Self for method chaining
        """
        param_def = {
            "type": param_type,
            "description": description
        }
        
        if enum:
            param_def["enum"] = enum
            
        self._parameters[name] = param_def

        if required:
            if "_required" not in self._parameters:
                self._parameters["_required"] = []
            if name not in self._parameters["_required"]:
                self._parameters["_required"].append(name)
        
        return self
    
    def expression(self, test_value: str, pattern: Union[str, Pattern], output: SwaigFunctionResult, 
                   nomatch_output: Optional[SwaigFunctionResult] = None) -> 'DataMap':
        """
        Add an expression pattern for pattern-based responses
        
        Args:
            test_value: Template string to test (e.g., "${args.command}")
            pattern: Regex pattern string or compiled Pattern object to match against
            output: SwaigFunctionResult to return when pattern matches
            nomatch_output: Optional SwaigFunctionResult to return when pattern doesn't match
            
        Returns:
            Self for method chaining
        """
        if isinstance(pattern, Pattern):
            pattern_str = pattern.pattern
        else:
            pattern_str = str(pattern)
            
        expr_def = {
            "string": test_value,
            "pattern": pattern_str,
            "output": output.to_dict()
        }
        
        if nomatch_output:
            expr_def["nomatch-output"] = nomatch_output.to_dict()
            
        self._expressions.append(expr_def)
        return self
    
    def webhook(self, method: str, url: str, headers: Optional[Dict[str, str]] = None,
               form_param: Optional[str] = None, 
               input_args_as_params: bool = False,
               require_args: Optional[List[str]] = None) -> 'DataMap':
        """
        Add a webhook API call
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            url: API endpoint URL (can include ${variable} substitutions)
            headers: Optional HTTP headers
            form_param: Send JSON body as single form parameter with this name
            input_args_as_params: Merge function arguments into params
            require_args: Only execute if these arguments are present
            
        Returns:
            Self for method chaining
        """
        webhook_def = {
            "url": url,
            "method": method.upper()
        }
        
        if headers:
            webhook_def["headers"] = headers
        if form_param:
            webhook_def["form_param"] = form_param
        if input_args_as_params:
            webhook_def["input_args_as_params"] = True
        if require_args:
            webhook_def["require_args"] = require_args
            
        self._webhooks.append(webhook_def)
        return self
    
    def webhook_expressions(self, expressions: List[Dict[str, Any]]) -> 'DataMap':
        """
        Add expressions that run after the most recent webhook completes
        
        Args:
            expressions: List of expression definitions to check post-webhook
            
        Returns:
            Self for method chaining
        """
        if not self._webhooks:
            raise ValueError("Must add webhook before setting webhook expressions")
            
        self._webhooks[-1]["expressions"] = expressions
        return self
    
    def body(self, data: Dict[str, Any]) -> 'DataMap':
        """
        Set request body for the last added webhook (POST/PUT requests)
        
        Args:
            data: Request body data (can include ${variable} substitutions)
            
        Returns:
            Self for method chaining
        """
        if not self._webhooks:
            raise ValueError("Must add webhook before setting body")
            
        self._webhooks[-1]["body"] = data
        return self
    
    def params(self, data: Dict[str, Any]) -> 'DataMap':
        """
        Set request params for the last added webhook (alias for body)
        
        Args:
            data: Request params data (can include ${variable} substitutions)
            
        Returns:
            Self for method chaining
        """
        if not self._webhooks:
            raise ValueError("Must add webhook before setting params")
            
        self._webhooks[-1]["params"] = data
        return self
    
    def foreach(self, foreach_config: Dict[str, Any]) -> 'DataMap':
        """
        Process an array from the webhook response using foreach mechanism
        
        Args:
            foreach_config: Either:
                - Dict: Foreach configuration with keys:
                    - input_key: Key in API response containing the array
                    - output_key: Name for the built string variable
                    - max: Maximum number of items to process (optional)
                    - append: Template string to append for each item
            
        Returns:
            Self for method chaining
            
        Example:
            .foreach({
                "input_key": "results",
                "output_key": "formatted_results", 
                "max": 3,
                "append": "Result: ${this.title} - ${this.summary}\n"
            })
        """
        if not self._webhooks:
            raise ValueError("Must add webhook before setting foreach")
            
        if isinstance(foreach_config, dict):
            # New format - validate required keys
            required_keys = ["input_key", "output_key", "append"]
            missing_keys = [key for key in required_keys if key not in foreach_config]
            if missing_keys:
                raise ValueError(f"foreach config missing required keys: {missing_keys}")
            
            foreach_data = foreach_config
        else:
            raise ValueError("foreach_config must be a dictionary")
            
        self._webhooks[-1]["foreach"] = foreach_data
        return self
    
    def output(self, result: SwaigFunctionResult) -> 'DataMap':
        """
        Set the output result for the most recent webhook
        
        Args:
            result: SwaigFunctionResult defining the response for this webhook
            
        Returns:
            Self for method chaining
        """
        if not self._webhooks:
            raise ValueError("Must add webhook before setting output")
            
        self._webhooks[-1]["output"] = result.to_dict()
        return self
    
    def fallback_output(self, result: SwaigFunctionResult) -> 'DataMap':
        """
        Set a fallback output result at the top level (used when all webhooks fail)
        
        Args:
            result: SwaigFunctionResult defining the fallback response
            
        Returns:
            Self for method chaining
        """
        self._output = result.to_dict()
        return self

    def error_keys(self, keys: List[str]) -> 'DataMap':
        """
        Set error keys for the most recent webhook (if webhooks exist) or top-level
        
        Args:
            keys: List of JSON keys whose presence indicates an error
            
        Returns:
            Self for method chaining
        """
        if self._webhooks:
            # Add to most recent webhook
            self._webhooks[-1]["error_keys"] = keys
        else:
            # Store as top-level error keys
            self._error_keys = keys
        return self
    
    def global_error_keys(self, keys: List[str]) -> 'DataMap':
        """
        Set top-level error keys (applies to all webhooks)
        
        Args:
            keys: List of JSON keys whose presence indicates an error
            
        Returns:
            Self for method chaining
        """
        self._error_keys = keys
        return self
    
    def to_swaig_function(self) -> Dict[str, Any]:
        """
        Convert this DataMap to a SWAIG function definition
        
        Returns:
            Dictionary with function definition and data_map instead of url
        """
        # Build parameter schema
        if self._parameters:
            # Extract required params without mutating original dict
            required_params = self._parameters.get("_required", [])
            param_properties = {k: v for k, v in self._parameters.items() if k != "_required"}
            
            param_schema = {
                "type": "object",
                "properties": param_properties
            }
            if required_params:
                param_schema["required"] = required_params
        else:
            param_schema = {"type": "object", "properties": {}}
        
        # Build data_map structure
        data_map = {}
        
        # Add expressions if present
        if self._expressions:
            data_map["expressions"] = self._expressions
            
        # Add webhooks if present  
        if self._webhooks:
            data_map["webhooks"] = self._webhooks
            
        # Add output if present
        if self._output:
            data_map["output"] = self._output
            
        # Add error_keys if present
        if self._error_keys:
            data_map["error_keys"] = self._error_keys
        
        # Build final function definition with correct field names
        function_def = {
            "function": self.function_name,
            "description": self._purpose or f"Execute {self.function_name}",
            "parameters": param_schema,
            "data_map": data_map
        }
        
        return function_def


def create_simple_api_tool(name: str, url: str, response_template: str, 
                          parameters: Optional[Dict[str, Dict]] = None,
                          method: str = "GET", headers: Optional[Dict[str, str]] = None,
                          body: Optional[Dict[str, Any]] = None,
                          error_keys: Optional[List[str]] = None) -> DataMap:
    """
    Create a simple API tool with minimal configuration
    
    Args:
        name: Function name
        url: API endpoint URL
        response_template: Template for formatting the response
        parameters: Optional parameter definitions
        method: HTTP method (default: GET)
        headers: Optional HTTP headers
        body: Optional request body (for POST/PUT)
        error_keys: Optional list of error indicator keys
        
    Returns:
        Configured DataMap object
    """
    data_map = DataMap(name)
    
    # Add parameters if provided
    if parameters:
        for param_name, param_def in parameters.items():
            required = param_def.get("required", False)
            data_map.parameter(
                param_name, 
                param_def.get("type", "string"),
                param_def.get("description", f"{param_name} parameter"),
                required=required
            )
    
    # Add webhook
    data_map.webhook(method, url, headers)
    
    # Add body if provided
    if body:
        data_map.body(body)
        
    # Add error keys if provided
    if error_keys:
        data_map.error_keys(error_keys)
    
    # Set output
    data_map.output(SwaigFunctionResult(response_template))
    
    return data_map


def create_expression_tool(name: str, patterns: Dict[str, Tuple[str, SwaigFunctionResult]],
                          parameters: Optional[Dict[str, Dict]] = None) -> DataMap:
    """
    Create an expression-based tool for pattern matching responses
    
    Args:
        name: Function name
        patterns: Dictionary mapping test_values to (pattern, SwaigFunctionResult) tuples
        parameters: Optional parameter definitions
        
    Returns:
        Configured DataMap object
    """
    data_map = DataMap(name)
    
    # Add parameters if provided
    if parameters:
        for param_name, param_def in parameters.items():
            required = param_def.get("required", False)
            data_map.parameter(
                param_name,
                param_def.get("type", "string"), 
                param_def.get("description", f"{param_name} parameter"),
                required=required
            )
    
    # Add expressions with corrected signature
    for test_value, (pattern, result) in patterns.items():
        data_map.expression(test_value, pattern, result)
        
    return data_map 