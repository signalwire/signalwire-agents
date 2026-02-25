#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
DataMap function execution and template expansion
"""

import re
import json
import requests
from typing import Dict, Any
from ..config import HTTP_REQUEST_TIMEOUT


def simple_template_expand(template: str, data: Dict[str, Any]) -> str:
    """
    Simple template expansion for DataMap testing
    Supports both ${key} and %{key} syntax with nested object access and array indexing
    
    Args:
        template: Template string with ${} or %{} variables
        data: Data dictionary for expansion
        
    Returns:
        Expanded string
    """
    if not template:
        return ""
        
    result = template
    
    # Handle both ${variable.path} and %{variable.path} syntax
    patterns = [
        r'\$\{([^}]+)\}',  # ${variable} syntax
        r'%\{([^}]+)\}'    # %{variable} syntax
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, result):
            var_path = match.group(1)
            
            # Handle array indexing syntax like "array[0].joke"
            if '[' in var_path and ']' in var_path:
                # Split path with array indexing
                parts = []
                current_part = ""
                i = 0
                while i < len(var_path):
                    if var_path[i] == '[':
                        if current_part:
                            parts.append(current_part)
                            current_part = ""
                        # Find the closing bracket
                        j = i + 1
                        while j < len(var_path) and var_path[j] != ']':
                            j += 1
                        if j < len(var_path):
                            index = var_path[i+1:j]
                            parts.append(f"[{index}]")
                            i = j + 1
                            if i < len(var_path) and var_path[i] == '.':
                                i += 1  # Skip the dot after ]
                        else:
                            current_part += var_path[i]
                            i += 1
                    elif var_path[i] == '.':
                        if current_part:
                            parts.append(current_part)
                            current_part = ""
                        i += 1
                    else:
                        current_part += var_path[i]
                        i += 1
                
                if current_part:
                    parts.append(current_part)
                    
                # Navigate through the data structure
                value = data
                try:
                    for part in parts:
                        if part.startswith('[') and part.endswith(']'):
                            # Array index
                            index = int(part[1:-1])
                            if isinstance(value, list) and 0 <= index < len(value):
                                value = value[index]
                            else:
                                value = f"<MISSING:{var_path}>"
                                break
                        else:
                            # Object property
                            if isinstance(value, dict) and part in value:
                                value = value[part]
                            else:
                                value = f"<MISSING:{var_path}>"
                                break
                except (ValueError, TypeError, IndexError):
                    value = f"<MISSING:{var_path}>"
                    
            else:
                # Regular nested object access (no array indexing)
                path_parts = var_path.split('.')
                value = data
                for part in path_parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        value = f"<MISSING:{var_path}>"
                        break
            
            # Replace the variable with its value
            result = result.replace(match.group(0), str(value))
    
    return result


def execute_datamap_function(datamap_config: Dict[str, Any], args: Dict[str, Any], 
                           verbose: bool = False) -> Dict[str, Any]:
    """
    Execute a DataMap function following the actual DataMap processing pipeline:
    1. Expressions (pattern matching)
    2. Webhooks (try each sequentially until one succeeds)
    3. Foreach (within successful webhook)
    4. Output (from successful webhook)
    5. Fallback output (if all webhooks fail)
    
    Args:
        datamap_config: DataMap configuration dictionary
        args: Function arguments
        verbose: Enable verbose output
        
    Returns:
        Function result (should be string or dict with 'response' key)
    """
    if verbose:
        print("=== DataMap Function Execution ===")
        print(f"Config: {json.dumps(datamap_config, indent=2)}")
        print(f"Args: {json.dumps(args, indent=2)}")
    
    # Extract the actual data_map configuration
    # DataMap configs have the structure: {"function": "...", "data_map": {...}}
    actual_datamap = datamap_config.get("data_map", datamap_config)
    
    if verbose:
        print(f"Extracted data_map: {json.dumps(actual_datamap, indent=2)}")
    
    # Initialize context with function arguments
    context = {"args": args}
    context.update(args)  # Also make args available at top level for backward compatibility
    
    if verbose:
        print(f"Initial context: {json.dumps(context, indent=2)}")
    
    # Step 1: Process expressions first (pattern matching)
    if "expressions" in actual_datamap:
        if verbose:
            print("\n--- Processing Expressions ---")
        for expr in actual_datamap["expressions"]:
            # Simple expression evaluation - in real implementation this would be more sophisticated
            if "pattern" in expr and "output" in expr:
                # For testing, we'll just match simple strings
                pattern = expr["pattern"]
                if pattern in str(args):
                    if verbose:
                        print(f"Expression matched: {pattern}")
                    result = simple_template_expand(str(expr["output"]), context)
                    if verbose:
                        print(f"Expression result: {result}")
                    return result
    
    # Step 2: Process webhooks sequentially
    if "webhooks" in actual_datamap:
        if verbose:
            print("\n--- Processing Webhooks ---")
        
        for i, webhook in enumerate(actual_datamap["webhooks"]):
            if verbose:
                print(f"\n=== Webhook {i+1}/{len(actual_datamap['webhooks'])} ===")
            
            url = webhook.get("url", "")
            method = webhook.get("method", "POST").upper()
            headers = webhook.get("headers", {})
            
            # Expand template variables in URL and headers
            url = simple_template_expand(url, context)
            expanded_headers = {}
            for key, value in headers.items():
                expanded_headers[key] = simple_template_expand(str(value), context)
            
            if verbose:
                print(f"Making {method} request to: {url}")
                print(f"Headers: {json.dumps(expanded_headers, indent=2)}")
            
            # Prepare request data
            request_data = None
            if method in ["POST", "PUT", "PATCH"]:
                # Check for 'params' (SignalWire style) or 'data' (generic style) or 'body'
                if "params" in webhook:
                    # Expand template variables in params
                    expanded_params = {}
                    for key, value in webhook["params"].items():
                        expanded_params[key] = simple_template_expand(str(value), context)
                    request_data = json.dumps(expanded_params)
                elif "body" in webhook:
                    # Expand template variables in body
                    if isinstance(webhook["body"], str):
                        request_data = simple_template_expand(webhook["body"], context)
                    else:
                        expanded_body = {}
                        for key, value in webhook["body"].items():
                            expanded_body[key] = simple_template_expand(str(value), context)
                        request_data = json.dumps(expanded_body)
                elif "data" in webhook:
                    # Expand template variables in data
                    if isinstance(webhook["data"], str):
                        request_data = simple_template_expand(webhook["data"], context)
                    else:
                        request_data = json.dumps(webhook["data"])
                
                if verbose and request_data:
                    print(f"Request data: {request_data}")
            
            webhook_failed = False
            response_data = None
            
            try:
                # SSRF protection
                from signalwire_agents.utils.url_validator import validate_url
                if not validate_url(url):
                    raise ValueError(f"URL rejected by SSRF protection: {url}")

                # Make the HTTP request
                if method == "GET":
                    response = requests.get(url, headers=expanded_headers, timeout=HTTP_REQUEST_TIMEOUT)
                elif method == "POST":
                    response = requests.post(url, data=request_data, headers=expanded_headers, timeout=HTTP_REQUEST_TIMEOUT)
                elif method == "PUT":
                    response = requests.put(url, data=request_data, headers=expanded_headers, timeout=HTTP_REQUEST_TIMEOUT)
                elif method == "PATCH":
                    response = requests.patch(url, data=request_data, headers=expanded_headers, timeout=HTTP_REQUEST_TIMEOUT)
                elif method == "DELETE":
                    response = requests.delete(url, headers=expanded_headers, timeout=HTTP_REQUEST_TIMEOUT)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                if verbose:
                    print(f"Response status: {response.status_code}")
                    print(f"Response headers: {dict(response.headers)}")
                
                # Parse response
                try:
                    response_data = response.json()
                except json.JSONDecodeError:
                    response_data = {"text": response.text, "status_code": response.status_code}
                    # Add parse_error like server does
                    response_data["parse_error"] = True
                    response_data["raw_response"] = response.text
                
                if verbose:
                    print(f"Response data: {json.dumps(response_data, indent=2)}")
                
                # Check for webhook failure following server logic
                
                # 1. Check HTTP status code (fix the server bug - should be OR not AND)
                if response.status_code < 200 or response.status_code > 299:
                    webhook_failed = True
                    if verbose:
                        print(f"Webhook failed: HTTP status {response.status_code} outside 200-299 range")
                
                # 2. Check for explicit error keys (parse_error, protocol_error)
                if not webhook_failed:
                    explicit_error_keys = ["parse_error", "protocol_error"]
                    for error_key in explicit_error_keys:
                        if error_key in response_data and response_data[error_key]:
                            webhook_failed = True
                            if verbose:
                                print(f"Webhook failed: Found explicit error key '{error_key}' = {response_data[error_key]}")
                            break
                
                # 3. Check for custom error_keys from webhook config
                if not webhook_failed and "error_keys" in webhook:
                    error_keys = webhook["error_keys"]
                    if isinstance(error_keys, str):
                        error_keys = [error_keys]  # Convert single string to list
                    elif not isinstance(error_keys, list):
                        error_keys = []
                    
                    for error_key in error_keys:
                        if error_key in response_data and response_data[error_key]:
                            webhook_failed = True
                            if verbose:
                                print(f"Webhook failed: Found custom error key '{error_key}' = {response_data[error_key]}")
                            break
                
            except Exception as e:
                webhook_failed = True
                if verbose:
                    print(f"Webhook failed: HTTP request exception: {e}")
                # Create error response like server does
                response_data = {
                    "protocol_error": True,
                    "error": str(e)
                }
            
            # If webhook succeeded, process its output
            if not webhook_failed:
                if verbose:
                    print(f"Webhook {i+1} succeeded!")
                
                # Add response data to context
                webhook_context = context.copy()
                
                # Handle different response types
                if isinstance(response_data, list):
                    # For array responses, use ${array[0].field} syntax
                    webhook_context["array"] = response_data
                    if verbose:
                        print(f"Array response: {len(response_data)} items")
                else:
                    # For object responses, use ${response.field} syntax
                    webhook_context["response"] = response_data
                    if verbose:
                        print("Object response")
                
                # Step 3: Process webhook-level foreach (if present)
                if "foreach" in webhook:
                    foreach_config = webhook["foreach"]
                    if verbose:
                        print(f"\n--- Processing Webhook Foreach ---")
                        print(f"Foreach config: {json.dumps(foreach_config, indent=2)}")
                    
                    input_key = foreach_config.get("input_key", "data")
                    output_key = foreach_config.get("output_key", "result")
                    max_items = foreach_config.get("max", 100)
                    append_template = foreach_config.get("append", "${this.value}")
                    
                    # Look for the input data in the response
                    input_data = None
                    if input_key in response_data and isinstance(response_data[input_key], list):
                        input_data = response_data[input_key]
                        if verbose:
                            print(f"Found array data in response.{input_key}: {len(input_data)} items")
                    
                    if input_data:
                        result_parts = []
                        items_to_process = input_data[:max_items]
                        
                        for item in items_to_process:
                            if isinstance(item, dict):
                                # For objects, make properties available as ${this.property}
                                item_context = {"this": item}
                                expanded = simple_template_expand(append_template, item_context)
                            else:
                                # For non-dict items, make them available as ${this.value}
                                item_context = {"this": {"value": item}}
                                expanded = simple_template_expand(append_template, item_context)
                            result_parts.append(expanded)
                        
                        # Store the concatenated result
                        foreach_result = "".join(result_parts)
                        webhook_context[output_key] = foreach_result
                        
                        if verbose:
                            print(f"Processed {len(items_to_process)} items")
                            print(f"Foreach result ({output_key}): {foreach_result[:200]}{'...' if len(foreach_result) > 200 else ''}")
                    else:
                        if verbose:
                            print(f"No array data found for foreach input_key: {input_key}")
                
                # Step 4: Process webhook-level output (this is the final result)
                if "output" in webhook:
                    webhook_output = webhook["output"]
                    if verbose:
                        print(f"\n--- Processing Webhook Output ---")
                        print(f"Output template: {json.dumps(webhook_output, indent=2)}")
                    
                    if isinstance(webhook_output, dict):
                        # Process each key-value pair in the output
                        final_result = {}
                        for key, template in webhook_output.items():
                            expanded_value = simple_template_expand(str(template), webhook_context)
                            final_result[key] = expanded_value
                            if verbose:
                                print(f"Set {key} = {expanded_value}")
                    else:
                        # Single output value (string template)
                        final_result = simple_template_expand(str(webhook_output), webhook_context)
                        if verbose:
                            print(f"Final result = {final_result}")
                    
                    if verbose:
                        print(f"\n--- Webhook {i+1} Final Result ---")
                        print(f"Result: {json.dumps(final_result, indent=2) if isinstance(final_result, dict) else final_result}")
                    
                    return final_result
                
                else:
                    # No output template defined, return the response data
                    if verbose:
                        print("No output template defined, returning response data")
                    return response_data
            
            else:
                # This webhook failed, try next webhook
                if verbose:
                    print(f"Webhook {i+1} failed, trying next webhook...")
                continue
    
    # Step 5: All webhooks failed, use fallback output if available
    if "output" in actual_datamap:
        if verbose:
            print(f"\n--- Using DataMap Fallback Output ---")
        datamap_output = actual_datamap["output"]
        if verbose:
            print(f"Fallback output template: {json.dumps(datamap_output, indent=2)}")
        
        if isinstance(datamap_output, dict):
            # Process each key-value pair in the fallback output
            final_result = {}
            for key, template in datamap_output.items():
                expanded_value = simple_template_expand(str(template), context)
                final_result[key] = expanded_value
                if verbose:
                    print(f"Fallback: Set {key} = {expanded_value}")
            result = final_result
        else:
            # Single fallback output value
            result = simple_template_expand(str(datamap_output), context)
            if verbose:
                print(f"Fallback result = {result}")
        
        if verbose:
            print(f"\n--- DataMap Fallback Final Result ---")
            print(f"Result: {json.dumps(result, indent=2) if isinstance(result, dict) else result}")
        
        return result
    
    # No fallback defined, return generic error
    error_result = {"error": "All webhooks failed and no fallback output defined", "status": "failed"}
    if verbose:
        print(f"\n--- DataMap Error Result ---")
        print(f"Result: {json.dumps(error_result, indent=2)}")
    
    return error_result