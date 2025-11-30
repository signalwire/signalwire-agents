"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import os
import json
import sys
from typing import Optional, Dict, Any

from signalwire_agents.core.logging_config import get_execution_mode
from signalwire_agents.core.function_result import SwaigFunctionResult


class ServerlessMixin:
    """
    Mixin class containing all serverless/cloud platform methods for AgentBase
    """
    
    def handle_serverless_request(self, event=None, context=None, mode=None):
        """
        Handle serverless environment requests (CGI, Lambda, Cloud Functions)
        
        Args:
            event: Serverless event object (Lambda, Cloud Functions)
            context: Serverless context object (Lambda, Cloud Functions)
            mode: Override execution mode (from force_mode in run())
            
        Returns:
            Response appropriate for the serverless platform
        """
        if mode is None:
            mode = get_execution_mode()
        
        try:
            if mode == 'cgi':
                # Check authentication in CGI mode
                if not self._check_cgi_auth():
                    return self._send_cgi_auth_challenge()
                
                path_info = os.getenv('PATH_INFO', '').strip('/')
                if not path_info:
                    return self._render_swml()
                else:
                    # Parse CGI request for SWAIG function call
                    args = {}
                    call_id = None
                    raw_data = None
                    
                    # Try to parse POST data from stdin for CGI
                    import sys
                    content_length = os.getenv('CONTENT_LENGTH')
                    if content_length and content_length.isdigit():
                        try:
                            post_data = sys.stdin.read(int(content_length))
                            if post_data:
                                raw_data = json.loads(post_data)
                                call_id = raw_data.get("call_id")
                                
                                # Extract arguments like the FastAPI handler does
                                if "argument" in raw_data and isinstance(raw_data["argument"], dict):
                                    if "parsed" in raw_data["argument"] and isinstance(raw_data["argument"]["parsed"], list) and raw_data["argument"]["parsed"]:
                                        args = raw_data["argument"]["parsed"][0]
                                    elif "raw" in raw_data["argument"]:
                                        try:
                                            args = json.loads(raw_data["argument"]["raw"])
                                        except Exception:
                                            pass
                        except Exception:
                            # If parsing fails, continue with empty args
                            pass
                    
                    return self._execute_swaig_function(path_info, args, call_id, raw_data)
            
            elif mode == 'lambda':
                # Check authentication in Lambda mode
                if not self._check_lambda_auth(event):
                    return self._send_lambda_auth_challenge()

                if event:
                    # Support both HTTP API (v2) and REST API (v1) payload formats
                    # HTTP API v2 uses rawPath, REST API v1 uses pathParameters.proxy
                    path = event.get('rawPath', '').strip('/')
                    if not path and event.get('pathParameters'):
                        path = event.get('pathParameters', {}).get('proxy', '')

                    # Parse request body if present
                    args = {}
                    call_id = None
                    raw_data = None
                    function_name = None

                    body_content = event.get('body')
                    if body_content:
                        try:
                            if event.get('isBase64Encoded'):
                                import base64
                                body_content = base64.b64decode(body_content).decode('utf-8')
                            if isinstance(body_content, str):
                                raw_data = json.loads(body_content)
                            else:
                                raw_data = body_content

                            call_id = raw_data.get("call_id")
                            function_name = raw_data.get("function")

                            # Extract arguments like the FastAPI handler does
                            if "argument" in raw_data and isinstance(raw_data["argument"], dict):
                                if "parsed" in raw_data["argument"] and isinstance(raw_data["argument"]["parsed"], list) and raw_data["argument"]["parsed"]:
                                    args = raw_data["argument"]["parsed"][0]
                                elif "raw" in raw_data["argument"]:
                                    try:
                                        args = json.loads(raw_data["argument"]["raw"])
                                    except Exception:
                                        pass
                        except Exception:
                            # If parsing fails, continue with empty args
                            pass

                    # Determine if this is a SWAIG function call
                    # Case 1: /swaig endpoint with function name in body
                    # Case 2: /{function_name} path-based routing
                    # Case 3: Root path - return SWML

                    if path in ('swaig', 'swaig/') and function_name:
                        # /swaig endpoint with function name in body
                        result = self._execute_swaig_function(function_name, args, call_id, raw_data)
                        return {
                            "statusCode": 200,
                            "headers": {"Content-Type": "application/json"},
                            "body": json.dumps(result) if isinstance(result, dict) else str(result)
                        }
                    elif path and path not in ('', 'swaig', 'swaig/'):
                        # Path-based function routing (e.g., /say_hello)
                        result = self._execute_swaig_function(path, args, call_id, raw_data)
                        return {
                            "statusCode": 200,
                            "headers": {"Content-Type": "application/json"},
                            "body": json.dumps(result) if isinstance(result, dict) else str(result)
                        }
                    else:
                        # Root path or /swaig without function - return SWML
                        swml_response = self._render_swml()
                        return {
                            "statusCode": 200,
                            "headers": {"Content-Type": "application/json"},
                            "body": swml_response
                        }
                else:
                    # Handle case when event is None (direct Lambda call with no event)
                    swml_response = self._render_swml()
                    return {
                        "statusCode": 200,
                        "headers": {"Content-Type": "application/json"},
                        "body": swml_response
                    }
            
            elif mode == 'google_cloud_function':
                # Check authentication in Google Cloud Functions mode
                if not self._check_google_cloud_function_auth(event):
                    return self._send_google_cloud_function_auth_challenge()
                
                return self._handle_google_cloud_function_request(event)
            
            elif mode == 'azure_function':
                # Check authentication in Azure Functions mode
                if not self._check_azure_function_auth(event):
                    return self._send_azure_function_auth_challenge()
                
                return self._handle_azure_function_request(event)
            
                
        except Exception as e:
            import logging
            import traceback
            logging.error(f"Error in serverless request handler: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            if mode == 'lambda':
                return {
                    "statusCode": 500,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps({"error": str(e)})
                }
            else:
                raise

    def _execute_swaig_function(self, function_name: str, args: Optional[Dict[str, Any]] = None, call_id: Optional[str] = None, raw_data: Optional[Dict[str, Any]] = None):
        """
        Execute a SWAIG function in serverless context
        
        Args:
            function_name: Name of the function to execute
            args: Function arguments dictionary
            call_id: Optional call ID
            raw_data: Optional raw request data
            
        Returns:
            Function execution result
        """
        # Use the existing logger
        req_log = self.log.bind(
            endpoint="serverless_swaig",
            function=function_name
        )
        
        if call_id:
            req_log = req_log.bind(call_id=call_id)
            
        req_log.debug("serverless_function_call_received")
        
        try:
            # Validate function exists
            if function_name not in self._tool_registry._swaig_functions:
                req_log.warning("function_not_found", available_functions=list(self._tool_registry._swaig_functions.keys()))
                return {"error": f"Function '{function_name}' not found"}
            
            # Use empty args if not provided
            if args is None:
                args = {}
                
            # Use empty raw_data if not provided, but include function call structure
            if raw_data is None:
                raw_data = {
                    "function": function_name,
                    "argument": {
                        "parsed": [args] if args else [],
                        "raw": json.dumps(args) if args else "{}"
                    }
                }
                if call_id:
                    raw_data["call_id"] = call_id
            
            req_log.debug("executing_function", args=json.dumps(args))
            
            # Call the function using the existing on_function_call method
            result = self.on_function_call(function_name, args, raw_data)
            
            # Convert result to dict if needed (same logic as in _handle_swaig_request)
            if isinstance(result, SwaigFunctionResult):
                result_dict = result.to_dict()
            elif isinstance(result, dict):
                result_dict = result
            else:
                result_dict = {"response": str(result)}
            
            req_log.info("serverless_function_executed_successfully")
            req_log.debug("function_result", result=json.dumps(result_dict))
            return result_dict
            
        except Exception as e:
            req_log.error("serverless_function_execution_error", error=str(e))
            return {"error": str(e), "function": function_name}

    def _handle_google_cloud_function_request(self, request):
        """
        Handle Google Cloud Functions specific requests

        Args:
            request: Flask request object from Google Cloud Functions

        Returns:
            Flask response object
        """
        try:
            from urllib.parse import urlparse

            # Get the path from the request
            path = request.path.strip('/')

            # Try to detect and set the base URL from the request for webhook URLs
            base_url = None
            if hasattr(request, 'url') and request.url:
                parsed = urlparse(request.url)
                # Get the base URL without the path
                base_url = f"{parsed.scheme}://{parsed.netloc}"

            # Set the proxy URL base so SWML renders correct webhook URLs
            if base_url and not getattr(self, '_proxy_url_base_from_env', False):
                self._proxy_url_base = base_url

            # Parse request body if present
            args = {}
            call_id = None
            raw_data = None
            function_name = None

            if request.method == 'POST':
                try:
                    if request.is_json:
                        raw_data = request.get_json()
                    else:
                        raw_data = json.loads(request.get_data(as_text=True))

                    call_id = raw_data.get("call_id")
                    function_name = raw_data.get("function")

                    # Extract arguments like the FastAPI handler does
                    if "argument" in raw_data and isinstance(raw_data["argument"], dict):
                        if "parsed" in raw_data["argument"] and isinstance(raw_data["argument"]["parsed"], list) and raw_data["argument"]["parsed"]:
                            args = raw_data["argument"]["parsed"][0]
                        elif "raw" in raw_data["argument"]:
                            try:
                                args = json.loads(raw_data["argument"]["raw"])
                            except Exception:
                                pass
                except Exception:
                    # If parsing fails, continue with empty args
                    pass

            # Determine if this is a SWAIG function call
            # Case 1: /swaig endpoint with function name in body
            # Case 2: /{function_name} path-based routing
            # Case 3: Root path - return SWML

            from flask import Response

            if path in ('swaig', 'swaig/') and function_name:
                # /swaig endpoint with function name in body
                result = self._execute_swaig_function(function_name, args, call_id, raw_data)
                return Response(
                    response=json.dumps(result) if isinstance(result, dict) else str(result),
                    status=200,
                    headers={"Content-Type": "application/json"}
                )
            elif path and path not in ('', 'swaig', 'swaig/'):
                # Path-based function routing (e.g., /say_hello)
                result = self._execute_swaig_function(path, args, call_id, raw_data)
                return Response(
                    response=json.dumps(result) if isinstance(result, dict) else str(result),
                    status=200,
                    headers={"Content-Type": "application/json"}
                )
            else:
                # Root path or /swaig without function - return SWML
                swml_response = self._render_swml()
                return Response(
                    response=swml_response,
                    status=200,
                    headers={"Content-Type": "application/json"}
                )

        except Exception as e:
            import logging
            logging.error(f"Error in Google Cloud Function request handler: {e}")
            from flask import Response
            return Response(
                response=json.dumps({"error": str(e)}),
                status=500,
                headers={"Content-Type": "application/json"}
            )

    def _handle_azure_function_request(self, req):
        """
        Handle Azure Functions specific requests

        Args:
            req: Azure Functions HttpRequest object

        Returns:
            Azure Functions HttpResponse object
        """
        try:
            import azure.functions as func
            from urllib.parse import urlparse

            # Get the path from the request URL
            # Azure Functions URLs look like: https://app.azurewebsites.net/api/function_name/path
            path = ''
            base_url = None
            if req.url:
                parsed = urlparse(req.url)
                # Full path after /api/ e.g. "function_app" or "function_app/swaig"
                url_parts = req.url.split('/api/')
                if len(url_parts) > 1:
                    full_path = url_parts[1].strip('/')
                    # Split into function name and sub-path
                    path_parts = full_path.split('/', 1)
                    function_app_name = path_parts[0] if path_parts else ''
                    path = path_parts[1] if len(path_parts) > 1 else ''

                    # Base URL includes the function app name for webhook URLs
                    # e.g., https://app.azurewebsites.net/api/function_app
                    base_url = f"{parsed.scheme}://{parsed.netloc}/api/{function_app_name}"
                else:
                    base_url = f"{parsed.scheme}://{parsed.netloc}/api"

            # Set the proxy URL base so SWML renders correct webhook URLs
            if base_url and not getattr(self, '_proxy_url_base_from_env', False):
                self._proxy_url_base = base_url

            # Parse request body if present
            args = {}
            call_id = None
            raw_data = None
            function_name = None

            if req.method == 'POST':
                try:
                    body = req.get_body()
                    if body:
                        raw_data = json.loads(body.decode('utf-8'))
                        call_id = raw_data.get("call_id")
                        function_name = raw_data.get("function")

                        # Extract arguments like the FastAPI handler does
                        if "argument" in raw_data and isinstance(raw_data["argument"], dict):
                            if "parsed" in raw_data["argument"] and isinstance(raw_data["argument"]["parsed"], list) and raw_data["argument"]["parsed"]:
                                args = raw_data["argument"]["parsed"][0]
                            elif "raw" in raw_data["argument"]:
                                try:
                                    args = json.loads(raw_data["argument"]["raw"])
                                except Exception:
                                    pass
                except Exception:
                    # If parsing fails, continue with empty args
                    pass

            # Determine if this is a SWAIG function call
            # Case 1: /swaig endpoint with function name in body
            # Case 2: /{function_name} path-based routing
            # Case 3: Root path - return SWML

            if path in ('swaig', 'swaig/') and function_name:
                # /swaig endpoint with function name in body
                result = self._execute_swaig_function(function_name, args, call_id, raw_data)
                return func.HttpResponse(
                    body=json.dumps(result) if isinstance(result, dict) else str(result),
                    status_code=200,
                    headers={"Content-Type": "application/json"}
                )
            elif path and path not in ('', 'api', 'swaig', 'swaig/'):
                # Path-based function routing (e.g., /say_hello)
                result = self._execute_swaig_function(path, args, call_id, raw_data)
                return func.HttpResponse(
                    body=json.dumps(result) if isinstance(result, dict) else str(result),
                    status_code=200,
                    headers={"Content-Type": "application/json"}
                )
            else:
                # Root path or /swaig without function - return SWML
                swml_response = self._render_swml()
                return func.HttpResponse(
                    body=swml_response,
                    status_code=200,
                    headers={"Content-Type": "application/json"}
                )

        except Exception as e:
            import logging
            logging.error(f"Error in Azure Function request handler: {e}")
            import azure.functions as func
            return func.HttpResponse(
                body=json.dumps({"error": str(e)}),
                status_code=500,
                headers={"Content-Type": "application/json"}
            )