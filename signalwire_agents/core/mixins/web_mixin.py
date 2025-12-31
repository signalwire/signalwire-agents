"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import os
import json
import base64
import signal
import sys
from typing import Optional, Dict, Any, Callable
from urllib.parse import urlparse, urlunparse

from fastapi import FastAPI, APIRouter, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from signalwire_agents.core.logging_config import get_execution_mode
from signalwire_agents.core.function_result import SwaigFunctionResult


class WebMixin:
    """
    Mixin class containing all web server and routing-related methods for AgentBase
    """
    
    def get_app(self):
        """
        Get the FastAPI application instance for deployment adapters like Lambda/Mangum
        
        This method ensures the FastAPI app is properly initialized and configured,
        then returns it for use with deployment adapters like Mangum for AWS Lambda.
        
        Returns:
            FastAPI: The configured FastAPI application instance
        """
        if self._app is None:
            # Initialize the app if it hasn't been created yet
            # This follows the same initialization logic as serve() but without running uvicorn
            from fastapi import FastAPI
            from fastapi.middleware.cors import CORSMiddleware
            
            # Create a FastAPI app with explicit redirect_slashes=False
            app = FastAPI(redirect_slashes=False)
            
            # Add health and ready endpoints directly to the main app to avoid conflicts with catch-all
            @app.get("/health")
            @app.post("/health")
            async def health_check():
                """Health check endpoint for Kubernetes liveness probe"""
                return {
                    "status": "healthy",
                    "agent": self.get_name(),
                    "route": self.route,
                    "functions": len(self._tool_registry._swaig_functions)
                }
            
            @app.get("/ready")
            @app.post("/ready")
            async def readiness_check():
                """Readiness check endpoint for Kubernetes readiness probe"""
                return {
                    "status": "ready",
                    "agent": self.get_name(),
                    "route": self.route,
                    "functions": len(self._tool_registry._swaig_functions)
                }
            
            # Add CORS middleware if needed
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            # Create router and register routes
            router = self.as_router()
            
            # Log registered routes for debugging
            self.log.debug("router_routes_registered")
            for route in router.routes:
                if hasattr(route, "path"):
                    self.log.debug("router_route", path=route.path)
            
            # Include the router
            app.include_router(router, prefix=self.route)
            
            # Register a catch-all route for debugging and troubleshooting
            @app.get("/{full_path:path}")
            @app.post("/{full_path:path}")
            async def handle_all_routes(request: Request, full_path: str):
                self.log.debug("request_received", path=full_path)
                
                # Check if the path is meant for this agent
                if not full_path.startswith(self.route.lstrip("/")):
                    return {"error": "Invalid route"}
                
                # Extract the path relative to this agent's route
                relative_path = full_path[len(self.route.lstrip("/")):]
                relative_path = relative_path.lstrip("/")
                self.log.debug("relative_path_extracted", path=relative_path)
            
            # Log all app routes for debugging
            self.log.debug("app_routes_registered")
            for route in app.routes:
                if hasattr(route, "path"):
                    self.log.debug("app_route", path=route.path)
            
            self._app = app
        
        return self._app
    
    def as_router(self) -> APIRouter:
        """
        Get a FastAPI router for this agent
        
        Returns:
            FastAPI router
        """
        # Create a router with explicit redirect_slashes=False
        router = APIRouter(redirect_slashes=False)
        
        # Register routes explicitly
        self._register_routes(router)
        
        # Log all registered routes for debugging
        self.log.debug("routes_registered", agent=self.name)
        for route in router.routes:
            self.log.debug("route_registered", path=route.path)
        
        return router

    def serve(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """
        Start a web server for this agent
        
        Args:
            host: Optional host to override the default
            port: Optional port to override the default
        """
        import uvicorn
        
        if self._app is None:
            # Create a FastAPI app with explicit redirect_slashes=False
            app = FastAPI(redirect_slashes=False)
            
            # Add health and ready endpoints directly to the main app to avoid conflicts with catch-all
            @app.get("/health")
            @app.post("/health")
            async def health_check():
                """Health check endpoint for Kubernetes liveness probe"""
                return {
                    "status": "healthy",
                    "agent": self.get_name(),
                    "route": self.route,
                    "functions": len(self._tool_registry._swaig_functions)
                }
            
            @app.get("/ready")
            @app.post("/ready")
            async def readiness_check():
                """Readiness check endpoint for Kubernetes readiness probe"""
                # Check if agent is properly initialized
                ready = (
                    hasattr(self, '_tool_registry') and
                    hasattr(self, 'schema_utils') and
                    self.schema_utils is not None
                )
                
                status_code = 200 if ready else 503
                return Response(
                    content=json.dumps({
                        "status": "ready" if ready else "not_ready",
                        "agent": self.get_name(),
                        "initialized": ready
                    }),
                    status_code=status_code,
                    media_type="application/json"
                )
            
            # Get router for this agent
            router = self.as_router()
            
            # Register a catch-all route for debugging and troubleshooting
            @app.get("/{full_path:path}")
            @app.post("/{full_path:path}")
            async def handle_all_routes(request: Request, full_path: str):
                self.log.debug("request_received", path=full_path)
                
                # Check if the path is meant for this agent
                if not full_path.startswith(self.route.lstrip("/")):
                    return {"error": "Invalid route"}
                
                # Extract the path relative to this agent's route
                relative_path = full_path[len(self.route.lstrip("/")):]
                relative_path = relative_path.lstrip("/")
                self.log.debug("path_extracted", relative_path=relative_path)
                
                # Perform routing based on the relative path
                if not relative_path or relative_path == "/":
                    # Root endpoint
                    return await self._handle_root_request(request)
                
                # Strip trailing slash for processing
                clean_path = relative_path.rstrip("/")
                
                # Check for standard endpoints
                if clean_path == "debug":
                    return await self._handle_debug_request(request)
                elif clean_path == "swaig":
                    return await self._handle_swaig_request(request, Response())
                elif clean_path == "post_prompt":
                    return await self._handle_post_prompt_request(request)
                elif clean_path == "check_for_input":
                    return await self._handle_check_for_input_request(request)
                
                # Check for custom routing callbacks
                if hasattr(self, '_routing_callbacks'):
                    for callback_path, callback_fn in self._routing_callbacks.items():
                        cb_path_clean = callback_path.strip("/")
                        if clean_path == cb_path_clean:
                            # Found a matching callback
                            request.state.callback_path = callback_path
                            return await self._handle_root_request(request)
                
                # Default: 404
                return {"error": "Path not found"}
            
            # Include router with prefix (handle root route special case)
            if self.route == "/":
                app.include_router(router)
            else:
                app.include_router(router, prefix=self.route)
            
            # Log all app routes for debugging
            self.log.debug("app_routes_registered")
            for route in app.routes:
                if hasattr(route, "path"):
                    self.log.debug("app_route", path=route.path)
            
            self._app = app
        
        host = host or self.host
        port = port or self.port
        
        # Print the auth credentials with source
        username, password, source = self.get_basic_auth_credentials(include_source=True)
        
        # Get the proper URL using unified URL building
        startup_url = self.get_full_url(include_auth=False)
        
        # Log startup information using structured logging
        self.log.info("agent_starting",
                     agent=self.name,
                     url=startup_url,
                     username=username,
                     password_length=len(password),
                     auth_source=source,
                     ssl_enabled=getattr(self, 'ssl_enabled', False))
        
        # Print user-friendly startup message (keep this for development UX)
        print(f"Agent '{self.name}' is available at:")
        print(f"URL: {startup_url}")
        print(f"Basic Auth: {username}:{password} (source: {source})")
        
        # Check if SSL is enabled and start uvicorn accordingly
        if getattr(self, 'ssl_enabled', False) and getattr(self, 'ssl_cert_path', None) and getattr(self, 'ssl_key_path', None):
            self.log.info("starting_with_ssl", cert=self.ssl_cert_path, key=self.ssl_key_path)
            uvicorn.run(
                self._app,
                host=host,
                port=port,
                ssl_certfile=self.ssl_cert_path,
                ssl_keyfile=self.ssl_key_path
            )
        else:
            uvicorn.run(self._app, host=host, port=port)

    def run(self, event=None, context=None, force_mode=None, host: Optional[str] = None, port: Optional[int] = None):
        """
        Smart run method that automatically detects environment and handles accordingly
        
        Args:
            event: Serverless event object (Lambda, Cloud Functions)
            context: Serverless context object (Lambda, Cloud Functions)
            force_mode: Override automatic mode detection for testing
            host: Host override for server mode
            port: Port override for server mode
            
        Returns:
            Response for serverless modes, None for server mode
        """
        mode = force_mode or get_execution_mode()
        
        try:
            if mode in ['cgi', 'azure_function']:
                response = self.handle_serverless_request(event, context, mode)
                print(response)
                return response
            elif mode in ['lambda', 'google_cloud_function']:
                return self.handle_serverless_request(event, context, mode)
            else:
                # Server mode - use existing serve method
                self.serve(host, port)
        except Exception as e:
            import logging
            logging.error(f"Error in run method: {e}")
            if mode == 'lambda':
                return {
                    "statusCode": 500,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps({"error": str(e)})
                }
            else:
                raise

    def _register_routes(self, router):
        """
        Register routes for this agent
        
        This method ensures proper route registration by handling the routes 
        directly in AgentBase rather than inheriting from SWMLService.
        
        Args:
            router: FastAPI router to register routes with
        """
        # Health check endpoints are now registered directly on the main app
        
        # Root endpoint (handles both with and without trailing slash)
        @router.get("/")
        @router.post("/")
        async def handle_root(request: Request, response: Response):
            """Handle GET/POST requests to the root endpoint"""
            return await self._handle_root_request(request)
            
        # Debug endpoint - Both versions
        @router.get("/debug")
        @router.get("/debug/")
        @router.post("/debug")
        @router.post("/debug/")
        async def handle_debug(request: Request):
            """Handle GET/POST requests to the debug endpoint"""
            return await self._handle_debug_request(request)
            
        # SWAIG endpoint - Both versions 
        @router.get("/swaig")
        @router.get("/swaig/")
        @router.post("/swaig")
        @router.post("/swaig/")
        async def handle_swaig(request: Request, response: Response):
            """Handle GET/POST requests to the SWAIG endpoint"""
            return await self._handle_swaig_request(request, response)
            
        # Post prompt endpoint - Both versions
        @router.get("/post_prompt")
        @router.get("/post_prompt/")
        @router.post("/post_prompt")
        @router.post("/post_prompt/")
        async def handle_post_prompt(request: Request):
            """Handle GET/POST requests to the post_prompt endpoint"""
            return await self._handle_post_prompt_request(request)
            
        # Check for input endpoint - Both versions
        @router.get("/check_for_input")
        @router.get("/check_for_input/")
        @router.post("/check_for_input")
        @router.post("/check_for_input/")
        async def handle_check_for_input(request: Request):
            """Handle GET/POST requests to the check_for_input endpoint"""
            return await self._handle_check_for_input_request(request)
        
        # Register callback routes for routing callbacks if available
        if hasattr(self, '_routing_callbacks') and self._routing_callbacks:
            for callback_path, callback_fn in self._routing_callbacks.items():
                # Skip the root path as it's already handled
                if callback_path == "/":
                    continue
                
                # Register both with and without trailing slash
                path = callback_path.rstrip("/")
                path_with_slash = f"{path}/"
                
                @router.get(path)
                @router.get(path_with_slash)
                @router.post(path)
                @router.post(path_with_slash)
                async def handle_callback(request: Request, response: Response, cb_path=callback_path):
                    """Handle GET/POST requests to a registered callback path"""
                    # Store the callback path in request state for _handle_request to use
                    request.state.callback_path = cb_path
                    return await self._handle_root_request(request)
                
                self.log.info("callback_endpoint_registered", path=callback_path)
    
    async def _handle_root_request(self, request: Request):
        """Handle GET/POST requests to the root endpoint"""
        # Debug logging to understand the state before any changes
        self.log.debug("_handle_root_request entry",
                      proxy_url_base=getattr(self, '_proxy_url_base', None),
                      proxy_url_base_from_env=getattr(self, '_proxy_url_base_from_env', False),
                      env_var=os.environ.get('SWML_PROXY_URL_BASE'))
        
        # Always detect proxy from current request headers - this allows mixing direct and proxied access
        # Check for proxy headers
        forwarded_host = request.headers.get("X-Forwarded-Host")
        forwarded_proto = request.headers.get("X-Forwarded-Proto", "http")
        
        if forwarded_host:
            # Only update proxy URL if it wasn't set from environment
            if not getattr(self, '_proxy_url_base_from_env', False):
                # Set proxy_url_base on both self and super() to ensure it's shared
                self._proxy_url_base = f"{forwarded_proto}://{forwarded_host}"
                self._current_request = request  # Store current request for get_full_url
                if hasattr(super(), '_proxy_url_base'):
                    # Ensure parent class has the same proxy URL
                    super()._proxy_url_base = self._proxy_url_base
                
                self.log.debug("proxy_detected_for_request", proxy_url_base=self._proxy_url_base, 
                             source="X-Forwarded headers")
            else:
                self.log.debug("proxy headers present but keeping env proxy URL",
                             forwarded_proto=forwarded_proto,
                             forwarded_host=forwarded_host,
                             keeping_proxy_url=self._proxy_url_base,
                             source="environment variable")
        else:
            # No proxy headers - only clear proxy URL if it wasn't set from environment
            if not getattr(self, '_proxy_url_base_from_env', False):
                self.log.debug("No proxy headers found, clearing proxy URL",
                             proxy_url_base_from_env=getattr(self, '_proxy_url_base_from_env', False))
                self._proxy_url_base = None
                if hasattr(super(), '_proxy_url_base'):
                    super()._proxy_url_base = None
            else:
                self.log.debug("No proxy headers found, but keeping env proxy URL",
                             proxy_url_base=getattr(self, '_proxy_url_base', None),
                             proxy_url_base_from_env=True)
            self._current_request = request  # Store current request for get_full_url
            
            # Try the parent class detection method if it exists
            if hasattr(super(), '_detect_proxy_from_request'):
                # Call the parent's detection method
                super()._detect_proxy_from_request(request)
                # Copy the result to our class if found
                if hasattr(super(), '_proxy_url_base') and getattr(super(), '_proxy_url_base', None):
                    self._proxy_url_base = super()._proxy_url_base
        
        # Check if this is a callback path request
        callback_path = getattr(request.state, "callback_path", None)
        
        req_log = self.log.bind(
            endpoint="root" if not callback_path else f"callback:{callback_path}",
            method=request.method,
            path=request.url.path
        )
        
        req_log.debug("endpoint_called")
        
        try:
            # Check auth
            if not self._check_basic_auth(request):
                req_log.warning("unauthorized_access_attempt")
                return Response(
                    content=json.dumps({"error": "Unauthorized"}),
                    status_code=401,
                    headers={"WWW-Authenticate": "Basic"},
                    media_type="application/json"
                )
            
            # Try to parse request body for POST
            body = {}
            call_id = None
            
            if request.method == "POST":
                # Check if body is empty first
                raw_body = await request.body()
                if raw_body:
                    try:
                        body = await request.json()
                        req_log.debug("request_body_received", body_size=len(str(body)))
                        if body:
                            req_log.debug("request_body")
                    except Exception as e:
                        req_log.warning("error_parsing_request_body", error=str(e))
                        # Continue processing with empty body
                        body = {}
                else:
                    req_log.debug("empty_request_body")
                    
                # Get call_id from body if present
                call_id = body.get("call_id")
                if not call_id and "call" in body:
                    # Sometimes it might be nested under 'call'
                    call_id = body.get("call", {}).get("call_id")
                req_log.debug("extracted_call_id_from_body", call_id=call_id, body_keys=list(body.keys()))
            else:
                # Get call_id from query params for GET
                call_id = request.query_params.get("call_id")
                
            # Add call_id to logger if any
            if call_id:
                req_log = req_log.bind(call_id=call_id)
                req_log.debug("call_id_identified")
            
            # Check if this is a callback path and we need to apply routing
            if callback_path and hasattr(self, '_routing_callbacks') and callback_path in self._routing_callbacks:
                callback_fn = self._routing_callbacks[callback_path]
                
                if request.method == "POST" and body:
                    req_log.debug("processing_routing_callback", path=callback_path)
                    # Call the routing callback
                    try:
                        route = callback_fn(request, body)
                        if route is not None:
                            req_log.info("routing_request", route=route)
                            # Return a redirect to the new route
                            return Response(
                                status_code=307,  # 307 Temporary Redirect preserves the method and body
                                headers={"Location": route}
                            )
                    except Exception as e:
                        req_log.error("error_in_routing_callback", error=str(e))
            
            # Allow subclasses to inspect/modify the request
            modifications = None
            try:
                modifications = self.on_swml_request(body, callback_path, request)
                if modifications:
                    req_log.debug("request_modifications_applied")
            except Exception as e:
                req_log.error("error_in_request_modifier", error=str(e))
            
            # Render SWML
            swml = self._render_swml(call_id, modifications)
            req_log.debug("swml_rendered", swml_size=len(swml))
            
            # Return as JSON
            req_log.info("request_successful")
            return Response(
                content=swml,
                media_type="application/json"
            )
        except Exception as e:
            req_log.error("request_failed", error=str(e))
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=500,
                media_type="application/json"
            )
    
    async def _handle_debug_request(self, request: Request):
        """Handle GET/POST requests to the debug endpoint"""
        req_log = self.log.bind(
            endpoint="debug",
            method=request.method,
            path=request.url.path
        )
        
        req_log.debug("endpoint_called")
        
        try:
            # Check auth
            if not self._check_basic_auth(request):
                req_log.warning("unauthorized_access_attempt")
                return Response(
                    content=json.dumps({"error": "Unauthorized"}),
                    status_code=401,
                    headers={"WWW-Authenticate": "Basic"},
                    media_type="application/json"
                )
            
            # Get call_id from either query params (GET) or body (POST)
            call_id = None
            body = {}
            
            if request.method == "POST":
                try:
                    body = await request.json()
                    req_log.debug("request_body_received", body_size=len(str(body)))
                    call_id = body.get("call_id")
                except Exception as e:
                    req_log.warning("error_parsing_request_body", error=str(e))
            else:
                call_id = request.query_params.get("call_id")
            
            # Add call_id to logger if any
            if call_id:
                req_log = req_log.bind(call_id=call_id)
                req_log.debug("call_id_identified")
                
            # Allow subclasses to inspect/modify the request
            modifications = None
            try:
                modifications = self.on_swml_request(body, None, request)
                if modifications:
                    req_log.debug("request_modifications_applied")
            except Exception as e:
                req_log.error("error_in_request_modifier", error=str(e))
                
            # Render SWML
            swml = self._render_swml(call_id, modifications)
            req_log.debug("swml_rendered", swml_size=len(swml))
            
            # Return as JSON
            req_log.info("request_successful")
            return Response(
                content=swml,
                media_type="application/json",
                headers={"X-Debug": "true"}
            )
        except Exception as e:
            req_log.error("request_failed", error=str(e))
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=500,
                media_type="application/json"
            )
    
    async def _handle_swaig_request(self, request: Request, response: Response):
        """Handle GET/POST requests to the SWAIG endpoint"""
        req_log = self.log.bind(
            endpoint="swaig",
            method=request.method,
            path=request.url.path
        )
        
        req_log.debug("endpoint_called")
        
        try:
            # Check auth
            if not self._check_basic_auth(request):
                req_log.warning("unauthorized_access_attempt")
                response.headers["WWW-Authenticate"] = "Basic"
                return Response(
                    content=json.dumps({"error": "Unauthorized"}),
                    status_code=401,
                    headers={"WWW-Authenticate": "Basic"},
                    media_type="application/json"
                )
            
            # Handle differently based on method
            if request.method == "GET":
                # For GET requests, return the SWML document (same as root endpoint)
                call_id = request.query_params.get("call_id")
                swml = self._render_swml(call_id)
                req_log.debug("swml_rendered", swml_size=len(swml))
                return Response(
                    content=swml,
                    media_type="application/json"
                )
            
            # For POST requests, process SWAIG function calls
            try:
                body = await request.json()
                req_log.debug("request_body_received", body_size=len(str(body)))
                if body:
                    req_log.debug("request_body", body=json.dumps(body))
            except Exception as e:
                req_log.error("error_parsing_request_body", error=str(e))
                body = {}
            
            # Extract function name
            function_name = body.get("function")
            if not function_name:
                req_log.warning("missing_function_name")
                return Response(
                    content=json.dumps({"error": "Missing function name"}),
                    status_code=400,
                    media_type="application/json"
                )
            
            # Add function info to logger
            req_log = req_log.bind(function=function_name)
            req_log.debug("function_call_received")
            
            # Extract arguments
            args = {}
            if "argument" in body and isinstance(body["argument"], dict):
                if "parsed" in body["argument"] and isinstance(body["argument"]["parsed"], list) and body["argument"]["parsed"]:
                    args = body["argument"]["parsed"][0]
                    req_log.debug("parsed_arguments", args=json.dumps(args))
                elif "raw" in body["argument"]:
                    try:
                        args = json.loads(body["argument"]["raw"])
                        req_log.debug("raw_arguments_parsed", args=json.dumps(args))
                    except Exception as e:
                        req_log.error("error_parsing_raw_arguments", error=str(e), raw=body["argument"]["raw"])
            
            # Get call_id from body
            call_id = body.get("call_id")
            if call_id:
                req_log = req_log.bind(call_id=call_id)
                req_log.debug("call_id_identified")
            
            # SECURITY BYPASS FOR DEBUGGING - make all functions work regardless of token
            # We'll log the attempt but allow it through
            token = request.query_params.get("__token") or request.query_params.get("token")  # Check __token first, fallback to token
            if token:
                req_log.debug("token_found", token_length=len(token))
                
                # Check token validity but don't reject the request
                if hasattr(self, '_session_manager') and function_name in self._tool_registry._swaig_functions:
                    is_valid = self._session_manager.validate_tool_token(function_name, token, call_id)
                    if is_valid:
                        req_log.debug("token_valid")
                    else:
                        # Log but continue anyway for debugging
                        req_log.warning("token_invalid")
                        if hasattr(self._session_manager, 'debug_token'):
                            debug_info = self._session_manager.debug_token(token)
                            req_log.debug("token_debug", debug=json.dumps(debug_info))
            
            # Check if we need to use an ephemeral agent for dynamic configuration
            agent_to_use = self
            if self._dynamic_config_callback and request:
                # Check query params to see if this request came from a dynamically configured agent
                # This would have been preserved through add_swaig_query_params
                # For example, conversation_type=chat would be in the query params
                agent_to_use = self._create_ephemeral_copy()
                
                try:
                    # Extract request data
                    query_params = dict(request.query_params)
                    headers = dict(request.headers)
                    
                    # Call the dynamic config callback with the ephemeral agent
                    # Pass the body as body_params for context
                    self._dynamic_config_callback(query_params, body, headers, agent_to_use)
                    
                except Exception as e:
                    req_log.error("dynamic_config_error", error=str(e))
            
            # Call the function
            try:
                result = agent_to_use.on_function_call(function_name, args, body)
                
                # Convert result to dict if needed
                if isinstance(result, SwaigFunctionResult):
                    result_dict = result.to_dict()
                elif isinstance(result, dict):
                    result_dict = result
                else:
                    result_dict = {"response": str(result)}
                
                req_log.info("function_executed_successfully")
                req_log.debug("function_result", result=json.dumps(result_dict))
                return result_dict
            except Exception as e:
                req_log.error("function_execution_error", error=str(e))
                return {"error": str(e), "function": function_name}
                
        except Exception as e:
            req_log.error("request_failed", error=str(e))
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=500,
                media_type="application/json"
            )

    async def _handle_post_prompt_request(self, request: Request):
        """Handle GET/POST requests to the post_prompt endpoint"""
        req_log = self.log.bind(
            endpoint="post_prompt",
            method=request.method,
            path=request.url.path
        )
        
        # Only log if not suppressed
        if not getattr(self, '_suppress_logs', False):
            req_log.debug("endpoint_called")
        
        try:
            # Check auth
            if not self._check_basic_auth(request):
                req_log.warning("unauthorized_access_attempt")
                return Response(
                    content=json.dumps({"error": "Unauthorized"}),
                    status_code=401,
                    headers={"WWW-Authenticate": "Basic"},
                    media_type="application/json"
                )
                
            # Extract call_id for use with token validation
            call_id = request.query_params.get("call_id")
            
            # For POST requests, try to also get call_id from body
            if request.method == "POST":
                try:
                    body_text = await request.body()
                    if body_text:
                        body_data = json.loads(body_text)
                        if call_id is None:
                            call_id = body_data.get("call_id")
                        # Save body_data for later use
                        setattr(request, "_post_prompt_body", body_data)
                except Exception as e:
                    req_log.error("error_extracting_call_id", error=str(e))
                    
            # If we have a call_id, add it to the logger context
            if call_id:
                req_log = req_log.bind(call_id=call_id)
                
            # Check token if provided
            token = request.query_params.get("__token") or request.query_params.get("token")  # Check __token first, fallback to token
            token_validated = False
            
            if token:
                req_log.debug("token_found", token_length=len(token))
                
                # Try to validate token, but continue processing regardless
                if call_id and hasattr(self, '_session_manager'):
                    try:
                        is_valid = self._session_manager.validate_tool_token("post_prompt", token, call_id)
                        if is_valid:
                            req_log.debug("token_valid")
                            token_validated = True
                        else:
                            req_log.warning("invalid_token")
                            # Debug information for token validation issues
                            if hasattr(self._session_manager, 'debug_token'):
                                debug_info = self._session_manager.debug_token(token)
                                req_log.debug("token_debug", debug=json.dumps(debug_info))
                    except Exception as e:
                        req_log.error("token_validation_error", error=str(e))
                        
            # For GET requests, return the SWML document
            if request.method == "GET":
                # Check if we should use dynamic config via on_swml_request
                modifications = self.on_swml_request(None, None, request)
                swml = self._render_swml(call_id, modifications)
                req_log.debug("swml_rendered", swml_size=len(swml))
                return Response(
                    content=swml,
                    media_type="application/json"
                )
            
            # For POST requests, process the post-prompt data
            try:
                # Try to reuse the body we already parsed for call_id extraction
                if hasattr(request, "_post_prompt_body"):
                    body = getattr(request, "_post_prompt_body")
                else:
                    body = await request.json()
                
                # Only log if not suppressed
                if not getattr(self, '_suppress_logs', False):
                    req_log.debug("request_body_received", body_size=len(str(body)))
                    # Log the raw body directly (let the logger handle the JSON encoding)
                    req_log.info("post_prompt_body", body=body)
            except Exception as e:
                req_log.error("error_parsing_request_body", error=str(e))
                body = {}
                
            # Check if we need to use an ephemeral agent for dynamic configuration
            agent_to_use = self
            if self._dynamic_config_callback and request:
                # Create ephemeral copy and apply dynamic config
                agent_to_use = self._create_ephemeral_copy()
                
                try:
                    # Extract request data
                    query_params = dict(request.query_params)
                    headers = dict(request.headers)
                    
                    # Call the dynamic config callback with the ephemeral agent
                    self._dynamic_config_callback(query_params, body, headers, agent_to_use)
                    
                except Exception as e:
                    req_log.error("dynamic_config_error", error=str(e))
            
            # Extract summary from the correct location in the request
            summary = agent_to_use._find_summary_in_post_data(body, req_log)

            # Call the summary handler with the summary and the full body
            result = None
            try:
                if summary:
                    result = agent_to_use.on_summary(summary, body)
                    req_log.debug("summary_handler_called_successfully")
                else:
                    # If no summary found but still want to process the data
                    result = agent_to_use.on_summary(None, body)
                    req_log.debug("summary_handler_called_with_null_summary")
            except Exception as e:
                req_log.error("error_in_summary_handler", error=str(e))

            # For fetch_conversation, return the result from on_summary
            # SignalWire expects conversation_summary in the response
            action = body.get("action", "")
            if action == "fetch_conversation" and result is not None:
                req_log.info("request_successful", action=action, returning_result=True)
                return result

            # Return success for save/post actions
            req_log.info("request_successful")
            return {"success": True}
        except Exception as e:
            req_log.error("request_failed", error=str(e))
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=500,
                media_type="application/json"
            )
    
    async def _handle_check_for_input_request(self, request: Request):
        """Handle GET/POST requests to the check_for_input endpoint"""
        req_log = self.log.bind(
            endpoint="check_for_input",
            method=request.method,
            path=request.url.path
        )
        
        req_log.debug("endpoint_called")
        
        try:
            # Check auth
            if not self._check_basic_auth(request):
                req_log.warning("unauthorized_access_attempt")
                return Response(
                    content=json.dumps({"error": "Unauthorized"}),
                    status_code=401,
                    headers={"WWW-Authenticate": "Basic"},
                    media_type="application/json"
                )
            
            # For both GET and POST requests, process input check
            conversation_id = None
            
            if request.method == "POST":
                try:
                    body = await request.json()
                    req_log.debug("request_body_received", body_size=len(str(body)))
                    conversation_id = body.get("conversation_id")
                except Exception as e:
                    req_log.error("error_parsing_request_body", error=str(e))
            else:
                conversation_id = request.query_params.get("conversation_id")
            
            if not conversation_id:
                req_log.warning("missing_conversation_id")
                return Response(
                    content=json.dumps({"error": "Missing conversation_id parameter"}),
                    status_code=400,
                    media_type="application/json"
                )
            
            # Here you would typically check for new input in some external system
            # For this implementation, we'll return an empty result
            return {
                "status": "success",
                "conversation_id": conversation_id,
                "new_input": False,
                "messages": []
            }
        except Exception as e:
            req_log.error("request_failed", error=str(e))
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=500,
                media_type="application/json"
            )
    
    def on_request(self, request_data: Optional[dict] = None, callback_path: Optional[str] = None) -> Optional[dict]:
        """
        Called when SWML is requested, with request data when available
        
        This method overrides SWMLService's on_request to properly handle SWML generation
        for AI Agents.
        
        Args:
            request_data: Optional dictionary containing the parsed POST body
            callback_path: Optional callback path
            
        Returns:
            None to use the default SWML rendering (which will call _render_swml)
        """
        # Call on_swml_request for customization
        if hasattr(self, 'on_swml_request') and callable(getattr(self, 'on_swml_request')):
            return self.on_swml_request(request_data, callback_path, None)
            
        # If no on_swml_request or it returned None, we'll proceed with default rendering
        return None
    
    def on_swml_request(self, request_data: Optional[dict] = None, callback_path: Optional[str] = None, request: Optional[Request] = None) -> Optional[dict]:
        """
        Customization point for subclasses to modify SWML based on request data
        
        Args:
            request_data: Optional dictionary containing the parsed POST body
            callback_path: Optional callback path
            request: Optional FastAPI Request object for accessing query params, headers, etc.
            
        Returns:
            Optional dict with modifications to apply to the SWML document
        """
        # Dynamic config is handled differently now - we don't modify the agent here
        # Instead, we'll return a marker that tells _render_swml to use an ephemeral copy
        # UNLESS we're already on an ephemeral agent (to prevent infinite recursion)
        # 
        # IMPORTANT: We ALWAYS return the marker if we have a dynamic config callback,
        # even if request is None. This ensures the first request gets dynamic config too.
        self.log.debug("on_swml_request_called", 
                      has_dynamic_callback=bool(self._dynamic_config_callback),
                      is_ephemeral=getattr(self, '_is_ephemeral', False),
                      has_request=bool(request))
        
        if self._dynamic_config_callback and not getattr(self, '_is_ephemeral', False):
            # Return a special marker that _render_swml will recognize
            self.log.debug("returning_ephemeral_marker")
            return {"__use_ephemeral_agent": True, "__request": request, "__request_data": request_data}
        
        # Return None to indicate no SWML modifications needed
        return None

    def register_routing_callback(self, callback_fn: Callable[[Request, Dict[str, Any]], Optional[str]], 
                                 path: str = "/sip") -> None:
        """
        Register a callback function that will be called to determine routing
        based on POST data.
        
        When a routing callback is registered, an endpoint at the specified path is automatically
        created that will handle requests. This endpoint will use the callback to
        determine if the request should be processed by this service or redirected.
        
        The callback should take a request object and request body dictionary and return:
        - A route string if it should be routed to a different endpoint
        - None if normal processing should continue
        
        Args:
            callback_fn: The callback function to register
            path: The path where this callback should be registered (default: "/sip")
        """
        # Normalize the path (remove trailing slash)
        normalized_path = path.rstrip("/")
        if not normalized_path.startswith("/"):
            normalized_path = f"/{normalized_path}"
            
        # Store the callback with the normalized path (without trailing slash)
        self.log.info("registering_routing_callback", path=normalized_path)
        if not hasattr(self, '_routing_callbacks'):
            self._routing_callbacks = {}
        self._routing_callbacks[normalized_path] = callback_fn

    def set_dynamic_config_callback(self, callback: Callable[[dict, dict, dict, 'AgentBase'], None]) -> 'AgentBase':
        """
        Set a callback function for dynamic agent configuration
        
        This callback receives the actual agent instance, allowing you to dynamically
        configure ANY aspect of the agent including adding skills, modifying prompts,
        changing parameters, etc. based on request data.
        
        Args:
            callback: Function that takes (query_params, body_params, headers, agent)
                     and configures the agent using any available methods like:
                     - agent.add_skill(...)
                     - agent.add_language(...)
                     - agent.prompt_add_section(...)
                     - agent.set_params(...)
                     - agent.set_global_data(...)
                     - agent.define_tool(...)
                     
        Example:
            def my_config(query_params, body_params, headers, agent):
                if query_params.get('tier') == 'premium':
                    agent.add_skill("advanced_search")
                    agent.add_language("English", "en-US", "premium_voice")
                    agent.set_params({"end_of_speech_timeout": 500})
                agent.set_global_data({"tier": query_params.get('tier', 'standard')})
            
            my_agent.set_dynamic_config_callback(my_config)
        """
        self._dynamic_config_callback = callback
        return self

    def manual_set_proxy_url(self, proxy_url: str) -> 'AgentBase':
        """
        Manually set the proxy URL base for webhook callbacks
        
        This can be called at runtime to set or update the proxy URL
        
        Args:
            proxy_url: The base URL to use for webhooks (e.g., https://example.ngrok.io)
            
        Returns:
            Self for method chaining
        """
        if proxy_url:
            # Set on self
            self._proxy_url_base = proxy_url.rstrip('/')
            self._proxy_detection_done = True
            
            # Set on parent if it has these attributes
            if hasattr(super(), '_proxy_url_base'):
                super()._proxy_url_base = self._proxy_url_base
            if hasattr(super(), '_proxy_detection_done'):
                super()._proxy_detection_done = True
                
            self.log.info("proxy_url_manually_set", proxy_url_base=self._proxy_url_base)
            
        return self

    def setup_graceful_shutdown(self) -> None:
        """
        Setup signal handlers for graceful shutdown (useful for Kubernetes)
        """
        def signal_handler(signum, frame):
            self.log.info("shutdown_signal_received", signal=signum)
            
            # Perform cleanup
            try:
                # Close any open resources
                if hasattr(self, '_session_manager'):
                    # Could add cleanup logic here
                    pass
                
                self.log.info("cleanup_completed")
            except Exception as e:
                self.log.error("cleanup_error", error=str(e))
            finally:
                sys.exit(0)
        
        # Register handlers for common termination signals
        signal.signal(signal.SIGTERM, signal_handler)  # Kubernetes sends this
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        
        self.log.debug("graceful_shutdown_handlers_registered")

    def enable_debug_routes(self) -> 'AgentBase':
        """
        Enable debug routes for testing and development
        
        Returns:
            Self for method chaining
        """
        # Debug routes are automatically registered in _register_routes
        # This method exists for backward compatibility
        return self