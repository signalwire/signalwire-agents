#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

# -*- coding: utf-8 -*-
"""
Base SWML Service for SignalWire Agents
"""

import os
import inspect
import json
import secrets
import base64
import logging
import sys
import types
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Type
from urllib.parse import urlparse

# Import centralized logging system
from signalwire_agents.core.logging_config import get_logger

# Create the module logger using centralized system
logger = get_logger("swml_service")

try:
    import fastapi
    from fastapi import FastAPI, APIRouter, Depends, HTTPException, Query, Body, Request, Response
    from fastapi.security import HTTPBasic, HTTPBasicCredentials
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "fastapi is required. Install it with: pip install fastapi"
    )

from signalwire_agents.utils.schema_utils import SchemaUtils
from signalwire_agents.core.swml_handler import VerbHandlerRegistry, SWMLVerbHandler
from signalwire_agents.core.security_config import SecurityConfig


class SWMLService:
    """
    Base class for creating and serving SWML documents.
    
    This class provides core functionality for:
    - Loading and validating SWML schema
    - Creating SWML documents
    - Setting up web endpoints for serving SWML
    - Managing authentication
    - Registering SWML functions
    
    It serves as the foundation for more specialized services like AgentBase.
    """
    
    def __init__(
        self,
        name: str,
        route: str = "/",
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        basic_auth: Optional[Tuple[str, str]] = None,
        schema_path: Optional[str] = None,
        config_file: Optional[str] = None
    ):
        """
        Initialize a new SWML service
        
        Args:
            name: Service name/identifier
            route: HTTP route path for this service
            host: Host to bind the web server to
            port: Port to bind the web server to
            basic_auth: Optional (username, password) tuple for basic auth
            schema_path: Optional path to the schema file
            config_file: Optional path to configuration file
        """
        self.name = name
        self.route = route.rstrip("/")  # Ensure no trailing slash
        self.host = host
        # Use provided port, or PORT env var, or default to 3000
        self.port = port if port is not None else int(os.environ.get("PORT", 3000))
        
        # Initialize logger for this instance FIRST before using it
        self.log = logger.bind(service=name)
        
        # Load unified security configuration with optional config file
        self.security = SecurityConfig(config_file=config_file, service_name=name)
        self.security.log_config("SWMLService")
        
        # For backward compatibility, expose SSL settings as instance attributes
        self.ssl_enabled = self.security.ssl_enabled
        self.domain = self.security.domain
        self.ssl_cert_path = self.security.ssl_cert_path
        self.ssl_key_path = self.security.ssl_key_path
        
        # Initialize proxy detection attributes
        self._proxy_url_base = os.environ.get('SWML_PROXY_URL_BASE')
        self._proxy_url_base_from_env = bool(self._proxy_url_base)  # Track if it came from environment
        if self._proxy_url_base:
            self.log.warning("SWML_PROXY_URL_BASE is set in environment - This overrides SSL configuration and port settings. Remove this variable to use automatic detection.", 
                           proxy_url_base=self._proxy_url_base)
        self._proxy_detection_done = False
        self._proxy_debug = os.environ.get('SWML_PROXY_DEBUG', '').lower() in ('true', '1', 'yes')
        self.log.info("service_initializing", route=self.route, host=self.host, port=self.port)
        
        # Set basic auth credentials
        if basic_auth is not None:
            # Use provided credentials
            self._basic_auth = basic_auth
        else:
            # Use unified security config for auth credentials
            self._basic_auth = self.security.get_basic_auth()
        
        # Find the schema file if not provided
        if schema_path is None:
            schema_path = self._find_schema_path()
            if schema_path:
                self.log.debug("schema_found", path=schema_path)
            else:
                self.log.warning("schema_not_found")
        
        # Initialize schema utils
        self.schema_utils = SchemaUtils(schema_path)
        
        # Initialize verb handler registry
        self.verb_registry = VerbHandlerRegistry()
        
        # Server state
        self._app = None
        self._router = None
        self._running = False
        
        # Initialize SWML document state
        self._current_document = self._create_empty_document()
        
        # Dictionary to cache dynamically created methods (instance level cache)
        self._verb_methods_cache = {}
        
        # Create auto-vivified methods for all verbs
        self._create_verb_methods()
        
        # Initialize routing callbacks dictionary (path -> callback)
        self._routing_callbacks = {}
    
    def _create_verb_methods(self) -> None:
        """
        Create auto-vivified methods for all verbs at initialization time
        """
        self.log.debug("creating_verb_methods")
        
        # Get all verb names from the schema
        if not self.schema_utils:
            self.log.warning("no_schema_utils_available")
            return
            
        verb_names = self.schema_utils.get_all_verb_names()
        self.log.debug("found_verbs_in_schema", count=len(verb_names))
        
        # Create a method for each verb
        for verb_name in verb_names:
            # Skip verbs that already have specific methods
            if hasattr(self, verb_name):
                self.log.debug("skipping_verb_has_method", verb=verb_name)
                continue
            
            # Handle sleep verb specially since it takes an integer directly
            if verb_name == "sleep":
                def sleep_method(self_instance, duration=None, **kwargs):
                    """
                    Add the sleep verb to the document.
                    
                    Args:
                        duration: The amount of time to sleep in milliseconds
                    """
                    self.log.debug("executing_sleep_verb", duration=duration)
                    # Sleep verb takes a direct integer parameter in SWML
                    if duration is not None:
                        return self_instance.add_verb("sleep", duration)
                    elif kwargs:
                        # Try to get the value from kwargs
                        return self_instance.add_verb("sleep", next(iter(kwargs.values())))
                    else:
                        raise TypeError("sleep() missing required argument: 'duration'")
                
                # Set it as an attribute of self
                setattr(self, verb_name, types.MethodType(sleep_method, self))
                
                # Also cache it for later
                self._verb_methods_cache[verb_name] = sleep_method
                
                self.log.debug("created_special_method", verb=verb_name)
                continue
                
            # Generate the method implementation for normal verbs
            def make_verb_method(name):
                def verb_method(self_instance, **kwargs):
                    """
                    Dynamically generated method for SWML verb
                    """
                    self.log.debug("executing_verb_method", verb=name, kwargs_count=len(kwargs))
                    config = {}
                    for key, value in kwargs.items():
                        if value is not None:
                            config[key] = value
                    return self_instance.add_verb(name, config)
                
                # Add docstring to the method
                verb_properties = self.schema_utils.get_verb_properties(name)
                if "description" in verb_properties:
                    verb_method.__doc__ = f"Add the {name} verb to the document.\n\n{verb_properties['description']}"
                else:
                    verb_method.__doc__ = f"Add the {name} verb to the document."
                
                return verb_method
            
            # Create the method with closure over the verb name
            method = make_verb_method(verb_name)
            
            # Set it as an attribute of self
            setattr(self, verb_name, types.MethodType(method, self))
            
            # Also cache it for later
            self._verb_methods_cache[verb_name] = method
            
            self.log.debug("created_verb_method", verb=verb_name)
    
    def __getattr__(self, name: str) -> Any:
        """
        Dynamically generate and return SWML verb methods when accessed
        
        This method is called when an attribute lookup fails through the normal
        mechanisms. It checks if the attribute name corresponds to a SWML verb
        defined in the schema, and if so, dynamically creates a method for that verb.
        
        Args:
            name: The name of the attribute being accessed
            
        Returns:
            The dynamically created verb method if name is a valid SWML verb,
            otherwise raises AttributeError
            
        Raises:
            AttributeError: If name is not a valid SWML verb
        """
        self.log.debug("getattr_called", attribute=name)
        
        # Simple version to match our test script
        # First check if this is a valid SWML verb
        if not self.schema_utils:
            msg = f"'{self.__class__.__name__}' object has no attribute '{name}' (no schema available)"
            self.log.debug("getattr_no_schema", attribute=name)
            raise AttributeError(msg)
            
        verb_names = self.schema_utils.get_all_verb_names()
        
        if name in verb_names:
            self.log.debug("getattr_valid_verb", verb=name)
            
            # Check if we already have this method in the cache
            if not hasattr(self, '_verb_methods_cache'):
                self._verb_methods_cache = {}
                
            if name in self._verb_methods_cache:
                self.log.debug("getattr_cached_method", verb=name)
                return types.MethodType(self._verb_methods_cache[name], self)
            
            # Handle sleep verb specially since it takes an integer directly
            if name == "sleep":
                def sleep_method(self_instance, duration=None, **kwargs):
                    """
                    Add the sleep verb to the document.
                    
                    Args:
                        duration: The amount of time to sleep in milliseconds
                    """
                    self.log.debug("executing_sleep_method", duration=duration)
                    # Sleep verb takes a direct integer parameter in SWML
                    if duration is not None:
                        return self_instance.add_verb("sleep", duration)
                    elif kwargs:
                        # Try to get the value from kwargs
                        return self_instance.add_verb("sleep", next(iter(kwargs.values())))
                    else:
                        raise TypeError("sleep() missing required argument: 'duration'")
                
                # Cache the method for future use
                self.log.debug("caching_sleep_method", verb=name)
                self._verb_methods_cache[name] = sleep_method
                
                # Return the bound method
                return types.MethodType(sleep_method, self)
            
            # Generate the method implementation for normal verbs
            def verb_method(self_instance, **kwargs):
                """
                Dynamically generated method for SWML verb
                """
                self.log.debug("executing_dynamic_verb", verb=name, kwargs_count=len(kwargs))
                config = {}
                for key, value in kwargs.items():
                    if value is not None:
                        config[key] = value
                return self_instance.add_verb(name, config)
            
            # Add docstring to the method
            verb_properties = self.schema_utils.get_verb_properties(name)
            if "description" in verb_properties:
                verb_method.__doc__ = f"Add the {name} verb to the document.\n\n{verb_properties['description']}"
            else:
                verb_method.__doc__ = f"Add the {name} verb to the document."
            
            # Cache the method for future use
            self.log.debug("caching_verb_method", verb=name)
            self._verb_methods_cache[name] = verb_method
            
            # Return the bound method
            return types.MethodType(verb_method, self)
        
        # Not a valid verb
        msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
        self.log.debug("getattr_invalid_attribute", attribute=name, error=msg)
        raise AttributeError(msg)
    
    def _find_schema_path(self) -> Optional[str]:
        """
        Find the schema.json file location
        
        Returns:
            Path to schema.json if found, None otherwise
        """
        # Try package resources first (most reliable after pip install)
        try:
            import importlib.resources
            try:
                # Python 3.9+
                try:
                    # Python 3.13+
                    path = importlib.resources.files("signalwire_agents").joinpath("schema.json")
                    return str(path)
                except Exception:
                    # Python 3.9-3.12
                    with importlib.resources.files("signalwire_agents").joinpath("schema.json") as path:
                        return str(path)
            except AttributeError:
                # Python 3.7-3.8
                with importlib.resources.path("signalwire_agents", "schema.json") as path:
                    return str(path)
        except (ImportError, ModuleNotFoundError):
            pass
            
        # Fall back to pkg_resources for older Python or alternative lookup
        try:
            import pkg_resources
            return pkg_resources.resource_filename("signalwire_agents", "schema.json")
        except (ImportError, ModuleNotFoundError, pkg_resources.DistributionNotFound):
            pass

        # Fall back to manual search in various locations
        import sys
        
        # Get package directory
        package_dir = os.path.dirname(os.path.dirname(__file__))
        
        # Potential locations for schema.json
        potential_paths = [
            os.path.join(os.getcwd(), "schema.json"),  # Current working directory
            os.path.join(package_dir, "schema.json"),  # Package directory
            os.path.join(os.path.dirname(package_dir), "schema.json"),  # Parent of package directory
            os.path.join(sys.prefix, "schema.json"),  # Python installation directory
            os.path.join(package_dir, "data", "schema.json"),  # Data subdirectory
            os.path.join(os.path.dirname(package_dir), "data", "schema.json"),  # Parent's data subdirectory
        ]
        
        # Try to find the schema file
        for path in potential_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _create_empty_document(self) -> Dict[str, Any]:
        """
        Create an empty SWML document
        
        Returns:
            Empty SWML document structure
        """
        return {
            "version": "1.0.0",
            "sections": {
                "main": []
            }
        }
    
    def reset_document(self) -> None:
        """
        Reset the current document to an empty state
        """
        self._current_document = self._create_empty_document()
    
    def add_verb(self, verb_name: str, config: Union[Dict[str, Any], int]) -> bool:
        """
        Add a verb to the main section of the current document
        
        Args:
            verb_name: The name of the verb to add
            config: Configuration for the verb or direct value for certain verbs (e.g., sleep)
            
        Returns:
            True if the verb was added successfully, False otherwise
        """
        # Special case for verbs that take direct values (like sleep)
        if verb_name == "sleep" and isinstance(config, int):
            # Sleep verb takes a direct integer value
            verb_obj = {verb_name: config}
            self._current_document["sections"]["main"].append(verb_obj)
            return True
            
        # Ensure config is a dictionary for other verbs
        if not isinstance(config, dict):
            self.log.warning(f"invalid_config_type", verb=verb_name, 
                            expected="dict", got=type(config).__name__)
            return False
        
        # Check if we have a specialized handler for this verb
        if self.verb_registry.has_handler(verb_name):
            handler = self.verb_registry.get_handler(verb_name)
            is_valid, errors = handler.validate_config(config)
        else:
            # Use schema-based validation for standard verbs
            is_valid, errors = self.schema_utils.validate_verb(verb_name, config)
        
        if not is_valid:
            # Log validation errors
            self.log.warning(f"verb_validation_error", verb=verb_name, errors=errors)
            return False
        
        # Add the verb to the main section
        verb_obj = {verb_name: config}
        self._current_document["sections"]["main"].append(verb_obj)
        return True
    
    def add_section(self, section_name: str) -> bool:
        """
        Add a new section to the document
        
        Args:
            section_name: Name of the section to add
            
        Returns:
            True if the section was added, False if it already exists
        """
        if section_name in self._current_document["sections"]:
            return False
        
        self._current_document["sections"][section_name] = []
        return True
    
    def add_verb_to_section(self, section_name: str, verb_name: str, config: Union[Dict[str, Any], int]) -> bool:
        """
        Add a verb to a specific section
        
        Args:
            section_name: Name of the section to add to
            verb_name: The name of the verb to add
            config: Configuration for the verb or direct value for certain verbs (e.g., sleep)
            
        Returns:
            True if the verb was added successfully, False otherwise
        """
        # Make sure the section exists
        if section_name not in self._current_document["sections"]:
            self.add_section(section_name)
        
        # Special case for verbs that take direct values (like sleep)
        if verb_name == "sleep" and isinstance(config, int):
            # Sleep verb takes a direct integer value
            verb_obj = {verb_name: config}
            self._current_document["sections"][section_name].append(verb_obj)
            return True
            
        # Ensure config is a dictionary for other verbs
        if not isinstance(config, dict):
            self.log.warning(f"invalid_config_type", verb=verb_name, section=section_name,
                            expected="dict", got=type(config).__name__)
            return False
        
        # Check if we have a specialized handler for this verb
        if self.verb_registry.has_handler(verb_name):
            handler = self.verb_registry.get_handler(verb_name)
            is_valid, errors = handler.validate_config(config)
        else:
            # Use schema-based validation for standard verbs
            is_valid, errors = self.schema_utils.validate_verb(verb_name, config)
        
        if not is_valid:
            # Log validation errors
            self.log.warning(f"verb_validation_error", verb=verb_name, section=section_name, errors=errors)
            return False
        
        # Add the verb to the section
        verb_obj = {verb_name: config}
        self._current_document["sections"][section_name].append(verb_obj)
        return True
    
    def get_document(self) -> Dict[str, Any]:
        """
        Get the current SWML document
        
        Returns:
            The current SWML document as a dictionary
        """
        return self._current_document
    
    def render_document(self) -> str:
        """
        Render the current SWML document as a JSON string
        
        Returns:
            The current SWML document as a JSON string
        """
        return json.dumps(self._current_document)
    
    def register_verb_handler(self, handler: SWMLVerbHandler) -> None:
        """
        Register a custom verb handler
        
        Args:
            handler: The verb handler to register
        """
        self.verb_registry.register_handler(handler)
    
    def as_router(self) -> APIRouter:
        """
        Create a FastAPI router for this service
        
        Returns:
            APIRouter: FastAPI router
        """
        router = APIRouter(redirect_slashes=False)
        
        # Root endpoint with and without trailing slash
        @router.get("/")
        @router.post("/")
        async def handle_root(request: Request, response: Response):
            """Handle requests to the root endpoint"""
            return await self._handle_request(request, response)
        
        # Register routing callbacks as needed
        if hasattr(self, '_routing_callbacks') and self._routing_callbacks:
            for callback_path, callback_fn in self._routing_callbacks.items():
                # Skip the root path which is already handled
                if callback_path == "/":
                    continue
                    
                # Register both versions: with and without trailing slash
                path = callback_path.rstrip("/")
                path_with_slash = f"{path}/"
                
                @router.get(path)
                @router.get(path_with_slash)
                @router.post(path)
                @router.post(path_with_slash)
                async def handle_callback(request: Request, response: Response, cb_path=callback_path):
                    """Handle requests to callback endpoints"""
                    # Store the callback path in the request state
                    request.state.callback_path = cb_path
                    return await self._handle_request(request, response)
        
        return router
    
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
        # Normalize the path to ensure consistent lookup
        normalized_path = path.rstrip("/")
        if not normalized_path.startswith("/"):
            normalized_path = f"/{normalized_path}"
            
        self.log.info("registering_routing_callback", path=normalized_path)
        self._routing_callbacks[normalized_path] = callback_fn

    @staticmethod
    def extract_sip_username(request_body: Dict[str, Any]) -> Optional[str]:
        """
        Extract SIP username from request body
        
        This extracts the username portion of a SIP URI from the 'to' field
        in the call data of a request body.
        
        Args:
            request_body: The parsed JSON body of the request
            
        Returns:
            The extracted SIP username, or None if not found
        """
        try:
            # Check if we have call data with a 'to' field
            if "call" in request_body and "to" in request_body["call"]:
                to_field = request_body["call"]["to"]
                
                # Handle SIP URIs like "sip:username@domain"
                if to_field.startswith("sip:"):
                    # Extract username part (between "sip:" and "@")
                    uri_parts = to_field[4:].split("@", 1)
                    if uri_parts:
                        return uri_parts[0]
                        
                # Handle TEL URIs like "tel:+1234567890"
                elif to_field.startswith("tel:"):
                    # Extract phone number part
                    return to_field[4:]
                    
                # Otherwise, return the whole 'to' field
                else:
                    return to_field
        except (KeyError, AttributeError):
            # If any exception occurs during extraction, return None
            pass
            
        return None
        
    async def _handle_request(self, request: Request, response: Response):
        """
        Internal handler for both GET and POST requests
        
        Args:
            request: FastAPI Request object
            response: FastAPI Response object
            
        Returns:
            Response with SWML document or error
        """
        # Always detect proxy from current request - allows mixing direct and proxied access
        self._detect_proxy_from_request(request)
            
        # Check auth
        if not self._check_basic_auth(request):
            response.headers["WWW-Authenticate"] = "Basic"
            return HTTPException(status_code=401, detail="Unauthorized")
        
        # Get callback path from request state
        callback_path = getattr(request.state, "callback_path", None)
        
        # Process request body if it's a POST
        body = {}
        if request.method == "POST":
            try:
                raw_body = await request.body()
                if raw_body:
                    body = await request.json()
                    
                    # Check if this is a callback path and we have a callback registered for it
                    if callback_path and hasattr(self, '_routing_callbacks') and callback_path in self._routing_callbacks:
                        callback_fn = self._routing_callbacks[callback_path]
                        self.log.debug("checking_routing", 
                                      path=callback_path, 
                                      body_keys=list(body.keys()))
                        
                        # Call the callback function
                        try:
                            route = callback_fn(request, body)
                            
                            if route is not None:
                                self.log.info("routing_request", route=route)
                                # We should return a redirect to the new route
                                # Use 307 to preserve the POST method and its body
                                response = Response(status_code=307)
                                response.headers["Location"] = route
                                return response
                        except Exception as e:
                            self.log.error("error_in_routing_callback", error=str(e))
            except Exception as e:
                self.log.error("error_parsing_body", error=str(e))
                # Continue with empty body if parsing fails
                pass
        
        # Allow for customized handling in subclasses
        modifications = self.on_request(body, callback_path)
        
        # Apply any modifications if needed
        if modifications and isinstance(modifications, dict):
            # Get a copy of the current document
            document = self.get_document()
            
            # Apply modifications (simplified implementation)
            # In a real implementation, you might want a more sophisticated merge strategy
            for key, value in modifications.items():
                if key in document:
                    document[key] = value
            
            # Create a new document with the modifications
            modified_doc = json.dumps(document)
            return Response(content=modified_doc, media_type="application/json")
        
        # Get the current SWML document
        swml = self.render_document()
        
        # Return the SWML document
        return Response(content=swml, media_type="application/json")
    
    def on_request(self, request_data: Optional[dict] = None, callback_path: Optional[str] = None) -> Optional[dict]:
        """
        Called when SWML is requested, with request data when available
        
        Subclasses can override this to inspect or modify SWML based on the request
        
        Args:
            request_data: Optional dictionary containing the parsed POST body
            callback_path: Optional callback path
            
        Returns:
            Optional dict to modify/augment the SWML document
        """
        # Default implementation does nothing
        return None
    
    def serve(self, host: Optional[str] = None, port: Optional[int] = None, 
              ssl_cert: Optional[str] = None, ssl_key: Optional[str] = None, 
              ssl_enabled: Optional[bool] = None, domain: Optional[str] = None) -> None:
        """
        Start a web server for this service
        
        Args:
            host: Host to bind to (defaults to self.host)
            port: Port to bind to (defaults to self.port)
            ssl_cert: Path to SSL certificate file
            ssl_key: Path to SSL key file
            ssl_enabled: Whether to enable SSL
            domain: Domain name for SSL certificate
        """
        import uvicorn
        
        # Store SSL configuration (override environment if explicitly provided)
        if ssl_enabled is not None:
            self.ssl_enabled = ssl_enabled
        if domain is not None:
            self.domain = domain
        
        # Set SSL paths (use provided paths or fall back to environment)
        ssl_cert_path = ssl_cert or getattr(self, 'ssl_cert_path', None)
        ssl_key_path = ssl_key or getattr(self, 'ssl_key_path', None)
        
        # Validate SSL configuration if enabled
        if self.ssl_enabled:
            is_valid, error = self.security.validate_ssl_config()
            if not is_valid:
                self.log.warning("ssl_config_invalid", error=error)
                self.ssl_enabled = False
            elif not self.domain:
                self.log.warning("ssl_domain_not_specified")
                # We'll continue, but URLs might not be correctly generated
        
        if self._app is None:
            # Use redirect_slashes=False to be consistent with AgentBase
            app = FastAPI(redirect_slashes=False)
            router = self.as_router()
            
            # Normalize the route to ensure it starts with a slash and doesn't end with one
            # This avoids the FastAPI error about prefixes ending with slashes
            normalized_route = "/" + self.route.strip("/")
            
            # Include router with the normalized prefix (handle root route special case)
            if normalized_route == "/":
                app.include_router(router)
            else:
                app.include_router(router, prefix=normalized_route)
            
            # Add a catch-all route handler that will handle both /path and /path/ formats
            # This provides the same behavior without using a trailing slash in the prefix
            @app.get("/{full_path:path}")
            @app.post("/{full_path:path}")
            async def handle_all_routes(request: Request, response: Response, full_path: str):
                # Get our route path without leading slash for comparison
                route_path = normalized_route.lstrip("/")
                route_with_slash = route_path + "/"
                
                # Log the incoming path for debugging
                self.log.debug("catch_all_route_hit", path=full_path, route=route_path)
                
                # Check for exact match to our route (without trailing slash)
                if full_path == route_path:
                    # This is our exact route - handle it directly
                    return await self._handle_request(request, response)
                    
                # Check for our route with a trailing slash or subpaths
                elif full_path == route_with_slash or full_path.startswith(route_with_slash):
                    # This is our route with a trailing slash
                    # Extract the path after our route prefix
                    sub_path = full_path[len(route_with_slash):]
                    
                    # Forward to the appropriate handler in our router
                    if not sub_path:
                        # Root endpoint
                        return await self._handle_request(request, response)
                    
                    # Check for routing callbacks if there are any
                    if hasattr(self, '_routing_callbacks'):
                        for callback_path, callback_fn in self._routing_callbacks.items():
                            cb_path_clean = callback_path.strip("/")
                            if sub_path == cb_path_clean or sub_path.startswith(cb_path_clean + "/"):
                                # Store the callback path in request state for handlers to use
                                request.state.callback_path = callback_path
                                return await self._handle_request(request, response)
                
                # Not our route or not matching our patterns
                self.log.debug("no_route_match", path=full_path)
                return {"error": "Path not found"}
            
            # Log all routes for debugging
            self.log.debug("registered_routes", service=self.name)
            for route in app.routes:
                if hasattr(route, "path"):
                    self.log.debug("route_registered", path=route.path)
            
            self._app = app
        
        host = host or self.host
        port = port or self.port
        
        # Get the auth credentials
        username, password = self._basic_auth
        
        # Get the proper URL using unified URL building
        startup_url = self._build_full_url(include_auth=False)
        
        self.log.info("starting_server", 
                     url=startup_url,
                     ssl_enabled=self.ssl_enabled,
                     username=username,
                     password_length=len(password))
        
        # Print user-friendly startup message (keep for UX)
        print(f"Service '{self.name}' is available at:")
        print(f"URL: {startup_url}")
        print(f"URL with trailing slash: {startup_url}/")
        print(f"Basic Auth: {username}:{password}")
        
        # Check if SIP routing is enabled and log additional info
        if self._routing_callbacks:
            print(f"Callback endpoints:")
            for path in self._routing_callbacks:
                callback_url = self._build_full_url(endpoint=path.lstrip('/'), include_auth=False)
                print(f"  {callback_url}")
        
        # Start uvicorn with or without SSL
        if self.ssl_enabled and ssl_cert_path and ssl_key_path:
            self.log.info("starting_with_ssl", cert=ssl_cert_path, key=ssl_key_path)
            uvicorn.run(
                self._app, 
                host=host, 
                port=port,
                ssl_certfile=ssl_cert_path,
                ssl_keyfile=ssl_key_path
            )
        else:
            uvicorn.run(self._app, host=host, port=port)
    
    def stop(self) -> None:
        """
        Stop the web server
        """
        self._running = False
    
    def _check_basic_auth(self, request: Request) -> bool:
        """
        Check if the request has valid basic auth credentials
        
        Args:
            request: FastAPI Request object
            
        Returns:
            True if auth is valid, False otherwise
        """
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return False
        
        # Extract the credentials from the header
        try:
            scheme, credentials = auth_header.split()
            if scheme.lower() != "basic":
                return False
            
            decoded = base64.b64decode(credentials).decode("utf-8")
            username, password = decoded.split(":")
            
            # Compare with our credentials
            expected_username, expected_password = self._basic_auth
            return username == expected_username and password == expected_password
        except Exception:
            return False
    
    def get_basic_auth_credentials(self, include_source: bool = False) -> Union[Tuple[str, str], Tuple[str, str, str]]:
        """
        Get the basic auth credentials
        
        Args:
            include_source: Whether to include the source of the credentials
            
        Returns:
            (username, password) tuple or (username, password, source) tuple if include_source is True
        """
        username, password = self._basic_auth
        
        if include_source:
            # Determine source
            env_user = os.environ.get('SWML_BASIC_AUTH_USER')
            env_pass = os.environ.get('SWML_BASIC_AUTH_PASSWORD')
            
            if env_user and env_pass and env_user == username and env_pass == password:
                source = "environment"
            else:
                source = "auto-generated"
                
            return username, password, source
        
        return username, password
    
        
    def _get_base_url(self, include_auth: bool = True) -> str:
        """
        Get the base URL for this service, using proxy info if available or falling back to configured values
        
        This is the central method for URL building that handles both startup configuration
        and per-request proxy detection.
        
        Args:
            include_auth: Whether to include authentication credentials in the URL
            
        Returns:
            Base URL string (protocol://[auth@]host[:port])
        """
        # Debug logging to understand state
        self.log.debug("_get_base_url called",
                      has_proxy_url_base=hasattr(self, '_proxy_url_base'),
                      proxy_url_base=getattr(self, '_proxy_url_base', None),
                      proxy_url_base_from_env=getattr(self, '_proxy_url_base_from_env', False),
                      env_var=os.environ.get('SWML_PROXY_URL_BASE'),
                      include_auth=include_auth,
                      caller=inspect.stack()[1].function if len(inspect.stack()) > 1 else "unknown")
        
        # Check if we have proxy information from a request
        if hasattr(self, '_proxy_url_base') and self._proxy_url_base:
            base = self._proxy_url_base.rstrip('/')
            self.log.debug("Using proxy URL base", proxy_url_base=base)

            # Add auth credentials if requested
            if include_auth:
                username, password = self._basic_auth
                url = urlparse(base)
                base = url._replace(netloc=f"{username}:{password}@{url.netloc}").geturl()
            
            return base
        
        # No proxy, use configured values
        # Determine protocol based on SSL settings
        protocol = "https" if self.ssl_enabled else "http"
        
        # Debug logging
        self.log.debug("_get_base_url",
                      ssl_enabled=self.ssl_enabled,
                      domain=self.domain,
                      port=self.port,
                      protocol=protocol)
        
        # Determine host part
        if self.ssl_enabled and self.domain:
            # Use domain for SSL
            if protocol == "https" and self.port == 443:
                host_part = self.domain  # Don't include port for standard HTTPS
            elif protocol == "http" and self.port == 80:
                host_part = self.domain  # Don't include port for standard HTTP
            else:
                host_part = f"{self.domain}:{self.port}"
                self.log.debug("Using domain with port", domain=self.domain, port=self.port, host_part=host_part)
        else:
            # Use configured host
            if self.host in ("0.0.0.0", "127.0.0.1", "localhost"):
                host = "localhost"
            else:
                host = self.host
            
            # Include port unless it's the standard port for the protocol
            if (protocol == "https" and self.port == 443) or (protocol == "http" and self.port == 80):
                host_part = host
            else:
                host_part = f"{host}:{self.port}"
        
        # Build base URL
        if include_auth:
            username, password = self._basic_auth
            base = f"{protocol}://{username}:{password}@{host_part}"
        else:
            base = f"{protocol}://{host_part}"
        
        return base
    
    def _build_full_url(self, endpoint: str = "", include_auth: bool = True, query_params: Optional[Dict[str, str]] = None) -> str:
        """
        Build the full URL for this service or a specific endpoint
        
        This is the internal implementation used by both get_full_url (for AgentBase compatibility)
        and _build_webhook_url.
        
        Args:
            endpoint: Optional endpoint path (e.g., "swaig", "post_prompt")
            include_auth: Whether to include authentication credentials in the URL
            query_params: Optional query parameters to append
            
        Returns:
            Full URL string
        """
        # Get base URL using central method
        base = self._get_base_url(include_auth=include_auth)

        # Build path
        if endpoint:
            # Ensure endpoint doesn't start with slash
            endpoint = endpoint.lstrip('/')
            # Add trailing slash to endpoint to prevent redirects
            if not endpoint.endswith('/'):
                endpoint = f"{endpoint}/"
            path = f"{self.route}/{endpoint}"
        else:
            # Just the route itself
            path = self.route if self.route != "/" else ""
        
        # Construct full URL
        url = f"{base}{path}"
        
        # Add query parameters if any
        if query_params:
            filtered_params = {k: v for k, v in query_params.items() if v}
            if filtered_params:
                params = "&".join([f"{k}={v}" for k, v in filtered_params.items()])
                url = f"{url}?{params}"
        
        return url
    
    def _build_webhook_url(self, endpoint: str, query_params: Optional[Dict[str, str]] = None) -> str:
        """
        Helper method to build webhook URLs consistently
        
        Args:
            endpoint: The endpoint path (e.g., "swaig", "post_prompt")
            query_params: Optional query parameters to append
            
        Returns:
            Fully constructed webhook URL
        """
        self.log.debug("_build_webhook_url called",
                      endpoint=endpoint,
                      query_params=query_params,
                      proxy_url_base=getattr(self, '_proxy_url_base', None),
                      proxy_url_base_from_env=getattr(self, '_proxy_url_base_from_env', False))
        
        # Use the central URL building method
        return self._build_full_url(endpoint=endpoint, include_auth=True, query_params=query_params) 

    def _detect_proxy_from_request(self, request: Request) -> None:
        """
        Detect if we're behind a proxy by examining request headers
        and auto-configure proxy_url_base if needed
        
        Args:
            request: FastAPI Request object
        """
        # If SWML_PROXY_URL_BASE was already set (e.g., from environment), don't override it
        if self._proxy_url_base:
            return
            
        # First check for standard X-Forwarded headers (used by most proxies including ngrok)
        forwarded_host = request.headers.get("X-Forwarded-Host")
        forwarded_proto = request.headers.get("X-Forwarded-Proto", "http")
        
        if forwarded_host:
            # Direct X-Forwarded-* headers - most common case
            self._proxy_url_base = f"{forwarded_proto}://{forwarded_host}"
            self.log.info("proxy_auto_detected", proxy_url_base=self._proxy_url_base, 
                        source="X-Forwarded headers")
            return
            
        # If no standard headers, check other proxy detection methods
        
        # Check for Forwarded header (RFC 7239)
        forwarded = request.headers.get("Forwarded")
        if forwarded:
            # Parse RFC 7239 Forwarded header
            try:
                # Extract host and proto from Forwarded: for=X;host=Y;proto=Z
                parts = [p.strip() for p in forwarded.split(';')]
                host_part = next((p for p in parts if p.startswith("host=")), None)
                proto_part = next((p for p in parts if p.startswith("proto=")), None)
                
                if host_part:
                    host = host_part.split('=', 1)[1].strip('"')
                    proto = proto_part.split('=', 1)[1].strip('"') if proto_part else "http"
                    self._proxy_url_base = f"{proto}://{host}"
                    self.log.info("proxy_auto_detected", proxy_url_base=self._proxy_url_base,
                                source="Forwarded header")
                    return
            except Exception as e:
                self.log.warning("forwarded_header_parse_error", error=str(e))
        
        # Try to detect from the URL itself for transparent proxies
        if str(request.url).startswith(("https://", "http://")) and not any(
            str(request.url).startswith(f"http://{h}") for h in ["localhost", "127.0.0.1", self.host, "0.0.0.0"]
        ):
            # This is likely a transparent proxy - extract base URL
            parsed = urlparse(str(request.url))
            base_url = f"{parsed.scheme}://{parsed.netloc}"
            self._proxy_url_base = base_url
            self.log.info("proxy_auto_detected", proxy_url_base=base_url,
                        source="request URL (transparent proxy)")
            return
        
        # Check for other common proxy setups
        original_host = request.headers.get("X-Original-Host") or request.headers.get("Host")
        if original_host:
            # Only use Host if it doesn't look like our local server
            local_hosts = [self.host, "localhost", "127.0.0.1", "0.0.0.0"]
            local_port = f":{self.port}"
            
            # If host doesn't look like local server or doesn't contain our port
            if not any(h in original_host for h in local_hosts) and local_port not in original_host:
                proto = "https" if request.url.scheme == "https" else "http"
                self._proxy_url_base = f"{proto}://{original_host}"
                self.log.info("proxy_auto_detected", proxy_url_base=self._proxy_url_base,
                            source="Host header")
                return
        
        # If forward_for header exists, we're likely behind a proxy but couldn't determine the URL
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            self.log.warning("proxy_detected_but_url_unknown", 
                          client_ip=forwarded_for,
                          message="Proxy detected via X-Forwarded-For header but could not determine public URL")
            
        # No proxy detected, or unable to determine the public URL
        if self._proxy_debug:
            self.log.info("proxy_detection_failed", 
                        message="Could not auto-detect proxy. If you are behind a proxy, set SWML_PROXY_URL_BASE manually.")
    
    def manual_set_proxy_url(self, proxy_url: str) -> None:
        """
        Manually set the proxy URL base for webhook callbacks
        
        This can be called at runtime to set or update the proxy URL
        
        Args:
            proxy_url: The base URL to use for webhooks (e.g., https://example.ngrok.io)
        """
        if proxy_url:
            self._proxy_url_base = proxy_url.rstrip('/')
            self.log.info("proxy_url_manually_set", proxy_url_base=self._proxy_url_base)
            self._proxy_detection_done = True 
