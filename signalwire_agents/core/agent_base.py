#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

# -*- coding: utf-8 -*-
"""
Base class for all SignalWire AI Agents
"""

import os
import json
import time
import uuid
import base64
import logging
import inspect
import functools
import re
import signal
import sys
from typing import Optional, Union, List, Dict, Any, Tuple, Callable, Type
from urllib.parse import urlparse, urlencode, urlunparse

try:
    import fastapi
    from fastapi import FastAPI, APIRouter, Depends, HTTPException, Query, Body, Request, Response
    from fastapi.security import HTTPBasic, HTTPBasicCredentials
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "fastapi is required. Install it with: pip install fastapi"
    )

try:
    import uvicorn
except ImportError:
    raise ImportError(
        "uvicorn is required. Install it with: pip install uvicorn"
    )



from signalwire_agents.core.pom_builder import PomBuilder
from signalwire_agents.core.swaig_function import SWAIGFunction
from signalwire_agents.core.function_result import SwaigFunctionResult
from signalwire_agents.core.swml_renderer import SwmlRenderer
from signalwire_agents.core.security.session_manager import SessionManager
from signalwire_agents.core.state import StateManager, FileStateManager
from signalwire_agents.core.swml_service import SWMLService
from signalwire_agents.core.swml_handler import AIVerbHandler
from signalwire_agents.core.skill_manager import SkillManager
from signalwire_agents.utils.schema_utils import SchemaUtils
from signalwire_agents.core.logging_config import get_logger, get_execution_mode

# Import refactored components
from signalwire_agents.core.agent.config.ephemeral import EphemeralAgentConfig
from signalwire_agents.core.agent.prompt.manager import PromptManager
from signalwire_agents.core.agent.tools.registry import ToolRegistry
from signalwire_agents.core.agent.tools.decorator import ToolDecorator

# Create a logger using centralized system
logger = get_logger("agent_base")


class AgentBase(SWMLService):
    """
    Base class for all SignalWire AI Agents.
    
    This class extends SWMLService and provides enhanced functionality for building agents including:
    - Prompt building and customization
    - SWML rendering
    - SWAIG function definition and execution
    - Web service for serving SWML and handling webhooks
    - Security and session management
    
    Subclassing options:
    1. Simple override of get_prompt() for raw text
    2. Using prompt_* methods for structured prompts
    3. Declarative PROMPT_SECTIONS class attribute
    """
    
    # Subclasses can define this to declaratively set prompt sections
    PROMPT_SECTIONS = None
    
    def __init__(
        self,
        name: str,
        route: str = "/",
        host: str = "0.0.0.0",
        port: int = 3000,
        basic_auth: Optional[Tuple[str, str]] = None,
        use_pom: bool = True,
        enable_state_tracking: bool = False,
        token_expiry_secs: int = 3600,
        auto_answer: bool = True,
        record_call: bool = False,
        record_format: str = "mp4",
        record_stereo: bool = True,
        state_manager: Optional[StateManager] = None,
        default_webhook_url: Optional[str] = None,
        agent_id: Optional[str] = None,
        native_functions: Optional[List[str]] = None,
        schema_path: Optional[str] = None,
        suppress_logs: bool = False,
        enable_post_prompt_override: bool = False,
        check_for_input_override: bool = False
    ):
        """
        Initialize a new agent
        
        Args:
            name: Agent name/identifier
            route: HTTP route path for this agent
            host: Host to bind the web server to
            port: Port to bind the web server to
            basic_auth: Optional (username, password) tuple for basic auth
            use_pom: Whether to use POM for prompt building
            enable_state_tracking: Whether to register startup_hook and hangup_hook SWAIG functions to track conversation state
            token_expiry_secs: Seconds until tokens expire
            auto_answer: Whether to automatically answer calls
            record_call: Whether to record calls
            record_format: Recording format
            record_stereo: Whether to record in stereo
            state_manager: Optional state manager for this agent
            default_webhook_url: Optional default webhook URL for all SWAIG functions
            agent_id: Optional unique ID for this agent, generated if not provided
            native_functions: Optional list of native functions to include in the SWAIG object
            schema_path: Optional path to the schema file
            suppress_logs: Whether to suppress structured logs
            enable_post_prompt_override: Whether to enable post-prompt override
            check_for_input_override: Whether to enable check-for-input override
        """
        # Import SWMLService here to avoid circular imports
        from signalwire_agents.core.swml_service import SWMLService
        
        # If schema_path is not provided, we'll let SWMLService find it through its _find_schema_path method
        # which will be called in its __init__
        
        # Initialize the SWMLService base class
        super().__init__(
            name=name,
            route=route,
            host=host,
            port=port,
            basic_auth=basic_auth,
            schema_path=schema_path
        )
        
        # Log the schema path if found and not suppressing logs
        if self.schema_utils and self.schema_utils.schema_path and not suppress_logs:
            self.log.debug("using_schema_path", path=self.schema_utils.schema_path)
        
        # Setup logger for this instance
        self.log = logger.bind(agent=name)
        self.log.info("agent_initializing", route=route, host=host, port=port)
        
        # Store agent-specific parameters
        self._default_webhook_url = default_webhook_url
        self._suppress_logs = suppress_logs
        
        # Generate or use the provided agent ID
        self.agent_id = agent_id or str(uuid.uuid4())
        
        # Check for proxy URL base in environment
        self._proxy_url_base = os.environ.get('SWML_PROXY_URL_BASE')
        
        # Initialize prompt handling
        self._use_pom = use_pom
        
        # Initialize POM if needed
        if self._use_pom:
            try:
                from signalwire_pom.pom import PromptObjectModel
                self.pom = PromptObjectModel()
            except ImportError:
                raise ImportError(
                    "signalwire-pom package is required for use_pom=True. "
                    "Install it with: pip install signalwire-pom"
                )
        else:
            self.pom = None
        
        # Initialize tool registry (separate from SWMLService verb registry)
        
        # Initialize session manager
        self._session_manager = SessionManager(token_expiry_secs=token_expiry_secs)
        self._enable_state_tracking = enable_state_tracking
        
        # URL override variables
        self._web_hook_url_override = None
        self._post_prompt_url_override = None
        
        # Register the tool decorator on this instance
        self.tool = self._tool_decorator
        
        # Call settings
        self._auto_answer = auto_answer
        self._record_call = record_call
        self._record_format = record_format
        self._record_stereo = record_stereo
        
        # Initialize refactored managers early
        self._prompt_manager = PromptManager(self)
        self._tool_registry = ToolRegistry(self)
        
        # Process declarative PROMPT_SECTIONS if defined in subclass
        self._process_prompt_sections()
        
        # Initialize state manager
        self._state_manager = state_manager or FileStateManager()
        
        # Process class-decorated tools (using @AgentBase.tool)
        self._tool_registry.register_class_decorated_tools()
        
        # Add native_functions parameter
        self.native_functions = native_functions or []
        
        # Register state tracking tools if enabled
        if enable_state_tracking:
            self._register_state_tracking_tools()
        
        # Initialize new configuration containers
        self._hints = []
        self._languages = []
        self._pronounce = []
        self._params = {}
        self._global_data = {}
        self._function_includes = []
        
        # Dynamic configuration callback
        self._dynamic_config_callback = None
        
        # Initialize skill manager
        self.skill_manager = SkillManager(self)
        
        # Initialize contexts system
        self._contexts_builder = None
        self._contexts_defined = False
        
        self.schema_utils = SchemaUtils(schema_path)
        if self.schema_utils and self.schema_utils.schema:
            self.log.debug("schema_loaded", path=self.schema_utils.schema_path)
        
    
    def _process_prompt_sections(self):
        """
        Process declarative PROMPT_SECTIONS attribute from a subclass
        
        This auto-vivifies section methods and bootstraps the prompt
        from class declaration, allowing for declarative agents.
        """
        # Skip if no PROMPT_SECTIONS defined or not using POM
        cls = self.__class__
        if not hasattr(cls, 'PROMPT_SECTIONS') or cls.PROMPT_SECTIONS is None or not self._use_pom:
            return
            
        sections = cls.PROMPT_SECTIONS
        
        # If sections is a dictionary mapping section names to content
        if isinstance(sections, dict):
            for title, content in sections.items():
                # Handle different content types
                if isinstance(content, str):
                    # Plain text - add as body
                    self.prompt_add_section(title, body=content)
                elif isinstance(content, list) and content:  # Only add if non-empty
                    # List of strings - add as bullets
                    self.prompt_add_section(title, bullets=content)
                elif isinstance(content, dict):
                    # Dictionary with body/bullets/subsections
                    body = content.get('body', '')
                    bullets = content.get('bullets', [])
                    numbered = content.get('numbered', False)
                    numbered_bullets = content.get('numberedBullets', False)
                    
                    # Only create section if it has content
                    if body or bullets or 'subsections' in content:
                        # Create the section
                        self.prompt_add_section(
                            title, 
                            body=body, 
                            bullets=bullets if bullets else None,
                            numbered=numbered,
                            numbered_bullets=numbered_bullets
                        )
                        
                        # Process subsections if any
                        subsections = content.get('subsections', [])
                        for subsection in subsections:
                            if 'title' in subsection:
                                sub_title = subsection['title']
                                sub_body = subsection.get('body', '')
                                sub_bullets = subsection.get('bullets', [])
                                
                                # Only add subsection if it has content
                                if sub_body or sub_bullets:
                                    self.prompt_add_subsection(
                                        title, 
                                        sub_title,
                                        body=sub_body,
                                        bullets=sub_bullets if sub_bullets else None
                                    )
        # If sections is a list of section objects, use the POM format directly
        elif isinstance(sections, list):
            if self.pom:
                # Process each section using auto-vivifying methods
                for section in sections:
                    if 'title' in section:
                        title = section['title']
                        body = section.get('body', '')
                        bullets = section.get('bullets', [])
                        numbered = section.get('numbered', False)
                        numbered_bullets = section.get('numberedBullets', False)
                        
                        # Only create section if it has content
                        if body or bullets or 'subsections' in section:
                            self.prompt_add_section(
                                title,
                                body=body,
                                bullets=bullets if bullets else None,
                                numbered=numbered,
                                numbered_bullets=numbered_bullets
                            )
                            
                            # Process subsections if any
                            subsections = section.get('subsections', [])
                            for subsection in subsections:
                                if 'title' in subsection:
                                    sub_title = subsection['title']
                                    sub_body = subsection.get('body', '')
                                    sub_bullets = subsection.get('bullets', [])
                                    
                                    # Only add subsection if it has content
                                    if sub_body or sub_bullets:
                                        self.prompt_add_subsection(
                                            title,
                                            sub_title,
                                            body=sub_body,
                                            bullets=sub_bullets if sub_bullets else None
                                        )
    
    # ----------------------------------------------------------------------
    # Prompt Building Methods
    # ----------------------------------------------------------------------
    
    def define_contexts(self, contexts=None) -> Optional['ContextBuilder']:
        """
        Define contexts and steps for this agent (alternative to POM/prompt)
        
        Args:
            contexts: Optional context configuration (dict or ContextBuilder)
            
        Returns:
            ContextBuilder for method chaining if no contexts provided
            
        Note:
            Contexts can coexist with traditional prompts. The restriction is only
            that you can't mix POM sections with raw text in the main prompt.
        """
        if contexts is not None:
            # New behavior - set contexts
            self._prompt_manager.define_contexts(contexts)
            return self
        else:
            # Legacy behavior - return ContextBuilder
            # Import here to avoid circular imports
            from signalwire_agents.core.contexts import ContextBuilder
            
            if self._contexts_builder is None:
                self._contexts_builder = ContextBuilder(self)
                self._contexts_defined = True
            
            return self._contexts_builder
    
    def _validate_prompt_mode_exclusivity(self):
        """
        Validate that POM sections and raw text are not mixed in the main prompt
        
        Note: This does NOT prevent contexts from being used alongside traditional prompts
        """
        # Delegate to prompt manager
        self._prompt_manager._validate_prompt_mode_exclusivity()
    
    def set_prompt_text(self, text: str) -> 'AgentBase':
        """
        Set the prompt as raw text instead of using POM
        
        Args:
            text: The raw prompt text
            
        Returns:
            Self for method chaining
        """
        self._prompt_manager.set_prompt_text(text)
        return self
    
    def set_post_prompt(self, text: str) -> 'AgentBase':
        """
        Set the post-prompt text for summary generation
        
        Args:
            text: The post-prompt text
            
        Returns:
            Self for method chaining
        """
        self._prompt_manager.set_post_prompt(text)
        return self
    
    def set_prompt_pom(self, pom: List[Dict[str, Any]]) -> 'AgentBase':
        """
        Set the prompt as a POM dictionary
        
        Args:
            pom: POM dictionary structure
            
        Returns:
            Self for method chaining
        """
        self._prompt_manager.set_prompt_pom(pom)
        return self
    
    def prompt_add_section(
        self, 
        title: str, 
        body: str = "", 
        bullets: Optional[List[str]] = None,
        numbered: bool = False,
        numbered_bullets: bool = False,
        subsections: Optional[List[Dict[str, Any]]] = None
    ) -> 'AgentBase':
        """
        Add a section to the prompt
        
        Args:
            title: Section title
            body: Optional section body text
            bullets: Optional list of bullet points
            numbered: Whether this section should be numbered
            numbered_bullets: Whether bullets should be numbered
            subsections: Optional list of subsection objects
            
        Returns:
            Self for method chaining
        """
        self._prompt_manager.prompt_add_section(
            title=title,
            body=body,
            bullets=bullets,
            numbered=numbered,
            numbered_bullets=numbered_bullets,
            subsections=subsections
        )
        return self
        
    def prompt_add_to_section(
        self,
        title: str,
        body: Optional[str] = None,
        bullet: Optional[str] = None,
        bullets: Optional[List[str]] = None
    ) -> 'AgentBase':
        """
        Add content to an existing section (creating it if needed)
        
        Args:
            title: Section title
            body: Optional text to append to section body
            bullet: Optional single bullet point to add
            bullets: Optional list of bullet points to add
            
        Returns:
            Self for method chaining
        """
        self._prompt_manager.prompt_add_to_section(
            title=title,
            body=body,
            bullet=bullet,
            bullets=bullets
        )
        return self
        
    def prompt_add_subsection(
        self,
        parent_title: str,
        title: str,
        body: str = "",
        bullets: Optional[List[str]] = None
    ) -> 'AgentBase':
        """
        Add a subsection to an existing section (creating parent if needed)
        
        Args:
            parent_title: Parent section title
            title: Subsection title
            body: Optional subsection body text
            bullets: Optional list of bullet points
            
        Returns:
            Self for method chaining
        """
        self._prompt_manager.prompt_add_subsection(
            parent_title=parent_title,
            title=title,
            body=body,
            bullets=bullets
        )
        return self
    
    def prompt_has_section(self, title: str) -> bool:
        """
        Check if a section exists in the prompt
        
        Args:
            title: Section title to check
            
        Returns:
            True if section exists, False otherwise
        """
        return self._prompt_manager.prompt_has_section(title)
    
    # ----------------------------------------------------------------------
    # Tool/Function Management
    # ----------------------------------------------------------------------
    
    def define_tool(
        self, 
        name: str, 
        description: str, 
        parameters: Dict[str, Any], 
        handler: Callable,
        secure: bool = True,
        fillers: Optional[Dict[str, List[str]]] = None,
        webhook_url: Optional[str] = None,
        **swaig_fields
    ) -> 'AgentBase':
        """
        Define a SWAIG function that the AI can call
        
        Args:
            name: Function name (must be unique)
            description: Function description for the AI
            parameters: JSON Schema of parameters
            handler: Function to call when invoked
            secure: Whether to require token validation
            fillers: Optional dict mapping language codes to arrays of filler phrases
            webhook_url: Optional external webhook URL to use instead of local handling
            **swaig_fields: Additional SWAIG fields to include in function definition
            
        Returns:
            Self for method chaining
        """
        self._tool_registry.define_tool(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            secure=secure,
            fillers=fillers,
            webhook_url=webhook_url,
            **swaig_fields
        )
        return self
    
    def register_swaig_function(self, function_dict: Dict[str, Any]) -> 'AgentBase':
        """
        Register a raw SWAIG function dictionary (e.g., from DataMap.to_swaig_function())
        
        Args:
            function_dict: Complete SWAIG function definition dictionary
            
        Returns:
            Self for method chaining
        """
        self._tool_registry.register_swaig_function(function_dict)
        return self
    
    def _tool_decorator(self, name=None, **kwargs):
        """
        Decorator for defining SWAIG tools in a class
        
        Used as:
        
        @agent.tool(name="example_function", parameters={...})
        def example_function(self, param1):
            # ...
        """
        return ToolDecorator.create_instance_decorator(self._tool_registry)(name, **kwargs)
    
    
    @classmethod
    def tool(cls, name=None, **kwargs):
        """
        Class method decorator for defining SWAIG tools
        
        Used as:
        
        @AgentBase.tool(name="example_function", parameters={...})
        def example_function(self, param1):
            # ...
        """
        return ToolDecorator.create_class_decorator()(name, **kwargs)
    
    # ----------------------------------------------------------------------
    # Override Points for Subclasses
    # ----------------------------------------------------------------------
    
    def get_name(self) -> str:
        """
        Get agent name
        
        Returns:
            Agent name
        """
        return self.name

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
                
                return {"error": "Path not found"}
            
            # Log all app routes for debugging
            self.log.debug("app_routes_registered")
            for route in app.routes:
                if hasattr(route, "path"):
                    self.log.debug("app_route", path=route.path)
            
            self._app = app
        
        return self._app
    
    def get_prompt(self) -> Union[str, List[Dict[str, Any]]]:
        """
        Get the prompt for the agent
        
        Returns:
            Either a string prompt or a POM object as list of dicts
        """
        # First check if prompt manager has a prompt
        prompt_result = self._prompt_manager.get_prompt()
        if prompt_result is not None:
            return prompt_result
            
        # If using POM, return the POM structure
        if self._use_pom and self.pom:
            try:
                # Try different methods that might be available on the POM implementation
                if hasattr(self.pom, 'render_dict'):
                    return self.pom.render_dict()
                elif hasattr(self.pom, 'to_dict'):
                    return self.pom.to_dict()
                elif hasattr(self.pom, 'to_list'):
                    return self.pom.to_list()
                elif hasattr(self.pom, 'render'):
                    render_result = self.pom.render()
                    # If render returns a string, we need to convert it to JSON
                    if isinstance(render_result, str):
                        try:
                            import json
                            return json.loads(render_result)
                        except:
                            # If we can't parse as JSON, fall back to raw text
                            pass
                    return render_result
                else:
                    # Last resort: attempt to convert the POM object directly to a list/dict
                    # This assumes the POM object has a reasonable __str__ or __repr__ method
                    pom_data = self.pom.__dict__
                    if '_sections' in pom_data and isinstance(pom_data['_sections'], list):
                        return pom_data['_sections']
                    # Fall through to default if nothing worked
            except Exception as e:
                self.log.error("pom_rendering_failed", error=str(e))
                # Fall back to raw text if POM fails
                
        # Return default text
        return f"You are {self.name}, a helpful AI assistant."
    
    def get_post_prompt(self) -> Optional[str]:
        """
        Get the post-prompt for the agent
        
        Returns:
            Post-prompt text or None if not set
        """
        return self._prompt_manager.get_post_prompt()
    
    def define_tools(self) -> List[SWAIGFunction]:
        """
        Define the tools this agent can use
        
        Returns:
            List of SWAIGFunction objects or raw dictionaries (for data_map tools)
            
        This method can be overridden by subclasses.
        """
        tools = []
        for func in self._tool_registry._swaig_functions.values():
            if isinstance(func, dict):
                # Raw dictionary from register_swaig_function (e.g., DataMap)
                tools.append(func)
            else:
                # SWAIGFunction object from define_tool
                tools.append(func)
        return tools
    
    def on_summary(self, summary: Optional[Dict[str, Any]], raw_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Called when a post-prompt summary is received
        
        Args:
            summary: The summary object or None if no summary was found
            raw_data: The complete raw POST data from the request
        """
        # Default implementation does nothing
        pass
    
    def on_function_call(self, name: str, args: Dict[str, Any], raw_data: Optional[Dict[str, Any]] = None) -> Any:
        """
        Called when a SWAIG function is invoked
        
        Args:
            name: Function name
            args: Function arguments
            raw_data: Raw request data
            
        Returns:
            Function result
        """
        # Check if the function is registered
        if name not in self._tool_registry._swaig_functions:
            # If the function is not found, return an error
            return {"response": f"Function '{name}' not found"}
            
        # Get the function
        func = self._tool_registry._swaig_functions[name]
        
        # Check if this is a data_map function (raw dictionary)
        if isinstance(func, dict):
            # Data_map functions execute on SignalWire's server, not here
            # This should never be called, but if it is, return an error
            return {"response": f"Data map function '{name}' should be executed by SignalWire server, not locally"}
        
        # Check if this is an external webhook function
        if hasattr(func, 'webhook_url') and func.webhook_url:
            # External webhook functions should be called directly by SignalWire, not locally
            return {"response": f"External webhook function '{name}' should be executed by SignalWire at {func.webhook_url}, not locally"}
        
        # Call the handler for regular SWAIG functions
        try:
            result = func.handler(args, raw_data)
            if result is None:
                # If the handler returns None, create a default response
                result = SwaigFunctionResult("Function executed successfully")
            return result
        except Exception as e:
            # If the handler raises an exception, return an error response
            return {"response": f"Error executing function '{name}': {str(e)}"}
    
    def validate_basic_auth(self, username: str, password: str) -> bool:
        """
        Validate basic auth credentials
        
        Args:
            username: Username from request
            password: Password from request
            
        Returns:
            True if valid, False otherwise
            
        This method can be overridden by subclasses.
        """
        return (username, password) == self._basic_auth
    
    def _create_tool_token(self, tool_name: str, call_id: str) -> str:
        """
        Create a secure token for a tool call
        
        Args:
            tool_name: Name of the tool
            call_id: Call ID for this session
            
        Returns:
            Secure token string
        """
        try:
            # Ensure we have a session manager
            if not hasattr(self, '_session_manager'):
                self.log.error("no_session_manager")
                return ""
                
            # Create the token using the session manager
            return self._session_manager.create_tool_token(tool_name, call_id)
        except Exception as e:
            self.log.error("token_creation_error", error=str(e), tool=tool_name, call_id=call_id)
            return ""
    
    def validate_tool_token(self, function_name: str, token: str, call_id: str) -> bool:
        """
        Validate a tool token
        
        Args:
            function_name: Name of the function/tool
            token: Token to validate
            call_id: Call ID for the session
            
        Returns:
            True if token is valid, False otherwise
        """
        try:
            # Skip validation for non-secure tools
            if function_name not in self._tool_registry._swaig_functions:
                self.log.warning("unknown_function", function=function_name)
                return False
                
            # Get the function and check if it's secure
            func = self._tool_registry._swaig_functions[function_name]
            is_secure = True  # Default to secure
            
            if isinstance(func, dict):
                # For raw dictionaries (DataMap functions), they're always secure
                is_secure = True
            else:
                # For SWAIGFunction objects, check the secure attribute
                is_secure = func.secure
                
            # Always allow non-secure functions
            if not is_secure:
                self.log.debug("non_secure_function_allowed", function=function_name)
                return True
            
            # Check if we have a session manager
            if not hasattr(self, '_session_manager'):
                self.log.error("no_session_manager")
                return False
            
            # Handle missing token
            if not token:
                self.log.warning("missing_token", function=function_name)
                return False
                
            # For debugging: Log token details
            try:
                # Capture original parameters
                self.log.debug("token_validate_input", 
                              function=function_name, 
                              call_id=call_id, 
                              token_length=len(token))
                
                # Try to decode token for debugging
                if hasattr(self._session_manager, 'debug_token'):
                    debug_info = self._session_manager.debug_token(token)
                    self.log.debug("token_debug", debug_info=debug_info)
                    
                    # Extract token components
                    if debug_info.get("valid_format") and "components" in debug_info:
                        components = debug_info["components"]
                        token_call_id = components.get("call_id")
                        token_function = components.get("function")
                        token_expiry = components.get("expiry")
                        
                        # Log parameter mismatches
                        if token_function != function_name:
                            self.log.warning("token_function_mismatch", 
                                           expected=function_name, 
                                           actual=token_function)
                        
                        if token_call_id != call_id:
                            self.log.warning("token_call_id_mismatch", 
                                           expected=call_id, 
                                           actual=token_call_id)
                            
                        # Check expiration
                        if debug_info.get("status", {}).get("is_expired"):
                            self.log.warning("token_expired", 
                                           expires_in=debug_info["status"].get("expires_in_seconds"))
            except Exception as e:
                self.log.error("token_debug_error", error=str(e))
                
            # Use call_id from token if the provided one is empty
            if not call_id and hasattr(self._session_manager, 'debug_token'):
                try:
                    debug_info = self._session_manager.debug_token(token)
                    if debug_info.get("valid_format") and "components" in debug_info:
                        token_call_id = debug_info["components"].get("call_id")
                        if token_call_id:
                            self.log.debug("using_call_id_from_token", call_id=token_call_id)
                            is_valid = self._session_manager.validate_tool_token(function_name, token, token_call_id)
                            if is_valid:
                                self.log.debug("token_valid_with_extracted_call_id")
                                return True
                except Exception as e:
                    self.log.error("error_using_call_id_from_token", error=str(e))
            
            # Normal validation with provided call_id
            is_valid = self._session_manager.validate_tool_token(function_name, token, call_id)
            
            if is_valid:
                self.log.debug("token_valid", function=function_name)
            else:
                self.log.warning("token_invalid", function=function_name)
                
            return is_valid
        except Exception as e:
            self.log.error("token_validation_error", error=str(e), function=function_name)
            return False
    
    # ----------------------------------------------------------------------
    # Web Server and Routing
    # ----------------------------------------------------------------------
    
    def get_basic_auth_credentials(self, include_source: bool = False) -> Union[Tuple[str, str], Tuple[str, str, str]]:
        """
        Get the basic auth credentials
        
        Args:
            include_source: Whether to include the source of the credentials
            
        Returns:
            If include_source is False:
                (username, password) tuple
            If include_source is True:
                (username, password, source) tuple, where source is one of:
                "provided", "environment", or "generated"
        """
        username, password = self._basic_auth
        
        if not include_source:
            return (username, password)
            
        # Determine source of credentials
        env_user = os.environ.get('SWML_BASIC_AUTH_USER')
        env_pass = os.environ.get('SWML_BASIC_AUTH_PASSWORD')
        
        # More robust source detection
        if env_user and env_pass and username == env_user and password == env_pass:
            source = "environment"
        elif username.startswith("user_") and len(password) > 20:  # Format of generated credentials
            source = "generated"
        else:
            source = "provided"
            
        return (username, password, source)
    
    def get_full_url(self, include_auth: bool = False) -> str:
        """
        Get the full URL for this agent's endpoint
        
        Args:
            include_auth: Whether to include authentication credentials in the URL
            
        Returns:
            Full URL including host, port, and route (with auth if requested)
        """
        mode = get_execution_mode()
        
        if mode == 'cgi':
            protocol = 'https' if os.getenv('HTTPS') == 'on' else 'http'
            host = os.getenv('HTTP_HOST') or os.getenv('SERVER_NAME') or 'localhost'
            script_name = os.getenv('SCRIPT_NAME', '')
            base_url = f"{protocol}://{host}{script_name}"
        elif mode == 'lambda':
            # AWS Lambda Function URL format
            lambda_url = os.getenv('AWS_LAMBDA_FUNCTION_URL')
            if lambda_url:
                base_url = lambda_url.rstrip('/')
            else:
                # Fallback construction for Lambda
                region = os.getenv('AWS_REGION', 'us-east-1')
                function_name = os.getenv('AWS_LAMBDA_FUNCTION_NAME', 'unknown')
                base_url = f"https://{function_name}.lambda-url.{region}.on.aws"
        elif mode == 'google_cloud_function':
            # Google Cloud Functions URL format
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT') or os.getenv('GCP_PROJECT')
            region = os.getenv('FUNCTION_REGION') or os.getenv('GOOGLE_CLOUD_REGION', 'us-central1')
            service_name = os.getenv('K_SERVICE') or os.getenv('FUNCTION_TARGET', 'unknown')
            
            if project_id:
                base_url = f"https://{region}-{project_id}.cloudfunctions.net/{service_name}"
            else:
                # Fallback for local testing or incomplete environment
                base_url = f"https://localhost:8080"
        elif mode == 'azure_function':
            # Azure Functions URL format
            function_app_name = os.getenv('WEBSITE_SITE_NAME') or os.getenv('AZURE_FUNCTIONS_APP_NAME')
            function_name = os.getenv('AZURE_FUNCTION_NAME', 'unknown')
            
            if function_app_name:
                base_url = f"https://{function_app_name}.azurewebsites.net/api/{function_name}"
            else:
                # Fallback for local testing
                base_url = f"https://localhost:7071/api/{function_name}"
        else:
            # Server mode - check for proxy URL first
            if hasattr(self, '_proxy_url_base') and self._proxy_url_base:
                # Use proxy URL when available (from reverse proxy detection)
                base_url = self._proxy_url_base.rstrip('/')
            else:
                # Fallback to local URL construction
                protocol = 'https' if getattr(self, 'ssl_enabled', False) else 'http'
                
                # Determine host part - include port unless it's the standard port for the protocol
                if getattr(self, 'ssl_enabled', False) and getattr(self, 'domain', None):
                    # Use domain, but include port if it's not the standard HTTPS port (443)
                    host_part = f"{self.domain}:{self.port}" if self.port != 443 else self.domain
                else:
                    # Use host:port for HTTP or when no domain is specified
                    host_part = f"{self.host}:{self.port}"
                
                base_url = f"{protocol}://{host_part}"
        
        # Add route if not already included (for server mode)
        if mode == 'server' and self.route and not base_url.endswith(self.route):
            base_url = f"{base_url}/{self.route.lstrip('/')}"
        
        # Add authentication if requested
        if include_auth:
            username, password = self.get_basic_auth_credentials()
            if username and password:
                # Parse URL to insert auth
                from urllib.parse import urlparse, urlunparse
                parsed = urlparse(base_url)
                # Reconstruct with auth
                base_url = urlunparse((
                    parsed.scheme,
                    f"{username}:{password}@{parsed.netloc}",
                    parsed.path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment
                ))
        
        return base_url
        
    def _build_webhook_url(self, endpoint: str, query_params: Optional[Dict[str, str]] = None) -> str:
        """
        Helper method to build webhook URLs consistently
        
        Args:
            endpoint: The endpoint path (e.g., "swaig", "post_prompt")
            query_params: Optional query parameters to append
            
        Returns:
            Fully constructed webhook URL
        """
        # Check for serverless environment and use appropriate URL generation
        mode = get_execution_mode()
        
        if mode != 'server':
            # In serverless mode, use the serverless-appropriate URL with auth
            base_url = self.get_full_url(include_auth=True)
            
            # Ensure the endpoint has a trailing slash to prevent redirects
            if endpoint in ["swaig", "post_prompt"]:
                endpoint = f"{endpoint}/"
                
            # Build the full webhook URL
            url = f"{base_url}/{endpoint}"
            
            # Add query parameters if any (only if they have values)
            if query_params:
                # Remove any call_id from query params
                filtered_params = {k: v for k, v in query_params.items() if k != "call_id" and v}
                if filtered_params:
                    params = "&".join([f"{k}={v}" for k, v in filtered_params.items()])
                    url = f"{url}?{params}"
            
            return url
        
        # Server mode - use existing logic with proxy/auth support
        # Use the parent class's implementation if available and has the same method
        if hasattr(super(), '_build_webhook_url'):
            # Ensure _proxy_url_base is synchronized
            if getattr(self, '_proxy_url_base', None) and hasattr(super(), '_proxy_url_base'):
                super()._proxy_url_base = self._proxy_url_base
                
            # Call parent's implementation
            return super()._build_webhook_url(endpoint, query_params)
            
        # Otherwise, fall back to our own implementation for server mode
        # Base URL construction
        if hasattr(self, '_proxy_url_base') and self._proxy_url_base:
            # For proxy URLs
            base = self._proxy_url_base.rstrip('/')
            
            # Always add auth credentials
            username, password = self._basic_auth
            url = urlparse(base)
            base = url._replace(netloc=f"{username}:{password}@{url.netloc}").geturl()
        else:
            # Determine protocol based on SSL settings
            protocol = "https" if getattr(self, 'ssl_enabled', False) else "http"
            
            # Determine host part - include port unless it's the standard port for the protocol
            if getattr(self, 'ssl_enabled', False) and getattr(self, 'domain', None):
                # Use domain, but include port if it's not the standard HTTPS port (443)
                host_part = f"{self.domain}:{self.port}" if self.port != 443 else self.domain
            else:
                # For local URLs
                if self.host in ("0.0.0.0", "127.0.0.1", "localhost"):
                    host = "localhost"
                else:
                    host = self.host
                
                host_part = f"{host}:{self.port}"
                
            # Always include auth credentials
            username, password = self._basic_auth
            base = f"{protocol}://{username}:{password}@{host_part}"
        
        # Ensure the endpoint has a trailing slash to prevent redirects
        if endpoint in ["swaig", "post_prompt"]:
            endpoint = f"{endpoint}/"
            
        # Simple path - use the route directly with the endpoint
        path = f"{self.route}/{endpoint}"
            
        # Construct full URL
        url = f"{base}{path}"
        
        # Add query parameters if any (only if they have values)
        # But NEVER add call_id parameter - it should be in the body, not the URL
        if query_params:
            # Remove any call_id from query params
            filtered_params = {k: v for k, v in query_params.items() if k != "call_id" and v}
            if filtered_params:
                params = "&".join([f"{k}={v}" for k, v in filtered_params.items()])
                url = f"{url}?{params}"
            
        return url

    def _render_swml(self, call_id: str = None, modifications: Optional[dict] = None) -> str:
        """
        Render the complete SWML document using SWMLService methods
        
        Args:
            call_id: Optional call ID for session-specific tokens
            modifications: Optional dict of modifications to apply to the SWML
            
        Returns:
            SWML document as a string
        """
        # Reset the document to a clean state
        self.reset_document()
        
        # Get prompt
        prompt = self.get_prompt()
        prompt_is_pom = isinstance(prompt, list)
        
        # Get post-prompt
        post_prompt = self.get_post_prompt()
        
        # Generate a call ID if needed
        if self._enable_state_tracking and call_id is None:
            call_id = self._session_manager.create_session()
            
        # Empty query params - no need to include call_id in URLs
        query_params = {}
        
        # Get the default webhook URL with auth
        default_webhook_url = self._build_webhook_url("swaig", query_params)
        
        # Use override if set
        if hasattr(self, '_web_hook_url_override') and self._web_hook_url_override:
            default_webhook_url = self._web_hook_url_override
        
        # Prepare SWAIG object (correct format)
        swaig_obj = {}
        
        # Add defaults if we have functions
        if self._tool_registry._swaig_functions:
            swaig_obj["defaults"] = {
                "web_hook_url": default_webhook_url
            }
            
        # Add native_functions if any are defined
        if self.native_functions:
            swaig_obj["native_functions"] = self.native_functions
        
        # Add includes if any are defined
        if self._function_includes:
            swaig_obj["includes"] = self._function_includes
        
        # Add internal_fillers if any are defined
        if hasattr(self, '_internal_fillers') and self._internal_fillers:
            swaig_obj["internal_fillers"] = self._internal_fillers
        
        # Create functions array
        functions = []
        
        # Add each function to the functions array
        for name, func in self._tool_registry._swaig_functions.items():
            if isinstance(func, dict):
                # For raw dictionaries (DataMap functions), use the entire dictionary as-is
                # This preserves data_map and any other special fields
                function_entry = func.copy()
                
                # Ensure the function name is set correctly
                function_entry["function"] = name
                
            else:
                # For SWAIGFunction objects, build the entry manually
                # Check if it's secure and get token for secure functions when we have a call_id
                token = None
                if func.secure and call_id:
                    token = self._create_tool_token(tool_name=name, call_id=call_id)
                    
                # Prepare function entry
                function_entry = {
                    "function": name,
                    "description": func.description,
                    "parameters": {
                        "type": "object",
                        "properties": func.parameters
                    }
                }
                
                # Add fillers if present
                if func.fillers:
                    function_entry["fillers"] = func.fillers
                
                # Handle webhook URL
                if hasattr(func, 'webhook_url') and func.webhook_url:
                    # External webhook function - use the provided URL directly
                    function_entry["web_hook_url"] = func.webhook_url
                elif token:
                    # Local function with token - build local webhook URL
                    token_params = {"token": token}
                    function_entry["web_hook_url"] = self._build_webhook_url("swaig", token_params)
            
            functions.append(function_entry)
        
        # Add functions array to SWAIG object if we have any
        if functions:
            swaig_obj["functions"] = functions
        
        # Add post-prompt URL with token if we have a post-prompt
        post_prompt_url = None
        if post_prompt:
            # Create a token for post_prompt if we have a call_id
            query_params = {}
            if call_id and hasattr(self, '_session_manager'):
                try:
                    token = self._session_manager.create_tool_token("post_prompt", call_id)
                    if token:
                        query_params["token"] = token
                except Exception as e:
                    self.log.error("post_prompt_token_creation_error", error=str(e))
            
            # Build the URL with the token (if any)
            post_prompt_url = self._build_webhook_url("post_prompt", query_params)
            
            # Use override if set
            if hasattr(self, '_post_prompt_url_override') and self._post_prompt_url_override:
                post_prompt_url = self._post_prompt_url_override
                
        # Add answer verb with auto-answer enabled
        self.add_verb("answer", {})
        
        # Use the AI verb handler to build and validate the AI verb config
        ai_config = {}
        
        # Get the AI verb handler
        ai_handler = self.verb_registry.get_handler("ai")
        if ai_handler:
            try:
                # Check if we're in contexts mode
                if self._contexts_defined and self._contexts_builder:
                    # Generate contexts and combine with base prompt
                    contexts_dict = self._contexts_builder.to_dict()
                    
                    # Determine base prompt (required when using contexts)
                    base_prompt_text = None
                    base_prompt_pom = None
                    
                    if prompt_is_pom:
                        base_prompt_pom = prompt
                    elif prompt:
                        base_prompt_text = prompt
                    else:
                        # Provide default base prompt if none exists
                        base_prompt_text = f"You are {self.name}, a helpful AI assistant that follows structured workflows."
                    
                    # Build AI config with base prompt + contexts
                    ai_config = ai_handler.build_config(
                        prompt_text=base_prompt_text,
                        prompt_pom=base_prompt_pom,
                        contexts=contexts_dict,
                        post_prompt=post_prompt,
                        post_prompt_url=post_prompt_url,
                        swaig=swaig_obj if swaig_obj else None
                    )
                else:
                    # Build AI config using the traditional prompt approach
                    ai_config = ai_handler.build_config(
                        prompt_text=None if prompt_is_pom else prompt,
                        prompt_pom=prompt if prompt_is_pom else None,
                        post_prompt=post_prompt,
                        post_prompt_url=post_prompt_url,
                        swaig=swaig_obj if swaig_obj else None
                    )
                
                # Add new configuration parameters to the AI config
                
                # Add hints if any
                if self._hints:
                    ai_config["hints"] = self._hints
                
                # Add languages if any
                if self._languages:
                    ai_config["languages"] = self._languages
                
                # Add pronunciation rules if any
                if self._pronounce:
                    ai_config["pronounce"] = self._pronounce
                
                # Add params if any
                if self._params:
                    ai_config["params"] = self._params
                
                # Add global_data if any
                if self._global_data:
                    ai_config["global_data"] = self._global_data
                    
            except ValueError as e:
                if not self._suppress_logs:
                    self.log.error("ai_verb_config_error", error=str(e))
        else:
            # Fallback if no handler (shouldn't happen but just in case)
            ai_config = {
                "prompt": {
                    "text" if not prompt_is_pom else "pom": prompt
                }
            }
            
            if post_prompt:
                ai_config["post_prompt"] = {"text": post_prompt}
                if post_prompt_url:
                    ai_config["post_prompt_url"] = post_prompt_url
                
            if swaig_obj:
                ai_config["SWAIG"] = swaig_obj
        
        # Add the new configurations if not already added by the handler
        if self._hints and "hints" not in ai_config:
            ai_config["hints"] = self._hints
        
        if self._languages and "languages" not in ai_config:
            ai_config["languages"] = self._languages
        
        if self._pronounce and "pronounce" not in ai_config:
            ai_config["pronounce"] = self._pronounce
        
        if self._params and "params" not in ai_config:
            ai_config["params"] = self._params
        
        if self._global_data and "global_data" not in ai_config:
            ai_config["global_data"] = self._global_data
        
        # Add the AI verb to the document
        self.add_verb("ai", ai_config)
        
        # Apply any modifications from the callback to agent state
        if modifications and isinstance(modifications, dict):
            # Handle global_data modifications by updating the AI config directly
            if "global_data" in modifications:
                if modifications["global_data"]:
                    # Merge the modification global_data with existing global_data
                    ai_config["global_data"] = {**ai_config.get("global_data", {}), **modifications["global_data"]}
            
            # Handle other modifications by updating the AI config
            for key, value in modifications.items():
                if key != "global_data":  # global_data handled above
                    ai_config[key] = value
            
            # Clear and rebuild the document with the modified AI config
            self.reset_document()
            self.add_verb("answer", {})
            self.add_verb("ai", ai_config)
        
        # Return the rendered document as a string
        return self.render_document()
    
    def _check_basic_auth(self, request: Request) -> bool:
        """
        Check basic auth from a request
        
        Args:
            request: FastAPI request object
            
        Returns:
            True if auth is valid, False otherwise
        """
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Basic "):
            return False
            
        try:
            # Decode the base64 credentials
            credentials = base64.b64decode(auth_header[6:]).decode("utf-8")
            username, password = credentials.split(":", 1)
            return self.validate_basic_auth(username, password)
        except Exception:
            return False
    
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
            
            # Include router with prefix
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
        
        # Log startup information using structured logging
        self.log.info("agent_starting",
                     agent=self.name,
                     url=f"http://{host}:{port}{self.route}",
                     username=username,
                     password_length=len(password),
                     auth_source=source)
        
        # Print user-friendly startup message (keep this for development UX)
        print(f"Agent '{self.name}' is available at:")
        print(f"URL: http://{host}:{port}{self.route}")
        print(f"Basic Auth: {username}:{password} (source: {source})")
        
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
            elif mode == 'lambda':
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

    def _check_cgi_auth(self) -> bool:
        """
        Check basic auth in CGI mode using environment variables
        
        Returns:
            True if auth is valid, False otherwise
        """
        # Check for HTTP_AUTHORIZATION environment variable
        auth_header = os.getenv('HTTP_AUTHORIZATION')
        if not auth_header:
            # Also check for REMOTE_USER (if web server handled auth)
            remote_user = os.getenv('REMOTE_USER')
            if remote_user:
                # If web server handled auth, trust it
                return True
            return False
        
        if not auth_header.startswith('Basic '):
            return False
            
        try:
            # Decode the base64 credentials
            credentials = base64.b64decode(auth_header[6:]).decode("utf-8")
            username, password = credentials.split(":", 1)
            return self.validate_basic_auth(username, password)
        except Exception:
            return False
    
    def _send_cgi_auth_challenge(self) -> str:
        """
        Send authentication challenge in CGI mode
        
        Returns:
            HTTP response with 401 status and WWW-Authenticate header
        """
        # In CGI, we need to output the complete HTTP response
        response = "Status: 401 Unauthorized\r\n"
        response += "WWW-Authenticate: Basic realm=\"SignalWire Agent\"\r\n"
        response += "Content-Type: application/json\r\n"
        response += "\r\n"
        response += json.dumps({"error": "Unauthorized"})
        return response

    def _check_lambda_auth(self, event) -> bool:
        """
        Check basic auth in Lambda mode using event headers
        
        Args:
            event: Lambda event object containing headers
            
        Returns:
            True if auth is valid, False otherwise
        """
        if not event or 'headers' not in event:
            return False
            
        headers = event['headers']
        
        # Check for authorization header (case-insensitive)
        auth_header = None
        for key, value in headers.items():
            if key.lower() == 'authorization':
                auth_header = value
                break
                
        if not auth_header or not auth_header.startswith('Basic '):
            return False
            
        try:
            # Decode the base64 credentials
            credentials = base64.b64decode(auth_header[6:]).decode("utf-8")
            username, password = credentials.split(":", 1)
            return self.validate_basic_auth(username, password)
        except Exception:
            return False
    
    def _send_lambda_auth_challenge(self) -> dict:
        """
        Send authentication challenge in Lambda mode
        
        Returns:
            Lambda response with 401 status and WWW-Authenticate header
        """
        return {
            "statusCode": 401,
            "headers": {
                "WWW-Authenticate": "Basic realm=\"SignalWire Agent\"",
                "Content-Type": "application/json"
            },
            "body": json.dumps({"error": "Unauthorized"})
        }
    

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
                    path = event.get('pathParameters', {}).get('proxy', '') if event.get('pathParameters') else ''
                    if not path:
                        swml_response = self._render_swml()
                        return {
                            "statusCode": 200,
                            "headers": {"Content-Type": "application/json"},
                            "body": swml_response
                        }
                    else:
                        # Parse Lambda event for SWAIG function call
                        args = {}
                        call_id = None
                        raw_data = None
                        
                        # Parse request body if present
                        body_content = event.get('body')
                        if body_content:
                            try:
                                if isinstance(body_content, str):
                                    raw_data = json.loads(body_content)
                                else:
                                    raw_data = body_content
                                    
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
                        
                        result = self._execute_swaig_function(path, args, call_id, raw_data)
                        return {
                            "statusCode": 200,
                            "headers": {"Content-Type": "application/json"},
                            "body": json.dumps(result) if isinstance(result, dict) else str(result)
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
            logging.error(f"Error in serverless request handler: {e}")
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

    def on_swml_request(self, request_data: Optional[dict] = None, callback_path: Optional[str] = None) -> Optional[dict]:
        """
        Customization point for subclasses to modify SWML based on request data
        
        Args:
            request_data: Optional dictionary containing the parsed POST body
            callback_path: Optional callback path
            
        Returns:
            Optional dict with modifications to apply to the SWML document
        """
        # Default implementation does nothing
        return None
    
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
    
    # ----------------------------------------------------------------------
    # AI Verb Configuration Methods
    # ----------------------------------------------------------------------

    def add_hint(self, hint: str) -> 'AgentBase':
        """
        Add a simple string hint to help the AI agent understand certain words better
        
        Args:
            hint: The hint string to add
            
        Returns:
            Self for method chaining
        """
        if isinstance(hint, str) and hint:
            self._hints.append(hint)
        return self

    def add_hints(self, hints: List[str]) -> 'AgentBase':
        """
        Add multiple string hints
        
        Args:
            hints: List of hint strings
            
        Returns:
            Self for method chaining
        """
        if hints and isinstance(hints, list):
            for hint in hints:
                if isinstance(hint, str) and hint:
                    self._hints.append(hint)
        return self

    def add_pattern_hint(self, 
                         hint: str, 
                         pattern: str, 
                         replace: str, 
                         ignore_case: bool = False) -> 'AgentBase':
        """
        Add a complex hint with pattern matching
        
        Args:
            hint: The hint to match
            pattern: Regular expression pattern
            replace: Text to replace the hint with
            ignore_case: Whether to ignore case when matching
            
        Returns:
            Self for method chaining
        """
        if hint and pattern and replace:
            self._hints.append({
                "hint": hint,
                "pattern": pattern,
                "replace": replace,
                "ignore_case": ignore_case
            })
        return self

    def add_language(self, 
                     name: str, 
                     code: str, 
                     voice: str,
                     speech_fillers: Optional[List[str]] = None,
                     function_fillers: Optional[List[str]] = None,
                     engine: Optional[str] = None,
                     model: Optional[str] = None) -> 'AgentBase':
        """
        Add a language configuration to support multilingual conversations
        
        Args:
            name: Name of the language (e.g., "English", "French")
            code: Language code (e.g., "en-US", "fr-FR")
            voice: TTS voice to use. Can be a simple name (e.g., "en-US-Neural2-F") 
                  or a combined format "engine.voice:model" (e.g., "elevenlabs.josh:eleven_turbo_v2_5")
            speech_fillers: Optional list of filler phrases for natural speech
            function_fillers: Optional list of filler phrases during function calls
            engine: Optional explicit engine name (e.g., "elevenlabs", "rime")
            model: Optional explicit model name (e.g., "eleven_turbo_v2_5", "arcana")
            
        Returns:
            Self for method chaining
            
        Examples:
            # Simple voice name
            agent.add_language("English", "en-US", "en-US-Neural2-F")
            
            # Explicit parameters
            agent.add_language("English", "en-US", "josh", engine="elevenlabs", model="eleven_turbo_v2_5")
            
            # Combined format
            agent.add_language("English", "en-US", "elevenlabs.josh:eleven_turbo_v2_5")
        """
        language = {
            "name": name,
            "code": code
        }
        
        # Handle voice formatting (either explicit params or combined string)
        if engine or model:
            # Use explicit parameters if provided
            language["voice"] = voice
            if engine:
                language["engine"] = engine
            if model:
                language["model"] = model
        elif "." in voice and ":" in voice:
            # Parse combined string format: "engine.voice:model"
            try:
                engine_voice, model_part = voice.split(":", 1)
                engine_part, voice_part = engine_voice.split(".", 1)
                
                language["voice"] = voice_part
                language["engine"] = engine_part
                language["model"] = model_part
            except ValueError:
                # If parsing fails, use the voice string as-is
                language["voice"] = voice
        else:
            # Simple voice string
            language["voice"] = voice
        
        # Add fillers if provided
        if speech_fillers and function_fillers:
            language["speech_fillers"] = speech_fillers
            language["function_fillers"] = function_fillers
        elif speech_fillers or function_fillers:
            # If only one type of fillers is provided, use the deprecated "fillers" field
            fillers = speech_fillers or function_fillers
            language["fillers"] = fillers
        
        self._languages.append(language)
        return self

    def set_languages(self, languages: List[Dict[str, Any]]) -> 'AgentBase':
        """
        Set all language configurations at once
        
        Args:
            languages: List of language configuration dictionaries
            
        Returns:
            Self for method chaining
        """
        if languages and isinstance(languages, list):
            self._languages = languages
        return self

    def add_pronunciation(self, 
                         replace: str, 
                         with_text: str, 
                         ignore_case: bool = False) -> 'AgentBase':
        """
        Add a pronunciation rule to help the AI speak certain words correctly
        
        Args:
            replace: The expression to replace
            with_text: The phonetic spelling to use instead
            ignore_case: Whether to ignore case when matching
            
        Returns:
            Self for method chaining
        """
        if replace and with_text:
            rule = {
                "replace": replace,
                "with": with_text
            }
            if ignore_case:
                rule["ignore_case"] = True
            
            self._pronounce.append(rule)
        return self

    def set_pronunciations(self, pronunciations: List[Dict[str, Any]]) -> 'AgentBase':
        """
        Set all pronunciation rules at once
        
        Args:
            pronunciations: List of pronunciation rule dictionaries
            
        Returns:
            Self for method chaining
        """
        if pronunciations and isinstance(pronunciations, list):
            self._pronounce = pronunciations
        return self

    def set_param(self, key: str, value: Any) -> 'AgentBase':
        """
        Set a single AI parameter
        
        Args:
            key: Parameter name
            value: Parameter value
            
        Returns:
            Self for method chaining
        """
        if key:
            self._params[key] = value
        return self

    def set_params(self, params: Dict[str, Any]) -> 'AgentBase':
        """
        Set multiple AI parameters at once
        
        Args:
            params: Dictionary of parameter name/value pairs
            
        Returns:
            Self for method chaining
        """
        if params and isinstance(params, dict):
            self._params.update(params)
        return self

    def set_global_data(self, data: Dict[str, Any]) -> 'AgentBase':
        """
        Set the global data available to the AI throughout the conversation
        
        Args:
            data: Dictionary of global data
            
        Returns:
            Self for method chaining
        """
        if data and isinstance(data, dict):
            self._global_data = data
        return self

    def update_global_data(self, data: Dict[str, Any]) -> 'AgentBase':
        """
        Update the global data with new values
        
        Args:
            data: Dictionary of global data to update
            
        Returns:
            Self for method chaining
        """
        if data and isinstance(data, dict):
            self._global_data.update(data)
        return self

    def set_native_functions(self, function_names: List[str]) -> 'AgentBase':
        """
        Set the list of native functions to enable
        
        Args:
            function_names: List of native function names
            
        Returns:
            Self for method chaining
        """
        if function_names and isinstance(function_names, list):
            self.native_functions = [name for name in function_names if isinstance(name, str)]
        return self

    def set_internal_fillers(self, internal_fillers: Dict[str, Dict[str, List[str]]]) -> 'AgentBase':
        """
        Set internal fillers for native SWAIG functions
        
        Internal fillers provide custom phrases the AI says while executing
        internal/native functions like check_time, wait_for_user, next_step, etc.
        
        Args:
            internal_fillers: Dictionary mapping function names to language-specific filler phrases
                            Format: {"function_name": {"language_code": ["phrase1", "phrase2"]}}
                            Example: {"next_step": {"en-US": ["Moving to the next step...", "Great, let's continue..."]}}
            
        Returns:
            Self for method chaining
            
        Example:
            agent.set_internal_fillers({
                "next_step": {
                    "en-US": ["Moving to the next step...", "Great, let's continue..."],
                    "es": ["Pasando al siguiente paso...", "Excelente, continuemos..."]
                },
                "check_time": {
                    "en-US": ["Let me check the time...", "Getting the current time..."]
                }
            })
        """
        if internal_fillers and isinstance(internal_fillers, dict):
            if not hasattr(self, '_internal_fillers'):
                self._internal_fillers = {}
            self._internal_fillers.update(internal_fillers)
        return self

    def add_internal_filler(self, function_name: str, language_code: str, fillers: List[str]) -> 'AgentBase':
        """
        Add internal fillers for a specific function and language
        
        Args:
            function_name: Name of the internal function (e.g., 'next_step', 'check_time')
            language_code: Language code (e.g., 'en-US', 'es', 'fr')
            fillers: List of filler phrases for this function and language
            
        Returns:
            Self for method chaining
            
        Example:
            agent.add_internal_filler("next_step", "en-US", ["Moving to the next step...", "Great, let's continue..."])
        """
        if function_name and language_code and fillers and isinstance(fillers, list):
            if not hasattr(self, '_internal_fillers'):
                self._internal_fillers = {}
            
            if function_name not in self._internal_fillers:
                self._internal_fillers[function_name] = {}
                
            self._internal_fillers[function_name][language_code] = fillers
        return self

    def add_function_include(self, url: str, functions: List[str], meta_data: Optional[Dict[str, Any]] = None) -> 'AgentBase':
        """
        Add a remote function include to the SWAIG configuration
        
        Args:
            url: URL to fetch remote functions from
            functions: List of function names to include
            meta_data: Optional metadata to include with the function include
            
        Returns:
            Self for method chaining
        """
        if url and functions and isinstance(functions, list):
            include = {
                "url": url,
                "functions": functions
            }
            if meta_data and isinstance(meta_data, dict):
                include["meta_data"] = meta_data
            
            self._function_includes.append(include)
        return self

    def set_function_includes(self, includes: List[Dict[str, Any]]) -> 'AgentBase':
        """
        Set the complete list of function includes
        
        Args:
            includes: List of include objects, each with url and functions properties
            
        Returns:
            Self for method chaining
        """
        if includes and isinstance(includes, list):
            # Validate each include has required properties
            valid_includes = []
            for include in includes:
                if isinstance(include, dict) and "url" in include and "functions" in include:
                    if isinstance(include["functions"], list):
                        valid_includes.append(include)
            
            self._function_includes = valid_includes
        return self

    def enable_sip_routing(self, auto_map: bool = True, path: str = "/sip") -> 'AgentBase':
        """
        Enable SIP-based routing for this agent
        
        This allows the agent to automatically route SIP requests based on SIP usernames.
        When enabled, an endpoint at the specified path is automatically created
        that will handle SIP requests and deliver them to this agent.
        
        Args:
            auto_map: Whether to automatically map common SIP usernames to this agent
                     (based on the agent name and route path)
            path: The path to register the SIP routing endpoint (default: "/sip")
        
        Returns:
            Self for method chaining
        """
        # Create a routing callback that handles SIP usernames
        def sip_routing_callback(request: Request, body: Dict[str, Any]) -> Optional[str]:
            # Extract SIP username from the request body
            sip_username = self.extract_sip_username(body)
            
            if sip_username:
                self.log.info("sip_username_extracted", username=sip_username)
                
                # Check if this username is registered with this agent
                if hasattr(self, '_sip_usernames') and sip_username.lower() in self._sip_usernames:
                    self.log.info("sip_username_matched", username=sip_username)
                    # This route is already being handled by the agent, no need to redirect
                    return None
                else:
                    self.log.info("sip_username_not_matched", username=sip_username)
                    # Not registered with this agent, let routing continue
                    
            return None
            
        # Register the callback with the SWMLService, specifying the path
        self.register_routing_callback(sip_routing_callback, path=path)
        
        # Auto-map common usernames if requested
        if auto_map:
            self.auto_map_sip_usernames()
            
        return self
        
    def register_sip_username(self, sip_username: str) -> 'AgentBase':
        """
        Register a SIP username that should be routed to this agent
        
        Args:
            sip_username: SIP username to register
            
        Returns:
            Self for method chaining
        """
        if not hasattr(self, '_sip_usernames'):
            self._sip_usernames = set()
            
        self._sip_usernames.add(sip_username.lower())
        self.log.info("sip_username_registered", username=sip_username)
        
        return self
        
    def auto_map_sip_usernames(self) -> 'AgentBase':
        """
        Automatically register common SIP usernames based on this agent's 
        name and route
        
        Returns:
            Self for method chaining
        """
        # Register username based on agent name
        clean_name = re.sub(r'[^a-z0-9_]', '', self.name.lower())
        if clean_name:
            self.register_sip_username(clean_name)
            
        # Register username based on route (without slashes)
        clean_route = re.sub(r'[^a-z0-9_]', '', self.route.lower())
        if clean_route and clean_route != clean_name:
            self.register_sip_username(clean_route)
            
        # Register common variations if they make sense
        if len(clean_name) > 3:
            # Register without vowels
            no_vowels = re.sub(r'[aeiou]', '', clean_name)
            if no_vowels != clean_name and len(no_vowels) > 2:
                self.register_sip_username(no_vowels)
                
        return self

    def set_web_hook_url(self, url: str) -> 'AgentBase':
        """
        Override the default web_hook_url with a supplied URL string
        
        Args:
            url: The URL to use for SWAIG function webhooks
            
        Returns:
            Self for method chaining
        """
        self._web_hook_url_override = url
        return self
        
    def set_post_prompt_url(self, url: str) -> 'AgentBase':
        """
        Override the default post_prompt_url with a supplied URL string
        
        Args:
            url: The URL to use for post-prompt summary delivery
            
        Returns:
            Self for method chaining
        """
        self._post_prompt_url_override = url
        return self

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
            token = request.query_params.get("token")
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
            
            # Call the function
            try:
                result = self.on_function_call(function_name, args, body)
                
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

    async def _handle_root_request(self, request: Request):
        """Handle GET/POST requests to the root endpoint"""
        # Auto-detect proxy on first request if not explicitly configured
        if not getattr(self, '_proxy_detection_done', False) and not getattr(self, '_proxy_url_base', None):
            # Check for proxy headers
            forwarded_host = request.headers.get("X-Forwarded-Host")
            forwarded_proto = request.headers.get("X-Forwarded-Proto", "http")
            
            if forwarded_host:
                # Set proxy_url_base on both self and super() to ensure it's shared
                self._proxy_url_base = f"{forwarded_proto}://{forwarded_host}"
                if hasattr(super(), '_proxy_url_base'):
                    # Ensure parent class has the same proxy URL
                    super()._proxy_url_base = self._proxy_url_base
                
                self.log.info("proxy_auto_detected", proxy_url_base=self._proxy_url_base, 
                             source="X-Forwarded headers")
                self._proxy_detection_done = True
                
                # Also set the detection flag on parent
                if hasattr(super(), '_proxy_detection_done'):
                    super()._proxy_detection_done = True
            # If no explicit proxy headers, try the parent class detection method if it exists
            elif hasattr(super(), '_detect_proxy_from_request'):
                # Call the parent's detection method
                super()._detect_proxy_from_request(request)
                # Copy the result to our class
                if hasattr(super(), '_proxy_url_base') and getattr(super(), '_proxy_url_base', None):
                    self._proxy_url_base = super()._proxy_url_base
                self._proxy_detection_done = True
        
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
            token = request.query_params.get("token")
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
                swml = self._render_swml(call_id)
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
                
            # Extract summary from the correct location in the request
            summary = self._find_summary_in_post_data(body, req_log)
            
            # Call the summary handler with the summary and the full body
            try:
                if summary:
                    self.on_summary(summary, body)
                    req_log.debug("summary_handler_called_successfully")
                else:
                    # If no summary found but still want to process the data
                    self.on_summary(None, body)
                    req_log.debug("summary_handler_called_with_null_summary")
            except Exception as e:
                req_log.error("error_in_summary_handler", error=str(e))
            
            # Return success
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
    
    def _find_summary_in_post_data(self, body, logger):
        """
        Attempt to find a summary in the post-prompt response data
        
        Args:
            body: The request body
            logger: Logger instance
            
        Returns:
            Summary data or None if not found
        """
        if not body:
            return None

        # Various ways to get summary data
        if "summary" in body:
            return body["summary"]
            
        if "post_prompt_data" in body:
            pdata = body["post_prompt_data"]
            if isinstance(pdata, dict):
                if "parsed" in pdata and isinstance(pdata["parsed"], list) and pdata["parsed"]:
                    return pdata["parsed"][0]
                elif "raw" in pdata and pdata["raw"]:
                    try:
                        # Try to parse JSON from raw text
                        parsed = json.loads(pdata["raw"])
                        return parsed
                    except:
                        return pdata["raw"]
                        
        return None

    def _register_state_tracking_tools(self):
        """
        Register special tools for state tracking
        
        This adds startup_hook and hangup_hook SWAIG functions that automatically
        activate and deactivate the session when called. These are useful for
        tracking call state and cleaning up resources when a call ends.
        """
        # Register startup hook to activate session
        self.define_tool(
            name="startup_hook",
            description="Called when a new conversation starts to initialize state",
            parameters={},
            handler=lambda args, raw_data: self._handle_startup_hook(args, raw_data),
            secure=False  # No auth needed for this system function
        )
        
        # Register hangup hook to end session
        self.define_tool(
            name="hangup_hook",
            description="Called when conversation ends to clean up resources",
            parameters={},
            handler=lambda args, raw_data: self._handle_hangup_hook(args, raw_data),
            secure=False  # No auth needed for this system function
        )
    
    def _handle_startup_hook(self, args, raw_data):
        """
        Handle the startup hook function call
        
        Args:
            args: Function arguments (empty for this hook)
            raw_data: Raw request data containing call_id
            
        Returns:
            Success response
        """
        call_id = raw_data.get("call_id") if raw_data else None
        if call_id:
            self.log.info("session_activated", call_id=call_id)
            self._session_manager.activate_session(call_id)
            return SwaigFunctionResult("Session activated")
        else:
            self.log.warning("session_activation_failed", error="No call_id provided")
            return SwaigFunctionResult("Failed to activate session: No call_id provided")
    
    def _handle_hangup_hook(self, args, raw_data):
        """
        Handle the hangup hook function call
        
        Args:
            args: Function arguments (empty for this hook)
            raw_data: Raw request data containing call_id
            
        Returns:
            Success response
        """
        call_id = raw_data.get("call_id") if raw_data else None
        if call_id:
            self.log.info("session_ended", call_id=call_id)
            self._session_manager.end_session(call_id)
            return SwaigFunctionResult("Session ended")
        else:
            self.log.warning("session_end_failed", error="No call_id provided")
            return SwaigFunctionResult("Failed to end session: No call_id provided")

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
        # Handle dynamic configuration callback if set
        if self._dynamic_config_callback and request:
            try:
                # Extract request data
                query_params = dict(request.query_params)
                body_params = request_data or {}
                headers = dict(request.headers)
                
                # Create ephemeral configurator
                agent_config = EphemeralAgentConfig()
                
                # Call the user's configuration callback
                self._dynamic_config_callback(query_params, body_params, headers, agent_config)
                
                # Extract the configuration
                config = agent_config.extract_config()
                if config:
                    # Handle ephemeral prompt sections by applying them to this agent instance
                    if "_ephemeral_prompt_sections" in config:
                        for section in config["_ephemeral_prompt_sections"]:
                            self.prompt_add_section(
                                section["title"],
                                section.get("body", ""),
                                section.get("bullets"),
                                **{k: v for k, v in section.items() if k not in ["title", "body", "bullets"]}
                            )
                        del config["_ephemeral_prompt_sections"]
                    
                    if "_ephemeral_raw_prompt" in config:
                        self._raw_prompt = config["_ephemeral_raw_prompt"]
                        del config["_ephemeral_raw_prompt"]
                    
                    if "_ephemeral_post_prompt" in config:
                        self._post_prompt = config["_ephemeral_post_prompt"]
                        del config["_ephemeral_post_prompt"]
                    
                    return config
                    
            except Exception as e:
                self.log.error("dynamic_config_error", error=str(e))
        
        # Default implementation does nothing
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

    def set_dynamic_config_callback(self, callback: Callable[[dict, dict, dict, EphemeralAgentConfig], None]) -> 'AgentBase':
        """
        Set a callback function for dynamic agent configuration
        
        This callback receives an EphemeralAgentConfig object that provides the same
        configuration methods as AgentBase, allowing you to dynamically configure
        the agent's voice, prompt, parameters, etc. based on request data.
        
        Args:
            callback: Function that takes (query_params, body_params, headers, agent_config)
                     and configures the agent_config object using familiar methods like:
                     - agent_config.add_language(...)
                     - agent_config.prompt_add_section(...)
                     - agent_config.set_params(...)
                     - agent_config.set_global_data(...)
                     
        Example:
            def my_config(query_params, body_params, headers, agent):
                if query_params.get('tier') == 'premium':
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

    # ----------------------------------------------------------------------
    # Skill Management Methods
    # ----------------------------------------------------------------------

    def add_skill(self, skill_name: str, params: Optional[Dict[str, Any]] = None) -> 'AgentBase':
        """
        Add a skill to this agent
        
        Args:
            skill_name: Name of the skill to add
            params: Optional parameters to pass to the skill for configuration
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If skill not found or failed to load with detailed error message
        """
        success, error_message = self.skill_manager.load_skill(skill_name, params=params)
        if not success:
            raise ValueError(f"Failed to load skill '{skill_name}': {error_message}")
        return self

    def remove_skill(self, skill_name: str) -> 'AgentBase':
        """Remove a skill from this agent"""
        self.skill_manager.unload_skill(skill_name)
        return self

    def list_skills(self) -> List[str]:
        """List currently loaded skills"""
        return self.skill_manager.list_loaded_skills()

    def has_skill(self, skill_name: str) -> bool:
        """Check if skill is loaded"""
        return self.skill_manager.has_skill(skill_name)

    def _check_google_cloud_function_auth(self, request) -> bool:
        """
        Check basic auth in Google Cloud Functions mode using request headers
        
        Args:
            request: Flask request object or similar containing headers
            
        Returns:
            True if auth is valid, False otherwise
        """
        if not hasattr(request, 'headers'):
            return False
            
        # Check for authorization header (case-insensitive)
        auth_header = None
        for key in request.headers:
            if key.lower() == 'authorization':
                auth_header = request.headers[key]
                break
                
        if not auth_header or not auth_header.startswith('Basic '):
            return False
            
        try:
            import base64
            encoded_credentials = auth_header[6:]  # Remove 'Basic '
            decoded_credentials = base64.b64decode(encoded_credentials).decode('utf-8')
            provided_username, provided_password = decoded_credentials.split(':', 1)
            
            expected_username, expected_password = self.get_basic_auth_credentials()
            return (provided_username == expected_username and 
                    provided_password == expected_password)
        except Exception:
            return False

    def _check_azure_function_auth(self, req) -> bool:
        """
        Check basic auth in Azure Functions mode using request object
        
        Args:
            req: Azure Functions request object containing headers
            
        Returns:
            True if auth is valid, False otherwise
        """
        if not hasattr(req, 'headers'):
            return False
            
        # Check for authorization header (case-insensitive)
        auth_header = None
        for key, value in req.headers.items():
            if key.lower() == 'authorization':
                auth_header = value
                break
                
        if not auth_header or not auth_header.startswith('Basic '):
            return False
            
        try:
            import base64
            encoded_credentials = auth_header[6:]  # Remove 'Basic '
            decoded_credentials = base64.b64decode(encoded_credentials).decode('utf-8')
            provided_username, provided_password = decoded_credentials.split(':', 1)
            
            expected_username, expected_password = self.get_basic_auth_credentials()
            return (provided_username == expected_username and 
                    provided_password == expected_password)
        except Exception:
            return False

    def _send_google_cloud_function_auth_challenge(self):
        """
        Send authentication challenge in Google Cloud Functions mode
        
        Returns:
            Flask-compatible response with 401 status and WWW-Authenticate header
        """
        from flask import Response
        return Response(
            response=json.dumps({"error": "Unauthorized"}),
            status=401,
            headers={
                "WWW-Authenticate": "Basic realm=\"SignalWire Agent\"",
                "Content-Type": "application/json"
            }
        )
    
    def _send_azure_function_auth_challenge(self):
        """
        Send authentication challenge in Azure Functions mode
        
        Returns:
            Azure Functions response with 401 status and WWW-Authenticate header
        """
        import azure.functions as func
        return func.HttpResponse(
            body=json.dumps({"error": "Unauthorized"}),
            status_code=401,
            headers={
                "WWW-Authenticate": "Basic realm=\"SignalWire Agent\"",
                "Content-Type": "application/json"
            }
        )

    def _handle_google_cloud_function_request(self, request):
        """
        Handle Google Cloud Functions specific requests
        
        Args:
            request: Flask request object from Google Cloud Functions
            
        Returns:
            Flask response object
        """
        try:
            # Get the path from the request
            path = request.path.strip('/')
            
            if not path:
                # Root request - return SWML
                swml_response = self._render_swml()
                from flask import Response
                return Response(
                    response=swml_response,
                    status=200,
                    headers={"Content-Type": "application/json"}
                )
            else:
                # SWAIG function call
                args = {}
                call_id = None
                raw_data = None
                
                # Parse request data
                if request.method == 'POST':
                    try:
                        if request.is_json:
                            raw_data = request.get_json()
                        else:
                            raw_data = json.loads(request.get_data(as_text=True))
                        
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
                
                result = self._execute_swaig_function(path, args, call_id, raw_data)
                from flask import Response
                return Response(
                    response=json.dumps(result) if isinstance(result, dict) else str(result),
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
            
            # Get the path from the request
            path = req.url.split('/')[-1] if req.url else ''
            
            if not path or path == 'api':
                # Root request - return SWML
                swml_response = self._render_swml()
                return func.HttpResponse(
                    body=swml_response,
                    status_code=200,
                    headers={"Content-Type": "application/json"}
                )
            else:
                # SWAIG function call
                args = {}
                call_id = None
                raw_data = None
                
                # Parse request data
                if req.method == 'POST':
                    try:
                        body = req.get_body()
                        if body:
                            raw_data = json.loads(body.decode('utf-8'))
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
                
                result = self._execute_swaig_function(path, args, call_id, raw_data)
                return func.HttpResponse(
                    body=json.dumps(result) if isinstance(result, dict) else str(result),
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
