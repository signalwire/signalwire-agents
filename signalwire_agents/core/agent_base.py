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
from signalwire_agents.core.swml_service import SWMLService
from signalwire_agents.core.swml_handler import AIVerbHandler
from signalwire_agents.core.skill_manager import SkillManager
from signalwire_agents.utils.schema_utils import SchemaUtils
from signalwire_agents.core.logging_config import get_logger, get_execution_mode

# Import refactored components
from signalwire_agents.core.agent.prompt.manager import PromptManager
from signalwire_agents.core.agent.tools.registry import ToolRegistry
from signalwire_agents.core.agent.tools.decorator import ToolDecorator

# Import all mixins
from signalwire_agents.core.mixins.prompt_mixin import PromptMixin
from signalwire_agents.core.mixins.tool_mixin import ToolMixin
from signalwire_agents.core.mixins.web_mixin import WebMixin
from signalwire_agents.core.mixins.auth_mixin import AuthMixin
from signalwire_agents.core.mixins.skill_mixin import SkillMixin
from signalwire_agents.core.mixins.ai_config_mixin import AIConfigMixin
from signalwire_agents.core.mixins.serverless_mixin import ServerlessMixin
from signalwire_agents.core.mixins.state_mixin import StateMixin

# Create a logger using centralized system
logger = get_logger("agent_base")


class AgentBase(
    AuthMixin,
    WebMixin,
    SWMLService,
    PromptMixin,
    ToolMixin,
    SkillMixin,
    AIConfigMixin,
    ServerlessMixin,
    StateMixin
):
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
        port: Optional[int] = None,
        basic_auth: Optional[Tuple[str, str]] = None,
        use_pom: bool = True,
        token_expiry_secs: int = 3600,
        auto_answer: bool = True,
        record_call: bool = False,
        record_format: str = "mp4",
        record_stereo: bool = True,
        default_webhook_url: Optional[str] = None,
        agent_id: Optional[str] = None,
        native_functions: Optional[List[str]] = None,
        schema_path: Optional[str] = None,
                suppress_logs: bool = False,
        enable_post_prompt_override: bool = False,
        check_for_input_override: bool = False,
        config_file: Optional[str] = None
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
            token_expiry_secs: Seconds until tokens expire
            auto_answer: Whether to automatically answer calls
            record_call: Whether to record calls
            record_format: Recording format
            record_stereo: Whether to record in stereo
            default_webhook_url: Optional default webhook URL for all SWAIG functions
            agent_id: Optional unique ID for this agent, generated if not provided
            native_functions: Optional list of native functions to include in the SWAIG object
            schema_path: Optional path to the schema file
            suppress_logs: Whether to suppress structured logs
            enable_post_prompt_override: Whether to enable post-prompt override
            check_for_input_override: Whether to enable check-for-input override
            config_file: Optional path to configuration file
        """
        # Import SWMLService here to avoid circular imports
        from signalwire_agents.core.swml_service import SWMLService
        
        # If schema_path is not provided, we'll let SWMLService find it through its _find_schema_path method
        # which will be called in its __init__
        
        # Load service configuration from config file before initializing SWMLService
        service_config = self._load_service_config(config_file, name)
        
        # Apply service config values, with constructor parameters taking precedence
        final_route = route if route != "/" else service_config.get('route', route)
        final_host = host if host != "0.0.0.0" else service_config.get('host', host)
        # For port: use explicit param if provided, else config file, else let SWMLService use PORT env var
        final_port = port if port is not None else service_config.get('port', None)
        final_name = service_config.get('name', name)
        
        # Initialize the SWMLService base class
        super().__init__(
            name=final_name,
            route=final_route,
            host=final_host,
            port=final_port,
            basic_auth=basic_auth,
            schema_path=schema_path,
            config_file=config_file
        )
        
        # Log the schema path if found and not suppressing logs
        if self.schema_utils and self.schema_utils.schema_path and not suppress_logs:
            self.log.debug("using_schema_path", path=self.schema_utils.schema_path)
        
        # Setup logger for this instance
        self.log = logger.bind(agent=name)
        self.log.info("agent_initializing", agent=name, route=route, host=self.host, port=self.port)
        
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
        
        
        # Process class-decorated tools (using @AgentBase.tool)
        self._tool_registry.register_class_decorated_tools()
        
        # Add native_functions parameter
        self.native_functions = native_functions or []
        
        
        # Initialize new configuration containers
        self._hints = []
        self._languages = []
        self._pronounce = []
        self._params = {}
        self._global_data = {}
        self._function_includes = []
        # Initialize LLM params as empty - only send if explicitly set
        self._prompt_llm_params = {}
        self._post_prompt_llm_params = {}
        
        # Dynamic configuration callback
        self._dynamic_config_callback = None
        
        # Initialize skill manager
        self.skill_manager = SkillManager(self)
        
        # Initialize contexts system
        self._contexts_builder = None
        self._contexts_defined = False
        
        # Initialize SWAIG query params for dynamic config
        self._swaig_query_params = {}

        # Initialize verb insertion points for call flow customization
        self._pre_answer_verbs = []    # Verbs to run before answer (e.g., ringback, screening)
        self._answer_config = {}        # Configuration for the answer verb
        self._post_answer_verbs = []    # Verbs to run after answer, before AI (e.g., announcements)
        self._post_ai_verbs = []        # Verbs to run after AI ends (e.g., cleanup, transfers)

    # Verb categories for pre-answer validation
    _PRE_ANSWER_SAFE_VERBS = {
        "transfer", "execute", "return", "label", "goto", "request",
        "switch", "cond", "if", "eval", "set", "unset", "hangup",
        "send_sms", "sleep", "stop_record_call", "stop_denoise", "stop_tap"
    }
    _AUTO_ANSWER_VERBS = {"play", "connect"}

    @staticmethod
    def _load_service_config(config_file: Optional[str], service_name: str) -> dict:
        """Load service configuration from config file if available"""
        from signalwire_agents.core.config_loader import ConfigLoader
        
        # Find config file
        if not config_file:
            config_file = ConfigLoader.find_config_file(service_name)
        
        if not config_file:
            return {}
            
        # Load config
        config_loader = ConfigLoader([config_file])
        if not config_loader.has_config():
            return {}
        
        # Get service section
        service_config = config_loader.get_section('service')
        if service_config:
            return service_config
            
        return {}
    
    def get_name(self) -> str:
        """
        Get agent name
        
        Returns:
            Agent name
        """
        return self.name
    
    def get_full_url(self, include_auth: bool = False) -> str:
        """
        Get the full URL for this agent's endpoint

        Args:
            include_auth: Whether to include authentication credentials in the URL

        Returns:
            Full URL including host, port, and route (with auth if requested)
        """
        # If _proxy_url_base is set (e.g., from request URL detection), use it
        if hasattr(self, '_proxy_url_base') and self._proxy_url_base:
            base_url = self._proxy_url_base.rstrip('/')
            # Add authentication if requested
            if include_auth:
                username, password = self.get_basic_auth_credentials()
                if username and password:
                    from urllib.parse import urlparse, urlunparse
                    parsed = urlparse(base_url)
                    base_url = urlunparse((
                        parsed.scheme,
                        f"{username}:{password}@{parsed.netloc}",
                        parsed.path,
                        parsed.params,
                        parsed.query,
                        parsed.fragment
                    ))
            return base_url

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
            # Server mode - use the SWMLService's unified URL building
            # Build the full URL using the parent's method
            base_url = self._build_full_url(endpoint="", include_auth=include_auth)
            return base_url
        
        # For serverless modes, add authentication if requested
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
        
        # Add route for serverless modes
        if self.route and self.route != "/" and not base_url.endswith(self.route):
            base_url = f"{base_url}/{self.route.lstrip('/')}"
        
        return base_url
    
    def on_summary(self, summary: Optional[Dict[str, Any]], raw_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Called when a post-prompt summary is received

        Args:
            summary: The summary object or None if no summary was found
            raw_data: The complete raw POST data from the request
        """
        # Default implementation does nothing
        pass

    # ==================== Call Flow Verb Insertion Methods ====================

    def add_pre_answer_verb(self, verb_name: str, config: Dict[str, Any]) -> 'AgentBase':
        """
        Add a verb to run before the call is answered.

        Pre-answer verbs execute while the call is still ringing. Only certain
        verbs are safe to use before answering:

        Safe verbs: transfer, execute, return, label, goto, request, switch,
                   cond, if, eval, set, unset, hangup, send_sms, sleep,
                   stop_record_call, stop_denoise, stop_tap

        Verbs with auto_answer option (play, connect): Must include
        "auto_answer": False in config to prevent automatic answering.

        Args:
            verb_name: The SWML verb name (e.g., "play", "sleep", "request")
            config: Verb configuration dictionary

        Returns:
            Self for method chaining

        Raises:
            ValueError: If verb is not safe for pre-answer use

        Example:
            # Play ringback tone before answering
            agent.add_pre_answer_verb("play", {
                "urls": ["ring:us"],
                "auto_answer": False
            })
        """
        # Validate verb is safe for pre-answer use
        if verb_name in self._AUTO_ANSWER_VERBS:
            if not config.get("auto_answer") is False:
                self.log.warning(
                    "pre_answer_verb_will_answer",
                    verb=verb_name,
                    hint=f"Add 'auto_answer': False to prevent {verb_name} from answering the call"
                )
        elif verb_name not in self._PRE_ANSWER_SAFE_VERBS:
            raise ValueError(
                f"Verb '{verb_name}' is not safe for pre-answer use. "
                f"Safe verbs: {', '.join(sorted(self._PRE_ANSWER_SAFE_VERBS))}"
            )

        self._pre_answer_verbs.append((verb_name, config))
        return self

    def add_answer_verb(self, config: Optional[Dict[str, Any]] = None) -> 'AgentBase':
        """
        Configure the answer verb.

        The answer verb connects the call. Use this method to customize
        answer behavior, such as setting max_duration.

        Args:
            config: Optional answer verb configuration (e.g., {"max_duration": 3600})

        Returns:
            Self for method chaining

        Example:
            # Set maximum call duration to 1 hour
            agent.add_answer_verb({"max_duration": 3600})
        """
        self._answer_config = config or {}
        return self

    def add_post_answer_verb(self, verb_name: str, config: Dict[str, Any]) -> 'AgentBase':
        """
        Add a verb to run after the call is answered but before the AI starts.

        Post-answer verbs run after the call is connected. Common uses include
        welcome messages, legal disclaimers, and hold music.

        Args:
            verb_name: The SWML verb name (e.g., "play", "sleep")
            config: Verb configuration dictionary

        Returns:
            Self for method chaining

        Example:
            # Play welcome message
            agent.add_post_answer_verb("play", {
                "url": "say:Welcome to Acme Corporation."
            })
            # Brief pause
            agent.add_post_answer_verb("sleep", {"time": 500})
        """
        self._post_answer_verbs.append((verb_name, config))
        return self

    def add_post_ai_verb(self, verb_name: str, config: Dict[str, Any]) -> 'AgentBase':
        """
        Add a verb to run after the AI conversation ends.

        Post-AI verbs run when the AI completes its conversation. Common uses
        include clean disconnects, transfers, and logging.

        Args:
            verb_name: The SWML verb name (e.g., "hangup", "transfer", "request")
            config: Verb configuration dictionary

        Returns:
            Self for method chaining

        Example:
            # Log call completion and hang up
            agent.add_post_ai_verb("request", {
                "url": "https://api.example.com/call-complete",
                "method": "POST"
            })
            agent.add_post_ai_verb("hangup", {})
        """
        self._post_ai_verbs.append((verb_name, config))
        return self

    def clear_pre_answer_verbs(self) -> 'AgentBase':
        """
        Remove all pre-answer verbs.

        Returns:
            Self for method chaining
        """
        self._pre_answer_verbs = []
        return self

    def clear_post_answer_verbs(self) -> 'AgentBase':
        """
        Remove all post-answer verbs.

        Returns:
            Self for method chaining
        """
        self._post_answer_verbs = []
        return self

    def clear_post_ai_verbs(self) -> 'AgentBase':
        """
        Remove all post-AI verbs.

        Returns:
            Self for method chaining
        """
        self._post_ai_verbs = []
        return self

    # ==================== End Call Flow Verb Insertion Methods ====================

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
    
    def add_swaig_query_params(self, params: Dict[str, str]) -> 'AgentBase':
        """
        Add query parameters that will be included in all SWAIG webhook URLs
        
        This is particularly useful for preserving dynamic configuration state
        across SWAIG callbacks. For example, if your dynamic config adds skills
        based on query parameters, you can pass those same parameters through
        to the SWAIG webhook so the same configuration is applied.
        
        Args:
            params: Dictionary of query parameters to add to SWAIG URLs
            
        Returns:
            Self for method chaining
            
        Example:
            def dynamic_config(query_params, body_params, headers, agent):
                if query_params.get('tier') == 'premium':
                    agent.add_skill('advanced_search')
                    # Preserve the tier param so SWAIG callbacks work
                    agent.add_swaig_query_params({'tier': 'premium'})
        """
        if params and isinstance(params, dict):
            self._swaig_query_params.update(params)
        return self
    
    def clear_swaig_query_params(self) -> 'AgentBase':
        """
        Clear all SWAIG query parameters
        
        Returns:
            Self for method chaining
        """
        self._swaig_query_params = {}
        return self
    
    def _render_swml(self, call_id: str = None, modifications: Optional[dict] = None) -> str:
        """
        Render the complete SWML document using SWMLService methods
        
        Args:
            call_id: Optional call ID for session-specific tokens
            modifications: Optional dict of modifications to apply to the SWML
            
        Returns:
            SWML document as a string
        """
        self.log.debug("_render_swml_called", 
                      call_id=call_id,
                      has_modifications=bool(modifications),
                      use_ephemeral=bool(modifications and modifications.get("__use_ephemeral_agent")),
                      has_dynamic_callback=bool(self._dynamic_config_callback))
        
        # Check if we need to use an ephemeral agent for dynamic configuration
        agent_to_use = self
        if modifications and modifications.get("__use_ephemeral_agent"):
            # Create an ephemeral copy for this request
            self.log.debug("creating_ephemeral_agent", 
                          original_sections=len(self._prompt_manager._sections) if hasattr(self._prompt_manager, '_sections') else 0)
            agent_to_use = self._create_ephemeral_copy()
            self.log.debug("ephemeral_agent_created", 
                          ephemeral_sections=len(agent_to_use._prompt_manager._sections) if hasattr(agent_to_use._prompt_manager, '_sections') else 0)
            
            # Extract the request data
            request = modifications.get("__request")
            request_data = modifications.get("__request_data", {})
            
            if self._dynamic_config_callback:
                try:
                    # Extract request data
                    if request:
                        query_params = dict(request.query_params)
                        headers = dict(request.headers)
                    else:
                        # No request object - use empty defaults
                        query_params = {}
                        headers = {}
                    body_params = request_data
                    
                    # Call the dynamic config callback with the ephemeral agent
                    # This allows FULL dynamic configuration including adding skills
                    self.log.debug("calling_dynamic_config_on_ephemeral", has_request=bool(request))
                    self._dynamic_config_callback(query_params, body_params, headers, agent_to_use)
                    self.log.debug("dynamic_config_complete",
                                  ephemeral_sections_after=len(agent_to_use._prompt_manager._sections) if hasattr(agent_to_use._prompt_manager, '_sections') else 0)
                    
                except Exception as e:
                    self.log.error("dynamic_config_error", error=str(e))
            
            # Clear the special markers so they don't affect rendering
            modifications = None
        
        # Reset the document to a clean state
        agent_to_use.reset_document()
        
        # Get prompt
        prompt = agent_to_use.get_prompt()
        prompt_is_pom = isinstance(prompt, list)
        
        # Get post-prompt
        post_prompt = agent_to_use.get_post_prompt()
        
        # Generate a call ID if needed
        if call_id is None:
            call_id = agent_to_use._session_manager.create_session()
            self.log.debug("generated_call_id", call_id=call_id)
        else:
            self.log.debug("using_provided_call_id", call_id=call_id)
            
        # Start with any SWAIG query params that were set
        query_params = agent_to_use._swaig_query_params.copy() if agent_to_use._swaig_query_params else {}
        
        # Get the default webhook URL with auth
        default_webhook_url = agent_to_use._build_webhook_url("swaig", query_params)
        
        # Use override if set
        if hasattr(agent_to_use, '_web_hook_url_override') and agent_to_use._web_hook_url_override:
            default_webhook_url = agent_to_use._web_hook_url_override
        
        # Prepare SWAIG object (correct format)
        swaig_obj = {}
        
        # Add native_functions if any are defined
        if agent_to_use.native_functions:
            swaig_obj["native_functions"] = agent_to_use.native_functions
        
        # Add includes if any are defined
        if agent_to_use._function_includes:
            swaig_obj["includes"] = agent_to_use._function_includes
        
        # Add internal_fillers if any are defined
        if hasattr(agent_to_use, '_internal_fillers') and agent_to_use._internal_fillers:
            swaig_obj["internal_fillers"] = agent_to_use._internal_fillers
        
        # Create functions array
        functions = []
        
        # Debug logging to see what functions we have
        self.log.debug("checking_swaig_functions", 
                      agent_name=agent_to_use.name,
                      is_ephemeral=getattr(agent_to_use, '_is_ephemeral', False),
                      registry_id=id(agent_to_use._tool_registry),
                      agent_id=id(agent_to_use),
                      function_count=len(agent_to_use._tool_registry._swaig_functions) if hasattr(agent_to_use._tool_registry, '_swaig_functions') else 0,
                      functions=list(agent_to_use._tool_registry._swaig_functions.keys()) if hasattr(agent_to_use._tool_registry, '_swaig_functions') else [])
        
        # Add each function to the functions array
        # Check if the registry has the _swaig_functions attribute
        if not hasattr(agent_to_use._tool_registry, '_swaig_functions'):
            self.log.warning("tool_registry_missing_swaig_functions",
                           registry_id=id(agent_to_use._tool_registry),
                           agent_id=id(agent_to_use))
            agent_to_use._tool_registry._swaig_functions = {}
        
        for name, func in agent_to_use._tool_registry._swaig_functions.items():
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
                    token = agent_to_use._create_tool_token(tool_name=name, call_id=call_id)
                    self.log.debug("created_token_for_function", function=name, call_id=call_id, token_prefix=token[:20] if token else None)
                    
                # Prepare function entry
                function_entry = {
                    "function": name,
                    "description": func.description,
                    "parameters": func._ensure_parameter_structure()
                }
                
                # Add wait_file if present (audio/video file URL)
                if hasattr(func, 'wait_file') and func.wait_file:
                    wait_file_url = func.wait_file
                    # If wait_file is a relative URL, convert it to absolute using agent's base URL
                    if wait_file_url and not wait_file_url.startswith(('http://', 'https://', '//')):
                        # Build full URL using the agent's base URL
                        base_url = agent_to_use._get_base_url(include_auth=False)
                        # Handle relative paths appropriately
                        if not wait_file_url.startswith('/'):
                            wait_file_url = '/' + wait_file_url
                        wait_file_url = f"{base_url}{wait_file_url}"
                    function_entry["wait_file"] = wait_file_url
                
                # Add fillers if present (text phrases to say while processing)
                if hasattr(func, 'fillers') and func.fillers:
                    function_entry["fillers"] = func.fillers
                
                # Add wait_file_loops if present
                if hasattr(func, 'wait_file_loops') and func.wait_file_loops is not None:
                    function_entry["wait_file_loops"] = func.wait_file_loops
                
                # Handle webhook URL
                if hasattr(func, 'webhook_url') and func.webhook_url:
                    # External webhook function - use the provided URL directly
                    function_entry["web_hook_url"] = func.webhook_url
                elif token or agent_to_use._swaig_query_params:
                    # Local function with token OR SWAIG query params - build local webhook URL
                    # Start with SWAIG query params
                    url_params = agent_to_use._swaig_query_params.copy() if agent_to_use._swaig_query_params else {}
                    if token:
                        url_params["__token"] = token  # Use __token to avoid collision
                    function_entry["web_hook_url"] = agent_to_use._build_webhook_url("swaig", url_params)
            
            functions.append(function_entry)
        
        # Add functions array to SWAIG object if we have any
        if functions:
            swaig_obj["functions"] = functions
            # Add defaults section now that we know we have functions
            if "defaults" not in swaig_obj:
                swaig_obj["defaults"] = {
                    "web_hook_url": default_webhook_url
                }
        
        # Add post-prompt URL with token if we have a post-prompt
        post_prompt_url = None
        if post_prompt:
            # Create a token for post_prompt if we have a call_id
            # Start with SWAIG query params
            query_params = agent_to_use._swaig_query_params.copy() if agent_to_use._swaig_query_params else {}
            if call_id and hasattr(agent_to_use, '_session_manager'):
                try:
                    token = agent_to_use._session_manager.create_tool_token("post_prompt", call_id)
                    if token:
                        query_params["__token"] = token  # Use __token to avoid collision
                except Exception as e:
                    agent_to_use.log.error("post_prompt_token_creation_error", error=str(e))
            
            # Build the URL with the token (if any)
            post_prompt_url = agent_to_use._build_webhook_url("post_prompt", query_params)
            
            # Use override if set
            if hasattr(agent_to_use, '_post_prompt_url_override') and agent_to_use._post_prompt_url_override:
                post_prompt_url = agent_to_use._post_prompt_url_override
                
        # ========== PHASE 1: PRE-ANSWER VERBS ==========
        # These run while the call is still ringing
        for verb_name, verb_config in agent_to_use._pre_answer_verbs:
            agent_to_use.add_verb(verb_name, verb_config)

        # ========== PHASE 2: ANSWER VERB ==========
        # Only add answer verb if auto_answer is enabled
        if agent_to_use._auto_answer:
            agent_to_use.add_verb("answer", agent_to_use._answer_config)

        # ========== PHASE 3: POST-ANSWER VERBS ==========
        # These run after answer but before AI

        # Add recording if enabled (this is a post-answer verb)
        if agent_to_use._record_call:
            agent_to_use.add_verb("record_call", {
                "format": agent_to_use._record_format,
                "stereo": agent_to_use._record_stereo
            })

        # Add user-defined post-answer verbs
        for verb_name, verb_config in agent_to_use._post_answer_verbs:
            agent_to_use.add_verb(verb_name, verb_config)
        
        # Use the AI verb handler to build and validate the AI verb config
        ai_config = {}
        
        # Get the AI verb handler
        ai_handler = agent_to_use.verb_registry.get_handler("ai")
        if ai_handler:
            try:
                # Check if we're in contexts mode
                if agent_to_use._contexts_defined and agent_to_use._contexts_builder:
                    # Generate contexts and combine with base prompt
                    contexts_dict = agent_to_use._contexts_builder.to_dict()
                    
                    # Determine base prompt (required when using contexts)
                    base_prompt_text = None
                    base_prompt_pom = None
                    
                    if prompt_is_pom:
                        base_prompt_pom = prompt
                    elif prompt:
                        base_prompt_text = prompt
                    else:
                        # Provide default base prompt if none exists
                        base_prompt_text = f"You are {agent_to_use.name}, a helpful AI assistant that follows structured workflows."
                    
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
                if agent_to_use._hints:
                    ai_config["hints"] = agent_to_use._hints
                
                # Add languages if any
                if agent_to_use._languages:
                    ai_config["languages"] = agent_to_use._languages
                
                # Add pronunciation rules if any
                if agent_to_use._pronounce:
                    ai_config["pronounce"] = agent_to_use._pronounce
                
                # Add params if any
                if agent_to_use._params:
                    ai_config["params"] = agent_to_use._params
                
                # Add global_data if any
                if agent_to_use._global_data:
                    ai_config["global_data"] = agent_to_use._global_data
                
                # Always add LLM parameters to prompt
                if "prompt" in ai_config:
                    # Only add LLM params if explicitly set
                    if agent_to_use._prompt_llm_params:
                        if isinstance(ai_config["prompt"], dict):
                            ai_config["prompt"].update(agent_to_use._prompt_llm_params)
                        elif isinstance(ai_config["prompt"], str):
                            # Convert string prompt to dict format
                            ai_config["prompt"] = {
                                "text": ai_config["prompt"],
                                **agent_to_use._prompt_llm_params
                            }
                    
                # Only add LLM parameters to post_prompt if explicitly set
                if post_prompt and "post_prompt" in ai_config:
                    # Only add LLM params if explicitly set
                    if agent_to_use._post_prompt_llm_params:
                        if isinstance(ai_config["post_prompt"], dict):
                            ai_config["post_prompt"].update(agent_to_use._post_prompt_llm_params)
                        elif isinstance(ai_config["post_prompt"], str):
                            # Convert string post_prompt to dict format
                            ai_config["post_prompt"] = {
                                "text": ai_config["post_prompt"],
                                **agent_to_use._post_prompt_llm_params
                            }
                    
            except ValueError as e:
                if not agent_to_use._suppress_logs:
                    agent_to_use.log.error("ai_verb_config_error", error=str(e))
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
        if agent_to_use._hints and "hints" not in ai_config:
            ai_config["hints"] = agent_to_use._hints
        
        if agent_to_use._languages and "languages" not in ai_config:
            ai_config["languages"] = agent_to_use._languages
        
        if agent_to_use._pronounce and "pronounce" not in ai_config:
            ai_config["pronounce"] = agent_to_use._pronounce
        
        if agent_to_use._params and "params" not in ai_config:
            ai_config["params"] = agent_to_use._params
        
        if agent_to_use._global_data and "global_data" not in ai_config:
            ai_config["global_data"] = agent_to_use._global_data

        # ========== PHASE 4: AI VERB ==========
        # Add the AI verb to the document
        agent_to_use.add_verb("ai", ai_config)

        # ========== PHASE 5: POST-AI VERBS ==========
        # These run after the AI conversation ends
        for verb_name, verb_config in agent_to_use._post_ai_verbs:
            agent_to_use.add_verb(verb_name, verb_config)

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
            agent_to_use.reset_document()

            # Rebuild with 5-phase approach
            # PHASE 1: Pre-answer verbs
            for verb_name, verb_config in agent_to_use._pre_answer_verbs:
                agent_to_use.add_verb(verb_name, verb_config)

            # PHASE 2: Answer verb (if auto_answer enabled)
            if agent_to_use._auto_answer:
                agent_to_use.add_verb("answer", agent_to_use._answer_config)

            # PHASE 3: Post-answer verbs
            if agent_to_use._record_call:
                agent_to_use.add_verb("record_call", {
                    "format": agent_to_use._record_format,
                    "stereo": agent_to_use._record_stereo
                })
            for verb_name, verb_config in agent_to_use._post_answer_verbs:
                agent_to_use.add_verb(verb_name, verb_config)

            # PHASE 4: AI verb
            agent_to_use.add_verb("ai", ai_config)

            # PHASE 5: Post-AI verbs
            for verb_name, verb_config in agent_to_use._post_ai_verbs:
                agent_to_use.add_verb(verb_name, verb_config)
        
        # Return the rendered document as a string
        return agent_to_use.render_document()
    
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
        
        # Server mode - use the parent class's implementation from SWMLService
        # which properly handles SWML_PROXY_URL_BASE environment variable
        return super()._build_webhook_url(endpoint, query_params)
    
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
    
    def _create_ephemeral_copy(self):
        """
        Create a lightweight copy of this agent for ephemeral configuration.
        
        This creates a partial copy that shares most resources but has independent
        configuration for SWML generation. Used when dynamic configuration callbacks
        need to modify the agent without affecting the persistent state.
        
        Returns:
            A lightweight copy of the agent suitable for ephemeral modifications
        """
        import copy
        
        # Create a new instance of the same class
        cls = self.__class__
        ephemeral_agent = cls.__new__(cls)
        
        # Copy all attributes as shallow references first
        for key, value in self.__dict__.items():
            setattr(ephemeral_agent, key, value)
        
        # Deep copy only the configuration that affects SWML generation
        # These are the parts that dynamic config might modify
        ephemeral_agent._params = copy.deepcopy(self._params)
        ephemeral_agent._hints = copy.deepcopy(self._hints)
        ephemeral_agent._languages = copy.deepcopy(self._languages)
        ephemeral_agent._pronounce = copy.deepcopy(self._pronounce)
        ephemeral_agent._global_data = copy.deepcopy(self._global_data)
        ephemeral_agent._function_includes = copy.deepcopy(self._function_includes)

        # Deep copy verb insertion points for call flow customization
        ephemeral_agent._pre_answer_verbs = copy.deepcopy(self._pre_answer_verbs)
        ephemeral_agent._answer_config = copy.deepcopy(self._answer_config)
        ephemeral_agent._post_answer_verbs = copy.deepcopy(self._post_answer_verbs)
        ephemeral_agent._post_ai_verbs = copy.deepcopy(self._post_ai_verbs)

        # Deep copy LLM parameters
        ephemeral_agent._prompt_llm_params = copy.deepcopy(self._prompt_llm_params)
        ephemeral_agent._post_prompt_llm_params = copy.deepcopy(self._post_prompt_llm_params)
        
        # Copy internal fillers if they exist
        if hasattr(self, '_internal_fillers'):
            ephemeral_agent._internal_fillers = copy.deepcopy(self._internal_fillers)
        
        # Don't deep copy _contexts_builder - it has a circular reference to the agent
        # The contexts are already copied via _prompt_manager._contexts (below)
        # Just copy the flag indicating contexts are defined
        if hasattr(self, '_contexts_defined'):
            ephemeral_agent._contexts_defined = self._contexts_defined
        
        # Deep copy the POM object if it exists to prevent sharing prompt sections
        if hasattr(self, 'pom') and self.pom:
            ephemeral_agent.pom = copy.deepcopy(self.pom)
        # Handle native_functions which might be stored as an attribute or property
        if hasattr(self, '_native_functions'):
            ephemeral_agent._native_functions = copy.deepcopy(self._native_functions)
        elif hasattr(self, 'native_functions'):
            ephemeral_agent.native_functions = copy.deepcopy(self.native_functions)
        ephemeral_agent._swaig_query_params = copy.deepcopy(self._swaig_query_params)
        
        # Create new manager instances that point to the ephemeral agent
        # This breaks the circular reference and allows independent modification
        from signalwire_agents.core.agent.prompt.manager import PromptManager
        from signalwire_agents.core.agent.tools.registry import ToolRegistry
        
        # Create new prompt manager for the ephemeral agent
        ephemeral_agent._prompt_manager = PromptManager(ephemeral_agent)
        # Copy ALL PromptManager state
        if hasattr(self._prompt_manager, '_sections'):
            ephemeral_agent._prompt_manager._sections = copy.deepcopy(self._prompt_manager._sections)
        ephemeral_agent._prompt_manager._prompt_text = copy.deepcopy(self._prompt_manager._prompt_text)
        ephemeral_agent._prompt_manager._post_prompt_text = copy.deepcopy(self._prompt_manager._post_prompt_text)
        ephemeral_agent._prompt_manager._contexts = copy.deepcopy(self._prompt_manager._contexts)
        
        # Create new tool registry for the ephemeral agent
        ephemeral_agent._tool_registry = ToolRegistry(ephemeral_agent)
        # Copy the SWAIG functions - we need a shallow copy here because
        # the functions themselves can be shared, we just need a new dict
        if hasattr(self._tool_registry, '_swaig_functions'):
            ephemeral_agent._tool_registry._swaig_functions = self._tool_registry._swaig_functions.copy()
        if hasattr(self._tool_registry, '_tool_instances'):
            ephemeral_agent._tool_registry._tool_instances = self._tool_registry._tool_instances.copy()
        
        # Create a new skill manager for the ephemeral agent
        # This is important because skills register tools with the agent's registry
        from signalwire_agents.core.skill_manager import SkillManager
        ephemeral_agent.skill_manager = SkillManager(ephemeral_agent)
        
        # Copy any already loaded skills from the original agent
        # This ensures skills loaded during __init__ are available in the ephemeral agent
        if hasattr(self.skill_manager, 'loaded_skills'):
            for skill_key, skill_instance in self.skill_manager.loaded_skills.items():
                # Re-load the skill in the ephemeral agent's context
                # We need to get the skill name and params from the existing instance
                skill_name = skill_instance.SKILL_NAME
                skill_params = getattr(skill_instance, 'params', {})
                try:
                    ephemeral_agent.skill_manager.load_skill(skill_name, type(skill_instance), skill_params)
                except Exception as e:
                    self.log.warning("failed_to_copy_skill_to_ephemeral", 
                                   skill_name=skill_name, 
                                   error=str(e))
        
        # Re-bind the tool decorator method to the new instance
        ephemeral_agent.tool = ephemeral_agent._tool_decorator
        
        # Share the logger but bind it to indicate ephemeral copy
        ephemeral_agent.log = self.log.bind(ephemeral=True)
        
        # Mark this as an ephemeral agent to prevent double application of dynamic config
        ephemeral_agent._is_ephemeral = True
        
        return ephemeral_agent
    
    async def _handle_request(self, request: Request, response: Response):
        """
        Override SWMLService's _handle_request to use AgentBase's _render_swml
        
        This ensures that when routes are handled by SWMLService's router,
        they still use AgentBase's SWML rendering logic.
        """
        # Use WebMixin's implementation if available
        if hasattr(super(), '_handle_root_request'):
            return await self._handle_root_request(request)
        
        # Fallback to basic implementation
        try:
            # Parse body if POST request
            body = {}
            if request.method == "POST":
                try:
                    body = await request.json()
                except:
                    pass
            
            # Get call_id
            call_id = body.get("call_id") if body else request.query_params.get("call_id")
            
            # Check auth
            if not self._check_basic_auth(request):
                return Response(
                    content=json.dumps({"error": "Unauthorized"}),
                    status_code=401,
                    headers={"WWW-Authenticate": "Basic"},
                    media_type="application/json"
                )
            
            # Render SWML using AgentBase's method
            swml = self._render_swml(call_id)
            
            return Response(
                content=swml,
                media_type="application/json"
            )
        except Exception as e:
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=500,
                media_type="application/json"
            )
    
