"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
AgentServer - Class for hosting multiple SignalWire AI Agents in a single server
"""

import os
import re
from typing import Dict, Any, Optional, List, Tuple, Callable

try:
    from fastapi import FastAPI, Request, Response
    import uvicorn
except ImportError:
    raise ImportError(
        "fastapi and uvicorn are required. Install them with: pip install fastapi uvicorn"
    )

from signalwire_agents.core.agent_base import AgentBase
from signalwire_agents.core.swml_service import SWMLService
from signalwire_agents.core.logging_config import get_logger, get_execution_mode


class AgentServer:
    """
    Server for hosting multiple SignalWire AI Agents under a single FastAPI application.
    
    This allows you to run multiple agents on different routes of the same server,
    which is useful for deployment and resource management.
    
    Example:
        server = AgentServer()
        server.register(SupportAgent(), "/support")
        server.register(SalesAgent(), "/sales") 
        server.run()
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 3000, log_level: str = "info"):
        """
        Initialize a new agent server
        
        Args:
            host: Host to bind the server to
            port: Port to bind the server to
            log_level: Logging level (debug, info, warning, error, critical)
        """
        self.host = host
        self.port = port
        self.log_level = log_level.lower()
        
        self.logger = get_logger("AgentServer")
        
        # Create FastAPI app
        self.app = FastAPI(
            title="SignalWire AI Agents",
            description="Hosted SignalWire AI Agents",
            version="0.1.2",
            redirect_slashes=False
        )
        
        # Keep track of registered agents
        self.agents: Dict[str, AgentBase] = {}

        # Keep track of SIP routing configuration
        self._sip_routing_enabled = False
        self._sip_route = None
        self._sip_username_mapping: Dict[str, str] = {}  # Maps SIP usernames to routes

        # Register health endpoints immediately so they're available
        # whether using server.run() or server.app with gunicorn
        self._register_health_endpoints()

        # Register catch-all handler on startup (not in __init__) so it runs AFTER
        # all other routes are registered. This ensures custom routes like /get_token
        # don't get overshadowed by the catch-all /{full_path:path} route.
        @self.app.on_event("startup")
        async def _setup_catch_all():
            self._register_catch_all_handler()
    
    def register(self, agent: AgentBase, route: Optional[str] = None) -> None:
        """
        Register an agent with the server
        
        Args:
            agent: The agent to register
            route: Optional route to override the agent's default route
            
        Raises:
            ValueError: If the route is already in use
        """
        # Use agent's route if none provided
        if route is None:
            route = agent.route
            
        # Normalize route format
        if not route.startswith("/"):
            route = f"/{route}"
            
        route = route.rstrip("/")
        
        # Check for conflicts
        if route in self.agents:
            raise ValueError(f"Route '{route}' is already in use")
            
        # Store the agent
        self.agents[route] = agent

        # Get the router and register it using the standard approach
        # The agent's router already handles both trailing slash versions properly
        router = agent.as_router()
        self.app.include_router(router, prefix=route)

        self.logger.info(f"Registered agent '{agent.get_name()}' at route '{route}'")
        
        # If SIP routing is enabled and auto-mapping is on, register SIP usernames for this agent
        if self._sip_routing_enabled:
            # Auto-map SIP usernames if enabled
            if getattr(self, '_sip_auto_map', False):
                self._auto_map_agent_sip_usernames(agent, route)
            
            # Register the SIP routing callback with this agent if we have one
            if hasattr(self, '_sip_routing_callback') and self._sip_routing_callback:
                agent.register_routing_callback(self._sip_routing_callback, path=self._sip_route)
            
    def setup_sip_routing(self, route: str = "/sip", auto_map: bool = True) -> None:
        """
        Set up central SIP-based routing for the server
        
        This configures all agents to handle SIP requests at the specified path,
        using a coordinated routing system where each agent checks if it can
        handle SIP requests for specific usernames.
        
        Args:
            route: The path for SIP routing (default: "/sip")
            auto_map: Whether to automatically map SIP usernames to agent routes
        """
        if self._sip_routing_enabled:
            self.logger.warning("SIP routing is already enabled")
            return
            
        # Normalize the route
        if not route.startswith("/"):
            route = f"/{route}"
            
        route = route.rstrip("/")
        
        # Store configuration
        self._sip_routing_enabled = True
        self._sip_route = route
        self._sip_auto_map = auto_map
        
        # If auto-mapping is enabled, map existing agents
        if auto_map:
            for agent_route, agent in self.agents.items():
                self._auto_map_agent_sip_usernames(agent, agent_route)
        
        # Create a unified routing callback that checks all registered usernames
        def server_sip_routing_callback(request: Request, body: Dict[str, Any]) -> Optional[str]:
            """Unified SIP routing callback that checks all registered usernames"""
            # Extract the SIP username
            sip_username = SWMLService.extract_sip_username(body)
            
            if sip_username:
                self.logger.info(f"Extracted SIP username: {sip_username}")
                
                # Look up the route for this username
                target_route = self._lookup_sip_route(sip_username)
                
                if target_route:
                    self.logger.info(f"Routing SIP request to {target_route}")
                    return target_route
                else:
                    self.logger.warning(f"No route found for SIP username: {sip_username}")
            
            # No routing needed (will be handled by the current agent)
            return None
        
        # Save the callback for later use with new agents
        self._sip_routing_callback = server_sip_routing_callback
        
        # Register this callback with each agent
        for agent in self.agents.values():
            # Each agent gets the same routing callback but at their own path
            agent.register_routing_callback(server_sip_routing_callback, path=route)
        
        self.logger.info(f"SIP routing enabled at {route} on all agents")
        
    def register_sip_username(self, username: str, route: str) -> None:
        """
        Register a mapping from SIP username to agent route
        
        Args:
            username: The SIP username
            route: The route to the agent
        """
        if not self._sip_routing_enabled:
            self.logger.warning("SIP routing is not enabled. Call setup_sip_routing() first.")
            return
            
        # Normalize the route
        if not route.startswith("/"):
            route = f"/{route}"
            
        route = route.rstrip("/")
        
        # Check if the route exists
        if route not in self.agents:
            self.logger.warning(f"Route {route} not found. SIP username will be registered but may not work.")
            
        # Add the mapping
        self._sip_username_mapping[username.lower()] = route
        self.logger.info(f"Registered SIP username '{username}' to route '{route}'")
        
    def _lookup_sip_route(self, username: str) -> Optional[str]:
        """
        Look up the route for a SIP username
        
        Args:
            username: The SIP username
            
        Returns:
            The route or None if not found
        """
        return self._sip_username_mapping.get(username.lower())
        
    def _auto_map_agent_sip_usernames(self, agent: AgentBase, route: str) -> None:
        """
        Automatically map SIP usernames for an agent
        
        This creates mappings based on the agent name and route.
        
        Args:
            agent: The agent to map
            route: The route to the agent
        """
        # Get the agent name and clean it for use as a SIP username
        agent_name = agent.get_name().lower()
        clean_name = re.sub(r'[^a-z0-9_]', '', agent_name)
        
        if clean_name:
            self.register_sip_username(clean_name, route)
            
        # Also use the route path (without slashes) as a username
        if route:
            # Extract just the last part of the route
            route_part = route.split("/")[-1]
            clean_route = re.sub(r'[^a-z0-9_]', '', route_part)
            
            if clean_route and clean_route != clean_name:
                self.register_sip_username(clean_route, route)
    
    def unregister(self, route: str) -> bool:
        """
        Unregister an agent from the server
        
        Args:
            route: The route of the agent to unregister
            
        Returns:
            True if the agent was unregistered, False if not found
        """
        # Normalize route format
        if not route.startswith("/"):
            route = f"/{route}"
            
        route = route.rstrip("/")
        
        # Check if the agent exists
        if route not in self.agents:
            return False
            
        # FastAPI doesn't support unregistering routes, so we'll just track it ourselves
        # and rebuild the app if needed
        del self.agents[route]
        
        self.logger.info(f"Unregistered agent at route '{route}'")
        return True
    
    def get_agents(self) -> List[Tuple[str, AgentBase]]:
        """
        Get all registered agents
        
        Returns:
            List of (route, agent) tuples
        """
        return [(route, agent) for route, agent in self.agents.items()]
    
    def get_agent(self, route: str) -> Optional[AgentBase]:
        """
        Get an agent by route
        
        Args:
            route: The route of the agent
            
        Returns:
            The agent or None if not found
        """
        # Normalize route format
        if not route.startswith("/"):
            route = f"/{route}"
            
        route = route.rstrip("/")
        
        return self.agents.get(route)
    
    def run(self, event=None, context=None, host: Optional[str] = None, port: Optional[int] = None) -> Any:
        """
        Universal run method that automatically detects environment and handles accordingly
        
        Detects execution mode and routes appropriately:
        - Server mode: Starts uvicorn server with FastAPI
        - CGI mode: Uses same routing logic but outputs CGI headers  
        - Lambda mode: Uses same routing logic but returns Lambda response
        
        Args:
            event: Serverless event object (Lambda, Cloud Functions)
            context: Serverless context object (Lambda, Cloud Functions)  
            host: Optional host to override the default (server mode only)
            port: Optional port to override the default (server mode only)
            
        Returns:
            Response for serverless modes, None for server mode
        """
        from signalwire_agents.core.logging_config import get_execution_mode
        import os
        import json
        
        # Detect execution mode
        mode = get_execution_mode()
        
        if mode == 'cgi':
            return self._handle_cgi_request()
        elif mode == 'lambda':
            return self._handle_lambda_request(event, context)
        else:
            # Server mode - use existing logic
            return self._run_server(host, port)
    
    def _handle_cgi_request(self) -> str:
        """Handle CGI request using same routing logic as server"""
        import os
        import sys
        import json
        
        # Get PATH_INFO to determine routing
        path_info = os.getenv('PATH_INFO', '').strip('/')
        
        # Use same routing logic as the server
        if not path_info:
            # Root request - return basic info or 404
            response = {"error": "No agent specified in path"}
            return self._format_cgi_response(response, status="404 Not Found")
        
        # Find matching agent using same logic as server
        for route, agent in self.agents.items():
            route_clean = route.lstrip("/")
            
            if path_info == route_clean:
                # Request to agent root - return SWML
                try:
                    swml = agent._render_swml()
                    return self._format_cgi_response(swml, content_type="application/json")
                except Exception as e:
                    error_response = {"error": f"Failed to generate SWML: {str(e)}"}
                    return self._format_cgi_response(error_response, status="500 Internal Server Error")
                    
            elif path_info.startswith(route_clean + "/"):
                # Request to agent sub-path
                relative_path = path_info[len(route_clean):].lstrip("/")
                
                if relative_path == "swaig":
                    # SWAIG function call - parse stdin for POST data
                    try:
                        # Read POST data from stdin
                        content_length = os.getenv('CONTENT_LENGTH')
                        if content_length:
                            raw_data = sys.stdin.buffer.read(int(content_length))
                            try:
                                post_data = json.loads(raw_data.decode('utf-8'))
                            except:
                                post_data = {}
                        else:
                            post_data = {}
                        
                        # Execute SWAIG function
                        result = agent._execute_swaig_function("", post_data, None, None)
                        return self._format_cgi_response(result, content_type="application/json")
                        
                    except Exception as e:
                        error_response = {"error": f"SWAIG function failed: {str(e)}"}
                        return self._format_cgi_response(error_response, status="500 Internal Server Error")
                
                elif relative_path.startswith("swaig/"):
                    # Direct function call like /matti/swaig/function_name
                    function_name = relative_path[6:]  # Remove "swaig/"
                    try:
                        # Read POST data from stdin  
                        content_length = os.getenv('CONTENT_LENGTH')
                        if content_length:
                            raw_data = sys.stdin.buffer.read(int(content_length))
                            try:
                                post_data = json.loads(raw_data.decode('utf-8'))
                            except:
                                post_data = {}
                        else:
                            post_data = {}
                            
                        result = agent._execute_swaig_function(function_name, post_data, None, None)
                        return self._format_cgi_response(result, content_type="application/json")
                        
                    except Exception as e:
                        error_response = {"error": f"Function call failed: {str(e)}"}
                        return self._format_cgi_response(error_response, status="500 Internal Server Error")
        
        # No matching agent found
        error_response = {"error": "Not Found"}
        return self._format_cgi_response(error_response, status="404 Not Found")
    
    def _handle_lambda_request(self, event, context) -> dict:
        """Handle Lambda request using same routing logic as server"""
        import json
        
        # Extract path from Lambda event
        path = ""
        if event and 'pathParameters' in event and event['pathParameters']:
            path = event['pathParameters'].get('proxy', '')
        elif event and 'path' in event:
            path = event['path']
        
        path = path.strip('/')
        
        # Use same routing logic as server
        if not path:
            return {
                "statusCode": 404,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "No agent specified in path"})
            }
        
        # Find matching agent
        for route, agent in self.agents.items():
            route_clean = route.lstrip("/")
            
            if path == route_clean:
                # Request to agent root - return SWML
                try:
                    swml = agent._render_swml()
                    return {
                        "statusCode": 200,
                        "headers": {"Content-Type": "application/json"},
                        "body": json.dumps(swml) if isinstance(swml, dict) else swml
                    }
                except Exception as e:
                    return {
                        "statusCode": 500,
                        "headers": {"Content-Type": "application/json"},
                        "body": json.dumps({"error": f"Failed to generate SWML: {str(e)}"})
                    }
                    
            elif path.startswith(route_clean + "/"):
                # Request to agent sub-path
                relative_path = path[len(route_clean):].lstrip("/")
                
                if relative_path == "swaig" or relative_path.startswith("swaig/"):
                    # SWAIG function call
                    try:
                        # Parse function name and body from event
                        function_name = relative_path[6:] if relative_path.startswith("swaig/") else ""
                        
                        # Get POST data from Lambda event body
                        post_data = {}
                        if event and 'body' in event and event['body']:
                            try:
                                post_data = json.loads(event['body'])
                            except:
                                pass
                        
                        result = agent._execute_swaig_function(function_name, post_data, None, None)
                        return {
                            "statusCode": 200,
                            "headers": {"Content-Type": "application/json"},
                            "body": json.dumps(result) if isinstance(result, dict) else result
                        }
                        
                    except Exception as e:
                        return {
                            "statusCode": 500,
                            "headers": {"Content-Type": "application/json"},
                            "body": json.dumps({"error": f"Function call failed: {str(e)}"})
                        }
        
        # No matching agent found
        return {
            "statusCode": 404,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Not Found"})
        }
    
    def _format_cgi_response(self, data, content_type: str = "application/json", status: str = "200 OK") -> str:
        """Format response for CGI output"""
        import json
        import sys
        
        # Format the body
        if isinstance(data, dict):
            body = json.dumps(data)
        else:
            body = str(data)
        
        # Build CGI response with headers
        response_lines = [
            f"Status: {status}",
            f"Content-Type: {content_type}",
            f"Content-Length: {len(body.encode('utf-8'))}",
            "",  # Empty line separates headers from body
            body
        ]
        
        response = "\n".join(response_lines)
        
        # Write directly to stdout and flush to ensure immediate output
        sys.stdout.write(response)
        sys.stdout.flush()
        
        return response

    def _register_health_endpoints(self) -> None:
        """Register health and readiness endpoints.

        Called during __init__ so endpoints are available whether using
        server.run() or accessing server.app directly with gunicorn.
        """
        @self.app.get("/health")
        def health_check():
            return {
                "status": "ok",
                "agents": len(self.agents),
                "routes": list(self.agents.keys())
            }

        @self.app.get("/ready")
        def readiness_check():
            return {
                "status": "ready",
                "agents": len(self.agents)
            }

    def _run_server(self, host: Optional[str] = None, port: Optional[int] = None) -> None:
        """Original server mode logic"""
        if not self.agents:
            self.logger.warning("Starting server with no registered agents")

        # Set host and port
        host = host or self.host
        port = port or self.port
        
        # Check for SSL configuration from environment variables
        ssl_enabled_env = os.environ.get('SWML_SSL_ENABLED', '').lower()
        ssl_enabled = ssl_enabled_env in ('true', '1', 'yes')
        ssl_cert_path = os.environ.get('SWML_SSL_CERT_PATH')
        ssl_key_path = os.environ.get('SWML_SSL_KEY_PATH')
        domain = os.environ.get('SWML_DOMAIN')
        
        # Validate SSL configuration if enabled
        if ssl_enabled:
            if not ssl_cert_path or not os.path.exists(ssl_cert_path):
                self.logger.warning(f"SSL cert not found: {ssl_cert_path}")
                ssl_enabled = False
            elif not ssl_key_path or not os.path.exists(ssl_key_path):
                self.logger.warning(f"SSL key not found: {ssl_key_path}")
                ssl_enabled = False
        
        # Update server info display with correct protocol
        protocol = "https" if ssl_enabled else "http"
        
        # Determine display host - include port unless it's the standard port for the protocol
        if ssl_enabled and domain:
            # Use domain, but include port if it's not the standard HTTPS port (443)
            display_host = f"{domain}:{port}" if port != 443 else domain
        else:
            # Use host:port for HTTP or when no domain is specified
            display_host = f"{host}:{port}"
        
        self.logger.info(f"Starting server on {protocol}://{display_host}")
        for route, agent in self.agents.items():
            username, password = agent.get_basic_auth_credentials()
            agent_url = agent.get_full_url(include_auth=False)
            self.logger.info(f"Agent '{agent.get_name()}' available at:")
            self.logger.info(f"URL: {agent_url}")
            self.logger.info(f"Basic Auth: {username}:{password}")
        
        # Start the server with or without SSL
        if ssl_enabled and ssl_cert_path and ssl_key_path:
            self.logger.info(f"Starting with SSL - cert: {ssl_cert_path}, key: {ssl_key_path}")
            uvicorn.run(
                self.app,
                host=host,
                port=port,
                log_level=self.log_level,
                ssl_certfile=ssl_cert_path,
                ssl_keyfile=ssl_key_path
            )
        else:
            uvicorn.run(
                self.app,
                host=host,
                port=port,
                log_level=self.log_level
            )

    def register_global_routing_callback(self, callback_fn: Callable[[Request, Dict[str, Any]], Optional[str]],
                                        path: str) -> None:
        """
        Register a routing callback across all agents

        This allows you to add unified routing logic to all agents at the same path.

        Args:
            callback_fn: The callback function to register
            path: The path to register the callback at
        """
        # Normalize the path
        if not path.startswith("/"):
            path = f"/{path}"

        path = path.rstrip("/")

        # Register with all existing agents
        for agent in self.agents.values():
            agent.register_routing_callback(callback_fn, path=path)

        self.logger.info(f"Registered global routing callback at {path} on all agents")

    def serve_static_files(self, directory: str, route: str = "/") -> None:
        """
        Serve static files from a directory.

        This method properly integrates static file serving with agent routes,
        ensuring that agent routes take priority over static files.

        Unlike using StaticFiles.mount("/", ...) directly on self.app, this method
        uses explicit route handlers that work correctly with agent routes.

        Args:
            directory: Path to the directory containing static files
            route: URL path prefix for static files (default: "/" for root)

        Example:
            server = AgentServer()
            server.register(SupportAgent(), "/support")
            server.serve_static_files("./web")  # Serves at /
            # /support -> SupportAgent
            # /index.html -> ./web/index.html
            # / -> ./web/index.html
        """
        from pathlib import Path
        from fastapi.responses import FileResponse
        from fastapi import HTTPException

        # Normalize directory path
        static_dir = Path(directory).resolve()

        if not static_dir.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        if not static_dir.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        # Normalize route
        if not route.startswith("/"):
            route = f"/{route}"
        route = route.rstrip("/")

        # Store static directory config for use by catch-all handler
        if not hasattr(self, '_static_directories'):
            self._static_directories = {}

        self._static_directories[route] = static_dir

        self.logger.info(f"Serving static files from '{directory}' at route '{route or '/'}'")


    def _serve_static_file(self, file_path: str, route: str = "/") -> Optional[Response]:
        """
        Internal method to serve a static file.

        Args:
            file_path: The requested file path
            route: The route prefix

        Returns:
            FileResponse if file exists, None otherwise
        """
        from pathlib import Path
        from fastapi.responses import FileResponse

        if not hasattr(self, '_static_directories'):
            return None

        static_dir = self._static_directories.get(route)
        if not static_dir:
            return None

        # Default to index.html for empty path
        if not file_path:
            file_path = "index.html"

        full_path = static_dir / file_path

        # Security: prevent path traversal
        try:
            full_path = full_path.resolve()
            if not str(full_path).startswith(str(static_dir)):
                return None
        except Exception:
            return None

        # Handle directory requests
        if full_path.is_dir():
            index_path = full_path / "index.html"
            if index_path.exists():
                full_path = index_path
            else:
                return None

        if not full_path.exists():
            return None

        return FileResponse(full_path)

    def _register_catch_all_handler(self) -> None:
        """
        Register catch-all route handler for agent routing and static files.

        This handler is needed for:
        1. Routing requests without trailing slashes to agents (e.g., /santa instead of /santa/)
        2. Serving static files from directories registered with serve_static_files()

        Called via startup event (not __init__) to ensure it runs AFTER all other routes
        are registered. This prevents the catch-all from overshadowing custom routes
        like /get_token that users may add to server.app.
        """
        @self.app.get("/{full_path:path}")
        @self.app.post("/{full_path:path}")
        async def handle_all_routes(request: Request, full_path: str):
            """Handle requests that don't match registered routes (e.g. /matti instead of /matti/)"""
            # Check if this path maps to one of our registered agents
            for route, agent in self.agents.items():
                # Check for exact match with registered route
                if full_path == route.lstrip("/"):
                    # This is a request to an agent's root without trailing slash
                    return await agent._handle_root_request(request)
                elif full_path.startswith(route.lstrip("/") + "/"):
                    # This is a request to an agent's sub-path
                    relative_path = full_path[len(route.lstrip("/")):]
                    relative_path = relative_path.lstrip("/")

                    # Route to appropriate handler based on path
                    if not relative_path or relative_path == "/":
                        return await agent._handle_root_request(request)

                    clean_path = relative_path.rstrip("/")
                    if clean_path == "debug":
                        return await agent._handle_debug_request(request)
                    elif clean_path == "swaig":
                        from fastapi import Response
                        return await agent._handle_swaig_request(request, Response())
                    elif clean_path == "post_prompt":
                        return await agent._handle_post_prompt_request(request)
                    elif clean_path == "check_for_input":
                        return await agent._handle_check_for_input_request(request)

                    # Check for custom routing callbacks
                    if hasattr(agent, '_routing_callbacks'):
                        for callback_path, callback_fn in agent._routing_callbacks.items():
                            cb_path_clean = callback_path.strip("/")
                            if clean_path == cb_path_clean:
                                request.state.callback_path = callback_path
                                return await agent._handle_root_request(request)

            # No matching agent - check for static files
            if hasattr(self, '_static_directories'):
                # Check each static directory route
                for static_route, static_dir in self._static_directories.items():
                    # For root static route, serve any unmatched path
                    if static_route == "" or static_route == "/":
                        response = self._serve_static_file(full_path, "")
                        if response:
                            return response
                    # For prefixed static routes, check if path matches
                    elif full_path.startswith(static_route.lstrip("/") + "/") or full_path == static_route.lstrip("/"):
                        relative_path = full_path[len(static_route.lstrip("/")):].lstrip("/")
                        response = self._serve_static_file(relative_path, static_route)
                        if response:
                            return response

            # No matching agent or static file found
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Not Found")
