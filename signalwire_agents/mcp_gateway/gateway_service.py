#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
MCP-SWAIG Gateway Service

HTTP/HTTPS server that bridges MCP servers with SignalWire SWAIG functions.
Manages sessions, handles authentication, and translates between protocols.
"""

import os
import sys
import json
import logging
import argparse
import signal
import re
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime
import base64
import ssl
import concurrent.futures

from flask import Flask, request, jsonify, Response
from werkzeug.serving import make_server
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
import threading

from signalwire_agents.mcp_gateway.session_manager import SessionManager
from signalwire_agents.mcp_gateway.mcp_manager import MCPManager
from signalwire_agents.core.config_loader import ConfigLoader
from signalwire_agents.core.security_config import SecurityConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gateway_service')


class MCPGateway:
    """Main gateway service class"""
    
    def __init__(self, config_path: str = "config.json"):
        # Use unified config loader
        self.config_loader = ConfigLoader([config_path])
        
        # Load config with fallback to old method if needed
        if self.config_loader.has_config():
            # Use new config loader with variable substitution
            self.config = self.config_loader.substitute_vars(self.config_loader.get_config())
            logger.info(f"Loaded config using unified ConfigLoader from {config_path}")
        else:
            # Fall back to old method for backward compatibility
            self.config = self._load_config(config_path)
            
        # Load security configuration
        self.security = SecurityConfig(config_file=config_path, service_name="mcp")
        self.security.log_config("MCPGateway")
        
        self.app = Flask(__name__)
        self.mcp_manager = MCPManager(self.config)
        self.session_manager = SessionManager(self.config)
        self.server = None
        self._shutdown_requested = False
        
        # Configure rate limiting from config
        self.rate_config = self.config.get('rate_limiting', {})
        default_limits = self.rate_config.get('default_limits', ["200 per day", "50 per hour"])
        storage_uri = self.rate_config.get('storage_uri', "memory://")
        
        self.limiter = Limiter(
            app=self.app,
            key_func=get_remote_address,
            default_limits=default_limits,
            storage_uri=storage_uri
        )
        
        # Configure security headers
        @self.app.after_request
        def set_security_headers(response):
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            response.headers['Content-Security-Policy'] = "default-src 'none'; frame-ancestors 'none';"
            if request.is_secure:
                response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
            return response
        
        # Configure request size limit (10MB)
        self.app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
        
        # Configure logging
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        logging.getLogger().setLevel(log_level)
        
        if log_config.get('file'):
            file_handler = logging.FileHandler(log_config['file'])
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logging.getLogger().addHandler(file_handler)
        
        # Set up routes
        self._setup_routes()
        
        # Validate services on startup
        logger.info("Validating MCP services...")
        validation_results = self.mcp_manager.validate_services()
        for service, valid in validation_results.items():
            if not valid:
                logger.warning(f"Service '{service}' failed validation")
    
    def _validate_service_name(self, name: str) -> str:
        """Validate service name to prevent injection attacks"""
        if not name or len(name) > 64:
            raise ValueError("Invalid service name length")
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            raise ValueError("Service name contains invalid characters")
        return name
    
    def _validate_session_id(self, session_id: str) -> str:
        """Validate session ID format"""
        if not session_id or len(session_id) > 128:
            raise ValueError("Invalid session ID length")
        if not re.match(r'^[a-zA-Z0-9_.-]+$', session_id):
            raise ValueError("Session ID contains invalid characters")
        return session_id
    
    def _validate_tool_name(self, name: str) -> str:
        """Validate tool name"""
        if not name or len(name) > 64:
            raise ValueError("Invalid tool name length")
        if not re.match(r'^[a-zA-Z0-9_-]+$', name):
            raise ValueError("Tool name contains invalid characters")
        return name
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security-relevant events with proper sanitization"""
        # Sanitize any user input in details
        sanitized = {}
        for key, value in details.items():
            if isinstance(value, str):
                # Truncate long strings and remove control characters
                sanitized[key] = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', str(value)[:256])
            else:
                sanitized[key] = value
        
        # Add timestamp and event type
        sanitized['event_type'] = event_type
        sanitized['timestamp'] = datetime.now().isoformat()
        
        # Log with SECURITY prefix for easy filtering
        logger.info(f"SECURITY_EVENT: {json.dumps(sanitized)}")
    
    def _substitute_env_vars(self, value: Any) -> Any:
        """Recursively substitute environment variables in config values
        
        Supports format: ${VAR_NAME|default_value}
        """
        if isinstance(value, str):
            # Check for environment variable pattern
            if value.startswith('${') and value.endswith('}'):
                var_expr = value[2:-1]
                if '|' in var_expr:
                    var_name, default = var_expr.split('|', 1)
                    return os.environ.get(var_name, default)
                else:
                    # No default provided
                    return os.environ.get(var_expr, value)
            return value
        elif isinstance(value, dict):
            return {k: self._substitute_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._substitute_env_vars(item) for item in value]
        else:
            return value
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file with environment variable substitution"""
        if not os.path.exists(config_path):
            # Check if sample_config.json exists
            sample_path = "sample_config.json"
            if os.path.exists(sample_path):
                # Copy sample to config
                import shutil
                shutil.copy(sample_path, config_path)
                logger.info(f"Created config.json from sample_config.json")
            else:
                # Create minimal default config if no sample exists
                default_config = {
                    "server": {
                        "host": "0.0.0.0",
                        "port": 8080,
                        "auth_user": "admin",
                        "auth_password": "changeme"
                    },
                    "services": {
                        "todo": {
                            "command": ["python3", "./test/todo_mcp.py"],
                            "description": "Simple todo list for testing",
                            "enabled": True
                        }
                    },
                    "session": {
                        "default_timeout": 300,
                        "max_sessions_per_service": 100,
                        "cleanup_interval": 60
                    },
                    "rate_limiting": {
                        "default_limits": ["200 per day", "50 per hour"],
                        "tools_limit": "30 per minute",
                        "call_limit": "10 per minute",
                        "session_delete_limit": "20 per minute",
                        "storage_uri": "memory://"
                    },
                    "logging": {
                        "level": "INFO"
                    }
                }
                
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                
                logger.info(f"Created default config at {config_path}")
                return default_config
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Apply environment variable substitution
        config = self._substitute_env_vars(config)
        
        # Handle special cases for numeric values
        # Port
        if isinstance(config.get('server', {}).get('port'), str):
            try:
                config['server']['port'] = int(config['server']['port'])
            except ValueError:
                logger.warning(f"Invalid port value: {config['server']['port']}, using default 8080")
                config['server']['port'] = 8080
        
        # Session timeouts and limits
        session_config = config.get('session', {})
        for key in ['default_timeout', 'max_sessions_per_service', 'cleanup_interval']:
            if key in session_config and isinstance(session_config[key], str):
                try:
                    session_config[key] = int(session_config[key])
                except ValueError:
                    logger.warning(f"Invalid {key} value: {session_config[key]}, using default")
                    # Set sensible defaults
                    defaults = {
                        'default_timeout': 300,
                        'max_sessions_per_service': 100,
                        'cleanup_interval': 60
                    }
                    session_config[key] = defaults.get(key, 60)
        
        return config
    
    def _check_auth(self, f):
        """Decorator for authentication using Bearer tokens or Basic auth"""
        @wraps(f)
        def decorated(*args, **kwargs):
            # Try Bearer token first
            auth_header = request.headers.get('Authorization', '')
            server_config = self.config.get('server', {})
            
            if auth_header.startswith('Bearer '):
                token = auth_header[7:]
                expected_token = server_config.get('auth_token')
                if expected_token and token == expected_token:
                    return f(*args, **kwargs)
            
            # Fall back to Basic auth
            auth = request.authorization
            if auth and auth.username == server_config.get('auth_user') and \
               auth.password == server_config.get('auth_password'):
                return f(*args, **kwargs)
            
            # Log failed auth attempt
            self._log_security_event('auth_failed', {
                'ip': request.remote_addr,
                'method': request.method,
                'path': request.path
            })
            
            return Response(
                'Authentication required',
                401,
                {'WWW-Authenticate': 'Basic realm="MCP Gateway"'}
            )
        
        return decorated
    
    def _setup_routes(self):
        """Set up Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Health check endpoint"""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            })
        
        @self.app.route('/services', methods=['GET'])
        @self._check_auth
        def list_services():
            """List available MCP services"""
            services = self.mcp_manager.list_services()
            return jsonify(services)
        
        @self.app.route('/services/<service_name>/tools', methods=['GET'])
        @self.limiter.limit(self.rate_config.get('tools_limit', "30 per minute"))
        @self._check_auth
        def get_service_tools(service_name):
            """Get tools for a specific service"""
            try:
                # Validate input
                service_name = self._validate_service_name(service_name)
                
                tools = self.mcp_manager.get_service_tools(service_name)
                return jsonify({"service": service_name, "tools": tools})
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                logger.error(f"Error getting tools for {service_name}: {e}")
                return jsonify({"error": "Service error"}), 500
        
        @self.app.route('/services/<service_name>/call', methods=['POST'])
        @self.limiter.limit(self.rate_config.get('call_limit', "10 per minute"))
        @self._check_auth
        def call_service_tool(service_name):
            """Call a tool on a service"""
            try:
                # Validate service name
                service_name = self._validate_service_name(service_name)
                
                data = request.get_json()
                if not data:
                    return jsonify({"error": "Invalid JSON"}), 400
                
                # Validate tool name
                tool_name = data.get('tool')
                if not tool_name:
                    return jsonify({"error": "Missing 'tool' parameter"}), 400
                tool_name = self._validate_tool_name(tool_name)
                
                # Validate session ID
                session_id = data.get('session_id')
                if not session_id:
                    return jsonify({"error": "Missing 'session_id' parameter"}), 400
                session_id = self._validate_session_id(session_id)
                
                # Validate other parameters
                arguments = data.get('arguments', {})
                if not isinstance(arguments, dict):
                    return jsonify({"error": "Invalid 'arguments' parameter"}), 400
                
                timeout = data.get('timeout', self.session_manager.default_timeout)
                if not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > 3600:
                    return jsonify({"error": "Invalid 'timeout' parameter"}), 400
                
                metadata = data.get('metadata', {})
                if not isinstance(metadata, dict):
                    return jsonify({"error": "Invalid 'metadata' parameter"}), 400
                
                # Log the tool call
                self._log_security_event('tool_call', {
                    'service': service_name,
                    'tool': tool_name,
                    'session_id': session_id,
                    'ip': request.remote_addr
                })
                
                # Get or create session
                session = self.session_manager.get_session(session_id)
                
                if not session:
                    # Create new session
                    logger.info(f"Creating new session {session_id} for service {service_name}")
                    
                    try:
                        client = self.mcp_manager.create_client(service_name)
                        session = self.session_manager.create_session(
                            session_id=session_id,
                            service_name=service_name,
                            process=client,
                            timeout=timeout,
                            metadata=metadata
                        )
                    except Exception as e:
                        logger.error(f"Failed to create session: {e}")
                        return jsonify({"error": f"Failed to create session: {str(e)}"}), 500
                
                elif session.service_name != service_name:
                    return jsonify({
                        "error": f"Session {session_id} is for service '{session.service_name}', not '{service_name}'"
                    }), 400
                
                # Get the MCP client from the session
                client = session.process
                
                # Call the tool
                logger.info(f"Calling {service_name}.{tool_name} for session {session_id}")
                result = client.call_tool(tool_name, arguments)
                
                # Extract text content if it's in MCP format
                if isinstance(result, dict) and 'content' in result:
                    content = result['content']
                    if isinstance(content, list) and len(content) > 0:
                        if content[0].get('type') == 'text':
                            result = content[0].get('text', result)
                
                return jsonify({
                    "session_id": session_id,
                    "service": service_name,
                    "tool": tool_name,
                    "result": result
                })
                
            except Exception as e:
                logger.error(f"Error calling tool: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/sessions', methods=['GET'])
        @self._check_auth
        def list_sessions():
            """List active sessions"""
            sessions = self.session_manager.list_sessions()
            return jsonify(sessions)
        
        @self.app.route('/sessions/<session_id>', methods=['DELETE'])
        @self.limiter.limit(self.rate_config.get('session_delete_limit', "20 per minute"))
        @self._check_auth
        def close_session(session_id):
            """Close a specific session"""
            try:
                # Validate session ID
                session_id = self._validate_session_id(session_id)
                
                if self.session_manager.close_session(session_id):
                    self._log_security_event('session_closed', {
                        'session_id': session_id,
                        'ip': request.remote_addr
                    })
                    return jsonify({"message": f"Session {session_id} closed"})
                else:
                    return jsonify({"error": f"Session {session_id} not found"}), 404
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
        
        @self.app.errorhandler(Exception)
        def handle_error(error):
            logger.error(f"Unhandled error: {error}")
            return jsonify({"error": "Internal server error"}), 500
    
    def run(self):
        """Run the gateway service"""
        server_config = self.config.get('server', {})
        host = server_config.get('host', '0.0.0.0')
        port = server_config.get('port', 8080)
        
        # Check for SSL certificate
        ssl_cert_path = "certs/server.pem"
        ssl_context = None
        protocol = "http"
        
        if os.path.exists(ssl_cert_path):
            logger.info(f"Found SSL certificate at {ssl_cert_path}, enabling HTTPS")
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(ssl_cert_path)
            protocol = "https"
        
        # Create server with threaded=True for better shutdown handling
        self.server = make_server(host, port, self.app, ssl_context=ssl_context, threaded=True)
        
        logger.info(f"MCP Gateway starting on {protocol}://{host}:{port}")
        logger.info(f"Basic Auth - User: {server_config.get('auth_user')}")
        
        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Received interrupt, shutting down...")
        finally:
            self.shutdown()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        self._shutdown_requested = True
        # Request server shutdown in a thread to avoid blocking
        if self.server:
            threading.Thread(target=self.server.shutdown, daemon=True).start()
    
    def shutdown(self):
        """Shutdown the gateway service"""
        if self._shutdown_requested:
            # Already shutting down
            return
            
        self._shutdown_requested = True
        logger.info("Shutting down MCP Gateway...")
        
        # Shutdown components in parallel for faster shutdown
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit shutdown tasks
            session_future = executor.submit(self.session_manager.shutdown)
            mcp_future = executor.submit(self.mcp_manager.shutdown)
            
            # Wait for completion with timeout
            try:
                session_future.result(timeout=5)
            except concurrent.futures.TimeoutError:
                logger.warning("Session manager shutdown timed out")
            
            try:
                mcp_future.result(timeout=5)
            except concurrent.futures.TimeoutError:
                logger.warning("MCP manager shutdown timed out")
        
        # Shutdown server
        if self.server and hasattr(self.server, 'shutdown'):
            try:
                self.server.shutdown()
            except:
                pass
        
        logger.info("MCP Gateway shutdown complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='MCP-SWAIG Gateway Service')
    parser.add_argument('-c', '--config', default='config.json',
                        help='Path to configuration file (default: config.json)')
    
    args = parser.parse_args()
    
    gateway = MCPGateway(args.config)
    gateway.run()


if __name__ == '__main__':
    main()