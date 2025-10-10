"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import json
import requests
import logging
from typing import List, Dict, Any, Optional
from requests.auth import HTTPBasicAuth

from signalwire_agents.core.skill_base import SkillBase
from signalwire_agents.core.function_result import SwaigFunctionResult

logger = logging.getLogger(__name__)


class MCPGatewaySkill(SkillBase):
    """
    MCP Gateway Skill - Bridge MCP servers with SWAIG functions
    
    This skill connects SignalWire agents to MCP (Model Context Protocol) servers
    through a gateway service, dynamically creating SWAIG functions for MCP tools.
    """
    
    SKILL_NAME = "mcp_gateway"
    SKILL_DESCRIPTION = "Bridge MCP servers with SWAIG functions"
    SKILL_VERSION = "1.0.0"
    REQUIRED_PACKAGES = ["requests"]
    REQUIRED_ENV_VARS = []
    
    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Get parameter schema for MCP Gateway skill"""
        schema = super().get_parameter_schema()
        schema.update({
            "gateway_url": {
                "type": "string",
                "description": "URL of the MCP Gateway service",
                "required": True
            },
            "auth_token": {
                "type": "string",
                "description": "Bearer token for authentication (alternative to basic auth)",
                "required": False,
                "hidden": True,
                "env_var": "MCP_GATEWAY_AUTH_TOKEN"
            },
            "auth_user": {
                "type": "string",
                "description": "Username for basic authentication (required if auth_token not provided)",
                "required": False,
                "env_var": "MCP_GATEWAY_AUTH_USER"
            },
            "auth_password": {
                "type": "string",
                "description": "Password for basic authentication (required if auth_token not provided)",
                "required": False,
                "hidden": True,
                "env_var": "MCP_GATEWAY_AUTH_PASSWORD"
            },
            "services": {
                "type": "array",
                "description": "List of MCP services to connect to (empty for all available)",
                "default": [],
                "required": False,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Service name"
                        },
                        "tools": {
                            "type": ["string", "array"],
                            "description": "Tools to expose ('*' for all, or list of tool names)"
                        }
                    }
                }
            },
            "session_timeout": {
                "type": "integer",
                "description": "Session timeout in seconds",
                "default": 300,
                "required": False
            },
            "tool_prefix": {
                "type": "string",
                "description": "Prefix for registered SWAIG function names",
                "default": "mcp_",
                "required": False
            },
            "retry_attempts": {
                "type": "integer",
                "description": "Number of retry attempts for failed requests",
                "default": 3,
                "required": False
            },
            "request_timeout": {
                "type": "integer",
                "description": "Request timeout in seconds",
                "default": 30,
                "required": False
            },
            "verify_ssl": {
                "type": "boolean",
                "description": "Verify SSL certificates",
                "default": True,
                "required": False
            }
        })
        return schema
    
    def setup(self) -> bool:
        """Setup and validate skill configuration"""
        # Check for auth method - either token or basic auth
        self.auth_token = self.params.get('auth_token')
        if not self.auth_token:
            # Require basic auth if no token
            required_params = ['gateway_url', 'auth_user', 'auth_password']
            missing_params = [param for param in required_params if not self.params.get(param)]
            if missing_params:
                self.logger.error(f"Missing required parameters: {missing_params}")
                return False
            self.auth = HTTPBasicAuth(self.params['auth_user'], self.params['auth_password'])
        else:
            # Just need gateway URL with token auth
            if not self.params.get('gateway_url'):
                self.logger.error("Missing required parameter: gateway_url")
                return False
            self.auth = None
        
        # Store configuration
        self.gateway_url = self.params['gateway_url'].rstrip('/')
        self.services = self.params.get('services', [])
        self.session_timeout = self.params.get('session_timeout', 300)
        self.tool_prefix = self.params.get('tool_prefix', 'mcp_')
        self.retry_attempts = self.params.get('retry_attempts', 3)
        self.request_timeout = self.params.get('request_timeout', 30)
        self.verify_ssl = self.params.get('verify_ssl', True)
        
        # Session ID will be set from call_id when first tool is used
        self.session_id = None
        
        # Validate gateway connection
        try:
            response = requests.get(
                f"{self.gateway_url}/health",
                timeout=self.request_timeout,
                verify=self.verify_ssl
            )
            response.raise_for_status()
            self.logger.info(f"Connected to MCP Gateway at {self.gateway_url}")
        except Exception as e:
            self.logger.error(f"Failed to connect to gateway: {e}")
            return False
        
        return True
    
    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make HTTP request with appropriate authentication"""
        headers = kwargs.get('headers', {})
        if self.auth_token:
            headers['Authorization'] = f'Bearer {self.auth_token}'
        kwargs['headers'] = headers
        
        if not self.auth_token:
            kwargs['auth'] = self.auth
        
        kwargs['timeout'] = kwargs.get('timeout', self.request_timeout)
        kwargs['verify'] = kwargs.get('verify', self.verify_ssl)
        
        return requests.request(method, url, **kwargs)
    
    def register_tools(self) -> None:
        """Register SWAIG tools from MCP services"""
        # If no services specified, get all available
        if not self.services:
            try:
                response = self._make_request('GET', f"{self.gateway_url}/services")
                response.raise_for_status()
                all_services = response.json()
                self.services = [{"name": name} for name in all_services.keys()]
            except Exception as e:
                self.logger.error(f"Failed to list services: {e}")
                return
        
        # Process each service
        for service_config in self.services:
            service_name = service_config.get('name')
            if not service_name:
                continue
            
            # Get tools for this service
            try:
                response = self._make_request('GET', f"{self.gateway_url}/services/{service_name}/tools")
                response.raise_for_status()
                tools_data = response.json()
                tools = tools_data.get('tools', [])
                
                # Filter tools if specified
                tool_filter = service_config.get('tools', '*')
                if tool_filter != '*' and isinstance(tool_filter, list):
                    tools = [t for t in tools if t['name'] in tool_filter]
                
                # Register each tool as a SWAIG function
                for tool in tools:
                    self._register_mcp_tool(service_name, tool)
                    
            except Exception as e:
                self.logger.error(f"Failed to get tools for service '{service_name}': {e}")
        
        # Register the hangup hook for session cleanup
        self.define_tool(
            name="_mcp_gateway_hangup",
            description="Internal cleanup function for MCP sessions",
            parameters={},
            handler=self._hangup_handler,
            is_hangup_hook=True
        )
    
    def _register_mcp_tool(self, service_name: str, tool_def: Dict[str, Any]):
        """Register a single MCP tool as a SWAIG function"""
        tool_name = tool_def.get('name')
        if not tool_name:
            return
        
        # Create SWAIG function name
        swaig_name = f"{self.tool_prefix}{service_name}_{tool_name}"
        
        # Build SWAIG parameters from MCP input schema
        input_schema = tool_def.get('inputSchema', {})
        properties = input_schema.get('properties', {})
        required = input_schema.get('required', [])
        
        # Convert MCP schema to SWAIG parameters
        swaig_params = {}
        for prop_name, prop_def in properties.items():
            param_def = {
                "type": prop_def.get('type', 'string'),
                "description": prop_def.get('description', '')
            }
            
            # Add enum if present
            if 'enum' in prop_def:
                param_def['enum'] = prop_def['enum']
            
            # Add default if present and not required
            if 'default' in prop_def and prop_name not in required:
                param_def['default'] = prop_def['default']
            
            swaig_params[prop_name] = param_def
        
        # Create handler function
        def handler(args, raw_data):
            return self._call_mcp_tool(service_name, tool_name, args, raw_data)
        
        # Register the SWAIG function
        self.define_tool(
            name=swaig_name,
            description=f"[{service_name}] {tool_def.get('description', tool_name)}",
            parameters=swaig_params,
            handler=handler
        )
        
        self.logger.info(f"Registered SWAIG function: {swaig_name}")
    
    def _call_mcp_tool(self, service_name: str, tool_name: str, args: Dict[str, Any], 
                       raw_data: Dict[str, Any]) -> SwaigFunctionResult:
        """Call an MCP tool through the gateway"""
        # Check for mcp_call_id in global_data first, then fall back to top-level call_id
        global_data = raw_data.get('global_data', {})
        if 'mcp_call_id' in global_data:
            self.session_id = global_data['mcp_call_id']
            self.logger.info(f"Using session ID from global_data.mcp_call_id: {self.session_id}")
        else:
            self.session_id = raw_data.get('call_id', 'unknown')
            self.logger.info(f"Using session ID from call_id: {self.session_id}")
        self.logger.debug(f"Raw data keys: {list(raw_data.keys())}")
        if 'global_data' in raw_data:
            self.logger.debug(f"global_data keys: {list(global_data.keys())}")
        
        # Prepare request
        request_data = {
            "tool": tool_name,
            "arguments": args,
            "session_id": self.session_id,
            "timeout": self.session_timeout,
            "metadata": {
                "agent_id": self.agent.name,
                "timestamp": raw_data.get('timestamp'),
                "call_id": raw_data.get('call_id')
            }
        }
        
        # Call the gateway with retries
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                response = self._make_request(
                    'POST',
                    f"{self.gateway_url}/services/{service_name}/call",
                    json=request_data
                )
                
                if response.status_code == 200:
                    result_data = response.json()
                    result_text = result_data.get('result', 'No response')
                    
                    # Create SWAIG result
                    return SwaigFunctionResult(result_text)
                
                else:
                    error_data = response.json()
                    error_msg = error_data.get('error', f'HTTP {response.status_code}')
                    last_error = error_msg
                    
                    if response.status_code >= 500:
                        # Server error, retry
                        self.logger.warning(f"Gateway error (attempt {attempt + 1}): {error_msg}")
                        continue
                    else:
                        # Client error, don't retry
                        break
                        
            except requests.exceptions.Timeout:
                last_error = "Request timeout"
                self.logger.warning(f"Timeout calling MCP tool (attempt {attempt + 1})")
                
            except requests.exceptions.ConnectionError:
                last_error = "Connection error"
                self.logger.warning(f"Connection error (attempt {attempt + 1})")
                
            except Exception as e:
                last_error = str(e)
                self.logger.error(f"Unexpected error: {e}")
                break
        
        # All attempts failed
        error_msg = f"Failed to call {service_name}.{tool_name}: {last_error}"
        self.logger.error(error_msg)
        return SwaigFunctionResult(error_msg)
    
    def _hangup_handler(self, args: Dict[str, Any], raw_data: Dict[str, Any]) -> SwaigFunctionResult:
        """Handle call hangup - cleanup MCP session"""
        # Check for mcp_call_id in global_data first, then fall back to top-level call_id
        global_data = raw_data.get('global_data', {})
        if 'mcp_call_id' in global_data:
            session_id = global_data['mcp_call_id']
            self.logger.info(f"Cleanup using session ID from global_data.mcp_call_id: {session_id}")
        else:
            session_id = raw_data.get('call_id', 'unknown')
            self.logger.info(f"Cleanup using session ID from call_id: {session_id}")
        
        try:
            response = self._make_request('DELETE', f"{self.gateway_url}/sessions/{session_id}")
            
            if response.status_code in [200, 404]:
                self.logger.info(f"Cleaned up MCP session: {session_id}")
            else:
                self.logger.warning(f"Failed to cleanup session: HTTP {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up session: {e}")
        
        return SwaigFunctionResult("Session cleanup complete")
    
    def get_hints(self) -> List[str]:
        """Return speech recognition hints"""
        hints = ["MCP", "gateway"]
        
        # Add service names as hints
        for service in self.services:
            if isinstance(service, dict) and 'name' in service:
                hints.append(service['name'])
        
        return hints
    
    def get_global_data(self) -> Dict[str, Any]:
        """Return global data for DataMap variables"""
        return {
            "mcp_gateway_url": self.gateway_url,
            "mcp_session_id": self.session_id,
            "mcp_services": [s.get('name') if isinstance(s, dict) else str(s) 
                            for s in self.services]
        }
    
    def get_prompt_sections(self) -> List[Dict[str, Any]]:
        """Return prompt sections to add to agent"""
        sections = []
        
        # Build service list for prompt
        service_descriptions = []
        for service in self.services:
            if isinstance(service, dict):
                name = service.get('name', 'Unknown')
                tools = service.get('tools', '*')
                if tools == '*':
                    service_descriptions.append(f"{name} (all tools)")
                elif isinstance(tools, list):
                    service_descriptions.append(f"{name} ({len(tools)} tools)")
            else:
                service_descriptions.append(str(service))
        
        if service_descriptions:
            sections.append({
                "title": "MCP Gateway Integration",
                "body": "You have access to external MCP (Model Context Protocol) services through a gateway.",
                "bullets": [
                    f"Connected to gateway at {self.gateway_url}",
                    f"Available services: {', '.join(service_descriptions)}",
                    f"Functions are prefixed with '{self.tool_prefix}' followed by service name",
                    "Each service maintains its own session state throughout the call"
                ]
            })
        
        return sections