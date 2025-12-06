#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
MCP Server Manager for MCP Gateway

Handles spawning, communication, and management of MCP server processes.
Implements the MCP protocol client functionality.
"""

import os
import sys
import subprocess
import json
import threading
import queue
import logging
import time
import pwd
import tempfile
import shutil
import resource
import select
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MCPService:
    """Configuration for an MCP service"""
    name: str
    command: List[str]
    description: str
    enabled: bool = True
    sandbox_config: Dict[str, Any] = None
    
    def __post_init__(self):
        # Default sandbox config if not provided
        if self.sandbox_config is None:
            self.sandbox_config = {
                'enabled': True,
                'resource_limits': True,
                'restricted_env': True
            }
    
    def __hash__(self):
        return hash(self.name)


class MCPClient:
    """Client for communicating with a single MCP server process"""
    
    def __init__(self, service: MCPService, sandbox_base_dir: str = './sandbox'):
        self.service = service
        self.process = None
        self.request_id = 0
        self.pending_requests = {}
        self.response_queue = queue.Queue()
        self.reader_thread = None
        self.lock = threading.Lock()
        self.tools = []
        self.sandbox_base_dir = sandbox_base_dir
        self.sandbox_dir = None
        self._shutdown = threading.Event()
        
    def _setup_sandbox_env(self) -> Tuple[Dict[str, str], Optional[str]]:
        """Create environment for the MCP process based on sandbox config
        
        Returns:
            Tuple of (environment dict, working directory)
        """
        sandbox_config = self.service.sandbox_config
        
        # Check if sandboxing is disabled
        if not sandbox_config.get('enabled', True):
            logger.info(f"Sandboxing disabled for '{self.service.name}'")
            return os.environ.copy(), sandbox_config.get('working_dir', os.getcwd())
        
        # Create a subdirectory in the sandbox base for this process
        self.sandbox_dir = os.path.join(self.sandbox_base_dir, f"mcp_{self.service.name}_{os.getpid()}")
        os.makedirs(self.sandbox_dir, exist_ok=True)
        
        # Start with appropriate environment
        if sandbox_config.get('restricted_env', True):
            # Restricted environment
            env = {
                'PATH': os.environ.get('PATH', '/usr/bin:/bin'),
                'HOME': self.sandbox_dir,
                'TMPDIR': self.sandbox_dir,
                'TEMP': self.sandbox_dir,
                'TMP': self.sandbox_dir,
                'USER': os.environ.get('USER', 'nobody'),
                'LANG': 'C.UTF-8',
            }
            # Copy only essential env vars if they exist
            for key in ['PYTHONPATH', 'NODE_PATH', 'JAVA_HOME']:
                if key in os.environ:
                    env[key] = os.environ[key]
        else:
            # Full environment with some overrides
            env = os.environ.copy()
            env['HOME'] = self.sandbox_dir
            env['TMPDIR'] = self.sandbox_dir
            env['TEMP'] = self.sandbox_dir
            env['TMP'] = self.sandbox_dir
        
        # Always remove potentially dangerous env vars
        dangerous_vars = ['LD_PRELOAD', 'LD_LIBRARY_PATH', 'DYLD_INSERT_LIBRARIES']
        for var in dangerous_vars:
            env.pop(var, None)
        
        # Determine working directory
        working_dir = sandbox_config.get('working_dir', os.getcwd())
        
        return env, working_dir
    
    def _sandbox_preexec(self):
        """Pre-exec function to sandbox the process"""
        sandbox_config = self.service.sandbox_config
        
        # Check if sandboxing is disabled
        if not sandbox_config.get('enabled', True):
            return
        
        # Check if resource limits are enabled
        if sandbox_config.get('resource_limits', True):
            try:
                # Set resource limits
                # Limit CPU time (300 seconds)
                resource.setrlimit(resource.RLIMIT_CPU, (300, 300))
                # Limit memory (512MB)
                resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
                # Limit number of processes (10)
                resource.setrlimit(resource.RLIMIT_NPROC, (10, 10))
                # Limit file size (10MB)
                resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
            except Exception as e:
                # Log but don't fail - some systems might not support all operations
                logger.warning(f"Resource limit warning: {e}")
        
        # Drop privileges if running as root (currently disabled)
        # NOTE: Disabled for now - needs more testing with MCP servers
        # if os.getuid() == 0:
        #     nobody_uid = pwd.getpwnam('nobody').pw_uid
        #     os.setgroups([])  # Remove supplementary groups
        #     os.setgid(nobody_uid)  # Set GID first
        #     os.setuid(nobody_uid)  # Then UID
    
    def start(self) -> bool:
        """Start the MCP server process and initialize connection"""
        try:
            logger.info(f"Starting MCP service '{self.service.name}' with command: {self.service.command}")
            
            # Set up environment and working directory
            env, working_dir = self._setup_sandbox_env()
            
            # Log the command we're trying to run
            logger.info(f"Starting command: {' '.join(self.service.command)}")
            logger.info(f"Working directory: {working_dir}")
            if self.sandbox_dir:
                logger.info(f"Sandbox directory: {self.sandbox_dir}")
            
            # Check if we should use preexec function
            use_preexec = self.service.sandbox_config.get('enabled', True) and sys.platform != 'win32'
            
            self.process = subprocess.Popen(
                self.service.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                cwd=working_dir,
                env=env,
                preexec_fn=self._sandbox_preexec if use_preexec else None
            )
            
            # Start reader thread
            self.reader_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.reader_thread.start()
            
            # Initialize the MCP session
            result = self._initialize()
            if not result:
                logger.error(f"Failed to initialize MCP service '{self.service.name}'")
                self.stop()
                return False
            
            # Get available tools
            self.tools = self._list_tools()
            logger.info(f"MCP service '{self.service.name}' started with {len(self.tools)} tools")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting MCP service '{self.service.name}': {e}")
            return False
    
    def stop(self):
        """Stop the MCP server process and clean up sandbox"""
        # Signal shutdown to reader thread
        self._shutdown.set()
        
        if self.process:
            try:
                # Try graceful shutdown first (with short timeout)
                try:
                    # Send shutdown with very short timeout
                    with self.lock:
                        request_id = self.request_id
                        self.request_id += 1
                        request = {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "method": "shutdown",
                            "params": {}
                        }
                    self._send_message(request)
                    time.sleep(0.2)  # Very brief wait
                except:
                    pass
                
                # Check if still running
                if self.process.poll() is None:
                    # Terminate process
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        # Force kill if terminate didn't work
                        logger.warning(f"Force killing MCP service '{self.service.name}'")
                        self.process.kill()
                        self.process.wait(timeout=1)
                
            except Exception as e:
                logger.error(f"Error stopping process: {e}")
                # Last resort - force kill
                try:
                    if self.process.poll() is None:
                        self.process.kill()
                except:
                    pass
            
            self.process = None
            logger.info(f"Stopped MCP service '{self.service.name}'")
        
        # Wait for reader thread to finish (with timeout)
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=1.0)
            if self.reader_thread.is_alive():
                logger.warning(f"Reader thread for '{self.service.name}' did not stop gracefully")
        
        # Clean up sandbox directory
        if hasattr(self, 'sandbox_dir') and self.sandbox_dir and os.path.exists(self.sandbox_dir):
            try:
                shutil.rmtree(self.sandbox_dir)
                logger.debug(f"Cleaned up sandbox directory: {self.sandbox_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up sandbox directory: {e}")
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
        return self.call_method("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
    
    def call_method(self, method: str, params: Dict[str, Any]) -> Any:
        """Call an RPC method and wait for response"""
        if self._shutdown.is_set():
            raise RuntimeError("Client is shutting down")
            
        with self.lock:
            request_id = self.request_id
            self.request_id += 1
            
            request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params
            }
            
            # Create future for response
            future = threading.Event()
            self.pending_requests[request_id] = {"event": future, "response": None}
            
        # Send request
        self._send_message(request)
        
        # Wait for response (with timeout)
        if not future.wait(timeout=30):
            with self.lock:
                self.pending_requests.pop(request_id, None)
            raise TimeoutError(f"Timeout waiting for response to {method}")
        
        # Get response
        with self.lock:
            response_data = self.pending_requests.pop(request_id)
            response = response_data["response"]
        
        if "error" in response:
            raise Exception(f"MCP Error: {response['error']}")
        
        return response.get("result")
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get the list of available tools"""
        return self.tools.copy()
    
    def _send_message(self, message: Dict[str, Any]):
        """Send a JSON-RPC message to the server"""
        if not self.process or self.process.poll() is not None:
            raise RuntimeError("MCP server process is not running")
        
        json_str = json.dumps(message)
        self.process.stdin.write(json_str + '\n')
        self.process.stdin.flush()
        logger.debug(f"Sent to '{self.service.name}': {json_str}")
    
    def _read_loop(self):
        """Background thread to read responses from the MCP server"""
        while not self._shutdown.is_set() and self.process and self.process.poll() is None:
            try:
                # Simple blocking read - the thread will be interrupted on shutdown
                line = self.process.stdout.readline()
                if not line:
                    break
                
                try:
                    message = json.loads(line.strip())
                    logger.debug(f"Received from '{self.service.name}': {message}")
                    
                    # Handle response
                    if "id" in message:
                        request_id = message["id"]
                        with self.lock:
                            if request_id in self.pending_requests:
                                self.pending_requests[request_id]["response"] = message
                                self.pending_requests[request_id]["event"].set()
                    else:
                        # Notification - log it
                        logger.info(f"Notification from '{self.service.name}': {message}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from '{self.service.name}': {line.strip()}")
                    
            except Exception as e:
                if not self._shutdown.is_set():
                    logger.error(f"Error in read loop for '{self.service.name}': {e}")
                break
        
        # Check if process died and capture any errors
        if self.process and self.process.poll() is not None and not self._shutdown.is_set():
            try:
                stderr_output = self.process.stderr.read()
                if stderr_output:
                    logger.error(f"Process '{self.service.name}' exited with code {self.process.returncode}")
                    logger.error(f"Stderr: {stderr_output}")
            except:
                pass
        
        logger.info(f"Read loop ended for '{self.service.name}'")
    
    def _initialize(self) -> bool:
        """Initialize the MCP session"""
        try:
            result = self.call_method("initialize", {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {
                    "name": "mcp-gateway",
                    "version": "1.0.0"
                }
            })
            
            logger.info(f"Initialized '{self.service.name}': {result.get('serverInfo', {})}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize '{self.service.name}': {e}")
            return False
    
    def _list_tools(self) -> List[Dict[str, Any]]:
        """Get the list of available tools from the server"""
        try:
            result = self.call_method("tools/list", {})
            return result.get("tools", [])
            
        except Exception as e:
            logger.error(f"Failed to list tools for '{self.service.name}': {e}")
            return []


class MCPManager:
    """Manages multiple MCP services and their lifecycles"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.services: Dict[str, MCPService] = {}
        self.clients: Dict[str, MCPClient] = {}
        
        # Get sandbox directory from config or use default
        self.sandbox_base_dir = config.get('session', {}).get('sandbox_dir', './sandbox')
        
        # Ensure sandbox directory exists
        os.makedirs(self.sandbox_base_dir, exist_ok=True)
        logger.info(f"Using sandbox base directory: {self.sandbox_base_dir}")
        
        # Load services from config
        self._load_services()
    
    def _load_services(self):
        """Load service definitions from configuration"""
        services_config = self.config.get('services', {})
        
        for name, service_data in services_config.items():
            if not service_data.get('enabled', True):
                logger.info(f"Service '{name}' is disabled, skipping")
                continue
            
            service = MCPService(
                name=name,
                command=service_data['command'],
                description=service_data.get('description', ''),
                enabled=service_data.get('enabled', True),
                sandbox_config=service_data.get('sandbox', None)
            )
            
            self.services[name] = service
            logger.info(f"Loaded service '{name}': {service.description}")
    
    def get_service(self, service_name: str) -> Optional[MCPService]:
        """Get a service definition by name"""
        return self.services.get(service_name)
    
    def list_services(self) -> Dict[str, Dict[str, Any]]:
        """List all available services"""
        result = {}
        
        for name, service in self.services.items():
            result[name] = {
                'description': service.description,
                'enabled': service.enabled,
                'command': service.command
            }
        
        return result
    
    def create_client(self, service_name: str) -> MCPClient:
        """Create a new MCP client for a service"""
        service = self.services.get(service_name)
        if not service:
            raise ValueError(f"Unknown service: {service_name}")
        
        if not service.enabled:
            raise ValueError(f"Service '{service_name}' is disabled")
        
        client = MCPClient(service, self.sandbox_base_dir)
        if not client.start():
            raise RuntimeError(f"Failed to start MCP service '{service_name}'")
        
        # Track active client for cleanup
        self.clients[f"{service_name}_{id(client)}"] = client
        
        return client
    
    def get_service_tools(self, service_name: str) -> List[Dict[str, Any]]:
        """Get tools for a service by starting a temporary instance"""
        client = None
        try:
            client = self.create_client(service_name)
            return client.get_tools()
        finally:
            if client:
                client.stop()
    
    def validate_services(self) -> Dict[str, bool]:
        """Validate that all services can be started"""
        results = {}
        
        for service_name in self.services:
            try:
                client = self.create_client(service_name)
                client.stop()
                results[service_name] = True
                logger.info(f"Service '{service_name}' validation: OK")
            except Exception as e:
                results[service_name] = False
                logger.error(f"Service '{service_name}' validation failed: {e}")
        
        return results
    
    def shutdown(self):
        """Shutdown all active MCP clients"""
        logger.info(f"Shutting down {len(self.clients)} active MCP clients")
        
        # Stop all clients
        for client_id, client in list(self.clients.items()):
            try:
                client.stop()
                self.clients.pop(client_id, None)
            except Exception as e:
                logger.error(f"Error stopping client {client_id}: {e}")