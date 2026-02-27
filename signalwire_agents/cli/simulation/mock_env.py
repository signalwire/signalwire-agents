#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Mock environment and serverless simulation functionality
"""

import os
import json
from typing import Optional, Dict, Any
from ..types import PostData


class MockQueryParams:
    """Mock FastAPI QueryParams (simple dict-like)"""
    def __init__(self, params: Optional[Dict[str, str]] = None):
        self._params = params or {}
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self._params.get(key, default)
    
    def __getitem__(self, key: str) -> str:
        return self._params[key]
    
    def __contains__(self, key: str) -> bool:
        return key in self._params
    
    def items(self):
        return self._params.items()
    
    def keys(self):
        return self._params.keys()
    
    def values(self):
        return self._params.values()


class MockHeaders:
    """Mock FastAPI Headers (case-insensitive dict-like)"""
    def __init__(self, headers: Optional[Dict[str, str]] = None):
        # Store headers with lowercase keys for case-insensitive lookup
        self._headers = {}
        if headers:
            for k, v in headers.items():
                self._headers[k.lower()] = v
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self._headers.get(key.lower(), default)
    
    def __getitem__(self, key: str) -> str:
        return self._headers[key.lower()]
    
    def __contains__(self, key: str) -> bool:
        return key.lower() in self._headers
    
    def items(self):
        return self._headers.items()
    
    def keys(self):
        return self._headers.keys()
    
    def values(self):
        return self._headers.values()


class MockURL:
    """Mock FastAPI URL object"""
    def __init__(self, url: str = "http://localhost:8080/swml"):
        self._url = url
        # Parse basic components
        if "?" in url:
            self.path, query_string = url.split("?", 1)
            self.query = query_string
        else:
            self.path = url
            self.query = ""
        
        # Extract scheme and netloc
        if "://" in url:
            self.scheme, rest = url.split("://", 1)
            if "/" in rest:
                self.netloc = rest.split("/", 1)[0]
            else:
                self.netloc = rest
        else:
            self.scheme = "http"
            self.netloc = "localhost:8080"
    
    def __str__(self):
        return self._url


class MockRequest:
    """Mock FastAPI Request object for dynamic agent testing"""
    def __init__(self, method: str = "POST", url: str = "http://localhost:8080/swml",
                 headers: Optional[Dict[str, str]] = None,
                 query_params: Optional[Dict[str, str]] = None,
                 json_body: Optional[Dict[str, Any]] = None):
        self.method = method
        self.url = MockURL(url)
        self.headers = MockHeaders(headers)
        self.query_params = MockQueryParams(query_params)
        self._json_body = json_body or {}
        self._body = json.dumps(self._json_body).encode('utf-8')
        # Add state object for request state (used by FastAPI)
        self.state = type('State', (), {})()
    
    async def json(self) -> Dict[str, Any]:
        """Return the JSON body"""
        return self._json_body
    
    async def body(self) -> bytes:
        """Return the raw body bytes"""
        return self._body
    
    def client(self):
        """Mock client property"""
        return type('MockClient', (), {'host': '127.0.0.1', 'port': 0})()


def create_mock_request(method: str = "POST", url: str = "http://localhost:8080/swml",
                       headers: Optional[Dict[str, str]] = None,
                       query_params: Optional[Dict[str, str]] = None,
                       body: Optional[Dict[str, Any]] = None) -> MockRequest:
    """
    Factory function to create a mock FastAPI Request object
    """
    return MockRequest(method=method, url=url, headers=headers, 
                      query_params=query_params, json_body=body)


class ServerlessSimulator:
    """Manages serverless environment simulation for different platforms"""
    
    # Default environment presets for each platform
    PLATFORM_PRESETS = {
        'lambda': {
            'AWS_LAMBDA_FUNCTION_NAME': 'test-agent-function',
            'AWS_LAMBDA_FUNCTION_URL': 'https://abc123.lambda-url.us-east-1.on.aws/',
            'AWS_REGION': 'us-east-1',
            '_HANDLER': 'lambda_function.lambda_handler'
        },
        'cgi': {
            'GATEWAY_INTERFACE': 'CGI/1.1',
            'HTTP_HOST': 'example.com',
            'SCRIPT_NAME': '/cgi-bin/agent.cgi',
            'HTTPS': 'on',
            'SERVER_NAME': 'example.com'
        },
        'cloud_function': {
            'GOOGLE_CLOUD_PROJECT': 'test-project',
            'FUNCTION_URL': 'https://my-function-abc123.cloudfunctions.net',
            'GOOGLE_CLOUD_REGION': 'us-central1',
            'K_SERVICE': 'agent'
        },
        'azure_function': {
            'AZURE_FUNCTIONS_ENVIRONMENT': 'Development',
            'FUNCTIONS_WORKER_RUNTIME': 'python',
            'WEBSITE_SITE_NAME': 'my-function-app'
        }
    }
    
    def __init__(self, platform: str, overrides: Optional[Dict[str, str]] = None):
        self.platform = platform
        self.original_env = dict(os.environ)
        self.preset_env = self.PLATFORM_PRESETS.get(platform, {}).copy()
        self.overrides = overrides or {}
        self.active = False
        self._cleared_vars = {}
    
    def activate(self, verbose: bool = False):
        """Apply serverless environment simulation"""
        if self.active:
            return
            
        # Clear conflicting environment variables
        self._clear_conflicting_env()
        
        # Apply preset environment
        os.environ.update(self.preset_env)
        
        # Apply user overrides
        os.environ.update(self.overrides)
        
        # Set appropriate logging mode for serverless simulation
        if self.platform == 'cgi' and 'SIGNALWIRE_LOG_MODE' not in self.overrides:
            # CGI mode should default to 'off' unless explicitly overridden
            os.environ['SIGNALWIRE_LOG_MODE'] = 'off'
        
        self.active = True
        
        if verbose:
            print(f"✓ Activated {self.platform} environment simulation")
            
            # Debug: Show key environment variables
            if self.platform == 'lambda':
                print(f"  AWS_LAMBDA_FUNCTION_NAME: {os.environ.get('AWS_LAMBDA_FUNCTION_NAME')}")
                print(f"  AWS_LAMBDA_FUNCTION_URL: {os.environ.get('AWS_LAMBDA_FUNCTION_URL')}")
                print(f"  AWS_REGION: {os.environ.get('AWS_REGION')}")
            elif self.platform == 'cgi':
                print(f"  GATEWAY_INTERFACE: {os.environ.get('GATEWAY_INTERFACE')}")
                print(f"  HTTP_HOST: {os.environ.get('HTTP_HOST')}")
                print(f"  SCRIPT_NAME: {os.environ.get('SCRIPT_NAME')}")
                print(f"  SIGNALWIRE_LOG_MODE: {os.environ.get('SIGNALWIRE_LOG_MODE')}")
            elif self.platform == 'cloud_function':
                print(f"  GOOGLE_CLOUD_PROJECT: {os.environ.get('GOOGLE_CLOUD_PROJECT')}")
                print(f"  FUNCTION_URL: {os.environ.get('FUNCTION_URL')}")
                print(f"  GOOGLE_CLOUD_REGION: {os.environ.get('GOOGLE_CLOUD_REGION')}")
            elif self.platform == 'azure_function':
                print(f"  AZURE_FUNCTIONS_ENVIRONMENT: {os.environ.get('AZURE_FUNCTIONS_ENVIRONMENT')}")
                print(f"  WEBSITE_SITE_NAME: {os.environ.get('WEBSITE_SITE_NAME')}")
            
            # Debug: Confirm SWML_PROXY_URL_BASE is cleared
            proxy_url = os.environ.get('SWML_PROXY_URL_BASE')
            if proxy_url:
                print(f"  WARNING: SWML_PROXY_URL_BASE still set: {proxy_url}")
            else:
                print(f"  ✓ SWML_PROXY_URL_BASE cleared successfully")
    
    def deactivate(self, verbose: bool = False):
        """Restore original environment"""
        if not self.active:
            return
            
        os.environ.clear()
        os.environ.update(self.original_env)
        self.active = False
        
        if verbose:
            print(f"✓ Deactivated {self.platform} environment simulation")
    
    def _clear_conflicting_env(self):
        """Clear environment variables that might conflict with simulation"""
        # Remove variables from other platforms
        conflicting_vars = []
        for platform, preset in self.PLATFORM_PRESETS.items():
            if platform != self.platform:
                conflicting_vars.extend(preset.keys())
        
        # Always clear SWML_PROXY_URL_BASE during serverless simulation
        # so that platform-specific URL generation takes precedence
        conflicting_vars.append('SWML_PROXY_URL_BASE')
        
        for var in conflicting_vars:
            if var in os.environ:
                self._cleared_vars[var] = os.environ[var]
                os.environ.pop(var)
    
    def add_override(self, key: str, value: str):
        """Add an environment variable override"""
        self.overrides[key] = value
        if self.active:
            os.environ[key] = value
    
    def get_current_env(self) -> Dict[str, str]:
        """Get the current environment that would be applied"""
        env = self.preset_env.copy()
        env.update(self.overrides)
        return env


def load_env_file(env_file_path: str) -> Dict[str, str]:
    """Load environment variables from a file"""
    env_vars = {}
    if not os.path.exists(env_file_path):
        raise FileNotFoundError(f"Environment file not found: {env_file_path}")
    
    with open(env_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
    
    return env_vars