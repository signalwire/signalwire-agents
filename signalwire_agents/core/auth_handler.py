"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import secrets
from typing import Optional, Tuple, Dict, Any, Callable
from functools import wraps

try:
    from fastapi import HTTPException, Depends
    from fastapi.security import HTTPBasic, HTTPBasicCredentials, HTTPBearer, HTTPAuthorizationCredentials
except ImportError:
    HTTPException = None
    Depends = None
    HTTPBasic = None
    HTTPBasicCredentials = None
    HTTPBearer = None
    HTTPAuthorizationCredentials = None

from signalwire_agents.core.logging_config import get_logger

logger = get_logger("auth_handler")


class AuthHandler:
    """
    Unified authentication handler supporting multiple auth methods.
    
    This class provides a clean pattern for handling Basic Auth, Bearer tokens,
    and API keys across all SignalWire services.
    """
    
    def __init__(self, security_config: 'SecurityConfig'):
        """
        Initialize auth handler with security configuration.
        
        Args:
            security_config: SecurityConfig instance with auth settings
        """
        self.security_config = security_config
        self.basic_auth = HTTPBasic(auto_error=False) if HTTPBasic else None
        self.bearer_auth = HTTPBearer(auto_error=False) if HTTPBearer else None
        
        # Get auth methods from config
        self._setup_auth_methods()
    
    def _setup_auth_methods(self):
        """Setup enabled authentication methods from config"""
        self.auth_methods = {}
        
        # Basic auth (always available for backward compatibility)
        username, password = self.security_config.get_basic_auth()
        self.auth_methods['basic'] = {
            'enabled': True,
            'username': username,
            'password': password
        }
        
        # Bearer token (if configured)
        bearer_token = getattr(self.security_config, 'bearer_token', None)
        if bearer_token:
            self.auth_methods['bearer'] = {
                'enabled': True,
                'token': bearer_token
            }
        
        # API key (if configured)
        api_key = getattr(self.security_config, 'api_key', None)
        if api_key:
            self.auth_methods['api_key'] = {
                'enabled': True,
                'key': api_key,
                'header': getattr(self.security_config, 'api_key_header', 'X-API-Key')
            }
    
    def verify_basic_auth(self, credentials: HTTPBasicCredentials) -> bool:
        """Verify basic auth credentials"""
        if not self.auth_methods.get('basic', {}).get('enabled'):
            return False
            
        basic_config = self.auth_methods['basic']
        username_correct = secrets.compare_digest(
            credentials.username, basic_config['username']
        )
        password_correct = secrets.compare_digest(
            credentials.password, basic_config['password']
        )
        
        return username_correct and password_correct
    
    def verify_bearer_token(self, credentials: HTTPAuthorizationCredentials) -> bool:
        """Verify bearer token"""
        if not self.auth_methods.get('bearer', {}).get('enabled'):
            return False
            
        bearer_config = self.auth_methods['bearer']
        return secrets.compare_digest(
            credentials.credentials, bearer_config['token']
        )
    
    def verify_api_key(self, api_key: str) -> bool:
        """Verify API key"""
        if not self.auth_methods.get('api_key', {}).get('enabled'):
            return False
            
        api_config = self.auth_methods['api_key']
        return secrets.compare_digest(api_key, api_config['key'])
    
    def get_fastapi_dependency(self, optional: bool = False):
        """
        Get FastAPI dependency for authentication.
        
        Args:
            optional: If True, authentication is optional
            
        Returns:
            FastAPI dependency function
        """
        if not Depends:
            return None
            
        async def auth_dependency(
            basic_credentials: Optional[HTTPBasicCredentials] = Depends(self.basic_auth) if self.basic_auth else None,
            bearer_credentials: Optional[HTTPAuthorizationCredentials] = Depends(self.bearer_auth) if self.bearer_auth else None,
            api_key: Optional[str] = None  # Get from header in request
        ):
            # Try each auth method
            authenticated = False
            auth_method = None
            
            # Try bearer token first (if provided)
            if bearer_credentials and self.verify_bearer_token(bearer_credentials):
                authenticated = True
                auth_method = 'bearer'
            
            # Try basic auth
            elif basic_credentials and self.verify_basic_auth(basic_credentials):
                authenticated = True
                auth_method = 'basic'
            
            # Try API key (would need to be extracted from request headers)
            # This is a simplified version - in practice, you'd get it from request
            
            if not authenticated and not optional:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Basic"},
                )
            
            return {'authenticated': authenticated, 'method': auth_method}
        
        return auth_dependency
    
    def flask_decorator(self, f: Callable) -> Callable:
        """
        Flask decorator for authentication.
        
        This provides compatibility with Flask-based services like MCP Gateway.
        """
        @wraps(f)
        def decorated(*args, **kwargs):
            from flask import request, Response
            
            # Try Bearer token first
            auth_header = request.headers.get('Authorization', '')
            
            if auth_header.startswith('Bearer ') and self.auth_methods.get('bearer', {}).get('enabled'):
                token = auth_header[7:]
                if secrets.compare_digest(token, self.auth_methods['bearer']['token']):
                    return f(*args, **kwargs)
            
            # Try API key
            if self.auth_methods.get('api_key', {}).get('enabled'):
                api_config = self.auth_methods['api_key']
                api_key = request.headers.get(api_config['header'])
                if api_key and secrets.compare_digest(api_key, api_config['key']):
                    return f(*args, **kwargs)
            
            # Fall back to Basic auth
            auth = request.authorization
            if auth and self.auth_methods.get('basic', {}).get('enabled'):
                basic_config = self.auth_methods['basic']
                if secrets.compare_digest(auth.username, basic_config['username']) and \
                   secrets.compare_digest(auth.password, basic_config['password']):
                    return f(*args, **kwargs)
            
            # Authentication failed
            logger.warning(
                "auth_failed",
                ip=request.remote_addr,
                method=request.method,
                path=request.path
            )
            
            return Response(
                'Authentication required',
                401,
                {'WWW-Authenticate': 'Basic realm="SignalWire Service"'}
            )
        
        return decorated
    
    def get_auth_info(self) -> Dict[str, Any]:
        """Get information about configured auth methods"""
        info = {}
        
        if self.auth_methods.get('basic', {}).get('enabled'):
            info['basic'] = {
                'enabled': True,
                'username': self.auth_methods['basic']['username']
            }
        
        if self.auth_methods.get('bearer', {}).get('enabled'):
            info['bearer'] = {
                'enabled': True,
                'hint': 'Use Authorization: Bearer <token>'
            }
        
        if self.auth_methods.get('api_key', {}).get('enabled'):
            api_config = self.auth_methods['api_key']
            info['api_key'] = {
                'enabled': True,
                'header': api_config['header'],
                'hint': f'Use {api_config["header"]}: <key>'
            }
        
        return info