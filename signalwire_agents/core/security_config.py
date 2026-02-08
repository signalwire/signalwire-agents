"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import os
import secrets
from typing import Dict, Any, Optional, Tuple, List, Union
from signalwire_agents.core.logging_config import get_logger
from signalwire_agents.core.config_loader import ConfigLoader

logger = get_logger("security_config")


class SecurityConfig:
    """
    Unified security configuration for SignalWire services.
    
    This class provides centralized security settings that can be used by
    both SWML and Search services, ensuring consistent security behavior.
    """
    
    # Security environment variable names
    SSL_ENABLED = 'SWML_SSL_ENABLED'
    SSL_CERT_PATH = 'SWML_SSL_CERT_PATH'
    SSL_KEY_PATH = 'SWML_SSL_KEY_PATH'
    SSL_DOMAIN = 'SWML_DOMAIN'
    SSL_VERIFY_MODE = 'SWML_SSL_VERIFY_MODE'
    
    # Additional security settings
    ALLOWED_HOSTS = 'SWML_ALLOWED_HOSTS'
    CORS_ORIGINS = 'SWML_CORS_ORIGINS'
    MAX_REQUEST_SIZE = 'SWML_MAX_REQUEST_SIZE'
    RATE_LIMIT = 'SWML_RATE_LIMIT'
    REQUEST_TIMEOUT = 'SWML_REQUEST_TIMEOUT'
    USE_HSTS = 'SWML_USE_HSTS'
    HSTS_MAX_AGE = 'SWML_HSTS_MAX_AGE'
    
    # Authentication
    BASIC_AUTH_USER = 'SWML_BASIC_AUTH_USER'
    BASIC_AUTH_PASSWORD = 'SWML_BASIC_AUTH_PASSWORD'
    
    # Defaults (secure by default)
    DEFAULTS = {
        SSL_ENABLED: False,  # Off by default, but secure when enabled
        SSL_VERIFY_MODE: 'CERT_REQUIRED',
        ALLOWED_HOSTS: '*',  # Accept all hosts by default for backward compatibility
        CORS_ORIGINS: '*',  # Accept all origins by default for backward compatibility
        MAX_REQUEST_SIZE: 10 * 1024 * 1024,  # 10MB
        RATE_LIMIT: 60,  # Requests per minute
        REQUEST_TIMEOUT: 30,  # Seconds
        USE_HSTS: True,  # Enable HSTS when HTTPS is on
        HSTS_MAX_AGE: 31536000,  # 1 year
    }
    
    def __init__(self, config_file: Optional[str] = None, service_name: Optional[str] = None):
        """
        Initialize security configuration.
        
        Args:
            config_file: Optional path to config file
            service_name: Optional service name for finding service-specific config
        """
        # First, set defaults
        self._set_defaults()
        
        # Then load from environment variables (backward compatibility)
        self.load_from_env()
        
        # Finally, apply config file if available (highest priority)
        self._load_config_file(config_file, service_name)
    
    def _set_defaults(self):
        """Set default values for all configuration"""
        # SSL configuration
        self.ssl_enabled = self.DEFAULTS[self.SSL_ENABLED]
        self.ssl_cert_path = None
        self.ssl_key_path = None
        self.domain = None
        self.ssl_verify_mode = self.DEFAULTS[self.SSL_VERIFY_MODE]
        
        # Additional settings
        self.allowed_hosts = self._parse_list(self.DEFAULTS[self.ALLOWED_HOSTS])
        self.cors_origins = self._parse_list(self.DEFAULTS[self.CORS_ORIGINS])
        self.max_request_size = self.DEFAULTS[self.MAX_REQUEST_SIZE]
        self.rate_limit = self.DEFAULTS[self.RATE_LIMIT]
        self.request_timeout = self.DEFAULTS[self.REQUEST_TIMEOUT]
        self.use_hsts = self.DEFAULTS[self.USE_HSTS]
        self.hsts_max_age = self.DEFAULTS[self.HSTS_MAX_AGE]
        
        # Authentication
        self.basic_auth_user = None
        self.basic_auth_password = None
    
    def _load_config_file(self, config_file: Optional[str], service_name: Optional[str]):
        """Load configuration from config file if available"""
        # Find config file
        if not config_file:
            config_file = ConfigLoader.find_config_file(service_name)
        
        if not config_file:
            return
        
        # Load config
        config_loader = ConfigLoader([config_file])
        if not config_loader.has_config():
            return
        
        logger.info("loading_config_from_file", file=config_file)
        
        # Get security section
        security_config = config_loader.get_section('security')
        if not security_config:
            return
        
        # Apply security settings (config file takes precedence)
        if 'ssl_enabled' in security_config:
            self.ssl_enabled = security_config['ssl_enabled']
        
        if 'ssl_cert_path' in security_config:
            self.ssl_cert_path = security_config['ssl_cert_path']
            
        if 'ssl_key_path' in security_config:
            self.ssl_key_path = security_config['ssl_key_path']
            
        if 'domain' in security_config:
            self.domain = security_config['domain']
            
        if 'ssl_verify_mode' in security_config:
            self.ssl_verify_mode = security_config['ssl_verify_mode']
        
        # Additional settings
        if 'allowed_hosts' in security_config:
            self.allowed_hosts = self._parse_list(security_config['allowed_hosts'])
            
        if 'cors_origins' in security_config:
            self.cors_origins = self._parse_list(security_config['cors_origins'])
            
        if 'max_request_size' in security_config:
            self.max_request_size = int(security_config['max_request_size'])
            
        if 'rate_limit' in security_config:
            self.rate_limit = int(security_config['rate_limit'])
            
        if 'request_timeout' in security_config:
            self.request_timeout = int(security_config['request_timeout'])
            
        if 'use_hsts' in security_config:
            self.use_hsts = security_config['use_hsts']
            
        if 'hsts_max_age' in security_config:
            self.hsts_max_age = int(security_config['hsts_max_age'])
        
        # Authentication from config
        auth_config = security_config.get('auth', {})
        if isinstance(auth_config, dict):
            basic_auth = auth_config.get('basic', {})
            if isinstance(basic_auth, dict):
                if 'user' in basic_auth:
                    self.basic_auth_user = basic_auth['user']
                if 'password' in basic_auth:
                    self.basic_auth_password = basic_auth['password']
        
    def load_from_env(self):
        """Load configuration from environment variables"""
        # SSL configuration
        ssl_enabled_env = os.environ.get(self.SSL_ENABLED, '').lower()
        self.ssl_enabled = ssl_enabled_env in ('true', '1', 'yes')
        self.ssl_cert_path = os.environ.get(self.SSL_CERT_PATH)
        self.ssl_key_path = os.environ.get(self.SSL_KEY_PATH)
        self.domain = os.environ.get(self.SSL_DOMAIN)
        self.ssl_verify_mode = os.environ.get(self.SSL_VERIFY_MODE, self.DEFAULTS[self.SSL_VERIFY_MODE])
        
        # Additional security settings
        self.allowed_hosts = self._parse_list(os.environ.get(self.ALLOWED_HOSTS, self.DEFAULTS[self.ALLOWED_HOSTS]))
        self.cors_origins = self._parse_list(os.environ.get(self.CORS_ORIGINS, self.DEFAULTS[self.CORS_ORIGINS]))
        self.max_request_size = int(os.environ.get(self.MAX_REQUEST_SIZE, self.DEFAULTS[self.MAX_REQUEST_SIZE]))
        self.rate_limit = int(os.environ.get(self.RATE_LIMIT, self.DEFAULTS[self.RATE_LIMIT]))
        self.request_timeout = int(os.environ.get(self.REQUEST_TIMEOUT, self.DEFAULTS[self.REQUEST_TIMEOUT]))
        
        # HSTS settings
        use_hsts_env = os.environ.get(self.USE_HSTS, '').lower()
        self.use_hsts = use_hsts_env != 'false' if use_hsts_env else self.DEFAULTS[self.USE_HSTS]
        self.hsts_max_age = int(os.environ.get(self.HSTS_MAX_AGE, self.DEFAULTS[self.HSTS_MAX_AGE]))
        
        # Authentication
        self.basic_auth_user = os.environ.get(self.BASIC_AUTH_USER)
        self.basic_auth_password = os.environ.get(self.BASIC_AUTH_PASSWORD)
        
    def _parse_list(self, value: Union[str, list]) -> list:
        """Parse comma-separated list from environment variable or list from config"""
        if isinstance(value, list):
            return value
        if value == '*':
            return ['*']
        return [item.strip() for item in value.split(',') if item.strip()]
    
    def validate_ssl_config(self) -> Tuple[bool, Optional[str]]:
        """
        Validate SSL configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.ssl_enabled:
            return True, None
            
        if not self.ssl_cert_path:
            return False, "SSL enabled but SWML_SSL_CERT_PATH not set"
            
        if not self.ssl_key_path:
            return False, "SSL enabled but SWML_SSL_KEY_PATH not set"
            
        if not os.path.exists(self.ssl_cert_path):
            return False, f"SSL certificate file not found: {self.ssl_cert_path}"
            
        if not os.path.exists(self.ssl_key_path):
            return False, f"SSL key file not found: {self.ssl_key_path}"
            
        return True, None
    
    def get_ssl_context_kwargs(self) -> Dict[str, Any]:
        """
        Get SSL context kwargs for uvicorn.
        
        Returns:
            Dictionary of SSL parameters for uvicorn
        """
        if not self.ssl_enabled:
            return {}
            
        is_valid, error = self.validate_ssl_config()
        if not is_valid:
            logger.error("ssl_validation_failed", error=error)
            return {}
            
        return {
            'ssl_certfile': self.ssl_cert_path,
            'ssl_keyfile': self.ssl_key_path,
            # Additional SSL options can be added here
        }
    
    def get_basic_auth(self) -> Tuple[str, str]:
        """
        Get basic auth credentials, generating if not set.
        
        Returns:
            Tuple of (username, password)
        """
        username = self.basic_auth_user or "signalwire"
        if not self.basic_auth_password:
            self.basic_auth_password = secrets.token_urlsafe(32)
        password = self.basic_auth_password

        return username, password
    
    def get_security_headers(self, is_https: bool = False) -> Dict[str, str]:
        """
        Get security headers to add to responses.
        
        Args:
            is_https: Whether the connection is over HTTPS
            
        Returns:
            Dictionary of security headers
        """
        headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
        }
        
        # Add HSTS header if HTTPS and enabled
        if is_https and self.use_hsts:
            headers['Strict-Transport-Security'] = f'max-age={self.hsts_max_age}; includeSubDomains'
            
        return headers
    
    def should_allow_host(self, host: str) -> bool:
        """
        Check if a host is allowed.
        
        Args:
            host: The host to check
            
        Returns:
            True if the host is allowed
        """
        if '*' in self.allowed_hosts:
            return True
            
        return host in self.allowed_hosts
    
    def get_cors_config(self) -> Dict[str, Any]:
        """
        Get CORS configuration for FastAPI.
        
        Returns:
            Dictionary of CORS settings
        """
        return {
            'allow_origins': self.cors_origins,
            'allow_credentials': True,
            'allow_methods': ['*'],
            'allow_headers': ['*'],
        }
    
    def get_url_scheme(self) -> str:
        """Get the URL scheme based on SSL configuration"""
        return 'https' if self.ssl_enabled else 'http'
    
    def log_config(self, service_name: str):
        """Log the current security configuration"""
        logger.info(
            "security_config_loaded",
            service=service_name,
            ssl_enabled=self.ssl_enabled,
            domain=self.domain,
            allowed_hosts=self.allowed_hosts,
            cors_origins=self.cors_origins,
            max_request_size=self.max_request_size,
            rate_limit=self.rate_limit,
            use_hsts=self.use_hsts,
            has_basic_auth=bool(self.basic_auth_user and self.basic_auth_password)
        )


# Global instance for easy access (backward compatibility)
# Services can create their own instances with specific config files
security_config = SecurityConfig()