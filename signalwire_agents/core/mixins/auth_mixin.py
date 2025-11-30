"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import os
import json
import base64
from typing import Union, Tuple

from fastapi import Request


class AuthMixin:
    """
    Mixin class containing all authentication-related methods for AgentBase
    """
    
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
        # Flask headers can be accessed directly with .get() which is case-insensitive
        auth_header = request.headers.get('Authorization')
                
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

        # Check for authorization header - use .get() which works with both dict and Flask headers
        auth_header = req.headers.get('Authorization')

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