"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Session manager for handling call sessions and security tokens
"""

from typing import Dict, Any, Optional, Tuple
import secrets
import time
import hmac
import hashlib
import base64
from datetime import datetime, timedelta


class SessionManager:
    """
    Manages security tokens for function calls
    
    This implementation is completely stateless - it does not track call sessions
    or store any information in memory. All validation is done using cryptographic
    signatures with the tokens containing all necessary information.
    """
    def __init__(self, token_expiry_secs: int = 900, secret_key: Optional[str] = None):
        """
        Initialize the session manager

        Args:
            token_expiry_secs: Seconds until tokens expire (default: 15 minutes)
            secret_key: Secret key for signing tokens (generated if not provided)
        """
        self.token_expiry_secs = token_expiry_secs
        # Use provided secret key or generate a secure one
        self.secret_key = secret_key or secrets.token_hex(32)
        self._debug_mode = False
    
    def create_session(self, call_id: Optional[str] = None) -> str:
        """
        Create a new session ID if one isn't provided
        
        Args:
            call_id: Optional call ID, generated if not provided
            
        Returns:
            The call_id for the session
        """
        # Generate call_id if not provided
        if not call_id:
            call_id = secrets.token_urlsafe(16)
        
        return call_id
    
    def generate_token(self, function_name: str, call_id: str) -> str:
        """
        Generate a secure self-contained token for a function call
        
        Args:
            function_name: Name of the function to generate a token for
            call_id: Call session ID
            
        Returns:
            A secure token
        """
        # Create token parts
        expiry = int(time.time()) + self.token_expiry_secs
        nonce = secrets.token_hex(8)
        
        # Create the message to sign
        message = f"{call_id}:{function_name}:{expiry}:{nonce}"
        
        # Sign the message
        signature = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Combine all parts into the token
        token = f"{call_id}.{function_name}.{expiry}.{nonce}.{signature}"
        
        # Base64 encode for URL safety
        return base64.urlsafe_b64encode(token.encode()).decode()
    
    # Alias for generate_token to maintain backward compatibility
    def create_tool_token(self, function_name: str, call_id: str) -> str:
        """
        Alias for generate_token to maintain backward compatibility
        
        Args:
            function_name: Name of the function to generate a token for
            call_id: Call session ID
            
        Returns:
            A secure token
        """
        return self.generate_token(function_name, call_id)
    
    def validate_token(self, call_id: str, function_name: str, token: str) -> bool:
        """
        Validate a function call token
        
        Args:
            call_id: Call session ID
            function_name: Name of the function being called
            token: Token to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Decode the token
            decoded_token = base64.urlsafe_b64decode(token.encode()).decode()
            
            # Split the token parts
            parts = decoded_token.split('.')
            if len(parts) != 5:
                return False
                
            token_call_id, token_function, token_expiry, token_nonce, token_signature = parts
            
            # Reject validation if call_id is not provided
            if not call_id:
                import logging
                logging.getLogger("signalwire.session_manager").warning(
                    "call_id not provided in request, rejecting token validation"
                )
                return False
            
            # Verify the function matches
            if token_function != function_name:
                return False
                
            # Check if the token has expired
            expiry = int(token_expiry)
            if expiry < time.time():
                return False
                
            # Recreate the message and verify the signature
            message = f"{token_call_id}:{token_function}:{token_expiry}:{token_nonce}"
            expected_signature = hmac.new(
                self.secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()  # Full HMAC digest for maximum security[:32]

            if not hmac.compare_digest(token_signature, expected_signature):
                return False
                
            # Finally, verify the call_id matches unless we're in special case
            # This check is done last to ensure the token is otherwise valid
            if token_call_id != call_id:
                return False
                
            return True
        except Exception:
            # Any exception during validation means the token is invalid
            return False
    
    # Alias for validate_token to maintain backward compatibility
    def validate_tool_token(self, function_name: str, token: str, call_id: str) -> bool:
        """
        Alias for validate_token to maintain backward compatibility
        
        Args:
            function_name: Name of the function being called
            token: Token to validate
            call_id: Call session ID
            
        Returns:
            True if valid, False otherwise
        """
        # Reorder parameters to match validate_token signature (call_id first, then function_name)
        return self.validate_token(call_id=call_id, function_name=function_name, token=token)
    
    # Legacy methods that now don't track state but provide API compatibility
    
    def activate_session(self, call_id: str) -> bool:
        """
        Legacy method, does nothing but returns success
        """
        return True
    
    def end_session(self, call_id: str) -> bool:
        """
        Legacy method, does nothing but returns success
        """
        return True
    
    def get_session_metadata(self, call_id: str) -> Optional[Dict[str, Any]]:
        """
        Legacy method, always returns empty metadata
        """
        return {}
    
    def set_session_metadata(self, call_id: str, key: str, value: Any) -> bool:
        """
        Legacy method, does nothing but returns success
        """
        return True
    
    def debug_token(self, token: str) -> Dict[str, Any]:
        """
        Debug a token without validating it

        This method decodes the token and extracts its components for debugging purposes
        without performing validation. Requires _debug_mode to be True.

        Args:
            token: The token to debug

        Returns:
            Dictionary with token components and analysis
        """
        if not self._debug_mode:
            return {"error": "debug mode not enabled"}
        try:
            # Decode the token
            decoded_token = base64.urlsafe_b64decode(token.encode()).decode()
            
            # Split the token parts
            parts = decoded_token.split('.')
            if len(parts) != 5:
                return {
                    "valid_format": False,
                    "parts_count": len(parts),
                    "token_length": len(token) if token else 0
                }
                
            token_call_id, token_function, token_expiry, token_nonce, token_signature = parts
            
            # Check expiration
            current_time = int(time.time())
            try:
                expiry = int(token_expiry)
                is_expired = expiry < current_time
                expires_in = expiry - current_time if not is_expired else 0
                expiry_date = datetime.fromtimestamp(expiry).isoformat()
            except ValueError:
                expiry = None
                is_expired = None
                expires_in = None
                expiry_date = None
            
            return {
                "valid_format": True,
                "components": {
                    "call_id": token_call_id[:8] + "..." if len(token_call_id) > 8 else token_call_id,
                    "function": token_function,
                    "expiry": token_expiry,
                    "expiry_date": expiry_date,
                    "nonce": token_nonce,
                    "signature": token_signature[:8] + "..."
                },
                "status": {
                    "current_time": current_time,
                    "is_expired": is_expired,
                    "expires_in_seconds": expires_in
                }
            }
        except Exception as e:
            # Any exception during parsing
            return {
                "valid_format": False,
                "error": str(e),
                "token_length": len(token) if token else 0
            }
