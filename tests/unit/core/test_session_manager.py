"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for session manager module
"""

import pytest
import time
import hmac
import hashlib
import base64
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from signalwire_agents.core.security.session_manager import SessionManager


class TestSessionManager:
    """Test SessionManager functionality"""
    
    def test_basic_initialization(self):
        """Test basic SessionManager initialization"""
        manager = SessionManager()
        
        assert manager.secret_key is not None
        assert len(manager.secret_key) >= 32  # Should be secure length
        assert manager.token_expiry_secs == 900  # Default 15 minutes
    
    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters"""
        custom_secret = "my_custom_secret_key_that_is_long_enough"
        custom_expiry = 7200  # 2 hours
        
        manager = SessionManager(
            secret_key=custom_secret,
            token_expiry_secs=custom_expiry
        )
        
        assert manager.secret_key == custom_secret
        assert manager.token_expiry_secs == custom_expiry
    
    def test_create_session(self):
        """Test session creation"""
        manager = SessionManager()
        
        # Test with provided call_id
        call_id = manager.create_session("test_call_123")
        assert call_id == "test_call_123"
        
        # Test with auto-generated call_id
        auto_call_id = manager.create_session()
        assert auto_call_id is not None
        assert len(auto_call_id) > 0
        assert isinstance(auto_call_id, str)
    
    def test_generate_token_basic(self):
        """Test basic token generation"""
        manager = SessionManager()
        
        token = manager.generate_token("test_function", "call_123")
        
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Should be base64 encoded
        try:
            decoded = base64.urlsafe_b64decode(token.encode()).decode()
            assert "call_123" in decoded
            assert "test_function" in decoded
        except Exception:
            pytest.fail("Token should be valid base64")
    
    def test_create_tool_token_alias(self):
        """Test create_tool_token alias"""
        manager = SessionManager()
        
        token1 = manager.generate_token("test_func", "call_123")
        token2 = manager.create_tool_token("test_func", "call_123")
        
        # Should both be valid tokens (though different due to nonce)
        assert isinstance(token1, str)
        assert isinstance(token2, str)
        assert len(token1) > 0
        assert len(token2) > 0
    
    def test_validate_token_valid(self):
        """Test validating valid token"""
        manager = SessionManager()
        
        token = manager.generate_token("test_function", "call_123")
        
        # Should be valid immediately
        assert manager.validate_token("call_123", "test_function", token) is True
    
    def test_validate_token_wrong_function(self):
        """Test validating token with wrong function name"""
        manager = SessionManager()
        
        token = manager.generate_token("test_function", "call_123")
        
        # Should be invalid for different function
        assert manager.validate_token("call_123", "other_function", token) is False
    
    def test_validate_token_wrong_call_id(self):
        """Test validating token with wrong call ID"""
        manager = SessionManager()
        
        token = manager.generate_token("test_function", "call_123")
        
        # Should be invalid for different call_id
        assert manager.validate_token("other_call", "test_function", token) is False
    
    def test_validate_token_expired(self):
        """Test validating expired token"""
        manager = SessionManager(token_expiry_secs=1)  # 1 second expiry
        
        token = manager.generate_token("test_function", "call_123")
        
        # Wait for token to expire
        time.sleep(2)
        
        # Should be invalid due to expiry
        assert manager.validate_token("call_123", "test_function", token) is False
    
    def test_validate_token_invalid_signature(self):
        """Test validating token with invalid signature"""
        manager1 = SessionManager(secret_key="secret1" + "x" * 24)
        manager2 = SessionManager(secret_key="secret2" + "x" * 24)
        
        token = manager1.generate_token("test_function", "call_123")
        
        # Should be invalid with different secret
        assert manager2.validate_token("call_123", "test_function", token) is False
    
    def test_validate_token_malformed(self):
        """Test validating malformed token"""
        manager = SessionManager()
        
        # Test various malformed tokens
        malformed_tokens = [
            "not_base64",
            "invalid_token",
            "",
            base64.urlsafe_b64encode(b"too.few.parts").decode(),
            base64.urlsafe_b64encode(b"too.many.parts.here.extra.stuff").decode()
        ]
        
        for token in malformed_tokens:
            assert manager.validate_token("call_123", "test_function", token) is False
    
    def test_validate_token_empty_call_id(self):
        """Test validating token with empty call_id (special case)"""
        manager = SessionManager()
        
        token = manager.generate_token("test_function", "call_123")
        
        # Should reject with empty call_id (no longer falls back to token's call_id)
        assert manager.validate_token("", "test_function", token) is False
        assert manager.validate_token(None, "test_function", token) is False
    
    def test_validate_tool_token_alias(self):
        """Test validate_tool_token alias"""
        manager = SessionManager()
        
        token = manager.generate_token("test_function", "call_123")
        
        # Test alias method (note parameter order difference)
        assert manager.validate_tool_token("test_function", token, "call_123") is True
        assert manager.validate_tool_token("wrong_function", token, "call_123") is False
    
    def test_debug_token(self):
        """Test token debugging functionality"""
        manager = SessionManager()
        manager._debug_mode = True

        token = manager.generate_token("test_function", "call_123")
        debug_info = manager.debug_token(token)
        
        assert isinstance(debug_info, dict)
        assert "components" in debug_info
        assert "status" in debug_info
        assert "valid_format" in debug_info
        
        # Check components
        components = debug_info["components"]
        assert components["call_id"] == "call_123"
        assert components["function"] == "test_function"
        assert isinstance(components["expiry"], str)
        assert isinstance(components["nonce"], str)
        assert isinstance(components["signature"], str)
        
        # Check status
        status = debug_info["status"]
        assert isinstance(status["current_time"], int)
        assert isinstance(status["expires_in_seconds"], int)
        assert isinstance(status["is_expired"], bool)
    
    def test_debug_token_invalid(self):
        """Test debugging invalid token"""
        manager = SessionManager()
        manager._debug_mode = True

        debug_info = manager.debug_token("invalid_token")
        
        assert debug_info is not None
        assert "error" in debug_info
        assert "valid_format" in debug_info
        assert debug_info["valid_format"] is False
    
    def test_legacy_methods(self):
        """Test legacy API compatibility methods"""
        manager = SessionManager()
        
        # These should all work but not do anything meaningful
        assert manager.activate_session("call_123") is True
        assert manager.end_session("call_123") is True
        
        metadata = manager.get_session_metadata("call_123")
        assert isinstance(metadata, dict)
        assert len(metadata) == 0
        
        assert manager.set_session_metadata("call_123", "key", "value") is True


class TestSessionManagerErrorHandling:
    """Test error handling in SessionManager"""
    
    def test_token_generation_edge_cases(self):
        """Test token generation with edge cases"""
        manager = SessionManager()
        
        # Empty function name
        token = manager.generate_token("", "call_123")
        assert isinstance(token, str)
        assert manager.validate_token("call_123", "", token) is True
        
        # Empty call_id - validation should now reject empty call_id
        token = manager.generate_token("test_func", "")
        assert isinstance(token, str)
        assert manager.validate_token("", "test_func", token) is False
    
    def test_validation_with_corrupted_token(self):
        """Test validation with corrupted token data"""
        manager = SessionManager()
        
        # Create valid token then corrupt it
        token = manager.generate_token("test_function", "call_123")
        
        # Corrupt the base64 data
        corrupted = token[:-5] + "XXXXX"
        assert manager.validate_token("call_123", "test_function", corrupted) is False
    
    def test_time_manipulation_resistance(self):
        """Test resistance to time manipulation attacks"""
        manager = SessionManager(token_expiry_secs=3600)
        
        # Generate token
        token = manager.generate_token("test_function", "call_123")
        
        # Mock time to be in the future (simulating clock skew)
        with patch('time.time', return_value=time.time() + 7200):  # 2 hours ahead
            # Token should be expired
            assert manager.validate_token("call_123", "test_function", token) is False


class TestSessionManagerIntegration:
    """Test integration scenarios"""
    
    def test_complete_token_workflow(self):
        """Test complete token management workflow"""
        manager = SessionManager()
        manager._debug_mode = True
        
        # 1. Create session
        call_id = manager.create_session()
        assert call_id is not None
        
        # 2. Generate token for function
        token = manager.generate_token("get_balance", call_id)
        assert token is not None
        
        # 3. Validate token
        assert manager.validate_token(call_id, "get_balance", token) is True
        
        # 4. Debug token
        debug_info = manager.debug_token(token)
        # call_id may be truncated in debug output ([:8] + "..." for long IDs)
        assert call_id.startswith(debug_info["components"]["call_id"].replace("...", ""))
        assert debug_info["components"]["function"] == "get_balance"
        
        # 5. Legacy session management
        assert manager.activate_session(call_id) is True
        assert manager.end_session(call_id) is True
    
    def test_multiple_function_tokens(self):
        """Test managing tokens for multiple functions"""
        manager = SessionManager()
        call_id = "multi_func_call"
        
        functions = ["get_balance", "transfer_funds", "get_history", "update_profile"]
        tokens = {}
        
        # Generate tokens for all functions
        for func in functions:
            tokens[func] = manager.generate_token(func, call_id)
        
        # Validate all tokens
        for func, token in tokens.items():
            assert manager.validate_token(call_id, func, token) is True
            
            # Should be invalid for other functions
            for other_func in functions:
                if other_func != func:
                    assert manager.validate_token(call_id, other_func, token) is False
    
    def test_concurrent_sessions(self):
        """Test managing multiple concurrent sessions"""
        manager = SessionManager()
        
        sessions = {}
        for i in range(5):
            call_id = f"call_{i}"
            sessions[call_id] = {
                "token": manager.generate_token("test_function", call_id)
            }
        
        # Validate all sessions
        for call_id, session_data in sessions.items():
            assert manager.validate_token(call_id, "test_function", session_data["token"]) is True
            
            # Should be invalid for other call_ids
            for other_call_id in sessions.keys():
                if other_call_id != call_id:
                    assert manager.validate_token(other_call_id, "test_function", session_data["token"]) is False
    
    def test_token_expiry_workflow(self):
        """Test token expiry workflow"""
        manager = SessionManager(token_expiry_secs=2)  # 2 second expiry
        
        call_id = "expiry_test"
        token = manager.generate_token("test_function", call_id)
        
        # Should be valid initially
        assert manager.validate_token(call_id, "test_function", token) is True
        
        # Wait for partial expiry
        time.sleep(1)
        assert manager.validate_token(call_id, "test_function", token) is True
        
        # Wait for full expiry
        time.sleep(2)
        assert manager.validate_token(call_id, "test_function", token) is False
        
        # Generate new token
        new_token = manager.generate_token("test_function", call_id)
        assert manager.validate_token(call_id, "test_function", new_token) is True
    
    def test_security_isolation(self):
        """Test security isolation between managers"""
        manager1 = SessionManager(secret_key="secret1" + "x" * 24)
        manager2 = SessionManager(secret_key="secret2" + "x" * 24)
        
        call_id = "security_test"
        function_name = "test_function"
        
        # Generate token with manager1
        token1 = manager1.generate_token(function_name, call_id)
        
        # Should be valid with manager1
        assert manager1.validate_token(call_id, function_name, token1) is True
        
        # Should be invalid with manager2
        assert manager2.validate_token(call_id, function_name, token1) is False
        
        # Generate token with manager2
        token2 = manager2.generate_token(function_name, call_id)
        
        # Should be valid with manager2
        assert manager2.validate_token(call_id, function_name, token2) is True
        
        # Should be invalid with manager1
        assert manager1.validate_token(call_id, function_name, token2) is False
    
    def test_performance_with_many_tokens(self):
        """Test performance with many token operations"""
        manager = SessionManager()
        
        # Generate many tokens
        tokens = []
        for i in range(100):
            call_id = f"call_{i}"
            function_name = f"function_{i % 10}"  # 10 different functions
            token = manager.generate_token(function_name, call_id)
            tokens.append((call_id, function_name, token))
        
        # Validate all tokens
        for call_id, function_name, token in tokens:
            assert manager.validate_token(call_id, function_name, token) is True
        
        # Test cross-validation (should all fail)
        for i, (call_id, function_name, token) in enumerate(tokens[:10]):
            for j, (other_call_id, other_function_name, _) in enumerate(tokens[10:20]):
                if i != j:
                    assert manager.validate_token(other_call_id, other_function_name, token) is False
    
    def test_token_structure_consistency(self):
        """Test token structure consistency"""
        manager = SessionManager()
        manager._debug_mode = True
        
        # Generate multiple tokens
        tokens = []
        for i in range(10):
            token = manager.generate_token(f"func_{i}", f"call_{i}")
            tokens.append(token)
        
        # All tokens should be valid base64
        for token in tokens:
            try:
                decoded = base64.urlsafe_b64decode(token.encode()).decode()
                parts = decoded.split('.')
                assert len(parts) == 5  # call_id.function.expiry.nonce.signature
            except Exception:
                pytest.fail(f"Token {token} should have valid structure")
        
        # Debug info should be consistent
        for i, token in enumerate(tokens):
            debug_info = manager.debug_token(token)
            assert debug_info is not None
            assert debug_info["components"]["call_id"] == f"call_{i}"
            assert debug_info["components"]["function"] == f"func_{i}"
            assert isinstance(debug_info["components"]["expiry"], str)
            assert isinstance(debug_info["components"]["nonce"], str)
            assert isinstance(debug_info["components"]["signature"], str) 