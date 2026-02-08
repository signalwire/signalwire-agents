"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for StateMixin
"""

import pytest
from unittest.mock import Mock, MagicMock, PropertyMock, patch


from signalwire_agents.core.mixins.state_mixin import StateMixin


class MockStateHost(StateMixin):
    """
    A minimal host class that inherits from StateMixin and provides
    all the attributes the mixin expects to find on self.
    """

    def __init__(self, session_manager=None, tool_registry=None):
        self.log = Mock()
        if session_manager is not None:
            self._session_manager = session_manager
        if tool_registry is not None:
            self._tool_registry = tool_registry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_session_manager():
    """Return a fresh Mock standing in for SessionManager."""
    sm = Mock()
    sm.create_tool_token.return_value = "test-token-abc123"
    sm.validate_tool_token.return_value = True
    sm.debug_token.return_value = {
        "valid_format": True,
        "components": {
            "call_id": "call-123",
            "function": "my_tool",
            "expiry": "2099-01-01",
        },
        "status": {
            "is_expired": False,
            "expires_in_seconds": 3600,
        },
    }
    return sm


@pytest.fixture
def mock_tool_registry():
    """Return a mock tool registry with some registered functions."""
    registry = Mock()
    # Create a SWAIGFunction-like mock (non-dict with secure attribute)
    secure_func = Mock()
    secure_func.secure = True

    non_secure_func = Mock()
    non_secure_func.secure = False

    data_map_func = {"function": "data_map_tool", "data_map": {}}

    registry._swaig_functions = {
        "secure_tool": secure_func,
        "non_secure_tool": non_secure_func,
        "data_map_tool": data_map_func,
    }
    return registry


@pytest.fixture
def host(mock_session_manager, mock_tool_registry):
    """Return a MockStateHost wired with mock dependencies."""
    return MockStateHost(
        session_manager=mock_session_manager,
        tool_registry=mock_tool_registry,
    )


# ===========================================================================
# Tests for _create_tool_token
# ===========================================================================

class TestCreateToolToken:
    """Tests for StateMixin._create_tool_token"""

    def test_creates_token_successfully(self, host, mock_session_manager):
        token = host._create_tool_token("my_tool", "call-123")
        mock_session_manager.create_tool_token.assert_called_once_with("my_tool", "call-123")
        assert token == "test-token-abc123"

    def test_returns_empty_string_when_no_session_manager(self):
        """When _session_manager attribute is missing, return empty string."""
        h = MockStateHost()  # No session_manager passed
        result = h._create_tool_token("tool", "call-1")
        assert result == ""
        h.log.error.assert_called_once()

    def test_returns_empty_string_on_exception(self, host, mock_session_manager):
        """When session_manager.create_tool_token raises, return empty string."""
        mock_session_manager.create_tool_token.side_effect = RuntimeError("boom")
        result = host._create_tool_token("tool", "call-1")
        assert result == ""
        host.log.error.assert_called_once()

    def test_passes_correct_args_to_session_manager(self, host, mock_session_manager):
        host._create_tool_token("func_name", "call-xyz")
        mock_session_manager.create_tool_token.assert_called_once_with("func_name", "call-xyz")

    def test_returns_whatever_session_manager_returns(self, host, mock_session_manager):
        mock_session_manager.create_tool_token.return_value = "custom-token-value"
        result = host._create_tool_token("t", "c")
        assert result == "custom-token-value"


# ===========================================================================
# Tests for validate_tool_token - basic validation
# ===========================================================================

class TestValidateToolTokenBasic:
    """Tests for StateMixin.validate_tool_token basic paths"""

    def test_returns_false_for_unknown_function(self, host):
        result = host.validate_tool_token("unknown_func", "token", "call-123")
        assert result is False
        host.log.warning.assert_called()

    def test_returns_true_for_non_secure_function(self, host):
        """Non-secure functions should always be allowed."""
        result = host.validate_tool_token("non_secure_tool", "any-token", "call-123")
        assert result is True

    def test_validates_secure_function_with_valid_token(self, host, mock_session_manager):
        mock_session_manager.validate_tool_token.return_value = True
        result = host.validate_tool_token("secure_tool", "valid-token", "call-123")
        assert result is True

    def test_rejects_secure_function_with_invalid_token(self, host, mock_session_manager):
        mock_session_manager.validate_tool_token.return_value = False
        result = host.validate_tool_token("secure_tool", "bad-token", "call-123")
        assert result is False


# ===========================================================================
# Tests for validate_tool_token - data_map functions
# ===========================================================================

class TestValidateToolTokenDataMap:
    """Tests for data_map function handling in validate_tool_token"""

    def test_data_map_functions_are_always_secure(self, host, mock_session_manager):
        """Data map functions (raw dicts) are treated as secure by default."""
        mock_session_manager.validate_tool_token.return_value = True
        result = host.validate_tool_token("data_map_tool", "valid-token", "call-123")
        assert result is True
        mock_session_manager.validate_tool_token.assert_called()

    def test_data_map_missing_token_returns_false(self, host):
        """Data map functions with missing token should fail validation."""
        result = host.validate_tool_token("data_map_tool", "", "call-123")
        assert result is False

    def test_data_map_none_token_returns_false(self, host):
        """Data map functions with None token should fail validation."""
        result = host.validate_tool_token("data_map_tool", None, "call-123")
        assert result is False


# ===========================================================================
# Tests for validate_tool_token - missing session manager
# ===========================================================================

class TestValidateToolTokenNoSessionManager:
    """Tests for validate_tool_token when session_manager is absent"""

    def test_returns_false_when_no_session_manager(self, mock_tool_registry):
        h = MockStateHost(tool_registry=mock_tool_registry)
        result = h.validate_tool_token("secure_tool", "token", "call-1")
        assert result is False

    def test_non_secure_still_allowed_without_session_manager(self, mock_tool_registry):
        """Non-secure functions should still be allowed even without session manager."""
        h = MockStateHost(tool_registry=mock_tool_registry)
        result = h.validate_tool_token("non_secure_tool", "token", "call-1")
        assert result is True


# ===========================================================================
# Tests for validate_tool_token - token debugging
# ===========================================================================

class TestValidateToolTokenDebug:
    """Tests for the debug_token branch in validate_tool_token"""

    def test_debug_token_called_when_available(self, host, mock_session_manager):
        host.validate_tool_token("secure_tool", "some-token", "call-123")
        mock_session_manager.debug_token.assert_called()

    def test_function_mismatch_logged(self, host, mock_session_manager):
        """When the token's function name doesn't match, a warning is logged."""
        mock_session_manager.debug_token.return_value = {
            "valid_format": True,
            "components": {
                "call_id": "call-123",
                "function": "different_tool",
                "expiry": "2099-01-01",
            },
            "status": {"is_expired": False},
        }
        host.validate_tool_token("secure_tool", "some-token", "call-123")
        # Should log the function mismatch
        assert any(
            "token_function_mismatch" in str(call)
            for call in host.log.warning.call_args_list
        )

    def test_call_id_mismatch_logged(self, host, mock_session_manager):
        """When the token's call_id doesn't match, a warning is logged."""
        mock_session_manager.debug_token.return_value = {
            "valid_format": True,
            "components": {
                "call_id": "different-call",
                "function": "secure_tool",
                "expiry": "2099-01-01",
            },
            "status": {"is_expired": False},
        }
        host.validate_tool_token("secure_tool", "some-token", "call-123")
        assert any(
            "token_call_id_mismatch" in str(call)
            for call in host.log.warning.call_args_list
        )

    def test_expired_token_logged(self, host, mock_session_manager):
        """When the token is expired, a warning is logged."""
        mock_session_manager.debug_token.return_value = {
            "valid_format": True,
            "components": {
                "call_id": "call-123",
                "function": "secure_tool",
                "expiry": "2020-01-01",
            },
            "status": {
                "is_expired": True,
                "expires_in_seconds": -100,
            },
        }
        host.validate_tool_token("secure_tool", "some-token", "call-123")
        assert any(
            "token_expired" in str(call)
            for call in host.log.warning.call_args_list
        )

    def test_debug_token_exception_handled(self, host, mock_session_manager):
        """If debug_token raises, it should be caught and logged."""
        mock_session_manager.debug_token.side_effect = RuntimeError("debug failed")
        # Should not raise, should still return validation result
        mock_session_manager.validate_tool_token.return_value = True
        result = host.validate_tool_token("secure_tool", "some-token", "call-123")
        # The validation may still succeed or fail based on the validate call
        # The important thing is no exception propagates
        assert isinstance(result, bool)


# ===========================================================================
# Tests for validate_tool_token - call_id extraction from token
# ===========================================================================

class TestValidateToolTokenCallIdExtraction:
    """Tests for extracting call_id from token when provided call_id is empty"""

    def test_uses_call_id_from_token_when_empty(self, host, mock_session_manager):
        """When call_id is empty, tries to extract from the token."""
        mock_session_manager.debug_token.return_value = {
            "valid_format": True,
            "components": {
                "call_id": "extracted-call-id",
                "function": "secure_tool",
            },
            "status": {"is_expired": False},
        }
        mock_session_manager.validate_tool_token.return_value = True
        result = host.validate_tool_token("secure_tool", "some-token", "")
        assert result is True

    def test_extracted_call_id_validation_fails_falls_through(self, host, mock_session_manager):
        """When extracted call_id validation fails, falls through to normal validation."""
        def validate_side_effect(fn, token, cid):
            if cid == "extracted-call-id":
                return False
            return False

        mock_session_manager.debug_token.return_value = {
            "valid_format": True,
            "components": {
                "call_id": "extracted-call-id",
                "function": "secure_tool",
            },
            "status": {"is_expired": False},
        }
        mock_session_manager.validate_tool_token.side_effect = validate_side_effect
        result = host.validate_tool_token("secure_tool", "some-token", "")
        assert result is False


# ===========================================================================
# Tests for validate_tool_token - exception handling
# ===========================================================================

class TestValidateToolTokenExceptions:
    """Tests for exception handling in validate_tool_token"""

    def test_returns_false_on_unexpected_exception(self, host, mock_session_manager):
        """Any unexpected exception should result in False."""
        mock_session_manager.validate_tool_token.side_effect = RuntimeError("unexpected")
        # Also need debug_token to not cause issue
        mock_session_manager.debug_token.return_value = {
            "valid_format": False,
        }
        result = host.validate_tool_token("secure_tool", "some-token", "call-123")
        assert result is False
        host.log.error.assert_called()
