"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for MCP Gateway Service (gateway_service.py)

Flask and flask_limiter are optional dependencies (mcp-gateway extra).
Tests are skipped automatically when they are not installed.
"""

import json
import os
import sys
import base64
import logging
import threading
import re

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime

# Skip the entire module when Flask is not installed
flask = pytest.importorskip("flask", reason="flask is required for MCP Gateway tests")
pytest.importorskip("flask_limiter", reason="flask_limiter is required for MCP Gateway tests")


# ---------------------------------------------------------------------------
# Helpers: build a minimal config dict and the standard set of constructor
# patches so that MCPGateway can be instantiated without touching the
# filesystem, network, or real MCP processes.
# ---------------------------------------------------------------------------

def _minimal_config():
    """Return a minimal valid configuration dictionary."""
    return {
        "server": {
            "host": "0.0.0.0",
            "port": 8080,
            "auth_user": "admin",
            "auth_password": "secret",
            "auth_token": "test-bearer-token",
        },
        "services": {},
        "session": {
            "default_timeout": 300,
            "max_sessions_per_service": 100,
            "cleanup_interval": 60,
        },
        "rate_limiting": {
            "default_limits": ["200 per day", "50 per hour"],
            "tools_limit": "30 per minute",
            "call_limit": "10 per minute",
            "session_delete_limit": "20 per minute",
            "storage_uri": "memory://",
        },
        "logging": {
            "level": "WARNING",
        },
    }


def _create_gateway(config=None):
    """
    Instantiate an ``MCPGateway`` with every external dependency mocked.

    Returns (gateway, mocks_dict) where *mocks_dict* maps friendly names to
    the active ``MagicMock`` instances so tests can configure or assert on them.
    """
    from signalwire_agents.mcp_gateway.gateway_service import MCPGateway

    if config is None:
        config = _minimal_config()

    mock_config_loader = MagicMock()
    mock_config_loader.has_config.return_value = True
    mock_config_loader.get_config.return_value = config
    mock_config_loader.substitute_vars.return_value = config

    patches = {
        "config_loader_cls": patch(
            "signalwire_agents.mcp_gateway.gateway_service.ConfigLoader",
            return_value=mock_config_loader,
        ),
        "security_config_cls": patch(
            "signalwire_agents.mcp_gateway.gateway_service.SecurityConfig",
        ),
        "mcp_manager_cls": patch(
            "signalwire_agents.mcp_gateway.gateway_service.MCPManager",
        ),
        "session_manager_cls": patch(
            "signalwire_agents.mcp_gateway.gateway_service.SessionManager",
        ),
    }

    mocks = {}
    managers = {}

    for name, p in patches.items():
        managers[name] = p.start()

    # MCPManager().validate_services() is called inside __init__
    managers["mcp_manager_cls"].return_value.validate_services.return_value = {}

    # Set JSON-serializable defaults so routes don't blow up with MagicMock
    managers["mcp_manager_cls"].return_value.list_services.return_value = {}
    managers["mcp_manager_cls"].return_value.get_service_tools.return_value = []

    # SessionManager needs a default_timeout attribute used in call_service_tool
    managers["session_manager_cls"].return_value.default_timeout = 300
    managers["session_manager_cls"].return_value.list_sessions.return_value = {}
    managers["session_manager_cls"].return_value.get_session.return_value = None
    managers["session_manager_cls"].return_value.close_session.return_value = False

    try:
        gateway = MCPGateway("fake_config.json")
    finally:
        for p in patches.values():
            p.stop()

    mocks["config_loader"] = managers["config_loader_cls"].return_value
    mocks["security_config"] = managers["security_config_cls"].return_value
    mocks["mcp_manager"] = managers["mcp_manager_cls"].return_value
    mocks["session_manager"] = managers["session_manager_cls"].return_value

    return gateway, mocks


def _auth_headers_basic(user="admin", password="secret"):
    """Return HTTP headers for Basic authentication."""
    creds = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {creds}"}


def _auth_headers_bearer(token="test-bearer-token"):
    """Return HTTP headers for Bearer token authentication."""
    return {"Authorization": f"Bearer {token}"}


# ===================================================================
# Tests: Initialization
# ===================================================================

class TestMCPGatewayInit:
    """Tests for MCPGateway construction and configuration."""

    def test_init_loads_config_via_config_loader(self):
        """When ConfigLoader has_config() returns True, config is loaded through it."""
        gateway, mocks = _create_gateway()
        assert mocks["config_loader"].has_config.called
        assert mocks["config_loader"].get_config.called
        assert mocks["config_loader"].substitute_vars.called

    def test_init_falls_back_to_load_config_when_no_config_loader(self):
        """When ConfigLoader has_config() is False, _load_config is used."""
        from signalwire_agents.mcp_gateway.gateway_service import MCPGateway

        config = _minimal_config()

        mock_config_loader = MagicMock()
        mock_config_loader.has_config.return_value = False

        with patch("signalwire_agents.mcp_gateway.gateway_service.ConfigLoader", return_value=mock_config_loader), \
             patch("signalwire_agents.mcp_gateway.gateway_service.SecurityConfig"), \
             patch("signalwire_agents.mcp_gateway.gateway_service.MCPManager") as mock_mcp_cls, \
             patch("signalwire_agents.mcp_gateway.gateway_service.SessionManager") as mock_session_cls, \
             patch.object(MCPGateway, "_load_config", return_value=config) as mock_load:

            mock_mcp_cls.return_value.validate_services.return_value = {}
            mock_session_cls.return_value.default_timeout = 300

            MCPGateway("missing.json")
            mock_load.assert_called_once_with("missing.json")

    def test_init_creates_flask_app(self):
        """A Flask app instance is created during init."""
        gateway, _ = _create_gateway()
        assert gateway.app is not None
        assert gateway.app.test_client() is not None

    def test_init_sets_max_content_length(self):
        """Request body is capped at 10 MB."""
        gateway, _ = _create_gateway()
        assert gateway.app.config["MAX_CONTENT_LENGTH"] == 10 * 1024 * 1024

    def test_init_rate_limiter_configured(self):
        """Rate limiter is wired to the Flask app."""
        gateway, _ = _create_gateway()
        assert gateway.limiter is not None

    def test_init_validates_services_on_startup(self):
        """validate_services() is called during __init__."""
        gateway, mocks = _create_gateway()
        mocks["mcp_manager"].validate_services.assert_called_once()

    def test_init_logs_warning_for_failed_validation(self):
        """A warning is logged when a service fails validation."""
        from signalwire_agents.mcp_gateway.gateway_service import MCPGateway

        config = _minimal_config()
        mock_config_loader = MagicMock()
        mock_config_loader.has_config.return_value = True
        mock_config_loader.get_config.return_value = config
        mock_config_loader.substitute_vars.return_value = config

        with patch("signalwire_agents.mcp_gateway.gateway_service.ConfigLoader", return_value=mock_config_loader), \
             patch("signalwire_agents.mcp_gateway.gateway_service.SecurityConfig"), \
             patch("signalwire_agents.mcp_gateway.gateway_service.MCPManager") as mcp_cls, \
             patch("signalwire_agents.mcp_gateway.gateway_service.SessionManager") as sm_cls, \
             patch("signalwire_agents.mcp_gateway.gateway_service.logger") as mock_logger:

            mcp_cls.return_value.validate_services.return_value = {"bad_svc": False, "good_svc": True}
            sm_cls.return_value.default_timeout = 300

            MCPGateway("fake.json")

            # At least one warning about 'bad_svc' failing validation
            warning_calls = [c for c in mock_logger.warning.call_args_list
                             if "bad_svc" in str(c)]
            assert len(warning_calls) >= 1

    def test_init_shutdown_flags_default(self):
        """Shutdown flags start as False / None."""
        gateway, _ = _create_gateway()
        assert gateway._shutdown_requested is False
        assert gateway._shutdown_cleanup_done is False
        assert gateway.server is None

    def test_init_rate_config_uses_defaults_when_missing(self):
        """When rate_limiting section is absent, sensible defaults are used."""
        config = _minimal_config()
        del config["rate_limiting"]
        gateway, _ = _create_gateway(config)
        assert gateway.rate_config == {}

    def test_init_security_config_created(self):
        """SecurityConfig is instantiated with correct parameters."""
        from signalwire_agents.mcp_gateway.gateway_service import MCPGateway

        config = _minimal_config()
        mock_cl = MagicMock()
        mock_cl.has_config.return_value = True
        mock_cl.get_config.return_value = config
        mock_cl.substitute_vars.return_value = config

        with patch("signalwire_agents.mcp_gateway.gateway_service.ConfigLoader", return_value=mock_cl), \
             patch("signalwire_agents.mcp_gateway.gateway_service.SecurityConfig") as sec_cls, \
             patch("signalwire_agents.mcp_gateway.gateway_service.MCPManager") as mcp_cls, \
             patch("signalwire_agents.mcp_gateway.gateway_service.SessionManager") as sm_cls:

            mcp_cls.return_value.validate_services.return_value = {}
            sm_cls.return_value.default_timeout = 300

            MCPGateway("myconfig.json")
            sec_cls.assert_called_once_with(config_file="myconfig.json", service_name="mcp")


# ===================================================================
# Tests: Input Validation Helpers
# ===================================================================

class TestValidationHelpers:
    """Tests for _validate_service_name, _validate_session_id, _validate_tool_name."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.gateway, self.mocks = _create_gateway()

    # -- service name -------------------------------------------------------

    def test_validate_service_name_valid(self):
        assert self.gateway._validate_service_name("my-service_1") == "my-service_1"

    def test_validate_service_name_empty(self):
        with pytest.raises(ValueError, match="Invalid service name length"):
            self.gateway._validate_service_name("")

    def test_validate_service_name_none(self):
        with pytest.raises(ValueError, match="Invalid service name length"):
            self.gateway._validate_service_name(None)

    def test_validate_service_name_too_long(self):
        with pytest.raises(ValueError, match="Invalid service name length"):
            self.gateway._validate_service_name("a" * 65)

    def test_validate_service_name_max_length(self):
        """Exactly 64 chars should be accepted."""
        name = "a" * 64
        assert self.gateway._validate_service_name(name) == name

    def test_validate_service_name_invalid_chars(self):
        with pytest.raises(ValueError, match="invalid characters"):
            self.gateway._validate_service_name("my service!")

    def test_validate_service_name_injection_attempt(self):
        with pytest.raises(ValueError, match="invalid characters"):
            self.gateway._validate_service_name("service; rm -rf /")

    def test_validate_service_name_path_traversal(self):
        with pytest.raises(ValueError, match="invalid characters"):
            self.gateway._validate_service_name("../etc/passwd")

    # -- session id ---------------------------------------------------------

    def test_validate_session_id_valid(self):
        assert self.gateway._validate_session_id("sess-123.abc_def") == "sess-123.abc_def"

    def test_validate_session_id_empty(self):
        with pytest.raises(ValueError, match="Invalid session ID length"):
            self.gateway._validate_session_id("")

    def test_validate_session_id_none(self):
        with pytest.raises(ValueError, match="Invalid session ID length"):
            self.gateway._validate_session_id(None)

    def test_validate_session_id_too_long(self):
        with pytest.raises(ValueError, match="Invalid session ID length"):
            self.gateway._validate_session_id("x" * 129)

    def test_validate_session_id_max_length(self):
        sid = "a" * 128
        assert self.gateway._validate_session_id(sid) == sid

    def test_validate_session_id_invalid_chars(self):
        with pytest.raises(ValueError, match="invalid characters"):
            self.gateway._validate_session_id("sess id with spaces")

    # -- tool name ----------------------------------------------------------

    def test_validate_tool_name_valid(self):
        assert self.gateway._validate_tool_name("add-todo_item") == "add-todo_item"

    def test_validate_tool_name_empty(self):
        with pytest.raises(ValueError, match="Invalid tool name length"):
            self.gateway._validate_tool_name("")

    def test_validate_tool_name_none(self):
        with pytest.raises(ValueError, match="Invalid tool name length"):
            self.gateway._validate_tool_name(None)

    def test_validate_tool_name_too_long(self):
        with pytest.raises(ValueError, match="Invalid tool name length"):
            self.gateway._validate_tool_name("t" * 65)

    def test_validate_tool_name_invalid_chars(self):
        with pytest.raises(ValueError, match="invalid characters"):
            self.gateway._validate_tool_name("tool name!")


# ===================================================================
# Tests: Security Event Logging
# ===================================================================

class TestLogSecurityEvent:
    """Tests for _log_security_event."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.gateway, self.mocks = _create_gateway()

    def test_log_security_event_basic(self):
        with patch("signalwire_agents.mcp_gateway.gateway_service.logger") as mock_logger:
            self.gateway._log_security_event("auth_failed", {"ip": "1.2.3.4"})
            mock_logger.info.assert_called_once()
            logged = mock_logger.info.call_args[0][0]
            assert "SECURITY_EVENT" in logged
            assert "auth_failed" in logged
            assert "1.2.3.4" in logged

    def test_log_security_event_truncates_long_strings(self):
        with patch("signalwire_agents.mcp_gateway.gateway_service.logger") as mock_logger:
            long_val = "x" * 500
            self.gateway._log_security_event("test", {"data": long_val})
            logged = mock_logger.info.call_args[0][0]
            parsed = json.loads(logged.split("SECURITY_EVENT: ")[1])
            assert len(parsed["data"]) <= 256

    def test_log_security_event_strips_control_chars(self):
        with patch("signalwire_agents.mcp_gateway.gateway_service.logger") as mock_logger:
            self.gateway._log_security_event("test", {"data": "hello\x00world\x1b"})
            logged = mock_logger.info.call_args[0][0]
            parsed = json.loads(logged.split("SECURITY_EVENT: ")[1])
            assert "\x00" not in parsed["data"]
            assert "\x1b" not in parsed["data"]
            assert "helloworld" in parsed["data"]

    def test_log_security_event_includes_timestamp(self):
        with patch("signalwire_agents.mcp_gateway.gateway_service.logger") as mock_logger:
            self.gateway._log_security_event("test", {})
            logged = mock_logger.info.call_args[0][0]
            parsed = json.loads(logged.split("SECURITY_EVENT: ")[1])
            assert "timestamp" in parsed

    def test_log_security_event_preserves_non_string_values(self):
        with patch("signalwire_agents.mcp_gateway.gateway_service.logger") as mock_logger:
            self.gateway._log_security_event("test", {"count": 42, "flag": True})
            logged = mock_logger.info.call_args[0][0]
            parsed = json.loads(logged.split("SECURITY_EVENT: ")[1])
            assert parsed["count"] == 42
            assert parsed["flag"] is True


# ===================================================================
# Tests: Environment Variable Substitution
# ===================================================================

class TestSubstituteEnvVars:
    """Tests for _substitute_env_vars."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.gateway, self.mocks = _create_gateway()

    def test_substitute_plain_string(self):
        assert self.gateway._substitute_env_vars("hello") == "hello"

    def test_substitute_env_var_present(self):
        with patch.dict(os.environ, {"MY_VAR": "my_value"}):
            assert self.gateway._substitute_env_vars("${MY_VAR}") == "my_value"

    def test_substitute_env_var_missing_no_default(self):
        """Missing var with no default returns the original placeholder."""
        env = os.environ.copy()
        env.pop("MISSING_VAR_XYZ", None)
        with patch.dict(os.environ, env, clear=True):
            result = self.gateway._substitute_env_vars("${MISSING_VAR_XYZ}")
            assert result == "${MISSING_VAR_XYZ}"

    def test_substitute_env_var_missing_with_default(self):
        env = os.environ.copy()
        env.pop("MISSING_VAR_XYZ", None)
        with patch.dict(os.environ, env, clear=True):
            assert self.gateway._substitute_env_vars("${MISSING_VAR_XYZ|fallback}") == "fallback"

    def test_substitute_env_var_present_with_default_ignored(self):
        with patch.dict(os.environ, {"MY_VAR": "real"}):
            assert self.gateway._substitute_env_vars("${MY_VAR|fallback}") == "real"

    def test_substitute_dict_recursion(self):
        with patch.dict(os.environ, {"A": "val_a"}):
            result = self.gateway._substitute_env_vars({"key": "${A}"})
            assert result == {"key": "val_a"}

    def test_substitute_list_recursion(self):
        with patch.dict(os.environ, {"B": "val_b"}):
            result = self.gateway._substitute_env_vars(["${B}", "plain"])
            assert result == ["val_b", "plain"]

    def test_substitute_nested_structures(self):
        with patch.dict(os.environ, {"X": "xval"}):
            data = {"outer": [{"inner": "${X}"}]}
            result = self.gateway._substitute_env_vars(data)
            assert result == {"outer": [{"inner": "xval"}]}

    def test_substitute_non_string_passthrough(self):
        assert self.gateway._substitute_env_vars(42) == 42
        assert self.gateway._substitute_env_vars(True) is True
        assert self.gateway._substitute_env_vars(None) is None


# ===================================================================
# Tests: _load_config
# ===================================================================

class TestLoadConfig:
    """Tests for the fallback _load_config method."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.gateway, self.mocks = _create_gateway()

    def test_load_config_reads_existing_file(self, tmp_path):
        config_data = _minimal_config()
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        loaded = self.gateway._load_config(str(config_file))
        assert loaded["server"]["port"] == 8080

    def test_load_config_creates_default_when_nothing_exists(self, tmp_path):
        config_path = str(tmp_path / "nonexistent.json")
        # Neither config_path nor sample_config.json exist
        with patch("os.path.exists", return_value=False):
            mock_open = MagicMock()
            with patch("builtins.open", mock_open):
                loaded = self.gateway._load_config(config_path)

        assert "server" in loaded
        assert loaded["server"]["port"] == 8080

    def test_load_config_copies_sample_when_available(self, tmp_path):
        config_path = str(tmp_path / "config.json")

        call_count = [0]
        def exists_side_effect(path):
            if path == config_path:
                # First call: config doesn't exist; after copy it does
                call_count[0] += 1
                return call_count[0] > 1
            if path == "sample_config.json":
                return True
            return False

        with patch("os.path.exists", side_effect=exists_side_effect), \
             patch("shutil.copy") as mock_copy:
            mock_file_content = json.dumps(_minimal_config())
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=MagicMock(
                read=MagicMock(return_value=mock_file_content)
            ))
            mock_file.__exit__ = MagicMock(return_value=False)

            with patch("builtins.open", return_value=mock_file):
                self.gateway._load_config(config_path)

            mock_copy.assert_called_once_with("sample_config.json", config_path)

    def test_load_config_converts_string_port_to_int(self, tmp_path):
        config_data = _minimal_config()
        config_data["server"]["port"] = "9090"
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        loaded = self.gateway._load_config(str(config_file))
        assert loaded["server"]["port"] == 9090

    def test_load_config_handles_invalid_port_string(self, tmp_path):
        config_data = _minimal_config()
        config_data["server"]["port"] = "not_a_number"
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        loaded = self.gateway._load_config(str(config_file))
        assert loaded["server"]["port"] == 8080  # falls back to default

    def test_load_config_converts_session_string_values(self, tmp_path):
        config_data = _minimal_config()
        config_data["session"]["default_timeout"] = "600"
        config_data["session"]["max_sessions_per_service"] = "50"
        config_data["session"]["cleanup_interval"] = "120"
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        loaded = self.gateway._load_config(str(config_file))
        assert loaded["session"]["default_timeout"] == 600
        assert loaded["session"]["max_sessions_per_service"] == 50
        assert loaded["session"]["cleanup_interval"] == 120

    def test_load_config_handles_invalid_session_values(self, tmp_path):
        config_data = _minimal_config()
        config_data["session"]["default_timeout"] = "nope"
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        loaded = self.gateway._load_config(str(config_file))
        assert loaded["session"]["default_timeout"] == 300  # default


# ===================================================================
# Tests: Authentication
# ===================================================================

class TestAuthentication:
    """Tests for _check_auth decorator and auth routes."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.gateway, self.mocks = _create_gateway()
        self.client = self.gateway.app.test_client()

    def test_bearer_token_auth_success(self):
        resp = self.client.get(
            "/services",
            headers=_auth_headers_bearer("test-bearer-token"),
        )
        assert resp.status_code == 200

    def test_bearer_token_auth_wrong_token(self):
        resp = self.client.get(
            "/services",
            headers=_auth_headers_bearer("wrong-token"),
        )
        assert resp.status_code == 401

    def test_basic_auth_success(self):
        resp = self.client.get(
            "/services",
            headers=_auth_headers_basic("admin", "secret"),
        )
        assert resp.status_code == 200

    def test_basic_auth_wrong_password(self):
        resp = self.client.get(
            "/services",
            headers=_auth_headers_basic("admin", "wrong"),
        )
        assert resp.status_code == 401

    def test_basic_auth_wrong_user(self):
        resp = self.client.get(
            "/services",
            headers=_auth_headers_basic("nobody", "secret"),
        )
        assert resp.status_code == 401

    def test_no_auth_header(self):
        resp = self.client.get("/services")
        assert resp.status_code == 401
        assert "WWW-Authenticate" in resp.headers

    def test_auth_failure_logs_security_event(self):
        with patch.object(self.gateway, "_log_security_event") as mock_log:
            self.client.get("/services")
            mock_log.assert_called()
            event_type = mock_log.call_args[0][0]
            assert event_type == "auth_failed"

    def test_bearer_auth_preferred_over_basic(self):
        """When a valid Bearer token is sent, Basic auth is not checked."""
        headers = {"Authorization": "Bearer test-bearer-token"}
        resp = self.client.get("/services", headers=headers)
        assert resp.status_code == 200

    def test_auth_without_token_config_falls_through_to_basic(self):
        """If no auth_token is configured, Bearer always fails, Basic is tried."""
        config = _minimal_config()
        del config["server"]["auth_token"]
        gateway, _ = _create_gateway(config)
        client = gateway.app.test_client()

        # Bearer with any token should fail (no expected token)
        resp = client.get("/services", headers=_auth_headers_bearer("anything"))
        assert resp.status_code == 401

        # Basic should still work
        resp = client.get("/services", headers=_auth_headers_basic("admin", "secret"))
        assert resp.status_code == 200


# ===================================================================
# Tests: Health Endpoint
# ===================================================================

class TestHealthEndpoint:
    """Tests for GET /health (no auth required)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.gateway, self.mocks = _create_gateway()
        self.client = self.gateway.app.test_client()

    def test_health_returns_200(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_json(self):
        resp = self.client.get("/health")
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"

    def test_health_no_auth_required(self):
        """Health endpoint must be accessible without credentials."""
        resp = self.client.get("/health")
        assert resp.status_code == 200


# ===================================================================
# Tests: Security Headers
# ===================================================================

class TestSecurityHeaders:
    """Verify that security headers are set on every response."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.gateway, self.mocks = _create_gateway()
        self.client = self.gateway.app.test_client()

    def test_x_content_type_options(self):
        resp = self.client.get("/health")
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options(self):
        resp = self.client.get("/health")
        assert resp.headers.get("X-Frame-Options") == "DENY"

    def test_x_xss_protection(self):
        resp = self.client.get("/health")
        assert resp.headers.get("X-XSS-Protection") == "1; mode=block"

    def test_content_security_policy(self):
        resp = self.client.get("/health")
        assert "default-src 'none'" in resp.headers.get("Content-Security-Policy", "")


# ===================================================================
# Tests: List Services
# ===================================================================

class TestListServicesEndpoint:
    """Tests for GET /services."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.gateway, self.mocks = _create_gateway()
        self.client = self.gateway.app.test_client()

    def test_list_services_success(self):
        self.mocks["mcp_manager"].list_services.return_value = {
            "todo": {"description": "Todo service", "enabled": True}
        }
        resp = self.client.get("/services", headers=_auth_headers_basic())
        assert resp.status_code == 200
        data = resp.get_json()
        assert "todo" in data

    def test_list_services_empty(self):
        self.mocks["mcp_manager"].list_services.return_value = {}
        resp = self.client.get("/services", headers=_auth_headers_basic())
        assert resp.status_code == 200
        assert resp.get_json() == {}


# ===================================================================
# Tests: Get Service Tools
# ===================================================================

class TestGetServiceToolsEndpoint:
    """Tests for GET /services/<service_name>/tools."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.gateway, self.mocks = _create_gateway()
        self.client = self.gateway.app.test_client()

    def test_get_tools_success(self):
        self.mocks["mcp_manager"].get_service_tools.return_value = [
            {"name": "add_todo", "description": "Add a todo"}
        ]
        resp = self.client.get(
            "/services/todo/tools",
            headers=_auth_headers_basic(),
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["service"] == "todo"
        assert len(data["tools"]) == 1

    def test_get_tools_invalid_service_name(self):
        resp = self.client.get(
            "/services/bad%20name!/tools",
            headers=_auth_headers_basic(),
        )
        assert resp.status_code == 400

    def test_get_tools_service_error(self):
        self.mocks["mcp_manager"].get_service_tools.side_effect = RuntimeError("boom")
        resp = self.client.get(
            "/services/todo/tools",
            headers=_auth_headers_basic(),
        )
        assert resp.status_code == 500
        data = resp.get_json()
        assert "error" in data


# ===================================================================
# Tests: Call Service Tool
# ===================================================================

class TestCallServiceToolEndpoint:
    """Tests for POST /services/<service_name>/call."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.gateway, self.mocks = _create_gateway()
        self.client = self.gateway.app.test_client()

    def _post_call(self, service_name="todo", payload=None, headers=None):
        if payload is None:
            payload = {
                "tool": "add_todo",
                "session_id": "sess-123",
                "arguments": {"text": "Buy milk"},
                "timeout": 30,
                "metadata": {},
            }
        if headers is None:
            headers = _auth_headers_basic()
        headers["Content-Type"] = "application/json"
        return self.client.post(
            f"/services/{service_name}/call",
            data=json.dumps(payload),
            headers=headers,
        )

    def test_call_tool_creates_session_when_missing(self):
        """When no session exists, a new one is created."""
        self.mocks["session_manager"].get_session.return_value = None

        mock_client = MagicMock()
        mock_client.call_tool.return_value = {"result": "done"}
        self.mocks["mcp_manager"].create_client.return_value = mock_client

        mock_session = MagicMock()
        mock_session.service_name = "todo"
        mock_session.process = mock_client
        self.mocks["session_manager"].create_session.return_value = mock_session

        resp = self._post_call()
        assert resp.status_code == 200
        self.mocks["session_manager"].create_session.assert_called_once()

    def test_call_tool_reuses_existing_session(self):
        """When session exists and matches, it is reused."""
        mock_client = MagicMock()
        mock_client.call_tool.return_value = "ok"

        mock_session = MagicMock()
        mock_session.service_name = "todo"
        mock_session.process = mock_client
        self.mocks["session_manager"].get_session.return_value = mock_session

        resp = self._post_call()
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["result"] == "ok"
        self.mocks["session_manager"].create_session.assert_not_called()

    def test_call_tool_service_mismatch(self):
        """Error when session belongs to a different service."""
        mock_session = MagicMock()
        mock_session.service_name = "other_service"
        self.mocks["session_manager"].get_session.return_value = mock_session

        resp = self._post_call()
        assert resp.status_code == 400
        assert "other_service" in resp.get_json()["error"]

    def test_call_tool_missing_tool_parameter(self):
        resp = self._post_call(payload={
            "session_id": "sess-1",
            "arguments": {},
        })
        assert resp.status_code == 400
        assert "tool" in resp.get_json()["error"].lower()

    def test_call_tool_missing_session_id(self):
        resp = self._post_call(payload={
            "tool": "add_todo",
            "arguments": {},
        })
        assert resp.status_code == 400
        assert "session_id" in resp.get_json()["error"].lower()

    def test_call_tool_invalid_json_body(self):
        headers = _auth_headers_basic()
        headers["Content-Type"] = "application/json"
        resp = self.client.post(
            "/services/todo/call",
            data="not json",
            headers=headers,
        )
        # Flask raises BadRequest on malformed JSON which is caught by the
        # outer except Exception handler, resulting in a 500 response
        assert resp.status_code == 500

    def test_call_tool_invalid_arguments_type(self):
        resp = self._post_call(payload={
            "tool": "add_todo",
            "session_id": "sess-1",
            "arguments": "not_a_dict",
        })
        assert resp.status_code == 400
        assert "arguments" in resp.get_json()["error"].lower()

    def test_call_tool_invalid_timeout_negative(self):
        resp = self._post_call(payload={
            "tool": "add_todo",
            "session_id": "sess-1",
            "arguments": {},
            "timeout": -5,
        })
        assert resp.status_code == 400
        assert "timeout" in resp.get_json()["error"].lower()

    def test_call_tool_invalid_timeout_too_large(self):
        resp = self._post_call(payload={
            "tool": "add_todo",
            "session_id": "sess-1",
            "arguments": {},
            "timeout": 9999,
        })
        assert resp.status_code == 400
        assert "timeout" in resp.get_json()["error"].lower()

    def test_call_tool_invalid_timeout_string(self):
        resp = self._post_call(payload={
            "tool": "add_todo",
            "session_id": "sess-1",
            "arguments": {},
            "timeout": "fast",
        })
        assert resp.status_code == 400

    def test_call_tool_invalid_metadata_type(self):
        resp = self._post_call(payload={
            "tool": "add_todo",
            "session_id": "sess-1",
            "arguments": {},
            "metadata": "not_a_dict",
        })
        assert resp.status_code == 400
        assert "metadata" in resp.get_json()["error"].lower()

    def test_call_tool_invalid_service_name(self):
        resp = self._post_call(service_name="bad%20name!")
        assert resp.status_code in (400, 500)

    def test_call_tool_extracts_mcp_text_content(self):
        """MCP-format result with text content is unwrapped."""
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {
            "content": [{"type": "text", "text": "hello world"}]
        }

        mock_session = MagicMock()
        mock_session.service_name = "todo"
        mock_session.process = mock_client
        self.mocks["session_manager"].get_session.return_value = mock_session

        resp = self._post_call()
        data = resp.get_json()
        assert data["result"] == "hello world"

    def test_call_tool_preserves_non_text_mcp_content(self):
        """MCP content that is not text type is returned as-is."""
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {
            "content": [{"type": "image", "data": "base64..."}]
        }

        mock_session = MagicMock()
        mock_session.service_name = "todo"
        mock_session.process = mock_client
        self.mocks["session_manager"].get_session.return_value = mock_session

        resp = self._post_call()
        data = resp.get_json()
        # Result should be the full dict since it's not text type
        assert isinstance(data["result"], dict)
        assert "content" in data["result"]

    def test_call_tool_empty_content_list(self):
        """MCP result with empty content list is returned as-is."""
        mock_client = MagicMock()
        mock_client.call_tool.return_value = {"content": []}

        mock_session = MagicMock()
        mock_session.service_name = "todo"
        mock_session.process = mock_client
        self.mocks["session_manager"].get_session.return_value = mock_session

        resp = self._post_call()
        data = resp.get_json()
        assert data["result"] == {"content": []}

    def test_call_tool_session_creation_failure(self):
        """Error when session creation fails."""
        self.mocks["session_manager"].get_session.return_value = None
        self.mocks["mcp_manager"].create_client.side_effect = RuntimeError("cannot start")

        resp = self._post_call()
        assert resp.status_code == 500
        assert "session" in resp.get_json()["error"].lower()

    def test_call_tool_logs_security_event(self):
        """Every tool call is logged as a security event."""
        mock_client = MagicMock()
        mock_client.call_tool.return_value = "ok"

        mock_session = MagicMock()
        mock_session.service_name = "todo"
        mock_session.process = mock_client
        self.mocks["session_manager"].get_session.return_value = mock_session

        with patch.object(self.gateway, "_log_security_event") as mock_log:
            self._post_call()
            # Find the tool_call log
            tool_calls = [c for c in mock_log.call_args_list if c[0][0] == "tool_call"]
            assert len(tool_calls) >= 1

    def test_call_tool_uses_default_timeout(self):
        """When timeout is not in payload, session_manager.default_timeout is used."""
        self.mocks["session_manager"].get_session.return_value = None
        mock_client = MagicMock()
        mock_client.call_tool.return_value = "ok"
        self.mocks["mcp_manager"].create_client.return_value = mock_client

        mock_session = MagicMock()
        mock_session.service_name = "todo"
        mock_session.process = mock_client
        self.mocks["session_manager"].create_session.return_value = mock_session

        payload = {
            "tool": "add_todo",
            "session_id": "sess-1",
            "arguments": {},
        }
        resp = self._post_call(payload=payload)
        assert resp.status_code == 200


# ===================================================================
# Tests: List Sessions
# ===================================================================

class TestListSessionsEndpoint:
    """Tests for GET /sessions."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.gateway, self.mocks = _create_gateway()
        self.client = self.gateway.app.test_client()

    def test_list_sessions_success(self):
        self.mocks["session_manager"].list_sessions.return_value = {
            "sess-1": {"service_name": "todo"}
        }
        resp = self.client.get("/sessions", headers=_auth_headers_basic())
        assert resp.status_code == 200
        data = resp.get_json()
        assert "sess-1" in data

    def test_list_sessions_empty(self):
        self.mocks["session_manager"].list_sessions.return_value = {}
        resp = self.client.get("/sessions", headers=_auth_headers_basic())
        assert resp.status_code == 200
        assert resp.get_json() == {}

    def test_list_sessions_requires_auth(self):
        resp = self.client.get("/sessions")
        assert resp.status_code == 401


# ===================================================================
# Tests: Close Session
# ===================================================================

class TestCloseSessionEndpoint:
    """Tests for DELETE /sessions/<session_id>."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.gateway, self.mocks = _create_gateway()
        self.client = self.gateway.app.test_client()

    def test_close_session_success(self):
        self.mocks["session_manager"].close_session.return_value = True
        resp = self.client.delete(
            "/sessions/sess-123",
            headers=_auth_headers_basic(),
        )
        assert resp.status_code == 200
        assert "closed" in resp.get_json()["message"].lower()

    def test_close_session_not_found(self):
        self.mocks["session_manager"].close_session.return_value = False
        resp = self.client.delete(
            "/sessions/sess-999",
            headers=_auth_headers_basic(),
        )
        assert resp.status_code == 404

    def test_close_session_invalid_id(self):
        resp = self.client.delete(
            "/sessions/bad%20id!",
            headers=_auth_headers_basic(),
        )
        assert resp.status_code == 400

    def test_close_session_requires_auth(self):
        resp = self.client.delete("/sessions/sess-1")
        assert resp.status_code == 401

    def test_close_session_logs_security_event(self):
        self.mocks["session_manager"].close_session.return_value = True
        with patch.object(self.gateway, "_log_security_event") as mock_log:
            self.client.delete(
                "/sessions/sess-123",
                headers=_auth_headers_basic(),
            )
            closed_calls = [c for c in mock_log.call_args_list if c[0][0] == "session_closed"]
            assert len(closed_calls) >= 1


# ===================================================================
# Tests: Signal Handler
# ===================================================================

class TestSignalHandler:
    """Tests for _signal_handler."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.gateway, self.mocks = _create_gateway()

    def test_signal_handler_sets_shutdown_flag(self):
        self.gateway._signal_handler(15, None)
        assert self.gateway._shutdown_requested is True

    def test_signal_handler_calls_server_shutdown_when_server_exists(self):
        mock_server = MagicMock()
        self.gateway.server = mock_server

        self.gateway._signal_handler(2, None)

        # Server.shutdown is called in a daemon thread; give it a moment
        import time
        time.sleep(0.2)
        mock_server.shutdown.assert_called()

    def test_signal_handler_tolerates_no_server(self):
        """No error when server is None."""
        self.gateway.server = None
        self.gateway._signal_handler(15, None)  # should not raise


# ===================================================================
# Tests: Shutdown
# ===================================================================

class TestShutdown:
    """Tests for shutdown()."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.gateway, self.mocks = _create_gateway()

    def test_shutdown_calls_session_manager_shutdown(self):
        self.gateway.shutdown()
        self.mocks["session_manager"].shutdown.assert_called_once()

    def test_shutdown_calls_mcp_manager_shutdown(self):
        self.gateway.shutdown()
        self.mocks["mcp_manager"].shutdown.assert_called_once()

    def test_shutdown_calls_server_shutdown(self):
        mock_server = MagicMock()
        self.gateway.server = mock_server
        self.gateway.shutdown()
        mock_server.shutdown.assert_called()

    def test_shutdown_idempotent(self):
        """Second call is a no-op."""
        self.gateway.shutdown()
        self.gateway.shutdown()
        # session_manager.shutdown should only be called once
        assert self.mocks["session_manager"].shutdown.call_count == 1

    def test_shutdown_sets_cleanup_done_flag(self):
        self.gateway.shutdown()
        assert self.gateway._shutdown_cleanup_done is True

    def test_shutdown_tolerates_timeout_on_session_manager(self):
        """If session_manager.shutdown hangs, we still complete."""
        import time as time_mod

        def hang():
            time_mod.sleep(30)

        self.mocks["session_manager"].shutdown.side_effect = hang
        # The 5-second timeout in the code should prevent indefinite blocking
        self.gateway.shutdown()
        assert self.gateway._shutdown_cleanup_done is True

    def test_shutdown_tolerates_no_server(self):
        self.gateway.server = None
        self.gateway.shutdown()  # should not raise


# ===================================================================
# Tests: Run Method
# ===================================================================

class TestRunMethod:
    """Tests for run()."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.gateway, self.mocks = _create_gateway()

    def test_run_creates_server_and_serves(self):
        mock_server = MagicMock()
        with patch("signalwire_agents.mcp_gateway.gateway_service.make_server", return_value=mock_server) as mock_make, \
             patch("signalwire_agents.mcp_gateway.gateway_service.signal") as mock_signal, \
             patch("os.path.exists", return_value=False):

            # simulate immediate shutdown via KeyboardInterrupt
            mock_server.serve_forever.side_effect = KeyboardInterrupt()
            self.gateway.run()

            mock_make.assert_called_once()
            call_kwargs = mock_make.call_args
            assert call_kwargs[0][0] == "0.0.0.0"
            assert call_kwargs[0][1] == 8080

    def test_run_enables_ssl_when_cert_exists(self):
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt()

        with patch("signalwire_agents.mcp_gateway.gateway_service.make_server", return_value=mock_server), \
             patch("signalwire_agents.mcp_gateway.gateway_service.signal"), \
             patch("os.path.exists", return_value=True), \
             patch("signalwire_agents.mcp_gateway.gateway_service.ssl") as mock_ssl:

            mock_ctx = MagicMock()
            mock_ssl.SSLContext.return_value = mock_ctx
            self.gateway.run()

            mock_ssl.SSLContext.assert_called_once()
            mock_ctx.load_cert_chain.assert_called_once_with("certs/server.pem")

    def test_run_calls_shutdown_on_exit(self):
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt()

        with patch("signalwire_agents.mcp_gateway.gateway_service.make_server", return_value=mock_server), \
             patch("signalwire_agents.mcp_gateway.gateway_service.signal"), \
             patch("os.path.exists", return_value=False), \
             patch.object(self.gateway, "shutdown") as mock_shutdown:

            self.gateway.run()
            mock_shutdown.assert_called()

    def test_run_registers_signal_handlers(self):
        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt()

        with patch("signalwire_agents.mcp_gateway.gateway_service.make_server", return_value=mock_server), \
             patch("signalwire_agents.mcp_gateway.gateway_service.signal") as mock_signal_mod, \
             patch("os.path.exists", return_value=False):

            self.gateway.run()

            # SIGTERM and SIGINT handlers should be registered
            signal_calls = mock_signal_mod.signal.call_args_list
            registered_signals = [c[0][0] for c in signal_calls]
            assert mock_signal_mod.SIGTERM in registered_signals
            assert mock_signal_mod.SIGINT in registered_signals

    def test_run_uses_config_host_and_port(self):
        config = _minimal_config()
        config["server"]["host"] = "127.0.0.1"
        config["server"]["port"] = 9999
        gateway, _ = _create_gateway(config)

        mock_server = MagicMock()
        mock_server.serve_forever.side_effect = KeyboardInterrupt()

        with patch("signalwire_agents.mcp_gateway.gateway_service.make_server", return_value=mock_server) as mock_make, \
             patch("signalwire_agents.mcp_gateway.gateway_service.signal"), \
             patch("os.path.exists", return_value=False):

            gateway.run()
            assert mock_make.call_args[0][0] == "127.0.0.1"
            assert mock_make.call_args[0][1] == 9999


# ===================================================================
# Tests: Error Handler
# ===================================================================

class TestErrorHandler:
    """Tests for the generic error handler."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.gateway, self.mocks = _create_gateway()
        self.client = self.gateway.app.test_client()

    def test_unhandled_exception_returns_500(self):
        """Routes that raise unexpected exceptions return a 500."""
        self.mocks["mcp_manager"].list_services.side_effect = RuntimeError("unexpected")
        resp = self.client.get("/services", headers=_auth_headers_basic())
        assert resp.status_code == 500

    def test_unknown_route_returns_error(self):
        """Unknown routes are caught by the generic error handler."""
        resp = self.client.get("/nonexistent", headers=_auth_headers_basic())
        # The generic @app.errorhandler(Exception) catches the 404 NotFound
        # and returns a 500 with {"error": "Internal server error"}
        assert resp.status_code == 500
        data = resp.get_json()
        assert "error" in data


# ===================================================================
# Tests: Edge Cases
# ===================================================================

class TestEdgeCases:
    """Miscellaneous edge case tests."""

    def test_multiple_gateway_instances_are_independent(self):
        """Two gateways do not share state."""
        gw1, mocks1 = _create_gateway()
        gw2, mocks2 = _create_gateway()
        assert gw1.app is not gw2.app
        assert gw1.mcp_manager is not gw2.mcp_manager

    def test_config_without_server_section_uses_defaults(self):
        config = _minimal_config()
        del config["server"]
        gateway, _ = _create_gateway(config)
        # Should not blow up; run() would use defaults

    def test_config_without_logging_section(self):
        config = _minimal_config()
        del config["logging"]
        gateway, _ = _create_gateway(config)
        # No error

    def test_config_with_log_file(self):
        config = _minimal_config()
        config["logging"] = {"level": "DEBUG", "file": "/tmp/test_gateway.log"}
        # Create a real-ish mock handler that won't break the logging system
        mock_handler = MagicMock(spec=logging.FileHandler)
        mock_handler.level = logging.DEBUG
        mock_handler.formatter = None
        mock_handler.filters = []
        with patch("logging.FileHandler", return_value=mock_handler):
            gateway, _ = _create_gateway(config)
        # Clean up: remove the mock handler from the root logger to avoid
        # poisoning other tests
        root = logging.getLogger()
        root.handlers = [h for h in root.handlers if h is not mock_handler]

    def test_call_tool_with_special_chars_in_arguments(self):
        """Arguments dict with varied types should be passed through."""
        gateway, mocks = _create_gateway()
        client = gateway.app.test_client()

        mock_client = MagicMock()
        mock_client.call_tool.return_value = "ok"

        mock_session = MagicMock()
        mock_session.service_name = "todo"
        mock_session.process = mock_client
        mocks["session_manager"].get_session.return_value = mock_session

        payload = {
            "tool": "add-todo",
            "session_id": "sess-1",
            "arguments": {
                "text": "Hello <script>alert(1)</script>",
                "count": 5,
                "nested": {"key": "value"},
            },
        }
        headers = _auth_headers_basic()
        headers["Content-Type"] = "application/json"
        resp = client.post(
            "/services/todo/call",
            data=json.dumps(payload),
            headers=headers,
        )
        assert resp.status_code == 200
        # The arguments should be passed through as-is to the tool
        passed_args = mock_client.call_tool.call_args[0][1]
        assert "<script>" in passed_args["text"]

    def test_call_tool_with_zero_timeout(self):
        """Timeout of 0 should be rejected."""
        gateway, mocks = _create_gateway()
        client = gateway.app.test_client()

        payload = {
            "tool": "add-todo",
            "session_id": "sess-1",
            "arguments": {},
            "timeout": 0,
        }
        headers = _auth_headers_basic()
        headers["Content-Type"] = "application/json"
        resp = client.post(
            "/services/todo/call",
            data=json.dumps(payload),
            headers=headers,
        )
        assert resp.status_code == 400

    def test_call_tool_with_max_valid_timeout(self):
        """Timeout of exactly 3600 should be accepted."""
        gateway, mocks = _create_gateway()
        client = gateway.app.test_client()

        mock_client = MagicMock()
        mock_client.call_tool.return_value = "ok"

        mock_session = MagicMock()
        mock_session.service_name = "todo"
        mock_session.process = mock_client
        mocks["session_manager"].get_session.return_value = mock_session

        payload = {
            "tool": "add-todo",
            "session_id": "sess-1",
            "arguments": {},
            "timeout": 3600,
        }
        headers = _auth_headers_basic()
        headers["Content-Type"] = "application/json"
        resp = client.post(
            "/services/todo/call",
            data=json.dumps(payload),
            headers=headers,
        )
        assert resp.status_code == 200
