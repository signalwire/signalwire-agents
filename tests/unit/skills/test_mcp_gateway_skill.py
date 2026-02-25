"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for MCP Gateway skill module
"""

import pytest
import json
import logging
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from requests.auth import HTTPBasicAuth

from signalwire_agents.skills.mcp_gateway.skill import MCPGatewaySkill
from signalwire_agents.core.function_result import SwaigFunctionResult


def _make_skill(params=None, skip_setup=True):
    """
    Helper to create an MCPGatewaySkill with a mock agent.

    Args:
        params: Dictionary of parameters to pass to the skill.
        skip_setup: If True, manually set attributes that setup() would set
                    so callers do not need to mock the health-check request.
    Returns:
        A tuple of (skill, mock_agent).
    """
    mock_agent = Mock()
    mock_agent.name = "test_agent"
    mock_agent.define_tool = Mock()

    default_params = {
        "gateway_url": "https://gateway.example.com",
        "auth_user": "user",
        "auth_password": "pass",
    }
    if params is not None:
        default_params.update(params)

    skill = MCPGatewaySkill(mock_agent, default_params)

    if skip_setup:
        # Manually assign attributes normally set in setup()
        skill.auth_token = default_params.get("auth_token")
        if skill.auth_token:
            skill.auth = None
        else:
            skill.auth = HTTPBasicAuth(
                default_params.get("auth_user", ""),
                default_params.get("auth_password", ""),
            )
        skill.gateway_url = default_params["gateway_url"].rstrip("/")
        skill.services = default_params.get("services", [])
        skill.session_timeout = default_params.get("session_timeout", 300)
        skill.tool_prefix = default_params.get("tool_prefix", "mcp_")
        skill.retry_attempts = default_params.get("retry_attempts", 3)
        skill.request_timeout = default_params.get("request_timeout", 30)
        skill.verify_ssl = default_params.get("verify_ssl", True)
        skill.session_id = None

    return skill, mock_agent


# ---------------------------------------------------------------------------
# Class-level attributes and parameter schema
# ---------------------------------------------------------------------------

class TestMCPGatewaySkillClassAttributes:
    """Test class-level attributes and metadata."""

    def test_skill_name(self):
        """SKILL_NAME should be 'mcp_gateway'."""
        assert MCPGatewaySkill.SKILL_NAME == "mcp_gateway"

    def test_skill_description(self):
        """SKILL_DESCRIPTION should be set."""
        assert MCPGatewaySkill.SKILL_DESCRIPTION == "Bridge MCP servers with SWAIG functions"

    def test_skill_version(self):
        """SKILL_VERSION should be '1.0.0'."""
        assert MCPGatewaySkill.SKILL_VERSION == "1.0.0"

    def test_required_packages(self):
        """REQUIRED_PACKAGES should include 'requests'."""
        assert "requests" in MCPGatewaySkill.REQUIRED_PACKAGES

    def test_required_env_vars_empty(self):
        """REQUIRED_ENV_VARS should be empty by default."""
        assert MCPGatewaySkill.REQUIRED_ENV_VARS == []


class TestParameterSchema:
    """Test get_parameter_schema returns the expected schema."""

    def test_schema_contains_gateway_url(self):
        schema = MCPGatewaySkill.get_parameter_schema()
        assert "gateway_url" in schema
        assert schema["gateway_url"]["required"] is True

    def test_schema_contains_auth_token(self):
        schema = MCPGatewaySkill.get_parameter_schema()
        assert "auth_token" in schema
        assert schema["auth_token"]["required"] is False
        assert schema["auth_token"].get("hidden") is True

    def test_schema_contains_auth_user_and_password(self):
        schema = MCPGatewaySkill.get_parameter_schema()
        assert "auth_user" in schema
        assert "auth_password" in schema
        assert schema["auth_password"].get("hidden") is True

    def test_schema_contains_services(self):
        schema = MCPGatewaySkill.get_parameter_schema()
        assert "services" in schema
        assert schema["services"]["type"] == "array"

    def test_schema_defaults(self):
        schema = MCPGatewaySkill.get_parameter_schema()
        assert schema["session_timeout"]["default"] == 300
        assert schema["tool_prefix"]["default"] == "mcp_"
        assert schema["retry_attempts"]["default"] == 3
        assert schema["request_timeout"]["default"] == 30
        assert schema["verify_ssl"]["default"] is True

    def test_schema_inherits_swaig_fields(self):
        """Should include swaig_fields from SkillBase."""
        schema = MCPGatewaySkill.get_parameter_schema()
        assert "swaig_fields" in schema


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestSkillInitialization:
    """Test MCPGatewaySkill __init__ via SkillBase."""

    def test_init_sets_agent(self):
        skill, agent = _make_skill()
        assert skill.agent is agent

    def test_init_sets_params(self):
        skill, _ = _make_skill({"gateway_url": "https://gw.test"})
        assert skill.params["gateway_url"] == "https://gw.test"

    def test_init_creates_logger(self):
        skill, _ = _make_skill()
        assert skill.logger is not None

    def test_init_extracts_swaig_fields(self):
        """swaig_fields should be popped from params into skill.swaig_fields."""
        skill, _ = _make_skill({"swaig_fields": {"web_hook_url": "https://x"}})
        assert skill.swaig_fields == {"web_hook_url": "https://x"}
        assert "swaig_fields" not in skill.params


# ---------------------------------------------------------------------------
# setup() method
# ---------------------------------------------------------------------------

class TestSetup:
    """Test the setup() method."""

    @patch("signalwire_agents.utils.url_validator.validate_url", return_value=True)
    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.get")
    def test_setup_success_with_basic_auth(self, mock_get, mock_validate):
        """setup() should succeed when basic auth params are provided and health check passes."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        skill, _ = _make_skill(skip_setup=False)
        result = skill.setup()

        assert result is True
        assert isinstance(skill.auth, HTTPBasicAuth)
        assert skill.auth_token is None
        assert skill.gateway_url == "https://gateway.example.com"
        mock_get.assert_called_once()

    @patch("signalwire_agents.utils.url_validator.validate_url", return_value=True)
    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.get")
    def test_setup_success_with_token_auth(self, mock_get, mock_validate):
        """setup() should succeed with auth_token and gateway_url."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        skill, _ = _make_skill(
            params={"auth_token": "mytoken", "gateway_url": "https://gw.test/"},
            skip_setup=False,
        )
        result = skill.setup()

        assert result is True
        assert skill.auth_token == "mytoken"
        assert skill.auth is None
        # trailing slash should be stripped
        assert skill.gateway_url == "https://gw.test"

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.get")
    def test_setup_fails_missing_gateway_url_with_token(self, mock_get):
        """setup() should fail if auth_token is provided but gateway_url is missing."""
        skill, _ = _make_skill(
            params={"auth_token": "tok", "gateway_url": ""},
            skip_setup=False,
        )
        result = skill.setup()
        assert result is False

    def test_setup_fails_missing_basic_auth_params(self):
        """setup() should fail if no auth_token and basic auth params are missing."""
        skill, _ = _make_skill(
            params={"gateway_url": "https://gw.test", "auth_user": "", "auth_password": ""},
            skip_setup=False,
        )
        result = skill.setup()
        assert result is False

    def test_setup_fails_missing_gateway_url_basic_auth(self):
        """setup() should fail when gateway_url is missing with basic auth."""
        skill, _ = _make_skill(
            params={"gateway_url": "", "auth_user": "u", "auth_password": "p"},
            skip_setup=False,
        )
        result = skill.setup()
        assert result is False

    @patch("signalwire_agents.utils.url_validator.validate_url", return_value=True)
    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.get")
    def test_setup_fails_on_health_check_error(self, mock_get, mock_validate):
        """setup() should return False when the health check raises an exception."""
        mock_get.side_effect = ConnectionError("unreachable")

        skill, _ = _make_skill(skip_setup=False)
        result = skill.setup()

        assert result is False

    @patch("signalwire_agents.utils.url_validator.validate_url", return_value=True)
    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.get")
    def test_setup_fails_on_health_check_http_error(self, mock_get, mock_validate):
        """setup() should return False when health check returns non-200."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("500 Server Error")
        mock_get.return_value = mock_response

        skill, _ = _make_skill(skip_setup=False)
        result = skill.setup()

        assert result is False

    @patch("signalwire_agents.utils.url_validator.validate_url", return_value=True)
    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.get")
    def test_setup_stores_configuration_defaults(self, mock_get, mock_validate):
        """setup() should store default values for optional params."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        skill, _ = _make_skill(skip_setup=False)
        skill.setup()

        assert skill.services == []
        assert skill.session_timeout == 300
        assert skill.tool_prefix == "mcp_"
        assert skill.retry_attempts == 3
        assert skill.request_timeout == 30
        assert skill.verify_ssl is True
        assert skill.session_id is None

    @patch("signalwire_agents.utils.url_validator.validate_url", return_value=True)
    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.get")
    def test_setup_stores_custom_configuration(self, mock_get, mock_validate):
        """setup() should store custom values for optional params."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        skill, _ = _make_skill(
            params={
                "services": [{"name": "svc1"}],
                "session_timeout": 600,
                "tool_prefix": "test_",
                "retry_attempts": 5,
                "request_timeout": 60,
                "verify_ssl": False,
            },
            skip_setup=False,
        )
        skill.setup()

        assert skill.services == [{"name": "svc1"}]
        assert skill.session_timeout == 600
        assert skill.tool_prefix == "test_"
        assert skill.retry_attempts == 5
        assert skill.request_timeout == 60
        assert skill.verify_ssl is False

    @patch("signalwire_agents.utils.url_validator.validate_url", return_value=True)
    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.get")
    def test_setup_health_check_url(self, mock_get, mock_validate):
        """setup() should call /health on the gateway URL."""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        skill, _ = _make_skill(
            params={"gateway_url": "https://my-gateway.io/"},
            skip_setup=False,
        )
        skill.setup()

        mock_get.assert_called_once_with(
            "https://my-gateway.io/health",
            timeout=30,
            verify=True,
        )


# ---------------------------------------------------------------------------
# _make_request
# ---------------------------------------------------------------------------

class TestMakeRequest:
    """Test the _make_request helper."""

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_make_request_with_basic_auth(self, mock_request):
        """_make_request should attach HTTPBasicAuth when no token is set."""
        skill, _ = _make_skill()
        skill._make_request("GET", "https://gw.test/services")

        _, kwargs = mock_request.call_args
        assert "auth" in kwargs
        assert isinstance(kwargs["auth"], HTTPBasicAuth)
        assert "Authorization" not in kwargs.get("headers", {})

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_make_request_with_bearer_token(self, mock_request):
        """_make_request should send Authorization header when token is set."""
        skill, _ = _make_skill(params={"auth_token": "mytoken"})
        skill._make_request("POST", "https://gw.test/call", json={"a": 1})

        _, kwargs = mock_request.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer mytoken"
        assert "auth" not in kwargs

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_make_request_default_timeout(self, mock_request):
        """_make_request should use skill.request_timeout as default."""
        skill, _ = _make_skill()
        skill._make_request("GET", "https://gw.test/x")

        _, kwargs = mock_request.call_args
        assert kwargs["timeout"] == 30

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_make_request_custom_timeout(self, mock_request):
        """_make_request should allow callers to override timeout."""
        skill, _ = _make_skill()
        skill._make_request("GET", "https://gw.test/x", timeout=5)

        _, kwargs = mock_request.call_args
        assert kwargs["timeout"] == 5

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_make_request_verify_ssl(self, mock_request):
        """_make_request should pass verify_ssl setting."""
        skill, _ = _make_skill(params={"verify_ssl": False})
        skill.verify_ssl = False
        skill._make_request("GET", "https://gw.test/x")

        _, kwargs = mock_request.call_args
        assert kwargs["verify"] is False

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_make_request_preserves_existing_headers(self, mock_request):
        """_make_request should preserve caller-supplied headers."""
        skill, _ = _make_skill(params={"auth_token": "tok"})
        skill._make_request("GET", "https://gw.test/x", headers={"X-Custom": "val"})

        _, kwargs = mock_request.call_args
        assert kwargs["headers"]["X-Custom"] == "val"
        assert kwargs["headers"]["Authorization"] == "Bearer tok"


# ---------------------------------------------------------------------------
# register_tools
# ---------------------------------------------------------------------------

class TestRegisterTools:
    """Test the register_tools() method."""

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_register_tools_fetches_all_services_when_none_specified(self, mock_request):
        """When services list is empty, register_tools should query /services."""
        skill, agent = _make_skill(params={"services": []})
        skill.services = []

        services_response = Mock()
        services_response.status_code = 200
        services_response.raise_for_status = Mock()
        services_response.json.return_value = {"svc1": {}, "svc2": {}}

        tools_response = Mock()
        tools_response.status_code = 200
        tools_response.raise_for_status = Mock()
        tools_response.json.return_value = {"tools": []}

        def side_effect(method, url, **kwargs):
            if "/services/" in url and "/tools" in url:
                return tools_response
            if url.endswith("/services"):
                return services_response
            return Mock()

        mock_request.side_effect = side_effect
        skill.register_tools()

        # Should have populated services
        assert len(skill.services) == 2

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_register_tools_registers_mcp_tools(self, mock_request):
        """register_tools should call _register_mcp_tool for each discovered tool."""
        skill, agent = _make_skill(params={"services": [{"name": "svc1"}]})
        skill.services = [{"name": "svc1"}]

        tools_response = Mock()
        tools_response.status_code = 200
        tools_response.raise_for_status = Mock()
        tools_response.json.return_value = {
            "tools": [
                {"name": "tool_a", "description": "Tool A", "inputSchema": {}},
                {"name": "tool_b", "description": "Tool B", "inputSchema": {}},
            ]
        }

        mock_request.return_value = tools_response

        with patch.object(skill, "_register_mcp_tool") as mock_register:
            skill.register_tools()
            assert mock_register.call_count == 2

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_register_tools_filters_tools(self, mock_request):
        """register_tools should filter tools when a list is specified."""
        skill, agent = _make_skill(
            params={"services": [{"name": "svc1", "tools": ["tool_a"]}]}
        )
        skill.services = [{"name": "svc1", "tools": ["tool_a"]}]

        tools_response = Mock()
        tools_response.status_code = 200
        tools_response.raise_for_status = Mock()
        tools_response.json.return_value = {
            "tools": [
                {"name": "tool_a", "description": "Tool A", "inputSchema": {}},
                {"name": "tool_b", "description": "Tool B", "inputSchema": {}},
            ]
        }
        mock_request.return_value = tools_response

        with patch.object(skill, "_register_mcp_tool") as mock_register:
            skill.register_tools()
            mock_register.assert_called_once()
            registered_tool = mock_register.call_args[0][1]
            assert registered_tool["name"] == "tool_a"

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_register_tools_wildcard_tools(self, mock_request):
        """When tools='*', all tools should be registered."""
        skill, agent = _make_skill(
            params={"services": [{"name": "svc1", "tools": "*"}]}
        )
        skill.services = [{"name": "svc1", "tools": "*"}]

        tools_response = Mock()
        tools_response.status_code = 200
        tools_response.raise_for_status = Mock()
        tools_response.json.return_value = {
            "tools": [
                {"name": "tool_a", "inputSchema": {}},
                {"name": "tool_b", "inputSchema": {}},
            ]
        }
        mock_request.return_value = tools_response

        with patch.object(skill, "_register_mcp_tool") as mock_register:
            skill.register_tools()
            assert mock_register.call_count == 2

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_register_tools_registers_hangup_hook(self, mock_request):
        """register_tools should register a hangup hook for session cleanup."""
        skill, agent = _make_skill(params={"services": []})
        skill.services = []

        services_response = Mock()
        services_response.raise_for_status = Mock()
        services_response.json.return_value = {}
        mock_request.return_value = services_response

        skill.register_tools()

        # The hangup hook should be defined via define_tool
        agent.define_tool.assert_called()
        hangup_call = None
        for call in agent.define_tool.call_args_list:
            if call.kwargs.get("name") == "_mcp_gateway_hangup":
                hangup_call = call
                break
        assert hangup_call is not None
        assert hangup_call.kwargs.get("is_hangup_hook") is True

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_register_tools_skips_service_without_name(self, mock_request):
        """Services without a 'name' key should be skipped."""
        skill, agent = _make_skill()
        skill.services = [{"tools": "*"}]  # no 'name' key

        skill.register_tools()

        # Only the hangup hook should be registered
        calls_with_hangup = [
            c for c in agent.define_tool.call_args_list
            if c.kwargs.get("name") == "_mcp_gateway_hangup"
        ]
        assert len(calls_with_hangup) == 1
        # No MCP tool registrations expected
        assert agent.define_tool.call_count == 1

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_register_tools_handles_service_list_error(self, mock_request):
        """register_tools should log error when fetching service list fails."""
        skill, agent = _make_skill()
        skill.services = []

        mock_request.side_effect = ConnectionError("cannot connect")

        skill.register_tools()
        # Should not raise; services list stays empty

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_register_tools_handles_tools_fetch_error(self, mock_request):
        """register_tools should log error when fetching tools fails."""
        skill, agent = _make_skill()
        skill.services = [{"name": "svc1"}]

        mock_request.side_effect = ConnectionError("cannot connect")

        skill.register_tools()
        # Should still register the hangup hook
        assert agent.define_tool.call_count == 1


# ---------------------------------------------------------------------------
# _register_mcp_tool
# ---------------------------------------------------------------------------

class TestRegisterMCPTool:
    """Test _register_mcp_tool method."""

    def test_register_mcp_tool_basic(self):
        """Should register a SWAIG function with correct name and description."""
        skill, agent = _make_skill()

        tool_def = {
            "name": "search",
            "description": "Search for items",
            "inputSchema": {
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        }

        skill._register_mcp_tool("my_service", tool_def)

        agent.define_tool.assert_called_once()
        call_kwargs = agent.define_tool.call_args.kwargs
        assert call_kwargs["name"] == "mcp_my_service_search"
        assert "[my_service]" in call_kwargs["description"]
        assert "Search for items" in call_kwargs["description"]

    def test_register_mcp_tool_with_custom_prefix(self):
        """Should use the configured tool_prefix."""
        skill, agent = _make_skill(params={"tool_prefix": "x_"})
        skill.tool_prefix = "x_"

        tool_def = {"name": "foo", "description": "Foo", "inputSchema": {}}
        skill._register_mcp_tool("svc", tool_def)

        call_kwargs = agent.define_tool.call_args.kwargs
        assert call_kwargs["name"] == "x_svc_foo"

    def test_register_mcp_tool_converts_schema_properties(self):
        """Should convert inputSchema properties to SWAIG parameters."""
        skill, agent = _make_skill()

        tool_def = {
            "name": "create",
            "description": "Create item",
            "inputSchema": {
                "properties": {
                    "name": {"type": "string", "description": "Item name"},
                    "count": {"type": "integer", "description": "Count", "default": 1},
                    "kind": {"type": "string", "description": "Kind", "enum": ["a", "b"]},
                },
                "required": ["name"],
            },
        }

        skill._register_mcp_tool("svc", tool_def)

        call_kwargs = agent.define_tool.call_args.kwargs
        params = call_kwargs["parameters"]

        assert params["name"]["type"] == "string"
        assert params["name"]["description"] == "Item name"
        # 'name' is required, so no default even if one were set
        assert "default" not in params["name"]

        assert params["count"]["type"] == "integer"
        assert params["count"]["default"] == 1

        assert params["kind"]["enum"] == ["a", "b"]

    def test_register_mcp_tool_skips_tool_without_name(self):
        """Should skip tools without a name."""
        skill, agent = _make_skill()

        tool_def = {"description": "No name tool", "inputSchema": {}}
        skill._register_mcp_tool("svc", tool_def)

        agent.define_tool.assert_not_called()

    def test_register_mcp_tool_handler_calls_call_mcp_tool(self):
        """The registered handler should delegate to _call_mcp_tool."""
        skill, agent = _make_skill()

        tool_def = {"name": "echo", "description": "Echo", "inputSchema": {}}
        skill._register_mcp_tool("svc", tool_def)

        # Extract the handler that was passed to define_tool
        call_kwargs = agent.define_tool.call_args.kwargs
        handler = call_kwargs["handler"]

        with patch.object(skill, "_call_mcp_tool", return_value=SwaigFunctionResult("ok")) as mock_call:
            result = handler({"text": "hi"}, {"call_id": "123"})
            mock_call.assert_called_once_with("svc", "echo", {"text": "hi"}, {"call_id": "123"})

    def test_register_mcp_tool_empty_input_schema(self):
        """Tools with no properties should register with empty parameters."""
        skill, agent = _make_skill()

        tool_def = {"name": "noop", "description": "No-op", "inputSchema": {}}
        skill._register_mcp_tool("svc", tool_def)

        call_kwargs = agent.define_tool.call_args.kwargs
        assert call_kwargs["parameters"] == {}

    def test_register_mcp_tool_missing_description(self):
        """Should fall back to tool name when description is absent."""
        skill, agent = _make_skill()

        tool_def = {"name": "mystery", "inputSchema": {}}
        skill._register_mcp_tool("svc", tool_def)

        call_kwargs = agent.define_tool.call_args.kwargs
        assert "mystery" in call_kwargs["description"]


# ---------------------------------------------------------------------------
# _call_mcp_tool
# ---------------------------------------------------------------------------

class TestCallMCPTool:
    """Test the _call_mcp_tool method."""

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_successful_call(self, mock_request):
        """Should return SwaigFunctionResult with the result text on success."""
        skill, _ = _make_skill()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "Hello from MCP"}
        mock_request.return_value = mock_response

        raw_data = {"call_id": "call_123", "timestamp": "2025-01-01"}
        result = skill._call_mcp_tool("svc", "echo", {"msg": "hi"}, raw_data)

        assert isinstance(result, SwaigFunctionResult)
        assert result.response == "Hello from MCP"

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_successful_call_no_result_field(self, mock_request):
        """Should use 'No response' fallback when result field is missing."""
        skill, _ = _make_skill()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_request.return_value = mock_response

        result = skill._call_mcp_tool("svc", "echo", {}, {"call_id": "c1"})
        assert result.response == "No response"

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_uses_mcp_call_id_from_global_data(self, mock_request):
        """Should prefer global_data.mcp_call_id for session_id."""
        skill, _ = _make_skill()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "ok"}
        mock_request.return_value = mock_response

        raw_data = {
            "call_id": "call_123",
            "global_data": {"mcp_call_id": "custom_session_id"},
        }
        skill._call_mcp_tool("svc", "tool", {}, raw_data)

        # Verify the session_id in the request body
        call_kwargs = mock_request.call_args
        request_body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert request_body["session_id"] == "custom_session_id"

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_falls_back_to_call_id(self, mock_request):
        """Should use call_id when mcp_call_id is not in global_data."""
        skill, _ = _make_skill()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "ok"}
        mock_request.return_value = mock_response

        raw_data = {"call_id": "fallback_id"}
        skill._call_mcp_tool("svc", "tool", {}, raw_data)

        call_kwargs = mock_request.call_args
        request_body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert request_body["session_id"] == "fallback_id"

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_falls_back_to_unknown(self, mock_request):
        """Should use 'unknown' when call_id is missing from raw_data."""
        skill, _ = _make_skill()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "ok"}
        mock_request.return_value = mock_response

        skill._call_mcp_tool("svc", "tool", {}, {})

        call_kwargs = mock_request.call_args
        request_body = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert request_body["session_id"] == "unknown"

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_request_contains_metadata(self, mock_request):
        """The POST body should contain correct metadata."""
        skill, _ = _make_skill()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "ok"}
        mock_request.return_value = mock_response

        raw_data = {"call_id": "c1", "timestamp": "ts1"}
        skill._call_mcp_tool("svc", "search", {"q": "test"}, raw_data)

        request_body = mock_request.call_args.kwargs.get("json") or mock_request.call_args[1].get("json")
        assert request_body["tool"] == "search"
        assert request_body["arguments"] == {"q": "test"}
        assert request_body["timeout"] == 300
        assert request_body["metadata"]["agent_id"] == "test_agent"
        assert request_body["metadata"]["call_id"] == "c1"
        assert request_body["metadata"]["timestamp"] == "ts1"

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_retries_on_server_error(self, mock_request):
        """Should retry on 5xx errors up to retry_attempts."""
        skill, _ = _make_skill()
        skill.retry_attempts = 3

        error_response = Mock()
        error_response.status_code = 500
        error_response.json.return_value = {"error": "internal error"}

        mock_request.return_value = error_response

        result = skill._call_mcp_tool("svc", "tool", {}, {"call_id": "c1"})

        assert mock_request.call_count == 3
        assert "Failed to call" in result.response

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_no_retry_on_client_error(self, mock_request):
        """Should not retry on 4xx errors."""
        skill, _ = _make_skill()
        skill.retry_attempts = 3

        error_response = Mock()
        error_response.status_code = 400
        error_response.json.return_value = {"error": "bad request"}

        mock_request.return_value = error_response

        result = skill._call_mcp_tool("svc", "tool", {}, {"call_id": "c1"})

        assert mock_request.call_count == 1
        assert "Failed to call" in result.response

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_handles_non_json_error_response(self, mock_request):
        """Should handle error responses that are not valid JSON."""
        skill, _ = _make_skill()
        skill.retry_attempts = 1

        import requests as real_requests

        error_response = Mock()
        error_response.status_code = 400
        error_response.json.side_effect = real_requests.exceptions.JSONDecodeError("", "", 0)
        error_response.text = "Bad Request: invalid payload"

        mock_request.return_value = error_response

        result = skill._call_mcp_tool("svc", "tool", {}, {"call_id": "c1"})
        assert "Failed to call" in result.response

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_handles_timeout(self, mock_request):
        """Should handle request timeouts gracefully."""
        import requests as real_requests

        skill, _ = _make_skill()
        skill.retry_attempts = 2

        mock_request.side_effect = real_requests.exceptions.Timeout("timed out")

        result = skill._call_mcp_tool("svc", "tool", {}, {"call_id": "c1"})

        assert mock_request.call_count == 2
        assert "Failed to call" in result.response

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_handles_connection_error(self, mock_request):
        """Should handle connection errors gracefully."""
        import requests as real_requests

        skill, _ = _make_skill()
        skill.retry_attempts = 2

        mock_request.side_effect = real_requests.exceptions.ConnectionError("refused")

        result = skill._call_mcp_tool("svc", "tool", {}, {"call_id": "c1"})

        assert mock_request.call_count == 2
        assert "Failed to call" in result.response

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_handles_unexpected_exception(self, mock_request):
        """Should handle unexpected exceptions and not retry."""
        skill, _ = _make_skill()
        skill.retry_attempts = 3

        mock_request.side_effect = RuntimeError("unexpected")

        result = skill._call_mcp_tool("svc", "tool", {}, {"call_id": "c1"})

        # Unexpected errors should break immediately (no retries)
        assert mock_request.call_count == 1
        assert "Failed to call" in result.response

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_posts_to_correct_url(self, mock_request):
        """Should POST to /services/{service_name}/call."""
        skill, _ = _make_skill()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "ok"}
        mock_request.return_value = mock_response

        skill._call_mcp_tool("my_svc", "my_tool", {}, {"call_id": "c1"})

        args, kwargs = mock_request.call_args
        assert args[0] == "POST"
        assert args[1] == "https://gateway.example.com/services/my_svc/call"


# ---------------------------------------------------------------------------
# _hangup_handler
# ---------------------------------------------------------------------------

class TestHangupHandler:
    """Test the _hangup_handler method."""

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_hangup_handler_success(self, mock_request):
        """Should send DELETE to /sessions/{session_id} and return result."""
        skill, _ = _make_skill()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        raw_data = {"call_id": "session_abc"}
        result = skill._hangup_handler({}, raw_data)

        assert isinstance(result, SwaigFunctionResult)
        assert result.response == "Session cleanup complete"

        args, _ = mock_request.call_args
        assert args[0] == "DELETE"
        assert "sessions/session_abc" in args[1]

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_hangup_handler_uses_mcp_call_id(self, mock_request):
        """Should prefer global_data.mcp_call_id for session cleanup."""
        skill, _ = _make_skill()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        raw_data = {
            "call_id": "call_id_val",
            "global_data": {"mcp_call_id": "mcp_session_xyz"},
        }
        skill._hangup_handler({}, raw_data)

        args, _ = mock_request.call_args
        assert "sessions/mcp_session_xyz" in args[1]

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_hangup_handler_404_is_ok(self, mock_request):
        """A 404 response (session already gone) should not raise."""
        skill, _ = _make_skill()

        mock_response = Mock()
        mock_response.status_code = 404
        mock_request.return_value = mock_response

        result = skill._hangup_handler({}, {"call_id": "gone"})
        assert result.response == "Session cleanup complete"

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_hangup_handler_server_error(self, mock_request):
        """Should handle non-200/404 responses without raising."""
        skill, _ = _make_skill()

        mock_response = Mock()
        mock_response.status_code = 500
        mock_request.return_value = mock_response

        result = skill._hangup_handler({}, {"call_id": "c1"})
        assert result.response == "Session cleanup complete"

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_hangup_handler_exception(self, mock_request):
        """Should handle exceptions during cleanup gracefully."""
        skill, _ = _make_skill()

        mock_request.side_effect = ConnectionError("failed")

        result = skill._hangup_handler({}, {"call_id": "c1"})
        assert result.response == "Session cleanup complete"


# ---------------------------------------------------------------------------
# get_hints
# ---------------------------------------------------------------------------

class TestGetHints:
    """Test the get_hints method."""

    def test_default_hints(self):
        """Should include 'MCP' and 'gateway' by default."""
        skill, _ = _make_skill()
        hints = skill.get_hints()
        assert "MCP" in hints
        assert "gateway" in hints

    def test_hints_include_service_names(self):
        """Should include configured service names."""
        skill, _ = _make_skill()
        skill.services = [
            {"name": "slack"},
            {"name": "github"},
        ]
        hints = skill.get_hints()
        assert "slack" in hints
        assert "github" in hints

    def test_hints_handles_non_dict_services(self):
        """Should handle non-dict entries in services list gracefully."""
        skill, _ = _make_skill()
        skill.services = [
            {"name": "slack"},
            "not_a_dict",  # edge case
        ]
        hints = skill.get_hints()
        assert "slack" in hints
        # 'not_a_dict' is not a dict, so it should not add to hints
        assert "not_a_dict" not in hints

    def test_hints_handles_dict_without_name(self):
        """Should handle dict entries missing 'name' key."""
        skill, _ = _make_skill()
        skill.services = [{"tools": "*"}]
        hints = skill.get_hints()
        # Only default hints
        assert len(hints) == 2


# ---------------------------------------------------------------------------
# get_global_data
# ---------------------------------------------------------------------------

class TestGetGlobalData:
    """Test the get_global_data method."""

    def test_global_data_structure(self):
        """Should return gateway URL, session ID, and service names."""
        skill, _ = _make_skill()
        skill.session_id = "sess_123"
        skill.services = [{"name": "svc1"}, {"name": "svc2"}]

        data = skill.get_global_data()

        assert data["mcp_gateway_url"] == "https://gateway.example.com"
        assert data["mcp_session_id"] == "sess_123"
        assert data["mcp_services"] == ["svc1", "svc2"]

    def test_global_data_handles_non_dict_services(self):
        """Should convert non-dict services to strings."""
        skill, _ = _make_skill()
        skill.services = ["raw_service"]

        data = skill.get_global_data()
        assert data["mcp_services"] == ["raw_service"]

    def test_global_data_empty_services(self):
        """Should return empty list when no services."""
        skill, _ = _make_skill()
        skill.services = []

        data = skill.get_global_data()
        assert data["mcp_services"] == []


# ---------------------------------------------------------------------------
# get_prompt_sections
# ---------------------------------------------------------------------------

class TestGetPromptSections:
    """Test the get_prompt_sections method."""

    def test_prompt_sections_with_services(self):
        """Should return a prompt section when services are configured."""
        skill, _ = _make_skill()
        skill.services = [{"name": "slack", "tools": "*"}]

        sections = skill.get_prompt_sections()

        assert len(sections) == 1
        section = sections[0]
        assert section["title"] == "MCP Gateway Integration"
        assert "MCP" in section["body"]
        assert any("slack (all tools)" in b for b in section["bullets"])
        assert any("mcp_" in b for b in section["bullets"])

    def test_prompt_sections_with_specific_tools(self):
        """Should describe tool count when tools are listed."""
        skill, _ = _make_skill()
        skill.services = [{"name": "github", "tools": ["create_issue", "list_repos"]}]

        sections = skill.get_prompt_sections()

        section = sections[0]
        assert any("github (2 tools)" in b for b in section["bullets"])

    def test_prompt_sections_empty_services(self):
        """Should return empty list when no services are configured."""
        skill, _ = _make_skill()
        skill.services = []

        sections = skill.get_prompt_sections()
        assert sections == []

    def test_prompt_sections_non_dict_service(self):
        """Should handle non-dict service entries by converting to string."""
        skill, _ = _make_skill()
        skill.services = ["plain_service"]

        sections = skill.get_prompt_sections()

        assert len(sections) == 1
        assert any("plain_service" in b for b in sections[0]["bullets"])

    def test_prompt_sections_include_gateway_url(self):
        """Bullets should include the gateway URL."""
        skill, _ = _make_skill()
        skill.services = [{"name": "svc"}]

        sections = skill.get_prompt_sections()
        assert any("gateway.example.com" in b for b in sections[0]["bullets"])

    def test_prompt_sections_multiple_services(self):
        """Should list multiple services in the prompt."""
        skill, _ = _make_skill()
        skill.services = [
            {"name": "svc1", "tools": "*"},
            {"name": "svc2", "tools": ["a"]},
        ]

        sections = skill.get_prompt_sections()
        assert len(sections) == 1
        available_line = [b for b in sections[0]["bullets"] if "Available services" in b]
        assert len(available_line) == 1
        assert "svc1" in available_line[0]
        assert "svc2" in available_line[0]


# ---------------------------------------------------------------------------
# Integration-style tests (methods interacting together)
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration-style tests combining multiple methods."""

    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    def test_register_and_call_tool_end_to_end(self, mock_request):
        """Register a tool, then invoke its handler through the full call path."""
        skill, agent = _make_skill()
        skill.services = [{"name": "math_svc"}]

        # First call: GET tools
        tools_response = Mock()
        tools_response.status_code = 200
        tools_response.raise_for_status = Mock()
        tools_response.json.return_value = {
            "tools": [
                {
                    "name": "add",
                    "description": "Add two numbers",
                    "inputSchema": {
                        "properties": {
                            "a": {"type": "number", "description": "First number"},
                            "b": {"type": "number", "description": "Second number"},
                        },
                        "required": ["a", "b"],
                    },
                }
            ]
        }

        # Second call: POST tool invocation
        call_response = Mock()
        call_response.status_code = 200
        call_response.json.return_value = {"result": "The sum is 5"}

        call_count = [0]

        def request_side_effect(method, url, **kwargs):
            call_count[0] += 1
            if method == "GET":
                return tools_response
            return call_response

        mock_request.side_effect = request_side_effect

        # Register tools
        skill.register_tools()

        # Find the tool call that registered the 'add' function
        tool_calls = [
            c for c in agent.define_tool.call_args_list
            if c.kwargs.get("name") == "mcp_math_svc_add"
        ]
        assert len(tool_calls) == 1

        handler = tool_calls[0].kwargs["handler"]

        # Call the handler
        result = handler({"a": 2, "b": 3}, {"call_id": "test_call"})

        assert isinstance(result, SwaigFunctionResult)
        assert result.response == "The sum is 5"

    @patch("signalwire_agents.utils.url_validator.validate_url", return_value=True)
    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.request")
    @patch("signalwire_agents.skills.mcp_gateway.skill.requests.get")
    def test_full_lifecycle(self, mock_get, mock_request, mock_validate):
        """Test full skill lifecycle: setup -> register -> call -> hangup."""
        # Health check
        health_response = Mock()
        health_response.raise_for_status = Mock()
        mock_get.return_value = health_response

        # Tools listing
        tools_response = Mock()
        tools_response.status_code = 200
        tools_response.raise_for_status = Mock()
        tools_response.json.return_value = {
            "tools": [{"name": "ping", "description": "Ping", "inputSchema": {}}]
        }

        # Tool call response
        call_response = Mock()
        call_response.status_code = 200
        call_response.json.return_value = {"result": "pong"}

        # Hangup response
        delete_response = Mock()
        delete_response.status_code = 200

        def request_side_effect(method, url, **kwargs):
            if method == "GET":
                return tools_response
            if method == "POST":
                return call_response
            if method == "DELETE":
                return delete_response
            return Mock()

        mock_request.side_effect = request_side_effect

        skill, agent = _make_skill(
            params={"services": [{"name": "test_svc"}]},
            skip_setup=False,
        )

        # Setup
        assert skill.setup() is True

        # Register tools
        skill.register_tools()

        # Verify hints
        hints = skill.get_hints()
        assert "test_svc" in hints

        # Verify global data
        gdata = skill.get_global_data()
        assert gdata["mcp_services"] == ["test_svc"]

        # Verify prompt sections
        sections = skill.get_prompt_sections()
        assert len(sections) == 1
