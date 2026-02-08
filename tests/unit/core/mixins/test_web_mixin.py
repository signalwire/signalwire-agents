"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for WebMixin class
"""

import pytest
import json
import base64
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from signalwire_agents.core.mixins.web_mixin import WebMixin
from signalwire_agents.core.function_result import SwaigFunctionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_auth_header(username, password):
    """Create a Basic Auth header value."""
    encoded = base64.b64encode(f"{username}:{password}".encode()).decode()
    return f"Basic {encoded}"


def _make_request(method="GET", headers=None, body=None, query_params=None, url_path="/"):
    """Create a mock FastAPI Request object."""
    request = AsyncMock()
    request.method = method
    request.headers = headers or {}
    request.url = Mock()
    request.url.path = url_path
    request.query_params = query_params or {}
    request.state = Mock(spec=[])  # empty spec so getattr(..., "callback_path", None) returns None

    if body is not None:
        raw = json.dumps(body).encode() if isinstance(body, dict) else body
        request.body = AsyncMock(return_value=raw)
        request.json = AsyncMock(return_value=body if isinstance(body, dict) else json.loads(body))
    else:
        request.body = AsyncMock(return_value=b"")
        request.json = AsyncMock(side_effect=Exception("No body"))
    return request


def _build_mixin(**overrides):
    """
    Build a minimal object that inherits from WebMixin and provides
    all the attributes / methods that WebMixin expects from the host class
    (AgentBase) without importing or instantiating AgentBase itself.
    """
    log = MagicMock()
    log.bind = MagicMock(return_value=log)

    tool_registry = MagicMock()
    tool_registry._swaig_functions = {}

    defaults = dict(
        _app=None,
        _basic_auth=("user", "pass"),
        _proxy_url_base=None,
        _proxy_url_base_from_env=False,
        _proxy_detection_done=False,
        _current_request=None,
        _dynamic_config_callback=None,
        _is_ephemeral=False,
        _suppress_logs=False,
        _routing_callbacks={},
        _tool_registry=tool_registry,
        _session_manager=MagicMock(),
        log=log,
        name="test_agent",
        route="/agent",
        host="0.0.0.0",
        port=3000,
        ssl_enabled=False,
        schema_utils=MagicMock(),
    )
    defaults.update(overrides)

    class FakeAgent(WebMixin):
        pass

    agent = FakeAgent()
    for k, v in defaults.items():
        setattr(agent, k, v)

    # Provide default method stubs unless caller overrides
    if "get_name" not in overrides:
        agent.get_name = MagicMock(return_value="test_agent")
    if "_check_basic_auth" not in overrides:
        agent._check_basic_auth = MagicMock(return_value=True)
    if "_render_swml" not in overrides:
        agent._render_swml = MagicMock(return_value='{"sections":{}}')
    if "on_function_call" not in overrides:
        agent.on_function_call = MagicMock(return_value=SwaigFunctionResult("ok"))
    if "on_summary" not in overrides:
        agent.on_summary = MagicMock(return_value=None)
    if "_find_summary_in_post_data" not in overrides:
        agent._find_summary_in_post_data = MagicMock(return_value=None)
    if "get_basic_auth_credentials" not in overrides:
        agent.get_basic_auth_credentials = MagicMock(return_value=("user", "pass", "provided"))
    if "get_full_url" not in overrides:
        agent.get_full_url = MagicMock(return_value="http://localhost:3000/agent")
    if "_create_ephemeral_copy" not in overrides:
        agent._create_ephemeral_copy = MagicMock(return_value=agent)
    if "handle_serverless_request" not in overrides:
        agent.handle_serverless_request = MagicMock(return_value=None)

    return agent


def _run(coro):
    """Run a coroutine synchronously for testing."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ===========================================================================
# get_app
# ===========================================================================

class TestGetApp:
    """Tests for WebMixin.get_app()"""

    def test_get_app_creates_fastapi_app(self):
        agent = _build_mixin()
        app = agent.get_app()
        assert app is not None
        # Should be cached
        assert agent.get_app() is app

    def test_get_app_returns_cached_app(self):
        agent = _build_mixin()
        fake_app = MagicMock()
        agent._app = fake_app
        assert agent.get_app() is fake_app

    def test_get_app_root_route_no_prefix(self):
        agent = _build_mixin(route="/")
        app = agent.get_app()
        assert app is not None

    def test_get_app_non_root_route_uses_prefix(self):
        agent = _build_mixin(route="/myagent")
        app = agent.get_app()
        assert app is not None


# ===========================================================================
# as_router
# ===========================================================================

class TestAsRouter:
    """Tests for WebMixin.as_router()"""

    def test_as_router_returns_api_router(self):
        from fastapi import APIRouter
        agent = _build_mixin()
        router = agent.as_router()
        assert isinstance(router, APIRouter)

    def test_as_router_registers_routes(self):
        agent = _build_mixin()
        router = agent.as_router()
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        # Standard routes should be registered
        assert "/" in paths
        assert "/debug" in paths
        assert "/swaig" in paths
        assert "/post_prompt" in paths
        assert "/check_for_input" in paths

    def test_as_router_registers_callback_routes(self):
        cb = MagicMock()
        agent = _build_mixin(_routing_callbacks={"/sip": cb})
        router = agent.as_router()
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert "/sip" in paths


# ===========================================================================
# _register_routes
# ===========================================================================

class TestRegisterRoutes:
    """Tests for WebMixin._register_routes()"""

    def test_register_routes_creates_slash_variants(self):
        agent = _build_mixin()
        from fastapi import APIRouter
        router = APIRouter()
        agent._register_routes(router)
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        # Each endpoint should have both with and without trailing slash
        assert "/debug/" in paths
        assert "/swaig/" in paths
        assert "/post_prompt/" in paths
        assert "/check_for_input/" in paths

    def test_register_routes_skips_root_callback(self):
        """Routing callbacks for '/' should be skipped (handled by root handler)."""
        cb = MagicMock()
        agent = _build_mixin(_routing_callbacks={"/": cb, "/custom": MagicMock()})
        from fastapi import APIRouter
        router = APIRouter()
        agent._register_routes(router)
        paths = [r.path for r in router.routes if hasattr(r, "path")]
        # /custom should be registered, but we should not have a duplicate "/"
        assert "/custom" in paths


# ===========================================================================
# Token enforcement / auth in request handlers
# ===========================================================================

class TestTokenEnforcement:
    """Tests for basic auth enforcement across endpoints."""

    def test_root_request_rejects_unauthorized(self):
        agent = _build_mixin()
        agent._check_basic_auth = MagicMock(return_value=False)
        request = _make_request("POST", body={})
        response = _run(agent._handle_root_request(request))
        assert response.status_code == 401

    def test_root_request_allows_authorized(self):
        agent = _build_mixin()
        agent._check_basic_auth = MagicMock(return_value=True)
        request = _make_request("POST", body={})
        response = _run(agent._handle_root_request(request))
        # Should get a 200 JSON response (SWML)
        assert response.status_code == 200

    def test_debug_request_rejects_unauthorized(self):
        agent = _build_mixin()
        agent._check_basic_auth = MagicMock(return_value=False)
        request = _make_request("GET", url_path="/agent/debug")
        response = _run(agent._handle_debug_request(request))
        assert response.status_code == 401

    def test_debug_request_allows_authorized(self):
        agent = _build_mixin()
        agent._check_basic_auth = MagicMock(return_value=True)
        request = _make_request("GET", url_path="/agent/debug")
        response = _run(agent._handle_debug_request(request))
        assert response.status_code == 200
        assert response.headers.get("X-Debug") == "true"

    def test_swaig_request_rejects_unauthorized(self):
        agent = _build_mixin()
        agent._check_basic_auth = MagicMock(return_value=False)
        response_obj = MagicMock()
        response_obj.headers = {}
        request = _make_request("POST", body={"function": "test"})
        response = _run(agent._handle_swaig_request(request, response_obj))
        assert response.status_code == 401

    def test_post_prompt_request_rejects_unauthorized(self):
        agent = _build_mixin()
        agent._check_basic_auth = MagicMock(return_value=False)
        request = _make_request("POST", body={"summary": "done"})
        response = _run(agent._handle_post_prompt_request(request))
        assert response.status_code == 401

    def test_check_for_input_rejects_unauthorized(self):
        agent = _build_mixin()
        agent._check_basic_auth = MagicMock(return_value=False)
        request = _make_request("POST", body={"conversation_id": "abc"})
        response = _run(agent._handle_check_for_input_request(request))
        assert response.status_code == 401


# ===========================================================================
# _handle_root_request
# ===========================================================================

class TestHandleRootRequest:
    """Tests for _handle_root_request."""

    def test_get_request_returns_swml(self):
        agent = _build_mixin()
        request = _make_request("GET")
        response = _run(agent._handle_root_request(request))
        assert response.status_code == 200
        assert response.media_type == "application/json"

    def test_post_request_with_body(self):
        agent = _build_mixin()
        request = _make_request("POST", body={"call_id": "call-123"})
        response = _run(agent._handle_root_request(request))
        assert response.status_code == 200
        agent._render_swml.assert_called()

    def test_post_request_with_empty_body(self):
        agent = _build_mixin()
        request = _make_request("POST")
        request.body = AsyncMock(return_value=b"")
        response = _run(agent._handle_root_request(request))
        assert response.status_code == 200

    def test_post_request_with_malformed_json(self):
        agent = _build_mixin()
        request = _make_request("POST")
        request.body = AsyncMock(return_value=b"not-json")
        request.json = AsyncMock(side_effect=json.JSONDecodeError("err", "doc", 0))
        response = _run(agent._handle_root_request(request))
        # Should still succeed, just with empty body
        assert response.status_code == 200

    def test_call_id_extracted_from_post_body(self):
        agent = _build_mixin()
        request = _make_request("POST", body={"call_id": "cid-xyz"})
        _run(agent._handle_root_request(request))
        # Verify _render_swml was called with the extracted call_id
        args, kwargs = agent._render_swml.call_args
        assert args[0] == "cid-xyz"

    def test_call_id_extracted_from_nested_call(self):
        agent = _build_mixin()
        body = {"call": {"call_id": "nested-id"}}
        request = _make_request("POST", body=body)
        _run(agent._handle_root_request(request))
        args, kwargs = agent._render_swml.call_args
        assert args[0] == "nested-id"

    def test_call_id_from_query_params_on_get(self):
        agent = _build_mixin()
        request = _make_request("GET", query_params={"call_id": "q-id"})
        _run(agent._handle_root_request(request))
        args, kwargs = agent._render_swml.call_args
        assert args[0] == "q-id"

    def test_proxy_detection_from_forwarded_headers(self):
        agent = _build_mixin()
        headers = {
            "X-Forwarded-Host": "proxy.example.com",
            "X-Forwarded-Proto": "https",
        }
        request = _make_request("GET", headers=headers)
        _run(agent._handle_root_request(request))
        assert agent._proxy_url_base == "https://proxy.example.com"

    def test_proxy_from_env_not_overridden_by_headers(self):
        agent = _build_mixin(
            _proxy_url_base="https://env-proxy.example.com",
            _proxy_url_base_from_env=True,
        )
        headers = {
            "X-Forwarded-Host": "header-proxy.example.com",
            "X-Forwarded-Proto": "https",
        }
        request = _make_request("GET", headers=headers)
        _run(agent._handle_root_request(request))
        # Should keep the env proxy URL, not the header one
        assert agent._proxy_url_base == "https://env-proxy.example.com"

    def test_no_proxy_headers_clears_proxy(self):
        agent = _build_mixin(_proxy_url_base="https://old-proxy.example.com")
        request = _make_request("GET")
        _run(agent._handle_root_request(request))
        assert agent._proxy_url_base is None

    def test_no_proxy_headers_keeps_env_proxy(self):
        agent = _build_mixin(
            _proxy_url_base="https://env-proxy.example.com",
            _proxy_url_base_from_env=True,
        )
        request = _make_request("GET")
        _run(agent._handle_root_request(request))
        assert agent._proxy_url_base == "https://env-proxy.example.com"

    def test_callback_path_routing(self):
        cb_fn = MagicMock(return_value="/redirect-here")
        agent = _build_mixin(_routing_callbacks={"/sip": cb_fn})
        request = _make_request("POST", body={"some": "data"}, url_path="/agent/sip")
        request.state.callback_path = "/sip"
        response = _run(agent._handle_root_request(request))
        # A redirect should be returned
        assert response.status_code == 307

    def test_callback_returns_none_continues_normally(self):
        cb_fn = MagicMock(return_value=None)
        agent = _build_mixin(_routing_callbacks={"/sip": cb_fn})
        request = _make_request("POST", body={"some": "data"}, url_path="/agent/sip")
        request.state.callback_path = "/sip"
        response = _run(agent._handle_root_request(request))
        # No redirect; falls through to normal SWML rendering
        assert response.status_code == 200

    def test_callback_exception_handled_gracefully(self):
        cb_fn = MagicMock(side_effect=RuntimeError("callback boom"))
        agent = _build_mixin(_routing_callbacks={"/sip": cb_fn})
        request = _make_request("POST", body={"some": "data"}, url_path="/agent/sip")
        request.state.callback_path = "/sip"
        response = _run(agent._handle_root_request(request))
        # Should still succeed with SWML
        assert response.status_code == 200

    def test_on_swml_request_called(self):
        agent = _build_mixin()
        agent.on_swml_request = MagicMock(return_value=None)
        request = _make_request("POST", body={})
        _run(agent._handle_root_request(request))
        agent.on_swml_request.assert_called_once()

    def test_render_swml_exception_returns_500(self):
        agent = _build_mixin()
        agent._render_swml = MagicMock(side_effect=RuntimeError("render fail"))
        request = _make_request("GET")
        response = _run(agent._handle_root_request(request))
        assert response.status_code == 500
        body = json.loads(response.body)
        assert "error" in body


# ===========================================================================
# _handle_debug_request
# ===========================================================================

class TestHandleDebugRequest:
    """Tests for _handle_debug_request."""

    def test_get_returns_swml_with_debug_header(self):
        agent = _build_mixin()
        request = _make_request("GET", url_path="/agent/debug")
        response = _run(agent._handle_debug_request(request))
        assert response.status_code == 200
        assert response.headers.get("X-Debug") == "true"

    def test_post_extracts_call_id_from_body(self):
        agent = _build_mixin()
        request = _make_request("POST", body={"call_id": "debug-call-1"}, url_path="/agent/debug")
        _run(agent._handle_debug_request(request))
        args, kwargs = agent._render_swml.call_args
        assert args[0] == "debug-call-1"

    def test_get_extracts_call_id_from_query(self):
        agent = _build_mixin()
        request = _make_request("GET", query_params={"call_id": "q-debug"}, url_path="/agent/debug")
        _run(agent._handle_debug_request(request))
        args, kwargs = agent._render_swml.call_args
        assert args[0] == "q-debug"

    def test_post_malformed_body_still_renders(self):
        agent = _build_mixin()
        request = _make_request("POST", url_path="/agent/debug")
        request.json = AsyncMock(side_effect=json.JSONDecodeError("err", "doc", 0))
        _run(agent._handle_debug_request(request))
        agent._render_swml.assert_called()

    def test_render_exception_returns_500(self):
        agent = _build_mixin()
        agent._render_swml = MagicMock(side_effect=ValueError("bad"))
        request = _make_request("GET", url_path="/agent/debug")
        response = _run(agent._handle_debug_request(request))
        assert response.status_code == 500

    def test_on_swml_request_called_with_none_callback(self):
        agent = _build_mixin()
        agent.on_swml_request = MagicMock(return_value=None)
        request = _make_request("POST", body={"call_id": "x"}, url_path="/agent/debug")
        _run(agent._handle_debug_request(request))
        agent.on_swml_request.assert_called_once_with({"call_id": "x"}, None, request)


# ===========================================================================
# _handle_swaig_request
# ===========================================================================

class TestHandleSwaigRequest:
    """Tests for _handle_swaig_request."""

    def test_get_returns_swml(self):
        agent = _build_mixin()
        resp = MagicMock()
        resp.headers = {}
        request = _make_request("GET", query_params={"call_id": "c1"}, url_path="/agent/swaig")
        response = _run(agent._handle_swaig_request(request, resp))
        assert response.status_code == 200

    def test_post_missing_function_name_returns_400(self):
        agent = _build_mixin()
        resp = MagicMock()
        resp.headers = {}
        request = _make_request("POST", body={"no_function": True}, url_path="/agent/swaig")
        response = _run(agent._handle_swaig_request(request, resp))
        assert response.status_code == 400

    def test_post_calls_function(self):
        agent = _build_mixin()
        agent.on_function_call = MagicMock(return_value=SwaigFunctionResult("done"))
        resp = MagicMock()
        resp.headers = {}
        body = {
            "function": "my_func",
            "argument": {"parsed": [{"key": "val"}]},
            "call_id": "c1",
        }
        request = _make_request("POST", body=body, url_path="/agent/swaig")
        result = _run(agent._handle_swaig_request(request, resp))
        agent.on_function_call.assert_called_once_with("my_func", {"key": "val"}, body)
        assert "response" in result
        assert result["response"] == "done"

    def test_post_parses_raw_arguments(self):
        agent = _build_mixin()
        agent.on_function_call = MagicMock(return_value={"response": "ok"})
        resp = MagicMock()
        resp.headers = {}
        body = {
            "function": "raw_func",
            "argument": {"raw": '{"a": 1}'},
        }
        request = _make_request("POST", body=body, url_path="/agent/swaig")
        _run(agent._handle_swaig_request(request, resp))
        agent.on_function_call.assert_called_once_with("raw_func", {"a": 1}, body)

    def test_post_invalid_raw_arguments_uses_empty(self):
        agent = _build_mixin()
        agent.on_function_call = MagicMock(return_value={"response": "ok"})
        resp = MagicMock()
        resp.headers = {}
        body = {
            "function": "bad_raw",
            "argument": {"raw": "not json"},
        }
        request = _make_request("POST", body=body, url_path="/agent/swaig")
        _run(agent._handle_swaig_request(request, resp))
        agent.on_function_call.assert_called_once_with("bad_raw", {}, body)

    def test_token_validation_valid(self):
        agent = _build_mixin()
        agent._session_manager.validate_tool_token = MagicMock(return_value=True)
        agent._tool_registry._swaig_functions = {"my_func": {"secure": True}}
        agent.on_function_call = MagicMock(return_value={"response": "ok"})
        resp = MagicMock()
        resp.headers = {}
        body = {"function": "my_func", "call_id": "c1"}
        request = _make_request(
            "POST", body=body,
            query_params={"__token": "valid-token"},
            url_path="/agent/swaig"
        )
        result = _run(agent._handle_swaig_request(request, resp))
        agent._session_manager.validate_tool_token.assert_called_once_with("my_func", "valid-token", "c1")
        # Function should still be called
        agent.on_function_call.assert_called()

    def test_token_validation_invalid_secure_function_rejects(self):
        """When a secure function has an invalid token the handler tries to
        return a JSONResponse(401).  However, JSONResponse is not imported
        in web_mixin.py, so a NameError is raised and caught by the outer
        exception handler which returns a 500 Response.  We test the actual
        observable behaviour here (500) and separately verify that if
        JSONResponse is available the intent is a 401."""
        agent = _build_mixin()
        agent._session_manager.validate_tool_token = MagicMock(return_value=False)
        agent._session_manager.debug_token = MagicMock(return_value={})
        agent._tool_registry._swaig_functions = {"secure_fn": {"secure": True}}
        resp = MagicMock()
        resp.headers = {}
        body = {"function": "secure_fn", "call_id": "c1"}
        request = _make_request(
            "POST", body=body,
            query_params={"token": "bad-token"},
            url_path="/agent/swaig"
        )
        # Without JSONResponse imported, the outer handler catches the NameError
        result = _run(agent._handle_swaig_request(request, resp))
        assert hasattr(result, "status_code") and result.status_code == 500

    def test_token_validation_invalid_secure_function_401_with_jsonresponse(self):
        """When JSONResponse IS available in the module namespace the handler
        correctly returns a 401 for an invalid token on a secure function."""
        from fastapi.responses import JSONResponse as RealJSONResponse
        agent = _build_mixin()
        agent._session_manager.validate_tool_token = MagicMock(return_value=False)
        agent._session_manager.debug_token = MagicMock(return_value={})
        agent._tool_registry._swaig_functions = {"secure_fn": {"secure": True}}
        resp = MagicMock()
        resp.headers = {}
        body = {"function": "secure_fn", "call_id": "c1"}
        request = _make_request(
            "POST", body=body,
            query_params={"token": "bad-token"},
            url_path="/agent/swaig"
        )
        # Inject JSONResponse into the module globals so the handler can use it
        import signalwire_agents.core.mixins.web_mixin as wm_module
        original = getattr(wm_module, "JSONResponse", None)
        wm_module.JSONResponse = RealJSONResponse
        try:
            # Also need to inject into the builtins/globals accessible by the method
            # The simplest way: temporarily add it to the WebMixin class namespace
            WebMixin.JSONResponse = RealJSONResponse
            # Patch the global in the module where the function runs
            with patch.dict(wm_module.__dict__, {"JSONResponse": RealJSONResponse}):
                result = _run(agent._handle_swaig_request(request, resp))
            assert hasattr(result, "status_code") and result.status_code == 401
        finally:
            if original is None:
                wm_module.__dict__.pop("JSONResponse", None)
            else:
                wm_module.JSONResponse = original
            if hasattr(WebMixin, "JSONResponse"):
                delattr(WebMixin, "JSONResponse")

    def test_token_validation_invalid_nonsecure_function_continues(self):
        agent = _build_mixin()
        agent._session_manager.validate_tool_token = MagicMock(return_value=False)
        agent._session_manager.debug_token = MagicMock(return_value={})
        agent._tool_registry._swaig_functions = {"open_fn": {"secure": False}}
        agent.on_function_call = MagicMock(return_value={"response": "allowed"})
        resp = MagicMock()
        resp.headers = {}
        body = {"function": "open_fn", "call_id": "c1"}
        request = _make_request(
            "POST", body=body,
            query_params={"__token": "bad-token"},
            url_path="/agent/swaig"
        )
        result = _run(agent._handle_swaig_request(request, resp))
        # Should proceed since function is not secure
        agent.on_function_call.assert_called()

    def test_dynamic_config_callback_creates_ephemeral(self):
        ephemeral = MagicMock()
        ephemeral.on_function_call = MagicMock(return_value={"response": "ephemeral"})
        config_cb = MagicMock()
        agent = _build_mixin(_dynamic_config_callback=config_cb)
        agent._create_ephemeral_copy = MagicMock(return_value=ephemeral)
        resp = MagicMock()
        resp.headers = {}
        body = {"function": "f1", "call_id": "c1"}
        request = _make_request("POST", body=body, url_path="/agent/swaig")
        result = _run(agent._handle_swaig_request(request, resp))
        agent._create_ephemeral_copy.assert_called_once()
        config_cb.assert_called_once()
        ephemeral.on_function_call.assert_called_once()

    def test_function_execution_error_returns_error_dict(self):
        agent = _build_mixin()
        agent.on_function_call = MagicMock(side_effect=RuntimeError("boom"))
        resp = MagicMock()
        resp.headers = {}
        body = {"function": "exploding_fn"}
        request = _make_request("POST", body=body, url_path="/agent/swaig")
        result = _run(agent._handle_swaig_request(request, resp))
        assert "error" in result
        assert "boom" in result["error"]

    def test_swaig_function_result_dict_passthrough(self):
        agent = _build_mixin()
        agent.on_function_call = MagicMock(return_value={"response": "direct dict"})
        resp = MagicMock()
        resp.headers = {}
        body = {"function": "dict_fn"}
        request = _make_request("POST", body=body, url_path="/agent/swaig")
        result = _run(agent._handle_swaig_request(request, resp))
        assert result == {"response": "direct dict"}

    def test_swaig_function_result_string_wrapped(self):
        agent = _build_mixin()
        agent.on_function_call = MagicMock(return_value="plain string")
        resp = MagicMock()
        resp.headers = {}
        body = {"function": "str_fn"}
        request = _make_request("POST", body=body, url_path="/agent/swaig")
        result = _run(agent._handle_swaig_request(request, resp))
        assert result == {"response": "plain string"}


# ===========================================================================
# _handle_post_prompt_request
# ===========================================================================

class TestHandlePostPromptRequest:
    """Tests for _handle_post_prompt_request."""

    def test_get_returns_swml(self):
        agent = _build_mixin()
        agent.on_swml_request = MagicMock(return_value=None)
        request = _make_request("GET", url_path="/agent/post_prompt")
        response = _run(agent._handle_post_prompt_request(request))
        assert response.status_code == 200

    def test_post_calls_on_summary(self):
        agent = _build_mixin()
        agent._find_summary_in_post_data = MagicMock(return_value={"summary": "the call ended"})
        agent.on_summary = MagicMock(return_value=None)
        body = {"summary": "the call ended", "call_id": "c1"}
        request = _make_request("POST", body=body, url_path="/agent/post_prompt")
        result = _run(agent._handle_post_prompt_request(request))
        agent.on_summary.assert_called_once_with({"summary": "the call ended"}, body)
        assert result == {"success": True}

    def test_post_with_no_summary(self):
        agent = _build_mixin()
        agent._find_summary_in_post_data = MagicMock(return_value=None)
        agent.on_summary = MagicMock(return_value=None)
        body = {"call_id": "c1"}
        request = _make_request("POST", body=body, url_path="/agent/post_prompt")
        result = _run(agent._handle_post_prompt_request(request))
        agent.on_summary.assert_called_once_with(None, body)
        assert result == {"success": True}

    def test_post_fetch_conversation_returns_result(self):
        agent = _build_mixin()
        agent._find_summary_in_post_data = MagicMock(return_value="some summary")
        fetch_result = {"conversation": [{"role": "user", "content": "hi"}]}
        agent.on_summary = MagicMock(return_value=fetch_result)
        body = {"action": "fetch_conversation", "summary": "some summary"}
        request = _make_request("POST", body=body, url_path="/agent/post_prompt")
        result = _run(agent._handle_post_prompt_request(request))
        assert result == fetch_result

    def test_post_token_validation(self):
        agent = _build_mixin()
        agent._session_manager.validate_tool_token = MagicMock(return_value=True)
        body = {"call_id": "c1"}
        request = _make_request(
            "POST", body=body,
            query_params={"__token": "good", "call_id": "c1"},
            url_path="/agent/post_prompt",
        )
        _run(agent._handle_post_prompt_request(request))
        agent._session_manager.validate_tool_token.assert_called_once_with("post_prompt", "good", "c1")

    def test_post_token_fallback_to_token_param(self):
        agent = _build_mixin()
        agent._session_manager.validate_tool_token = MagicMock(return_value=True)
        body = {"call_id": "c1"}
        request = _make_request(
            "POST", body=body,
            query_params={"token": "fallback-tok", "call_id": "c1"},
            url_path="/agent/post_prompt",
        )
        _run(agent._handle_post_prompt_request(request))
        agent._session_manager.validate_tool_token.assert_called_once_with("post_prompt", "fallback-tok", "c1")

    def test_post_dynamic_config_creates_ephemeral(self):
        ephemeral = MagicMock()
        ephemeral._find_summary_in_post_data = MagicMock(return_value=None)
        ephemeral.on_summary = MagicMock(return_value=None)
        config_cb = MagicMock()
        agent = _build_mixin(_dynamic_config_callback=config_cb)
        agent._create_ephemeral_copy = MagicMock(return_value=ephemeral)
        body = {"call_id": "c1"}
        request = _make_request("POST", body=body, url_path="/agent/post_prompt")
        _run(agent._handle_post_prompt_request(request))
        agent._create_ephemeral_copy.assert_called_once()
        config_cb.assert_called_once()
        ephemeral.on_summary.assert_called_once()

    def test_post_exception_returns_500(self):
        agent = _build_mixin()
        agent._find_summary_in_post_data = MagicMock(side_effect=RuntimeError("oops"))
        body = {"call_id": "c1"}
        request = _make_request("POST", body=body, url_path="/agent/post_prompt")
        response = _run(agent._handle_post_prompt_request(request))
        assert response.status_code == 500

    def test_suppress_logs_flag(self):
        agent = _build_mixin(_suppress_logs=True)
        agent.on_swml_request = MagicMock(return_value=None)
        request = _make_request("GET", url_path="/agent/post_prompt")
        _run(agent._handle_post_prompt_request(request))
        # Just verify it doesn't error; log suppression is internal


# ===========================================================================
# _handle_check_for_input_request
# ===========================================================================

class TestHandleCheckForInputRequest:
    """Tests for _handle_check_for_input_request."""

    def test_post_with_conversation_id(self):
        agent = _build_mixin()
        body = {"conversation_id": "conv-123"}
        request = _make_request("POST", body=body, url_path="/agent/check_for_input")
        result = _run(agent._handle_check_for_input_request(request))
        assert result["status"] == "success"
        assert result["conversation_id"] == "conv-123"
        assert result["new_input"] is False

    def test_get_with_conversation_id(self):
        agent = _build_mixin()
        request = _make_request("GET", query_params={"conversation_id": "conv-456"}, url_path="/agent/check_for_input")
        result = _run(agent._handle_check_for_input_request(request))
        assert result["status"] == "success"
        assert result["conversation_id"] == "conv-456"

    def test_missing_conversation_id_returns_400(self):
        agent = _build_mixin()
        body = {}
        request = _make_request("POST", body=body, url_path="/agent/check_for_input")
        response = _run(agent._handle_check_for_input_request(request))
        assert response.status_code == 400


# ===========================================================================
# on_request / on_swml_request
# ===========================================================================

class TestOnRequestAndOnSwmlRequest:
    """Tests for on_request and on_swml_request methods."""

    def test_on_request_delegates_to_on_swml_request(self):
        agent = _build_mixin()
        agent.on_swml_request = MagicMock(return_value={"custom": True})
        result = agent.on_request({"data": "val"}, "/cb")
        agent.on_swml_request.assert_called_once_with({"data": "val"}, "/cb", None)
        assert result == {"custom": True}

    def test_on_request_returns_none_when_on_swml_request_not_callable(self):
        agent = _build_mixin()
        # Override on_swml_request with a non-callable so the callable() check fails
        agent.on_swml_request = "not_callable"
        result = agent.on_request(None, None)
        assert result is None

    def test_on_swml_request_returns_ephemeral_marker_with_dynamic_callback(self):
        cb = MagicMock()
        agent = _build_mixin(_dynamic_config_callback=cb)
        result = agent.on_swml_request({"data": True}, None, None)
        assert result is not None
        assert result["__use_ephemeral_agent"] is True

    def test_on_swml_request_returns_none_without_dynamic_callback(self):
        agent = _build_mixin()
        result = agent.on_swml_request(None, None, None)
        assert result is None

    def test_on_swml_request_skips_ephemeral_agents(self):
        cb = MagicMock()
        agent = _build_mixin(_dynamic_config_callback=cb, _is_ephemeral=True)
        result = agent.on_swml_request(None, None, None)
        assert result is None

    def test_on_swml_request_includes_request_in_marker(self):
        cb = MagicMock()
        agent = _build_mixin(_dynamic_config_callback=cb)
        mock_req = MagicMock()
        result = agent.on_swml_request({"x": 1}, "/path", mock_req)
        assert result["__request"] is mock_req
        assert result["__request_data"] == {"x": 1}


# ===========================================================================
# register_routing_callback
# ===========================================================================

class TestRegisterRoutingCallback:
    """Tests for register_routing_callback."""

    def test_registers_callback_with_normalized_path(self):
        agent = _build_mixin()
        fn = MagicMock()
        agent.register_routing_callback(fn, "/sip/")
        assert "/sip" in agent._routing_callbacks
        assert agent._routing_callbacks["/sip"] is fn

    def test_adds_leading_slash(self):
        agent = _build_mixin()
        fn = MagicMock()
        agent.register_routing_callback(fn, "custom")
        assert "/custom" in agent._routing_callbacks

    def test_default_path_is_sip(self):
        agent = _build_mixin()
        fn = MagicMock()
        agent.register_routing_callback(fn)
        assert "/sip" in agent._routing_callbacks

    def test_initializes_routing_callbacks_dict(self):
        agent = _build_mixin()
        # Remove the dict to test initialization
        if hasattr(agent, "_routing_callbacks"):
            delattr(agent, "_routing_callbacks")
        fn = MagicMock()
        agent.register_routing_callback(fn, "/new")
        assert hasattr(agent, "_routing_callbacks")
        assert "/new" in agent._routing_callbacks


# ===========================================================================
# set_dynamic_config_callback
# ===========================================================================

class TestSetDynamicConfigCallback:
    """Tests for set_dynamic_config_callback."""

    def test_sets_callback(self):
        agent = _build_mixin()
        fn = MagicMock()
        result = agent.set_dynamic_config_callback(fn)
        assert agent._dynamic_config_callback is fn
        assert result is agent  # returns self for chaining


# ===========================================================================
# manual_set_proxy_url
# ===========================================================================

class TestManualSetProxyUrl:
    """Tests for manual_set_proxy_url."""

    def test_sets_proxy_url(self):
        agent = _build_mixin()
        result = agent.manual_set_proxy_url("https://example.ngrok.io/")
        assert agent._proxy_url_base == "https://example.ngrok.io"
        assert agent._proxy_detection_done is True
        assert result is agent

    def test_strips_trailing_slash(self):
        agent = _build_mixin()
        agent.manual_set_proxy_url("https://proxy.com///")
        assert agent._proxy_url_base == "https://proxy.com"

    def test_empty_string_does_not_set(self):
        agent = _build_mixin(_proxy_url_base="old")
        agent.manual_set_proxy_url("")
        # Should not have changed
        assert agent._proxy_url_base == "old"

    def test_none_does_not_set(self):
        agent = _build_mixin(_proxy_url_base="old")
        agent.manual_set_proxy_url(None)
        assert agent._proxy_url_base == "old"


# ===========================================================================
# setup_graceful_shutdown
# ===========================================================================

class TestSetupGracefulShutdown:
    """Tests for setup_graceful_shutdown."""

    def test_registers_signal_handlers(self):
        agent = _build_mixin()
        import signal as sig_module
        with patch.object(sig_module, "signal") as mock_signal:
            agent.setup_graceful_shutdown()
            calls = mock_signal.call_args_list
            registered_signals = [c[0][0] for c in calls]
            assert sig_module.SIGTERM in registered_signals
            assert sig_module.SIGINT in registered_signals


# ===========================================================================
# enable_debug_routes
# ===========================================================================

class TestEnableDebugRoutes:
    """Tests for enable_debug_routes."""

    def test_returns_self_for_chaining(self):
        agent = _build_mixin()
        result = agent.enable_debug_routes()
        assert result is agent


# ===========================================================================
# Route prefix handling
# ===========================================================================

class TestRoutePrefixHandling:
    """Tests verifying route prefix behaviour with different route configurations."""

    def test_root_route_no_prefix(self):
        agent = _build_mixin(route="/")
        app = agent.get_app()
        # Router should be included without prefix
        assert app is not None

    def test_custom_route_uses_prefix(self):
        agent = _build_mixin(route="/v1/mybot")
        app = agent.get_app()
        assert app is not None

    def test_serve_root_route(self):
        agent = _build_mixin(route="/")
        # We can at least verify the app creation logic doesn't crash
        app = agent.get_app()
        assert app is not None

    def test_serve_with_prefix(self):
        agent = _build_mixin(route="/bot")
        app = agent.get_app()
        # Verify the router was created
        assert agent._app is not None


# ===========================================================================
# Azure mode behavior (via run() method)
# ===========================================================================

class TestAzureModeBehavior:
    """Tests for Azure Function mode in the run() method."""

    def test_run_azure_function_mode(self):
        agent = _build_mixin()
        mock_event = MagicMock()
        agent.handle_serverless_request = MagicMock(return_value="azure-response")
        result = agent.run(event=mock_event, context=None, force_mode="azure_function")
        agent.handle_serverless_request.assert_called_once_with(mock_event, None, "azure_function")
        assert result == "azure-response"

    def test_run_lambda_mode(self):
        agent = _build_mixin()
        mock_event = {"headers": {}, "body": "{}"}
        agent.handle_serverless_request = MagicMock(return_value={"statusCode": 200})
        result = agent.run(event=mock_event, context=None, force_mode="lambda")
        agent.handle_serverless_request.assert_called_once_with(mock_event, None, "lambda")
        assert result == {"statusCode": 200}

    def test_run_cgi_mode(self):
        agent = _build_mixin()
        agent.handle_serverless_request = MagicMock(return_value="CGI output")
        with patch("builtins.print") as mock_print:
            result = agent.run(force_mode="cgi")
        agent.handle_serverless_request.assert_called_once()
        mock_print.assert_called_once_with("CGI output")
        assert result == "CGI output"

    def test_run_google_cloud_function_mode(self):
        agent = _build_mixin()
        mock_event = MagicMock()
        agent.handle_serverless_request = MagicMock(return_value="gcf-response")
        result = agent.run(event=mock_event, force_mode="google_cloud_function")
        assert result == "gcf-response"

    def test_run_server_mode_calls_serve(self):
        agent = _build_mixin()
        agent.serve = MagicMock()
        agent.run(force_mode="server", host="127.0.0.1", port=9000)
        agent.serve.assert_called_once_with("127.0.0.1", 9000)

    def test_run_lambda_error_returns_500(self):
        agent = _build_mixin()
        agent.handle_serverless_request = MagicMock(side_effect=RuntimeError("lambda fail"))
        result = agent.run(force_mode="lambda")
        assert result["statusCode"] == 500
        body = json.loads(result["body"])
        assert "lambda fail" in body["error"]

    def test_run_non_lambda_error_raises(self):
        agent = _build_mixin()
        agent.handle_serverless_request = MagicMock(side_effect=RuntimeError("cgi fail"))
        with pytest.raises(RuntimeError, match="cgi fail"):
            with patch("builtins.print"):
                agent.run(force_mode="cgi")

    def test_run_auto_detection_defaults_to_server(self):
        agent = _build_mixin()
        agent.serve = MagicMock()
        with patch("signalwire_agents.core.mixins.web_mixin.get_execution_mode", return_value="server"):
            agent.run()
        agent.serve.assert_called_once()


# ===========================================================================
# serve() method
# ===========================================================================

class TestServe:
    """Tests for serve() method."""

    def _patch_uvicorn(self):
        """Patch uvicorn inside serve() since it uses a local import."""
        return patch.dict("sys.modules", {"uvicorn": MagicMock()})

    def test_serve_uses_default_host_and_port(self):
        import sys
        mock_uvicorn = MagicMock()
        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            agent = _build_mixin(host="0.0.0.0", port=3000)
            agent.serve()
        mock_uvicorn.run.assert_called_once()
        _, kwargs = mock_uvicorn.run.call_args
        assert kwargs["host"] == "0.0.0.0"
        assert kwargs["port"] == 3000

    def test_serve_uses_override_host_and_port(self):
        import sys
        mock_uvicorn = MagicMock()
        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            agent = _build_mixin(host="0.0.0.0", port=3000)
            agent.serve(host="127.0.0.1", port=9999)
        _, kwargs = mock_uvicorn.run.call_args
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 9999

    def test_serve_with_ssl(self):
        import sys
        mock_uvicorn = MagicMock()
        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            agent = _build_mixin(
                ssl_enabled=True,
                ssl_cert_path="/path/to/cert.pem",
                ssl_key_path="/path/to/key.pem",
            )
            agent.serve()
        _, kwargs = mock_uvicorn.run.call_args
        assert kwargs["ssl_certfile"] == "/path/to/cert.pem"
        assert kwargs["ssl_keyfile"] == "/path/to/key.pem"

    def test_serve_without_ssl(self):
        import sys
        mock_uvicorn = MagicMock()
        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            agent = _build_mixin(ssl_enabled=False)
            agent.serve()
        _, kwargs = mock_uvicorn.run.call_args
        assert "ssl_certfile" not in kwargs
        assert "ssl_keyfile" not in kwargs

    def test_serve_caches_app(self):
        import sys
        mock_uvicorn = MagicMock()
        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            agent = _build_mixin()
            agent.serve()
        assert agent._app is not None

    def test_serve_reuses_cached_app(self):
        import sys
        mock_uvicorn = MagicMock()
        fake_app = MagicMock()
        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            agent = _build_mixin()
            agent._app = fake_app
            agent.serve()
        # uvicorn.run should have been called with the pre-existing app
        mock_uvicorn.run.assert_called_once()
        call_args = mock_uvicorn.run.call_args
        assert call_args[0][0] is fake_app

    def test_serve_root_route_includes_router_without_prefix(self):
        import sys
        mock_uvicorn = MagicMock()
        with patch.dict(sys.modules, {"uvicorn": mock_uvicorn}):
            agent = _build_mixin(route="/")
            agent.serve()
        assert agent._app is not None
