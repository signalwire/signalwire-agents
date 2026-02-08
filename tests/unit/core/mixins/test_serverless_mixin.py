"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for ServerlessMixin class
"""

import pytest
import json
import base64
import os
import sys
from unittest.mock import Mock, MagicMock, patch, PropertyMock

from signalwire_agents.core.mixins.serverless_mixin import ServerlessMixin
from signalwire_agents.core.function_result import SwaigFunctionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockLogger:
    """Minimal structured logger mock that supports .bind() chaining."""

    def bind(self, **kwargs):
        return self

    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class _MockToolRegistry:
    """Minimal tool registry mock."""

    def __init__(self, functions=None):
        self._swaig_functions = functions or {}


class ConcreteServerlessMixin(ServerlessMixin):
    """Concrete implementation of ServerlessMixin for testing."""

    def __init__(self, swaig_functions=None):
        self.log = _MockLogger()
        self._tool_registry = _MockToolRegistry(swaig_functions or {})
        self._proxy_url_base = None
        self._proxy_url_base_from_env = False
        self._swml_response = '{"sections": {}}'

    # Stubs for methods the mixin calls but that live in other mixins / base class
    def _check_cgi_auth(self):
        return True

    def _send_cgi_auth_challenge(self):
        return "Status: 401\r\n\r\n"

    def _check_lambda_auth(self, event):
        return True

    def _send_lambda_auth_challenge(self):
        return {"statusCode": 401, "body": "Unauthorized"}

    def _check_google_cloud_function_auth(self, request):
        return True

    def _send_google_cloud_function_auth_challenge(self):
        return Mock(status_code=401)

    def _check_azure_function_auth(self, req):
        return True

    def _send_azure_function_auth_challenge(self):
        return Mock(status_code=401)

    def _render_swml(self, **kwargs):
        return self._swml_response

    def on_function_call(self, function_name, args, raw_data):
        fn = self._tool_registry._swaig_functions.get(function_name)
        if fn:
            return fn(args, raw_data)
        return {"error": f"Function '{function_name}' not found"}


def _make_flask_request(path="/", method="GET", json_data=None, url=None):
    """Create a mock Flask request for GCF tests."""
    request = Mock()
    request.path = path
    request.method = method
    request.url = url or f"https://us-central1-myproject.cloudfunctions.net{path}"

    if json_data is not None:
        request.is_json = True
        request.get_json = Mock(return_value=json_data)
        request.get_data = Mock(return_value=json.dumps(json_data).encode("utf-8"))
    else:
        request.is_json = False
        request.get_json = Mock(return_value=None)
        request.get_data = Mock(return_value=b"")

    return request


def _make_azure_request(url=None, method="GET", body=None):
    """Create a mock Azure Functions HttpRequest for Azure tests."""
    req = Mock()
    req.url = url or "https://myapp.azurewebsites.net/api/myagent"
    req.method = method
    if body is not None:
        req.get_body = Mock(return_value=json.dumps(body).encode("utf-8"))
    else:
        req.get_body = Mock(return_value=b"")
    return req


def _swaig_body(function_name, args=None, call_id=None):
    """Build a typical SWAIG request body dict."""
    body = {
        "function": function_name,
        "argument": {
            "parsed": [args] if args else [],
            "raw": json.dumps(args) if args else "{}",
        },
    }
    if call_id:
        body["call_id"] = call_id
    return body


# ---------------------------------------------------------------------------
# Lambda handler tests
# ---------------------------------------------------------------------------

class TestLambdaHandlerRootPath:
    """Lambda handler returns SWML for root path requests."""

    def test_root_path_returns_swml(self):
        """Root path returns SWML document with 200."""
        mixin = ConcreteServerlessMixin()
        event = {"rawPath": "/", "headers": {}}
        result = mixin.handle_serverless_request(event=event, mode="lambda")
        assert result["statusCode"] == 200
        assert result["headers"]["Content-Type"] == "application/json"
        assert result["body"] == mixin._swml_response

    def test_empty_raw_path_returns_swml(self):
        """Empty rawPath returns SWML."""
        mixin = ConcreteServerlessMixin()
        event = {"rawPath": "", "headers": {}}
        result = mixin.handle_serverless_request(event=event, mode="lambda")
        assert result["statusCode"] == 200
        assert result["body"] == mixin._swml_response

    def test_none_event_returns_swml(self):
        """None event returns SWML."""
        mixin = ConcreteServerlessMixin()
        result = mixin.handle_serverless_request(event=None, mode="lambda")
        assert result["statusCode"] == 200
        assert result["body"] == mixin._swml_response


class TestLambdaHandlerPathRouting:
    """Lambda handler routes path-based function calls."""

    def test_path_based_function_call(self):
        """Path like /say_hello invokes the named function."""
        mixin = ConcreteServerlessMixin(
            swaig_functions={"say_hello": lambda args, raw: {"response": "hi"}}
        )
        body = _swaig_body("say_hello", {"name": "Alice"})
        event = {
            "rawPath": "/say_hello",
            "headers": {},
            "body": json.dumps(body),
        }
        result = mixin.handle_serverless_request(event=event, mode="lambda")
        assert result["statusCode"] == 200
        result_body = json.loads(result["body"])
        assert result_body["response"] == "hi"

    def test_swaig_endpoint_with_function_in_body(self):
        """/swaig endpoint dispatches using function name from body."""
        mixin = ConcreteServerlessMixin(
            swaig_functions={"greet": lambda args, raw: {"response": "hello"}}
        )
        body = _swaig_body("greet", {"name": "Bob"})
        event = {
            "rawPath": "/swaig",
            "headers": {},
            "body": json.dumps(body),
        }
        result = mixin.handle_serverless_request(event=event, mode="lambda")
        assert result["statusCode"] == 200
        result_body = json.loads(result["body"])
        assert result_body["response"] == "hello"


class TestLambdaHandlerV1Payload:
    """Lambda handler supports API Gateway v1 (REST API) payload format."""

    def test_v1_path_parameters_proxy(self):
        """REST API v1 uses pathParameters.proxy for routing."""
        mixin = ConcreteServerlessMixin(
            swaig_functions={"lookup": lambda args, raw: {"status": "found"}}
        )
        body = _swaig_body("lookup")
        event = {
            "rawPath": "",
            "pathParameters": {"proxy": "lookup"},
            "headers": {},
            "body": json.dumps(body),
        }
        result = mixin.handle_serverless_request(event=event, mode="lambda")
        assert result["statusCode"] == 200
        result_body = json.loads(result["body"])
        assert result_body["status"] == "found"


class TestLambdaHandlerBase64Body:
    """Lambda handler decodes base64-encoded request bodies."""

    def test_base64_encoded_body(self):
        """isBase64Encoded flag triggers base64 decoding."""
        mixin = ConcreteServerlessMixin(
            swaig_functions={"echo": lambda args, raw: {"args": args}}
        )
        body = _swaig_body("echo", {"val": 42})
        encoded = base64.b64encode(json.dumps(body).encode("utf-8")).decode("utf-8")
        event = {
            "rawPath": "/echo",
            "headers": {},
            "body": encoded,
            "isBase64Encoded": True,
        }
        result = mixin.handle_serverless_request(event=event, mode="lambda")
        assert result["statusCode"] == 200
        result_body = json.loads(result["body"])
        assert result_body["args"] == {"val": 42}


class TestLambdaHandlerArgumentExtraction:
    """Lambda handler extracts arguments from parsed and raw formats."""

    def test_parsed_arguments(self):
        """Arguments from argument.parsed[0] are used."""
        mixin = ConcreteServerlessMixin(
            swaig_functions={"fn": lambda args, raw: {"got": args}}
        )
        body = {
            "function": "fn",
            "argument": {"parsed": [{"key": "value"}], "raw": "{}"},
        }
        event = {"rawPath": "/fn", "headers": {}, "body": json.dumps(body)}
        result = mixin.handle_serverless_request(event=event, mode="lambda")
        result_body = json.loads(result["body"])
        assert result_body["got"] == {"key": "value"}

    def test_raw_arguments_fallback(self):
        """When parsed is empty, argument.raw is used as fallback."""
        mixin = ConcreteServerlessMixin(
            swaig_functions={"fn": lambda args, raw: {"got": args}}
        )
        body = {
            "function": "fn",
            "argument": {"parsed": [], "raw": '{"from_raw": true}'},
        }
        event = {"rawPath": "/fn", "headers": {}, "body": json.dumps(body)}
        result = mixin.handle_serverless_request(event=event, mode="lambda")
        result_body = json.loads(result["body"])
        assert result_body["got"] == {"from_raw": True}

    def test_invalid_body_json_continues_with_empty_args(self):
        """If body is not valid JSON, parsing continues with empty args."""
        mixin = ConcreteServerlessMixin(
            swaig_functions={"fn": lambda args, raw: {"got": args}}
        )
        event = {"rawPath": "/fn", "headers": {}, "body": "NOT-JSON!!!"}
        result = mixin.handle_serverless_request(event=event, mode="lambda")
        assert result["statusCode"] == 200
        result_body = json.loads(result["body"])
        assert result_body["got"] == {}

    def test_body_as_dict(self):
        """Body that is already a dict (not a string) is handled."""
        mixin = ConcreteServerlessMixin(
            swaig_functions={"fn": lambda args, raw: {"got": args}}
        )
        body = _swaig_body("fn", {"x": 1})
        event = {"rawPath": "/fn", "headers": {}, "body": body}
        result = mixin.handle_serverless_request(event=event, mode="lambda")
        assert result["statusCode"] == 200


class TestLambdaHandlerAuth:
    """Lambda handler authentication checks."""

    def test_auth_failure_returns_challenge(self):
        """When auth fails, the challenge response is returned."""
        mixin = ConcreteServerlessMixin()
        mixin._check_lambda_auth = Mock(return_value=False)
        challenge = {"statusCode": 401, "body": "Unauthorized"}
        mixin._send_lambda_auth_challenge = Mock(return_value=challenge)
        event = {"rawPath": "/", "headers": {}}
        result = mixin.handle_serverless_request(event=event, mode="lambda")
        assert result["statusCode"] == 401
        mixin._check_lambda_auth.assert_called_once_with(event)


class TestLambdaHandlerErrors:
    """Lambda handler error handling."""

    def test_exception_returns_500(self):
        """Exceptions in lambda mode return 500 with error body."""
        mixin = ConcreteServerlessMixin()
        mixin._check_lambda_auth = Mock(side_effect=RuntimeError("boom"))
        event = {"rawPath": "/", "headers": {}}
        result = mixin.handle_serverless_request(event=event, mode="lambda")
        assert result["statusCode"] == 500
        body = json.loads(result["body"])
        assert "boom" in body["error"]

    def test_render_swml_error_returns_500(self):
        """Exception in _render_swml returns 500."""
        mixin = ConcreteServerlessMixin()
        mixin._render_swml = Mock(side_effect=ValueError("swml error"))
        event = {"rawPath": "/", "headers": {}}
        result = mixin.handle_serverless_request(event=event, mode="lambda")
        assert result["statusCode"] == 500


# ---------------------------------------------------------------------------
# Google Cloud Function handler tests
# ---------------------------------------------------------------------------

class TestGCFHandlerRootPath:
    """GCF handler returns SWML for root path requests."""

    def test_root_path_get_returns_swml(self):
        """GET to root path returns SWML."""
        mock_response_cls = Mock()
        mock_response_instance = Mock()
        mock_response_cls.return_value = mock_response_instance

        mixin = ConcreteServerlessMixin()
        request = _make_flask_request(path="/", method="GET")

        with patch.dict("sys.modules", {"flask": Mock(Response=mock_response_cls)}):
            result = mixin._handle_google_cloud_function_request(request)

        mock_response_cls.assert_called_once()
        call_kwargs = mock_response_cls.call_args[1]
        assert call_kwargs["status"] == 200
        assert call_kwargs["response"] == mixin._swml_response

    def test_root_path_post_no_body_returns_swml(self):
        """POST to root path with no useful body returns SWML."""
        mock_response_cls = Mock()
        mock_response_cls.return_value = Mock()

        mixin = ConcreteServerlessMixin()
        request = _make_flask_request(path="/", method="POST")

        with patch.dict("sys.modules", {"flask": Mock(Response=mock_response_cls)}):
            result = mixin._handle_google_cloud_function_request(request)

        call_kwargs = mock_response_cls.call_args[1]
        assert call_kwargs["status"] == 200


class TestGCFHandlerFunctionRouting:
    """GCF handler function routing."""

    def test_path_based_function_call(self):
        """POST to /say_hello routes to the named function."""
        mock_response_cls = Mock()
        mock_response_cls.return_value = Mock()

        mixin = ConcreteServerlessMixin(
            swaig_functions={"say_hello": lambda args, raw: {"response": "hi"}}
        )
        body = _swaig_body("say_hello", {"name": "Alice"})
        request = _make_flask_request(path="/say_hello", method="POST", json_data=body)

        with patch.dict("sys.modules", {"flask": Mock(Response=mock_response_cls)}):
            mixin._handle_google_cloud_function_request(request)

        call_kwargs = mock_response_cls.call_args[1]
        assert call_kwargs["status"] == 200
        response_body = json.loads(call_kwargs["response"])
        assert response_body["response"] == "hi"

    def test_swaig_endpoint_with_function_in_body(self):
        """POST to /swaig with function in body dispatches correctly."""
        mock_response_cls = Mock()
        mock_response_cls.return_value = Mock()

        mixin = ConcreteServerlessMixin(
            swaig_functions={"greet": lambda args, raw: {"response": "hello"}}
        )
        body = _swaig_body("greet", {"name": "Bob"})
        request = _make_flask_request(path="/swaig", method="POST", json_data=body)

        with patch.dict("sys.modules", {"flask": Mock(Response=mock_response_cls)}):
            mixin._handle_google_cloud_function_request(request)

        call_kwargs = mock_response_cls.call_args[1]
        response_body = json.loads(call_kwargs["response"])
        assert response_body["response"] == "hello"


class TestGCFHandlerBodyParsing:
    """GCF handler request body parsing."""

    def test_non_json_body_fallback(self):
        """When is_json is False, get_data(as_text=True) is used."""
        mock_response_cls = Mock()
        mock_response_cls.return_value = Mock()

        mixin = ConcreteServerlessMixin(
            swaig_functions={"fn": lambda args, raw: {"ok": True}}
        )
        body = _swaig_body("fn", {"x": 1})
        request = Mock()
        request.path = "/fn"
        request.method = "POST"
        request.url = "https://example.com/fn"
        request.is_json = False
        request.get_data = Mock(return_value=json.dumps(body))

        with patch.dict("sys.modules", {"flask": Mock(Response=mock_response_cls)}):
            mixin._handle_google_cloud_function_request(request)

        call_kwargs = mock_response_cls.call_args[1]
        assert call_kwargs["status"] == 200

    def test_malformed_body_continues(self):
        """Malformed POST body does not crash; continues with empty args."""
        mock_response_cls = Mock()
        mock_response_cls.return_value = Mock()

        mixin = ConcreteServerlessMixin()
        request = Mock()
        request.path = "/"
        request.method = "POST"
        request.url = "https://example.com/"
        request.is_json = True
        request.get_json = Mock(side_effect=Exception("bad json"))
        request.get_data = Mock(return_value="NOT-JSON")

        with patch.dict("sys.modules", {"flask": Mock(Response=mock_response_cls)}):
            mixin._handle_google_cloud_function_request(request)

        call_kwargs = mock_response_cls.call_args[1]
        assert call_kwargs["status"] == 200


class TestGCFHandlerURLBaseDetection:
    """GCF handler detects base URL from request."""

    def test_base_url_set_from_request(self):
        """Proxy URL base is derived from the request URL."""
        mock_response_cls = Mock()
        mock_response_cls.return_value = Mock()

        mixin = ConcreteServerlessMixin()
        assert mixin._proxy_url_base is None
        request = _make_flask_request(
            path="/",
            url="https://us-central1-myproject.cloudfunctions.net/agent"
        )

        with patch.dict("sys.modules", {"flask": Mock(Response=mock_response_cls)}):
            mixin._handle_google_cloud_function_request(request)

        assert mixin._proxy_url_base == "https://us-central1-myproject.cloudfunctions.net"

    def test_base_url_not_overridden_when_env_set(self):
        """When _proxy_url_base_from_env is True, URL is not overridden."""
        mock_response_cls = Mock()
        mock_response_cls.return_value = Mock()

        mixin = ConcreteServerlessMixin()
        mixin._proxy_url_base = "https://original.example.com"
        mixin._proxy_url_base_from_env = True
        request = _make_flask_request(
            path="/",
            url="https://different.example.com/agent"
        )

        with patch.dict("sys.modules", {"flask": Mock(Response=mock_response_cls)}):
            mixin._handle_google_cloud_function_request(request)

        assert mixin._proxy_url_base == "https://original.example.com"


class TestGCFHandlerAuth:
    """GCF handler authentication via handle_serverless_request dispatch."""

    def test_auth_failure_returns_challenge(self):
        """When auth fails, the challenge response is returned."""
        mixin = ConcreteServerlessMixin()
        mixin._check_google_cloud_function_auth = Mock(return_value=False)
        challenge = Mock(status_code=401)
        mixin._send_google_cloud_function_auth_challenge = Mock(return_value=challenge)
        request = _make_flask_request(path="/")
        result = mixin.handle_serverless_request(event=request, mode="google_cloud_function")
        assert result.status_code == 401


class TestGCFHandlerErrors:
    """GCF handler error handling."""

    def test_exception_returns_500(self):
        """Exceptions in GCF handler return 500 Flask response."""
        mock_response_cls = Mock()
        mock_response_cls.return_value = Mock()

        mixin = ConcreteServerlessMixin()
        mixin._render_swml = Mock(side_effect=RuntimeError("boom"))
        request = _make_flask_request(path="/")

        with patch.dict("sys.modules", {"flask": Mock(Response=mock_response_cls)}):
            mixin._handle_google_cloud_function_request(request)

        # The error handler creates a Response with status 500
        call_kwargs = mock_response_cls.call_args[1]
        assert call_kwargs["status"] == 500
        error_body = json.loads(call_kwargs["response"])
        assert "boom" in error_body["error"]


# ---------------------------------------------------------------------------
# Azure Function handler tests
# ---------------------------------------------------------------------------

class TestAzureHandlerRootPath:
    """Azure handler returns SWML for root path."""

    def test_root_path_returns_swml(self):
        """GET to root azure path returns SWML."""
        mock_http_response = Mock()
        mock_func = Mock()
        mock_func.HttpResponse = mock_http_response

        mixin = ConcreteServerlessMixin()
        req = _make_azure_request(
            url="https://myapp.azurewebsites.net/api/myagent",
            method="GET"
        )

        saved = {}
        for mod_name in ["azure", "azure.functions"]:
            saved[mod_name] = sys.modules.pop(mod_name, None)
        try:
            with patch.dict("sys.modules", {
                "azure": Mock(functions=mock_func),
                "azure.functions": mock_func,
            }):
                mixin._handle_azure_function_request(req)
        finally:
            for mod_name, original in saved.items():
                if original is not None:
                    sys.modules[mod_name] = original
                else:
                    sys.modules.pop(mod_name, None)

        mock_http_response.assert_called_once()
        call_kwargs = mock_http_response.call_args[1]
        assert call_kwargs["status_code"] == 200
        assert call_kwargs["body"] == mixin._swml_response


class TestAzureHandlerFunctionRouting:
    """Azure handler function routing."""

    def test_swaig_endpoint_with_function(self):
        """POST to /api/myagent/swaig with function name dispatches correctly."""
        mock_http_response = Mock()
        mock_func = Mock()
        mock_func.HttpResponse = mock_http_response

        mixin = ConcreteServerlessMixin(
            swaig_functions={"greet": lambda args, raw: {"response": "hi"}}
        )
        body = _swaig_body("greet", {"name": "Alice"})
        req = _make_azure_request(
            url="https://myapp.azurewebsites.net/api/myagent/swaig",
            method="POST",
            body=body,
        )

        saved = {}
        for mod_name in ["azure", "azure.functions"]:
            saved[mod_name] = sys.modules.pop(mod_name, None)
        try:
            with patch.dict("sys.modules", {
                "azure": Mock(functions=mock_func),
                "azure.functions": mock_func,
            }):
                mixin._handle_azure_function_request(req)
        finally:
            for mod_name, original in saved.items():
                if original is not None:
                    sys.modules[mod_name] = original
                else:
                    sys.modules.pop(mod_name, None)

        call_kwargs = mock_http_response.call_args[1]
        assert call_kwargs["status_code"] == 200
        response_body = json.loads(call_kwargs["body"])
        assert response_body["response"] == "hi"

    def test_path_based_function_routing(self):
        """POST to /api/myagent/say_hello routes by path."""
        mock_http_response = Mock()
        mock_func = Mock()
        mock_func.HttpResponse = mock_http_response

        mixin = ConcreteServerlessMixin(
            swaig_functions={"say_hello": lambda args, raw: {"response": "hello"}}
        )
        body = _swaig_body("say_hello", {"name": "Bob"})
        req = _make_azure_request(
            url="https://myapp.azurewebsites.net/api/myagent/say_hello",
            method="POST",
            body=body,
        )

        saved = {}
        for mod_name in ["azure", "azure.functions"]:
            saved[mod_name] = sys.modules.pop(mod_name, None)
        try:
            with patch.dict("sys.modules", {
                "azure": Mock(functions=mock_func),
                "azure.functions": mock_func,
            }):
                mixin._handle_azure_function_request(req)
        finally:
            for mod_name, original in saved.items():
                if original is not None:
                    sys.modules[mod_name] = original
                else:
                    sys.modules.pop(mod_name, None)

        call_kwargs = mock_http_response.call_args[1]
        response_body = json.loads(call_kwargs["body"])
        assert response_body["response"] == "hello"


class TestAzureHandlerURLParsing:
    """Azure handler URL parsing and base URL detection."""

    def test_base_url_set_from_request(self):
        """Proxy URL base includes /api/function_app_name."""
        mock_http_response = Mock()
        mock_func = Mock()
        mock_func.HttpResponse = mock_http_response

        mixin = ConcreteServerlessMixin()
        req = _make_azure_request(
            url="https://myapp.azurewebsites.net/api/myagent",
            method="GET",
        )

        saved = {}
        for mod_name in ["azure", "azure.functions"]:
            saved[mod_name] = sys.modules.pop(mod_name, None)
        try:
            with patch.dict("sys.modules", {
                "azure": Mock(functions=mock_func),
                "azure.functions": mock_func,
            }):
                mixin._handle_azure_function_request(req)
        finally:
            for mod_name, original in saved.items():
                if original is not None:
                    sys.modules[mod_name] = original
                else:
                    sys.modules.pop(mod_name, None)

        assert mixin._proxy_url_base == "https://myapp.azurewebsites.net/api/myagent"

    def test_url_without_api_prefix(self):
        """URL without /api/ sets base URL to scheme://netloc/api."""
        mock_http_response = Mock()
        mock_func = Mock()
        mock_func.HttpResponse = mock_http_response

        mixin = ConcreteServerlessMixin()
        req = _make_azure_request(
            url="https://myapp.azurewebsites.net/myagent",
            method="GET",
        )

        saved = {}
        for mod_name in ["azure", "azure.functions"]:
            saved[mod_name] = sys.modules.pop(mod_name, None)
        try:
            with patch.dict("sys.modules", {
                "azure": Mock(functions=mock_func),
                "azure.functions": mock_func,
            }):
                mixin._handle_azure_function_request(req)
        finally:
            for mod_name, original in saved.items():
                if original is not None:
                    sys.modules[mod_name] = original
                else:
                    sys.modules.pop(mod_name, None)

        assert mixin._proxy_url_base == "https://myapp.azurewebsites.net/api"

    def test_base_url_not_overridden_when_env_set(self):
        """When _proxy_url_base_from_env is True, URL is not overridden."""
        mock_http_response = Mock()
        mock_func = Mock()
        mock_func.HttpResponse = mock_http_response

        mixin = ConcreteServerlessMixin()
        mixin._proxy_url_base = "https://original.example.com"
        mixin._proxy_url_base_from_env = True
        req = _make_azure_request(
            url="https://myapp.azurewebsites.net/api/myagent",
            method="GET",
        )

        saved = {}
        for mod_name in ["azure", "azure.functions"]:
            saved[mod_name] = sys.modules.pop(mod_name, None)
        try:
            with patch.dict("sys.modules", {
                "azure": Mock(functions=mock_func),
                "azure.functions": mock_func,
            }):
                mixin._handle_azure_function_request(req)
        finally:
            for mod_name, original in saved.items():
                if original is not None:
                    sys.modules[mod_name] = original
                else:
                    sys.modules.pop(mod_name, None)

        assert mixin._proxy_url_base == "https://original.example.com"


class TestAzureHandlerAuth:
    """Azure handler authentication via handle_serverless_request dispatch."""

    def test_auth_failure_returns_challenge(self):
        """When auth fails, the challenge response is returned."""
        mixin = ConcreteServerlessMixin()
        mixin._check_azure_function_auth = Mock(return_value=False)
        challenge = Mock(status_code=401)
        mixin._send_azure_function_auth_challenge = Mock(return_value=challenge)
        req = _make_azure_request()
        result = mixin.handle_serverless_request(event=req, mode="azure_function")
        assert result.status_code == 401


class TestAzureHandlerErrors:
    """Azure handler error handling."""

    def test_exception_returns_500(self):
        """Exceptions in Azure handler return 500 HttpResponse."""
        mock_http_response = Mock()
        mock_func = Mock()
        mock_func.HttpResponse = mock_http_response

        mixin = ConcreteServerlessMixin()
        mixin._render_swml = Mock(side_effect=RuntimeError("azure boom"))
        req = _make_azure_request(
            url="https://myapp.azurewebsites.net/api/myagent",
            method="GET",
        )

        saved = {}
        for mod_name in ["azure", "azure.functions"]:
            saved[mod_name] = sys.modules.pop(mod_name, None)
        try:
            with patch.dict("sys.modules", {
                "azure": Mock(functions=mock_func),
                "azure.functions": mock_func,
            }):
                mixin._handle_azure_function_request(req)
        finally:
            for mod_name, original in saved.items():
                if original is not None:
                    sys.modules[mod_name] = original
                else:
                    sys.modules.pop(mod_name, None)

        call_kwargs = mock_http_response.call_args[1]
        assert call_kwargs["status_code"] == 500
        error_body = json.loads(call_kwargs["body"])
        assert "azure boom" in error_body["error"]

    def test_malformed_body_continues(self):
        """Malformed POST body does not crash; continues with empty args."""
        mock_http_response = Mock()
        mock_func = Mock()
        mock_func.HttpResponse = mock_http_response

        mixin = ConcreteServerlessMixin()
        req = Mock()
        req.url = "https://myapp.azurewebsites.net/api/myagent"
        req.method = "POST"
        req.get_body = Mock(return_value=b"NOT-JSON!!!")

        saved = {}
        for mod_name in ["azure", "azure.functions"]:
            saved[mod_name] = sys.modules.pop(mod_name, None)
        try:
            with patch.dict("sys.modules", {
                "azure": Mock(functions=mock_func),
                "azure.functions": mock_func,
            }):
                mixin._handle_azure_function_request(req)
        finally:
            for mod_name, original in saved.items():
                if original is not None:
                    sys.modules[mod_name] = original
                else:
                    sys.modules.pop(mod_name, None)

        # Should succeed with SWML since no function name extracted
        call_kwargs = mock_http_response.call_args[1]
        assert call_kwargs["status_code"] == 200


# ---------------------------------------------------------------------------
# _execute_swaig_function tests
# ---------------------------------------------------------------------------

class TestExecuteSwaigFunction:
    """Tests for _execute_swaig_function."""

    def test_function_not_found(self):
        """Unknown function name returns error dict."""
        mixin = ConcreteServerlessMixin()
        result = mixin._execute_swaig_function("nonexistent")
        assert "error" in result
        assert "nonexistent" in result["error"]

    def test_successful_dict_result(self):
        """Function returning a dict passes it through."""
        mixin = ConcreteServerlessMixin(
            swaig_functions={"fn": lambda args, raw: {"key": "val"}}
        )
        result = mixin._execute_swaig_function("fn", {"x": 1})
        assert result == {"key": "val"}

    def test_successful_swaig_function_result(self):
        """Function returning SwaigFunctionResult is converted to dict."""
        def handler(args, raw):
            return SwaigFunctionResult("Done")

        mixin = ConcreteServerlessMixin(swaig_functions={"fn": handler})
        result = mixin._execute_swaig_function("fn", {"x": 1})
        assert "response" in result
        assert result["response"] == "Done"

    def test_successful_string_result(self):
        """Function returning a string is wrapped in response dict."""
        mixin = ConcreteServerlessMixin(
            swaig_functions={"fn": lambda args, raw: "just a string"}
        )
        result = mixin._execute_swaig_function("fn")
        assert result == {"response": "just a string"}

    def test_none_args_default_to_empty_dict(self):
        """None args are replaced with empty dict."""
        received = {}

        def handler(args, raw):
            received["args"] = args
            return {"ok": True}

        mixin = ConcreteServerlessMixin(swaig_functions={"fn": handler})
        mixin._execute_swaig_function("fn", None)
        assert received["args"] == {}

    def test_none_raw_data_builds_default(self):
        """None raw_data is replaced with structured default."""
        received = {}

        def handler(args, raw):
            received["raw"] = raw
            return {"ok": True}

        mixin = ConcreteServerlessMixin(swaig_functions={"fn": handler})
        mixin._execute_swaig_function("fn", {"key": "val"}, call_id="c123", raw_data=None)
        raw = received["raw"]
        assert raw["function"] == "fn"
        assert raw["call_id"] == "c123"
        assert raw["argument"]["parsed"] == [{"key": "val"}]

    def test_exception_during_execution(self):
        """Exception in function returns error dict."""
        def handler(args, raw):
            raise ValueError("function error")

        mixin = ConcreteServerlessMixin(swaig_functions={"fn": handler})
        result = mixin._execute_swaig_function("fn")
        assert "error" in result
        assert "function error" in result["error"]
        assert result["function"] == "fn"


# ---------------------------------------------------------------------------
# Mode detection / dispatch
# ---------------------------------------------------------------------------

class TestModeDetection:
    """handle_serverless_request dispatches based on mode."""

    def test_lambda_mode_dispatch(self):
        """mode='lambda' dispatches to lambda handler."""
        mixin = ConcreteServerlessMixin()
        result = mixin.handle_serverless_request(event=None, mode="lambda")
        assert result["statusCode"] == 200

    def test_gcf_mode_dispatch(self):
        """mode='google_cloud_function' dispatches to GCF handler."""
        mock_response_cls = Mock()
        mock_response_cls.return_value = Mock()

        mixin = ConcreteServerlessMixin()
        request = _make_flask_request(path="/")

        with patch.dict("sys.modules", {"flask": Mock(Response=mock_response_cls)}):
            mixin.handle_serverless_request(event=request, mode="google_cloud_function")

        mock_response_cls.assert_called_once()

    def test_azure_mode_dispatch(self):
        """mode='azure_function' dispatches to Azure handler."""
        mock_http_response = Mock()
        mock_func = Mock()
        mock_func.HttpResponse = mock_http_response

        mixin = ConcreteServerlessMixin()
        req = _make_azure_request()

        saved = {}
        for mod_name in ["azure", "azure.functions"]:
            saved[mod_name] = sys.modules.pop(mod_name, None)
        try:
            with patch.dict("sys.modules", {
                "azure": Mock(functions=mock_func),
                "azure.functions": mock_func,
            }):
                mixin.handle_serverless_request(event=req, mode="azure_function")
        finally:
            for mod_name, original in saved.items():
                if original is not None:
                    sys.modules[mod_name] = original
                else:
                    sys.modules.pop(mod_name, None)

        mock_http_response.assert_called_once()

    def test_cgi_mode_dispatch_root(self):
        """mode='cgi' with empty PATH_INFO renders SWML."""
        mixin = ConcreteServerlessMixin()
        with patch.dict(os.environ, {"PATH_INFO": ""}, clear=False):
            result = mixin.handle_serverless_request(mode="cgi")
        assert result == mixin._swml_response

    def test_cgi_mode_auth_failure(self):
        """mode='cgi' with auth failure returns challenge."""
        mixin = ConcreteServerlessMixin()
        mixin._check_cgi_auth = Mock(return_value=False)
        mixin._send_cgi_auth_challenge = Mock(return_value="Status: 401\r\n\r\n")
        result = mixin.handle_serverless_request(mode="cgi")
        assert result == "Status: 401\r\n\r\n"

    def test_mode_auto_detection_lambda(self):
        """When mode is None, get_execution_mode() is called."""
        mixin = ConcreteServerlessMixin()
        with patch("signalwire_agents.core.mixins.serverless_mixin.get_execution_mode", return_value="lambda"):
            result = mixin.handle_serverless_request(event=None)
        assert result["statusCode"] == 200

    def test_non_lambda_exception_reraises(self):
        """Exceptions in non-lambda modes are re-raised (not wrapped in 500)."""
        mixin = ConcreteServerlessMixin()
        mixin._check_cgi_auth = Mock(side_effect=RuntimeError("cgi boom"))
        with pytest.raises(RuntimeError, match="cgi boom"):
            mixin.handle_serverless_request(mode="cgi")


# ---------------------------------------------------------------------------
# CGI mode body parsing
# ---------------------------------------------------------------------------

class TestCGIModeBodyParsing:
    """CGI mode parses POST data from stdin."""

    def test_cgi_function_call_with_post_body(self):
        """CGI mode parses function call from stdin POST data."""
        mixin = ConcreteServerlessMixin(
            swaig_functions={"hello": lambda args, raw: {"response": "world"}}
        )
        body = _swaig_body("hello", {"name": "Alice"}, call_id="c1")
        body_str = json.dumps(body)

        import io
        mock_stdin = io.StringIO(body_str)

        env = {
            "PATH_INFO": "/hello",
            "CONTENT_LENGTH": str(len(body_str)),
        }
        with patch.dict(os.environ, env, clear=False), \
             patch("sys.stdin", mock_stdin):
            result = mixin.handle_serverless_request(mode="cgi")

        assert result["response"] == "world"

    def test_cgi_function_call_with_raw_args(self):
        """CGI mode falls back to argument.raw when parsed is empty."""
        mixin = ConcreteServerlessMixin(
            swaig_functions={"hello": lambda args, raw: {"got": args}}
        )
        body = {
            "function": "hello",
            "argument": {"parsed": [], "raw": '{"from_raw": true}'},
        }
        body_str = json.dumps(body)

        import io
        mock_stdin = io.StringIO(body_str)

        env = {
            "PATH_INFO": "/hello",
            "CONTENT_LENGTH": str(len(body_str)),
        }
        with patch.dict(os.environ, env, clear=False), \
             patch("sys.stdin", mock_stdin):
            result = mixin.handle_serverless_request(mode="cgi")

        assert result["got"] == {"from_raw": True}

    def test_cgi_missing_content_length(self):
        """CGI mode with no CONTENT_LENGTH still works (no body parsed)."""
        mixin = ConcreteServerlessMixin(
            swaig_functions={"hello": lambda args, raw: {"response": "ok"}}
        )
        env = {"PATH_INFO": "/hello"}
        with patch.dict(os.environ, env, clear=False):
            result = mixin.handle_serverless_request(mode="cgi")

        # Function called with empty args
        assert result["response"] == "ok"
