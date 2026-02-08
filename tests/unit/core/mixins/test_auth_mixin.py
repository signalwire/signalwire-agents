"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for AuthMixin class
"""

import pytest
import json
import base64
import os
from unittest.mock import Mock, patch, MagicMock

from signalwire_agents.core.mixins.auth_mixin import AuthMixin


class ConcreteAuthMixin(AuthMixin):
    """Concrete implementation of AuthMixin for testing purposes."""

    def __init__(self, basic_auth=None):
        self._basic_auth = basic_auth or ("testuser", "testpass")


def _make_basic_auth_header(username, password):
    """Helper to create a Base64-encoded Basic auth header value."""
    credentials = f"{username}:{password}"
    encoded = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
    return f"Basic {encoded}"


# ---------------------------------------------------------------------------
# validate_basic_auth
# ---------------------------------------------------------------------------

class TestValidateBasicAuth:
    """Tests for validate_basic_auth method."""

    def test_valid_credentials(self):
        """Valid username and password returns True."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        assert mixin.validate_basic_auth("admin", "secret") is True

    def test_invalid_username(self):
        """Wrong username returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        assert mixin.validate_basic_auth("wrong", "secret") is False

    def test_invalid_password(self):
        """Wrong password returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        assert mixin.validate_basic_auth("admin", "wrong") is False

    def test_both_invalid(self):
        """Both wrong username and password returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        assert mixin.validate_basic_auth("wrong", "alsowrong") is False

    def test_empty_credentials(self):
        """Empty strings match if _basic_auth is also empty strings."""
        mixin = ConcreteAuthMixin(("", ""))
        assert mixin.validate_basic_auth("", "") is True

    def test_empty_credentials_mismatch(self):
        """Empty strings do not match non-empty credentials."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        assert mixin.validate_basic_auth("", "") is False

    def test_credentials_with_special_characters(self):
        """Credentials containing colons and special characters work."""
        mixin = ConcreteAuthMixin(("user@domain.com", "p@ss:w0rd!"))
        assert mixin.validate_basic_auth("user@domain.com", "p@ss:w0rd!") is True

    def test_delegation_can_be_overridden(self):
        """Subclass can override validate_basic_auth to change behavior."""

        class CustomAuth(AuthMixin):
            def __init__(self):
                self._basic_auth = ("unused", "unused")

            def validate_basic_auth(self, username, password):
                # Always accept a specific master key
                if password == "master-key":
                    return True
                return super().validate_basic_auth(username, password)

        custom = CustomAuth()
        assert custom.validate_basic_auth("anyone", "master-key") is True
        assert custom.validate_basic_auth("unused", "unused") is True
        assert custom.validate_basic_auth("anyone", "wrong") is False


# ---------------------------------------------------------------------------
# get_basic_auth_credentials
# ---------------------------------------------------------------------------

class TestGetBasicAuthCredentials:
    """Tests for get_basic_auth_credentials method."""

    def test_returns_tuple_without_source(self):
        """Without include_source, returns a 2-tuple."""
        mixin = ConcreteAuthMixin(("myuser", "mypass"))
        result = mixin.get_basic_auth_credentials()
        assert result == ("myuser", "mypass")

    def test_returns_tuple_with_source_provided(self):
        """With include_source and non-env, non-generated credentials, source is 'provided'."""
        mixin = ConcreteAuthMixin(("myuser", "mypass"))
        with patch.dict(os.environ, {}, clear=True):
            result = mixin.get_basic_auth_credentials(include_source=True)
        assert result == ("myuser", "mypass", "provided")

    def test_source_environment_when_matching_env_vars(self):
        """When credentials match environment variables, source is 'environment'."""
        mixin = ConcreteAuthMixin(("envuser", "envpass"))
        env = {"SWML_BASIC_AUTH_USER": "envuser", "SWML_BASIC_AUTH_PASSWORD": "envpass"}
        with patch.dict(os.environ, env, clear=True):
            result = mixin.get_basic_auth_credentials(include_source=True)
        assert result == ("envuser", "envpass", "environment")

    def test_source_not_environment_when_env_partially_matches(self):
        """If only one env var matches, source is not 'environment'."""
        mixin = ConcreteAuthMixin(("envuser", "differentpass"))
        env = {"SWML_BASIC_AUTH_USER": "envuser", "SWML_BASIC_AUTH_PASSWORD": "envpass"}
        with patch.dict(os.environ, env, clear=True):
            result = mixin.get_basic_auth_credentials(include_source=True)
        assert result[2] != "environment"

    def test_source_generated_when_looks_generated(self):
        """Credentials that look generated (user_ prefix, long password) get source 'generated'."""
        long_password = "a" * 25  # Longer than 20 characters
        mixin = ConcreteAuthMixin(("user_abc123", long_password))
        with patch.dict(os.environ, {}, clear=True):
            result = mixin.get_basic_auth_credentials(include_source=True)
        assert result == ("user_abc123", long_password, "generated")

    def test_source_not_generated_short_password(self):
        """user_ prefix but short password does not count as 'generated'."""
        mixin = ConcreteAuthMixin(("user_abc", "short"))
        with patch.dict(os.environ, {}, clear=True):
            result = mixin.get_basic_auth_credentials(include_source=True)
        assert result[2] == "provided"

    def test_source_not_generated_no_prefix(self):
        """Long password but no user_ prefix does not count as 'generated'."""
        mixin = ConcreteAuthMixin(("admin", "a" * 25))
        with patch.dict(os.environ, {}, clear=True):
            result = mixin.get_basic_auth_credentials(include_source=True)
        assert result[2] == "provided"

    def test_environment_takes_priority_over_generated(self):
        """Even if credentials look generated, environment match takes priority."""
        long_password = "a" * 25
        mixin = ConcreteAuthMixin(("user_abc123", long_password))
        env = {
            "SWML_BASIC_AUTH_USER": "user_abc123",
            "SWML_BASIC_AUTH_PASSWORD": long_password,
        }
        with patch.dict(os.environ, env, clear=True):
            result = mixin.get_basic_auth_credentials(include_source=True)
        assert result[2] == "environment"

    def test_include_source_false_explicit(self):
        """Explicitly passing include_source=False returns 2-tuple."""
        mixin = ConcreteAuthMixin(("u", "p"))
        result = mixin.get_basic_auth_credentials(include_source=False)
        assert len(result) == 2
        assert result == ("u", "p")


# ---------------------------------------------------------------------------
# _check_basic_auth (FastAPI request)
# ---------------------------------------------------------------------------

class TestCheckBasicAuth:
    """Tests for _check_basic_auth with FastAPI request objects."""

    def _make_request(self, auth_header=None):
        """Create a mock FastAPI request with optional Authorization header."""
        request = Mock()
        headers = {}
        if auth_header is not None:
            headers["Authorization"] = auth_header
        request.headers = Mock()
        request.headers.get = Mock(side_effect=lambda key, default=None: headers.get(key, default))
        return request

    def test_valid_credentials(self):
        """Valid Basic auth header returns True."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        header = _make_basic_auth_header("admin", "secret")
        request = self._make_request(header)
        assert mixin._check_basic_auth(request) is True

    def test_invalid_credentials(self):
        """Invalid Basic auth header returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        header = _make_basic_auth_header("admin", "wrong")
        request = self._make_request(header)
        assert mixin._check_basic_auth(request) is False

    def test_missing_auth_header(self):
        """Missing Authorization header returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        request = self._make_request(None)
        assert mixin._check_basic_auth(request) is False

    def test_non_basic_scheme(self):
        """Authorization header with non-Basic scheme returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        request = self._make_request("Bearer sometoken123")
        assert mixin._check_basic_auth(request) is False

    def test_malformed_base64(self):
        """Malformed base64 returns False instead of raising."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        request = self._make_request("Basic !!!not-base64!!!")
        assert mixin._check_basic_auth(request) is False

    def test_base64_without_colon(self):
        """Base64 content without colon separator returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        no_colon = base64.b64encode(b"nocolonhere").decode("utf-8")
        request = self._make_request(f"Basic {no_colon}")
        assert mixin._check_basic_auth(request) is False

    def test_password_with_colon(self):
        """Password containing colons is handled correctly (split on first colon only)."""
        mixin = ConcreteAuthMixin(("user", "pass:with:colons"))
        header = _make_basic_auth_header("user", "pass:with:colons")
        request = self._make_request(header)
        assert mixin._check_basic_auth(request) is True

    def test_delegates_to_validate_basic_auth(self):
        """_check_basic_auth delegates validation to validate_basic_auth."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        mixin.validate_basic_auth = Mock(return_value=True)
        header = _make_basic_auth_header("admin", "secret")
        request = self._make_request(header)

        result = mixin._check_basic_auth(request)

        assert result is True
        mixin.validate_basic_auth.assert_called_once_with("admin", "secret")

    def test_empty_auth_header(self):
        """Empty Authorization header string returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        request = self._make_request("")
        assert mixin._check_basic_auth(request) is False


# ---------------------------------------------------------------------------
# _check_cgi_auth
# ---------------------------------------------------------------------------

class TestCheckCgiAuth:
    """Tests for _check_cgi_auth method."""

    def test_valid_http_authorization(self):
        """Valid HTTP_AUTHORIZATION env var returns True."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        auth_header = _make_basic_auth_header("admin", "secret")
        env = {"HTTP_AUTHORIZATION": auth_header}
        with patch.dict(os.environ, env, clear=True):
            assert mixin._check_cgi_auth() is True

    def test_invalid_http_authorization(self):
        """Invalid HTTP_AUTHORIZATION credentials return False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        auth_header = _make_basic_auth_header("admin", "wrong")
        env = {"HTTP_AUTHORIZATION": auth_header}
        with patch.dict(os.environ, env, clear=True):
            assert mixin._check_cgi_auth() is False

    def test_remote_user_trusted(self):
        """REMOTE_USER without HTTP_AUTHORIZATION returns True (web server trust)."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        env = {"REMOTE_USER": "someuser"}
        with patch.dict(os.environ, env, clear=True):
            assert mixin._check_cgi_auth() is True

    def test_no_auth_env_vars(self):
        """No auth env vars returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        with patch.dict(os.environ, {}, clear=True):
            assert mixin._check_cgi_auth() is False

    def test_non_basic_scheme_in_env(self):
        """Non-Basic scheme in HTTP_AUTHORIZATION returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        env = {"HTTP_AUTHORIZATION": "Bearer token123"}
        with patch.dict(os.environ, env, clear=True):
            assert mixin._check_cgi_auth() is False

    def test_malformed_base64_in_env(self):
        """Malformed base64 in HTTP_AUTHORIZATION returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        env = {"HTTP_AUTHORIZATION": "Basic !!!invalid!!!"}
        with patch.dict(os.environ, env, clear=True):
            assert mixin._check_cgi_auth() is False

    def test_delegates_to_validate_basic_auth(self):
        """_check_cgi_auth delegates to validate_basic_auth."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        mixin.validate_basic_auth = Mock(return_value=True)
        auth_header = _make_basic_auth_header("admin", "secret")
        env = {"HTTP_AUTHORIZATION": auth_header}
        with patch.dict(os.environ, env, clear=True):
            result = mixin._check_cgi_auth()
        assert result is True
        mixin.validate_basic_auth.assert_called_once_with("admin", "secret")

    def test_http_authorization_takes_precedence_over_remote_user(self):
        """When both HTTP_AUTHORIZATION and REMOTE_USER are set, auth header is used."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        auth_header = _make_basic_auth_header("admin", "wrong")
        env = {"HTTP_AUTHORIZATION": auth_header, "REMOTE_USER": "someuser"}
        with patch.dict(os.environ, env, clear=True):
            # HTTP_AUTHORIZATION is checked first; wrong password -> False
            assert mixin._check_cgi_auth() is False


# ---------------------------------------------------------------------------
# _send_cgi_auth_challenge
# ---------------------------------------------------------------------------

class TestSendCgiAuthChallenge:
    """Tests for _send_cgi_auth_challenge method."""

    def test_returns_string_response(self):
        """Challenge response is a string."""
        mixin = ConcreteAuthMixin()
        result = mixin._send_cgi_auth_challenge()
        assert isinstance(result, str)

    def test_contains_401_status(self):
        """Response contains 401 Unauthorized status line."""
        mixin = ConcreteAuthMixin()
        result = mixin._send_cgi_auth_challenge()
        assert "401 Unauthorized" in result

    def test_contains_www_authenticate_header(self):
        """Response contains WWW-Authenticate header."""
        mixin = ConcreteAuthMixin()
        result = mixin._send_cgi_auth_challenge()
        assert "WWW-Authenticate: Basic" in result

    def test_contains_json_content_type(self):
        """Response contains JSON content type."""
        mixin = ConcreteAuthMixin()
        result = mixin._send_cgi_auth_challenge()
        assert "Content-Type: application/json" in result

    def test_contains_error_body(self):
        """Response body contains error JSON."""
        mixin = ConcreteAuthMixin()
        result = mixin._send_cgi_auth_challenge()
        # The body is after double CRLF
        body_start = result.index("\r\n\r\n") + 4
        body = result[body_start:]
        parsed = json.loads(body)
        assert parsed == {"error": "Unauthorized"}

    def test_uses_crlf_line_endings(self):
        """Response uses CRLF line endings for HTTP compliance."""
        mixin = ConcreteAuthMixin()
        result = mixin._send_cgi_auth_challenge()
        assert "\r\n" in result


# ---------------------------------------------------------------------------
# _check_lambda_auth
# ---------------------------------------------------------------------------

class TestCheckLambdaAuth:
    """Tests for _check_lambda_auth method."""

    def test_valid_credentials(self):
        """Valid auth header in event returns True."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        header = _make_basic_auth_header("admin", "secret")
        event = {"headers": {"Authorization": header}}
        assert mixin._check_lambda_auth(event) is True

    def test_invalid_credentials(self):
        """Invalid credentials in event returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        header = _make_basic_auth_header("admin", "wrong")
        event = {"headers": {"Authorization": header}}
        assert mixin._check_lambda_auth(event) is False

    def test_none_event(self):
        """None event returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        assert mixin._check_lambda_auth(None) is False

    def test_empty_event(self):
        """Empty event dict returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        assert mixin._check_lambda_auth({}) is False

    def test_no_headers_key(self):
        """Event without 'headers' key returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        assert mixin._check_lambda_auth({"body": "data"}) is False

    def test_case_insensitive_header(self):
        """Authorization header lookup is case-insensitive."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        header = _make_basic_auth_header("admin", "secret")
        event = {"headers": {"authorization": header}}
        assert mixin._check_lambda_auth(event) is True

    def test_mixed_case_header(self):
        """Mixed case 'AUTHORIZATION' header works."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        header = _make_basic_auth_header("admin", "secret")
        event = {"headers": {"AUTHORIZATION": header}}
        assert mixin._check_lambda_auth(event) is True

    def test_missing_auth_in_headers(self):
        """Headers dict without authorization key returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        event = {"headers": {"Content-Type": "application/json"}}
        assert mixin._check_lambda_auth(event) is False

    def test_non_basic_scheme(self):
        """Non-Basic scheme returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        event = {"headers": {"Authorization": "Bearer token123"}}
        assert mixin._check_lambda_auth(event) is False

    def test_malformed_base64(self):
        """Malformed base64 returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        event = {"headers": {"Authorization": "Basic !!!bad!!!"}}
        assert mixin._check_lambda_auth(event) is False

    def test_delegates_to_validate_basic_auth(self):
        """_check_lambda_auth delegates to validate_basic_auth."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        mixin.validate_basic_auth = Mock(return_value=True)
        header = _make_basic_auth_header("admin", "secret")
        event = {"headers": {"Authorization": header}}

        result = mixin._check_lambda_auth(event)

        assert result is True
        mixin.validate_basic_auth.assert_called_once_with("admin", "secret")


# ---------------------------------------------------------------------------
# _send_lambda_auth_challenge
# ---------------------------------------------------------------------------

class TestSendLambdaAuthChallenge:
    """Tests for _send_lambda_auth_challenge method."""

    def test_returns_dict(self):
        """Challenge returns a dictionary."""
        mixin = ConcreteAuthMixin()
        result = mixin._send_lambda_auth_challenge()
        assert isinstance(result, dict)

    def test_status_code_401(self):
        """Response has statusCode 401."""
        mixin = ConcreteAuthMixin()
        result = mixin._send_lambda_auth_challenge()
        assert result["statusCode"] == 401

    def test_www_authenticate_header(self):
        """Response includes WWW-Authenticate header."""
        mixin = ConcreteAuthMixin()
        result = mixin._send_lambda_auth_challenge()
        assert "WWW-Authenticate" in result["headers"]
        assert "Basic" in result["headers"]["WWW-Authenticate"]

    def test_content_type_json(self):
        """Response includes JSON content type header."""
        mixin = ConcreteAuthMixin()
        result = mixin._send_lambda_auth_challenge()
        assert result["headers"]["Content-Type"] == "application/json"

    def test_body_is_json_error(self):
        """Response body is JSON-encoded error."""
        mixin = ConcreteAuthMixin()
        result = mixin._send_lambda_auth_challenge()
        body = json.loads(result["body"])
        assert body == {"error": "Unauthorized"}


# ---------------------------------------------------------------------------
# _check_google_cloud_function_auth
# ---------------------------------------------------------------------------

class TestCheckGoogleCloudFunctionAuth:
    """Tests for _check_google_cloud_function_auth method."""

    def _make_flask_request(self, auth_header=None):
        """Create a mock Flask-like request."""
        request = Mock()
        headers = Mock()
        if auth_header is not None:
            headers.get = Mock(side_effect=lambda key, default=None: auth_header if key == "Authorization" else default)
        else:
            headers.get = Mock(return_value=None)
        request.headers = headers
        return request

    def test_valid_credentials(self):
        """Valid credentials return True."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        header = _make_basic_auth_header("admin", "secret")
        request = self._make_flask_request(header)
        assert mixin._check_google_cloud_function_auth(request) is True

    def test_invalid_credentials(self):
        """Invalid credentials return False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        header = _make_basic_auth_header("admin", "wrong")
        request = self._make_flask_request(header)
        assert mixin._check_google_cloud_function_auth(request) is False

    def test_no_headers_attribute(self):
        """Object without headers attribute returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        request = object()  # No headers attribute
        assert mixin._check_google_cloud_function_auth(request) is False

    def test_missing_auth_header(self):
        """Missing Authorization header returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        request = self._make_flask_request(None)
        assert mixin._check_google_cloud_function_auth(request) is False

    def test_non_basic_scheme(self):
        """Non-Basic scheme returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        request = self._make_flask_request("Bearer token123")
        assert mixin._check_google_cloud_function_auth(request) is False

    def test_malformed_base64(self):
        """Malformed base64 returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        request = self._make_flask_request("Basic !!!bad!!!")
        assert mixin._check_google_cloud_function_auth(request) is False

    def test_delegates_to_validate_basic_auth(self):
        """Delegates validation to validate_basic_auth."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        mixin.validate_basic_auth = Mock(return_value=True)
        header = _make_basic_auth_header("admin", "secret")
        request = self._make_flask_request(header)

        result = mixin._check_google_cloud_function_auth(request)

        assert result is True
        mixin.validate_basic_auth.assert_called_once_with("admin", "secret")

    def test_password_with_colon(self):
        """Password containing colons is handled correctly."""
        mixin = ConcreteAuthMixin(("user", "pass:with:colons"))
        header = _make_basic_auth_header("user", "pass:with:colons")
        request = self._make_flask_request(header)
        assert mixin._check_google_cloud_function_auth(request) is True


# ---------------------------------------------------------------------------
# _send_google_cloud_function_auth_challenge
# ---------------------------------------------------------------------------

class TestSendGoogleCloudFunctionAuthChallenge:
    """Tests for _send_google_cloud_function_auth_challenge method."""

    @patch("signalwire_agents.core.mixins.auth_mixin.AuthMixin._send_google_cloud_function_auth_challenge")
    def test_returns_response_object(self, mock_challenge):
        """Challenge returns a Flask Response-like object."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.headers = {
            "WWW-Authenticate": 'Basic realm="SignalWire Agent"',
            "Content-Type": "application/json",
        }
        mock_challenge.return_value = mock_response

        mixin = ConcreteAuthMixin()
        result = mock_challenge()
        assert result.status_code == 401
        assert "WWW-Authenticate" in result.headers

    def test_challenge_calls_flask_response(self):
        """The method constructs a Flask Response with correct parameters."""
        mock_response_cls = Mock()
        mock_response_instance = Mock()
        mock_response_cls.return_value = mock_response_instance

        mixin = ConcreteAuthMixin()
        with patch.dict("sys.modules", {"flask": Mock(Response=mock_response_cls)}):
            result = mixin._send_google_cloud_function_auth_challenge()

        mock_response_cls.assert_called_once()
        call_kwargs = mock_response_cls.call_args
        assert call_kwargs[1]["status"] == 401
        assert "WWW-Authenticate" in call_kwargs[1]["headers"]
        body = json.loads(call_kwargs[1]["response"])
        assert body == {"error": "Unauthorized"}


# ---------------------------------------------------------------------------
# _check_azure_function_auth
# ---------------------------------------------------------------------------

class TestCheckAzureFunctionAuth:
    """Tests for _check_azure_function_auth method."""

    def _make_azure_request(self, auth_header=None):
        """Create a mock Azure Functions request."""
        req = Mock()
        headers = Mock()
        if auth_header is not None:
            headers.get = Mock(side_effect=lambda key, default=None: auth_header if key == "Authorization" else default)
        else:
            headers.get = Mock(return_value=None)
        req.headers = headers
        return req

    def test_valid_credentials(self):
        """Valid credentials return True."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        header = _make_basic_auth_header("admin", "secret")
        req = self._make_azure_request(header)
        assert mixin._check_azure_function_auth(req) is True

    def test_invalid_credentials(self):
        """Invalid credentials return False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        header = _make_basic_auth_header("admin", "wrong")
        req = self._make_azure_request(header)
        assert mixin._check_azure_function_auth(req) is False

    def test_no_headers_attribute(self):
        """Object without headers attribute returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        req = object()  # No headers attribute
        assert mixin._check_azure_function_auth(req) is False

    def test_missing_auth_header(self):
        """Missing Authorization header returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        req = self._make_azure_request(None)
        assert mixin._check_azure_function_auth(req) is False

    def test_non_basic_scheme(self):
        """Non-Basic scheme returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        req = self._make_azure_request("Bearer token123")
        assert mixin._check_azure_function_auth(req) is False

    def test_malformed_base64(self):
        """Malformed base64 returns False."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        req = self._make_azure_request("Basic !!!bad!!!")
        assert mixin._check_azure_function_auth(req) is False

    def test_delegates_to_validate_basic_auth(self):
        """Delegates validation to validate_basic_auth."""
        mixin = ConcreteAuthMixin(("admin", "secret"))
        mixin.validate_basic_auth = Mock(return_value=True)
        header = _make_basic_auth_header("admin", "secret")
        req = self._make_azure_request(header)

        result = mixin._check_azure_function_auth(req)

        assert result is True
        mixin.validate_basic_auth.assert_called_once_with("admin", "secret")

    def test_password_with_colon(self):
        """Password containing colons is handled correctly."""
        mixin = ConcreteAuthMixin(("user", "pass:with:colons"))
        header = _make_basic_auth_header("user", "pass:with:colons")
        req = self._make_azure_request(header)
        assert mixin._check_azure_function_auth(req) is True


# ---------------------------------------------------------------------------
# _send_azure_function_auth_challenge
# ---------------------------------------------------------------------------

class TestSendAzureFunctionAuthChallenge:
    """Tests for _send_azure_function_auth_challenge method."""

    def test_challenge_calls_azure_http_response(self):
        """The method constructs an Azure HttpResponse with correct parameters."""
        mock_http_response_cls = Mock()
        mock_http_response_instance = Mock()
        mock_http_response_cls.return_value = mock_http_response_instance

        mock_func_module = MagicMock()
        mock_func_module.HttpResponse = mock_http_response_cls

        mock_azure = MagicMock()
        mock_azure.functions = mock_func_module

        mixin = ConcreteAuthMixin()
        # Remove any previously cached azure modules so the import inside
        # the method picks up our mocks.
        import sys
        modules_to_patch = {
            "azure": mock_azure,
            "azure.functions": mock_func_module,
        }
        saved = {}
        for mod_name in modules_to_patch:
            saved[mod_name] = sys.modules.pop(mod_name, None)
        try:
            with patch.dict("sys.modules", modules_to_patch):
                result = mixin._send_azure_function_auth_challenge()
        finally:
            for mod_name, original in saved.items():
                if original is not None:
                    sys.modules[mod_name] = original
                else:
                    sys.modules.pop(mod_name, None)

        mock_http_response_cls.assert_called_once()
        call_kwargs = mock_http_response_cls.call_args
        assert call_kwargs[1]["status_code"] == 401
        assert "WWW-Authenticate" in call_kwargs[1]["headers"]
        body = json.loads(call_kwargs[1]["body"])
        assert body == {"error": "Unauthorized"}


# ---------------------------------------------------------------------------
# Integration: validate_basic_auth delegation across all check methods
# ---------------------------------------------------------------------------

class TestValidateBasicAuthDelegationIntegration:
    """Verify that all auth check methods ultimately delegate to validate_basic_auth."""

    def test_check_basic_auth_uses_validate(self):
        """_check_basic_auth calls validate_basic_auth with decoded creds."""
        mixin = ConcreteAuthMixin(("u", "p"))
        mixin.validate_basic_auth = Mock(return_value=False)
        header = _make_basic_auth_header("u", "p")
        request = Mock()
        request.headers = Mock()
        request.headers.get = Mock(return_value=header)

        mixin._check_basic_auth(request)
        mixin.validate_basic_auth.assert_called_once_with("u", "p")

    def test_check_cgi_auth_uses_validate(self):
        """_check_cgi_auth calls validate_basic_auth with decoded creds."""
        mixin = ConcreteAuthMixin(("u", "p"))
        mixin.validate_basic_auth = Mock(return_value=False)
        header = _make_basic_auth_header("u", "p")
        with patch.dict(os.environ, {"HTTP_AUTHORIZATION": header}, clear=True):
            mixin._check_cgi_auth()
        mixin.validate_basic_auth.assert_called_once_with("u", "p")

    def test_check_lambda_auth_uses_validate(self):
        """_check_lambda_auth calls validate_basic_auth with decoded creds."""
        mixin = ConcreteAuthMixin(("u", "p"))
        mixin.validate_basic_auth = Mock(return_value=False)
        header = _make_basic_auth_header("u", "p")
        event = {"headers": {"Authorization": header}}

        mixin._check_lambda_auth(event)
        mixin.validate_basic_auth.assert_called_once_with("u", "p")

    def test_check_google_cloud_function_auth_uses_validate(self):
        """_check_google_cloud_function_auth calls validate_basic_auth."""
        mixin = ConcreteAuthMixin(("u", "p"))
        mixin.validate_basic_auth = Mock(return_value=False)
        header = _make_basic_auth_header("u", "p")
        request = Mock()
        request.headers = Mock()
        request.headers.get = Mock(return_value=header)

        mixin._check_google_cloud_function_auth(request)
        mixin.validate_basic_auth.assert_called_once_with("u", "p")

    def test_check_azure_function_auth_uses_validate(self):
        """_check_azure_function_auth calls validate_basic_auth."""
        mixin = ConcreteAuthMixin(("u", "p"))
        mixin.validate_basic_auth = Mock(return_value=False)
        header = _make_basic_auth_header("u", "p")
        req = Mock()
        req.headers = Mock()
        req.headers.get = Mock(return_value=header)

        mixin._check_azure_function_auth(req)
        mixin.validate_basic_auth.assert_called_once_with("u", "p")

    def test_overridden_validate_affects_all_checks(self):
        """Overriding validate_basic_auth changes behavior of all check methods."""

        class AlwaysAccept(AuthMixin):
            def __init__(self):
                self._basic_auth = ("admin", "secret")

            def validate_basic_auth(self, username, password):
                return True  # Accept anything

        mixin = AlwaysAccept()
        header = _make_basic_auth_header("wrong", "wrong")

        # FastAPI request
        request = Mock()
        request.headers = Mock()
        request.headers.get = Mock(return_value=header)
        assert mixin._check_basic_auth(request) is True

        # CGI
        with patch.dict(os.environ, {"HTTP_AUTHORIZATION": header}, clear=True):
            assert mixin._check_cgi_auth() is True

        # Lambda
        event = {"headers": {"Authorization": header}}
        assert mixin._check_lambda_auth(event) is True

        # Google Cloud Function
        gcf_request = Mock()
        gcf_request.headers = Mock()
        gcf_request.headers.get = Mock(return_value=header)
        assert mixin._check_google_cloud_function_auth(gcf_request) is True

        # Azure Function
        azure_req = Mock()
        azure_req.headers = Mock()
        azure_req.headers.get = Mock(return_value=header)
        assert mixin._check_azure_function_auth(azure_req) is True


# ---------------------------------------------------------------------------
# Integration: security_config credential flow
# ---------------------------------------------------------------------------

class TestSecurityConfigIntegration:
    """Test AuthMixin behavior when _basic_auth is set from SecurityConfig.get_basic_auth."""

    def test_credentials_from_security_config_provided(self):
        """When SecurityConfig provides explicit user/pass, validate_basic_auth works."""
        mixin = ConcreteAuthMixin(("configuser", "configpass"))
        assert mixin.validate_basic_auth("configuser", "configpass") is True
        assert mixin.validate_basic_auth("other", "other") is False

    def test_credentials_from_security_config_generated(self):
        """When SecurityConfig generates a long password, get_basic_auth_credentials detects it."""
        import secrets
        generated_pass = secrets.token_urlsafe(32)
        mixin = ConcreteAuthMixin(("user_abc123", generated_pass))
        with patch.dict(os.environ, {}, clear=True):
            _, _, source = mixin.get_basic_auth_credentials(include_source=True)
        assert source == "generated"

    def test_credentials_from_env_via_security_config(self):
        """When SecurityConfig loads from env, get_basic_auth_credentials detects environment source."""
        mixin = ConcreteAuthMixin(("envuser", "envpass"))
        env = {"SWML_BASIC_AUTH_USER": "envuser", "SWML_BASIC_AUTH_PASSWORD": "envpass"}
        with patch.dict(os.environ, env, clear=True):
            _, _, source = mixin.get_basic_auth_credentials(include_source=True)
        assert source == "environment"

    def test_end_to_end_auth_check_with_config_credentials(self):
        """Full flow: credentials set, request made, auth succeeds."""
        mixin = ConcreteAuthMixin(("myagent", "s3cret!"))
        header = _make_basic_auth_header("myagent", "s3cret!")
        request = Mock()
        request.headers = Mock()
        request.headers.get = Mock(return_value=header)
        assert mixin._check_basic_auth(request) is True

    def test_end_to_end_auth_check_rejects_wrong_creds(self):
        """Full flow: credentials set, wrong request made, auth fails."""
        mixin = ConcreteAuthMixin(("myagent", "s3cret!"))
        header = _make_basic_auth_header("myagent", "wrongpass")
        request = Mock()
        request.headers = Mock()
        request.headers.get = Mock(return_value=header)
        assert mixin._check_basic_auth(request) is False
