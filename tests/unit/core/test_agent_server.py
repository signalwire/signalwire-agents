"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Tests for AgentServer routing, initialization, agent management, SIP routing,
serverless mode handling, static file serving, and catch-all handler behavior.

These tests verify that:
1. Custom routes registered after AgentServer creation work correctly
2. The catch-all handler doesn't overshadow custom routes
3. Health endpoints work correctly
4. Agent registration, unregistration, and retrieval work
5. SIP routing setup and username mapping work
6. CGI and Lambda request handling work
7. Server run method dispatches correctly
8. Static file serving works

Note: Static file and agent route tests are separate from the core routing tests
as they involve additional complexity (auth, startup events, etc.)
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from io import StringIO

from fastapi.testclient import TestClient
from signalwire_agents import AgentBase, AgentServer


class SimpleTestAgent(AgentBase):
    """Simple agent for testing"""

    def __init__(self, name="test_agent", route="/test"):
        super().__init__(
            name=name,
            route=route,
            use_pom=False
        )
        # Disable auth for testing
        self._auth_enabled = False


class TestAgentServerInitialization:
    """Test AgentServer initialization and default state"""

    def test_default_initialization(self):
        """Test default host, port, and log_level"""
        server = AgentServer()
        assert server.host == "0.0.0.0"
        assert server.port == 3000
        assert server.log_level == "info"
        assert server.agents == {}
        assert server._sip_routing_enabled is False
        assert server._sip_route is None
        assert server._sip_username_mapping == {}

    def test_custom_initialization(self):
        """Test custom host, port, and log_level"""
        server = AgentServer(host="127.0.0.1", port=8080, log_level="DEBUG")
        assert server.host == "127.0.0.1"
        assert server.port == 8080
        assert server.log_level == "debug"  # Should be lowered

    def test_app_is_fastapi_instance(self):
        """Test that self.app is a FastAPI instance"""
        from fastapi import FastAPI
        server = AgentServer()
        assert isinstance(server.app, FastAPI)

    def test_health_endpoint_registered_on_init(self):
        """Test that health endpoints are available immediately after init"""
        server = AgentServer()
        client = TestClient(server.app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["agents"] == 0
        assert data["routes"] == []

    def test_ready_endpoint_registered_on_init(self):
        """Test that ready endpoint is available immediately after init"""
        server = AgentServer()
        client = TestClient(server.app)
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["agents"] == 0


class TestAgentRegistration:
    """Test agent registration, retrieval, and unregistration"""

    def test_register_agent_with_explicit_route(self):
        """Test registering an agent with an explicit route"""
        server = AgentServer()
        agent = SimpleTestAgent()
        server.register(agent, "/support")
        assert "/support" in server.agents
        assert server.agents["/support"] is agent

    def test_register_agent_uses_agent_default_route(self):
        """Test registering an agent without specifying a route uses agent's route"""
        server = AgentServer()
        agent = SimpleTestAgent(name="myagent", route="/myroute")
        server.register(agent)
        assert "/myroute" in server.agents

    def test_register_normalizes_route_adds_slash(self):
        """Test that routes without leading slash get normalized"""
        server = AgentServer()
        agent = SimpleTestAgent()
        server.register(agent, "noslash")
        assert "/noslash" in server.agents

    def test_register_normalizes_route_strips_trailing_slash(self):
        """Test that trailing slashes are stripped"""
        server = AgentServer()
        agent = SimpleTestAgent()
        server.register(agent, "/trailing/")
        assert "/trailing" in server.agents

    def test_register_duplicate_route_raises(self):
        """Test that registering two agents on same route raises ValueError"""
        server = AgentServer()
        agent1 = SimpleTestAgent(name="agent1")
        agent2 = SimpleTestAgent(name="agent2")
        server.register(agent1, "/shared")
        with pytest.raises(ValueError, match="Route '/shared' is already in use"):
            server.register(agent2, "/shared")

    def test_register_multiple_agents_different_routes(self):
        """Test registering multiple agents on different routes"""
        server = AgentServer()
        agent1 = SimpleTestAgent(name="support_agent")
        agent2 = SimpleTestAgent(name="sales_agent")
        server.register(agent1, "/support")
        server.register(agent2, "/sales")
        assert len(server.agents) == 2
        assert "/support" in server.agents
        assert "/sales" in server.agents

    def test_health_endpoint_reflects_registered_agents(self):
        """Test that health endpoint shows registered agent count and routes"""
        server = AgentServer()
        server.register(SimpleTestAgent(name="a1"), "/r1")
        server.register(SimpleTestAgent(name="a2"), "/r2")
        client = TestClient(server.app)
        data = client.get("/health").json()
        assert data["agents"] == 2
        assert "/r1" in data["routes"]
        assert "/r2" in data["routes"]


class TestAgentRetrieval:
    """Test get_agent and get_agents methods"""

    def test_get_agents_empty(self):
        """Test get_agents with no registered agents"""
        server = AgentServer()
        assert server.get_agents() == []

    def test_get_agents_returns_list_of_tuples(self):
        """Test get_agents returns route-agent tuples"""
        server = AgentServer()
        agent = SimpleTestAgent()
        server.register(agent, "/test")
        result = server.get_agents()
        assert len(result) == 1
        assert result[0][0] == "/test"
        assert result[0][1] is agent

    def test_get_agents_multiple(self):
        """Test get_agents with multiple agents"""
        server = AgentServer()
        a1 = SimpleTestAgent(name="a1")
        a2 = SimpleTestAgent(name="a2")
        server.register(a1, "/one")
        server.register(a2, "/two")
        result = server.get_agents()
        assert len(result) == 2
        routes = [r for r, _ in result]
        assert "/one" in routes
        assert "/two" in routes

    def test_get_agent_by_route(self):
        """Test get_agent returns the correct agent"""
        server = AgentServer()
        agent = SimpleTestAgent()
        server.register(agent, "/myagent")
        result = server.get_agent("/myagent")
        assert result is agent

    def test_get_agent_normalizes_route(self):
        """Test get_agent normalizes routes (adds slash, strips trailing)"""
        server = AgentServer()
        agent = SimpleTestAgent()
        server.register(agent, "/myagent")
        # Without leading slash
        assert server.get_agent("myagent") is agent
        # With trailing slash
        assert server.get_agent("/myagent/") is agent

    def test_get_agent_not_found(self):
        """Test get_agent returns None for unknown route"""
        server = AgentServer()
        assert server.get_agent("/nonexistent") is None


class TestAgentUnregistration:
    """Test unregister method"""

    def test_unregister_existing_agent(self):
        """Test unregistering an existing agent returns True"""
        server = AgentServer()
        server.register(SimpleTestAgent(), "/test")
        assert server.unregister("/test") is True
        assert "/test" not in server.agents

    def test_unregister_normalizes_route(self):
        """Test unregister normalizes route format"""
        server = AgentServer()
        server.register(SimpleTestAgent(), "/test")
        # Without leading slash
        assert server.unregister("test") is True
        assert len(server.agents) == 0

    def test_unregister_strips_trailing_slash(self):
        """Test unregister strips trailing slash"""
        server = AgentServer()
        server.register(SimpleTestAgent(), "/test")
        assert server.unregister("/test/") is True

    def test_unregister_nonexistent_returns_false(self):
        """Test unregistering a nonexistent route returns False"""
        server = AgentServer()
        assert server.unregister("/missing") is False


class TestSipRouting:
    """Test SIP routing setup and username mapping"""

    def test_setup_sip_routing_basic(self):
        """Test basic SIP routing setup"""
        server = AgentServer()
        server.setup_sip_routing(route="/sip")
        assert server._sip_routing_enabled is True
        assert server._sip_route == "/sip"
        assert server._sip_auto_map is True

    def test_setup_sip_routing_normalizes_route(self):
        """Test SIP routing normalizes the route"""
        server = AgentServer()
        server.setup_sip_routing(route="sip/")
        assert server._sip_route == "/sip"

    def test_setup_sip_routing_already_enabled_logs_warning(self):
        """Test that calling setup_sip_routing twice is a no-op"""
        server = AgentServer()
        server.setup_sip_routing()
        # Second call should just warn and return
        server.setup_sip_routing()
        assert server._sip_routing_enabled is True

    def test_setup_sip_routing_auto_map_existing_agents(self):
        """Test auto-mapping SIP usernames for existing agents"""
        server = AgentServer()
        agent = SimpleTestAgent(name="support_bot")
        server.register(agent, "/support")
        server.setup_sip_routing(auto_map=True)
        # Should have mapped agent name and route
        assert "support_bot" in server._sip_username_mapping or "supportbot" in server._sip_username_mapping
        assert "support" in server._sip_username_mapping

    def test_setup_sip_routing_no_auto_map(self):
        """Test SIP routing without auto-mapping"""
        server = AgentServer()
        agent = SimpleTestAgent(name="support")
        server.register(agent, "/support")
        server.setup_sip_routing(auto_map=False)
        assert server._sip_auto_map is False
        # Should not auto-map
        assert len(server._sip_username_mapping) == 0

    def test_register_sip_username(self):
        """Test manual SIP username registration"""
        server = AgentServer()
        server.register(SimpleTestAgent(), "/support")
        server.setup_sip_routing()
        server.register_sip_username("alice", "/support")
        assert server._sip_username_mapping["alice"] == "/support"

    def test_register_sip_username_case_insensitive(self):
        """Test SIP usernames are stored lowercase"""
        server = AgentServer()
        server.setup_sip_routing()
        server.register_sip_username("ALICE", "/support")
        assert "alice" in server._sip_username_mapping

    def test_register_sip_username_normalizes_route(self):
        """Test SIP username registration normalizes routes"""
        server = AgentServer()
        server.setup_sip_routing()
        server.register_sip_username("alice", "support/")
        assert server._sip_username_mapping["alice"] == "/support"

    def test_register_sip_username_without_sip_routing_enabled(self):
        """Test registering SIP username without enabling SIP routing first"""
        server = AgentServer()
        # Should just log a warning and return without adding
        server.register_sip_username("alice", "/support")
        assert len(server._sip_username_mapping) == 0

    def test_lookup_sip_route(self):
        """Test _lookup_sip_route returns correct route"""
        server = AgentServer()
        server._sip_username_mapping = {"alice": "/support", "bob": "/sales"}
        assert server._lookup_sip_route("alice") == "/support"
        assert server._lookup_sip_route("bob") == "/sales"
        assert server._lookup_sip_route("charlie") is None

    def test_lookup_sip_route_case_insensitive(self):
        """Test _lookup_sip_route is case insensitive"""
        server = AgentServer()
        server._sip_username_mapping = {"alice": "/support"}
        assert server._lookup_sip_route("ALICE") == "/support"

    def test_auto_map_agent_sip_usernames(self):
        """Test _auto_map_agent_sip_usernames creates proper mappings"""
        server = AgentServer()
        server._sip_routing_enabled = True
        agent = SimpleTestAgent(name="Sales Bot")
        server.register(agent, "/sales")
        # Now manually call auto-map
        server._auto_map_agent_sip_usernames(agent, "/sales")
        # "salesbot" (cleaned name) and "sales" (route part) should be mapped
        assert "salesbot" in server._sip_username_mapping or "sales_bot" in server._sip_username_mapping
        assert "sales" in server._sip_username_mapping

    def test_register_agent_with_sip_routing_enabled(self):
        """Test that registering an agent after SIP routing is set up auto-maps and registers callback"""
        server = AgentServer()
        server.setup_sip_routing(auto_map=True)
        agent = SimpleTestAgent(name="helpdesk")
        server.register(agent, "/helpdesk")
        assert "helpdesk" in server._sip_username_mapping

    def test_sip_routing_callback_with_valid_username(self):
        """Test the SIP routing callback resolves a known username"""
        server = AgentServer()
        agent = SimpleTestAgent(name="support")
        server.register(agent, "/support")
        server.setup_sip_routing(auto_map=True)

        # Directly invoke the callback
        mock_request = Mock()
        body = {"call": {"to": "sip:support@example.com"}}
        result = server._sip_routing_callback(mock_request, body)
        assert result == "/support"

    def test_sip_routing_callback_with_unknown_username(self):
        """Test the SIP routing callback returns None for unknown username"""
        server = AgentServer()
        agent = SimpleTestAgent(name="support")
        server.register(agent, "/support")
        server.setup_sip_routing(auto_map=True)

        mock_request = Mock()
        body = {"call": {"to": "sip:unknown@example.com"}}
        result = server._sip_routing_callback(mock_request, body)
        assert result is None

    def test_sip_routing_callback_no_sip_username(self):
        """Test the SIP routing callback with no extractable username"""
        server = AgentServer()
        server.setup_sip_routing()
        mock_request = Mock()
        body = {}
        result = server._sip_routing_callback(mock_request, body)
        assert result is None


class TestRunMethod:
    """Test the universal run method and execution mode detection"""

    @patch("signalwire_agents.agent_server.uvicorn")
    def test_run_server_mode(self, mock_uvicorn):
        """Test run() in server mode delegates to _run_server"""
        server = AgentServer()
        with patch("signalwire_agents.core.logging_config.get_execution_mode", return_value="server"):
            server.run()
        mock_uvicorn.run.assert_called_once()

    def test_run_cgi_mode(self):
        """Test run() in CGI mode delegates to _handle_cgi_request"""
        server = AgentServer()
        with patch("signalwire_agents.core.logging_config.get_execution_mode", return_value="cgi"):
            with patch.object(server, '_handle_cgi_request', return_value="cgi_response") as mock_cgi:
                result = server.run()
                mock_cgi.assert_called_once()
                assert result == "cgi_response"

    def test_run_lambda_mode(self):
        """Test run() in Lambda mode delegates to _handle_lambda_request"""
        server = AgentServer()
        event = {"path": "/test"}
        context = Mock()
        with patch("signalwire_agents.core.logging_config.get_execution_mode", return_value="lambda"):
            with patch.object(server, '_handle_lambda_request', return_value={"statusCode": 200}) as mock_lambda:
                result = server.run(event=event, context=context)
                mock_lambda.assert_called_once_with(event, context)
                assert result["statusCode"] == 200


class TestRunServer:
    """Test _run_server method"""

    @patch("signalwire_agents.agent_server.uvicorn")
    def test_run_server_default_host_port(self, mock_uvicorn):
        """Test _run_server uses default host and port"""
        server = AgentServer(host="0.0.0.0", port=3000)
        server._run_server()
        mock_uvicorn.run.assert_called_once_with(
            server.app,
            host="0.0.0.0",
            port=3000,
            log_level="info"
        )

    @patch("signalwire_agents.agent_server.uvicorn")
    def test_run_server_override_host_port(self, mock_uvicorn):
        """Test _run_server with overridden host and port"""
        server = AgentServer()
        server._run_server(host="127.0.0.1", port=9999)
        mock_uvicorn.run.assert_called_once_with(
            server.app,
            host="127.0.0.1",
            port=9999,
            log_level="info"
        )

    @patch("signalwire_agents.agent_server.uvicorn")
    def test_run_server_with_ssl(self, mock_uvicorn):
        """Test _run_server with SSL enabled via environment variables"""
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as cert_f, \
             tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as key_f:
            cert_path = cert_f.name
            key_path = key_f.name

        try:
            env = {
                "SWML_SSL_ENABLED": "true",
                "SWML_SSL_CERT_PATH": cert_path,
                "SWML_SSL_KEY_PATH": key_path,
                "SWML_DOMAIN": "example.com",
            }
            with patch.dict(os.environ, env, clear=False):
                server = AgentServer()
                server._run_server()
                mock_uvicorn.run.assert_called_once_with(
                    server.app,
                    host="0.0.0.0",
                    port=3000,
                    log_level="info",
                    ssl_certfile=cert_path,
                    ssl_keyfile=key_path,
                )
        finally:
            os.unlink(cert_path)
            os.unlink(key_path)

    @patch("signalwire_agents.agent_server.uvicorn")
    def test_run_server_ssl_disabled_bad_cert(self, mock_uvicorn):
        """Test _run_server falls back to non-SSL if cert not found"""
        env = {
            "SWML_SSL_ENABLED": "true",
            "SWML_SSL_CERT_PATH": "/nonexistent/cert.pem",
            "SWML_SSL_KEY_PATH": "/nonexistent/key.pem",
        }
        with patch.dict(os.environ, env, clear=False):
            server = AgentServer()
            server._run_server()
            # Should call without ssl params
            mock_uvicorn.run.assert_called_once_with(
                server.app,
                host="0.0.0.0",
                port=3000,
                log_level="info"
            )

    @patch("signalwire_agents.agent_server.uvicorn")
    def test_run_server_no_agents_warning(self, mock_uvicorn):
        """Test _run_server with no agents logs a warning"""
        server = AgentServer()
        # Should not raise; just warns
        server._run_server()
        mock_uvicorn.run.assert_called_once()

    @patch("signalwire_agents.agent_server.uvicorn")
    def test_run_server_ssl_missing_key(self, mock_uvicorn):
        """Test _run_server falls back when SSL key path is missing"""
        with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as cert_f:
            cert_path = cert_f.name

        try:
            env = {
                "SWML_SSL_ENABLED": "true",
                "SWML_SSL_CERT_PATH": cert_path,
                "SWML_SSL_KEY_PATH": "",
            }
            with patch.dict(os.environ, env, clear=False):
                server = AgentServer()
                server._run_server()
                # Should fall back to non-SSL
                mock_uvicorn.run.assert_called_once_with(
                    server.app,
                    host="0.0.0.0",
                    port=3000,
                    log_level="info"
                )
        finally:
            os.unlink(cert_path)


class TestHandleLambdaRequest:
    """Test _handle_lambda_request method"""

    def test_lambda_no_path_returns_404(self):
        """Test Lambda request with no path returns 404"""
        server = AgentServer()
        result = server._handle_lambda_request({"path": ""}, None)
        assert result["statusCode"] == 404
        body = json.loads(result["body"])
        assert "No agent specified" in body["error"]

    def test_lambda_none_event(self):
        """Test Lambda request with None event"""
        server = AgentServer()
        result = server._handle_lambda_request(None, None)
        assert result["statusCode"] == 404

    def test_lambda_matching_agent_returns_swml(self):
        """Test Lambda request that matches an agent returns SWML"""
        server = AgentServer()
        agent = SimpleTestAgent(name="myagent")
        agent._render_swml = Mock(return_value={"version": "1.0.0", "sections": {}})
        server.register(agent, "/myagent")
        event = {"path": "/myagent"}
        result = server._handle_lambda_request(event, None)
        assert result["statusCode"] == 200

    def test_lambda_matching_agent_render_error(self):
        """Test Lambda request when agent render fails returns 500"""
        server = AgentServer()
        agent = SimpleTestAgent(name="broken")
        agent._render_swml = Mock(side_effect=Exception("render failed"))
        server.register(agent, "/broken")
        event = {"path": "/broken"}
        result = server._handle_lambda_request(event, None)
        assert result["statusCode"] == 500
        body = json.loads(result["body"])
        assert "render failed" in body["error"]

    def test_lambda_no_matching_agent_returns_404(self):
        """Test Lambda request with no matching agent returns 404"""
        server = AgentServer()
        server.register(SimpleTestAgent(), "/test")
        event = {"path": "/nonexistent"}
        result = server._handle_lambda_request(event, None)
        assert result["statusCode"] == 404
        body = json.loads(result["body"])
        assert body["error"] == "Not Found"

    def test_lambda_path_from_path_parameters(self):
        """Test Lambda extracts path from pathParameters.proxy"""
        server = AgentServer()
        agent = SimpleTestAgent(name="myagent")
        agent._render_swml = Mock(return_value={"version": "1.0"})
        server.register(agent, "/myagent")
        event = {"pathParameters": {"proxy": "myagent"}}
        result = server._handle_lambda_request(event, None)
        assert result["statusCode"] == 200

    def test_lambda_swaig_subpath(self):
        """Test Lambda request to swaig subpath"""
        server = AgentServer()
        agent = SimpleTestAgent(name="myagent")
        agent._execute_swaig_function = Mock(return_value={"response": "ok"})
        server.register(agent, "/myagent")
        event = {"path": "/myagent/swaig", "body": json.dumps({"function": "test"})}
        result = server._handle_lambda_request(event, None)
        assert result["statusCode"] == 200

    def test_lambda_swaig_function_subpath(self):
        """Test Lambda request to swaig/<function_name> subpath"""
        server = AgentServer()
        agent = SimpleTestAgent(name="myagent")
        agent._execute_swaig_function = Mock(return_value={"response": "ok"})
        server.register(agent, "/myagent")
        event = {"path": "/myagent/swaig/my_func", "body": json.dumps({})}
        result = server._handle_lambda_request(event, None)
        assert result["statusCode"] == 200
        agent._execute_swaig_function.assert_called_once_with("my_func", {}, None, None)

    def test_lambda_swaig_exception(self):
        """Test Lambda request to swaig that raises exception returns 500"""
        server = AgentServer()
        agent = SimpleTestAgent(name="myagent")
        agent._execute_swaig_function = Mock(side_effect=Exception("swaig error"))
        server.register(agent, "/myagent")
        event = {"path": "/myagent/swaig", "body": json.dumps({})}
        result = server._handle_lambda_request(event, None)
        assert result["statusCode"] == 500

    def test_lambda_swaig_invalid_body(self):
        """Test Lambda request with invalid JSON body"""
        server = AgentServer()
        agent = SimpleTestAgent(name="myagent")
        agent._execute_swaig_function = Mock(return_value={"response": "ok"})
        server.register(agent, "/myagent")
        event = {"path": "/myagent/swaig", "body": "not valid json"}
        result = server._handle_lambda_request(event, None)
        assert result["statusCode"] == 200


class TestHandleCgiRequest:
    """Test _handle_cgi_request method"""

    def test_cgi_no_path_returns_404(self):
        """Test CGI request with no PATH_INFO returns 404"""
        server = AgentServer()
        with patch.dict(os.environ, {"PATH_INFO": ""}, clear=False):
            with patch("sys.stdout", new_callable=StringIO):
                result = server._handle_cgi_request()
                assert "404 Not Found" in result

    def test_cgi_matching_agent_returns_swml(self):
        """Test CGI request that matches an agent returns SWML"""
        server = AgentServer()
        agent = SimpleTestAgent(name="myagent")
        agent._render_swml = Mock(return_value={"version": "1.0.0"})
        server.register(agent, "/myagent")
        with patch.dict(os.environ, {"PATH_INFO": "/myagent"}, clear=False):
            with patch("sys.stdout", new_callable=StringIO):
                result = server._handle_cgi_request()
                assert "200 OK" in result

    def test_cgi_matching_agent_render_error(self):
        """Test CGI request when agent render fails returns 500"""
        server = AgentServer()
        agent = SimpleTestAgent(name="broken")
        agent._render_swml = Mock(side_effect=Exception("render failed"))
        server.register(agent, "/broken")
        with patch.dict(os.environ, {"PATH_INFO": "/broken"}, clear=False):
            with patch("sys.stdout", new_callable=StringIO):
                result = server._handle_cgi_request()
                assert "500 Internal Server Error" in result

    def test_cgi_no_matching_agent_returns_404(self):
        """Test CGI request with no matching agent returns 404"""
        server = AgentServer()
        server.register(SimpleTestAgent(), "/test")
        with patch.dict(os.environ, {"PATH_INFO": "/nonexistent"}, clear=False):
            with patch("sys.stdout", new_callable=StringIO):
                result = server._handle_cgi_request()
                assert "404 Not Found" in result

    def test_cgi_swaig_subpath(self):
        """Test CGI request to swaig subpath with no body"""
        server = AgentServer()
        agent = SimpleTestAgent(name="myagent")
        agent._execute_swaig_function = Mock(return_value={"response": "ok"})
        server.register(agent, "/myagent")
        env = {"PATH_INFO": "/myagent/swaig"}
        # Remove CONTENT_LENGTH to avoid stdin reading
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("CONTENT_LENGTH", None)
            with patch("sys.stdout", new_callable=StringIO):
                result = server._handle_cgi_request()
                assert "200 OK" in result

    def test_cgi_swaig_function_subpath(self):
        """Test CGI request to swaig/<function_name> subpath with no body"""
        server = AgentServer()
        agent = SimpleTestAgent(name="myagent")
        agent._execute_swaig_function = Mock(return_value={"response": "ok"})
        server.register(agent, "/myagent")
        env = {"PATH_INFO": "/myagent/swaig/my_func"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("CONTENT_LENGTH", None)
            with patch("sys.stdout", new_callable=StringIO):
                result = server._handle_cgi_request()
                assert "200 OK" in result

    def test_cgi_swaig_exception(self):
        """Test CGI request when SWAIG function raises exception"""
        server = AgentServer()
        agent = SimpleTestAgent(name="myagent")
        agent._execute_swaig_function = Mock(side_effect=Exception("swaig error"))
        server.register(agent, "/myagent")
        env = {"PATH_INFO": "/myagent/swaig", "CONTENT_LENGTH": "0"}
        with patch.dict(os.environ, env, clear=False):
            with patch("sys.stdout", new_callable=StringIO):
                result = server._handle_cgi_request()
                assert "500 Internal Server Error" in result

    def test_cgi_swaig_function_exception(self):
        """Test CGI request to swaig/<func> that raises exception"""
        server = AgentServer()
        agent = SimpleTestAgent(name="myagent")
        agent._execute_swaig_function = Mock(side_effect=Exception("func error"))
        server.register(agent, "/myagent")
        env = {"PATH_INFO": "/myagent/swaig/broken_func", "CONTENT_LENGTH": "0"}
        with patch.dict(os.environ, env, clear=False):
            with patch("sys.stdout", new_callable=StringIO):
                result = server._handle_cgi_request()
                assert "500 Internal Server Error" in result


class TestFormatCgiResponse:
    """Test _format_cgi_response method"""

    def test_format_cgi_response_dict(self):
        """Test formatting a dict response"""
        server = AgentServer()
        with patch("sys.stdout", new_callable=StringIO):
            result = server._format_cgi_response({"key": "value"})
        assert "Status: 200 OK" in result
        assert "Content-Type: application/json" in result
        assert '"key": "value"' in result

    def test_format_cgi_response_string(self):
        """Test formatting a string response"""
        server = AgentServer()
        with patch("sys.stdout", new_callable=StringIO):
            result = server._format_cgi_response("plain text")
        assert "plain text" in result

    def test_format_cgi_response_custom_status(self):
        """Test formatting with custom status"""
        server = AgentServer()
        with patch("sys.stdout", new_callable=StringIO):
            result = server._format_cgi_response({"error": "nope"}, status="404 Not Found")
        assert "Status: 404 Not Found" in result

    def test_format_cgi_response_custom_content_type(self):
        """Test formatting with custom content type"""
        server = AgentServer()
        with patch("sys.stdout", new_callable=StringIO):
            result = server._format_cgi_response({"data": 1}, content_type="text/html")
        assert "Content-Type: text/html" in result


class TestGlobalRoutingCallback:
    """Test register_global_routing_callback method"""

    def test_register_global_routing_callback(self):
        """Test registering a global routing callback on all agents"""
        server = AgentServer()
        agent1 = SimpleTestAgent(name="a1")
        agent2 = SimpleTestAgent(name="a2")
        server.register(agent1, "/a1")
        server.register(agent2, "/a2")

        callback = Mock()
        with patch.object(agent1, 'register_routing_callback') as mock1, \
             patch.object(agent2, 'register_routing_callback') as mock2:
            server.register_global_routing_callback(callback, path="/sip")
            mock1.assert_called_once_with(callback, path="/sip")
            mock2.assert_called_once_with(callback, path="/sip")

    def test_register_global_routing_callback_normalizes_path(self):
        """Test that path is normalized"""
        server = AgentServer()
        agent = SimpleTestAgent(name="a1")
        server.register(agent, "/a1")

        callback = Mock()
        with patch.object(agent, 'register_routing_callback') as mock_reg:
            server.register_global_routing_callback(callback, path="sip/")
            mock_reg.assert_called_once_with(callback, path="/sip")


class TestServeStaticFiles:
    """Test serve_static_files and _serve_static_file methods"""

    def test_serve_static_files_valid_directory(self):
        """Test serve_static_files with valid directory"""
        server = AgentServer()
        with tempfile.TemporaryDirectory() as tmpdir:
            server.serve_static_files(tmpdir)
            assert hasattr(server, '_static_directories')
            assert "" in server._static_directories or "/" in server._static_directories

    def test_serve_static_files_nonexistent_directory(self):
        """Test serve_static_files with nonexistent directory"""
        server = AgentServer()
        with pytest.raises(ValueError, match="does not exist"):
            server.serve_static_files("/nonexistent/directory")

    def test_serve_static_files_file_not_directory(self):
        """Test serve_static_files with a file path instead of directory"""
        server = AgentServer()
        with tempfile.NamedTemporaryFile() as tmpfile:
            with pytest.raises(ValueError, match="not a directory"):
                server.serve_static_files(tmpfile.name)

    def test_serve_static_files_custom_route(self):
        """Test serve_static_files with custom route prefix"""
        server = AgentServer()
        with tempfile.TemporaryDirectory() as tmpdir:
            server.serve_static_files(tmpdir, route="/static")
            assert "/static" in server._static_directories

    def test_serve_static_file_no_directories_configured(self):
        """Test _serve_static_file returns None when no directories configured"""
        server = AgentServer()
        result = server._serve_static_file("test.html")
        assert result is None

    def test_serve_static_file_existing_file(self):
        """Test _serve_static_file returns FileResponse for existing file"""
        server = AgentServer()
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved_dir = Path(tmpdir).resolve()
            # Create a test file
            test_file = resolved_dir / "hello.txt"
            test_file.write_text("Hello World")

            server._static_directories = {"": resolved_dir}
            result = server._serve_static_file("hello.txt", route="")
            assert result is not None

    def test_serve_static_file_nonexistent_file(self):
        """Test _serve_static_file returns None for nonexistent file"""
        server = AgentServer()
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved_dir = Path(tmpdir).resolve()
            server._static_directories = {"": resolved_dir}
            result = server._serve_static_file("nonexistent.txt", route="")
            assert result is None

    def test_serve_static_file_empty_path_serves_index(self):
        """Test _serve_static_file defaults to index.html for empty path"""
        server = AgentServer()
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved_dir = Path(tmpdir).resolve()
            index_file = resolved_dir / "index.html"
            index_file.write_text("<html></html>")

            server._static_directories = {"": resolved_dir}
            result = server._serve_static_file("", route="")
            assert result is not None

    def test_serve_static_file_directory_with_index(self):
        """Test _serve_static_file serves index.html from a directory"""
        server = AgentServer()
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved_dir = Path(tmpdir).resolve()
            subdir = resolved_dir / "subdir"
            subdir.mkdir()
            index_file = subdir / "index.html"
            index_file.write_text("<html></html>")

            server._static_directories = {"": resolved_dir}
            result = server._serve_static_file("subdir", route="")
            assert result is not None

    def test_serve_static_file_route_not_found(self):
        """Test _serve_static_file returns None for unknown route"""
        server = AgentServer()
        server._static_directories = {"/assets": Path("/tmp")}
        result = server._serve_static_file("test.txt", route="/other")
        assert result is None


class TestAgentServerRouting:
    """Test suite for AgentServer routing behavior"""

    def test_custom_route_not_overshadowed_by_catch_all(self):
        """
        Test that custom routes registered after AgentServer creation are not
        overshadowed by the catch-all handler.

        This was a bug where registering catch-all in __init__ would cause
        routes like /get_token to return 404.
        """
        server = AgentServer()
        server.register(SimpleTestAgent(), "/agent")

        # Add a custom route AFTER server creation (like santa's /get_token)
        @server.app.get('/get_token')
        def get_token():
            return {"token": "test-token-123", "success": True}

        # Add another custom route
        @server.app.get('/health_custom')
        def health_custom():
            return {"status": "healthy"}

        client = TestClient(server.app)

        # Test custom route works
        response = client.get('/get_token')
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        assert data["token"] == "test-token-123"
        assert data["success"] is True

        # Test another custom route
        response = client.get('/health_custom')
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_health_endpoints_work(self):
        """Test that built-in health endpoints work"""
        server = AgentServer()
        server.register(SimpleTestAgent(), "/agent")

        client = TestClient(server.app)

        # Health endpoint should work
        response = client.get('/health')
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

        # Ready endpoint should work
        response = client.get('/ready')
        assert response.status_code == 200
        assert response.json()["status"] == "ready"


    def test_multiple_custom_routes(self):
        """Test multiple custom routes all work correctly"""
        server = AgentServer()
        server.register(SimpleTestAgent(), "/agent")

        # Add multiple custom routes
        @server.app.get('/route1')
        def route1():
            return {"route": 1}

        @server.app.get('/route2')
        def route2():
            return {"route": 2}

        @server.app.post('/route3')
        def route3():
            return {"route": 3}

        @server.app.get('/nested/deep/route')
        def nested():
            return {"route": "nested"}

        client = TestClient(server.app)

        # All routes should work
        assert client.get('/route1').json()["route"] == 1
        assert client.get('/route2').json()["route"] == 2
        assert client.post('/route3').json()["route"] == 3
        assert client.get('/nested/deep/route').json()["route"] == "nested"

    def test_nonexistent_route_returns_404(self):
        """Test that truly nonexistent routes return 404"""
        server = AgentServer()
        server.register(SimpleTestAgent(), "/agent")

        client = TestClient(server.app)

        # Nonexistent route should 404
        response = client.get('/nonexistent/path')
        assert response.status_code == 404

    def test_post_custom_routes_work(self):
        """Test that POST custom routes work correctly"""
        server = AgentServer()
        server.register(SimpleTestAgent(), "/agent")

        @server.app.post('/webhook')
        def webhook():
            return {"received": True}

        client = TestClient(server.app)

        response = client.post('/webhook', json={"data": "test"})
        assert response.status_code == 200
        assert response.json()["received"] is True


class TestAgentServerGunicornCompatibility:
    """
    Tests specifically for gunicorn compatibility.

    These tests verify behavior when using server.app directly (as gunicorn does)
    instead of server.run().
    """

    def test_app_property_works_for_gunicorn(self):
        """Test that server.app can be used directly like gunicorn does"""
        server = AgentServer()
        server.register(SimpleTestAgent(), "/agent")

        # Gunicorn uses server.app directly
        app = server.app

        # Add routes to the app (like santa does)
        @app.get('/get_token')
        def get_token():
            return {"token": "gunicorn-test"}

        client = TestClient(app)

        # Custom route should work
        response = client.get('/get_token')
        assert response.status_code == 200
        assert response.json()["token"] == "gunicorn-test"

        # Health should work
        response = client.get('/health')
        assert response.status_code == 200

    def test_custom_routes_work_with_gunicorn_pattern(self):
        """Test custom routes work when using server.app (gunicorn pattern)"""
        server = AgentServer()
        server.register(SimpleTestAgent(), "/agent")

        # Add multiple custom endpoints like a real app would
        @server.app.get('/get_credentials')
        def get_credentials():
            return {"user": "test", "pass": "secret"}

        @server.app.get('/get_resource_info')
        def get_resource_info():
            return {"resource_id": "123"}

        @server.app.post('/webhook')
        def webhook():
            return {"status": "received"}

        # Use server.app directly like gunicorn
        client = TestClient(server.app)

        # All custom routes should work
        assert client.get('/get_credentials').status_code == 200
        assert client.get('/get_resource_info').status_code == 200
        assert client.post('/webhook').status_code == 200

        # Health should still work
        assert client.get('/health').status_code == 200
