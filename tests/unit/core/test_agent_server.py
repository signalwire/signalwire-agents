"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Tests for AgentServer routing, especially catch-all handler behavior.

These tests verify that:
1. Custom routes registered after AgentServer creation work correctly
2. The catch-all handler doesn't overshadow custom routes
3. Health endpoints work correctly

Note: Static file and agent route tests are separate from the core routing tests
as they involve additional complexity (auth, startup events, etc.)
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from signalwire_agents import AgentBase, AgentServer


class SimpleTestAgent(AgentBase):
    """Simple agent for testing"""

    def __init__(self):
        super().__init__(
            name="test_agent",
            route="/test",
            use_pom=False
        )
        # Disable auth for testing
        self._auth_enabled = False


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
