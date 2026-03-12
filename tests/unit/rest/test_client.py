"""Tests for client.py — SignalWireClient instantiation and namespace wiring."""

import os
import pytest
from unittest.mock import patch

from signalwire_agents.rest.client import SignalWireClient
from signalwire_agents.rest.namespaces.fabric import FabricNamespace
from signalwire_agents.rest.namespaces.calling import CallingNamespace
from signalwire_agents.rest.namespaces.video import VideoNamespace
from signalwire_agents.rest.namespaces.compat import CompatNamespace


class TestSignalWireClient:
    def test_requires_credentials(self):
        with pytest.raises(ValueError, match="project, token, and host"):
            SignalWireClient()

    def test_env_var_fallback(self, mock_session):
        env = {
            "SIGNALWIRE_PROJECT_ID": "env-project",
            "SIGNALWIRE_API_TOKEN": "env-token",
            "SIGNALWIRE_SPACE": "env.signalwire.com",
        }
        with patch.dict(os.environ, env):
            c = SignalWireClient()
            assert c._project == "env-project"

    def test_explicit_overrides_env(self, mock_session):
        env = {
            "SIGNALWIRE_PROJECT_ID": "env-project",
            "SIGNALWIRE_API_TOKEN": "env-token",
            "SIGNALWIRE_SPACE": "env.signalwire.com",
        }
        with patch.dict(os.environ, env):
            c = SignalWireClient(project="explicit", token="tok", host="h.com")
            assert c._project == "explicit"

    def test_namespaces_exist(self, client):
        assert isinstance(client.fabric, FabricNamespace)
        assert isinstance(client.calling, CallingNamespace)
        assert isinstance(client.video, VideoNamespace)
        assert isinstance(client.compat, CompatNamespace)
        assert hasattr(client, "phone_numbers")
        assert hasattr(client, "addresses")
        assert hasattr(client, "queues")
        assert hasattr(client, "recordings")
        assert hasattr(client, "number_groups")
        assert hasattr(client, "verified_callers")
        assert hasattr(client, "sip_profile")
        assert hasattr(client, "lookup")
        assert hasattr(client, "short_codes")
        assert hasattr(client, "imported_numbers")
        assert hasattr(client, "mfa")
        assert hasattr(client, "registry")
        assert hasattr(client, "datasphere")
        assert hasattr(client, "logs")
        assert hasattr(client, "project")
        assert hasattr(client, "pubsub")
        assert hasattr(client, "chat")
