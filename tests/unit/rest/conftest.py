"""Mock infrastructure and fixtures for REST client tests."""

from unittest.mock import MagicMock, patch
import pytest

from signalwire_agents.rest._base import HttpClient
from signalwire_agents.rest.client import SignalWireClient


class MockResponse:
    """Simulates a requests.Response."""

    def __init__(self, status_code=200, json_data=None, content=b"ok"):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.content = content
        self.ok = 200 <= status_code < 300
        self.text = str(json_data)

    def json(self):
        return self._json


@pytest.fixture
def mock_session():
    """Patches requests.Session and returns the mock session instance."""
    with patch("signalwire_agents.rest._base.requests.Session") as MockSession:
        session = MagicMock()
        MockSession.return_value = session
        # Default: return 200 with empty dict
        session.request.return_value = MockResponse(200, {})
        yield session


@pytest.fixture
def http(mock_session):
    """An HttpClient backed by a mock session."""
    return HttpClient("test-project-id", "test-token", "test.signalwire.com")


@pytest.fixture
def client(mock_session):
    """A SignalWireClient backed by a mock session."""
    return SignalWireClient(
        project="test-project-id",
        token="test-token",
        host="test.signalwire.com",
    )
