"""Mock infrastructure and fixtures for RELAY client tests."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Optional
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
import websockets.exceptions

from signalwire_agents.relay.client import RelayClient, _active_clients
from signalwire_agents.relay.call import Call
from signalwire_agents.relay.constants import METHOD_SIGNALWIRE_CONNECT


# ---------------------------------------------------------------------------
# Mock WebSocket
# ---------------------------------------------------------------------------

class MockWebSocket:
    """Simulates websockets.WebSocketClientProtocol for unit tests.

    - recv is backed by an asyncio.Queue so tests can inject messages.
    - send captures outgoing messages (parsed as dicts) for assertions.
    """

    def __init__(self) -> None:
        self._recv_queue: asyncio.Queue[str | Exception] = asyncio.Queue()
        self.sent_messages: list[dict[str, Any]] = []
        self.closed = False
        self._close_event = asyncio.Event()

    # --- sending (client → server) ---

    async def send(self, raw: str) -> None:
        if self.closed:
            raise websockets.exceptions.ConnectionClosed(None, None)
        self.sent_messages.append(json.loads(raw))

    # --- receiving (server → client) ---

    async def recv(self) -> str:
        item = await self._recv_queue.get()
        if isinstance(item, Exception):
            raise item
        return item

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        try:
            return await self.recv()
        except (websockets.exceptions.ConnectionClosed, StopAsyncIteration):
            raise StopAsyncIteration

    # --- test helpers ---

    def feed_message(self, msg: dict[str, Any]) -> None:
        """Inject a server message into the recv queue."""
        self._recv_queue.put_nowait(json.dumps(msg))

    def feed_close(self) -> None:
        """Inject a ConnectionClosed exception to end the recv loop."""
        self._recv_queue.put_nowait(
            websockets.exceptions.ConnectionClosed(None, None)
        )

    async def close(self) -> None:
        self.closed = True
        self._close_event.set()
        # Unblock any waiting recv
        self.feed_close()


class AutoAuthMockWebSocket(MockWebSocket):
    """MockWebSocket that auto-replies to signalwire.connect requests.

    Watches send() for a signalwire.connect request and immediately injects
    a matching success response into the recv queue.
    """

    def __init__(self, protocol: str = "test-protocol-abc123", auto_reply_all: bool = False) -> None:
        super().__init__()
        self.protocol = protocol
        self.auto_reply_all = auto_reply_all

    async def send(self, raw: str) -> None:
        await super().send(raw)
        msg = json.loads(raw)
        if msg.get("method") == METHOD_SIGNALWIRE_CONNECT:
            self.feed_message(make_jsonrpc_response(msg["id"], {
                "protocol": self.protocol,
                "identity": "test-identity",
            }))
        elif self.auto_reply_all and "id" in msg and "method" in msg:
            self.feed_message(make_jsonrpc_response(msg["id"], {
                "code": "200", "message": "OK",
            }))


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def make_jsonrpc_response(req_id: str, result: Any) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 success response."""
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def make_jsonrpc_error(req_id: str, code: int, message: str) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 error response."""
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def make_event(
    event_type: str,
    params: dict[str, Any] | None = None,
    msg_id: str | None = None,
) -> dict[str, Any]:
    """Build a signalwire.event server message."""
    return {
        "jsonrpc": "2.0",
        "id": msg_id or str(uuid.uuid4()),
        "method": "signalwire.event",
        "params": {
            "event_type": event_type,
            "params": params or {},
        },
    }


def make_server_ping(msg_id: str | None = None) -> dict[str, Any]:
    """Build a signalwire.ping server message."""
    return {
        "jsonrpc": "2.0",
        "id": msg_id or str(uuid.uuid4()),
        "method": "signalwire.ping",
        "params": {},
    }


def make_calling_response(
    req_id: str, code: str = "200", message: str = "OK", **extra: Any
) -> dict[str, Any]:
    """Build a calling API response (result with code field)."""
    result = {"code": code, "message": message}
    result.update(extra)
    return make_jsonrpc_response(req_id, result)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def relay_client():
    """Fresh RelayClient with test credentials.  Clears _active_clients."""
    _active_clients.clear()
    client = RelayClient(project="test-project", token="test-token")
    yield client
    _active_clients.clear()


@pytest_asyncio.fixture
async def connected_client():
    """A RelayClient that is connected via a mocked WebSocket.

    Yields ``(client, mock_ws)``.  Disconnects on teardown.
    """
    _active_clients.clear()
    mock_ws = AutoAuthMockWebSocket()

    with patch("signalwire_agents.relay.client.websockets.connect",
               new_callable=AsyncMock, return_value=mock_ws):
        client = RelayClient(project="test-project", token="test-token")
        await client.connect()
        yield client, mock_ws
        if client._connected:
            await client.disconnect()

    _active_clients.clear()


@pytest.fixture
def make_call():
    """Factory that creates a Call object on a connected client."""

    def _factory(
        client: RelayClient,
        call_id: str = "call-1",
        node_id: str = "node-1",
        state: str = "created",
        direction: str = "inbound",
    ) -> Call:
        call = Call(
            client=client,
            call_id=call_id,
            node_id=node_id,
            project_id=client.project,
            context="test-ctx",
            direction=direction,
            state=state,
        )
        client._calls[call_id] = call
        return call

    return _factory
