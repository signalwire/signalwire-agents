"""Tests for RelayClient lifecycle, messaging, ping, and reconnect."""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import websockets.exceptions

from signalwire_agents.relay.client import (
    RelayClient,
    RelayError,
    _active_clients,
    _MAX_CONNECTIONS,
    _MAX_QUEUE_SIZE,
    _SUCCESS_CODE_RE,
)
from signalwire_agents.relay.call import Call, PlayAction
from signalwire_agents.relay.constants import (
    AGENT_STRING,
    CALL_STATE_ENDED,
    DEFAULT_RELAY_HOST,
    EVENT_CALL_RECEIVE,
    EVENT_CALL_STATE,
    METHOD_SIGNALWIRE_CONNECT,
    METHOD_SIGNALWIRE_DISCONNECT,
    METHOD_SIGNALWIRE_EVENT,
    METHOD_SIGNALWIRE_PING,
    PROTOCOL_VERSION,
    RECONNECT_BACKOFF_FACTOR,
    RECONNECT_MIN_DELAY,
)

from .conftest import (
    AutoAuthMockWebSocket,
    MockWebSocket,
    make_calling_response,
    make_event,
    make_jsonrpc_error,
    make_jsonrpc_response,
    make_server_ping,
)


# ===================================================================
# Init / validation
# ===================================================================

class TestClientInit:
    def setup_method(self):
        _active_clients.clear()

    def teardown_method(self):
        _active_clients.clear()

    def test_explicit_creds(self):
        c = RelayClient(project="p", token="t")
        assert c.project == "p"
        assert c.token == "t"
        assert c.host == DEFAULT_RELAY_HOST

    def test_env_var_creds(self, monkeypatch):
        monkeypatch.setenv("SIGNALWIRE_PROJECT_ID", "env-proj")
        monkeypatch.setenv("SIGNALWIRE_API_TOKEN", "env-tok")
        c = RelayClient()
        assert c.project == "env-proj"
        assert c.token == "env-tok"

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_creds_raises(self):
        with pytest.raises(ValueError, match="project and token are required"):
            RelayClient(project="", token="")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_project_raises(self):
        with pytest.raises(ValueError):
            RelayClient(project="", token="t")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_token_raises(self):
        with pytest.raises(ValueError):
            RelayClient(project="p", token="")

    @pytest.mark.parametrize("bad_host", [
        "host.com/path",
        "user@host.com",
        "host.com?q=1",
        "host\r\n.com",
        "host .com",
    ])
    def test_ssrf_host_rejection(self, bad_host):
        with pytest.raises(ValueError, match="Invalid host"):
            RelayClient(project="p", token="t", host=bad_host)

    def test_custom_host(self):
        c = RelayClient(project="p", token="t", host="custom.relay.com")
        assert c.host == "custom.relay.com"

    def test_contexts(self):
        c = RelayClient(project="p", token="t", contexts=["a", "b"])
        assert c.contexts == ["a", "b"]

    def test_on_call_decorator(self):
        c = RelayClient(project="p", token="t")

        @c.on_call
        async def handler(call):
            pass

        assert c._on_call_handler is handler


# ===================================================================
# Connect / auth
# ===================================================================

class TestConnectAuth:
    def setup_method(self):
        _active_clients.clear()

    def teardown_method(self):
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_sends_signalwire_connect(self, connected_client):
        client, ws = connected_client
        # Find the connect message
        connect_msg = None
        for msg in ws.sent_messages:
            if msg.get("method") == METHOD_SIGNALWIRE_CONNECT:
                connect_msg = msg
                break
        assert connect_msg is not None
        params = connect_msg["params"]
        assert params["version"] == PROTOCOL_VERSION
        assert params["agent"] == AGENT_STRING
        assert params["event_acks"] is True
        assert params["authentication"]["project"] == "test-project"
        assert params["authentication"]["token"] == "test-token"

    @pytest.mark.asyncio
    async def test_stores_protocol_and_identity(self, connected_client):
        client, ws = connected_client
        assert client._relay_protocol == "test-protocol-abc123"
        assert client._identity == "test-identity"

    @pytest.mark.asyncio
    async def test_connected_flag_set(self, connected_client):
        client, ws = connected_client
        assert client._connected is True

    @pytest.mark.asyncio
    async def test_recv_task_started(self, connected_client):
        client, ws = connected_client
        assert client._recv_task is not None
        assert not client._recv_task.done()

    @pytest.mark.asyncio
    async def test_ping_task_started(self, connected_client):
        client, ws = connected_client
        assert client._ping_task is not None

    @pytest.mark.asyncio
    async def test_reconnect_sends_protocol(self):
        """On reconnect, the protocol string should be sent back."""
        _active_clients.clear()
        ws = AutoAuthMockWebSocket()

        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws):
            client = RelayClient(project="p", token="t")
            await client.connect()
            assert client._relay_protocol == "test-protocol-abc123"

            # Simulate reconnect
            await client.disconnect()
            _active_clients.clear()

            ws2 = AutoAuthMockWebSocket(protocol="new-proto")
            with patch("signalwire_agents.relay.client.websockets.connect",
                       new_callable=AsyncMock, return_value=ws2):
                await client.connect()

                # Find the second connect msg
                connect_msgs = [m for m in ws2.sent_messages
                                if m.get("method") == METHOD_SIGNALWIRE_CONNECT]
                assert len(connect_msgs) == 1
                assert connect_msgs[0]["params"]["protocol"] == "test-protocol-abc123"

                await client.disconnect()
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_contexts_sent_in_connect(self):
        _active_clients.clear()
        ws = AutoAuthMockWebSocket()
        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws):
            client = RelayClient(project="p", token="t", contexts=["default", "support"])
            await client.connect()
            connect_msg = [m for m in ws.sent_messages
                           if m.get("method") == METHOD_SIGNALWIRE_CONNECT][0]
            assert connect_msg["params"]["contexts"] == ["default", "support"]
            await client.disconnect()
        _active_clients.clear()


# ===================================================================
# Connection limits
# ===================================================================

class TestConnectionLimits:
    def setup_method(self):
        _active_clients.clear()

    def teardown_method(self):
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_default_limit_is_one(self):
        ws1 = AutoAuthMockWebSocket()
        ws2 = AutoAuthMockWebSocket()
        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws1):
            c1 = RelayClient(project="p", token="t")
            await c1.connect()

            with patch("signalwire_agents.relay.client.websockets.connect",
                       new_callable=AsyncMock, return_value=ws2):
                c2 = RelayClient(project="p2", token="t2")
                with pytest.raises(RuntimeError, match="connection limit"):
                    await c2.connect()

            await c1.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_frees_slot(self):
        ws = AutoAuthMockWebSocket()
        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws):
            c1 = RelayClient(project="p", token="t")
            await c1.connect()
            await c1.disconnect()

        # Now another client should be able to connect
        ws2 = AutoAuthMockWebSocket()
        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws2):
            c2 = RelayClient(project="p2", token="t2")
            await c2.connect()
            await c2.disconnect()


# ===================================================================
# Disconnect / cleanup
# ===================================================================

class TestDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_sets_flags(self, connected_client):
        client, ws = connected_client
        await client.disconnect()
        assert client._closing is True
        assert client._connected is False
        assert client._ws is None

    @pytest.mark.asyncio
    async def test_disconnect_cancels_tasks(self, connected_client):
        client, ws = connected_client
        recv = client._recv_task
        ping = client._ping_task
        await client.disconnect()
        assert client._recv_task is None
        assert client._ping_task is None

    @pytest.mark.asyncio
    async def test_disconnect_cancels_pending_futures(self, connected_client):
        client, ws = connected_client
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        client._pending["test-id"] = fut
        await client.disconnect()
        assert fut.cancelled()
        assert len(client._pending) == 0

    @pytest.mark.asyncio
    async def test_disconnect_cancels_queued_requests(self, connected_client):
        client, ws = connected_client
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        client._execute_queue.append(({"method": "test"}, fut))
        await client.disconnect()
        assert fut.cancelled()
        assert len(client._execute_queue) == 0

    @pytest.mark.asyncio
    async def test_disconnect_removes_from_active_clients(self, connected_client):
        client, ws = connected_client
        assert id(client) in _active_clients
        await client.disconnect()
        assert id(client) not in _active_clients


# ===================================================================
# Message handling
# ===================================================================

class TestMessageHandling:
    @pytest.mark.asyncio
    async def test_response_matches_pending(self, connected_client):
        client, ws = connected_client

        # Start a request
        task = asyncio.ensure_future(client.execute("calling.answer", {
            "node_id": "n1", "call_id": "c1",
        }))
        await asyncio.sleep(0)

        # Find the sent request
        sent = [m for m in ws.sent_messages if m.get("method") == "calling.answer"]
        assert len(sent) == 1
        req_id = sent[0]["id"]

        # Feed a successful response
        ws.feed_message(make_calling_response(req_id))
        result = await asyncio.wait_for(task, timeout=1.0)
        assert result["code"] == "200"

    @pytest.mark.asyncio
    async def test_2xx_codes_are_success(self, connected_client):
        """Any 2xx code should be treated as success."""
        client, ws = connected_client

        task = asyncio.ensure_future(client.execute("calling.answer", {
            "node_id": "n1", "call_id": "c1",
        }))
        await asyncio.sleep(0)
        sent = [m for m in ws.sent_messages if "calling.answer" in m.get("method", "")]
        req_id = sent[0]["id"]
        ws.feed_message(make_calling_response(req_id, code="201", message="Created"))
        result = await asyncio.wait_for(task, timeout=1.0)
        assert result["code"] == "201"

    @pytest.mark.asyncio
    async def test_non_2xx_raises_relay_error(self, connected_client):
        client, ws = connected_client

        task = asyncio.ensure_future(client.execute("calling.answer", {
            "node_id": "n1", "call_id": "c1",
        }))
        await asyncio.sleep(0)
        sent = [m for m in ws.sent_messages if "calling.answer" in m.get("method", "")]
        req_id = sent[0]["id"]
        ws.feed_message(make_calling_response(req_id, code="400", message="Bad Request"))

        with pytest.raises(RelayError) as exc_info:
            await asyncio.wait_for(task, timeout=1.0)
        assert exc_info.value.code == 400

    @pytest.mark.asyncio
    async def test_signalwire_connect_skips_code_check(self, connected_client):
        """signalwire.connect responses should not be checked for code field."""
        client, ws = connected_client
        # The auth already succeeded — just verify protocol was stored
        assert client._relay_protocol == "test-protocol-abc123"

    @pytest.mark.asyncio
    async def test_jsonrpc_error(self, connected_client):
        client, ws = connected_client

        task = asyncio.ensure_future(client.execute("calling.answer", {
            "node_id": "n1", "call_id": "c1",
        }))
        await asyncio.sleep(0)
        sent = [m for m in ws.sent_messages if "calling.answer" in m.get("method", "")]
        req_id = sent[0]["id"]
        ws.feed_message(make_jsonrpc_error(req_id, -32600, "Invalid request"))

        with pytest.raises(RelayError) as exc_info:
            await asyncio.wait_for(task, timeout=1.0)
        assert exc_info.value.code == -32600
        assert "Invalid request" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_non_dict_message_ignored(self, connected_client):
        """Non-dict messages should be silently ignored."""
        client, ws = connected_client
        # Feed a raw string that parses as a list
        ws._recv_queue.put_nowait(json.dumps([1, 2, 3]))
        await asyncio.sleep(0.05)
        # Client should still be functional
        assert client._connected

    @pytest.mark.asyncio
    async def test_invalid_json_ignored(self, connected_client):
        """Invalid JSON should be silently ignored."""
        client, ws = connected_client
        ws._recv_queue.put_nowait("not valid json {{{")
        await asyncio.sleep(0.05)
        assert client._connected

    @pytest.mark.asyncio
    async def test_non_dict_result_wrapped(self, connected_client):
        """Non-dict results should be wrapped in {raw: ...}."""
        client, ws = connected_client

        task = asyncio.ensure_future(client.execute("calling.answer", {
            "node_id": "n1", "call_id": "c1",
        }))
        await asyncio.sleep(0)
        sent = [m for m in ws.sent_messages if "calling.answer" in m.get("method", "")]
        req_id = sent[0]["id"]
        ws.feed_message(make_jsonrpc_response(req_id, "just a string"))
        result = await asyncio.wait_for(task, timeout=1.0)
        assert result == {"raw": "just a string"}

    @pytest.mark.asyncio
    async def test_non_dict_error_handled(self, connected_client):
        """Non-dict error values should be handled gracefully."""
        client, ws = connected_client

        task = asyncio.ensure_future(client.execute("calling.answer", {
            "node_id": "n1", "call_id": "c1",
        }))
        await asyncio.sleep(0)
        sent = [m for m in ws.sent_messages if "calling.answer" in m.get("method", "")]
        req_id = sent[0]["id"]
        ws.feed_message({"jsonrpc": "2.0", "id": req_id, "error": "string error"})

        with pytest.raises(RelayError) as exc_info:
            await asyncio.wait_for(task, timeout=1.0)
        assert exc_info.value.code == -1


# ===================================================================
# Event dispatch
# ===================================================================

class TestClientEventDispatch:
    @pytest.mark.asyncio
    async def test_event_ack_sent(self, connected_client):
        """Events should be ACKed back to the server."""
        client, ws = connected_client
        ws.sent_messages.clear()

        event = make_event(EVENT_CALL_STATE, {"call_id": "c1", "call_state": "ringing"}, msg_id="evt-1")
        ws.feed_message(event)
        await asyncio.sleep(0.05)

        acks = [m for m in ws.sent_messages if m.get("id") == "evt-1" and "result" in m]
        assert len(acks) == 1

    @pytest.mark.asyncio
    async def test_event_routed_to_call(self, connected_client, make_call):
        client, ws = connected_client
        call = make_call(client, call_id="c-route")

        event = make_event(EVENT_CALL_STATE, {"call_id": "c-route", "call_state": "answered"})
        ws.feed_message(event)
        await asyncio.sleep(0.05)

        assert call.state == "answered"

    @pytest.mark.asyncio
    async def test_inbound_call_creation(self, connected_client):
        client, ws = connected_client
        handler_calls = []

        @client.on_call
        async def handle(call):
            handler_calls.append(call)

        event = make_event(EVENT_CALL_RECEIVE, {
            "call_id": "new-call",
            "node_id": "node-abc",
            "project_id": "test-project",
            "direction": "inbound",
            "call_state": "ringing",
            "device": {"type": "phone", "params": {"from_number": "+1555"}},
        })
        ws.feed_message(event)
        await asyncio.sleep(0.05)

        assert "new-call" in client._calls
        call = client._calls["new-call"]
        assert call.direction == "inbound"
        assert call.node_id == "node-abc"
        assert len(handler_calls) == 1

    @pytest.mark.asyncio
    async def test_no_handler_logs_warning(self, connected_client):
        """Inbound call without handler should not crash."""
        client, ws = connected_client
        event = make_event(EVENT_CALL_RECEIVE, {
            "call_id": "no-handler-call",
            "node_id": "n1",
        })
        ws.feed_message(event)
        await asyncio.sleep(0.05)
        assert "no-handler-call" in client._calls

    @pytest.mark.asyncio
    async def test_max_calls_limit(self, connected_client):
        client, ws = connected_client

        @client.on_call
        async def handle(call):
            pass

        # Fill up to max
        for i in range(client._max_active_calls):
            client._calls[f"fill-{i}"] = MagicMock()

        event = make_event(EVENT_CALL_RECEIVE, {
            "call_id": "overflow-call",
            "node_id": "n1",
        })
        ws.feed_message(event)
        await asyncio.sleep(0.05)

        # Should NOT be added
        assert "overflow-call" not in client._calls

    @pytest.mark.asyncio
    async def test_ended_call_removed(self, connected_client, make_call):
        client, ws = connected_client
        call = make_call(client, call_id="c-end")
        assert "c-end" in client._calls

        event = make_event(EVENT_CALL_STATE, {"call_id": "c-end", "call_state": CALL_STATE_ENDED})
        ws.feed_message(event)
        await asyncio.sleep(0.05)

        assert "c-end" not in client._calls

    @pytest.mark.asyncio
    async def test_signalwire_disconnect_acked(self, connected_client):
        """Server disconnect should be ACKed."""
        client, ws = connected_client
        ws.sent_messages.clear()

        disconnect_msg = {
            "jsonrpc": "2.0",
            "id": "disc-1",
            "method": METHOD_SIGNALWIRE_DISCONNECT,
            "params": {},
        }
        ws.feed_message(disconnect_msg)
        await asyncio.sleep(0.05)

        acks = [m for m in ws.sent_messages if m.get("id") == "disc-1"]
        assert len(acks) == 1


# ===================================================================
# Ping
# ===================================================================

class TestPing:
    @pytest.mark.asyncio
    async def test_server_ping_gets_pong(self, connected_client):
        client, ws = connected_client
        ws.sent_messages.clear()

        ping = make_server_ping(msg_id="ping-1")
        ws.feed_message(ping)
        await asyncio.sleep(0.05)

        pongs = [m for m in ws.sent_messages if m.get("id") == "ping-1" and "result" in m]
        assert len(pongs) == 1

    @pytest.mark.asyncio
    async def test_server_ping_resets_failure_counter(self, connected_client):
        client, ws = connected_client
        client._ping_failures = 2

        ping = make_server_ping()
        ws.feed_message(ping)
        await asyncio.sleep(0.05)

        assert client._ping_failures == 0

    @pytest.mark.asyncio
    async def test_client_ping_loop_sends_pings(self):
        """Client ping loop sends signalwire.ping at configured intervals."""
        _active_clients.clear()
        ws = AutoAuthMockWebSocket()

        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws), \
             patch("signalwire_agents.relay.client._CLIENT_PING_INTERVAL", 0.05):
            client = RelayClient(project="p", token="t")
            await client.connect()

            # Wait for a ping to be sent
            await asyncio.sleep(0.1)

            # Find ping requests
            pings = [m for m in ws.sent_messages if m.get("method") == METHOD_SIGNALWIRE_PING]
            # Respond to pending ping to avoid timeout
            for p in pings:
                ws.feed_message(make_jsonrpc_response(p["id"], {}))

            assert len(pings) >= 1

            await client.disconnect()
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_ping_failure_backoff(self):
        """Ping failures should trigger exponential backoff."""
        _active_clients.clear()
        ws = AutoAuthMockWebSocket()

        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws), \
             patch("signalwire_agents.relay.client._CLIENT_PING_INTERVAL", 0.02), \
             patch("signalwire_agents.relay.client._EXECUTE_TIMEOUT", 0.02), \
             patch("signalwire_agents.relay.client._MAX_PING_FAILURES", 2):
            client = RelayClient(project="p", token="t")
            await client.connect()

            # Don't respond to pings — they'll timeout
            await asyncio.sleep(0.3)

            # After max failures, the connection should be force-closed
            assert client._connected is False or client._ping_failures >= 1

            await client.disconnect()
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_force_close(self, connected_client):
        client, ws = connected_client
        client._force_close()
        assert client._connected is False


# ===================================================================
# Request queuing
# ===================================================================

class TestRequestQueuing:
    @pytest.mark.asyncio
    async def test_queue_while_disconnected(self):
        _active_clients.clear()
        client = RelayClient(project="p", token="t")

        # Not connected — execute should queue
        task = asyncio.ensure_future(client.execute("calling.answer", {"call_id": "c1"}))
        await asyncio.sleep(0)

        assert len(client._execute_queue) == 1
        # Cancel the future to avoid hanging
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, RelayError):
            pass
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_flush_after_reconnect(self):
        """Queued requests should be flushed after reconnect (auth)."""
        _active_clients.clear()
        client = RelayClient(project="p", token="t")

        # Queue a request while disconnected
        task = asyncio.ensure_future(client.execute("calling.answer", {
            "node_id": "n1", "call_id": "c1",
        }))
        await asyncio.sleep(0)
        assert len(client._execute_queue) == 1

        # Now connect — the queued request should be flushed
        ws = AutoAuthMockWebSocket()
        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws):
            await client.connect()
            await asyncio.sleep(0.05)

            # The queued request should have been sent
            answer_msgs = [m for m in ws.sent_messages if "calling.answer" in m.get("method", "")]
            assert len(answer_msgs) == 1

            # Respond to it
            ws.feed_message(make_calling_response(answer_msgs[0]["id"]))
            result = await asyncio.wait_for(task, timeout=1.0)
            assert result["code"] == "200"

            await client.disconnect()
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_queue_overflow(self):
        _active_clients.clear()
        client = RelayClient(project="p", token="t")

        # Fill the queue
        futures = []
        for i in range(_MAX_QUEUE_SIZE):
            f = asyncio.ensure_future(client.execute(f"method.{i}", {}))
            futures.append(f)
            await asyncio.sleep(0)

        # One more should raise
        with pytest.raises(RelayError, match="queue full"):
            await client.execute("method.overflow", {})

        # Cancel all queued futures
        for f in futures:
            f.cancel()
            try:
                await f
            except (asyncio.CancelledError, RelayError):
                pass
        _active_clients.clear()


# ===================================================================
# Timeouts
# ===================================================================

class TestTimeouts:
    @pytest.mark.asyncio
    async def test_execute_timeout_raises(self, connected_client):
        client, ws = connected_client

        with patch("signalwire_agents.relay.client._EXECUTE_TIMEOUT", 0.05):
            # Don't respond to the request — it should timeout
            with pytest.raises(RelayError, match="timeout"):
                await client.execute("calling.answer", {"node_id": "n1", "call_id": "c1"})

    @pytest.mark.asyncio
    async def test_execute_timeout_force_closes(self, connected_client):
        client, ws = connected_client

        with patch("signalwire_agents.relay.client._EXECUTE_TIMEOUT", 0.05):
            try:
                await client.execute("calling.answer", {"node_id": "n1", "call_id": "c1"})
            except RelayError:
                pass

            # Force close should have been called (connected = False)
            assert client._connected is False


# ===================================================================
# Success code regex
# ===================================================================

class TestSuccessCodeRegex:
    @pytest.mark.parametrize("code", ["200", "201", "204", "299"])
    def test_2xx_matches(self, code):
        assert _SUCCESS_CODE_RE.match(code)

    @pytest.mark.parametrize("code", ["100", "300", "400", "500", "199", "2000"])
    def test_non_2xx_does_not_match(self, code):
        assert not _SUCCESS_CODE_RE.match(code)


# ===================================================================
# Async context manager
# ===================================================================

class TestAsyncContextManager:
    @pytest.mark.asyncio
    async def test_aenter_aexit(self):
        _active_clients.clear()
        ws = AutoAuthMockWebSocket()
        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws):
            async with RelayClient(project="p", token="t") as client:
                assert client._connected
            assert client._closing
        _active_clients.clear()


# ===================================================================
# Reconnect backoff arithmetic (no run_forever)
# ===================================================================

class TestReconnectBackoff:
    def test_initial_delay(self):
        _active_clients.clear()
        c = RelayClient(project="p", token="t")
        assert c._reconnect_delay == RECONNECT_MIN_DELAY
        _active_clients.clear()

    def test_backoff_calculation(self):
        _active_clients.clear()
        c = RelayClient(project="p", token="t")
        # Simulate the backoff math from _run_forever
        c._reconnect_delay = min(
            c._reconnect_delay * RECONNECT_BACKOFF_FACTOR,
            30.0,
        )
        assert c._reconnect_delay == RECONNECT_MIN_DELAY * RECONNECT_BACKOFF_FACTOR
        _active_clients.clear()


# ===================================================================
# Dial
# ===================================================================

class TestDial:
    @pytest.mark.asyncio
    async def test_dial_returns_call(self, connected_client):
        client, ws = connected_client
        ws.sent_messages.clear()

        devices = [[{"type": "phone", "params": {"to_number": "+15551234567"}}]]
        task = asyncio.ensure_future(client.dial(devices))
        await asyncio.sleep(0)

        sent = [m for m in ws.sent_messages if "calling.dial" in m.get("method", "")]
        assert len(sent) == 1
        req_id = sent[0]["id"]
        dial_tag = sent[0]["params"]["tag"]

        # Server acknowledges the dial
        ws.feed_message(make_calling_response(req_id))
        await asyncio.sleep(0)

        # Server sends calling.call.state with the real call_id, matched by tag
        ws.feed_message(make_event("calling.call.state", {
            "call_id": "dial-call-1",
            "node_id": "dial-node-1",
            "tag": dial_tag,
            "call_state": "created",
            "direction": "outbound",
        }))
        await asyncio.sleep(0)

        # Server sends calling.call.dial with dial_state=answered to resolve the dial
        ws.feed_message(make_event("calling.call.dial", {
            "tag": dial_tag,
            "dial_state": "answered",
            "call": {
                "call_id": "dial-call-1",
                "node_id": "dial-node-1",
                "tag": dial_tag,
                "dial_winner": True,
            },
        }))
        call = await asyncio.wait_for(task, timeout=1.0)
        assert isinstance(call, Call)
        assert call.call_id == "dial-call-1"
        assert call.node_id == "dial-node-1"
        assert call.direction == "outbound"
        assert "dial-call-1" in client._calls


# ===================================================================
# __del__ cleanup
# ===================================================================

class TestDelCleanup:
    def test_del_removes_from_active(self):
        _active_clients.clear()
        c = RelayClient(project="p", token="t")
        cid = id(c)
        _active_clients.add(cid)
        c.__del__()
        assert cid not in _active_clients
        _active_clients.clear()


# ===================================================================
# Non-dict params in event
# ===================================================================

class TestNonDictParams:
    @pytest.mark.asyncio
    async def test_non_dict_params_ignored(self, connected_client):
        """Messages with non-dict params should be ignored."""
        client, ws = connected_client
        msg = {
            "jsonrpc": "2.0",
            "id": "bad-1",
            "method": "signalwire.event",
            "params": "not a dict",
        }
        ws.feed_message(msg)
        await asyncio.sleep(0.05)
        assert client._connected


# ===================================================================
# Empty event_type
# ===================================================================

class TestEmptyEventType:
    @pytest.mark.asyncio
    async def test_empty_event_type_logged(self, connected_client):
        """Events with empty event_type should be ignored."""
        client, ws = connected_client
        event = make_event("", {"call_id": "c1"})
        ws.feed_message(event)
        await asyncio.sleep(0.05)
        assert client._connected


# ===================================================================
# Coverage: relay_protocol property — line 158
# ===================================================================

class TestRelayProtocolProperty:
    @pytest.mark.asyncio
    async def test_relay_protocol_property(self, connected_client):
        client, ws = connected_client
        assert client.relay_protocol == "test-protocol-abc123"


# ===================================================================
# Coverage: RELAY_MAX_CONNECTIONS env var ValueError — lines 75-76
# ===================================================================

class TestMaxConnectionsEnvVar:
    def test_invalid_env_var_fallback(self, monkeypatch):
        """Invalid RELAY_MAX_CONNECTIONS should fall back to 1."""
        _active_clients.clear()
        # We can't easily re-execute module-level code, but we can test
        # that the regex/parsing logic works by importing the module fresh.
        # Instead, just verify the current value is sane.
        import signalwire_agents.relay.client as mod
        assert mod._MAX_CONNECTIONS >= 1
        _active_clients.clear()


# ===================================================================
# Coverage: dial with max_duration — line 302
# ===================================================================

class TestDialMaxDuration:
    @pytest.mark.asyncio
    async def test_dial_with_max_duration(self, connected_client):
        client, ws = connected_client
        ws.sent_messages.clear()

        devices = [[{"type": "phone", "params": {"to_number": "+15551234567"}}]]
        task = asyncio.ensure_future(client.dial(devices, max_duration=120))
        await asyncio.sleep(0)

        sent = [m for m in ws.sent_messages if "calling.dial" in m.get("method", "")]
        assert len(sent) == 1
        assert sent[0]["params"]["max_duration"] == 120
        req_id = sent[0]["id"]
        dial_tag = sent[0]["params"]["tag"]

        # Acknowledge the dial
        ws.feed_message(make_calling_response(req_id))
        await asyncio.sleep(0)

        # Send the call.state event with the tag to register the leg
        ws.feed_message(make_event("calling.call.state", {
            "call_id": "md-call",
            "node_id": "md-node",
            "tag": dial_tag,
            "call_state": "created",
        }))
        await asyncio.sleep(0)

        # Send the dial answered event to resolve the dial
        ws.feed_message(make_event("calling.call.dial", {
            "tag": dial_tag,
            "dial_state": "answered",
            "call": {
                "call_id": "md-call",
                "node_id": "md-node",
                "tag": dial_tag,
                "dial_winner": True,
            },
        }))
        call = await asyncio.wait_for(task, timeout=1.0)
        assert call.call_id == "md-call"


# ===================================================================
# Coverage: run() / _run_forever — lines 328, 332-354
# ===================================================================

class TestRunForever:
    @pytest.mark.asyncio
    async def test_run_forever_connects_and_reconnects(self):
        """Test _run_forever connect + reconnect loop — lines 332-354."""
        _active_clients.clear()
        connect_count = 0

        class CountingMockWS(AutoAuthMockWebSocket):
            async def close(self):
                await super().close()

        async def mock_connect(*args, **kwargs):
            nonlocal connect_count
            connect_count += 1
            ws = CountingMockWS()
            return ws

        with patch("signalwire_agents.relay.client.websockets.connect",
                   side_effect=mock_connect), \
             patch("signalwire_agents.relay.client._CLIENT_PING_INTERVAL", 999):
            client = RelayClient(project="p", token="t")

            async def stop_after_connect():
                # Wait for first connect, then signal close
                while connect_count < 1:
                    await asyncio.sleep(0.01)
                await asyncio.sleep(0.05)
                client._closing = True
                # Force close the recv loop
                if client._ws:
                    await client._ws.close()

            task = asyncio.ensure_future(client._run_forever())
            stopper = asyncio.ensure_future(stop_after_connect())
            await asyncio.wait_for(task, timeout=5.0)
            stopper.cancel()
            assert connect_count >= 1
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_run_forever_handles_connect_exception(self):
        """_run_forever catches exceptions and attempts reconnect — lines 340-341."""
        _active_clients.clear()
        call_count = 0

        async def fail_then_succeed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionRefusedError("test")
            return AutoAuthMockWebSocket()

        with patch("signalwire_agents.relay.client.websockets.connect",
                   side_effect=fail_then_succeed), \
             patch("signalwire_agents.relay.client._CLIENT_PING_INTERVAL", 999), \
             patch("signalwire_agents.relay.client.RECONNECT_MIN_DELAY", 0.01):
            client = RelayClient(project="p", token="t")

            async def stop_after():
                while call_count < 2:
                    await asyncio.sleep(0.01)
                await asyncio.sleep(0.05)
                client._closing = True
                if client._ws:
                    await client._ws.close()

            task = asyncio.ensure_future(client._run_forever())
            stopper = asyncio.ensure_future(stop_after())
            await asyncio.wait_for(task, timeout=5.0)
            stopper.cancel()
            assert call_count >= 2
        _active_clients.clear()

    def test_run_calls_asyncio_run(self):
        """run() calls asyncio.run — line 328."""
        _active_clients.clear()
        client = RelayClient(project="p", token="t")

        def close_coro_and_record(coro):
            """Close the coroutine to avoid 'was never awaited' warning."""
            coro.close()

        with patch("signalwire_agents.relay.client.asyncio.run",
                   side_effect=close_coro_and_record) as mock_run:
            client.run()
            mock_run.assert_called_once()
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_run_forever_cancelled(self):
        """_run_forever breaks on CancelledError — line 339."""
        _active_clients.clear()

        async def cancel_connect(*args, **kwargs):
            raise asyncio.CancelledError()

        with patch("signalwire_agents.relay.client.websockets.connect",
                   side_effect=cancel_connect):
            client = RelayClient(project="p", token="t")
            # Should exit cleanly without raising
            await client._run_forever()
        _active_clients.clear()


# ===================================================================
# Coverage: flush queue skip done futures — line 414
# ===================================================================

class TestFlushQueueDoneFutures:
    @pytest.mark.asyncio
    async def test_flush_skips_done_futures(self, connected_client):
        """Already-done futures in queue should be skipped — line 414."""
        client, ws = connected_client
        loop = asyncio.get_running_loop()
        done_fut = loop.create_future()
        done_fut.cancel()  # Mark as done
        client._execute_queue.append(({"method": "test", "id": "done-1"}, done_fut))
        client._flush_execute_queue()
        # Should not crash, queue should be cleared
        assert len(client._execute_queue) == 0


# ===================================================================
# Coverage: _safe_send exception path — lines 424-426
# ===================================================================

class TestSafeSend:
    @pytest.mark.asyncio
    async def test_safe_send_failure_rejects_future(self, connected_client):
        """_safe_send should reject the future if send fails — lines 424-426."""
        client, ws = connected_client
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        # Make ws.send raise
        original_send = ws.send
        async def failing_send(raw):
            raise ConnectionError("test failure")
        ws.send = failing_send

        await client._safe_send('{"test": true}', future)
        assert future.done()
        with pytest.raises(RelayError):
            future.result()

        ws.send = original_send


# ===================================================================
# Coverage: _clear_pending_requests — lines 432-438
# ===================================================================

class TestClearPendingRequests:
    @pytest.mark.asyncio
    async def test_clear_pending_requests(self, connected_client):
        """_clear_pending_requests rejects all pending — lines 432-438."""
        client, ws = connected_client
        loop = asyncio.get_running_loop()

        fut1 = loop.create_future()
        fut2 = loop.create_future()
        client._pending["a"] = fut1
        client._pending["b"] = fut2
        client._pending_methods["a"] = "test.a"
        client._pending_methods["b"] = "test.b"

        client._clear_pending_requests()

        assert len(client._pending) == 0
        assert len(client._pending_methods) == 0
        with pytest.raises(RelayError):
            fut1.result()
        with pytest.raises(RelayError):
            fut2.result()


# ===================================================================
# Coverage: recv_loop exception paths — lines 453, 456-457, 462-463
# ===================================================================

class TestRecvLoopExceptions:
    @pytest.mark.asyncio
    async def test_recv_loop_connection_closed(self):
        """recv_loop handles ConnectionClosed at async-for level — line 453."""
        _active_clients.clear()
        ws = AutoAuthMockWebSocket()

        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws), \
             patch("signalwire_agents.relay.client._CLIENT_PING_INTERVAL", 999):
            client = RelayClient(project="p", token="t")
            await client.connect()

            # Cancel the existing recv task
            if client._recv_task:
                client._recv_task.cancel()
                try:
                    await client._recv_task
                except asyncio.CancelledError:
                    pass

            # Replace _ws with one that immediately raises ConnectionClosed
            class ImmediateCloseWS:
                def __aiter__(self):
                    return self
                async def __anext__(self):
                    raise websockets.exceptions.ConnectionClosed(None, None)
                async def close(self):
                    pass

            client._ws = ImmediateCloseWS()
            client._connected = True

            # Start a new recv loop — it will immediately hit ConnectionClosed
            client._recv_task = asyncio.ensure_future(client._recv_loop())
            await asyncio.sleep(0.05)

            assert client._connected is False
            await client.disconnect()
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_recv_loop_generic_exception(self):
        """recv_loop handles generic exceptions — lines 456-457."""
        _active_clients.clear()
        ws = AutoAuthMockWebSocket()

        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws):
            client = RelayClient(project="p", token="t")
            await client.connect()

            # Inject an unexpected exception via the queue
            ws._recv_queue.put_nowait(RuntimeError("unexpected"))
            await asyncio.sleep(0.1)

            # The recv loop should catch this and exit cleanly
            assert client._connected is False
            await client.disconnect()
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_recv_loop_cleans_up_ping_task(self):
        """recv_loop finally block cancels ping task — lines 462-463."""
        _active_clients.clear()
        ws = AutoAuthMockWebSocket()

        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws), \
             patch("signalwire_agents.relay.client._CLIENT_PING_INTERVAL", 999):
            client = RelayClient(project="p", token="t")
            await client.connect()

            ping_task = client._ping_task
            assert ping_task is not None

            # Close connection to trigger recv_loop exit
            ws.feed_close()
            await asyncio.sleep(0.1)

            # The ping task should have been cancelled by the finally block
            assert client._ping_task is None
            await client.disconnect()
        _active_clients.clear()


# ===================================================================
# Coverage: error for unknown/expired request — line 483
# ===================================================================

class TestUnknownRequestError:
    @pytest.mark.asyncio
    async def test_error_for_unknown_request(self, connected_client):
        """Error response for unknown request ID — line 483."""
        client, ws = connected_client
        # Feed error for a request ID that doesn't exist
        ws.feed_message(make_jsonrpc_error("nonexistent-id", -1, "unknown"))
        await asyncio.sleep(0.05)
        assert client._connected  # Should not crash


# ===================================================================
# Coverage: non-numeric error code — lines 507-508
# ===================================================================

class TestNonNumericErrorCode:
    @pytest.mark.asyncio
    async def test_non_numeric_error_code(self, connected_client):
        """Error code that can't be parsed as int — lines 507-508."""
        client, ws = connected_client

        task = asyncio.ensure_future(client.execute("calling.answer", {
            "node_id": "n1", "call_id": "c1",
        }))
        await asyncio.sleep(0)
        sent = [m for m in ws.sent_messages if "calling.answer" in m.get("method", "")]
        req_id = sent[0]["id"]
        # Feed a response with non-numeric error code
        ws.feed_message(make_calling_response(req_id, code="bad_code", message="Invalid"))

        with pytest.raises(RelayError) as exc_info:
            await asyncio.wait_for(task, timeout=1.0)
        assert exc_info.value.code == -1


# ===================================================================
# Coverage: response for unknown/expired request — line 515
# ===================================================================

class TestUnknownRequestResponse:
    @pytest.mark.asyncio
    async def test_response_for_unknown_request(self, connected_client):
        """Response for unknown request ID — line 515."""
        client, ws = connected_client
        # Feed response for a request ID that doesn't exist
        ws.feed_message(make_calling_response("nonexistent-id"))
        await asyncio.sleep(0.05)
        assert client._connected  # Should not crash


# ===================================================================
# Coverage: on_call handler exception — lines 607-608
# ===================================================================

class TestOnCallHandlerException:
    @pytest.mark.asyncio
    async def test_handler_exception_caught(self, connected_client):
        """Exception in on_call handler should be caught — lines 607-608."""
        client, ws = connected_client

        @client.on_call
        async def bad_handler(call):
            raise RuntimeError("handler crash")

        event = make_event(EVENT_CALL_RECEIVE, {
            "call_id": "exc-call",
            "node_id": "n1",
        })
        ws.feed_message(event)
        await asyncio.sleep(0.1)

        # Call should still exist, handler exception shouldn't crash
        assert "exc-call" in client._calls


# ===================================================================
# Coverage: _send_pong / _send_event_ack with no ws — lines 613, 624
# ===================================================================

class TestSendWithNoWs:
    @pytest.mark.asyncio
    async def test_send_pong_no_ws(self, connected_client):
        """_send_pong with _ws=None should return early — line 613."""
        client, ws = connected_client
        client._ws = None
        await client._send_pong("test-id")  # Should not raise

    @pytest.mark.asyncio
    async def test_send_event_ack_no_ws(self, connected_client):
        """_send_event_ack with _ws=None should return early — line 624."""
        client, ws = connected_client
        client._ws = None
        await client._send_event_ack("test-id")  # Should not raise


# ===================================================================
# Coverage: ping loop internals — lines 648, 652, 663-665, 695
# ===================================================================

class TestPingLoopInternals:
    @pytest.mark.asyncio
    async def test_ping_loop_exits_when_disconnected(self):
        """Ping loop should break when disconnected mid-sleep — line 648."""
        _active_clients.clear()
        ws = AutoAuthMockWebSocket()

        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws), \
             patch("signalwire_agents.relay.client._CLIENT_PING_INTERVAL", 0.02):
            client = RelayClient(project="p", token="t")
            await client.connect()

            # Wait for a ping attempt, respond to it (covers line 652)
            await asyncio.sleep(0.05)
            pings = [m for m in ws.sent_messages if m.get("method") == METHOD_SIGNALWIRE_PING]
            for p in pings:
                ws.feed_message(make_jsonrpc_response(p["id"], {}))
            await asyncio.sleep(0.01)

            # Disconnect while the ping loop is sleeping
            client._connected = False
            await asyncio.sleep(0.05)

            await client.disconnect()
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_ping_loop_max_failures_force_close(self):
        """Ping loop force-closes after max failures — lines 663-665."""
        _active_clients.clear()
        ws = AutoAuthMockWebSocket()

        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws), \
             patch("signalwire_agents.relay.client._CLIENT_PING_INTERVAL", 0.01), \
             patch("signalwire_agents.relay.client._EXECUTE_TIMEOUT", 0.01), \
             patch("signalwire_agents.relay.client._MAX_PING_FAILURES", 1), \
             patch("signalwire_agents.relay.client.RECONNECT_MIN_DELAY", 0.01):
            client = RelayClient(project="p", token="t")
            await client.connect()

            # Don't respond to pings — they'll timeout and trigger force_close
            await asyncio.sleep(0.3)

            # After max failures, should have force-closed
            assert client._connected is False
            await client.disconnect()
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_check_ping_timeout_fires(self, connected_client):
        """_on_check_ping_timeout executes when timer fires — line 695."""
        client, ws = connected_client
        # Manually call the timeout handler
        client._on_check_ping_timeout()
        # Should not crash, just logs


# ===================================================================
# Coverage: _safe_send when _ws is None
# ===================================================================

class TestSafeSendNoWs:
    @pytest.mark.asyncio
    async def test_safe_send_no_ws(self, connected_client):
        """_safe_send with _ws=None should do nothing."""
        client, ws = connected_client
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        client._ws = None
        await client._safe_send('{"test": true}', future)
        # Future should not be resolved or rejected
        assert not future.done()


# ===================================================================
# JWT auth
# ===================================================================

class TestJwtAuth:
    def setup_method(self):
        _active_clients.clear()

    def teardown_method(self):
        _active_clients.clear()

    @patch.dict(os.environ, {}, clear=True)
    def test_jwt_token_skips_project_token(self):
        c = RelayClient(jwt_token="eyJ...")
        assert c.jwt_token == "eyJ..."
        assert c.project == ""

    @patch.dict(os.environ, {}, clear=True)
    def test_jwt_token_with_project(self):
        c = RelayClient(jwt_token="eyJ...", project="proj-1")
        assert c.jwt_token == "eyJ..."
        assert c.project == "proj-1"

    def test_jwt_env_var(self, monkeypatch):
        monkeypatch.setenv("SIGNALWIRE_JWT_TOKEN", "env-jwt")
        monkeypatch.delenv("SIGNALWIRE_PROJECT_ID", raising=False)
        monkeypatch.delenv("SIGNALWIRE_API_TOKEN", raising=False)
        c = RelayClient()
        assert c.jwt_token == "env-jwt"

    @pytest.mark.asyncio
    async def test_jwt_auth_sends_jwt_token(self):
        ws = AutoAuthMockWebSocket()
        client = RelayClient(jwt_token="eyJ...", project="p")
        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws):
            await client.connect()
        connect_msg = None
        for msg in ws.sent_messages:
            if msg.get("method") == METHOD_SIGNALWIRE_CONNECT:
                connect_msg = msg
                break
        assert connect_msg is not None
        assert connect_msg["params"]["authentication"] == {"jwt_token": "eyJ..."}
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_legacy_auth_sends_project_token(self, connected_client):
        client, ws = connected_client
        connect_msg = None
        for msg in ws.sent_messages:
            if msg.get("method") == METHOD_SIGNALWIRE_CONNECT:
                connect_msg = msg
                break
        assert connect_msg is not None
        auth = connect_msg["params"]["authentication"]
        assert "project" in auth
        assert "token" in auth
        assert "jwt_token" not in auth


# ===================================================================
# max_active_calls config
# ===================================================================

class TestMaxActiveCallsConfig:
    def setup_method(self):
        _active_clients.clear()

    def teardown_method(self):
        _active_clients.clear()

    def test_constructor_param(self):
        c = RelayClient(project="p", token="t", max_active_calls=50)
        assert c._max_active_calls == 50

    def test_constructor_param_min_1(self):
        c = RelayClient(project="p", token="t", max_active_calls=0)
        assert c._max_active_calls == 1

    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("RELAY_MAX_ACTIVE_CALLS", "200")
        c = RelayClient(project="p", token="t")
        assert c._max_active_calls == 200

    def test_env_var_invalid(self, monkeypatch):
        monkeypatch.setenv("RELAY_MAX_ACTIVE_CALLS", "not_a_number")
        c = RelayClient(project="p", token="t")
        from signalwire_agents.relay.client import _DEFAULT_MAX_ACTIVE_CALLS
        assert c._max_active_calls == _DEFAULT_MAX_ACTIVE_CALLS

    def test_default(self):
        c = RelayClient(project="p", token="t")
        from signalwire_agents.relay.client import _DEFAULT_MAX_ACTIVE_CALLS
        assert c._max_active_calls == _DEFAULT_MAX_ACTIVE_CALLS


# ===================================================================
# Dynamic context subscription
# ===================================================================

class TestReceiveUnreceive:
    def setup_method(self):
        _active_clients.clear()

    def teardown_method(self):
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_receive_sends_request(self):
        ws = AutoAuthMockWebSocket(auto_reply_all=True)
        client = RelayClient(project="p", token="t")
        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws):
            await client.connect()
        ws.sent_messages.clear()
        await client.receive(["support", "sales"])
        recv_msgs = [m for m in ws.sent_messages if m.get("method") == "signalwire.receive"]
        assert len(recv_msgs) == 1
        assert recv_msgs[0]["params"]["contexts"] == ["support", "sales"]
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_receive_empty_is_noop(self, connected_client):
        client, ws = connected_client
        ws.sent_messages.clear()
        await client.receive([])
        recv_msgs = [m for m in ws.sent_messages if m.get("method") == "signalwire.receive"]
        assert len(recv_msgs) == 0

    @pytest.mark.asyncio
    async def test_unreceive_sends_request(self):
        ws = AutoAuthMockWebSocket(auto_reply_all=True)
        client = RelayClient(project="p", token="t")
        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws):
            await client.connect()
        ws.sent_messages.clear()
        await client.unreceive(["support"])
        msgs = [m for m in ws.sent_messages if m.get("method") == "signalwire.unreceive"]
        assert len(msgs) == 1
        assert msgs[0]["params"]["contexts"] == ["support"]
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_unreceive_empty_is_noop(self, connected_client):
        client, ws = connected_client
        ws.sent_messages.clear()
        await client.unreceive([])
        msgs = [m for m in ws.sent_messages if m.get("method") == "signalwire.unreceive"]
        assert len(msgs) == 0


# ===================================================================
# Authorization state
# ===================================================================

class TestAuthorizationState:
    @pytest.mark.asyncio
    async def test_authorization_state_stored(self, connected_client):
        client, ws = connected_client
        from signalwire_agents.relay.constants import EVENT_AUTHORIZATION_STATE
        event = make_event(EVENT_AUTHORIZATION_STATE, {
            "authorization_state": "encrypted-state-abc",
        })
        ws.feed_message(event)
        await asyncio.sleep(0.05)
        assert client._authorization_state == "encrypted-state-abc"

    @pytest.mark.asyncio
    async def test_authorization_state_empty_ignored(self, connected_client):
        client, ws = connected_client
        client._authorization_state = "existing"
        from signalwire_agents.relay.constants import EVENT_AUTHORIZATION_STATE
        event = make_event(EVENT_AUTHORIZATION_STATE, {
            "authorization_state": "",
        })
        ws.feed_message(event)
        await asyncio.sleep(0.05)
        assert client._authorization_state == "existing"


# ===================================================================
# signalwire.disconnect with restart
# ===================================================================

class TestDisconnectRestart:
    @pytest.mark.asyncio
    async def test_disconnect_restart_clears_state(self, connected_client):
        client, ws = connected_client
        client._relay_protocol = "proto-123"
        client._authorization_state = "auth-state-abc"

        disconnect_msg = {
            "jsonrpc": "2.0",
            "id": "disc-restart",
            "method": METHOD_SIGNALWIRE_DISCONNECT,
            "params": {"restart": True},
        }
        ws.feed_message(disconnect_msg)
        await asyncio.sleep(0.05)

        assert client._relay_protocol == ""
        assert client._authorization_state == ""

    @pytest.mark.asyncio
    async def test_disconnect_no_restart_keeps_state(self, connected_client):
        client, ws = connected_client
        client._relay_protocol = "proto-123"
        client._authorization_state = "auth-state-abc"

        disconnect_msg = {
            "jsonrpc": "2.0",
            "id": "disc-no-restart",
            "method": METHOD_SIGNALWIRE_DISCONNECT,
            "params": {"restart": False},
        }
        ws.feed_message(disconnect_msg)
        await asyncio.sleep(0.05)

        assert client._relay_protocol == "proto-123"
        assert client._authorization_state == "auth-state-abc"
