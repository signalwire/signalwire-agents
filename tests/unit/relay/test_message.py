"""Tests for RELAY messaging — Message class, send_message, inbound routing."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from signalwire_agents.relay.client import RelayClient, RelayError, _active_clients
from signalwire_agents.relay.message import Message
from signalwire_agents.relay.constants import (
    EVENT_MESSAGING_RECEIVE,
    EVENT_MESSAGING_STATE,
    MESSAGE_STATE_DELIVERED,
    MESSAGE_STATE_FAILED,
    MESSAGE_STATE_QUEUED,
    MESSAGE_STATE_SENT,
)

from .conftest import (
    AutoAuthMockWebSocket,
    make_event,
    make_jsonrpc_response,
)


# ---------------------------------------------------------------------------
# Message class unit tests
# ---------------------------------------------------------------------------

class TestMessage:
    """Tests for the Message data class and state tracking."""

    @pytest.mark.asyncio
    async def test_initial_state(self):
        msg = Message(
            message_id="msg-1",
            direction="outbound",
            from_number="+15551111111",
            to_number="+15552222222",
            body="Hello",
            state="queued",
        )
        assert msg.message_id == "msg-1"
        assert msg.state == "queued"
        assert msg.direction == "outbound"
        assert msg.body == "Hello"
        assert not msg.is_done
        assert msg.result is None

    @pytest.mark.asyncio
    async def test_dispatch_event_updates_state(self):
        msg = Message(message_id="msg-1", state="queued")
        await msg._dispatch_event({
            "event_type": EVENT_MESSAGING_STATE,
            "params": {"message_id": "msg-1", "message_state": "sent"},
        })
        assert msg.state == "sent"
        assert not msg.is_done

    @pytest.mark.asyncio
    async def test_dispatch_terminal_state_resolves(self):
        msg = Message(message_id="msg-1", state="queued")
        await msg._dispatch_event({
            "event_type": EVENT_MESSAGING_STATE,
            "params": {"message_id": "msg-1", "message_state": "delivered"},
        })
        assert msg.state == "delivered"
        assert msg.is_done
        assert msg.result is not None

    @pytest.mark.asyncio
    async def test_dispatch_failed_state(self):
        msg = Message(message_id="msg-1", state="queued")
        await msg._dispatch_event({
            "event_type": EVENT_MESSAGING_STATE,
            "params": {
                "message_id": "msg-1",
                "message_state": "failed",
                "reason": "spam",
            },
        })
        assert msg.state == "failed"
        assert msg.reason == "spam"
        assert msg.is_done

    @pytest.mark.asyncio
    async def test_wait_returns_terminal_event(self):
        msg = Message(message_id="msg-1", state="queued")

        async def deliver():
            await asyncio.sleep(0.01)
            await msg._dispatch_event({
                "event_type": EVENT_MESSAGING_STATE,
                "params": {"message_id": "msg-1", "message_state": "delivered"},
            })

        asyncio.ensure_future(deliver())
        event = await msg.wait(timeout=2.0)
        assert event.params["message_state"] == "delivered"

    @pytest.mark.asyncio
    async def test_wait_timeout(self):
        msg = Message(message_id="msg-1", state="queued")
        with pytest.raises(asyncio.TimeoutError):
            await msg.wait(timeout=0.01)

    @pytest.mark.asyncio
    async def test_on_completed_callback(self):
        results = []
        msg = Message(message_id="msg-1", state="queued")
        msg._on_completed = lambda event: results.append(event)

        await msg._dispatch_event({
            "event_type": EVENT_MESSAGING_STATE,
            "params": {"message_id": "msg-1", "message_state": "delivered"},
        })
        assert len(results) == 1
        assert results[0].params["message_state"] == "delivered"

    @pytest.mark.asyncio
    async def test_on_completed_async_callback(self):
        results = []
        msg = Message(message_id="msg-1", state="queued")

        async def on_done(event):
            results.append(event)

        msg._on_completed = on_done

        await msg._dispatch_event({
            "event_type": EVENT_MESSAGING_STATE,
            "params": {"message_id": "msg-1", "message_state": "delivered"},
        })
        await asyncio.sleep(0.01)  # let the ensure_future run
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_on_completed_error_is_caught(self):
        msg = Message(message_id="msg-1", state="queued")
        msg._on_completed = lambda event: 1 / 0  # raises ZeroDivisionError

        # Should not raise
        await msg._dispatch_event({
            "event_type": EVENT_MESSAGING_STATE,
            "params": {"message_id": "msg-1", "message_state": "delivered"},
        })
        assert msg.is_done

    @pytest.mark.asyncio
    async def test_listener_called_on_state_change(self):
        events = []
        msg = Message(message_id="msg-1", state="queued")
        msg.on(lambda event: events.append(event))

        await msg._dispatch_event({
            "event_type": EVENT_MESSAGING_STATE,
            "params": {"message_id": "msg-1", "message_state": "sent"},
        })
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_listener_error_is_caught(self):
        msg = Message(message_id="msg-1", state="queued")
        msg.on(lambda event: 1 / 0)

        # Should not raise
        await msg._dispatch_event({
            "event_type": EVENT_MESSAGING_STATE,
            "params": {"message_id": "msg-1", "message_state": "sent"},
        })
        assert msg.state == "sent"

    @pytest.mark.asyncio
    async def test_repr(self):
        msg = Message(
            message_id="msg-1",
            direction="outbound",
            state="queued",
            from_number="+15551111111",
            to_number="+15552222222",
        )
        r = repr(msg)
        assert "msg-1" in r
        assert "outbound" in r


# ---------------------------------------------------------------------------
# Client send_message tests
# ---------------------------------------------------------------------------

class TestSendMessage:
    """Tests for RelayClient.send_message()."""

    @pytest.mark.asyncio
    async def test_send_message_basic(self):
        _active_clients.clear()
        ws = AutoAuthMockWebSocket(auto_reply_all=True)

        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws):
            client = RelayClient(project="test-project", token="test-token")
            await client.connect()

            # Override auto_reply to include message_id
            original_send = ws.send

            async def custom_send(raw):
                await MockWebSocketSendBase.send(ws, raw)
                msg = json.loads(raw)
                if msg.get("method") == "messaging.send":
                    ws.feed_message(make_jsonrpc_response(msg["id"], {
                        "code": "200",
                        "message": "Message accepted",
                        "message_id": "msg-abc123",
                    }))
                elif msg.get("method") == "signalwire.connect":
                    ws.feed_message(make_jsonrpc_response(msg["id"], {
                        "protocol": "test-protocol",
                        "identity": "test-identity",
                    }))

            # Simpler approach: just use auto_reply_all which returns code 200
            message = await client.send_message(
                to_number="+15552222222",
                from_number="+15551111111",
                body="Hello World",
            )

            assert isinstance(message, Message)
            assert message.direction == "outbound"
            assert message.from_number == "+15551111111"
            assert message.to_number == "+15552222222"
            assert message.body == "Hello World"
            assert message.state == "queued"

            # Verify the RPC was sent correctly
            send_msgs = [m for m in ws.sent_messages if m.get("method") == "messaging.send"]
            assert len(send_msgs) == 1
            params = send_msgs[0]["params"]
            assert params["to_number"] == "+15552222222"
            assert params["from_number"] == "+15551111111"
            assert params["body"] == "Hello World"

            await client.disconnect()
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_send_message_with_media(self):
        _active_clients.clear()
        ws = AutoAuthMockWebSocket(auto_reply_all=True)

        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws):
            client = RelayClient(project="test-project", token="test-token")
            await client.connect()

            message = await client.send_message(
                to_number="+15552222222",
                from_number="+15551111111",
                media=["https://example.com/image.jpg"],
            )

            assert message.media == ["https://example.com/image.jpg"]

            send_msgs = [m for m in ws.sent_messages if m.get("method") == "messaging.send"]
            assert send_msgs[0]["params"]["media"] == ["https://example.com/image.jpg"]
            assert "body" not in send_msgs[0]["params"]

            await client.disconnect()
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_send_message_with_all_params(self):
        _active_clients.clear()
        ws = AutoAuthMockWebSocket(auto_reply_all=True)

        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws):
            client = RelayClient(project="test-project", token="test-token")
            await client.connect()

            message = await client.send_message(
                to_number="+15552222222",
                from_number="+15551111111",
                body="Hello",
                media=["https://example.com/image.jpg"],
                tags=["vip", "support"],
                region="us",
                context="my_context",
            )

            assert message.tags == ["vip", "support"]
            assert message.context == "my_context"

            send_msgs = [m for m in ws.sent_messages if m.get("method") == "messaging.send"]
            params = send_msgs[0]["params"]
            assert params["tags"] == ["vip", "support"]
            assert params["region"] == "us"
            assert params["context"] == "my_context"

            await client.disconnect()
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_send_message_requires_body_or_media(self):
        _active_clients.clear()
        ws = AutoAuthMockWebSocket(auto_reply_all=True)

        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws):
            client = RelayClient(project="test-project", token="test-token")
            await client.connect()

            with pytest.raises(ValueError, match="body or media"):
                await client.send_message(
                    to_number="+15552222222",
                    from_number="+15551111111",
                )

            await client.disconnect()
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_send_message_on_completed(self):
        _active_clients.clear()
        ws = AutoAuthMockWebSocket(auto_reply_all=True)

        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws):
            client = RelayClient(project="test-project", token="test-token")
            await client.connect()

            results = []
            message = await client.send_message(
                to_number="+15552222222",
                from_number="+15551111111",
                body="Hello",
                on_completed=lambda event: results.append(event),
            )

            assert message._on_completed is not None

            await client.disconnect()
        _active_clients.clear()


# ---------------------------------------------------------------------------
# Client event routing tests
# ---------------------------------------------------------------------------

class TestMessagingEventRouting:
    """Tests for messaging event dispatch in RelayClient."""

    @pytest.mark.asyncio
    async def test_inbound_message_routing(self):
        _active_clients.clear()
        ws = AutoAuthMockWebSocket(auto_reply_all=True)

        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws):
            client = RelayClient(project="test-project", token="test-token")

            received_messages = []

            @client.on_message
            async def handle_message(message):
                received_messages.append(message)

            await client.connect()

            # Inject an inbound message event
            ws.feed_message(make_event(EVENT_MESSAGING_RECEIVE, {
                "message_id": "msg-inbound-1",
                "context": "default",
                "direction": "inbound",
                "from_number": "+15553333333",
                "to_number": "+15551111111",
                "body": "Hi there",
                "media": [],
                "segments": 1,
                "message_state": "received",
            }))

            # Give the event loop time to process
            await asyncio.sleep(0.05)

            assert len(received_messages) == 1
            msg = received_messages[0]
            assert msg.message_id == "msg-inbound-1"
            assert msg.direction == "inbound"
            assert msg.from_number == "+15553333333"
            assert msg.body == "Hi there"
            assert msg.state == "received"
            assert msg.segments == 1

            await client.disconnect()
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_inbound_message_no_handler(self):
        """Inbound message with no handler should just log a warning."""
        _active_clients.clear()
        ws = AutoAuthMockWebSocket(auto_reply_all=True)

        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws):
            client = RelayClient(project="test-project", token="test-token")
            await client.connect()

            # No handler registered — should not crash
            ws.feed_message(make_event(EVENT_MESSAGING_RECEIVE, {
                "message_id": "msg-inbound-2",
                "from_number": "+15553333333",
                "to_number": "+15551111111",
                "body": "Hello",
                "message_state": "received",
            }))

            await asyncio.sleep(0.05)
            # No crash is the assertion

            await client.disconnect()
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_outbound_message_state_routing(self):
        """messaging.state events should route to tracked outbound messages."""
        _active_clients.clear()
        ws = AutoAuthMockWebSocket(auto_reply_all=True)

        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws):
            client = RelayClient(project="test-project", token="test-token")
            await client.connect()

            message = await client.send_message(
                to_number="+15552222222",
                from_number="+15551111111",
                body="Hello",
            )

            # The auto_reply_all doesn't return message_id, so manually register
            message.message_id = "msg-out-1"
            client._messages["msg-out-1"] = message

            # Send state updates
            ws.feed_message(make_event(EVENT_MESSAGING_STATE, {
                "message_id": "msg-out-1",
                "message_state": "sent",
            }))
            await asyncio.sleep(0.05)
            assert message.state == "sent"
            assert not message.is_done

            ws.feed_message(make_event(EVENT_MESSAGING_STATE, {
                "message_id": "msg-out-1",
                "message_state": "delivered",
            }))
            await asyncio.sleep(0.05)
            assert message.state == "delivered"
            assert message.is_done

            # Message should be cleaned up from tracking
            assert "msg-out-1" not in client._messages

            await client.disconnect()
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_state_event_for_unknown_message(self):
        """State event for unknown message_id should be ignored."""
        _active_clients.clear()
        ws = AutoAuthMockWebSocket(auto_reply_all=True)

        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws):
            client = RelayClient(project="test-project", token="test-token")
            await client.connect()

            ws.feed_message(make_event(EVENT_MESSAGING_STATE, {
                "message_id": "msg-unknown",
                "message_state": "delivered",
            }))
            await asyncio.sleep(0.05)
            # No crash is the assertion

            await client.disconnect()
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_message_handler_error_is_caught(self):
        """Error in on_message handler should be caught, not crash the loop."""
        _active_clients.clear()
        ws = AutoAuthMockWebSocket(auto_reply_all=True)

        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws):
            client = RelayClient(project="test-project", token="test-token")

            @client.on_message
            async def handle_message(message):
                raise RuntimeError("handler boom")

            await client.connect()

            ws.feed_message(make_event(EVENT_MESSAGING_RECEIVE, {
                "message_id": "msg-err",
                "from_number": "+15553333333",
                "to_number": "+15551111111",
                "body": "Hello",
                "message_state": "received",
            }))

            await asyncio.sleep(0.05)
            # No crash — handler error was caught

            await client.disconnect()
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_inbound_message_with_media_and_tags(self):
        """Verify all fields are populated on inbound messages."""
        _active_clients.clear()
        ws = AutoAuthMockWebSocket(auto_reply_all=True)

        with patch("signalwire_agents.relay.client.websockets.connect",
                   new_callable=AsyncMock, return_value=ws):
            client = RelayClient(project="test-project", token="test-token")

            received = []

            @client.on_message
            async def handle_message(message):
                received.append(message)

            await client.connect()

            ws.feed_message(make_event(EVENT_MESSAGING_RECEIVE, {
                "message_id": "msg-mms",
                "context": "support",
                "direction": "inbound",
                "from_number": "+15553333333",
                "to_number": "+15551111111",
                "body": "Check this out",
                "media": ["https://example.com/photo.jpg", "https://example.com/doc.pdf"],
                "segments": 2,
                "message_state": "received",
                "tags": ["vip"],
            }))

            await asyncio.sleep(0.05)
            msg = received[0]
            assert msg.media == ["https://example.com/photo.jpg", "https://example.com/doc.pdf"]
            assert msg.segments == 2
            assert msg.tags == ["vip"]
            assert msg.context == "support"

            await client.disconnect()
        _active_clients.clear()
