"""Tests for RelayClient dial() and event routing.

Focuses on the critical dial flow where calling.dial response has NO
call_id — the call info arrives via calling.call.state and
calling.call.dial events matched by tag.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from signalwire_agents.relay.client import RelayClient, RelayError
from signalwire_agents.relay.call import Call
from signalwire_agents.relay.constants import (
    EVENT_CALL_DIAL,
    EVENT_CALL_RECEIVE,
    EVENT_CALL_STATE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(**kwargs) -> RelayClient:
    """Create a RelayClient with mocked internals for unit testing."""
    client = RelayClient(project="test-proj", token="test-token", **kwargs)
    client._ws = MagicMock()
    client._ws.send = AsyncMock()
    client._connected = True
    client._relay_protocol = "test-protocol"
    return client


def _make_dial_event(tag: str, dial_state: str, call_id: str = "winner-call-id",
                     node_id: str = "winner-node-id") -> dict:
    """Build a calling.call.dial event payload."""
    return {
        "event_type": EVENT_CALL_DIAL,
        "params": {
            "node_id": node_id,
            "tag": tag,
            "dial_state": dial_state,
            "call": {
                "call_id": call_id,
                "node_id": node_id,
                "tag": tag,
                "device": {"type": "phone", "params": {
                    "from_number": "+15551234567",
                    "to_number": "+15559876543",
                }},
                "dial_winner": True,
            },
        },
    }


def _make_state_event(call_id: str, tag: str, call_state: str,
                      node_id: str = "node-1") -> dict:
    """Build a calling.call.state event payload."""
    return {
        "event_type": EVENT_CALL_STATE,
        "params": {
            "node_id": node_id,
            "call_id": call_id,
            "tag": tag,
            "call_state": call_state,
            "device": {"type": "phone", "params": {
                "from_number": "+15551234567",
                "to_number": "+15559876543",
            }},
        },
    }


# ---------------------------------------------------------------------------
# Tests for _handle_dial_event
# ---------------------------------------------------------------------------

class TestHandleDialEvent:
    @pytest.mark.asyncio
    async def test_dial_answered_resolves_future(self):
        client = _make_client()
        tag = "test-tag-1"

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        client._pending_dials[tag] = fut
        client._dial_calls_by_tag[tag] = []

        payload = _make_dial_event(tag, "answered")
        await client._handle_dial_event(payload)

        assert fut.done()
        call = fut.result()
        assert isinstance(call, Call)
        assert call.call_id == "winner-call-id"
        assert call.node_id == "winner-node-id"
        assert call.direction == "outbound"
        assert call.tag == tag
        # Call should be registered
        assert "winner-call-id" in client._calls

    @pytest.mark.asyncio
    async def test_dial_failed_rejects_future(self):
        client = _make_client()
        tag = "test-tag-2"

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        client._pending_dials[tag] = fut

        payload = _make_dial_event(tag, "failed")
        await client._handle_dial_event(payload)

        assert fut.done()
        with pytest.raises(RelayError, match="Dial failed"):
            fut.result()

    @pytest.mark.asyncio
    async def test_dial_dialing_does_not_resolve(self):
        client = _make_client()
        tag = "test-tag-3"

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        client._pending_dials[tag] = fut

        payload = _make_dial_event(tag, "dialing")
        await client._handle_dial_event(payload)

        assert not fut.done()

    @pytest.mark.asyncio
    async def test_dial_unknown_tag_ignored(self):
        client = _make_client()
        payload = _make_dial_event("unknown-tag", "answered")
        # Should not raise
        await client._handle_dial_event(payload)


# ---------------------------------------------------------------------------
# Tests for _handle_event routing with dial
# ---------------------------------------------------------------------------

class TestHandleEventDialRouting:
    @pytest.mark.asyncio
    async def test_state_event_creates_call_for_pending_dial(self):
        """calling.call.state events during dial create Call objects."""
        client = _make_client()
        tag = "dial-tag-1"

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        client._pending_dials[tag] = fut
        client._dial_calls_by_tag[tag] = []

        # Simulate a calling.call.state event with call_state: created
        payload = _make_state_event("leg-call-1", tag, "created")
        await client._handle_event(payload)

        # Call should be registered
        assert "leg-call-1" in client._calls
        call = client._calls["leg-call-1"]
        assert call.call_id == "leg-call-1"
        assert call.tag == tag
        assert call.direction == "outbound"
        assert call.state == "created"

    @pytest.mark.asyncio
    async def test_state_event_routes_to_existing_call(self):
        """Once a call is registered, state events route normally."""
        client = _make_client()
        tag = "dial-tag-2"

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        client._pending_dials[tag] = fut
        client._dial_calls_by_tag[tag] = []

        # First state event creates the call
        await client._handle_event(_make_state_event("leg-2", tag, "created"))
        call = client._calls["leg-2"]
        assert call.state == "created"

        # Second state event updates the call
        await client._handle_event(_make_state_event("leg-2", tag, "ringing"))
        assert call.state == "ringing"

    @pytest.mark.asyncio
    async def test_dial_event_uses_existing_call(self):
        """If the call was already created by state events, dial event reuses it."""
        client = _make_client()
        tag = "dial-tag-3"

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        client._pending_dials[tag] = fut
        client._dial_calls_by_tag[tag] = []

        # State events create the call
        await client._handle_event(_make_state_event("winner-id", tag, "created", "node-win"))
        await client._handle_event(_make_state_event("winner-id", tag, "answered", "node-win"))

        assert "winner-id" in client._calls

        # Dial answered event resolves with existing call
        payload = _make_dial_event(tag, "answered", call_id="winner-id", node_id="node-win")
        await client._handle_event(payload)

        assert fut.done()
        call = fut.result()
        assert call.call_id == "winner-id"
        # Should be the same object
        assert call is client._calls["winner-id"]

    @pytest.mark.asyncio
    async def test_ended_call_cleaned_up(self):
        """Calls that end during dial are cleaned up from _calls."""
        client = _make_client()
        tag = "dial-tag-4"

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        client._pending_dials[tag] = fut
        client._dial_calls_by_tag[tag] = []

        # Create a call leg that ends (loser in parallel dial)
        await client._handle_event(_make_state_event("loser-id", tag, "created"))
        assert "loser-id" in client._calls

        await client._handle_event(_make_state_event("loser-id", tag, "ended"))
        assert "loser-id" not in client._calls


# ---------------------------------------------------------------------------
# Tests for _handle_event with inbound calls
# ---------------------------------------------------------------------------

class TestHandleEventInbound:
    @pytest.mark.asyncio
    async def test_inbound_call_creates_call_and_invokes_handler(self):
        client = _make_client()
        received_calls = []

        @client.on_call
        async def handler(call):
            received_calls.append(call)

        payload = {
            "event_type": EVENT_CALL_RECEIVE,
            "params": {
                "call_id": "inbound-1",
                "node_id": "node-1",
                "call_state": "created",
                "context": "office",
                "device": {"type": "phone"},
            },
        }
        await client._handle_event(payload)

        # Give the handler task a chance to run
        await asyncio.sleep(0.01)

        assert "inbound-1" in client._calls
        assert len(received_calls) == 1
        assert received_calls[0].call_id == "inbound-1"


# ---------------------------------------------------------------------------
# Tests for event routing by call_id
# ---------------------------------------------------------------------------

class TestHandleEventCallIdRouting:
    @pytest.mark.asyncio
    async def test_event_routes_to_call_by_call_id(self):
        client = _make_client()

        # Pre-register a call
        call = Call(
            client=client,
            call_id="c1",
            node_id="n1",
            project_id="test-proj",
            context="test-protocol",
            state="answered",
        )
        client._calls["c1"] = call

        dispatched = []
        call.on("calling.call.play", lambda e: dispatched.append(e))

        await client._handle_event({
            "event_type": "calling.call.play",
            "params": {"call_id": "c1", "control_id": "ctl1", "state": "playing"},
        })

        assert len(dispatched) == 1

    @pytest.mark.asyncio
    async def test_unknown_call_id_ignored(self):
        client = _make_client()

        # Should not raise
        await client._handle_event({
            "event_type": "calling.call.play",
            "params": {"call_id": "unknown-id", "control_id": "ctl1", "state": "playing"},
        })


# ---------------------------------------------------------------------------
# Tests for _register_dial_leg
# ---------------------------------------------------------------------------

class TestRegisterDialLeg:
    @pytest.mark.asyncio
    async def test_creates_and_registers_call(self):
        client = _make_client()
        tag = "reg-tag-1"
        client._dial_calls_by_tag[tag] = []

        event_params = {
            "call_id": "new-leg-1",
            "node_id": "node-x",
            "call_state": "created",
            "device": {"type": "phone"},
        }
        call = client._register_dial_leg(tag, event_params)

        assert call.call_id == "new-leg-1"
        assert call.node_id == "node-x"
        assert call.tag == tag
        assert call.direction == "outbound"
        assert "new-leg-1" in client._calls
        assert call in client._dial_calls_by_tag[tag]


# ---------------------------------------------------------------------------
# Tests for disconnect cleanup
# ---------------------------------------------------------------------------

class TestDisconnectCleanup:
    @pytest.mark.asyncio
    async def test_disconnect_cancels_pending_dials(self):
        client = _make_client()
        client._ws.close = AsyncMock()
        loop = asyncio.get_running_loop()

        fut = loop.create_future()
        client._pending_dials["tag1"] = fut
        client._dial_calls_by_tag["tag1"] = []

        await client.disconnect()

        assert fut.cancelled()
        assert len(client._pending_dials) == 0
        assert len(client._dial_calls_by_tag) == 0

    @pytest.mark.asyncio
    async def test_clear_pending_rejects_dials(self):
        client = _make_client()
        loop = asyncio.get_running_loop()

        fut = loop.create_future()
        client._pending_dials["tag2"] = fut

        client._clear_pending_requests()

        assert fut.done()
        with pytest.raises(RelayError, match="Connection closed"):
            fut.result()
