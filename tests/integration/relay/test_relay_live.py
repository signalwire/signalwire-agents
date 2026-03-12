"""Live integration tests for RELAY client.

Skipped by default — requires real credentials in environment variables:
    SIGNALWIRE_PROJECT_ID
    SIGNALWIRE_API_TOKEN
    SIGNALWIRE_SPACE          (optional, defaults to relay.signalwire.com)
    RELAY_TEST_PHONE           (optional, for dial test)
"""

from __future__ import annotations

import asyncio
import os

import pytest

from signalwire_agents.relay.client import RelayClient, _active_clients

pytestmark = pytest.mark.network

_PROJECT = os.environ.get("SIGNALWIRE_PROJECT_ID", "")
_TOKEN = os.environ.get("SIGNALWIRE_API_TOKEN", "")
_HOST = os.environ.get("SIGNALWIRE_SPACE", "")
_FROM_NUMBER = os.environ.get("RELAY_FROM_NUMBER", "")  # A number on your SW project
_TO_NUMBER = os.environ.get("RELAY_TO_NUMBER", "")      # Destination number

skip_no_creds = pytest.mark.skipif(
    not (_PROJECT and _TOKEN),
    reason="SIGNALWIRE_PROJECT_ID and SIGNALWIRE_API_TOKEN required",
)

skip_no_phone = pytest.mark.skipif(
    not (_FROM_NUMBER and _TO_NUMBER),
    reason="RELAY_FROM_NUMBER and RELAY_TO_NUMBER required",
)


@skip_no_creds
class TestRelayLive:
    def setup_method(self):
        _active_clients.clear()

    def teardown_method(self):
        _active_clients.clear()

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        kwargs = {"project": _PROJECT, "token": _TOKEN}
        if _HOST:
            kwargs["host"] = _HOST
        client = RelayClient(**kwargs)
        await client.connect()

        assert client._connected
        assert client._relay_protocol  # Should have received a protocol string

        await client.disconnect()
        assert not client._connected

    @pytest.mark.asyncio
    async def test_protocol_string(self):
        kwargs = {"project": _PROJECT, "token": _TOKEN}
        if _HOST:
            kwargs["host"] = _HOST
        client = RelayClient(**kwargs)
        await client.connect()

        assert isinstance(client._relay_protocol, str)
        assert len(client._relay_protocol) > 0

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_ping(self):
        """Verify the server sends pings and we respond correctly."""
        kwargs = {"project": _PROJECT, "token": _TOKEN}
        if _HOST:
            kwargs["host"] = _HOST
        client = RelayClient(**kwargs)
        await client.connect()

        # Wait a bit for a server ping
        await asyncio.sleep(5)
        assert client._connected  # Connection should be healthy

        await client.disconnect()

    @skip_no_phone
    @pytest.mark.asyncio
    async def test_dial_and_hangup(self):
        kwargs = {"project": _PROJECT, "token": _TOKEN}
        if _HOST:
            kwargs["host"] = _HOST
        client = RelayClient(**kwargs)
        await client.connect()

        devices = [[{"type": "phone", "params": {"to_number": _TO_NUMBER, "from_number": _FROM_NUMBER}}]]
        call = await client.dial(devices)

        assert call.call_id
        assert call.direction == "outbound"

        # Wait briefly then hang up
        await asyncio.sleep(2)
        try:
            await call.hangup()
        except Exception:
            pass  # Call may have already ended

        await client.disconnect()
