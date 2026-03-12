#!/usr/bin/env python3
"""Dial a number and play 'Welcome to SignalWire' using the RELAY client.

Requires env vars:
    SIGNALWIRE_PROJECT_ID
    SIGNALWIRE_API_TOKEN
    RELAY_FROM_NUMBER   — a number on your SignalWire project
    RELAY_TO_NUMBER     — destination to call
"""

import asyncio
import os

from signalwire_agents.relay import RelayClient, CALL_STATE_ANSWERED, CALL_STATE_ENDED


async def main():
    from_number = os.environ["RELAY_FROM_NUMBER"]
    to_number = os.environ["RELAY_TO_NUMBER"]

    client = RelayClient()
    await client.connect()
    print(f"Connected — protocol: {client.relay_protocol}")

    # Dial the number
    devices = [[{"type": "phone", "params": {"to_number": to_number, "from_number": from_number}}]]
    call = await client.dial(devices)
    print(f"Dialing {to_number} from {from_number} — call_id: {call.call_id}")

    # Wait for the call to be answered
    try:
        await call.wait_for(
            "calling.call.state",
            predicate=lambda e: e.params.get("call_state") == CALL_STATE_ANSWERED,
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        print("No answer — timed out")
        await client.disconnect()
        return

    print("Call answered — playing TTS")

    # Play TTS
    play_action = await call.play([
        {"type": "tts", "params": {"text": "Welcome to SignalWire"}}
    ])

    # Wait for playback to finish
    await play_action.wait(timeout=15.0)
    print("Playback finished — hanging up")

    await call.hangup()
    await call.wait_for_ended(timeout=10.0)
    print("Call ended")

    await client.disconnect()
    print("Disconnected")


if __name__ == "__main__":
    asyncio.run(main())
