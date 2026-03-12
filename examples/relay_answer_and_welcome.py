"""Example: Answer an inbound call and say "Welcome to SignalWire!"

Set these env vars (or pass them directly to RelayClient):
  SIGNALWIRE_PROJECT_ID   - your SignalWire project ID
  SIGNALWIRE_API_TOKEN    - your SignalWire API token
  SIGNALWIRE_SPACE        - your SignalWire space (e.g. example.signalwire.com)

For full WebSocket / JSON-RPC debug output:
  SIGNALWIRE_LOG_LEVEL=debug
"""

import os

os.environ.setdefault("SIGNALWIRE_LOG_LEVEL", "debug")

from signalwire_agents.relay import RelayClient

client = RelayClient(contexts=["default"])


@client.on_call
async def on_incoming_call(call):
    print(f"Incoming call: {call}")
    await call.answer()

    action = await call.play([{"type": "tts", "params": {"text": "Welcome to SignalWire!"}}])
    await action.wait()

    await call.hangup()
    print(f"Call ended: {call.call_id}")


if __name__ == "__main__":
    print("Waiting for inbound calls on context 'default' ...")
    client.run()
