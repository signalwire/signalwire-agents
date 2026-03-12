"""Example: IVR menu with DTMF collection, playback, and call connect.

Answers an inbound call, plays a greeting, collects a digit, and
routes the caller based on their choice:
  1 - Hear a sales message
  2 - Hear a support message
  0 - Connect to a live agent at +19184238080

Set these env vars (or pass them directly to RelayClient):
  SIGNALWIRE_PROJECT_ID   - your SignalWire project ID
  SIGNALWIRE_API_TOKEN    - your SignalWire API token
  SIGNALWIRE_SPACE        - your SignalWire space (optional, defaults to relay.signalwire.com)

For full WebSocket / JSON-RPC debug output:
  SIGNALWIRE_LOG_LEVEL=debug
"""

import os

os.environ.setdefault("SIGNALWIRE_LOG_LEVEL", "debug")

from signalwire_agents.relay import RelayClient

AGENT_NUMBER = "+19184238080"

client = RelayClient(contexts=["default"])


def tts(text):
    """Helper to build a TTS play element."""
    return {"type": "tts", "params": {"text": text}}


@client.on_call
async def on_incoming_call(call):
    print(f"Incoming call: {call}")
    await call.answer()

    # Play greeting and collect a single digit
    collect_action = await call.play_and_collect(
        media=[
            tts("Welcome to SignalWire!"),
            tts("Press 1 for sales. Press 2 for support. Press 0 to speak with an agent."),
        ],
        collect={
            "digits": {
                "max": 1,
                "digit_timeout": 5.0,
            },
            "initial_timeout": 10.0,
        },
    )

    result_event = await collect_action.wait()
    result = result_event.params.get("result", {})
    result_type = result.get("type", "")
    digits = result.get("params", {}).get("digits", "")

    print(f"Collect result: type={result_type} digits={digits}")

    if result_type == "digit" and digits == "1":
        # Sales
        action = await call.play([tts("Thank you for your interest! A sales representative will be with you shortly.")])
        await action.wait()

    elif result_type == "digit" and digits == "2":
        # Support
        action = await call.play([tts("Please hold while we connect you to our support team.")])
        await action.wait()

    elif result_type == "digit" and digits == "0":
        # Connect to live agent
        action = await call.play([tts("Connecting you to an agent now. Please hold.")])
        await action.wait()

        # Get the inbound caller's number from the device info
        from_number = call.device.get("params", {}).get("to_number", "")

        print(f"Connecting to {AGENT_NUMBER} from {from_number}")

        await call.connect(
            devices=[[
                {
                    "type": "phone",
                    "params": {
                        "to_number": AGENT_NUMBER,
                        "from_number": from_number,
                        "timeout": 30,
                    },
                }
            ]],
            ringback=[tts("Please wait while we connect your call.")],
        )

        # Stay on the call until the bridge ends
        await call.wait_for_ended()
        print(f"Connected call ended: {call.call_id}")
        return

    else:
        # No input or invalid
        action = await call.play([tts("We didn't receive a valid selection.")])
        await action.wait()

    await call.hangup()
    print(f"Call ended: {call.call_id}")


if __name__ == "__main__":
    print("Waiting for inbound calls on context 'default' ...")
    client.run()
