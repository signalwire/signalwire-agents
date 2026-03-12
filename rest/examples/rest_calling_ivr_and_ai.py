"""Example: IVR input collection, AI operations, and advanced call control.

NOTE: These commands require an active call. The call_id used here is
illustrative — in production you would obtain it from a dial response or
inbound call event.

Set these env vars (or pass them directly to SignalWireClient):
  SIGNALWIRE_PROJECT_ID   - your SignalWire project ID
  SIGNALWIRE_API_TOKEN    - your SignalWire API token
  SIGNALWIRE_SPACE        - your SignalWire space (e.g. example.signalwire.com)

For full HTTP debug output:
  SIGNALWIRE_LOG_LEVEL=debug
"""

from signalwire_agents.rest import SignalWireClient, SignalWireRestError

client = SignalWireClient()

CALL_ID = "demo-call-id"


def safe(label, fn):
    """Execute a calling command, handling expected demo failures."""
    try:
        result = fn()
        print(f"  {label}: OK")
        return result
    except SignalWireRestError as e:
        print(f"  {label}: failed ({e.status_code})")
        return None


def main():
    # 1. Collect DTMF input
    print("Collecting DTMF input...")
    safe("Collect", lambda: client.calling.collect(
        CALL_ID,
        digits={"max": 4, "terminators": "#"},
        play=[{"type": "tts", "text": "Enter your PIN followed by pound."}],
    ))
    safe("Start input timers", lambda: client.calling.collect_start_input_timers(CALL_ID))
    safe("Stop collect", lambda: client.calling.collect_stop(CALL_ID))

    # 2. Answering machine detection
    print("\nDetecting answering machine...")
    safe("Detect", lambda: client.calling.detect(CALL_ID, type="machine"))
    safe("Stop detect", lambda: client.calling.detect_stop(CALL_ID))

    # 3. AI operations
    print("\nAI agent operations...")
    safe("AI message", lambda: client.calling.ai_message(
        CALL_ID,
        message="The customer wants to check their balance.",
    ))
    safe("AI hold", lambda: client.calling.ai_hold(CALL_ID))
    safe("AI unhold", lambda: client.calling.ai_unhold(CALL_ID))
    safe("AI stop", lambda: client.calling.ai_stop(CALL_ID))

    # 4. Live transcription and translation
    print("\nLive transcription and translation...")
    safe("Live transcribe", lambda: client.calling.live_transcribe(
        CALL_ID, language="en-US",
    ))
    safe("Live translate", lambda: client.calling.live_translate(
        CALL_ID, language="es",
    ))

    # 5. Tap (media fork)
    print("\nTap (media fork)...")
    safe("Tap start", lambda: client.calling.tap(
        CALL_ID,
        tap={"type": "audio", "direction": "both"},
        device={"type": "rtp", "addr": "192.168.1.100", "port": 9000},
    ))
    safe("Tap stop", lambda: client.calling.tap_stop(CALL_ID))

    # 6. Stream (WebSocket)
    print("\nStream (WebSocket)...")
    safe("Stream start", lambda: client.calling.stream(
        CALL_ID, url="wss://example.com/audio-stream",
    ))
    safe("Stream stop", lambda: client.calling.stream_stop(CALL_ID))

    # 7. User event
    print("\nSending user event...")
    safe("User event", lambda: client.calling.user_event(
        CALL_ID, event_name="agent_note", data={"note": "VIP caller"},
    ))

    # 8. SIP refer
    print("\nSIP refer...")
    safe("SIP refer", lambda: client.calling.refer(
        CALL_ID, sip_uri="sip:support@example.com",
    ))

    # 9. Fax stop commands
    print("\nFax stop commands...")
    safe("Send fax stop", lambda: client.calling.send_fax_stop(CALL_ID))
    safe("Receive fax stop", lambda: client.calling.receive_fax_stop(CALL_ID))

    # 10. Transfer and disconnect
    print("\nTransfer and disconnect...")
    safe("Transfer", lambda: client.calling.transfer(
        CALL_ID, dest="+15559999999",
    ))
    safe("Update call", lambda: client.calling.update(
        call_id=CALL_ID, metadata={"priority": "high"},
    ))
    safe("Disconnect", lambda: client.calling.disconnect(CALL_ID))


if __name__ == "__main__":
    main()
