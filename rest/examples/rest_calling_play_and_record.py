"""Example: Control an active call with media operations (play, record, transcribe, denoise).

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


def main():
    # 1. Dial an outbound call
    print("Dialing outbound call...")
    try:
        call = client.calling.dial(
            from_="+15559876543",
            to="+15551234567",
            url="https://example.com/call-handler",
        )
        call_id = call.get("id", "demo-call-id")
        print(f"  Call initiated: {call_id}")
    except SignalWireRestError as e:
        print(f"  Dial failed (expected in demo): {e.status_code}")
        call_id = "demo-call-id"

    # 2. Play TTS audio
    print("\nPlaying TTS on call...")
    try:
        client.calling.play(call_id, play=[{"type": "tts", "text": "Welcome to SignalWire."}])
        print("  Play started")
    except SignalWireRestError as e:
        print(f"  Play failed (expected in demo): {e.status_code}")

    # 3. Pause, resume, adjust volume, stop playback
    print("\nControlling playback...")
    for action, fn in [
        ("Pause", lambda: client.calling.play_pause(call_id)),
        ("Resume", lambda: client.calling.play_resume(call_id)),
        ("Volume +2dB", lambda: client.calling.play_volume(call_id, volume=2.0)),
        ("Stop", lambda: client.calling.play_stop(call_id)),
    ]:
        try:
            fn()
            print(f"  {action}: OK")
        except SignalWireRestError as e:
            print(f"  {action}: failed ({e.status_code})")

    # 4. Record the call
    print("\nRecording call...")
    try:
        client.calling.record(call_id, beep=True, format="mp3")
        print("  Recording started")
    except SignalWireRestError as e:
        print(f"  Record failed (expected in demo): {e.status_code}")

    # 5. Pause, resume, stop recording
    print("\nControlling recording...")
    for action, fn in [
        ("Pause", lambda: client.calling.record_pause(call_id)),
        ("Resume", lambda: client.calling.record_resume(call_id)),
        ("Stop", lambda: client.calling.record_stop(call_id)),
    ]:
        try:
            fn()
            print(f"  {action}: OK")
        except SignalWireRestError as e:
            print(f"  {action}: failed ({e.status_code})")

    # 6. Transcribe the call
    print("\nTranscribing call...")
    try:
        client.calling.transcribe(call_id, language="en-US")
        print("  Transcription started")
        client.calling.transcribe_stop(call_id)
        print("  Transcription stopped")
    except SignalWireRestError as e:
        print(f"  Transcribe failed (expected in demo): {e.status_code}")

    # 7. Denoise the call
    print("\nEnabling denoise...")
    try:
        client.calling.denoise(call_id)
        print("  Denoise started")
        client.calling.denoise_stop(call_id)
        print("  Denoise stopped")
    except SignalWireRestError as e:
        print(f"  Denoise failed (expected in demo): {e.status_code}")

    # 8. End the call
    print("\nEnding call...")
    try:
        client.calling.end(call_id, reason="hangup")
        print("  Call ended")
    except SignalWireRestError as e:
        print(f"  End call failed (expected in demo): {e.status_code}")


if __name__ == "__main__":
    main()
