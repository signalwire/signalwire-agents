# Call Methods Reference

A `Call` object represents a live phone call. You get one from `@client.on_call` (inbound) or `client.dial()` (outbound).

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `call_id` | `str` | Unique call identifier |
| `node_id` | `str` | Server node handling the call |
| `state` | `str` | Current state: `created`, `ringing`, `answered`, `ending`, `ended` |
| `direction` | `str` | `inbound` or `outbound` |
| `tag` | `str` | Correlation tag |
| `device` | `dict` | Device info (type, params) |
| `segment_id` | `str` | Segment identifier |

## Actions: Blocking vs Fire-and-Forget

Methods like `play()`, `record()`, `detect()`, etc. return **Action** objects. The `await call.play(...)` itself only waits for the server to accept the command — the actual operation runs asynchronously on the server. You choose how to handle completion:

### Wait inline (blocking)

```python
action = await call.play([{"type": "tts", "params": {"text": "Hello"}}])
await action.wait()  # blocks until playback finishes
# execution continues only after play is done
```

### Fire and forget (background)

```python
action = await call.play([{"type": "tts", "params": {"text": "Hello"}}])
# don't call action.wait() — continue immediately while audio plays
await call.send_digits("1234")

# check later if needed
if action.is_done:
    print(f"Play result: {action.result}")
```

### Fire with callback

```python
# Sync callback
action = await call.play(
    [{"type": "tts", "params": {"text": "Hello"}}],
    on_completed=lambda event: print(f"Done: {event.params}"),
)
# continues immediately; callback fires when playback finishes

# Async callback
async def on_recording_done(event):
    print(f"Recording URL: {event.params.get('url')}")
    await call.hangup()

action = await call.record(on_completed=on_recording_done)
```

The `on_completed` callback is available on all action-based methods: `play`, `record`, `play_and_collect`, `collect`, `detect`, `pay`, `send_fax`, `receive_fax`, `tap`, `stream`, `transcribe`, and `ai`. It accepts both sync and async functions. Errors in callbacks are caught and logged, never crash the event loop. The callback also fires when the call is gone (404/410).

### Action methods summary

| Method | Returns |
|--------|---------|
| `action.wait(timeout=None)` | Blocks until the action completes, returns the terminal `RelayEvent` |
| `action.is_done` | `True` if the action has completed |
| `action.result` | The terminal `RelayEvent` (or `None` if not done) |
| `action.completed` | `True` if the action reached a terminal state |
| `action.stop()` | Stop the operation on the server |

Some actions also have `pause()`, `resume()`, and `volume()`.

## Lifecycle

### `answer(**kwargs) -> dict`

Answer an inbound call.

```python
await call.answer()
```

### `hangup(reason="hangup") -> dict`

End the call.

```python
await call.hangup()
await call.hangup(reason="busy")
```

### `pass_() -> dict`

Decline control, returning the call to routing.

```python
await call.pass_()
```

## Audio Playback

### `play(media, *, volume=None, direction=None, loop=None, control_id=None) -> PlayAction`

Play audio. Returns a `PlayAction` with `stop()`, `pause()`, `resume()`, `volume()`, and `wait()`.

```python
# TTS
action = await call.play([{"type": "tts", "params": {"text": "Hello!"}}])
await action.wait()

# Audio file
action = await call.play([{"type": "audio", "params": {"url": "https://example.com/sound.mp3"}}])

# Silence
action = await call.play([{"type": "silence", "params": {"duration": 2}}])

# Ringtone
action = await call.play([{"type": "ringtone", "params": {"name": "us"}}])

# Control playback
await action.pause()
await action.resume()
await action.volume(-3.0)
await action.stop()
```

## Recording

### `record(audio=None, *, control_id=None) -> RecordAction`

Record the call. Returns a `RecordAction` with `stop()`, `pause()`, `resume()`, and `wait()`.

```python
action = await call.record(audio={"format": "wav", "stereo": True, "direction": "both"})
# ... later ...
await action.stop()
event = await action.wait()
print(f"Recording URL: {event.params.get('url')}")
```

## Input Collection

### `play_and_collect(media, collect, *, volume=None, control_id=None) -> CollectAction`

Play audio and collect DTMF or speech input. Returns a `CollectAction`.

```python
action = await call.play_and_collect(
    [{"type": "tts", "params": {"text": "Press 1 for sales, 2 for support."}}],
    {"digits": {"max": 1, "digit_timeout": 5.0}},
)
event = await action.wait()
digit = event.params.get("result", {}).get("params", {}).get("digits", "")
```

### `collect(*, digits=None, speech=None, ..., control_id=None) -> StandaloneCollectAction`

Collect input without playing audio.

```python
action = await call.collect(
    digits={"max": 4, "terminators": "#"},
    speech={"language": "en-US"},
    partial_results=True,
)
event = await action.wait()
```

## Bridging

### `connect(devices, *, ringback=None, tag=None, max_duration=None, max_price_per_minute=None, status_url=None) -> dict`

Bridge the call to another destination.

```python
await call.connect(
    [[{"type": "phone", "params": {"to_number": "+15551234567", "from_number": "+15559876543"}}]],
    ringback=[{"type": "ringtone", "params": {"name": "us"}}],
)
```

### `disconnect() -> dict`

Unbridge a connected call.

```python
await call.disconnect()
```

## DTMF

### `send_digits(digits, *, control_id=None) -> dict`

Send DTMF tones.

```python
await call.send_digits("1234#")
```

## Detection

### `detect(detect, *, timeout=None, control_id=None) -> DetectAction`

Detect machine, fax, or digits.

```python
action = await call.detect({"type": "machine"}, timeout=30.0)
event = await action.wait()
```

## SIP Refer

### `refer(device, *, status_url=None) -> dict`

Transfer via SIP REFER.

```python
await call.refer({"type": "sip", "params": {"to": "sip:user@example.com"}})
```

## Transfer

### `transfer(dest) -> dict`

Transfer call control to another RELAY app or SWML script.

```python
await call.transfer("https://example.com/swml-endpoint")
```

## Fax

### `send_fax(document, *, identity=None, header_info=None, control_id=None) -> FaxAction`

```python
action = await call.send_fax("https://example.com/document.pdf", identity="+15551234567")
event = await action.wait()
```

### `receive_fax(*, control_id=None) -> FaxAction`

```python
action = await call.receive_fax()
event = await action.wait()
```

## Tap (Media Interception)

### `tap(tap, device, *, control_id=None) -> TapAction`

Intercept call media and stream to an RTP endpoint.

```python
action = await call.tap(
    {"type": "audio", "params": {"direction": "both"}},
    {"type": "rtp", "params": {"addr": "192.168.1.100", "port": 5000}},
)
```

## Streaming

### `stream(url, *, name=None, codec=None, track=None, control_id=None, ...) -> StreamAction`

Stream call audio to a WebSocket endpoint.

```python
action = await call.stream(
    "wss://example.com/audio",
    name="my_stream",
    codec="PCMU",
    track="inbound_track",
)
# Stop streaming
await action.stop()
```

## Payment

### `pay(payment_connector_url, *, control_id=None, charge_amount=None, currency=None, ...) -> PayAction`

Collect a payment via DTMF.

```python
action = await call.pay(
    "https://pay.example.com",
    charge_amount="25.99",
    currency="usd",
    input_method="dtmf",
)
event = await action.wait()
```

## Conference

### `join_conference(name, *, muted=None, beep=None, max_participants=None, record=None, ...) -> dict`

```python
await call.join_conference("my_conference", muted=False, beep="onEnter")
```

### `leave_conference(conference_id) -> dict`

```python
await call.leave_conference("conf-123")
```

## Hold

### `hold() -> dict` / `unhold() -> dict`

```python
await call.hold()
# ... later ...
await call.unhold()
```

## Denoise

### `denoise() -> dict` / `denoise_stop() -> dict`

```python
await call.denoise()
# ... later ...
await call.denoise_stop()
```

## Transcription

### `transcribe(*, control_id=None, status_url=None) -> TranscribeAction`

```python
action = await call.transcribe(status_url="https://example.com/transcription")
# ... later ...
await action.stop()
```

## Live Transcribe / Translate

### `live_transcribe(action_obj) -> dict`

```python
await call.live_transcribe({"start": {"language": "en-US"}})
```

### `live_translate(action_obj, *, status_url=None) -> dict`

```python
await call.live_translate({"start": {"source": "en-US", "target": "es"}})
```

## Echo

### `echo(*, timeout=None, status_url=None) -> dict`

Echo audio back to the caller (useful for testing).

```python
await call.echo(timeout=30.0)
```

## AI Agent

### `ai(*, control_id=None, prompt=None, SWAIG=None, ai_params=None, ...) -> AIAction`

Start an AI agent session on the call.

```python
action = await call.ai(
    prompt={"text": "You are a helpful support agent."},
    SWAIG={"functions": [...]},
    ai_params={"end_of_speech_timeout": 3000},
)
event = await action.wait()
```

### `amazon_bedrock(*, prompt=None, SWAIG=None, ...) -> dict`

Connect to an Amazon Bedrock AI agent.

### `ai_message(*, message_text=None, role=None, ...) -> dict`

Send a message to an active AI session.

### `ai_hold(*, timeout=None, prompt=None) -> dict` / `ai_unhold(*, prompt=None) -> dict`

Put an AI session on/off hold.

## Rooms

### `join_room(name, *, status_url=None) -> dict`

```python
await call.join_room("my_room")
```

### `leave_room() -> dict`

```python
await call.leave_room()
```

## Queue

### `queue_enter(queue_name, *, control_id=None, status_url=None) -> dict`

```python
await call.queue_enter("support")
```

### `queue_leave(queue_name, *, control_id=None, queue_id=None, status_url=None) -> dict`

```python
await call.queue_leave("support", queue_id="q-123")
```

## Digit Bindings

### `bind_digit(digits, bind_method, *, bind_params=None, realm=None, max_triggers=None) -> dict`

Bind a DTMF sequence to trigger a RELAY method.

```python
await call.bind_digit(
    "*1",
    "calling.play",
    bind_params={"play": [{"type": "tts", "params": {"text": "You pressed star-1"}}]},
)
```

### `clear_digit_bindings(*, realm=None) -> dict`

```python
await call.clear_digit_bindings()
```

## User Events

### `user_event(*, event=None, **kwargs) -> dict`

Send a custom event.

```python
await call.user_event(event="order_placed", order_id="12345")
```

## Event Handling

### `on(event_type, handler)`

Register an event listener on this call.

```python
def on_play(event):
    print(f"Play state: {event.params.get('state')}")

call.on("calling.call.play", on_play)
```

### `wait_for(event_type, predicate=None, timeout=None) -> RelayEvent`

Wait for a specific event.

```python
event = await call.wait_for("calling.call.play", timeout=30.0)
```

### `wait_for_ended(timeout=None) -> RelayEvent`

Wait for the call to end.

```python
event = await call.wait_for_ended()
print(f"End reason: {event.params.get('end_reason')}")
```
