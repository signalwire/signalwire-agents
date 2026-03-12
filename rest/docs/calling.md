# Calling Commands

The Calling API provides REST-based call control. All commands are dispatched via a single `POST /api/calling/calls` endpoint with a `command` field. No WebSocket connection is needed.

## How It Works

Every method on `client.calling` sends a POST request with this structure:

```json
{
    "command": "calling.play",
    "id": "<call-uuid>",
    "params": { ... }
}
```

For `dial` and `update`, the call details are inside `params` (no top-level `id`). For all other commands, `id` is the UUID of the call to control.

## Call Lifecycle

### `dial(**params) -> dict`

Initiate an outbound call.

```python
result = client.calling.dial(
    from_="+15559876543",
    to="+15551234567",
    url="https://example.com/call-handler",
)
call_id = result.get("id")
```

### `update(**params) -> dict`

Update an active call's dialplan mid-call.

```python
client.calling.update(id=call_id, url="https://example.com/new-handler")
```

### `end(call_id, **params) -> dict`

Terminate a call.

```python
client.calling.end(call_id, reason="hangup")
```

### `transfer(call_id, **params) -> dict`

Transfer a call to a new destination.

```python
client.calling.transfer(call_id, dest="sip:agent@example.com")
```

### `disconnect(call_id) -> dict`

Disconnect bridged calls without hanging up either leg.

```python
client.calling.disconnect(call_id)
```

## Audio Playback

### `play(call_id, **params) -> dict`

Play audio, TTS, silence, or ringtone.

```python
client.calling.play(call_id,
    play=[{"type": "tts", "text": "Hello!"}],
    volume=5.0,
)
```

### `play_pause(call_id, **params)` / `play_resume(call_id, **params)`

Pause or resume active playback.

```python
client.calling.play_pause(call_id, control_id="ctrl-1")
client.calling.play_resume(call_id, control_id="ctrl-1")
```

### `play_stop(call_id, **params)`

Stop active playback.

```python
client.calling.play_stop(call_id, control_id="ctrl-1")
```

### `play_volume(call_id, **params)`

Adjust playback volume.

```python
client.calling.play_volume(call_id, control_id="ctrl-1", volume=-3.0)
```

## Recording

### `record(call_id, **params)` / `record_pause` / `record_resume` / `record_stop`

```python
client.calling.record(call_id,
    control_id="rec-1",
    audio={"beep": True, "format": "wav", "stereo": True},
)
client.calling.record_pause(call_id, control_id="rec-1")
client.calling.record_resume(call_id, control_id="rec-1")
client.calling.record_stop(call_id, control_id="rec-1")
```

## Input Collection

### `collect(call_id, **params)` / `collect_stop` / `collect_start_input_timers`

```python
client.calling.collect(call_id,
    control_id="coll-1",
    digits={"max": 4, "terminators": "#"},
    speech={"end_silence_timeout": 2.0},
)
client.calling.collect_stop(call_id, control_id="coll-1")
client.calling.collect_start_input_timers(call_id, control_id="coll-1")
```

## Detection

### `detect(call_id, **params)` / `detect_stop`

```python
client.calling.detect(call_id,
    control_id="det-1",
    detect={"type": "machine", "params": {"initial_timeout": 4.5}},
)
client.calling.detect_stop(call_id, control_id="det-1")
```

## Tap & Stream

### `tap(call_id, **params)` / `tap_stop`

```python
client.calling.tap(call_id,
    control_id="tap-1",
    tap={"type": "audio", "params": {"direction": "both"}},
    device={"type": "rtp", "params": {"addr": "192.168.1.1", "port": 1234}},
)
client.calling.tap_stop(call_id, control_id="tap-1")
```

### `stream(call_id, **params)` / `stream_stop`

```python
client.calling.stream(call_id,
    control_id="str-1",
    url="wss://example.com/audio-stream",
    codec="PCMU",
)
client.calling.stream_stop(call_id, control_id="str-1")
```

## Denoise

### `denoise(call_id)` / `denoise_stop(call_id)`

```python
client.calling.denoise(call_id)
client.calling.denoise_stop(call_id)
```

## Transcription

### `transcribe(call_id, **params)` / `transcribe_stop`

```python
client.calling.transcribe(call_id, control_id="tx-1", status_url="https://example.com/hook")
client.calling.transcribe_stop(call_id, control_id="tx-1")
```

## AI

### `ai_message(call_id, **params)`

Inject a message into an active AI session.

```python
client.calling.ai_message(call_id, role="user", message_text="Transfer me to billing")
```

### `ai_hold(call_id, **params)` / `ai_unhold(call_id, **params)`

```python
client.calling.ai_hold(call_id, timeout=60, prompt="Please wait while I transfer you.")
client.calling.ai_unhold(call_id, prompt="I'm back, how can I help?")
```

### `ai_stop(call_id, **params)`

```python
client.calling.ai_stop(call_id, control_id="ai-1")
```

## Live Transcribe & Translate

```python
client.calling.live_transcribe(call_id, action="start", lang="en")
client.calling.live_translate(call_id, action="start", from_lang="en", to_lang="es")
```

## Fax

```python
client.calling.send_fax_stop(call_id, control_id="fax-1")
client.calling.receive_fax_stop(call_id, control_id="fax-1")
```

## SIP & Custom Events

```python
# SIP REFER transfer
client.calling.refer(call_id, device={"to": "sip:agent@example.com"})

# Custom event
client.calling.user_event(call_id, event={"type": "custom", "data": {"key": "value"}})
```

## Complete Method List

| Method | Command | Requires call_id |
|--------|---------|:-:|
| `dial(**params)` | `dial` | No |
| `update(**params)` | `update` | No |
| `end(call_id, **params)` | `calling.end` | Yes |
| `transfer(call_id, **params)` | `calling.transfer` | Yes |
| `disconnect(call_id)` | `calling.disconnect` | Yes |
| `play(call_id, **params)` | `calling.play` | Yes |
| `play_pause(call_id, **params)` | `calling.play.pause` | Yes |
| `play_resume(call_id, **params)` | `calling.play.resume` | Yes |
| `play_stop(call_id, **params)` | `calling.play.stop` | Yes |
| `play_volume(call_id, **params)` | `calling.play.volume` | Yes |
| `record(call_id, **params)` | `calling.record` | Yes |
| `record_pause(call_id, **params)` | `calling.record.pause` | Yes |
| `record_resume(call_id, **params)` | `calling.record.resume` | Yes |
| `record_stop(call_id, **params)` | `calling.record.stop` | Yes |
| `collect(call_id, **params)` | `calling.collect` | Yes |
| `collect_stop(call_id, **params)` | `calling.collect.stop` | Yes |
| `collect_start_input_timers(call_id, **params)` | `calling.collect.start_input_timers` | Yes |
| `detect(call_id, **params)` | `calling.detect` | Yes |
| `detect_stop(call_id, **params)` | `calling.detect.stop` | Yes |
| `tap(call_id, **params)` | `calling.tap` | Yes |
| `tap_stop(call_id, **params)` | `calling.tap.stop` | Yes |
| `stream(call_id, **params)` | `calling.stream` | Yes |
| `stream_stop(call_id, **params)` | `calling.stream.stop` | Yes |
| `denoise(call_id)` | `calling.denoise` | Yes |
| `denoise_stop(call_id)` | `calling.denoise.stop` | Yes |
| `transcribe(call_id, **params)` | `calling.transcribe` | Yes |
| `transcribe_stop(call_id, **params)` | `calling.transcribe.stop` | Yes |
| `ai_message(call_id, **params)` | `calling.ai_message` | Yes |
| `ai_hold(call_id, **params)` | `calling.ai_hold` | Yes |
| `ai_unhold(call_id, **params)` | `calling.ai_unhold` | Yes |
| `ai_stop(call_id, **params)` | `calling.ai.stop` | Yes |
| `live_transcribe(call_id, **params)` | `calling.live_transcribe` | Yes |
| `live_translate(call_id, **params)` | `calling.live_translate` | Yes |
| `send_fax_stop(call_id, **params)` | `calling.send_fax.stop` | Yes |
| `receive_fax_stop(call_id, **params)` | `calling.receive_fax.stop` | Yes |
| `refer(call_id, **params)` | `calling.refer` | Yes |
| `user_event(call_id, **params)` | `calling.user_event` | Yes |
