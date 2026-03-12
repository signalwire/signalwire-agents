# Events

RELAY events are server-pushed notifications about call state changes and operation results. Events arrive over the WebSocket as `signalwire.event` JSON-RPC messages and are automatically routed to the correct `Call` object.

## Listening for Events

### On a Call

```python
@client.on_call
async def handle(call):
    # Register a listener
    call.on("calling.call.play", lambda event: print(f"Play: {event.params}"))

    # Or wait for a specific event
    event = await call.wait_for("calling.call.state",
        predicate=lambda e: e.params.get("call_state") == "ended",
        timeout=60.0,
    )
```

### Via Actions

Actions returned by `play()`, `record()`, etc. have a `wait()` method that resolves when the operation completes:

```python
action = await call.play([{"type": "tts", "params": {"text": "Hello"}}])
event = await action.wait(timeout=30.0)
# event is a RelayEvent with the terminal state
```

## Event Types

All event type constants are importable from `signalwire_agents.relay`:

| Constant | Value | Description |
|----------|-------|-------------|
| `EVENT_CALL_STATE` | `calling.call.state` | Call state changes (created, ringing, answered, ending, ended) |
| `EVENT_CALL_RECEIVE` | `calling.call.receive` | Inbound call notification |
| `EVENT_CALL_PLAY` | `calling.call.play` | Play operation state changes |
| `EVENT_CALL_RECORD` | `calling.call.record` | Record operation state changes |
| `EVENT_CALL_COLLECT` | `calling.call.collect` | Input collection results |
| `EVENT_CALL_CONNECT` | `calling.call.connect` | Bridge/connect state changes |
| `EVENT_CALL_DETECT` | `calling.call.detect` | Detection results |
| `EVENT_CALL_FAX` | `calling.call.fax` | Fax operation state changes |
| `EVENT_CALL_TAP` | `calling.call.tap` | Tap operation state changes |
| `EVENT_CALL_STREAM` | `calling.call.stream` | Stream operation state changes |
| `EVENT_CALL_SEND_DIGITS` | `calling.call.send_digits` | DTMF send completion |
| `EVENT_CALL_DIAL` | `calling.call.dial` | Outbound dial progress |
| `EVENT_CALL_REFER` | `calling.call.refer` | SIP REFER results |
| `EVENT_CALL_DENOISE` | `calling.call.denoise` | Denoise state changes |
| `EVENT_CALL_PAY` | `calling.call.pay` | Payment state changes |
| `EVENT_CALL_QUEUE` | `calling.call.queue` | Queue state changes |
| `EVENT_CALL_ECHO` | `calling.call.echo` | Echo state changes |
| `EVENT_CALL_TRANSCRIBE` | `calling.call.transcribe` | Transcription state changes |
| `EVENT_CONFERENCE` | `calling.conference` | Conference state changes |
| `EVENT_CALLING_ERROR` | `calling.error` | Error events |
| `EVENT_MESSAGING_RECEIVE` | `messaging.receive` | Inbound message received |
| `EVENT_MESSAGING_STATE` | `messaging.state` | Outbound message state change |

## Typed Event Classes

Raw events are always `RelayEvent` with a `params` dict. For convenience, typed event classes provide named properties:

```python
from signalwire_agents.relay import CallStateEvent, PlayEvent, RecordEvent, parse_event

# Automatic parsing
event = parse_event(raw_payload)

# Or construct directly
if event.event_type == "calling.call.state":
    state_event = CallStateEvent.from_payload(raw_payload)
    print(state_event.call_state)   # "answered"
    print(state_event.end_reason)   # "hangup" (only on ended)
```

### Available Typed Events

| Class | Key Properties |
|-------|---------------|
| `CallStateEvent` | `call_state`, `end_reason`, `direction`, `device` |
| `CallReceiveEvent` | `call_state`, `direction`, `device`, `node_id`, `context`, `tag` |
| `PlayEvent` | `control_id`, `state` |
| `RecordEvent` | `control_id`, `state`, `url`, `duration`, `size` |
| `CollectEvent` | `control_id`, `state`, `result`, `final` |
| `ConnectEvent` | `connect_state`, `peer` |
| `DetectEvent` | `control_id`, `detect` |
| `FaxEvent` | `control_id`, `fax` |
| `TapEvent` | `control_id`, `state`, `tap`, `device` |
| `StreamEvent` | `control_id`, `state`, `url`, `name` |
| `SendDigitsEvent` | `control_id`, `state` |
| `DialEvent` | `tag`, `dial_state`, `call` |
| `ReferEvent` | `state`, `sip_refer_to`, `sip_refer_response_code` |
| `DenoiseEvent` | `denoised` |
| `PayEvent` | `control_id`, `state` |
| `QueueEvent` | `control_id`, `status`, `queue_id`, `queue_name`, `position`, `size` |
| `EchoEvent` | `state` |
| `TranscribeEvent` | `control_id`, `state`, `url`, `duration`, `size` |
| `HoldEvent` | `state` |
| `ConferenceEvent` | `conference_id`, `name`, `status` |
| `CallingErrorEvent` | `code`, `message` |
| `MessageReceiveEvent` | `message_id`, `context`, `direction`, `from_number`, `to_number`, `body`, `media`, `segments`, `message_state`, `tags` |
| `MessageStateEvent` | `message_id`, `context`, `direction`, `from_number`, `to_number`, `body`, `media`, `segments`, `message_state`, `reason`, `tags` |

## Call States

```
created -> ringing -> answered -> ending -> ended
```

Constants: `CALL_STATE_CREATED`, `CALL_STATE_RINGING`, `CALL_STATE_ANSWERED`, `CALL_STATE_ENDING`, `CALL_STATE_ENDED`

## End Reasons

When a call reaches the `ended` state, the `end_reason` field indicates why:

| Reason | Description |
|--------|-------------|
| `hangup` | Normal hangup |
| `cancel` | Caller cancelled |
| `busy` | Destination busy |
| `noAnswer` | No answer |
| `decline` | Call declined |
| `error` | Error occurred |
| `abandoned` | Call abandoned |
| `max_duration` | Max duration reached |
| `not_found` | Destination not found |

## Message States

Outbound messages progress through: `queued` → `initiated` → `sent` → `delivered` (or `undelivered`/`failed`).

Constants: `MESSAGE_STATE_QUEUED`, `MESSAGE_STATE_INITIATED`, `MESSAGE_STATE_SENT`, `MESSAGE_STATE_DELIVERED`, `MESSAGE_STATE_UNDELIVERED`, `MESSAGE_STATE_FAILED`, `MESSAGE_STATE_RECEIVED`
