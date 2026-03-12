# SignalWire RELAY Client Implementation Guide

This document is for AI coding agents implementing a RELAY client in any language. It describes the wire protocol, correlation mechanisms, and critical pitfalls that are not obvious from the spec alone.

## Protocol Overview

RELAY uses JSON-RPC 2.0 over WebSocket. The server is at `wss://<host>` (no path). All communication is async — you send requests, get responses matched by `id`, and receive server-pushed events via `signalwire.event`.

## Authentication

```json
{
  "jsonrpc": "2.0",
  "id": "<uuid>",
  "method": "signalwire.connect",
  "params": {
    "version": {"major": 2, "minor": 0, "revision": 0},
    "agent": "your-sdk-name/1.0",
    "event_acks": true,
    "authentication": {"project": "<project_id>", "token": "<token>"},
    "contexts": ["office", "support"]
  }
}
```

The response contains a `protocol` string — save it and send it back on reconnect to resume the session.

## Four Correlation Mechanisms

Every method uses one or more of these to bind requests to responses and events. Getting these wrong is the #1 source of bugs.

### 1. JSON-RPC `id` (ALL methods)

Every request has a UUID `id`. The server responds with the same `id`. This is how you match RPC responses to pending requests.

```
Client → {"jsonrpc":"2.0", "id":"abc-123", "method":"calling.answer", "params":{...}}
Server → {"jsonrpc":"2.0", "id":"abc-123", "result":{"code":"200", "message":"Answered"}}
```

Implementation: maintain a `pending: Map<string, Future>` keyed by request `id`.

### 2. `call_id` routing (all call-level methods)

Every call method sends `node_id` + `call_id` in params. Events come back with `call_id` in their params. Route events to the correct Call object by `call_id`.

```
Client → {"method":"calling.play", "params":{"node_id":"n1", "call_id":"c1", "control_id":"ctl1", ...}}
Server event → {"method":"signalwire.event", "params":{"event_type":"calling.call.play", "params":{"call_id":"c1", "control_id":"ctl1", "state":"finished"}}}
```

Implementation: maintain a `calls: Map<string, Call>` keyed by `call_id`.

### 3. `control_id` action tracking (12 methods)

Methods that start long-running operations (play, record, detect, etc.) take a client-generated `control_id`. The server echoes it back in events. Multiple actions can run concurrently on the same call — `control_id` disambiguates.

Implementation: each Call maintains `actions: Map<string, Action>` keyed by `control_id`.

### 4. `tag` correlation (dial only)

**CRITICAL**: `calling.dial` is the only method where the RPC response does NOT contain a `call_id`. The response is just `{"code":"200", "message":"Dialing"}`. The real `call_id` and `node_id` arrive asynchronously via events matched by `tag`.

Implementation: maintain `pending_dials: Map<string, Future<Call>>` keyed by `tag`.

## Method Categories

### Simple fire-and-response (no async tracking)

These methods send an RPC, get a response with `code`/`message`, and are done. No control_id, no ongoing events to track.

| Method | RPC | Response |
|--------|-----|----------|
| `answer` | `calling.answer` | `{"code":"200", "message":"Answered"}` |
| `end` | `calling.end` | `{"code":"200", "message":"Disconnecting call"}` |
| `pass` | `calling.pass` | `{"code":"200", "message":"Passing call"}` |
| `connect` | `calling.connect` | `{"code":"200", "message":"connecting"}` |
| `disconnect` | `calling.disconnect` | `{"code":"200", "message":"Disconnecting"}` |
| `hold` | `calling.hold` | `{"code":"200", "message":"Call on hold"}` |
| `unhold` | `calling.unhold` | `{"code":"200", "message":"Call off hold"}` |
| `denoise` | `calling.denoise` | `{"code":"200", "message":"Denoiser on"}` |
| `denoise.stop` | `calling.denoise.stop` | `{"code":"200", "message":"Denoiser off"}` |
| `transfer` | `calling.transfer` | `{"code":"200", "message":"Transferring"}` |
| `join_conference` | `calling.join_conference` | `{"code":"200", "message":"Joining conference"}` |
| `leave_conference` | `calling.leave_conference` | `{"code":"200", "message":"Leaving conference"}` |
| `echo` | `calling.echo` | `{"code":"200", "message":"Echo started"}` |
| `bind_digit` | `calling.bind_digit` | `{"code":"200", "message":"Digit binding created"}` |
| `clear_digit_bindings` | `calling.clear_digit_bindings` | `{"code":"200", "message":"Digit bindings cleared"}` |
| `live_transcribe` | `calling.live_transcribe` | `{"code":"200", "message":"Live transcription started"}` |
| `live_translate` | `calling.live_translate` | `{"code":"200", "message":"Live translation started"}` |
| `join_room` | `calling.join_room` | `{"code":"200", "message":"Joining room"}` |
| `leave_room` | `calling.leave_room` | `{"code":"200", "message":"Leaving room"}` |
| `amazon_bedrock` | `calling.amazon_bedrock` | `{"code":"200", "message":"AI started"}` |
| `ai_message` | `calling.ai_message` | `{"code":"200", "message":"Message sent"}` |
| `ai_hold` | `calling.ai_hold` | `{"code":"200", "message":"AI on hold"}` |
| `ai_unhold` | `calling.ai_unhold` | `{"code":"200", "message":"AI resumed"}` |
| `user_event` | `calling.user_event` | `{"code":"200", "message":"Event sent"}` |
| `queue.enter` | `calling.queue.enter` | `{"code":"200", "message":"Entering Queue"}` |
| `queue.leave` | `calling.queue.leave` | `{"code":"200", "message":"Leaving Queue"}` |
| `refer` | `calling.refer` | `{"code":"200", "message":"Starting SIP REFER"}` |
| `send_digits` | `calling.send_digits` | `{"code":"200", "message":"Sending Digits"}` |

Note: `connect`, `refer`, `send_digits` etc. DO produce async events (`calling.call.connect`, `calling.call.refer`, `calling.call.send_digits`) but these route normally by `call_id`. You don't need special tracking — just the standard event dispatch.

### control_id action methods (require action tracking)

These methods start a long-running operation. The client generates a `control_id` UUID, sends it in the request, and the server echoes it back in all related events. You MUST track these as Action objects to support `stop()`, `pause()`, `resume()`, `wait()`.

**Blocking vs fire-and-forget**: The `await call.play(...)` call only waits for the server to accept the command (the JSON-RPC response). The actual operation runs asynchronously on the server. The user chooses how to handle completion:

1. **Wait inline** (blocking): `await action.wait()` — blocks until the terminal event arrives
2. **Fire and forget** (background): don't call `action.wait()`, continue immediately. Check `action.is_done` or `action.result` later if needed
3. **Callback** (background + notification): pass `on_completed=callback` to the method. The callback fires when the action reaches a terminal state. Accepts both sync and async functions. Errors in callbacks are caught and logged.

The `on_completed` callback MUST also fire when the call is gone (404/410) — the action is resolved immediately with an empty event so the callback still runs. Implementations MUST support all three patterns on every action-based method.

| Method | RPC | Event Type | Terminal States |
|--------|-----|------------|-----------------|
| `play` | `calling.play` | `calling.call.play` | `finished`, `error` |
| `record` | `calling.record` | `calling.call.record` | `finished`, `no_input` |
| `detect` | `calling.detect` | `calling.call.detect` | `finished`, `error` |
| `collect` | `calling.collect` | `calling.call.collect` | `finished`, `error`, `no_input`, `no_match` |
| `play_and_collect` | `calling.play_and_collect` | `calling.call.collect` | `finished`, `error`, `no_input`, `no_match` |
| `pay` | `calling.pay` | `calling.call.pay` | `finished`, `error` |
| `send_fax` | `calling.send_fax` | `calling.call.fax` | `finished`, `error` |
| `receive_fax` | `calling.receive_fax` | `calling.call.fax` | `finished`, `error` |
| `tap` | `calling.tap` | `calling.call.tap` | `finished` |
| `stream` | `calling.stream` | `calling.call.stream` | `finished` |
| `transcribe` | `calling.transcribe` | `calling.call.transcribe` | `finished` |
| `ai` | `calling.ai` | N/A (ends on call end or stop) | `finished`, `error` |

Action sub-commands (these reference an existing `control_id`):
- `play.stop`, `play.pause`, `play.resume`, `play.volume`
- `record.stop`, `record.pause`, `record.resume`
- `detect.stop`
- `collect.stop`, `collect.start_input_timers`
- `play_and_collect.stop`, `play_and_collect.volume`
- `pay.stop`
- `send_fax.stop`, `receive_fax.stop`
- `tap.stop`
- `stream.stop`
- `transcribe.stop`
- `ai.stop`

#### play_and_collect gotcha

`play_and_collect` shares one `control_id` across both the play and collect phases. Events arrive as BOTH `calling.call.play` (for play state) and `calling.call.collect` (for collect result). Your CollectAction must filter by event_type — only resolve on `calling.call.collect` events, NOT on `calling.call.play` events with `state: finished` (that just means playback ended, not that input was collected).

#### detect gotcha

`detect` delivers results continuously via `calling.call.detect` events with a `detect` object. A detect action should resolve on the FIRST meaningful result (e.g., `HUMAN`, `MACHINE`, `CED`, digit) OR on terminal states `finished`/`error`. Don't wait only for `finished` — the useful data comes in intermediate events.

### tag-based methods (dial)

**This is where most implementations break.**

#### calling.dial — the async dance

```
Client → {"method":"calling.dial", "params":{"tag":"my-tag-123", "devices":[[...]]}}
Server → {"result":{"code":"200", "message":"Dialing"}}   ← NO call_id here!
```

After the RPC response, the server sends a sequence of events:

1. **`calling.call.state`** events for each call leg (one per device being dialed):
```json
{"event_type":"calling.call.state", "params":{
  "call_id":"leg-uuid-1", "node_id":"node-uuid", "tag":"my-tag-123",
  "call_state":"created", "device":{...}
}}
```
Then `ringing`, then `answered` or `ended` for each leg.

2. **`calling.call.dial`** event when the dial operation completes:
```json
{"event_type":"calling.call.dial", "params":{
  "node_id":"node-uuid",
  "tag":"my-tag-123",
  "dial_state":"answered",
  "call":{
    "call_id":"winner-uuid",
    "node_id":"node-uuid",
    "tag":"my-tag-123",
    "device":{...},
    "dial_winner": true
  }
}}
```

**CRITICAL DETAILS:**
- The `calling.call.dial` event has NO top-level `call_id` in params. The call info is nested inside `params.call.call_id`.
- `dial_state` values: `dialing` (progress), `answered` (success), `failed` (all legs failed).
- With parallel dialing, multiple call legs are created. Each gets `calling.call.state` events. Only the winner appears in the `calling.call.dial` event with `dial_state: "answered"`. Losers get `calling.call.state` with `call_state: "ended"`.

#### Correct dial() implementation

```python
async def dial(devices, tag=None, timeout=120):
    tag = tag or generate_uuid()

    # 1. Register pending dial BEFORE sending RPC
    future = create_future()
    pending_dials[tag] = future

    # 2. Send the RPC — response is just {"code":"200","message":"Dialing"}
    await execute("calling.dial", {"tag": tag, "devices": devices})

    # 3. Wait for calling.call.dial event to resolve the future
    try:
        call = await wait_for(future, timeout=timeout)
        return call
    finally:
        pending_dials.pop(tag)
```

#### Event routing during dial

Your event handler needs three special cases:

```python
def handle_event(payload):
    event_type = payload["event_type"]
    event_params = payload["params"]
    call_id = event_params.get("call_id", "")

    # 1. Inbound call
    if event_type == "calling.call.receive":
        handle_inbound(payload)
        return

    # 2. Dial completion — call_id is NESTED at params.call.call_id
    if event_type == "calling.call.dial":
        tag = event_params.get("tag", "")
        dial_state = event_params.get("dial_state", "")
        call_info = event_params.get("call", {})
        future = pending_dials.get(tag)
        if future:
            if dial_state == "answered":
                call = find_or_create_call(call_info)
                future.resolve(call)
            elif dial_state == "failed":
                future.reject(Error("Dial failed"))
        return

    # 3. State events during dial — call not registered yet
    if event_type == "calling.call.state":
        tag = event_params.get("tag", "")
        if tag in pending_dials and call_id not in calls:
            # Create the Call object so events route correctly
            register_dial_leg(tag, event_params)
        # Fall through to normal routing

    # 4. Normal routing by call_id
    call = calls.get(call_id)
    if call:
        call.dispatch_event(payload)
        if call.state == "ended":
            calls.pop(call_id)
```

### calling.begin (DEPRECATED)

Unlike `calling.dial`, the deprecated `calling.begin` DOES return `call_id` and `node_id` in the response:
```json
{"code":"200", "message":"Call started", "call_id":"<UUID>", "node_id":"<UUID>"}
```
Do NOT assume `calling.dial` works the same way. It doesn't.

## Event ACK

The server expects an acknowledgment for every `signalwire.event`:
```json
{"jsonrpc":"2.0", "id":"<event_msg_id>", "result":{}}
```
Send this immediately when you receive the event, before processing it.

## Server Pings

The server sends `signalwire.ping` periodically. Respond with:
```json
{"jsonrpc":"2.0", "id":"<ping_msg_id>", "result":{}}
```

## Error Code Handling

The calling API always returns results with `code` and `message`. The code is a STRING, not an integer.

- Any `2xx` code = success
- `404` = call does not exist
- `410` = call existed but is gone
- `409` = conflict (call already connected, controlled by another client)

### Call-gone handling (404/410)

When a caller hangs up, the server destroys the call. If your client tries to send a command (play, record, etc.) after the caller hung up but before you received the `calling.call.state ended` event, you get a 404 or 410 error. **This is normal telephony flow, not an exceptional condition.**

Handle it gracefully: log it and return an empty result instead of raising an exception. This avoids noisy errors in production when callers hang up mid-IVR.

```python
async def execute_on_call(method, params):
    try:
        return await execute(f"calling.{method}", params)
    except RelayError as e:
        if e.code in (404, 410):
            log.info(f"Call gone during {method}")
            return {}
        raise
```

Also: if an action (play, record, etc.) gets a call-gone response, resolve the action's Future immediately so `action.wait()` doesn't hang forever.

## Event Structure Reference

All events arrive as:
```json
{
  "jsonrpc": "2.0",
  "method": "signalwire.event",
  "id": "<msg_id>",
  "params": {
    "event_type": "calling.call.play",
    "timestamp": 123457.1234,
    "params": {
      "call_id": "...",
      "control_id": "...",
      "state": "finished"
    }
  }
}
```

Note the nested `params.params` structure. The outer `params` has `event_type`, the inner `params` has event-specific data like `call_id`, `control_id`, `state`.

### Events with non-standard structure

| Event | Gotcha |
|-------|--------|
| `calling.call.dial` | No top-level `call_id`. Has `tag`, `dial_state`, and `call` object containing `call_id`. |
| `calling.conference` | Uses `conference_id` instead of `call_id` for conference-level events. Participant events DO have `call_id`. |
| `calling.call.detect` | Results are in `detect.params.event`, not in a `state` field. Terminal state is `finished`. |
| `calling.call.collect` | Result object has `type` (digit/speech/error/no_input/no_match) and `params`. |
| `calling.call.record` | `url`, `duration`, `size` may be at top level OR nested inside `record` object. Check both. |

## Authorization State (Fast Reconnection)

The server sends `signalwire.authorization.state` events containing an encrypted `authorization_state` string. Store this and send it back on reconnect for fast re-auth without a full authentication round-trip.

```json
// Server sends:
{"method":"signalwire.event", "params":{
  "event_type":"signalwire.authorization.state",
  "params":{"authorization_state":"<encryptedBase64>:<tagBase64>"}
}}
```

On reconnect, include `authorization_state` in the `signalwire.connect` params alongside `protocol`. If the authorization state is invalid or expired, the server ignores it and falls back to normal `authentication`.

## Server-Initiated Disconnect

The server sends `signalwire.disconnect` to gracefully shut down connections (e.g. during deployments). The client MUST:

1. Respond with an empty result `{"jsonrpc":"2.0","id":"...","result":{}}`
2. Check the `restart` flag:
   - `restart: false` (or absent) → reconnect normally, reuse `protocol` and `authorization_state`
   - `restart: true` → clear `protocol` and `authorization_state`, reconnect with fresh auth

```json
{"method":"signalwire.disconnect", "params":{"restart": true}}
```

Do NOT set a "closing" flag — the client should reconnect after the server closes the socket.

## Dynamic Context Subscription

Contexts can be sent in `signalwire.connect` for initial subscription, but you can also add/remove contexts dynamically after connecting:

### signalwire.receive

Subscribe to additional contexts for inbound events:
```json
{"method":"signalwire.receive", "params":{"contexts":["sales","support"]}}
// Response: {"code":"200","message":"Receiving events"}
// Error: {"code":"402","message":"Payment required"}
```

### signalwire.unreceive

Unsubscribe from contexts:
```json
{"method":"signalwire.unreceive", "params":{"contexts":["sales"]}}
// Response: {"code":"200","message":"Unreceiving events"}
```

These are sent on the assigned protocol (the one returned from `signalwire.connect`), not the signalwire protocol.

## Reconnection

On WebSocket disconnect:
1. Reject all pending request Futures
2. Reject all pending dial Futures
3. Wait with exponential backoff (1s → 2s → 4s → ... → 30s max)
4. Reconnect and re-authenticate with `signalwire.connect`, sending:
   - The previous `protocol` string (to resume the session)
   - The `authorization_state` (for fast re-auth)
   - Unless a `signalwire.disconnect` with `restart: true` was received — then connect fresh
5. Flush any queued requests

Call objects survive reconnect — the server tracks them by `call_id` across connections.

## Architecture Checklist

- [ ] `pending` map: request `id` → Future (for RPC response matching)
- [ ] `calls` map: `call_id` → Call (for event routing)
- [ ] Each Call has `actions` map: `control_id` → Action (for action event routing)
- [ ] `pending_dials` map: `tag` → Future<Call> (for dial event matching)
- [ ] `authorization_state` stored from events and sent on reconnect
- [ ] Event ACK sent for every `signalwire.event`
- [ ] Pong sent for every `signalwire.ping`
- [ ] `signalwire.disconnect` handled: respond, check `restart` flag, reconnect
- [ ] `signalwire.receive`/`signalwire.unreceive` for dynamic context management
- [ ] 404/410 errors handled gracefully (call-gone, not exceptional)
- [ ] `calling.call.dial` events routed by `tag`, not `call_id`
- [ ] `calling.call.state` events during dial create Call objects before dial completes
- [ ] `play_and_collect` CollectAction filters by event_type, ignores play events
- [ ] Auto-reconnect with exponential backoff
- [ ] All pending futures cleaned up on disconnect
- [ ] Actions support three completion patterns: `await action.wait()` (blocking), fire-and-forget (don't await), `on_completed` callback
- [ ] `on_completed` callback supports both sync and async functions, errors caught and logged
- [ ] `on_completed` fires on call-gone (404/410) with resolved empty event
- [ ] `messages` map: `message_id` → Message (for messaging.state routing)
- [ ] `on_message` handler for inbound SMS/MMS (like `on_call` for calls)
- [ ] `send_message()` returns Message, tracks by `message_id`, supports `on_completed`

## Complete Method Parameter Reference

Every calling method is sent as a JSON-RPC request. Unless otherwise noted, all methods require `node_id` (UUID) and `call_id` (UUID) in params. The `project_id` and `protocol` fields are added by the client automatically.

### Media Objects (reused across methods)

Used in `play`, `play_and_collect`, and `connect` (ringback):

```
audio:    { type: "audio",    params: { url: "https://..." } }
tts:      { type: "tts",      params: { text: "...", language?: "en-US", gender?: "male"|"female", voice?: "..." } }
silence:  { type: "silence",  params: { duration: <seconds> } }
ringtone: { type: "ringtone", params: { name: "<country-code>", duration?: <seconds> } }
```

Ringtone country codes: `at au bg br be ch cl cn cz de dk ee es fi fr gr hu il in it lt jp mx my nl no nz ph pl pt ru se sg th uk us tw ve za`

### Device Objects (reused across dial/connect)

```
phone:  { type: "phone",  params: { from_number: "+1...", to_number: "+1...", timeout?: 30, max_duration?: <sec>, call_state_url?: "https://...", call_state_events?: ["created","ringing","answered","ended"], confirm?: "<swml>" } }
sip:    { type: "sip",    params: { from: "sip:...", to: "sip:...", from_name?: "...", timeout?: 30, max_duration?: <sec>, headers?: [{name,value}], codecs?: ["PCMU","PCMA","OPUS","G729","G722","VP8","H264"], webrtc_media?: false, call_state_url?: "...", call_state_events?: [...], confirm?: "..." } }
webrtc: { type: "webrtc", params: { from: "+1...", to: "resource-name", timeout?: 30, codecs?: ["PCMU","PCMA","OPUS","VP8","H264"], call_state_url?: "...", call_state_events?: [...], confirm?: "..." } }
agora:  { type: "agora",  params: { from: "+1...", to?: "...", appid: "...", channel: "...", call_state_url?: "...", call_state_events?: [...], confirm?: "..." } }
call:   { type: "call",   params: { node_id: "...", call_id: "..." } }        // connect only
queue:  { type: "queue",  params: { node_id: "...", queue_name: "...", queue_id?: "..." } }  // connect only
stream: { type: "stream", params: { url: "wss://...", name?: "...", codec?: "PCMU", status_url?: "...", status_url_method?: "POST", realtime?: false, authorization_bearer_token?: "...", custom_parameters?: {} } }  // connect only
```

### Collect Object (reused in collect and play_and_collect)

```
{
  initial_timeout?: 4.0,        // seconds, used when start_input_timers is true
  digits?: {
    max: <int>,                  // required
    terminators?: "#*",
    digit_timeout?: 5.0
  },
  speech?: {
    end_silence_timeout?: 1.0,
    speech_timeout?: 60.0,
    language?: "en-US",
    hints?: ["word1", "word2"],
    engine?: "Deepgram"|"Google"
  }
}
```

---

### calling.dial

Dial outbound call(s). First device to answer wins.

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `tag` | string | YES | Identifier for matching async events |
| `devices` | array[array[Device]] | YES | Outer=sequential, inner=parallel |
| `region` | string | no | Region to originate from |
| `max_price_per_minute` | number | no | Price cap |

Does NOT take `node_id`/`call_id`. Response has NO `call_id` — see tag correlation section above.

### calling.begin (DEPRECATED)

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `tag` | string | no | |
| `region` | string | no | |
| `device` | Device | YES | Single device (not array) |

Response DOES return `call_id` and `node_id`. Do NOT assume dial works the same.

### calling.answer

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `codecs` | string[] | no | e.g. ["PCMU","PCMA","OPUS","G729","G722","VP8","H264"] |

### calling.end

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `reason` | string | no | hangup\|cancel\|busy\|noAnswer\|decline\|error (default: hangup) |

### calling.pass

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |

Pass on inbound call offer. SignalWire offers it to another RELAY consumer.

### calling.connect

Connect another device to this active call.

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `devices` | array[array[Device]] | YES | Sequential/parallel device arrays |
| `ringback` | Media[] | no | What caller hears while connecting |
| `tag` | string | no | Tag for created calls' events |
| `max_duration` | number | no | Max connected duration in minutes |
| `max_price_per_minute` | number | no | |
| `status_url` | URL | no | |

### calling.disconnect

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |

Disconnects all connected calls without hanging up on them.

### calling.play

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `control_id` | string | YES | Client-generated UUID |
| `play` | Media[] | YES | Array of media objects |
| `volume` | float | no | -40 to +40 dB |
| `direction` | string | no | listen\|speak\|both (default: listen) |
| `loop` | int | no | 0=infinite, default 1 |
| `status_url` | URL | no | |

### calling.play.pause / calling.play.resume / calling.play.stop

| Param | Type | Required |
|-------|------|----------|
| `node_id` | UUID | YES |
| `call_id` | UUID | YES |
| `control_id` | string | YES |

### calling.play.volume

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `control_id` | string | YES | |
| `volume` | float | YES | -40 to +40 dB |

### calling.record

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `control_id` | string | YES | |
| `record` | object | YES | See below |
| `status_url` | URL | no | |

Record object (`record.audio`):
| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `beep` | bool | false | |
| `format` | string | "mp3" | mp3\|wav |
| `stereo` | bool | false | |
| `direction` | string | "speak" | listen\|speak\|both |
| `initial_timeout` | float | 5.0 | seconds, 0=disable |
| `end_silence_timeout` | float | 1.0 | seconds, 0=disable |
| `terminators` | string | "#*" | DTMF digits |
| `input_sensitivity` | int | 44 | 0-100 |

### calling.record.pause

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `control_id` | string | YES | |
| `behavior` | string | no | skip\|silence (default: skip) |

### calling.record.resume / calling.record.stop

| Param | Type | Required |
|-------|------|----------|
| `node_id` | UUID | YES |
| `call_id` | UUID | YES |
| `control_id` | string | YES |

### calling.collect

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `control_id` | string | YES | |
| `initial_timeout` | float | no | seconds, default 4.0 |
| `digits` | object | conditional | Required if no speech |
| `speech` | object | conditional | Required if no digits |
| `partial_results` | bool | no | default false |
| `continuous` | bool | no | default false |
| `send_start_of_input` | bool | no | default false |
| `start_input_timers` | bool | no | default false |
| `status_url` | URL | no | |

See Collect Object section above for digits/speech schemas.

### calling.collect.stop

| Param | Type | Required |
|-------|------|----------|
| `node_id` | UUID | YES |
| `call_id` | UUID | YES |
| `control_id` | string | YES |

### calling.collect.start_input_timers

| Param | Type | Required |
|-------|------|----------|
| `node_id` | UUID | YES |
| `call_id` | UUID | YES |
| `control_id` | string | YES |

Starts the `initial_timeout` timer on an active collect.

### calling.play_and_collect

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `control_id` | string | YES | |
| `play` | Media[] | YES | Array of media objects |
| `collect` | object | YES | See Collect Object section |
| `volume` | float | no | -40 to +40 dB |
| `status_url` | URL | no | |

### calling.play_and_collect.stop

| Param | Type | Required |
|-------|------|----------|
| `node_id` | UUID | YES |
| `call_id` | UUID | YES |
| `control_id` | string | YES |

### calling.play_and_collect.volume

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `control_id` | string | YES | |
| `volume` | float | YES | -40 to +40 dB |

### calling.detect

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `control_id` | string | YES | |
| `detect` | object | YES | See below |
| `timeout` | float | no | seconds, default 30.0 |
| `status_url` | URL | no | |

Detect object — type "machine":
| Param | Type | Default |
|-------|------|---------|
| `initial_timeout` | float | 4.5 |
| `end_silence_timeout` | float | 1.0 |
| `machine_ready_timeout` | float | =end_silence_timeout |
| `machine_voice_threshold` | float | 1.25 |
| `machine_words_threshold` | int | 6 |
| `detect_interruptions` | bool | false |
| `detect_message_end` | bool | true |

Detect object — type "fax":
| Param | Type | Default |
|-------|------|---------|
| `tone` | string | "CED" | CED\|CNG |

Detect object — type "digit":
| Param | Type | Default |
|-------|------|---------|
| `digits` | string | "0123456789#*" |

### calling.detect.stop

| Param | Type | Required |
|-------|------|----------|
| `node_id` | UUID | YES |
| `call_id` | UUID | YES |
| `control_id` | string | YES |

### calling.send_digits

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `control_id` | string | YES | |
| `digits` | string | YES | 0-9, #, *, A-D, w (0.5s wait), W (1s wait) |

### calling.tap

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `control_id` | string | YES | |
| `tap` | object | YES | `{type:"audio", params:{direction:"listen"\|"speak"\|"both"}}` |
| `device` | object | YES | See below |
| `status_url` | URL | no | |

Tap device — type "rtp":
| Param | Type | Required |
|-------|------|----------|
| `addr` | string (IPv4) | YES |
| `port` | int | YES |
| `codec` | string | no |
| `ptime` | int | no |

Tap device — type "ws":
| Param | Type | Required |
|-------|------|----------|
| `uri` | string | YES |
| `codec` | string | no |
| `rate` | int | no |

### calling.tap.stop

| Param | Type | Required |
|-------|------|----------|
| `node_id` | UUID | YES |
| `call_id` | UUID | YES |
| `control_id` | string | YES |

### calling.stream

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `control_id` | string | YES | |
| `url` | string | YES | wss:// WebSocket URI |
| `name` | string | no | Friendly name |
| `codec` | string | no | Default: call's native codec |
| `track` | string | no | inbound_track\|outbound_track\|both_tracks (default: inbound_track) |
| `status_url` | URL | no | |
| `status_url_method` | string | no | GET\|POST (default: POST) |
| `authorization_bearer_token` | string | no | |
| `custom_parameters` | object | no | |

### calling.stream.stop

| Param | Type | Required |
|-------|------|----------|
| `node_id` | UUID | YES |
| `call_id` | UUID | YES |
| `control_id` | string | YES |

### calling.transcribe

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `control_id` | string | YES | |
| `status_url` | URL | no | |

Only one active transcription per call (409 if already active).

### calling.transcribe.stop

| Param | Type | Required |
|-------|------|----------|
| `node_id` | UUID | YES |
| `call_id` | UUID | YES |
| `control_id` | string | YES |

### calling.send_fax

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `control_id` | string | YES | |
| `document` | URL | YES | URL to PDF |
| `identity` | string | no | Display identity on receiver |
| `header_info` | string | no | Custom header (default: "SignalWire", ""=disable) |
| `status_url` | URL | no | |

### calling.send_fax.stop / calling.receive_fax.stop

| Param | Type | Required |
|-------|------|----------|
| `node_id` | UUID | YES |
| `call_id` | UUID | YES |
| `control_id` | string | YES |

### calling.receive_fax

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `control_id` | string | YES | |
| `status_url` | URL | no | |

### calling.refer

SIP REFER transfer.

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `device` | object | YES | `{type:"sip", params:{to:"sip:...", username?:"...", password?:"..."}}` |
| `status_url` | URL | no | |

### calling.transfer

Transfer call control to another app.

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `dest` | string | YES | https:// URL, SWML script, or `context:<app>` |

### calling.join_conference

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `name` | string | YES | Conference name |
| `muted` | bool | no | default false |
| `beep` | string | no | true\|false\|onEnter\|onExit (default: "true") |
| `start_on_enter` | bool | no | default true |
| `end_on_exit` | bool | no | default false |
| `wait_url` | URL | no | CXML/mp3/wav hold music |
| `max_participants` | int | no | 1-250, default 250 |
| `record` | string | no | do-not-record\|record-from-start |
| `region` | string | no | global\|us\|eu |
| `trim` | string | no | trim-silence\|do-not-trim |
| `coach` | string | no | Call ID to coach |
| `status_callback` | URL | no | |
| `status_callback_event` | string | no | Space-separated: start end join leave mute hold modify speaker announcement |
| `status_callback_event_type` | string | no | relay\|cxml (default: relay) |
| `status_callback_method` | string | no | GET\|POST |
| `recording_status_callback` | URL | no | |
| `recording_status_callback_event` | string | no | in-progress\|completed\|absent |
| `recording_status_callback_event_type` | string | no | relay\|cxml |
| `recording_status_callback_method` | string | no | GET\|POST |
| `stream` | object | no | `{url:"wss://...", name?:"...", codec?:"...", ...}` |

### calling.leave_conference

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `conference_id` | UUID | YES | From conference events |

### calling.hold / calling.unhold

| Param | Type | Required |
|-------|------|----------|
| `node_id` | UUID | YES |
| `call_id` | UUID | YES |

Note: marked NOT IMPLEMENTED in spec.

### calling.denoise / calling.denoise.stop

| Param | Type | Required |
|-------|------|----------|
| `node_id` | UUID | YES |
| `call_id` | UUID | YES |

### calling.echo

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `timeout` | int | no | seconds, 0=until call ends |
| `status_url` | URL | no | |

### calling.bind_digit

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `digits` | string | YES | DTMF sequence to bind |
| `bind_method` | string | YES | Method to invoke |
| `params` | object | no | Params for bound method |
| `realm` | string | no | Namespace for selective clearing |
| `max_triggers` | int | no | 0=unlimited |

### calling.clear_digit_bindings

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `realm` | string | no | Only clear this realm |

### calling.queue.enter

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `control_id` | string | YES | |
| `queue_name` | string | YES | |
| `status_url` | URL | no | |

### calling.queue.leave

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `control_id` | string | YES | |
| `queue_name` | string | YES | |
| `queue_id` | string | no | |
| `status_url` | URL | no | |

### calling.ai

Start an AI agent on the call.

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `control_id` | string | YES | |
| `agent` | UUID | no | Pre-configured agent |
| `prompt` | object | no | `{text, top_p?, temperature?, confidence?, barge_confidence?, presence_penalty?, frequency_penalty?, model?}` |
| `post_prompt` | object | no | Same structure as prompt |
| `post_prompt_url` | URL | no | |
| `post_prompt_auth_user` | string | no | |
| `post_prompt_auth_password` | string | no | |
| `global_data` | object | no | Data for SWAIG functions |
| `pronounce` | object | no | Pronunciation rules |
| `hints` | object | no | Context hints |
| `languages` | object | no | Language configs |
| `SWAIG` | object | no | `{defaults?, functions?, includes?, native_functions?}` |
| `params` | object | no | ASR, TTS, turn detection, barge-in, LLM config, video, etc. |

### calling.ai.stop

| Param | Type | Required |
|-------|------|----------|
| `node_id` | UUID | YES |
| `call_id` | UUID | YES |
| `control_id` | string | YES |

### calling.amazon_bedrock

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `prompt` | string | no | System prompt |
| `SWAIG` | object | no | Function config |
| `params` | object | no | AI params |
| `global_data` | object | no | |
| `post_prompt` | object | no | |
| `post_prompt_url` | URL | no | |

### calling.ai_message

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `message_text` | string | no | Text to inject |
| `role` | string | no | Message role |
| `reset` | object | no | `{full_reset?, user_prompt?, system_prompt?}` |
| `global_data` | object | no | Updated global data |

### calling.ai_hold

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `timeout` | string/number | no | |
| `prompt` | string | no | Hold prompt/music |

### calling.ai_unhold

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `prompt` | string | no | Resume prompt |

### calling.user_event

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `event` | string | no | Event name |

### calling.live_transcribe

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `action` | object | YES | One key: `start{}`, `stop{}`, or `summarize{}` |

### calling.live_translate

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `action` | object | YES | One key: `start{}`, `stop{}`, `summarize{}`, or `inject{}` |
| `status_url` | URL | no | |

### calling.join_room

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `name` | string | YES | Room name |
| `status_url` | URL | no | |

### calling.leave_room

| Param | Type | Required |
|-------|------|----------|
| `node_id` | UUID | YES |
| `call_id` | UUID | YES |

### calling.pay

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `node_id` | UUID | YES | |
| `call_id` | UUID | YES | |
| `control_id` | string | YES | |
| `payment_connector_url` | URL | YES | |
| `input` | string | no | dtmf (only option) |
| `status_url` | URL | no | |
| `payment_method` | string | no | credit-card (only option) |
| `timeout` | int | no | seconds, default 5 |
| `max_attempts` | int | no | default 1 |
| `security_code` | bool | no | default true |
| `postal_code` | bool/string | no | true/false/known-postcode |
| `min_postal_code_length` | int | no | default 0 |
| `token_type` | string | no | one-time\|reusable (default: reusable) |
| `charge_amount` | string | no | Decimal, no currency prefix |
| `currency` | string | no | default "usd" |
| `language` | string | no | default "en-US" |
| `voice` | string | no | TTS voice, default "woman" |
| `description` | string | no | |
| `valid_card_types` | string | no | Space-separated: visa mastercard amex maestro discover jcb diners-club |
| `parameters` | array | no | `[{name, value}]` |
| `prompts` | array | no | See spec for prompt structure |

### calling.pay.stop

| Param | Type | Required |
|-------|------|----------|
| `node_id` | UUID | YES |
| `call_id` | UUID | YES |
| `control_id` | string | YES |

---

## Messaging Namespace

The messaging namespace is separate from calling. It handles SMS/MMS and uses different methods and event types. Messages are identified by `message_id` (not `call_id`), and there is no `node_id` or `control_id`.

### Correlation

Messages use a simpler model than calls:
1. **JSON-RPC `id`** — matches the RPC response (same as calling)
2. **`message_id`** — returned in the `messaging.send` response and echoed in all `messaging.state` events. Route state events to the correct Message object by `message_id`.

Implementation: maintain a `messages: Map<string, Message>` keyed by `message_id`.

### messaging.send

Send an outbound SMS/MMS.

| Param | Type | Required | Notes |
|-------|------|----------|-------|
| `context` | string | YES | Context for receiving state events |
| `to_number` | string | YES | E.164 format |
| `from_number` | string | YES | E.164 format |
| `body` | string | conditional | Required if no media |
| `media` | string[] | conditional | Required if no body — array of URLs |
| `tags` | string[] | no | Tags for searching in UI |
| `region` | string | no | Origination region |

```
Client → {"method":"messaging.send", "params":{
  "context": "my_context",
  "from_number": "+15551111111",
  "to_number": "+15552222222",
  "body": "Hello World",
  "media": ["https://example.com/image.jpg"],
  "tags": ["vip"]
}}
Server → {"result":{"code":"200", "message":"Message accepted", "message_id":"<UUID>"}}
```

**IMPORTANT**: The response includes `message_id`. Store this to route subsequent `messaging.state` events.

### Messaging Events

Events arrive via the same `signalwire.event` mechanism as calling events, but with `messaging.*` event types.

#### messaging.receive

Inbound message received.

| Field | Type | Description |
|-------|------|-------------|
| `message_id` | UUID | Message identifier |
| `context` | string | Context |
| `direction` | string | Always `"inbound"` |
| `from_number` | string | Sender (E.164) |
| `to_number` | string | Recipient (E.164) |
| `body` | string | Message body |
| `media` | string[] | Media URLs |
| `segments` | int | Number of segments |
| `message_state` | string | Always `"received"` |
| `tags` | string[] | Tags |

```json
{"method":"signalwire.event", "params":{
  "event_type":"messaging.receive",
  "params":{
    "message_id":"<UUID>",
    "context":"my_context",
    "direction":"inbound",
    "from_number":"+15553333333",
    "to_number":"+15551111111",
    "body":"Hello",
    "media":[],
    "segments":1,
    "message_state":"received"
  }
}}
```

#### messaging.state

Outbound message state change.

| Field | Type | Description |
|-------|------|-------------|
| `message_id` | UUID | Message identifier |
| `context` | string | Context |
| `direction` | string | Always `"outbound"` |
| `from_number` | string | Sender (E.164) |
| `to_number` | string | Recipient (E.164) |
| `body` | string | Message body |
| `media` | string[] | Media URLs |
| `segments` | int | Number of segments |
| `message_state` | string | See states below |
| `reason` | string | Present on `undelivered` and `failed` |
| `tags` | string[] | Tags |

Message states: `queued` → `initiated` → `sent` → `delivered` (terminal)

Failed paths: `queued` → `initiated` → `failed` or `undelivered` (terminal)

Terminal states: `delivered`, `undelivered`, `failed`

### Message Object Design

The Message object is simpler than Call — it's a data holder with state tracking:

```python
class Message:
    message_id: str
    context: str
    direction: str       # "inbound" or "outbound"
    from_number: str
    to_number: str
    body: str
    media: list[str]
    segments: int
    state: str           # current message_state
    reason: str          # failure reason (on failed/undelivered)
    tags: list[str]

    # Completion
    is_done: bool        # True when terminal state reached
    result: RelayEvent   # Terminal event (or None)

    async def wait(timeout=None) -> RelayEvent
    def on(handler)      # Register state change listener
```

The `on_completed` callback pattern from calling actions applies here too — pass it to `send_message()` and it fires when a terminal state is reached.

### Messaging Checklist

- [ ] `messages` map: `message_id` → Message (for state event routing)
- [ ] `on_message` handler for inbound messages (like `on_call` for calls)
- [ ] `send_message()` returns a Message object with the `message_id` from the response
- [ ] `messaging.receive` events create a Message and invoke the `on_message` handler
- [ ] `messaging.state` events route to tracked Message by `message_id`
- [ ] Message tracks state progression and resolves on terminal state (delivered/undelivered/failed)
- [ ] Terminal messages are cleaned up from the tracking map
- [ ] `on_completed` callback supported on `send_message()`
- [ ] At least one of `body` or `media` is required for `send_message()`
