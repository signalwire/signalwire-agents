# Messaging

Send and receive SMS/MMS messages through the RELAY client.

## Sending Messages

Use `client.send_message()` to send an outbound SMS or MMS.

```python
message = await client.send_message(
    to_number="+15552222222",
    from_number="+15551111111",
    body="Hello from SignalWire!",
)
```

### Wait for delivery

```python
message = await client.send_message(
    to_number="+15552222222",
    from_number="+15551111111",
    body="Hello!",
)
event = await message.wait()  # blocks until delivered/failed
print(f"Final state: {message.state}")
if message.reason:
    print(f"Reason: {message.reason}")
```

### Fire and forget

```python
message = await client.send_message(
    to_number="+15552222222",
    from_number="+15551111111",
    body="Hello!",
)
# don't call message.wait() — continue immediately
```

### Callback on completion

```python
message = await client.send_message(
    to_number="+15552222222",
    from_number="+15551111111",
    body="Hello!",
    on_completed=lambda event: print(f"Delivery: {event.params.get('message_state')}"),
)
```

### MMS (media messages)

```python
message = await client.send_message(
    to_number="+15552222222",
    from_number="+15551111111",
    body="Check this out!",
    media=["https://example.com/image.jpg"],
)
```

### All parameters

```python
message = await client.send_message(
    to_number="+15552222222",       # required — E.164 format
    from_number="+15551111111",     # required — E.164 format
    body="Message text",            # required if no media
    media=["https://..."],          # required if no body
    context="my_context",           # context for state events (default: relay protocol)
    tags=["vip", "support"],        # optional tags for searching in UI
    region="us",                    # optional origination region
    on_completed=callback_fn,       # optional completion callback
)
```

## Receiving Messages

Register a handler with `@client.on_message` to receive inbound SMS/MMS.

```python
from signalwire_agents.relay import RelayClient

client = RelayClient(
    project="your-project-id",
    token="your-api-token",
    host="example.signalwire.com",
    contexts=["default"],
)

@client.on_message
async def handle_message(message):
    print(f"From: {message.from_number}")
    print(f"To: {message.to_number}")
    print(f"Body: {message.body}")
    if message.media:
        print(f"Media: {message.media}")

    # Reply back
    await client.send_message(
        to_number=message.from_number,
        from_number=message.to_number,
        body=f"You said: {message.body}",
    )

client.run()
```

## Message Object

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `message_id` | `str` | Unique message identifier |
| `context` | `str` | Context the message belongs to |
| `direction` | `str` | `inbound` or `outbound` |
| `from_number` | `str` | Sender phone number (E.164) |
| `to_number` | `str` | Recipient phone number (E.164) |
| `body` | `str` | Text body of the message |
| `media` | `list[str]` | Media URLs (MMS) |
| `segments` | `int` | Number of message segments |
| `state` | `str` | Current message state |
| `reason` | `str` | Failure reason (on `undelivered` or `failed`) |
| `tags` | `list[str]` | Tags attached to the message |
| `is_done` | `bool` | `True` if message reached a terminal state |
| `result` | `RelayEvent` | Terminal event (or `None` if not done) |

### Methods

| Method | Description |
|--------|-------------|
| `await message.wait(timeout=None)` | Block until terminal state. Returns the terminal `RelayEvent`. |
| `message.on(handler)` | Register a listener for state change events. |

### Message States

Outbound messages progress through these states:

| State | Description |
|-------|-------------|
| `queued` | Message accepted and queued for sending |
| `initiated` | Sending has started |
| `sent` | Message sent to carrier |
| `delivered` | Message delivered to recipient (terminal) |
| `undelivered` | Delivery failed (terminal) — check `reason` |
| `failed` | Message failed to send (terminal) — check `reason` |

Inbound messages always arrive with state `received`.

## Event Types

| Event | Description |
|-------|-------------|
| `MessageReceiveEvent` | Inbound message received |
| `MessageStateEvent` | Outbound message state change |

```python
from signalwire_agents.relay import MessageReceiveEvent, MessageStateEvent
```

## Combining Calls and Messages

The same `RelayClient` handles both calls and messages:

```python
client = RelayClient(project="...", token="...", contexts=["default"])

@client.on_call
async def handle_call(call):
    await call.answer()
    await call.play([{"type": "tts", "params": {"text": "Hello!"}}])
    await call.hangup()

@client.on_message
async def handle_message(message):
    print(f"SMS from {message.from_number}: {message.body}")

client.run()
```
