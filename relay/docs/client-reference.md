# RelayClient Reference

## Constructor

```python
RelayClient(
    project: str = None,          # SIGNALWIRE_PROJECT_ID
    token: str = None,            # SIGNALWIRE_API_TOKEN
    jwt_token: str = None,        # SIGNALWIRE_JWT_TOKEN
    host: str = None,             # SIGNALWIRE_SPACE (default: relay.signalwire.com)
    contexts: list[str] = None,   # Topics to subscribe to
    max_active_calls: int = None, # RELAY_MAX_ACTIVE_CALLS (default: 1000)
)
```

Authentication requires either `project` + `token` (legacy) or `jwt_token` (faster, no server roundtrip). All parameters fall back to their corresponding environment variables.

## Methods

### `run()`

Blocking entry point. Connects, authenticates, and runs the event loop with auto-reconnect until interrupted.

```python
client.run()
```

### `connect()` / `disconnect()`

Manual lifecycle control for use in async code.

```python
await client.connect()
# ... use client ...
await client.disconnect()
```

Also supports async context manager:

```python
async with RelayClient(contexts=["default"]) as client:
    ...
```

### `on_call(handler)`

Decorator to register the inbound call handler. The handler receives a `Call` object.

```python
@client.on_call
async def handle(call):
    await call.answer()
```

### `dial(devices, *, tag=None, max_duration=None, dial_timeout=None) -> Call`

Place an outbound call. Returns a `Call` once the remote party answers.

- `devices` -- nested list of device objects (serial/parallel dial)
- `tag` -- optional correlation tag (auto-generated if omitted)
- `max_duration` -- max call duration in minutes
- `dial_timeout` -- seconds to wait before raising `TimeoutError` (default: 120)

```python
call = await client.dial([
    [{"type": "phone", "params": {"to_number": "+15551234567", "from_number": "+15559876543"}}]
])
```

### `on_message(handler)`

Decorator to register the inbound message handler. The handler receives a `Message` object.

```python
@client.on_message
async def handle(message):
    print(f"SMS from {message.from_number}: {message.body}")
```

### `send_message(*, to_number, from_number, body=None, media=None, ...) -> Message`

Send an outbound SMS/MMS. Returns a `Message` that tracks delivery state.

```python
message = await client.send_message(
    to_number="+15552222222",
    from_number="+15551111111",
    body="Hello!",
)
event = await message.wait()  # block until delivered/failed
```

See [Messaging](messaging.md) for full details.

### `execute(method, params) -> dict`

Send a raw JSON-RPC request. Used internally by Call methods, but available for custom commands.

### `receive(contexts) / unreceive(contexts)`

Dynamically subscribe to or unsubscribe from contexts after connecting.

```python
await client.receive(["new-context"])
await client.unreceive(["old-context"])
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `relay_protocol` | `str` | Server-assigned protocol string from connect response |
| `project` | `str` | Project ID |
| `host` | `str` | Relay host |
| `contexts` | `list[str]` | Initial contexts |

## Connection Behavior

- **Auto-reconnect**: On connection loss, the client reconnects with exponential backoff (1s to 30s).
- **Ping/pong**: Client sends periodic pings and monitors server pings. After 3 consecutive failures, the connection is force-closed and reconnected.
- **Request queueing**: Requests made while disconnected are queued and sent after re-authentication.
- **Authorization state**: The server sends encrypted auth state via events. On reconnect, this is sent back for fast re-authentication without a full auth roundtrip.
- **Server disconnect**: The server can request a graceful disconnect (e.g. during deployment). The client auto-reconnects afterward.

## Concurrency

Each inbound call handler runs as an independent `asyncio.Task`, so multiple calls are handled concurrently. The `max_active_calls` parameter (default: 1000) caps concurrent calls to prevent unbounded memory growth.

For multiple WebSocket connections in one process, set `RELAY_MAX_CONNECTIONS` (default: 1).

## Error Handling

```python
from signalwire_agents.relay import RelayError

try:
    await call.play([...])
except RelayError as e:
    print(f"Error {e.code}: {e.message}")
```

`RelayError` is raised when the server returns a non-2xx response code. Errors 404 and 410 (call gone) are silently swallowed by Call methods since the call no longer exists.
