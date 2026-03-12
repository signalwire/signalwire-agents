# SignalWire RELAY Client

Real-time call control and messaging over WebSocket using Python's asyncio. The RELAY client connects to SignalWire via the Blade protocol (JSON-RPC 2.0 over WebSocket) and gives you imperative control over live phone calls and SMS/MMS messaging.

## Quick Start

```python
from signalwire_agents.relay import RelayClient

client = RelayClient(
    project="your-project-id",
    token="your-api-token",
    host="example.signalwire.com",
    contexts=["default"],
)

@client.on_call
async def handle(call):
    await call.answer()
    action = await call.play([{"type": "tts", "params": {"text": "Welcome to SignalWire!"}}])
    await action.wait()
    await call.hangup()

client.run()
```

## Features

- Asyncio-native with auto-reconnect and exponential backoff
- All 57+ calling methods: play, record, collect, connect, detect, fax, tap, stream, AI, conferencing, queues, and more
- SMS/MMS messaging: send outbound messages, receive inbound messages, track delivery state
- Action objects with `wait()`, `stop()`, `pause()`, `resume()` for controllable operations
- Typed event classes for all call events
- JWT and legacy authentication
- Dynamic context subscription/unsubscription
- Configurable concurrency limits

## Documentation

- [Getting Started](docs/getting-started.md) -- installation, configuration, first call
- [Call Methods Reference](docs/call-methods.md) -- every method available on a Call object
- [Events](docs/events.md) -- event types, typed event classes, call states
- [Messaging](docs/messaging.md) -- sending and receiving SMS/MMS messages
- [Client Reference](docs/client-reference.md) -- RelayClient configuration, methods, connection behavior

## Examples

- [relay_answer_and_welcome.py](examples/relay_answer_and_welcome.py) -- answer an inbound call and play a TTS greeting

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SIGNALWIRE_PROJECT_ID` | Project ID for authentication |
| `SIGNALWIRE_API_TOKEN` | API token for authentication |
| `SIGNALWIRE_JWT_TOKEN` | JWT token (alternative to project/token) |
| `SIGNALWIRE_SPACE` | Space hostname (default: `relay.signalwire.com`) |
| `RELAY_MAX_ACTIVE_CALLS` | Max concurrent calls per client (default: 1000) |
| `RELAY_MAX_CONNECTIONS` | Max WebSocket connections per process (default: 1) |
| `SIGNALWIRE_LOG_LEVEL` | Log level (`debug` for WebSocket traffic) |

## Module Structure

```
signalwire_agents/relay/
    __init__.py      # Public exports
    client.py        # RelayClient -- WebSocket connection, auth, event dispatch
    call.py          # Call object -- all calling methods and Action classes
    message.py       # Message object -- SMS/MMS message tracking
    event.py         # Typed event dataclasses
    constants.py     # Protocol constants, call states, event types
```
