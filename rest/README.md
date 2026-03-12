# SignalWire REST Client

Synchronous REST client for managing SignalWire resources, controlling live calls, and interacting with every SignalWire API surface from Python. No WebSocket required -- just standard HTTP requests with automatic connection pooling.

## Quick Start

```python
from signalwire_agents.rest import SignalWireClient

client = SignalWireClient(
    project="your-project-id",
    token="your-api-token",
    host="example.signalwire.com",
)

# Create an AI agent
agent = client.fabric.ai_agents.create(
    name="Support Bot",
    prompt={"text": "You are a helpful support agent."},
)

# Search for a phone number
results = client.phone_numbers.search(area_code="512")

# Place a call via REST
client.calling.dial(
    from_="+15559876543",
    to="+15551234567",
    url="https://example.com/call-handler",
)
```

## Features

- Single `SignalWireClient` with namespaced sub-objects for every API
- All 37 calling commands: dial, play, record, collect, detect, tap, stream, AI, transcribe, and more
- Full Fabric API: 13 resource types with CRUD + addresses, tokens, and generic resources
- Datasphere: document management and semantic search
- Video: rooms, sessions, recordings, conferences, tokens, streams
- Compatibility API: full Twilio-compatible LAML surface
- Phone number management, 10DLC registry, MFA, logs, and more
- Shared `requests.Session` for connection pooling across all calls
- Dict returns -- raw JSON, no wrapper objects to learn

## Documentation

- [Getting Started](docs/getting-started.md) -- installation, configuration, first API call
- [Client Reference](docs/client-reference.md) -- SignalWireClient constructor, namespaces, error handling
- [Fabric Resources](docs/fabric.md) -- managing AI agents, SWML scripts, subscribers, call flows, and more
- [Calling Commands](docs/calling.md) -- REST-based call control (dial, play, record, collect, AI, etc.)
- [Compatibility API](docs/compat.md) -- Twilio-compatible LAML endpoints
- [All Namespaces](docs/namespaces.md) -- phone numbers, video, datasphere, logs, registry, and more

## Examples

- [rest_manage_resources.py](examples/rest_manage_resources.py) -- create an AI agent, assign a phone number, and place a test call
- [rest_datasphere_search.py](examples/rest_datasphere_search.py) -- upload a document and run a semantic search

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SIGNALWIRE_PROJECT_ID` | Project ID for authentication |
| `SIGNALWIRE_API_TOKEN` | API token for authentication |
| `SIGNALWIRE_SPACE` | Space hostname (e.g. `example.signalwire.com`) |
| `SIGNALWIRE_LOG_LEVEL` | Log level (`debug` for HTTP request details) |

## Module Structure

```
signalwire_agents/rest/
    __init__.py          # Public exports: SignalWireClient, SignalWireRestError
    client.py            # SignalWireClient -- namespace wiring, env var resolution
    _base.py             # HttpClient, BaseResource, CrudResource, CrudWithAddresses
    _pagination.py       # PaginatedIterator for list endpoints
    namespaces/
        fabric.py        # 13 resource types + generic resources + addresses + tokens
        calling.py       # 37 command dispatch methods via single POST
        phone_numbers.py # Search, purchase, update, release
        compat.py        # Twilio-compatible LAML API
        video.py         # Rooms, sessions, recordings, conferences
        datasphere.py    # Documents, search, chunks
        ... and 15 more
```
