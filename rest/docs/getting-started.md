# Getting Started with the REST Client

The REST client provides synchronous access to all SignalWire APIs using standard HTTP requests. No WebSocket connection required.

## Installation

The REST client is included in the `signalwire-agents` package:

```bash
pip install signalwire-agents
```

The only additional dependency is `requests`, which is installed automatically.

## Configuration

You need three things to connect:

| Parameter | Env Var | Description |
|-----------|---------|-------------|
| `project` | `SIGNALWIRE_PROJECT_ID` | Your SignalWire project ID |
| `token` | `SIGNALWIRE_API_TOKEN` | Your SignalWire API token |
| `host` | `SIGNALWIRE_SPACE` | Your space hostname (e.g. `example.signalwire.com`) |

## Minimal Example

```python
from signalwire_agents.rest import SignalWireClient

client = SignalWireClient(
    project="your-project-id",
    token="your-api-token",
    host="example.signalwire.com",
)

# List your AI agents
agents = client.fabric.ai_agents.list()
print(agents)
```

Or use environment variables and skip the constructor args:

```bash
export SIGNALWIRE_PROJECT_ID=your-project-id
export SIGNALWIRE_API_TOKEN=your-api-token
export SIGNALWIRE_SPACE=example.signalwire.com
```

```python
from signalwire_agents.rest import SignalWireClient

client = SignalWireClient()
agents = client.fabric.ai_agents.list()
```

## CRUD Pattern

Most resources follow the same CRUD pattern:

```python
# List
items = client.fabric.ai_agents.list()

# Create
agent = client.fabric.ai_agents.create(name="Support", prompt={"text": "Be helpful"})

# Get by ID
agent = client.fabric.ai_agents.get("agent-uuid")

# Update
client.fabric.ai_agents.update("agent-uuid", name="Updated Name")

# Delete
client.fabric.ai_agents.delete("agent-uuid")
```

Fabric resources also support listing addresses:

```python
addresses = client.fabric.ai_agents.list_addresses("agent-uuid")
```

## Error Handling

```python
from signalwire_agents.rest import SignalWireClient, SignalWireRestError

client = SignalWireClient()

try:
    agent = client.fabric.ai_agents.get("nonexistent-id")
except SignalWireRestError as e:
    print(f"HTTP {e.status_code}: {e.body}")
    # HTTP 404: {'error': 'not found'}
```

## Debug Logging

Set the log level to see HTTP request details:

```bash
export SIGNALWIRE_LOG_LEVEL=debug
```

## Next Steps

- [Client Reference](client-reference.md) -- all namespaces and constructor options
- [Fabric Resources](fabric.md) -- managing AI agents, SWML scripts, and more
- [Calling Commands](calling.md) -- REST-based call control
- [All Namespaces](namespaces.md) -- phone numbers, video, datasphere, and more
