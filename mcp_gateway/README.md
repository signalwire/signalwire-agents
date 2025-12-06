# MCP-SWAIG Gateway

A gateway service that bridges Model Context Protocol (MCP) servers with SignalWire AI Gateway (SWAIG) functions.

> **Note**: This is the quick start guide. For comprehensive technical documentation including architecture details, protocol specifications, and advanced configuration, see [docs/mcp_gateway_reference.md](../docs/mcp_gateway_reference.md).

## Installation

The MCP Gateway is included in the SignalWire Agents SDK. Install with the gateway dependencies:

```bash
pip install "signalwire-agents[mcp-gateway]"
```

Or if you have the SDK installed, add the gateway dependencies:

```bash
pip install flask flask-limiter
```

## Quick Start

1. **Start the Gateway**:
   ```bash
   # Using the installed CLI command
   mcp-gateway -c config.json

   # Or run directly (for development)
   cd mcp_gateway
   python3 gateway_service.py
   ```

2. **Test with curl**:
   ```bash
   ./test/test_gateway.sh
   ```

3. **Test with SignalWire Agent**:
   ```bash
   swaig-test mcp_gateway/test/test_agent.py --list-tools
   swaig-test mcp_gateway/test/test_agent.py --exec mcp_todo_add_todo --text "Test item"
   ```

## Features

- **Multi-Service Support**: Manage multiple MCP servers from one gateway
- **Session Management**: Automatic session lifecycle tied to SignalWire calls
- **HTTP/HTTPS**: Automatic HTTPS when SSL certificate is present
- **Authentication**: Basic Auth for secure access
- **Dynamic Discovery**: SWAIG functions created dynamically from MCP tools
- **Resource Limits**: Configurable session limits and timeouts

## Configuration

The gateway uses a `config.json` file with support for environment variable substitution. If not present, it will be created from `sample_config.json`.

### Method 1: Using Environment Variables (Recommended)

```bash
# Copy the example .env file
cp .env.example .env

# Edit with your values
vim .env

# Run the gateway (Docker Compose automatically reads .env)
./mcp-docker.sh start

# Or without Docker
source .env
python3 gateway_service.py
```

### Method 2: Direct Configuration

```bash
cp sample_config.json config.json
# Edit config.json with your settings
```

### Configuration Format

The configuration supports environment variable substitution using `${VAR_NAME|default}` syntax:

```json
{
  "server": {
    "host": "${MCP_HOST|0.0.0.0}",
    "port": "${MCP_PORT|8080}",
    "auth_user": "${MCP_AUTH_USER|admin}",
    "auth_password": "${MCP_AUTH_PASSWORD|changeme}",
    "auth_token": "${MCP_AUTH_TOKEN|}"
  },
  "services": {
    "todo": {
      "command": ["python3", "./test/todo_mcp.py"],
      "description": "Simple todo list for testing",
      "enabled": true
    }
  },
  "session": {
    "default_timeout": "${MCP_SESSION_TIMEOUT|300}",
    "max_sessions_per_service": "${MCP_MAX_SESSIONS|100}",
    "cleanup_interval": "${MCP_CLEANUP_INTERVAL|60}"
  },
  "logging": {
    "level": "${MCP_LOG_LEVEL|INFO}",
    "file": "${MCP_LOG_FILE|gateway.log}"
  }
}
```

### Supported Environment Variables

- `MCP_HOST`: Server bind address (default: 0.0.0.0)
- `MCP_PORT`: Server port (default: 8080)
- `MCP_AUTH_USER`: Basic auth username (default: admin)
- `MCP_AUTH_PASSWORD`: Basic auth password (default: changeme)
- `MCP_AUTH_TOKEN`: Bearer token for API access (optional)
- `MCP_SESSION_TIMEOUT`: Session timeout in seconds (default: 300)
- `MCP_MAX_SESSIONS`: Max sessions per service (default: 100)
- `MCP_CLEANUP_INTERVAL`: Session cleanup interval (default: 60)
- `MCP_LOG_LEVEL`: Logging level (default: INFO)
- `MCP_LOG_FILE`: Log file path (default: gateway.log)

## HTTPS Setup

To enable HTTPS, place your SSL certificate at `certs/server.pem`:

```bash
mkdir -p certs
# Copy your certificate
cp /path/to/server.pem certs/

# Or generate a self-signed certificate for testing
openssl req -x509 -newkey rsa:4096 -keyout certs/key.pem -out certs/cert.pem -days 365 -nodes
cat certs/cert.pem certs/key.pem > certs/server.pem
rm certs/cert.pem certs/key.pem
```

## API Documentation

### Health Check
```
GET /health
```

### Authentication

The gateway supports two authentication methods:

1. **Basic Auth** (default):
   ```
   Authorization: Basic <base64(username:password)>
   ```

2. **Bearer Token** (when MCP_AUTH_TOKEN is set):
   ```
   Authorization: Bearer <token>
   ```

### List Services
```
GET /services
Authorization: Basic <credentials> OR Bearer <token>
```

### Get Service Tools
```
GET /services/{service_name}/tools
Authorization: Basic <credentials> OR Bearer <token>
```

### Call Tool
```
POST /services/{service_name}/call
Authorization: Basic <credentials> OR Bearer <token>
Content-Type: application/json

{
  "tool": "tool_name",
  "arguments": {...},
  "session_id": "unique-session-id",
  "timeout": 300
}
```

### List Sessions
```
GET /sessions
Authorization: Basic <credentials> OR Bearer <token>
```

### Close Session
```
DELETE /sessions/{session_id}
Authorization: Basic <credentials> OR Bearer <token>
```

## Adding MCP Services

1. Add the service to `config.json`:
   ```json
   "my_service": {
     "command": ["python3", "/path/to/my_mcp_server.py"],
     "description": "My custom MCP service",
     "enabled": true
   }
   ```

2. Restart the gateway

3. The service will be available to SignalWire agents

## Using with SignalWire Agents

Add the MCP Gateway skill to your agent:

```python
from signalwire_agents import AgentBase

class MyAgent(AgentBase):
    def __init__(self):
        super().__init__(name="My Agent")
        
        self.add_skill("mcp_gateway", {
            "gateway_url": "http://localhost:8080",
            "auth_user": "admin",
            "auth_password": "changeme",
            "services": [
                {"name": "todo", "tools": "*"}
            ]
        })

agent = MyAgent()
agent.run()
```

## Docker Deployment

### Using the Helper Script (Recommended)

```bash
# Using .env file for configuration
./mcp-docker.sh start

# Or with environment variables
MCP_PORT=9000 ./mcp-docker.sh start

# Other commands
./mcp-docker.sh stop
./mcp-docker.sh restart
./mcp-docker.sh logs -f
./mcp-docker.sh status
```

### Manual Docker Commands

```bash
# Build
docker build -t mcp-gateway .

# Run with environment variables
docker run -p ${MCP_PORT:-8080}:${MCP_PORT:-8080} \
  -v $(pwd)/config.json:/app/config.json \
  -e MCP_PORT=8080 \
  mcp-gateway

# Or use docker-compose (reads .env automatically)
docker-compose up
```

**Note**: The Docker port mapping uses `${MCP_PORT:-8080}`, so ensure your MCP_PORT environment variable matches the port in your config.json.

## Troubleshooting

### MCP Server Won't Start
- Check the command path in config.json
- Verify the MCP server script is executable
- Check gateway.log for error messages

### Authentication Failures
- Verify credentials match between config.json and skill parameters
- Check that Basic Auth header is being sent

### Session Issues
- Increase timeout if sessions are expiring too quickly
- Check max_sessions_per_service limit
- Monitor active sessions with GET /sessions

### SSL Certificate Errors
- For self-signed certificates, set `verify_ssl: false` in skill config
- Ensure certificate file includes both cert and key

## Security Notes

1. Always use HTTPS in production
2. Change default credentials
3. Limit network access to the gateway
4. Monitor logs for suspicious activity
5. Set appropriate resource limits

## License

This is part of the SignalWire AI Agents SDK, licensed under the MIT License.