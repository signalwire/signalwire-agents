# MCP to SWAIG Gateway

## Overview

The MCP-SWAIG Gateway bridges Model Context Protocol (MCP) servers with SignalWire AI Gateway (SWAIG) functions, allowing SignalWire AI agents to interact with MCP-based tools. This gateway acts as a translation layer and session manager between the two protocols.

## Installation

The MCP Gateway is included in the SignalWire Agents SDK. Install with the gateway dependencies:

```bash
pip install "signalwire-agents[mcp-gateway]"
```

Once installed, the `mcp-gateway` CLI command is available:

```bash
mcp-gateway -c config.json
```

## Architecture

### Components

1. **MCP Gateway Service** (`mcp_gateway/`)
   - HTTP/HTTPS server with Basic Authentication
   - Manages multiple MCP server instances
   - Handles session lifecycle per SignalWire call
   - Translates between SWAIG and MCP protocols

2. **MCP Gateway Skill** (`signalwire_agents/skills/mcp_gateway/`)
   - SignalWire skill that connects agents to the gateway
   - Dynamically creates SWAIG functions from MCP tools
   - Manages session lifecycle using call_id

3. **Test MCP Server** (`mcp_gateway/test/todo_mcp.py`)
   - Simple todo list MCP server for testing
   - Demonstrates stateful MCP server implementation

## Protocol Flow

```
SignalWire Agent                 Gateway Service              MCP Server
      |                                |                          |
      |---(1) Add Skill--------------->|                          |
      |<--(2) Query Tools--------------|                          |
      |                                |---(3) List Tools-------->|
      |                                |<--(4) Tool List----------|
      |---(5) Call SWAIG Function----->|                          |
      |                                |---(6) Spawn Session----->|
      |                                |---(7) Call MCP Tool----->|
      |                                |<--(8) MCP Response-------|
      |<--(9) SWAIG Response-----------|                          |
      |                                |                          |
      |---(10) Hangup Hook------------>|                          |
      |                                |---(11) Close Session---->|
```

## Message Envelope Format

The gateway uses a custom envelope format for routing and session management:

```json
{
    "session_id": "call_xyz123",  // From SWAIG call_id
    "service": "todo",             // MCP service name
    "tool": "add_todo",           // Tool name
    "arguments": {                 // Tool arguments
        "text": "Buy milk"
    },
    "timeout": 300,               // Session timeout in seconds
    "metadata": {                 // Optional metadata
        "agent_id": "agent_123",
        "timestamp": "2024-01-20T10:30:00Z"
    }
}
```

## Directory Structure

```
signalwire_agents/mcp_gateway/    # Core gateway package (installed with SDK)
├── __init__.py                   # Package exports
├── gateway_service.py            # Main HTTP/HTTPS server
├── mcp_manager.py                # MCP server lifecycle management
└── session_manager.py            # Session handling and timeouts

mcp_gateway/                      # Configuration and deployment files
├── config.json                   # Gateway configuration
├── sample_config.json            # Example configuration
├── Dockerfile                    # Docker container definition
├── docker-compose.yml            # Docker compose configuration
├── mcp-docker.sh                 # Docker management helper script
├── README.md                     # Gateway documentation
├── certs/                        # SSL certificates (optional)
│   └── .gitignore                # Ignore actual certificates
├── test/
│   ├── todo_mcp.py               # Test MCP server
│   ├── test_gateway.sh           # Curl test scripts
│   └── test_agent.py             # Test SignalWire agent
└── examples/
    └── generate_cert.sh          # Generate self-signed certificate

signalwire_agents/skills/mcp_gateway/
├── __init__.py
├── skill.py                      # MCP gateway skill
└── README.md                     # Skill documentation
```

## Configuration

### Gateway Configuration (`config.json`)

The configuration supports environment variable substitution using `${VAR_NAME|default}` syntax:

```json
{
    "server": {
        "host": "${MCP_HOST|0.0.0.0}",
        "port": "${MCP_PORT|8080}",
        "auth_user": "${MCP_AUTH_USER|admin}",
        "auth_password": "${MCP_AUTH_PASSWORD|changeme}",
        "auth_token": "${MCP_AUTH_TOKEN|optional-bearer-token}"
    },
    "services": {
        "todo": {
            "command": ["python3", "./test/todo_mcp.py"],
            "description": "Simple todo list for testing",
            "enabled": true,
            "sandbox": {
                "enabled": true,
                "resource_limits": true,
                "restricted_env": true
            }
        },
        "shell-mpc": {
            "command": ["python3", "/path/to/shell_mpc.py"],
            "description": "Shell PTY access",
            "enabled": false,
            "sandbox": {
                "enabled": false,
                "note": "Shell access needs full filesystem"
            }
        },
        "calculator": {
            "command": ["node", "/path/to/calculator.js"],
            "description": "Math calculations",
            "enabled": true,
            "sandbox": {
                "enabled": true,
                "resource_limits": true,
                "restricted_env": false,
                "note": "Needs NODE_PATH but can have resource limits"
            }
        }
    },
    "session": {
        "default_timeout": 300,
        "max_sessions_per_service": 100,
        "cleanup_interval": 60,
        "sandbox_dir": "./sandbox"
    },
    "rate_limiting": {
        "default_limits": ["200 per day", "50 per hour"],
        "tools_limit": "30 per minute",
        "call_limit": "10 per minute",
        "session_delete_limit": "20 per minute",
        "storage_uri": "memory://"
    },
    "logging": {
        "level": "INFO",
        "file": "gateway.log"
    }
}
```

### Environment Variable Substitution

The gateway supports environment variable substitution in config.json using the format `${VAR_NAME|default_value}`.

Example usage:

**Method 1: Using .env file (recommended)**
```bash
# Copy the example
cp .env.example .env

# Edit with your values
vim .env

# Run - Docker Compose automatically reads .env
./mcp-docker.sh start

# Or for non-Docker
source .env
python3 gateway_service.py
```

**Method 2: Export environment variables**
```bash
# Set environment variables
export MCP_PORT=9000
export MCP_AUTH_PASSWORD=mysecret

# Run the gateway
python3 gateway_service.py
```

**Method 3: Inline variables**
```bash
# Set variables for just this command
MCP_PORT=9000 MCP_AUTH_PASSWORD=mysecret ./mcp-docker.sh start
```

Supported variables:
- `MCP_HOST`: Server bind address (default: 0.0.0.0)
- `MCP_PORT`: Server port (default: 8080)
- `MCP_AUTH_USER`: Basic auth username (default: admin)
- `MCP_AUTH_PASSWORD`: Basic auth password (default: changeme)
- `MCP_AUTH_TOKEN`: Bearer token for API access (default: empty)
- `MCP_SESSION_TIMEOUT`: Session timeout in seconds (default: 300)
- `MCP_MAX_SESSIONS`: Max sessions per service (default: 100)
- `MCP_CLEANUP_INTERVAL`: Session cleanup interval in seconds (default: 60)
- `MCP_LOG_LEVEL`: Logging level (default: INFO)
- `MCP_LOG_FILE`: Log file path (default: gateway.log)

### Sandbox Configuration Options

Each service can have its own sandbox configuration:

| Option | Default | Description |
|--------|---------|-------------|
| `enabled` | `true` | Enable/disable sandboxing completely |
| `resource_limits` | `true` | Apply CPU, memory, process limits |
| `restricted_env` | `true` | Use minimal environment variables |
| `working_dir` | Current dir | Working directory for the process |
| `allowed_paths` | N/A | Future: Path access restrictions |

#### Sandbox Profiles

1. **High Security** (Default)
```json
"sandbox": {
    "enabled": true,
    "resource_limits": true,
    "restricted_env": true
}
```

2. **Medium Security** (For services needing env vars)
```json
"sandbox": {
    "enabled": true,
    "resource_limits": true,
    "restricted_env": false
}
```

3. **No Sandbox** (For trusted services needing full access)
```json
"sandbox": {
    "enabled": false
}
```

### Skill Configuration

```python
agent.add_skill("mcp_gateway", {
    "gateway_url": "https://localhost:8080",
    "auth_user": "admin",
    "auth_password": "changeme",
    "services": [
        {
            "name": "todo",
            "tools": ["add_todo", "list_todos"]  # Specific tools only
        },
        {
            "name": "calculator",
            "tools": "*"  # All tools
        }
    ],
    "session_timeout": 300,     # Override default timeout
    "tool_prefix": "mcp_",      # Prefix for SWAIG function names
    "retry_attempts": 3,        # Gateway connection retries
    "request_timeout": 30,      # Individual request timeout
    "verify_ssl": True         # SSL certificate verification
})
```

## API Endpoints

### Gateway Service Endpoints

#### GET /health
Health check endpoint
```bash
curl http://localhost:8080/health
```

#### GET /services
List available MCP services
```bash
curl -u admin:changeme http://localhost:8080/services
```

#### GET /services/{service_name}/tools
Get tools for a specific service
```bash
curl -u admin:changeme http://localhost:8080/services/todo/tools
```

#### POST /services/{service_name}/call
Call a tool on a service

Using Basic Auth:
```bash
curl -u admin:changeme -X POST http://localhost:8080/services/todo/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "add_todo",
    "arguments": {"text": "Test item"},
    "session_id": "test-123",
    "timeout": 300
  }'
```

Using Bearer Token:
```bash
curl -X POST http://localhost:8080/services/todo/call \
  -H "Authorization: Bearer your-token-here" \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "add_todo",
    "arguments": {"text": "Test item"},
    "session_id": "test-123"
  }'
```

#### GET /sessions
List active sessions
```bash
curl -u admin:changeme http://localhost:8080/sessions
```

#### DELETE /sessions/{session_id}
Close a specific session
```bash
curl -u admin:changeme -X DELETE http://localhost:8080/sessions/test-123
```

## Security Features

### Authentication
- **Basic Auth**: Username/password authentication
- **Bearer Token**: Alternative token-based authentication
- **Dual Support**: Can use either Basic Auth or Bearer tokens

### Input Validation
- Service name validation (alphanumeric + dash/underscore, max 64 chars)
- Session ID validation (alphanumeric + dot/dash/underscore, max 128 chars)
- Tool name validation (alphanumeric + dash/underscore, max 64 chars)
- Request size limits (10MB max)

### Rate Limiting
Fully configurable through the `rate_limiting` section in config.json:

```json
"rate_limiting": {
    "default_limits": ["200 per day", "50 per hour"],
    "tools_limit": "30 per minute",
    "call_limit": "10 per minute", 
    "session_delete_limit": "20 per minute",
    "storage_uri": "memory://"
}
```

- `default_limits`: Global rate limits per IP address
- `tools_limit`: Rate limit for `/services/*/tools` endpoints
- `call_limit`: Rate limit for `/services/*/call` endpoints
- `session_delete_limit`: Rate limit for session deletion
- `storage_uri`: Storage backend for rate limit counters (memory:// or redis://)

### Security Headers
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection: 1; mode=block
- Content-Security-Policy: default-src 'none'
- Strict-Transport-Security (HTTPS only)

### Process Sandboxing
Configurable per MCP service with three security levels:

1. **High Security** (Default)
   - Process isolation with resource limits
   - Restricted environment variables
   - CPU time: 300s, Memory: 512MB, Processes: 10
   - File size: 10MB max

2. **Medium Security**
   - Resource limits enabled
   - Full environment variables
   - For services needing PATH, NODE_PATH, etc.

3. **No Sandbox**
   - Disabled sandboxing for trusted services
   - Full filesystem and resource access

### Other Security Features
- HTTPS support with SSL/TLS
- Session isolation between calls
- Automatic session cleanup
- Security event logging
- Dangerous environment variable filtering

## Testing

### 1. Unit Testing the Gateway

```bash
# Start the gateway
cd mcp_gateway
python3 gateway_service.py

# Test with curl
./test/test_gateway.sh
```

### 2. Testing with SWAIG CLI

```bash
# Test the agent with MCP skill
swaig-test test/test_agent.py --list-tools

# IMPORTANT: --call-id must come BEFORE --exec for session persistence
swaig-test test/test_agent.py --call-id test-session --exec mcp_todo_add_todo --text "Buy milk"
swaig-test test/test_agent.py --call-id test-session --exec mcp_todo_list_todos

# WRONG: This won't work - --call-id after --exec is treated as function argument
swaig-test test/test_agent.py --exec mcp_todo_add_todo --text "Buy milk" --call-id test-session

# Generate SWML document
swaig-test test/test_agent.py --dump-swml
```

### 3. End-to-End Testing

```python
# test/test_agent.py
from signalwire_agents import AgentBase

class TestMCPAgent(AgentBase):
    def __init__(self):
        super().__init__(name="MCP Test Agent")
        
        self.add_skill("mcp_gateway", {
            "gateway_url": "http://localhost:8080",
            "auth_user": "admin",
            "auth_password": "changeme",
            "services": [{"name": "todo"}]
        })

if __name__ == "__main__":
    agent = TestMCPAgent()
    agent.run()
```

## Deployment

### Local Development
```bash
cd mcp_gateway
python3 gateway_service.py
```

### Docker Deployment

#### Configuration Options
The Docker setup supports three configuration scenarios:

1. **Runtime Config** (highest priority): Mount config.json at runtime
2. **Build-time Config**: Include config.json when building the image
3. **Default Config**: Falls back to sample_config.json

To pre-configure the image at build time:
```bash
# Edit your config.json
cp sample_config.json config.json
vim config.json

# Build with config included
./mcp-docker.sh build  # Will include config.json in image
```

**Port Configuration**: The Docker setup automatically reads the port from your config.json file. If your config specifies port 8100, Docker will expose the service on port 8100.

The mcp-docker.sh script automatically detects the port from config.json. You can also override it using an environment variable:
```bash
# Override port at runtime (must match what's in config.json)
MCP_PORT=8100 ./mcp-docker.sh start
```

Note: The port in the MCP_PORT environment variable should match the port configured in your config.json file, as the container internally listens on the configured port.

#### Using mcp-docker.sh Helper Script
The easiest way to manage the Docker deployment is using the provided helper script:

```bash
cd mcp_gateway

# Show available commands
./mcp-docker.sh help

# Build the Docker image
./mcp-docker.sh build

# Start in foreground (Ctrl+C to stop)
./mcp-docker.sh start

# Start in background
./mcp-docker.sh start -d

# View logs
./mcp-docker.sh logs
./mcp-docker.sh logs -f  # Follow logs

# Check status
./mcp-docker.sh status

# Restart the container
./mcp-docker.sh restart

# Stop the container
./mcp-docker.sh stop

# Open shell in running container
./mcp-docker.sh shell

# Clean up (remove container and volumes)
./mcp-docker.sh clean
```

#### Manual Docker Commands
```bash
cd mcp_gateway
docker build -t mcp-gateway .
docker run -p 8080:8080 -v $(pwd)/config.json:/app/config.json mcp-gateway
```

#### Docker Compose
```bash
cd mcp_gateway
docker-compose up
docker-compose up -d  # Run in background
docker-compose logs -f  # Follow logs
docker-compose down  # Stop and remove
```

### Production with HTTPS
```bash
# Generate or place certificates
mkdir -p certs
# Place server.pem in certs/

# Run with HTTPS
python3 gateway_service.py
```

## Implementation Details

### Session Management

1. **Session Creation**: First tool call creates session with call_id
2. **Session Persistence**: Sessions maintained across multiple tool calls
3. **Session Cleanup**: Automatic cleanup on timeout or hangup hook
4. **State Isolation**: Each session gets separate MCP server instance

### Error Handling

1. **MCP Server Failures**: Automatic restart with backoff
2. **Network Errors**: Retry logic with configurable attempts
3. **Invalid Requests**: Clear error messages returned to SWAIG
4. **Resource Exhaustion**: Reject new sessions when at limit

### Performance Optimization

1. **Connection Pooling**: Reuse HTTP connections to gateway
2. **Lazy Loading**: MCP servers started only when needed
3. **Efficient Cleanup**: Background thread for session management
4. **Response Caching**: Optional caching for read-only operations

## Troubleshooting

### Common Issues

1. **MCP Server Won't Start**
   - Check command path in config.json
   - Verify MCP server is executable
   - Check logs for import errors
   - Ensure working directory is correct for sandboxed processes

2. **Authentication Failures**
   - Verify credentials match in config and skill
   - Check Basic Auth header format
   - For Bearer tokens, ensure "Bearer " prefix is included

3. **Session Timeouts**
   - Increase timeout in skill configuration
   - Check gateway logs for premature cleanup
   - Monitor for stuck MCP processes

4. **SSL Certificate Errors**
   - For self-signed certs, set `verify_ssl: false`
   - Ensure cert path is correct

5. **Gateway Shutdown Hangs**
   - Fixed in latest version with improved thread management
   - Ensure you're running the updated code
   - Check for zombie MCP processes: `ps aux | grep mcp`

6. **Session Persistence Issues**
   - Ensure MCP process doesn't die between calls
   - Check reader thread isn't creating thread leaks
   - Monitor process count with `ps aux | grep todo_mcp`

### Debug Mode

Enable debug logging:
```json
{
    "logging": {
        "level": "DEBUG",
        "file": "gateway.log"
    }
}
```

## Examples

- `examples/mcp_gateway_demo.py` - Agent connecting to MCP servers through the `mcp_gateway` skill

## Future Enhancements

1. **WebSocket Support**: Real-time bidirectional communication
2. **Multi-tenant**: Separate auth/permissions per tenant
3. **Metrics/Monitoring**: Prometheus endpoints
4. **Load Balancing**: Multiple gateway instances
5. **Plugin System**: Custom transformations/middleware