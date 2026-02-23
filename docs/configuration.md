# Configuration Guide

This guide explains the unified configuration system available in SignalWire AI Agents SDK.

## Overview

All SignalWire services (SWML-based agents, Search, MCP Gateway) now support optional JSON configuration files with environment variable substitution. SWML (SignalWire Markup Language) is the JSON document format that defines agent behavior during calls. Services continue to work without any configuration file, maintaining full backward compatibility.

## Quick Start

### Zero Configuration (Default)
```python
# Works exactly as before - no config needed
agent = MyAgent()
agent.run()
```

### With Configuration File
```python
# Automatically detects config.json if present
agent = MyAgent()

# Or specify a config file
agent = MyAgent(config_file="production_config.json")
```

## Configuration Files

### File Locations

Services look for configuration files in this order:
1. Service-specific: `{service_name}_config.json` (e.g., `search_config.json`)
2. Generic: `config.json`
3. Hidden: `.swml/config.json`
4. User home: `~/.swml/config.json`
5. System: `/etc/swml/config.json`

### Configuration Structure

```json
{
  "service": {
    "name": "my-service",
    "host": "${HOST|0.0.0.0}",
    "port": "${PORT|3000}"
  },
  "security": {
    "ssl_enabled": "${SSL_ENABLED|false}",
    "ssl_cert_path": "${SSL_CERT|/etc/ssl/cert.pem}",
    "ssl_key_path": "${SSL_KEY|/etc/ssl/key.pem}",
    "auth": {
      "basic": {
        "enabled": true,
        "user": "${AUTH_USER|signalwire}",
        "password": "${AUTH_PASSWORD}"
      },
      "bearer": {
        "enabled": "${BEARER_ENABLED|false}",
        "token": "${BEARER_TOKEN}"
      }
    },
    "allowed_hosts": ["${PRIMARY_HOST}", "${SECONDARY_HOST|localhost}"],
    "cors_origins": "${CORS_ORIGINS|*}",
    "rate_limit": "${RATE_LIMIT|60}"
  }
}
```

## Environment Variable Substitution

The configuration system supports `${VAR|default}` syntax:

- `${VAR}` - Use environment variable VAR (error if not set)
- `${VAR|default}` - Use VAR or "default" if not set
- `${VAR|}` - Use VAR or empty string if not set

### Examples

```json
{
  "database": {
    "host": "${DB_HOST|localhost}",
    "port": "${DB_PORT|5432}",
    "password": "${DB_PASSWORD}"
  }
}
```

## Priority Order

Configuration values are applied in this order (highest to lowest):

1. **Constructor parameters** - Explicitly passed to service
2. **Config file values** - From JSON configuration
3. **Environment variables** - Direct env vars (backward compatibility)
4. **Defaults** - Hard-coded defaults

## Service-Specific Configuration

### SWML/Agent Configuration

```json
{
  "service": {
    "name": "my-agent",
    "route": "/agent",
    "port": "${AGENT_PORT|3000}"
  },
  "security": {
    "ssl_enabled": "${SSL_ENABLED|false}",
    "auth": {
      "basic": {
        "user": "${AGENT_USER|agent}",
        "password": "${AGENT_PASSWORD}"
      }
    }
  }
}
```

### Search Service Configuration

```json
{
  "service": {
    "port": "${SEARCH_PORT|8001}",
    "indexes": {
      "docs": "${DOCS_INDEX|./docs.swsearch}",
      "api": "${API_INDEX|./api.swsearch}"
    }
  },
  "security": {
    "ssl_enabled": "${SEARCH_SSL|false}",
    "auth": {
      "basic": {
        "enabled": true,
        "user": "${SEARCH_USER|search}",
        "password": "${SEARCH_PASSWORD}"
      }
    }
  }
}
```

### MCP Gateway Configuration

```json
{
  "server": {
    "host": "${MCP_HOST|0.0.0.0}",
    "port": "${MCP_PORT|8080}",
    "auth_user": "${MCP_USER|admin}",
    "auth_password": "${MCP_PASSWORD}",
    "auth_token": "${MCP_TOKEN}"
  },
  "services": {
    "example": {
      "command": ["python", "${SERVICE_PATH|./service.py}"],
      "enabled": "${SERVICE_ENABLED|true}"
    }
  },
  "session": {
    "default_timeout": "${SESSION_TIMEOUT|300}",
    "max_sessions_per_service": "${MAX_SESSIONS|100}"
  }
}
```

## Security Configuration

All services share the same security configuration options:

```json
{
  "security": {
    "ssl_enabled": true,
    "ssl_cert_path": "/etc/ssl/cert.pem",
    "ssl_key_path": "/etc/ssl/key.pem",
    "domain": "api.example.com",
    
    "allowed_hosts": ["api.example.com", "app.example.com"],
    "cors_origins": ["https://app.example.com"],
    
    "max_request_size": 5242880,
    "rate_limit": 30,
    "request_timeout": 60,
    
    "use_hsts": true,
    "hsts_max_age": 31536000
  }
}
```

## Migration Guide

### From Environment Variables Only

Before:
```bash
export SWML_SSL_ENABLED=true
export SWML_SSL_CERT_PATH=/etc/ssl/cert.pem
python my_agent.py
```

After (Option 1 - Keep using env vars):
```bash
# Still works exactly the same
export SWML_SSL_ENABLED=true
export SWML_SSL_CERT_PATH=/etc/ssl/cert.pem
python my_agent.py
```

After (Option 2 - Use config file):
```json
// config.json
{
  "security": {
    "ssl_enabled": true,
    "ssl_cert_path": "/etc/ssl/cert.pem"
  }
}
```

After (Option 3 - Mix config and env vars):
```json
// config.json
{
  "security": {
    "ssl_enabled": true,
    "ssl_cert_path": "${SSL_CERT|/etc/ssl/cert.pem}"
  }
}
```

## Best Practices

1. **Keep secrets in environment variables**
   ```json
   {
     "security": {
       "auth": {
         "basic": {
           "user": "admin",
           "password": "${ADMIN_PASSWORD}"
         }
       }
     }
   }
   ```

2. **Use defaults for development**
   ```json
   {
     "service": {
       "port": "${PORT|3000}",
       "host": "${HOST|localhost}"
     }
   }
   ```

3. **Environment-specific configs**
   - `dev_config.json` - Development settings
   - `prod_config.json` - Production settings
   - Use `${ENV}` to switch between them

4. **Version control**
   - Commit config files WITHOUT secrets
   - Use `.gitignore` for local overrides
   - Document required environment variables

## Programmatic Usage

### Loading Configuration

```python
from signalwire_agents.core.config_loader import ConfigLoader

# Load config
loader = ConfigLoader(["my_config.json"])
if loader.has_config():
    config = loader.get_config()
    
    # Get specific value with substitution
    port = loader.get("service.port", default=3000)
    
    # Get entire section
    security = loader.get_section("security")
```

### Using with Services

```python
# SWML Service
from signalwire_agents import AgentBase

class MyAgent(AgentBase):
    def __init__(self):
        # Auto-detects config.json if present
        super().__init__(name="my-agent", config_file="agent_config.json")

# Search Service
from signalwire_agents.search import SearchService

service = SearchService(config_file="search_config.json")

# MCP Gateway
from mcp_gateway.gateway_service import MCPGateway

gateway = MCPGateway(config_path="mcp_config.json")
```

## Troubleshooting

### Config Not Loading

1. Check file exists and is valid JSON:
   ```bash
   python -m json.tool config.json
   ```

2. Enable debug logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. Check for syntax errors in variable substitution

### Environment Variables Not Substituting

1. Ensure correct syntax: `${VAR}` or `${VAR|default}`
2. Check environment variable is exported:
   ```bash
   echo $MY_VAR
   ```

3. Remember config file values override env vars

### Authentication Issues

1. Config file auth settings override env vars
2. Check which auth method is enabled
3. Verify credentials match