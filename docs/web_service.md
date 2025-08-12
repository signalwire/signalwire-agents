# WebService Documentation

The `WebService` class provides static file serving capabilities for the SignalWire AI Agents SDK. It follows the same architectural pattern as `SearchService`, allowing it to run as a standalone service or alongside your AI agents.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Security Features](#security-features)
- [HTTPS/SSL Support](#httpsssl-support)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Deployment Patterns](#deployment-patterns)

## Overview

WebService is designed to serve static files with enterprise-grade security features. It's perfect for:
- Serving agent documentation and API specs
- Hosting static assets (images, CSS, JavaScript)
- Serving generated reports and exports
- Providing configuration files and templates
- Hosting agent UI components

### Key Features
- **Multiple directory mounting** - Serve different directories at different URL paths
- **Security-first design** - Authentication, CORS, security headers, file filtering
- **HTTPS support** - Full SSL/TLS support with PEM files
- **Directory browsing** - Optional HTML directory listings
- **MIME type handling** - Automatic content-type detection
- **Path traversal protection** - Prevents access outside designated directories
- **File filtering** - Allow/block specific file extensions

## Installation

WebService is included in the core SignalWire AI Agents SDK:

```bash
pip install signalwire-agents
```

## Quick Start

```python
from signalwire_agents import WebService

# Create a service to serve files
service = WebService(
    port=8002,
    directories={
        "/docs": "./documentation",
        "/assets": "./static/assets"
    }
)

# Start the service
service.start()
# Service available at http://localhost:8002
# Basic Auth: dev:w00t (auto-generated)
```

## Configuration

WebService can be configured through multiple methods (in order of priority):

### 1. Constructor Parameters

```python
service = WebService(
    port=8002,                          # Port to bind to
    directories={                       # URL path to directory mappings
        "/docs": "./documentation",
        "/assets": "./static"
    },
    basic_auth=("admin", "secret"),    # Custom authentication
    enable_directory_browsing=True,     # Allow directory listings
    allowed_extensions=['.html', '.css', '.js'],  # Whitelist extensions
    blocked_extensions=['.env', '.key'],          # Blacklist extensions
    max_file_size=100 * 1024 * 1024,   # Max file size (100MB)
    enable_cors=True                    # Enable CORS headers
)
```

### 2. Environment Variables

```bash
# Basic authentication
export SWML_BASIC_AUTH_USER="admin"
export SWML_BASIC_AUTH_PASS="secretpassword"

# SSL/HTTPS configuration
export SWML_SSL_ENABLED=true
export SWML_SSL_CERT="/path/to/cert.pem"
export SWML_SSL_KEY="/path/to/key.pem"

# Security settings
export SWML_ALLOWED_HOSTS="example.com,*.example.com"
export SWML_CORS_ORIGINS="https://app.example.com"
```

### 3. Configuration File

Create a `web.json` or `swml_web.json` file:

```json
{
    "service": {
        "port": 8002,
        "directories": {
            "/docs": "./documentation",
            "/api": "./api-specs",
            "/reports": "./generated/reports"
        },
        "enable_directory_browsing": true,
        "max_file_size": 52428800,
        "allowed_extensions": [".html", ".css", ".js", ".json", ".pdf"],
        "blocked_extensions": [".env", ".key", ".pem"]
    },
    "security": {
        "basic_auth": {
            "username": "admin",
            "password": "secure123"
        },
        "ssl_enabled": true,
        "ssl_cert": "/etc/ssl/certs/server.crt",
        "ssl_key": "/etc/ssl/private/server.key",
        "allowed_hosts": ["*"],
        "cors_origins": ["*"]
    }
}
```

## Security Features

### Basic Authentication

WebService implements HTTP Basic Authentication. Credentials can be set via:

1. **Constructor**: `basic_auth=("username", "password")`
2. **Environment**: `SWML_BASIC_AUTH_USER` and `SWML_BASIC_AUTH_PASS`
3. **Config file**: `security.basic_auth` section
4. **Auto-generated**: If not specified, generates random credentials

### File Security

#### Default Blocked Extensions/Files
- `.env`, `.git`, `.gitignore`
- `.key`, `.pem`, `.crt`
- `.pyc`, `__pycache__`
- `.DS_Store`, `.swp`

#### Path Traversal Protection
WebService prevents access outside designated directories:
```python
# These attempts will be blocked:
# GET /docs/../../../etc/passwd
# GET /docs/./././../config.json
```

#### File Size Limits
Default maximum file size is 100MB. Configure with:
```python
service = WebService(max_file_size=50 * 1024 * 1024)  # 50MB
```

### Security Headers

Automatically adds security headers to all responses:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security` (when HTTPS is enabled)

## HTTPS/SSL Support

WebService provides multiple ways to enable HTTPS:

### Method 1: Environment Variables

```bash
# Using file paths
export SWML_SSL_CERT="/path/to/cert.pem"
export SWML_SSL_KEY="/path/to/key.pem"

# Or using inline PEM content
export SWML_SSL_CERT_INLINE="-----BEGIN CERTIFICATE-----
MIIDXTCCAkWgAwIBAgIJAKLdQVPy...
-----END CERTIFICATE-----"
export SWML_SSL_KEY_INLINE="-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQE...
-----END PRIVATE KEY-----"
```

### Method 2: Direct Parameters

```python
service = WebService(directories={"/docs": "./docs"})
service.start(
    ssl_cert="/path/to/cert.pem",
    ssl_key="/path/to/key.pem"
)
# Service available at https://localhost:8002
```

### Method 3: Configuration File

```json
{
    "security": {
        "ssl_enabled": true,
        "ssl_cert": "/etc/ssl/certs/server.crt",
        "ssl_key": "/etc/ssl/private/server.key"
    }
}
```

### Generating Self-Signed Certificates

For development/testing:

```bash
# Generate a self-signed certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem \
    -days 365 -nodes -subj "/CN=localhost"

# Use with WebService
export SWML_SSL_CERT="cert.pem"
export SWML_SSL_KEY="key.pem"
```

## API Endpoints

### GET /health
Health check endpoint (no authentication required)

**Response:**
```json
{
    "status": "healthy",
    "directories": ["/docs", "/assets"],
    "ssl_enabled": false,
    "auth_required": true,
    "directory_browsing": true
}
```

### GET /
Root endpoint showing available directories

**Response:** HTML page listing all mounted directories

### GET /{route}/{file_path}
Serve files from mounted directories

**Parameters:**
- `route`: The mounted directory route (e.g., `/docs`)
- `file_path`: Path to file within the directory

**Response:**
- File content with appropriate MIME type
- 404 if file not found
- 403 if file type blocked or directory browsing disabled

## Usage Examples

### Basic File Serving

```python
from signalwire_agents import WebService

# Serve documentation
service = WebService(
    directories={
        "/docs": "./documentation",
        "/api": "./api-specs"
    }
)
service.start()

# Files accessible at:
# http://localhost:8002/docs/index.html
# http://localhost:8002/api/swagger.json
```

### With Directory Browsing

```python
service = WebService(
    directories={"/files": "./public"},
    enable_directory_browsing=True  # Allow browsing directories
)
service.start()

# Browse files at: http://localhost:8002/files/
```

### Restricted File Types

```python
# Only serve web assets
service = WebService(
    directories={"/web": "./www"},
    allowed_extensions=['.html', '.css', '.js', '.png', '.jpg', '.woff2'],
    enable_directory_browsing=False
)
```

### Dynamic Directory Management

```python
service = WebService()

# Add directories after initialization
service.add_directory("/docs", "./documentation")
service.add_directory("/reports", "./generated/reports")

# Remove a directory
service.remove_directory("/reports")

service.start()
```

### With Custom Authentication

```python
service = WebService(
    directories={"/private": "./sensitive-docs"},
    basic_auth=("admin", "super-secret-password")
)
service.start()
```

### HTTPS with Let's Encrypt

```python
# Assuming you have Let's Encrypt certificates
service = WebService(
    directories={"/secure": "./secure-files"}
)
service.start(
    ssl_cert="/etc/letsencrypt/live/example.com/fullchain.pem",
    ssl_key="/etc/letsencrypt/live/example.com/privkey.pem"
)
# Service available at https://example.com:8002
```

### Multi-Environment Configuration

```python
import os

# Development vs Production
if os.getenv("ENVIRONMENT") == "production":
    service = WebService(
        port=443,
        directories={"/": "./dist"},
        enable_directory_browsing=False
    )
    service.start(
        host="0.0.0.0",
        ssl_cert="/etc/ssl/certs/production.crt",
        ssl_key="/etc/ssl/private/production.key"
    )
else:
    service = WebService(
        port=8002,
        directories={"/": "./src"},
        enable_directory_browsing=True
    )
    service.start()
```

## Deployment Patterns

### Standalone Service

Run WebService as a dedicated static file server:

```python
# web_server.py
from signalwire_agents import WebService

if __name__ == "__main__":
    service = WebService(
        port=8002,
        directories={
            "/docs": "/var/www/docs",
            "/assets": "/var/www/assets",
            "/downloads": "/var/www/downloads"
        }
    )
    service.start()
```

### Alongside AI Agents

Run WebService alongside your AI agents on different ports:

```python
# main.py
from signalwire_agents import AgentBase, WebService
import threading

# Start WebService in background
def run_web_service():
    web = WebService(
        port=8002,
        directories={"/docs": "./agent-docs"}
    )
    web.start()

# Start web service thread
web_thread = threading.Thread(target=run_web_service, daemon=True)
web_thread.start()

# Run your agent
class MyAgent(AgentBase):
    def __init__(self):
        super().__init__(name="My Agent")

agent = MyAgent()
agent.serve(port=3000)  # Agent on port 3000, WebService on 8002
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install SDK
RUN pip install signalwire-agents

# Copy static files
COPY ./static /app/static
COPY ./web_config.json /app/web_config.json

# Expose port
EXPOSE 8002

# Run WebService
CMD ["python", "-c", "from signalwire_agents import WebService; WebService(config_file='web_config.json').start()"]
```

### Systemd Service

Create `/etc/systemd/system/signalwire-web.service`:

```ini
[Unit]
Description=SignalWire Web Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/signalwire
Environment="SWML_SSL_CERT=/etc/ssl/certs/server.crt"
Environment="SWML_SSL_KEY=/etc/ssl/private/server.key"
ExecStart=/usr/bin/python3 -c "from signalwire_agents import WebService; WebService(directories={'/': '/var/www/html'}).start()"
Restart=always

[Install]
WantedBy=multi-user.target
```

### Nginx Reverse Proxy

For production, use Nginx as a reverse proxy:

```nginx
server {
    listen 80;
    server_name static.example.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name static.example.com;
    
    ssl_certificate /etc/ssl/certs/example.com.crt;
    ssl_certificate_key /etc/ssl/private/example.com.key;
    
    location / {
        proxy_pass http://localhost:8002;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Cache static assets
        location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
            proxy_pass http://localhost:8002;
            expires 1h;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

## Best Practices

### Security
1. **Always use HTTPS in production** - Protect data in transit
2. **Change default credentials** - Never use auto-generated auth in production
3. **Restrict file types** - Use `allowed_extensions` to whitelist safe files
4. **Disable directory browsing** - Turn off in production environments
5. **Use reverse proxy** - Put Nginx/Apache in front for additional security

### Performance
1. **Set appropriate cache headers** - WebService adds 1-hour cache by default
2. **Limit file sizes** - Adjust `max_file_size` based on your needs
3. **Use CDN for static assets** - Offload traffic for better performance
4. **Compress large files** - Use gzip/brotli at reverse proxy level

### Organization
1. **Separate content types** - Use different routes for different file types
2. **Version your assets** - Include version in path (e.g., `/assets/v1/`)
3. **Use index.html** - Provide default files for directories
4. **Document your structure** - Maintain clear directory organization

## Troubleshooting

### Common Issues

**Issue: "FastAPI not available"**
```bash
# Install FastAPI and uvicorn
pip install fastapi uvicorn
```

**Issue: SSL certificate errors**
```python
# Check certificate paths
import os
print(os.path.exists("/path/to/cert.pem"))  # Should be True
print(os.path.exists("/path/to/key.pem"))   # Should be True
```

**Issue: Permission denied**
```bash
# Ensure read permissions on directories
chmod -R 755 /path/to/static/files
```

**Issue: Directory not found**
```python
# Use absolute paths
import os
service = WebService(
    directories={
        "/docs": os.path.abspath("./documentation")
    }
)
```

### Debug Logging

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

service = WebService(directories={"/test": "./test"})
service.start()
```

## API Reference

### WebService Class

```python
class WebService:
    def __init__(self,
                 port: int = 8002,
                 directories: Dict[str, str] = None,
                 basic_auth: Optional[Tuple[str, str]] = None,
                 config_file: Optional[str] = None,
                 enable_directory_browsing: bool = False,
                 allowed_extensions: Optional[list] = None,
                 blocked_extensions: Optional[list] = None,
                 max_file_size: int = 100 * 1024 * 1024,
                 enable_cors: bool = True)
```

#### Parameters
- `port`: Port to bind to (default: 8002)
- `directories`: Dictionary mapping URL paths to local directories
- `basic_auth`: Tuple of (username, password) for authentication
- `config_file`: Path to JSON configuration file
- `enable_directory_browsing`: Allow directory listing (default: False)
- `allowed_extensions`: List of allowed file extensions
- `blocked_extensions`: List of blocked file extensions
- `max_file_size`: Maximum file size in bytes (default: 100MB)
- `enable_cors`: Enable CORS headers (default: True)

#### Methods

##### start()
```python
def start(self,
          host: str = "0.0.0.0",
          port: Optional[int] = None,
          ssl_cert: Optional[str] = None,
          ssl_key: Optional[str] = None)
```
Start the web service.

##### add_directory()
```python
def add_directory(self, route: str, directory: str) -> None
```
Add a new directory to serve.

##### remove_directory()
```python
def remove_directory(self, route: str) -> None
```
Remove a directory from being served.

## Integration with SignalWire Agents

WebService complements AI agents by providing static file serving:

```python
from signalwire_agents import AgentBase, WebService

class DocumentationAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Documentation Assistant")
        
        # Reference documentation served by WebService
        self.prompt_add_section(
            "Documentation",
            "User documentation is available at https://example.com:8002/docs/"
        )
    
    @self.tool()
    def get_doc_link(args, raw_data):
        doc_name = args.get('doc_name')
        return SwaigFunctionResult(
            f"Documentation available at: https://example.com:8002/docs/{doc_name}.html"
        )

# Run both services
if __name__ == "__main__":
    # Start WebService for documentation
    web = WebService(
        port=8002,
        directories={"/docs": "./documentation"}
    )
    
    # Start agent
    agent = DocumentationAgent()
    
    # Run in threads or separate processes
    import threading
    web_thread = threading.Thread(target=web.start, daemon=True)
    web_thread.start()
    
    agent.serve(port=3000)
```

## Summary

WebService provides a secure, configurable static file server that integrates seamlessly with the SignalWire AI Agents SDK. It follows the same architectural patterns as other SDK services, making it familiar and easy to use while providing enterprise-grade security features and flexible deployment options.