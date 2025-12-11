#!/usr/bin/env python3
"""
SignalWire Agent Dokku Deployment Tool

CLI tool for deploying SignalWire agents to Dokku with support for:
- Simple git push deployment
- Full CI/CD with GitHub Actions
- Service provisioning (PostgreSQL, Redis)
- Preview environments for PRs

Usage:
    sw-agent-dokku init myagent                    # Simple mode
    sw-agent-dokku init myagent --cicd             # With GitHub Actions CI/CD
    sw-agent-dokku deploy                          # Deploy current directory
    sw-agent-dokku logs                            # Tail logs
    sw-agent-dokku config set KEY=value            # Set environment variables
    sw-agent-dokku scale web=2                     # Scale processes
"""

import os
import sys
import subprocess
import secrets
import argparse
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Any


# =============================================================================
# ANSI Colors
# =============================================================================

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    NC = '\033[0m'


def print_step(msg: str):
    print(f"{Colors.BLUE}==>{Colors.NC} {msg}")


def print_success(msg: str):
    print(f"{Colors.GREEN}✓{Colors.NC} {msg}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}!{Colors.NC} {msg}")


def print_error(msg: str):
    print(f"{Colors.RED}✗{Colors.NC} {msg}")


def print_header(msg: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{msg}{Colors.NC}")


def prompt(question: str, default: str = "") -> str:
    if default:
        result = input(f"{question} [{default}]: ").strip()
        return result if result else default
    return input(f"{question}: ").strip()


def prompt_yes_no(question: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    result = input(f"{question} [{hint}]: ").strip().lower()
    if not result:
        return default
    return result in ('y', 'yes')


def generate_password(length: int = 32) -> str:
    return secrets.token_urlsafe(length)[:length]


# =============================================================================
# Templates - Core Files
# =============================================================================

PROCFILE_TEMPLATE = """web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --worker-class uvicorn.workers.UvicornWorker
"""

RUNTIME_TEMPLATE = """python-3.11
"""

REQUIREMENTS_TEMPLATE = """signalwire-agents>=1.0.16
gunicorn>=21.0.0
uvicorn>=0.24.0
python-dotenv>=1.0.0
requests>=2.28.0
"""

CHECKS_TEMPLATE = """WAIT=5
TIMEOUT=30
ATTEMPTS=5

/health
"""

GITIGNORE_TEMPLATE = """# Environment
.env
.venv/
venv/
__pycache__/
*.pyc
*.pyo

# IDE
.vscode/
.idea/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/

# Build
dist/
build/
*.egg-info/

# Logs
*.log

# OS
.DS_Store
Thumbs.db
"""

ENV_EXAMPLE_TEMPLATE = """# SignalWire Agent Configuration
# =============================================================================

# SignalWire Credentials (required for WebRTC calling)
SIGNALWIRE_SPACE_NAME=your-space
SIGNALWIRE_PROJECT_ID=your-project-id
SIGNALWIRE_TOKEN=your-api-token

# Public URL for SWML callbacks (required for WebRTC calling)
# This should be your publicly accessible URL (e.g., ngrok, dokku domain)
SWML_PROXY_URL_BASE=https://your-app.example.com

# Basic Auth for SWML endpoints (password required, user defaults to 'signalwire')
# SWML_BASIC_AUTH_USER=signalwire
SWML_BASIC_AUTH_PASSWORD=your-secure-password

# Agent Configuration
AGENT_NAME={app_name}

# App Configuration
APP_ENV=production
APP_NAME={app_name}

# Optional: External Services
# DATABASE_URL=postgres://user:pass@host:5432/db
# REDIS_URL=redis://host:6379
"""

APP_TEMPLATE = '''#!/usr/bin/env python3
"""
{agent_name} - SignalWire AI Agent

Deployed to Dokku with automatic health checks and SWAIG support.
"""

import os
from dotenv import load_dotenv
from signalwire_agents import AgentBase, SwaigFunctionResult

# Load environment variables from .env file
load_dotenv()


class {agent_class}(AgentBase):
    """{agent_name} agent for Dokku deployment."""

    def __init__(self):
        super().__init__(name="{agent_slug}")

        self._configure_prompts()
        self.add_language("English", "en-US", "rime.spore")
        self._setup_functions()

    def _configure_prompts(self):
        self.prompt_add_section(
            "Role",
            "You are a helpful AI assistant deployed on Dokku."
        )

        self.prompt_add_section(
            "Guidelines",
            bullets=[
                "Be professional and courteous",
                "Ask clarifying questions when needed",
                "Keep responses concise and helpful"
            ]
        )

    def _setup_functions(self):
        @self.tool(
            description="Get information about a topic",
            parameters={{
                "type": "object",
                "properties": {{
                    "topic": {{
                        "type": "string",
                        "description": "The topic to get information about"
                    }}
                }},
                "required": ["topic"]
            }}
        )
        def get_info(args, raw_data):
            topic = args.get("topic", "")
            return SwaigFunctionResult(
                f"Information about {{topic}}: This is a placeholder response."
            )

        @self.tool(description="Get deployment information")
        def get_deployment_info(args, raw_data):
            app_name = os.getenv("APP_NAME", "unknown")
            app_env = os.getenv("APP_ENV", "unknown")

            return SwaigFunctionResult(
                f"Running on Dokku. App: {{app_name}}, Environment: {{app_env}}."
            )


# Create agent instance
agent = {agent_class}()

# Expose the FastAPI app for gunicorn/uvicorn
app = agent.get_app()

if __name__ == "__main__":
    agent.run()
'''

APP_TEMPLATE_WITH_WEB = '''#!/usr/bin/env python3
"""
{agent_name} - SignalWire AI Agent

Deployed to Dokku with automatic health checks, SWAIG support, and web interface.
Includes WebRTC calling support with dynamic token generation.
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv
import requests
from starlette.responses import JSONResponse
from signalwire_agents import AgentBase, AgentServer, SwaigFunctionResult

# Load environment variables from .env file
load_dotenv()


class {agent_class}(AgentBase):
    """{agent_name} agent for Dokku deployment."""

    def __init__(self):
        super().__init__(name="{agent_slug}", route="/swml")

        self._configure_prompts()
        self.add_language("English", "en-US", "rime.spore")
        self._setup_functions()

    def _configure_prompts(self):
        self.prompt_add_section(
            "Role",
            "You are a helpful AI assistant deployed on Dokku."
        )

        self.prompt_add_section(
            "Guidelines",
            bullets=[
                "Be professional and courteous",
                "Ask clarifying questions when needed",
                "Keep responses concise and helpful"
            ]
        )

    def _setup_functions(self):
        @self.tool(
            description="Get information about a topic",
            parameters={{
                "type": "object",
                "properties": {{
                    "topic": {{
                        "type": "string",
                        "description": "The topic to get information about"
                    }}
                }},
                "required": ["topic"]
            }}
        )
        def get_info(args, raw_data):
            topic = args.get("topic", "")
            return SwaigFunctionResult(
                f"Information about {{topic}}: This is a placeholder response."
            )

        @self.tool(description="Get deployment information")
        def get_deployment_info(args, raw_data):
            app_name = os.getenv("APP_NAME", "unknown")
            app_env = os.getenv("APP_ENV", "unknown")

            return SwaigFunctionResult(
                f"Running on Dokku. App: {{app_name}}, Environment: {{app_env}}."
            )


# =============================================================================
# SignalWire SWML Handler Management
# =============================================================================

def get_signalwire_host():
    """Get the full SignalWire host from space name."""
    space = os.getenv("SIGNALWIRE_SPACE_NAME", "")
    if not space:
        return None
    if "." in space:
        return space
    return f"{{space}}.signalwire.com"


def find_existing_handler(sw_host, auth, agent_name):
    """Find an existing SWML handler by name."""
    try:
        resp = requests.get(
            f"https://{{sw_host}}/api/fabric/resources/external_swml_handlers",
            auth=auth,
            headers={{"Accept": "application/json"}}
        )
        if resp.status_code != 200:
            return None

        handlers = resp.json().get("data", [])
        for handler in handlers:
            swml_webhook = handler.get("swml_webhook", {{}})
            handler_name = swml_webhook.get("name") or handler.get("display_name")
            if handler_name == agent_name:
                handler_id = handler.get("id")
                handler_url = swml_webhook.get("primary_request_url", "")
                addr_resp = requests.get(
                    f"https://{{sw_host}}/api/fabric/resources/external_swml_handlers/{{handler_id}}/addresses",
                    auth=auth,
                    headers={{"Accept": "application/json"}}
                )
                if addr_resp.status_code == 200:
                    addresses = addr_resp.json().get("data", [])
                    if addresses:
                        return {{
                            "id": handler_id,
                            "name": handler_name,
                            "url": handler_url,
                            "address_id": addresses[0]["id"],
                            "address": addresses[0]["channels"]["audio"]
                        }}
    except Exception as e:
        print(f"Error checking existing handlers: {{e}}")
    return None


# Store SWML handler info
swml_handler_info = {{"id": None, "address_id": None, "address": None}}


def setup_swml_handler():
    """Set up SWML handler on startup."""
    sw_host = get_signalwire_host()
    project = os.getenv("SIGNALWIRE_PROJECT_ID", "")
    token = os.getenv("SIGNALWIRE_TOKEN", "")
    agent_name = os.getenv("AGENT_NAME", "{agent_slug}")
    proxy_url = os.getenv("SWML_PROXY_URL_BASE", os.getenv("APP_URL", ""))
    auth_user = os.getenv("SWML_BASIC_AUTH_USER", "signalwire")
    auth_pass = os.getenv("SWML_BASIC_AUTH_PASSWORD", "")

    if not all([sw_host, project, token]):
        print("SignalWire credentials not configured - skipping SWML handler setup")
        return

    if not proxy_url:
        print("SWML_PROXY_URL_BASE not set - skipping SWML handler setup")
        return

    # Build SWML URL with basic auth (user defaults to 'signalwire' if not set)
    if auth_pass and "://" in proxy_url:
        scheme, rest = proxy_url.split("://", 1)
        swml_url = f"{{scheme}}://{{auth_user}}:{{auth_pass}}@{{rest}}/swml"
    else:
        swml_url = proxy_url + "/swml"

    auth = (project, token)
    headers = {{"Content-Type": "application/json", "Accept": "application/json"}}

    existing = find_existing_handler(sw_host, auth, agent_name)
    if existing:
        swml_handler_info["id"] = existing["id"]
        swml_handler_info["address_id"] = existing["address_id"]
        swml_handler_info["address"] = existing["address"]

        # Always update the URL on startup to ensure credentials are current
        try:
            requests.put(
                f"https://{{sw_host}}/api/fabric/resources/external_swml_handlers/{{existing['id']}}",
                json={{"primary_request_url": swml_url, "primary_request_method": "POST"}},
                auth=auth,
                headers=headers
            )
            print(f"Updated SWML handler: {{existing['name']}}")
        except Exception as e:
            print(f"Failed to update handler URL: {{e}}")
        print(f"Call address: {{existing['address']}}")
    else:
        try:
            handler_resp = requests.post(
                f"https://{{sw_host}}/api/fabric/resources/external_swml_handlers",
                json={{
                    "name": agent_name,
                    "used_for": "calling",
                    "primary_request_url": swml_url,
                    "primary_request_method": "POST"
                }},
                auth=auth,
                headers=headers
            )
            handler_resp.raise_for_status()
            handler_id = handler_resp.json().get("id")
            swml_handler_info["id"] = handler_id

            addr_resp = requests.get(
                f"https://{{sw_host}}/api/fabric/resources/external_swml_handlers/{{handler_id}}/addresses",
                auth=auth,
                headers={{"Accept": "application/json"}}
            )
            addr_resp.raise_for_status()
            addresses = addr_resp.json().get("data", [])
            if addresses:
                swml_handler_info["address_id"] = addresses[0]["id"]
                swml_handler_info["address"] = addresses[0]["channels"]["audio"]
                print(f"Created SWML handler: {{agent_name}}")
                print(f"Call address: {{swml_handler_info['address']}}")
        except Exception as e:
            print(f"Failed to create SWML handler: {{e}}")


# =============================================================================
# Server Setup
# =============================================================================

server = AgentServer(host="0.0.0.0", port=int(os.getenv("PORT", 3000)))
server.register({agent_class}())

# Serve static files from web/ directory
web_dir = Path(__file__).parent / "web"
if web_dir.exists():
    server.serve_static_files(str(web_dir))


# =============================================================================
# API Endpoints
# =============================================================================

@server.app.get("/get_token")
def get_token():
    """Get a guest token for WebRTC calls."""
    sw_host = get_signalwire_host()
    project = os.getenv("SIGNALWIRE_PROJECT_ID", "")
    token = os.getenv("SIGNALWIRE_TOKEN", "")

    if not all([sw_host, project, token]):
        return JSONResponse({{"error": "SignalWire credentials not configured"}}, status_code=500)

    if not swml_handler_info["address_id"]:
        return JSONResponse({{"error": "SWML handler not configured - check startup logs"}}, status_code=500)

    auth = (project, token)
    headers = {{"Content-Type": "application/json", "Accept": "application/json"}}

    try:
        expire_at = int(time.time()) + 3600 * 24  # 24 hours

        guest_resp = requests.post(
            f"https://{{sw_host}}/api/fabric/guests/tokens",
            json={{
                "allowed_addresses": [swml_handler_info["address_id"]],
                "expire_at": expire_at
            }},
            auth=auth,
            headers=headers
        )
        guest_resp.raise_for_status()
        guest_token = guest_resp.json().get("token", "")

        return {{
            "token": guest_token,
            "address": swml_handler_info["address"]
        }}

    except requests.exceptions.RequestException as e:
        print(f"Token request failed: {{e}}")
        return JSONResponse({{"error": str(e)}}, status_code=500)


@server.app.get("/get_credentials")
def get_credentials():
    """Get basic auth credentials for curl examples."""
    return {{
        "user": os.getenv("SWML_BASIC_AUTH_USER", ""),
        "password": os.getenv("SWML_BASIC_AUTH_PASSWORD", "")
    }}


@server.app.get("/get_resource_info")
def get_resource_info():
    """Get SWML handler resource info for dashboard link."""
    sw_host = get_signalwire_host()
    space_name = os.getenv("SIGNALWIRE_SPACE_NAME", "")
    return {{
        "space_name": space_name,
        "resource_id": swml_handler_info["id"],
        "dashboard_url": f"https://{{sw_host}}/neon/resources/{{swml_handler_info['id']}}/edit?t=addresses" if sw_host and swml_handler_info["id"] else None
    }}


# =============================================================================
# Static File Handling
# =============================================================================

from fastapi import Request, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
import mimetypes

@server.app.api_route("/swml", methods=["GET", "POST"])
async def swml_redirect():
    return RedirectResponse(url="/swml/", status_code=307)

@server.app.get("/{{full_path:path}}")
async def serve_static(request: Request, full_path: str):
    """Serve static files from web/ directory"""
    if not web_dir.exists():
        raise HTTPException(status_code=404, detail="Not Found")

    if not full_path or full_path == "/":
        full_path = "index.html"

    file_path = web_dir / full_path

    try:
        file_path = file_path.resolve()
        if not str(file_path).startswith(str(web_dir.resolve())):
            raise HTTPException(status_code=404, detail="Not Found")
    except Exception:
        raise HTTPException(status_code=404, detail="Not Found")

    if file_path.exists() and file_path.is_file():
        media_type, _ = mimetypes.guess_type(str(file_path))
        return FileResponse(file_path, media_type=media_type)

    if (web_dir / full_path / "index.html").exists():
        return FileResponse(web_dir / full_path / "index.html", media_type="text/html")

    raise HTTPException(status_code=404, detail="Not Found")


# Set up SWML handler on startup
setup_swml_handler()

# Expose ASGI app for gunicorn
app = server.app

if __name__ == "__main__":
    server.run()
'''

WEB_INDEX_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{agent_name}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #f8f9fa;
            color: #333;
        }}
        h1 {{
            color: #044cf6;
            border-bottom: 3px solid #044cf6;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #333;
            margin-top: 40px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 8px;
        }}
        .status {{
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .call-section {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 30px;
            margin: 20px 0;
            color: white;
            text-align: center;
        }}
        .call-section h2 {{
            color: white;
            border: none;
            margin-top: 0;
        }}
        .call-controls {{
            display: flex;
            gap: 15px;
            justify-content: center;
            align-items: center;
            flex-wrap: wrap;
        }}
        .call-btn {{
            padding: 15px 40px;
            font-size: 18px;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        .call-btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        .call-btn.connect {{
            background: #10b981;
            color: white;
        }}
        .call-btn.connect:hover:not(:disabled) {{
            background: #059669;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(16, 185, 129, 0.4);
        }}
        .call-btn.disconnect {{
            background: #ef4444;
            color: white;
        }}
        .call-btn.disconnect:hover:not(:disabled) {{
            background: #dc2626;
        }}
        .call-status {{
            margin-top: 15px;
            font-size: 14px;
            opacity: 0.9;
        }}
        .destination-input {{
            padding: 12px 15px;
            font-size: 14px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 8px;
            background: rgba(255,255,255,0.1);
            color: white;
            width: 250px;
        }}
        .destination-input::placeholder {{
            color: rgba(255,255,255,0.6);
        }}
        .destination-input:focus {{
            outline: none;
            border-color: rgba(255,255,255,0.6);
            background: rgba(255,255,255,0.2);
        }}
        .endpoint {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
        }}
        .endpoint h3 {{
            margin-top: 0;
            color: #044cf6;
        }}
        .method {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 12px;
            margin-right: 10px;
        }}
        .method.get {{ background: #61affe; color: white; }}
        .method.post {{ background: #49cc90; color: white; }}
        .path {{
            font-family: monospace;
            font-size: 16px;
            color: #333;
        }}
        code, pre {{
            font-family: 'SF Mono', Monaco, 'Courier New', monospace;
        }}
        code {{
            background: #e9ecef;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 14px;
        }}
        pre {{
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 13px;
            line-height: 1.5;
            margin: 0;
        }}
        pre .comment {{ color: #6a9955; }}
        .tabs {{
            display: flex;
            gap: 0;
            margin-top: 15px;
        }}
        .tab {{
            padding: 8px 16px;
            background: #e9ecef;
            border: 1px solid #ddd;
            border-bottom: none;
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            color: #666;
            transition: all 0.2s;
        }}
        .tab:hover {{ background: #dee2e6; }}
        .tab.active {{
            background: #1e1e1e;
            color: #d4d4d4;
            border-color: #1e1e1e;
        }}
        .tab-content {{
            display: none;
            border-radius: 0 8px 8px 8px;
        }}
        .tab-content.active {{ display: block; }}
        .browser-panel {{
            background: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 0 8px 8px 8px;
            padding: 15px;
        }}
        .try-btn {{
            padding: 10px 20px;
            background: #044cf6;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            font-size: 14px;
            transition: all 0.2s;
        }}
        .try-btn:hover {{ background: #0339c2; }}
        .try-btn:disabled {{
            background: #ccc;
            cursor: not-allowed;
        }}
        .response-area {{
            margin-top: 15px;
            display: none;
        }}
        .response-area.visible {{ display: block; }}
        .response-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}
        .response-status {{
            font-size: 13px;
            font-weight: 500;
        }}
        .response-status.success {{ color: #10b981; }}
        .response-status.error {{ color: #ef4444; }}
        .response-time {{
            font-size: 12px;
            color: #666;
        }}
        .response-body {{
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 15px;
            border-radius: 8px;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 12px;
            max-height: 300px;
            overflow: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        .curl-panel {{
            position: relative;
        }}
        .copy-btn {{
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 5px 10px;
            background: rgba(255,255,255,0.1);
            color: #999;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
            transition: all 0.2s;
        }}
        .copy-btn:hover {{
            background: rgba(255,255,255,0.2);
            color: #fff;
        }}
        .audio-settings {{
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-top: 15px;
            flex-wrap: wrap;
        }}
        .audio-setting {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            color: rgba(255,255,255,0.9);
        }}
        .audio-setting input[type="checkbox"] {{
            width: 18px;
            height: 18px;
            cursor: pointer;
        }}
        .audio-setting label {{
            cursor: pointer;
        }}
        .phone-info {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid #044cf6;
            border-radius: 12px;
            padding: 20px 25px;
            margin: 30px 0;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }}
        .phone-info h3 {{
            margin: 0 0 10px 0;
            color: #fff;
            font-size: 16px;
        }}
        .phone-info p {{
            margin: 0 0 15px 0;
            color: rgba(255,255,255,0.8);
            font-size: 14px;
            line-height: 1.5;
        }}
        .phone-info a {{
            display: inline-block;
            padding: 10px 20px;
            background: #044cf6;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            font-weight: 500;
            font-size: 14px;
            transition: all 0.2s;
        }}
        .phone-info a:hover {{
            background: #0339c2;
        }}
        .phone-info.hidden {{
            display: none;
        }}
        .footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <h1>{agent_name}</h1>

    <div class="status">
        Your agent is running and ready to receive calls!
    </div>

    <div class="call-section">
        <h2>Call Your Agent</h2>
        <p>Test your agent directly from the browser using WebRTC.</p>
        <div class="call-controls">
            <input type="text" id="destination" class="destination-input" placeholder="Address will be auto-filled" />
            <button id="connectBtn" class="call-btn connect">Call Agent</button>
            <button id="disconnectBtn" class="call-btn disconnect" disabled>Hang Up</button>
        </div>
        <div class="audio-settings">
            <div class="audio-setting">
                <input type="checkbox" id="echoCancellation" checked>
                <label for="echoCancellation">Echo Cancellation</label>
            </div>
            <div class="audio-setting">
                <input type="checkbox" id="noiseSuppression">
                <label for="noiseSuppression">Noise Suppression</label>
            </div>
            <div class="audio-setting">
                <input type="checkbox" id="autoGainControl">
                <label for="autoGainControl">Auto Gain Control</label>
            </div>
        </div>
        <div id="callStatus" class="call-status"></div>
    </div>

    <div id="phoneInfo" class="phone-info hidden">
        <h3>Want to call from a phone number?</h3>
        <p>You can assign a SignalWire phone number to this agent. Click below to add a number in the dashboard.</p>
        <a id="dashboardLink" href="#" target="_blank">Add Phone Number in Dashboard</a>
    </div>

    <h2>Endpoints</h2>

    <div class="endpoint">
        <h3><span class="method post">POST</span> <span class="path">/swml</span></h3>
        <p>Main SWML endpoint for SignalWire to fetch agent configuration.</p>
        <div class="tabs">
            <div class="tab active" onclick="switchTab(this, 'swml-browser')">Browser</div>
            <div class="tab" onclick="switchTab(this, 'swml-curl')">curl</div>
        </div>
        <div id="swml-browser" class="tab-content active">
            <div class="browser-panel">
                <button class="try-btn" onclick="tryEndpoint('POST', '/swml', {{}}, 'swml-response', true)">Try it</button>
                <div id="swml-response" class="response-area"></div>
            </div>
        </div>
        <div id="swml-curl" class="tab-content">
            <div class="curl-panel">
                <button class="copy-btn" onclick="copyCode(this)">Copy</button>
                <pre><span class="comment"># Get the SWML configuration</span>
curl -X POST <span class="base-url"></span>/swml \\
  -u <span class="auth-creds"></span> \\
  -H "Content-Type: application/json" \\
  -d '{{}}'</pre>
            </div>
        </div>
    </div>

    <div class="endpoint">
        <h3><span class="method get">GET</span> <span class="path">/get_token</span></h3>
        <p>Get a guest token for WebRTC calls. Returns a token and call address.</p>
        <div class="tabs">
            <div class="tab active" onclick="switchTab(this, 'token-browser')">Browser</div>
            <div class="tab" onclick="switchTab(this, 'token-curl')">curl</div>
        </div>
        <div id="token-browser" class="tab-content active">
            <div class="browser-panel">
                <button class="try-btn" onclick="tryEndpoint('GET', '/get_token', null, 'token-response')">Try it</button>
                <div id="token-response" class="response-area"></div>
            </div>
        </div>
        <div id="token-curl" class="tab-content">
            <div class="curl-panel">
                <button class="copy-btn" onclick="copyCode(this)">Copy</button>
                <pre><span class="comment"># Get a guest token</span>
curl <span class="base-url"></span>/get_token</pre>
            </div>
        </div>
    </div>

    <div class="endpoint">
        <h3><span class="method post">POST</span> <span class="path">/swml/swaig/</span></h3>
        <p>SWAIG function endpoint. Test your agent's functions.</p>
        <div class="tabs">
            <div class="tab active" onclick="switchTab(this, 'swaig-browser')">Browser</div>
            <div class="tab" onclick="switchTab(this, 'swaig-curl')">curl</div>
        </div>
        <div id="swaig-browser" class="tab-content active">
            <div class="browser-panel">
                <button class="try-btn" onclick="tryEndpoint('POST', '/swml/swaig/', {{function: 'get_info', argument: {{parsed: [{{topic: 'SignalWire'}}]}}}}, 'swaig-response', true)">Try it</button>
                <div id="swaig-response" class="response-area"></div>
            </div>
        </div>
        <div id="swaig-curl" class="tab-content">
            <div class="curl-panel">
                <button class="copy-btn" onclick="copyCode(this)">Copy</button>
                <pre><span class="comment"># Call a SWAIG function</span>
curl -X POST <span class="base-url"></span>/swml/swaig/ \\
  -u <span class="auth-creds"></span> \\
  -H "Content-Type: application/json" \\
  -d '{{"function": "get_info", "argument": {{"parsed": [{{"topic": "SignalWire"}}]}}}}'</pre>
            </div>
        </div>
    </div>

    <div class="endpoint">
        <h3><span class="method get">GET</span> <span class="path">/health</span></h3>
        <p>Health check endpoint for load balancers and monitoring.</p>
    </div>

    <div class="footer">
        Powered by <a href="https://signalwire.com">SignalWire</a> and the
        <a href="https://github.com/signalwire/signalwire-agents">SignalWire Agents SDK</a>
    </div>

    <script src="https://cdn.signalwire.com/@signalwire/client"></script>
    <script>
        let authCreds = null;

        function switchTab(tabEl, contentId) {{
            const endpoint = tabEl.closest('.endpoint');
            endpoint.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            endpoint.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            tabEl.classList.add('active');
            document.getElementById(contentId).classList.add('active');
        }}

        function copyCode(btn) {{
            const pre = btn.parentElement.querySelector('pre');
            const text = pre.textContent;
            navigator.clipboard.writeText(text).then(() => {{
                const orig = btn.textContent;
                btn.textContent = 'Copied!';
                setTimeout(() => btn.textContent = orig, 1500);
            }});
        }}

        async function tryEndpoint(method, path, body, responseId, requiresAuth) {{
            const responseArea = document.getElementById(responseId);
            responseArea.classList.add('visible');
            responseArea.innerHTML = '<div class="response-status">Loading...</div>';

            const startTime = performance.now();
            try {{
                const options = {{
                    method: method,
                    headers: {{}}
                }};
                if (body) {{
                    options.headers['Content-Type'] = 'application/json';
                    options.body = JSON.stringify(body);
                }}
                if (requiresAuth && authCreds) {{
                    options.headers['Authorization'] = 'Basic ' + btoa(authCreds.user + ':' + authCreds.password);
                }}

                const resp = await fetch(path, options);
                const endTime = performance.now();
                const duration = Math.round(endTime - startTime);

                let data;
                const contentType = resp.headers.get('content-type') || '';
                if (contentType.includes('json')) {{
                    data = await resp.json();
                    data = JSON.stringify(data, null, 2);
                }} else {{
                    data = await resp.text();
                }}

                const statusClass = resp.ok ? 'success' : 'error';
                responseArea.innerHTML = `
                    <div class="response-header">
                        <span class="response-status ${{statusClass}}">${{resp.status}} ${{resp.statusText}}</span>
                        <span class="response-time">${{duration}}ms</span>
                    </div>
                    <div class="response-body">${{escapeHtml(data)}}</div>
                `;
            }} catch (err) {{
                responseArea.innerHTML = `
                    <div class="response-header">
                        <span class="response-status error">Error</span>
                    </div>
                    <div class="response-body">${{escapeHtml(err.message)}}</div>
                `;
            }}
        }}

        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}

        // WebRTC calling - robust pattern from santa app.js
        let client = null;
        let roomSession = null;
        let currentToken = null;
        let currentDestination = null;

        const connectBtn = document.getElementById('connectBtn');
        const disconnectBtn = document.getElementById('disconnectBtn');
        const destinationInput = document.getElementById('destination');
        const callStatus = document.getElementById('callStatus');

        function updateCallStatus(message) {{
            callStatus.textContent = message;
        }}

        async function connect() {{
            // Debounce - prevent double-clicks
            if (connectBtn.disabled) {{
                console.log('Call already in progress');
                return;
            }}
            connectBtn.disabled = true;
            connectBtn.textContent = 'Connecting...';

            try {{
                updateCallStatus('Getting token...');

                const tokenResp = await fetch('/get_token');
                const tokenData = await tokenResp.json();

                if (tokenData.error) {{
                    throw new Error(tokenData.error);
                }}

                currentToken = tokenData.token;
                currentDestination = tokenData.address;

                if (tokenData.address) {{
                    destinationInput.value = tokenData.address;
                }}

                console.log('Got token, destination:', currentDestination);
                updateCallStatus('Connecting...');

                // Initialize SignalWire client
                if (window.SignalWire && typeof window.SignalWire.SignalWire === 'function') {{
                    console.log('Initializing SignalWire client...');
                    client = await window.SignalWire.SignalWire({{
                        token: currentToken
                    }});
                }} else {{
                    console.error('SignalWire SDK structure:', window.SignalWire);
                    throw new Error('SignalWire.SignalWire function not found');
                }}

                const destination = currentDestination || destinationInput.value;
                roomSession = await client.dial({{
                    to: destination,
                    audio: {{
                        echoCancellation: document.getElementById('echoCancellation').checked,
                        noiseSuppression: document.getElementById('noiseSuppression').checked,
                        autoGainControl: document.getElementById('autoGainControl').checked
                    }},
                    video: false
                }});

                console.log('Room session created:', roomSession);

                roomSession.on('call.joined', () => {{
                    console.log('Call joined');
                    updateCallStatus('Connected');
                    disconnectBtn.disabled = false;
                }});

                roomSession.on('call.left', () => {{
                    console.log('Call left');
                    updateCallStatus('Call ended');
                    cleanup();
                }});

                roomSession.on('destroy', () => {{
                    console.log('Session destroyed');
                    updateCallStatus('Call ended');
                    cleanup();
                }});

                roomSession.on('room.left', () => {{
                    console.log('Room left');
                    cleanup();
                }});

                await roomSession.start();
                console.log('Call started successfully');

                // Update UI
                connectBtn.style.display = 'none';
                disconnectBtn.style.display = 'inline-block';

            }} catch (err) {{
                console.error('Connection error:', err);
                updateCallStatus('Error: ' + err.message);
                cleanup();
            }}
        }}

        async function disconnect() {{
            console.log('Disconnect called');
            await hangup();
        }}

        async function hangup() {{
            try {{
                if (roomSession) {{
                    console.log('Hanging up call...');
                    await roomSession.hangup();
                    console.log('Call hung up successfully');
                }}
            }} catch (err) {{
                console.error('Hangup error:', err);
            }}
            cleanup();
        }}

        function cleanup() {{
            console.log('Cleanup called');

            // Clean up local stream if it exists
            if (roomSession && roomSession.localStream) {{
                console.log('Stopping local stream tracks');
                roomSession.localStream.getTracks().forEach(track => {{
                    track.stop();
                }});
            }}

            roomSession = null;

            // Disconnect the client properly
            if (client) {{
                try {{
                    console.log('Disconnecting client');
                    client.disconnect();
                }} catch (e) {{
                    console.log('Client disconnect error:', e);
                }}
                client = null;
            }}

            // Reset UI
            connectBtn.disabled = false;
            connectBtn.textContent = 'Call Agent';
            connectBtn.style.display = 'inline-block';
            disconnectBtn.disabled = true;
            disconnectBtn.style.display = 'none';
        }}

        connectBtn.addEventListener('click', connect);
        disconnectBtn.addEventListener('click', disconnect);

        // Clean up on page unload
        window.addEventListener('beforeunload', () => {{
            if (roomSession) {{
                hangup();
            }}
        }});

        // Initialize on load
        document.addEventListener('DOMContentLoaded', async function() {{
            const baseUrl = window.location.origin;
            document.querySelectorAll('.base-url').forEach(function(el) {{
                el.textContent = baseUrl;
            }});

            // Fetch auth credentials
            try {{
                const credsResp = await fetch('/get_credentials');
                if (credsResp.ok) {{
                    authCreds = await credsResp.json();
                    document.querySelectorAll('.auth-creds').forEach(function(el) {{
                        el.textContent = authCreds.user + ':' + authCreds.password;
                    }});
                }}
            }} catch (e) {{
                document.querySelectorAll('.auth-creds').forEach(function(el) {{
                    el.textContent = 'user:password';
                }});
            }}

            // Fetch resource info for dashboard link
            try {{
                const resourceResp = await fetch('/get_resource_info');
                if (resourceResp.ok) {{
                    const resourceInfo = await resourceResp.json();
                    if (resourceInfo.dashboard_url) {{
                        document.getElementById('dashboardLink').href = resourceInfo.dashboard_url;
                        document.getElementById('phoneInfo').classList.remove('hidden');
                    }}
                }}
            }} catch (e) {{
                console.log('Could not fetch resource info:', e);
            }}
        }});
    </script>
</body>
</html>
'''

APP_JSON_TEMPLATE = '''{{
  "name": "{app_name}",
  "description": "SignalWire AI Agent with WebRTC calling support",
  "keywords": ["signalwire", "ai", "agent", "python", "webrtc"],
  "env": {{
    "APP_ENV": {{
      "description": "Application environment",
      "value": "production"
    }},
    "AGENT_NAME": {{
      "description": "Name for the SWML handler resource",
      "value": "{app_name}"
    }},
    "SIGNALWIRE_SPACE_NAME": {{
      "description": "SignalWire space name (e.g., 'myspace' or 'myspace.signalwire.com')",
      "required": true
    }},
    "SIGNALWIRE_PROJECT_ID": {{
      "description": "SignalWire project ID",
      "required": true
    }},
    "SIGNALWIRE_TOKEN": {{
      "description": "SignalWire API token",
      "required": true
    }},
    "SWML_PROXY_URL_BASE": {{
      "description": "Public URL base for SWML callbacks (e.g., 'https://myapp.example.com')",
      "required": true
    }},
    "SWML_BASIC_AUTH_USER": {{
      "description": "Basic auth username for SWML endpoints",
      "required": true
    }},
    "SWML_BASIC_AUTH_PASSWORD": {{
      "description": "Basic auth password for SWML endpoints",
      "required": true
    }}
  }},
  "formation": {{
    "web": {{
      "quantity": 1
    }}
  }},
  "buildpacks": [
    {{
      "url": "heroku/python"
    }}
  ],
  "healthchecks": {{
    "web": [
      {{
        "type": "startup",
        "name": "port listening",
        "listening": true,
        "attempts": 10,
        "wait": 5,
        "timeout": 60
      }}
    ]
  }}
}}
'''

# =============================================================================
# Templates - Simple Mode
# =============================================================================

DEPLOY_SCRIPT_TEMPLATE = '''#!/bin/bash
# Dokku deployment helper for {app_name}
set -e

APP_NAME="${{1:-{app_name}}}"
DOKKU_HOST="${{2:-{dokku_host}}}"

echo "═══════════════════════════════════════════════════════════"
echo "  Deploying $APP_NAME to $DOKKU_HOST"
echo "═══════════════════════════════════════════════════════════"

# Initialize git if needed
if [ ! -d .git ]; then
    echo "→ Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit"
fi

# Create app if it doesn't exist
echo "→ Creating app (if not exists)..."
ssh dokku@$DOKKU_HOST apps:create $APP_NAME 2>/dev/null || true

# Set environment variables
echo "→ Setting environment variables..."
AUTH_PASS=$(openssl rand -base64 24 | tr -d '/+=' | head -c 24)
ssh dokku@$DOKKU_HOST config:set --no-restart $APP_NAME \\
    APP_ENV=production \\
    APP_NAME=$APP_NAME \\
    SWML_BASIC_AUTH_USER=admin \\
    SWML_BASIC_AUTH_PASSWORD=$AUTH_PASS

# Add dokku remote
echo "→ Configuring git remote..."
git remote add dokku dokku@$DOKKU_HOST:$APP_NAME 2>/dev/null || \\
git remote set-url dokku dokku@$DOKKU_HOST:$APP_NAME

# Deploy
echo "→ Pushing to Dokku..."
git push dokku main --force

# Enable SSL
echo "→ Enabling Let's Encrypt SSL..."
ssh dokku@$DOKKU_HOST letsencrypt:enable $APP_NAME 2>/dev/null || \\
echo "  (SSL setup may require manual configuration)"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ Deployment complete!"
echo ""
echo "  🌐 URL: https://$APP_NAME.$DOKKU_HOST"
echo "  🔑 Auth: admin / $AUTH_PASS"
echo ""
echo "  Configure SignalWire phone number SWML URL to:"
echo "  https://admin:$AUTH_PASS@$APP_NAME.$DOKKU_HOST/{route}"
echo "═══════════════════════════════════════════════════════════"
'''

README_SIMPLE_TEMPLATE = '''# {app_name}

A SignalWire AI Agent deployed to Dokku.

## Quick Deploy

```bash
./deploy.sh {app_name} {dokku_host}
```

## Manual Deployment

1. **Create the app:**
   ```bash
   ssh dokku@{dokku_host} apps:create {app_name}
   ```

2. **Set environment variables:**
   ```bash
   ssh dokku@{dokku_host} config:set {app_name} \\
     SWML_BASIC_AUTH_USER=admin \\
     SWML_BASIC_AUTH_PASSWORD=secure-password \\
     APP_ENV=production
   ```

3. **Add git remote and deploy:**
   ```bash
   git remote add dokku dokku@{dokku_host}:{app_name}
   git push dokku main
   ```

4. **Enable SSL:**
   ```bash
   ssh dokku@{dokku_host} letsencrypt:enable {app_name}
   ```

## Usage

Your agent is available at: `https://{app_name}.{dokku_host_domain}`

Configure your SignalWire phone number:
- **SWML URL:** `https://{app_name}.{dokku_host_domain}/{route}`
- **Auth:** Basic auth with your configured credentials

## Useful Commands

```bash
# View logs
ssh dokku@{dokku_host} logs {app_name} -t

# Restart app
ssh dokku@{dokku_host} ps:restart {app_name}

# View environment variables
ssh dokku@{dokku_host} config:show {app_name}

# Scale workers
ssh dokku@{dokku_host} ps:scale {app_name} web=2

# Rollback to previous release
ssh dokku@{dokku_host} releases:rollback {app_name}
```

## Local Development

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8080
```

Test with swaig-test:
```bash
swaig-test app.py --list-tools
```
'''

# =============================================================================
# Templates - CI/CD Mode
# =============================================================================

DEPLOY_WORKFLOW_TEMPLATE = '''# Deploy to Dokku - calls reusable workflow from dokku-deploy-system
name: Deploy

on:
  workflow_dispatch:
  push:
    branches: [main, staging, develop]

permissions:
  contents: read
  deployments: write

concurrency:
  group: deploy-${{{{ github.ref }}}}
  cancel-in-progress: true

jobs:
  deploy:
    uses: signalwire-demos/dokku-deploy-system/.github/workflows/deploy.yml@main
    secrets: inherit
    # Optional: customize health check path
    # with:
    #   health_check_path: '/health'
'''

PREVIEW_WORKFLOW_TEMPLATE = '''# Preview environments for pull requests - calls reusable workflow from dokku-deploy-system
name: Preview

on:
  pull_request:
    types: [opened, synchronize, reopened, closed]

concurrency:
  group: preview-${{{{ github.event.pull_request.number }}}}

jobs:
  preview:
    uses: signalwire-demos/dokku-deploy-system/.github/workflows/preview.yml@main
    secrets: inherit
    # Optional: customize memory limit for previews
    # with:
    #   memory_limit: '256m'
'''

DOKKU_CONFIG_TEMPLATE = '''# ═══════════════════════════════════════════════════════════════════════════════
# Dokku App Configuration
# ═══════════════════════════════════════════════════════════════════════════════
#
# Configuration for your Dokku app deployment.
# These settings are applied during the deployment workflow.
#
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# Resource Limits
# ─────────────────────────────────────────────────────────────────────────────
# Memory: 256m, 512m, 1g, 2g, etc.
# CPU: Number of cores (can be fractional, e.g., 0.5)
resources:
  memory: 512m
  cpu: 1

# ─────────────────────────────────────────────────────────────────────────────
# Health Check
# ─────────────────────────────────────────────────────────────────────────────
# Path that returns 200 OK when app is healthy
healthcheck:
  path: /health
  timeout: 30
  attempts: 5

# ─────────────────────────────────────────────────────────────────────────────
# Scaling
# ─────────────────────────────────────────────────────────────────────────────
# Number of web workers (dynos)
scale:
  web: 1
  # worker: 1  # Uncomment for background workers

# ─────────────────────────────────────────────────────────────────────────────
# Custom Domains
# ─────────────────────────────────────────────────────────────────────────────
# Additional domains for this app (requires DNS configuration)
# custom_domains:
#   - www.example.com
#   - api.example.com

# ─────────────────────────────────────────────────────────────────────────────
# Environment-Specific Overrides
# ─────────────────────────────────────────────────────────────────────────────
environments:
  production:
    resources:
      memory: 1g
      cpu: 2
    scale:
      web: 2

  staging:
    resources:
      memory: 512m
      cpu: 1

  preview:
    resources:
      memory: 256m
      cpu: 0.5
'''

SERVICES_TEMPLATE = '''# ═══════════════════════════════════════════════════════════════════════════════
# Dokku Services Configuration
# ═══════════════════════════════════════════════════════════════════════════════
#
# Define which backing services your app needs.
# Services are automatically provisioned and linked during deployment.
#
# When a service is linked, its connection URL is automatically
# injected as an environment variable (e.g., DATABASE_URL, REDIS_URL).
#
# ═══════════════════════════════════════════════════════════════════════════════

services:
  # ─────────────────────────────────────────────────────────────────────────────
  # PostgreSQL Database
  # ─────────────────────────────────────────────────────────────────────────────
  # Environment variable: DATABASE_URL
  # Format: postgres://user:pass@host:5432/database
  postgres:
    enabled: false  # Set to true to enable
    environments:
      production:
        # Production gets its own dedicated database
        dedicated: true
      staging:
        # Staging gets its own database
        dedicated: true
      preview:
        # All preview apps share a single database to save resources
        shared: true

  # ─────────────────────────────────────────────────────────────────────────────
  # Redis Cache/Queue
  # ─────────────────────────────────────────────────────────────────────────────
  # Environment variable: REDIS_URL
  # Format: redis://host:6379
  redis:
    enabled: false  # Set to true to enable
    environments:
      production:
        dedicated: true
      staging:
        dedicated: true
      preview:
        shared: true

  # ─────────────────────────────────────────────────────────────────────────────
  # MySQL Database
  # ─────────────────────────────────────────────────────────────────────────────
  # Environment variable: DATABASE_URL
  # Format: mysql://user:pass@host:3306/database
  mysql:
    enabled: false
    environments:
      preview:
        shared: true

  # ─────────────────────────────────────────────────────────────────────────────
  # MongoDB
  # ─────────────────────────────────────────────────────────────────────────────
  # Environment variable: MONGO_URL
  # Format: mongodb://user:pass@host:27017/database
  mongo:
    enabled: false
    environments:
      preview:
        shared: true

  # ─────────────────────────────────────────────────────────────────────────────
  # RabbitMQ Message Queue
  # ─────────────────────────────────────────────────────────────────────────────
  # Environment variable: RABBITMQ_URL
  # Format: amqp://user:pass@host:5672
  rabbitmq:
    enabled: false
    environments:
      preview:
        # Don't provision RabbitMQ for previews (too expensive)
        enabled: false

  # ─────────────────────────────────────────────────────────────────────────────
  # Elasticsearch
  # ─────────────────────────────────────────────────────────────────────────────
  # Environment variable: ELASTICSEARCH_URL
  # Format: http://host:9200
  elasticsearch:
    enabled: false
    environments:
      preview:
        enabled: false

# ═══════════════════════════════════════════════════════════════════════════════
# External/Managed Services
# ═══════════════════════════════════════════════════════════════════════════════
#
# For production, you may want to use managed services like AWS RDS,
# ElastiCache, etc. Define the connection URLs here (reference GitHub secrets).
#
# These override the Dokku-managed services for the specified environment.
#
# ═══════════════════════════════════════════════════════════════════════════════

# external_services:
#   production:
#     DATABASE_URL: "${secrets.PROD_DATABASE_URL}"
#     REDIS_URL: "${secrets.PROD_REDIS_URL}"
#   staging:
#     DATABASE_URL: "${secrets.STAGING_DATABASE_URL}"
'''

README_CICD_TEMPLATE = '''# {app_name}

A SignalWire AI Agent with automated GitHub → Dokku deployments.

## Features

- ✅ Auto-deploy on push to main/staging/develop
- ✅ Preview environments for pull requests
- ✅ Automatic SSL via Let's Encrypt
- ✅ Zero-downtime deployments
- ✅ Multi-environment support

## Setup

### 1. GitHub Secrets

Add these secrets to your repository (Settings → Secrets → Actions):

| Secret | Description |
|--------|-------------|
| `DOKKU_HOST` | Your Dokku server hostname |
| `DOKKU_SSH_PRIVATE_KEY` | SSH private key for deployments |
| `BASE_DOMAIN` | Base domain (e.g., `yourdomain.com`) |
| `SWML_BASIC_AUTH_USER` | Basic auth username |
| `SWML_BASIC_AUTH_PASSWORD` | Basic auth password |

### 2. GitHub Environments

Create these environments (Settings → Environments):
- `production` - Deploy from `main` branch
- `staging` - Deploy from `staging` branch
- `development` - Deploy from `develop` branch
- `preview` - Deploy from pull requests

### 3. Deploy

Just push to a branch:

```bash
git push origin main      # → {app_name}.yourdomain.com
git push origin staging   # → {app_name}-staging.yourdomain.com
git push origin develop   # → {app_name}-dev.yourdomain.com
```

Or open a PR for a preview environment.

## Branch → Environment Mapping

| Branch | App Name | URL |
|--------|----------|-----|
| `main` | `{app_name}` | `{app_name}.yourdomain.com` |
| `staging` | `{app_name}-staging` | `{app_name}-staging.yourdomain.com` |
| `develop` | `{app_name}-dev` | `{app_name}-dev.yourdomain.com` |
| PR #42 | `{app_name}-pr-42` | `{app_name}-pr-42.yourdomain.com` |

## Manual Operations

```bash
# View logs
ssh dokku@server logs {app_name} -t

# SSH into container
ssh dokku@server enter {app_name}

# Restart
ssh dokku@server ps:restart {app_name}

# Rollback
ssh dokku@server releases:rollback {app_name}

# Scale
ssh dokku@server ps:scale {app_name} web=2
```

## Local Development

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8080
```

Test with swaig-test:
```bash
swaig-test app.py --list-tools
```
'''


# =============================================================================
# Project Generator
# =============================================================================

class DokkuProjectGenerator:
    """Generates Dokku deployment files for SignalWire agents."""

    def __init__(self, app_name: str, options: Dict[str, Any]):
        self.app_name = app_name
        self.options = options
        self.project_dir = Path(options.get('project_dir', f'./{app_name}'))

        # Derived names
        self.agent_slug = app_name.lower().replace(' ', '-').replace('_', '-')
        self.agent_class = ''.join(
            word.capitalize()
            for word in app_name.replace('-', ' ').replace('_', ' ').split()
        ) + 'Agent'

    def generate(self) -> bool:
        """Generate the project files."""
        try:
            self.project_dir.mkdir(parents=True, exist_ok=True)
            print_success(f"Created {self.project_dir}/")

            # Core files (both modes)
            self._write_core_files()

            # Mode-specific files
            if self.options.get('cicd'):
                self._write_cicd_files()
            else:
                self._write_simple_files()

            return True
        except Exception as e:
            print_error(f"Failed to generate project: {e}")
            return False

    def _write_core_files(self):
        """Write files common to both modes."""
        # Procfile
        self._write_file('Procfile', PROCFILE_TEMPLATE)

        # runtime.txt
        self._write_file('runtime.txt', RUNTIME_TEMPLATE)

        # requirements.txt
        self._write_file('requirements.txt', REQUIREMENTS_TEMPLATE)

        # CHECKS
        self._write_file('CHECKS', CHECKS_TEMPLATE)

        # .gitignore
        self._write_file('.gitignore', GITIGNORE_TEMPLATE)

        # .env.example
        self._write_file('.env.example', ENV_EXAMPLE_TEMPLATE.format(
            app_name=self.app_name
        ))

        # app.json
        self._write_file('app.json', APP_JSON_TEMPLATE.format(
            app_name=self.app_name
        ))

        # app.py - use web template if web option is enabled
        if self.options.get('web'):
            self._write_file('app.py', APP_TEMPLATE_WITH_WEB.format(
                agent_name=self.app_name,
                agent_class=self.agent_class,
                agent_slug=self.agent_slug
            ))
            self._write_web_files()
        else:
            self._write_file('app.py', APP_TEMPLATE.format(
                agent_name=self.app_name,
                agent_class=self.agent_class,
                agent_slug=self.agent_slug
            ))

    def _write_web_files(self):
        """Write web interface files."""
        # Create web directory
        web_dir = self.project_dir / 'web'
        web_dir.mkdir(parents=True, exist_ok=True)

        route = self.options.get('route', 'swaig')

        # index.html
        self._write_file('web/index.html', WEB_INDEX_TEMPLATE.format(
            agent_name=self.app_name,
            route=route
        ))

    def _write_simple_files(self):
        """Write files for simple deployment mode."""
        dokku_host = self.options.get('dokku_host', 'dokku.yourdomain.com')
        route = self.options.get('route', 'swaig')

        # deploy.sh
        deploy_script = DEPLOY_SCRIPT_TEMPLATE.format(
            app_name=self.app_name,
            dokku_host=dokku_host,
            route=route
        )
        self._write_file('deploy.sh', deploy_script, executable=True)

        # README.md
        readme = README_SIMPLE_TEMPLATE.format(
            app_name=self.app_name,
            dokku_host=dokku_host,
            dokku_host_domain=dokku_host.replace('dokku.', ''),
            route=route
        )
        self._write_file('README.md', readme)

    def _write_cicd_files(self):
        """Write files for CI/CD deployment mode."""
        # Create .github/workflows directory
        workflows_dir = self.project_dir / '.github' / 'workflows'
        workflows_dir.mkdir(parents=True, exist_ok=True)

        # deploy.yml
        deploy_workflow = DEPLOY_WORKFLOW_TEMPLATE.format(app_name=self.app_name)
        self._write_file('.github/workflows/deploy.yml', deploy_workflow)

        # preview.yml
        preview_workflow = PREVIEW_WORKFLOW_TEMPLATE.format(app_name=self.app_name)
        self._write_file('.github/workflows/preview.yml', preview_workflow)

        # Create .dokku directory
        dokku_dir = self.project_dir / '.dokku'
        dokku_dir.mkdir(parents=True, exist_ok=True)

        # config.yml
        self._write_file('.dokku/config.yml', DOKKU_CONFIG_TEMPLATE)

        # services.yml
        self._write_file('.dokku/services.yml', SERVICES_TEMPLATE)

        # README.md
        readme = README_CICD_TEMPLATE.format(app_name=self.app_name)
        self._write_file('README.md', readme)

    def _write_file(self, path: str, content: str, executable: bool = False):
        """Write a file to the project directory."""
        file_path = self.project_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

        if executable:
            file_path.chmod(0o755)

        print_success(f"Created {path}")


# =============================================================================
# CLI Commands
# =============================================================================

def cmd_init(args):
    """Initialize a new Dokku project."""
    app_name = args.name

    print_header(f"Creating Dokku project: {app_name}")

    # Gather options
    options = {
        'project_dir': args.dir or f'./{app_name}',
        'cicd': args.cicd,
        'dokku_host': args.host or 'dokku.yourdomain.com',
        'route': 'swaig',
        'web': args.web,
    }

    # Interactive mode if not all options provided
    if not args.host and not args.cicd:
        print("\n")
        if prompt_yes_no("Enable GitHub Actions CI/CD?", default=False):
            options['cicd'] = True
        else:
            options['dokku_host'] = prompt("Dokku server hostname", "dokku.yourdomain.com")

        # Ask about web interface if not specified
        if not args.web:
            options['web'] = prompt_yes_no("Include web interface (static files at /)?", default=True)

    # Check if directory exists
    project_dir = Path(options['project_dir'])
    if project_dir.exists():
        if not args.force:
            if not prompt_yes_no(f"Directory {project_dir} exists. Overwrite?", default=False):
                print("Aborted.")
                return 1
        shutil.rmtree(project_dir)

    # Generate project
    generator = DokkuProjectGenerator(app_name, options)
    if generator.generate():
        print(f"\n{Colors.GREEN}{Colors.BOLD}Project created successfully!{Colors.NC}\n")

        if options['cicd']:
            _print_cicd_instructions(app_name)
        else:
            _print_simple_instructions(app_name, options['dokku_host'], project_dir)

        return 0
    return 1


def _print_simple_instructions(app_name: str, dokku_host: str, project_dir: Path):
    """Print instructions for simple mode."""
    print(f"""To deploy your agent:

  {Colors.CYAN}cd {project_dir}{Colors.NC}
  {Colors.CYAN}./deploy.sh{Colors.NC}

Or manually:

  {Colors.DIM}git init && git add . && git commit -m "Initial commit"
  git remote add dokku dokku@{dokku_host}:{app_name}
  git push dokku main{Colors.NC}
""")


def _print_cicd_instructions(app_name: str):
    """Print instructions for CI/CD mode."""
    print(f"""
{Colors.BOLD}═══════════════════════════════════════════════════════════{Colors.NC}
{Colors.BOLD}  CI/CD Setup Instructions{Colors.NC}
{Colors.BOLD}═══════════════════════════════════════════════════════════{Colors.NC}

1. Push this repository to GitHub

2. Add these secrets to your GitHub repository:
   (Settings → Secrets → Actions)

   • {Colors.CYAN}DOKKU_HOST{Colors.NC}              - Your Dokku server hostname
   • {Colors.CYAN}DOKKU_SSH_PRIVATE_KEY{Colors.NC}  - SSH key for deployments
   • {Colors.CYAN}BASE_DOMAIN{Colors.NC}            - Base domain (e.g., yourdomain.com)
   • {Colors.CYAN}SWML_BASIC_AUTH_USER{Colors.NC}   - Basic auth username
   • {Colors.CYAN}SWML_BASIC_AUTH_PASSWORD{Colors.NC} - Basic auth password

3. Create GitHub environments:
   (Settings → Environments)

   • production
   • staging
   • development
   • preview

4. Push to deploy:

   {Colors.CYAN}git push origin main{Colors.NC}  # Deploys to production

5. Open a PR for preview environments!

{Colors.BOLD}═══════════════════════════════════════════════════════════{Colors.NC}
""")


def cmd_deploy(args):
    """Deploy to Dokku."""
    # Check if we're in a Dokku project
    if not Path('Procfile').exists():
        print_error("No Procfile found. Are you in a Dokku project directory?")
        print("Run 'sw-agent-dokku init <name>' to create a new project.")
        return 1

    dokku_host = args.host
    app_name = args.app

    # Try to get app name from app.json
    if not app_name and Path('app.json').exists():
        import json
        try:
            with open('app.json') as f:
                app_json = json.load(f)
                app_name = app_json.get('name')
        except:
            pass

    if not app_name:
        app_name = prompt("App name", Path.cwd().name)

    if not dokku_host:
        dokku_host = prompt("Dokku host", "dokku.yourdomain.com")

    print_header(f"Deploying {app_name} to {dokku_host}")

    # Check git status
    if not Path('.git').exists():
        print_step("Initializing git repository...")
        subprocess.run(['git', 'init'], check=True)
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], check=True)

    # Create app
    print_step("Creating app (if not exists)...")
    subprocess.run(
        ['ssh', f'dokku@{dokku_host}', 'apps:create', app_name],
        capture_output=True
    )

    # Set up git remote
    print_step("Configuring git remote...")
    remote_url = f'dokku@{dokku_host}:{app_name}'
    subprocess.run(['git', 'remote', 'remove', 'dokku'], capture_output=True)
    subprocess.run(['git', 'remote', 'add', 'dokku', remote_url], check=True)

    # Deploy
    print_step("Pushing to Dokku...")
    result = subprocess.run(
        ['git', 'push', 'dokku', 'HEAD:main', '--force'],
        capture_output=False
    )

    if result.returncode == 0:
        print_success(f"Deployed to https://{app_name}.{dokku_host.replace('dokku.', '')}")
    else:
        print_error("Deployment failed")
        return 1

    return 0


def cmd_logs(args):
    """Tail Dokku logs."""
    app_name = args.app
    dokku_host = args.host

    if not app_name:
        app_name = _get_app_name()
    if not dokku_host:
        dokku_host = prompt("Dokku host", "dokku.yourdomain.com")

    print_header(f"Tailing logs for {app_name}")

    cmd = ['ssh', f'dokku@{dokku_host}', 'logs', app_name]
    if args.tail:
        cmd.append('-t')
    if args.num:
        cmd.extend(['--num', str(args.num)])

    subprocess.run(cmd)
    return 0


def cmd_config(args):
    """Manage Dokku config."""
    app_name = args.app
    dokku_host = args.host

    if not app_name:
        app_name = _get_app_name()
    if not dokku_host:
        dokku_host = prompt("Dokku host", "dokku.yourdomain.com")

    if args.config_action == 'show':
        subprocess.run(['ssh', f'dokku@{dokku_host}', 'config:show', app_name])
    elif args.config_action == 'set':
        if not args.vars:
            print_error("No variables provided. Use: sw-agent-dokku config set KEY=value")
            return 1
        cmd = ['ssh', f'dokku@{dokku_host}', 'config:set', app_name] + args.vars
        subprocess.run(cmd)
    elif args.config_action == 'unset':
        if not args.vars:
            print_error("No variables provided. Use: sw-agent-dokku config unset KEY")
            return 1
        cmd = ['ssh', f'dokku@{dokku_host}', 'config:unset', app_name] + args.vars
        subprocess.run(cmd)

    return 0


def cmd_scale(args):
    """Scale Dokku processes."""
    app_name = args.app
    dokku_host = args.host

    if not app_name:
        app_name = _get_app_name()
    if not dokku_host:
        dokku_host = prompt("Dokku host", "dokku.yourdomain.com")

    if not args.scale_args:
        # Show current scale
        subprocess.run(['ssh', f'dokku@{dokku_host}', 'ps:scale', app_name])
    else:
        # Set scale
        cmd = ['ssh', f'dokku@{dokku_host}', 'ps:scale', app_name] + args.scale_args
        subprocess.run(cmd)

    return 0


def _get_app_name() -> str:
    """Try to get app name from app.json or prompt."""
    if Path('app.json').exists():
        import json
        try:
            with open('app.json') as f:
                return json.load(f).get('name', '')
        except:
            pass
    return prompt("App name", Path.cwd().name)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SignalWire Agent Dokku Deployment Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  sw-agent-dokku init myagent                    # Create simple project (with prompts)
  sw-agent-dokku init myagent --web              # Create with web interface at /
  sw-agent-dokku init myagent --cicd             # Create with CI/CD workflows
  sw-agent-dokku init myagent --host dokku.example.com
  sw-agent-dokku deploy                          # Deploy current directory
  sw-agent-dokku logs -t                         # Tail logs
  sw-agent-dokku config show                     # Show config
  sw-agent-dokku config set KEY=value            # Set config
  sw-agent-dokku scale web=2                     # Scale processes
'''
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # init command
    init_parser = subparsers.add_parser('init', help='Initialize a new Dokku project')
    init_parser.add_argument('name', help='Project/app name')
    init_parser.add_argument('--cicd', action='store_true',
                            help='Include GitHub Actions CI/CD workflows')
    init_parser.add_argument('--web', action='store_true',
                            help='Include web interface (static files at /)')
    init_parser.add_argument('--host', help='Dokku server hostname')
    init_parser.add_argument('--dir', help='Project directory')
    init_parser.add_argument('--force', '-f', action='store_true',
                            help='Overwrite existing directory')

    # deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy to Dokku')
    deploy_parser.add_argument('--app', '-a', help='App name')
    deploy_parser.add_argument('--host', '-H', help='Dokku server hostname')

    # logs command
    logs_parser = subparsers.add_parser('logs', help='View Dokku logs')
    logs_parser.add_argument('--app', '-a', help='App name')
    logs_parser.add_argument('--host', '-H', help='Dokku server hostname')
    logs_parser.add_argument('--tail', '-t', action='store_true', help='Tail logs')
    logs_parser.add_argument('--num', '-n', type=int, help='Number of lines')

    # config command
    config_parser = subparsers.add_parser('config', help='Manage config variables')
    config_parser.add_argument('config_action', choices=['show', 'set', 'unset'],
                              help='Config action')
    config_parser.add_argument('vars', nargs='*', help='Variables (KEY=value)')
    config_parser.add_argument('--app', '-a', help='App name')
    config_parser.add_argument('--host', '-H', help='Dokku server hostname')

    # scale command
    scale_parser = subparsers.add_parser('scale', help='Scale processes')
    scale_parser.add_argument('scale_args', nargs='*', help='Scale args (web=2)')
    scale_parser.add_argument('--app', '-a', help='App name')
    scale_parser.add_argument('--host', '-H', help='Dokku server hostname')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        'init': cmd_init,
        'deploy': cmd_deploy,
        'logs': cmd_logs,
        'config': cmd_config,
        'scale': cmd_scale,
    }

    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
