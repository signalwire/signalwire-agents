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

REQUIREMENTS_TEMPLATE = """signalwire-agents>=1.0.12
gunicorn>=21.0.0
uvicorn>=0.24.0
python-dotenv>=1.0.0
"""

CHECKS_TEMPLATE = """WAIT=5
TIMEOUT=30
ATTEMPTS=5

/health
/ready
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
SWML_BASIC_AUTH_USER=admin
SWML_BASIC_AUTH_PASSWORD=your-secure-password

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
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from signalwire_agents import AgentBase, SwaigFunctionResult
from fastapi.staticfiles import StaticFiles

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

# Get the FastAPI app
app = agent.get_app()

# Mount static files from web/ directory at root
web_dir = Path(__file__).parent / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="static")

if __name__ == "__main__":
    agent.run()
'''

WEB_INDEX_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{agent_name}</title>
    <style>
        :root {{
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --bg: #f8fafc;
            --card: #ffffff;
            --text: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }}
        .container {{
            max-width: 600px;
            width: 100%;
        }}
        .card {{
            background: var(--card);
            border-radius: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            padding: 2rem;
            text-align: center;
        }}
        .logo {{
            width: 80px;
            height: 80px;
            background: var(--primary);
            border-radius: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1.5rem;
        }}
        .logo svg {{
            width: 48px;
            height: 48px;
            fill: white;
        }}
        h1 {{
            font-size: 1.75rem;
            margin-bottom: 0.5rem;
        }}
        .subtitle {{
            color: var(--text-muted);
            margin-bottom: 2rem;
        }}
        .status {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: #dcfce7;
            color: #166534;
            border-radius: 2rem;
            font-size: 0.875rem;
            font-weight: 500;
        }}
        .status::before {{
            content: '';
            width: 8px;
            height: 8px;
            background: #22c55e;
            border-radius: 50%;
        }}
        .endpoints {{
            margin-top: 2rem;
            text-align: left;
            border-top: 1px solid var(--border);
            padding-top: 1.5rem;
        }}
        .endpoints h2 {{
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 1rem;
        }}
        .endpoint {{
            display: flex;
            align-items: center;
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--border);
        }}
        .endpoint:last-child {{
            border-bottom: none;
        }}
        .method {{
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            margin-right: 1rem;
            min-width: 50px;
            text-align: center;
        }}
        .method.get {{
            background: #dbeafe;
            color: #1d4ed8;
        }}
        .method.post {{
            background: #dcfce7;
            color: #166534;
        }}
        .path {{
            font-family: monospace;
            color: var(--text);
        }}
        .desc {{
            margin-left: auto;
            color: var(--text-muted);
            font-size: 0.875rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <div class="logo">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                </svg>
            </div>
            <h1>{agent_name}</h1>
            <p class="subtitle">SignalWire AI Agent</p>
            <span class="status">Running on Dokku</span>

            <div class="endpoints">
                <h2>API Endpoints</h2>
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span class="path">/health</span>
                    <span class="desc">Health check</span>
                </div>
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <span class="path">/ready</span>
                    <span class="desc">Readiness check</span>
                </div>
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <span class="path">/{route}</span>
                    <span class="desc">SWML endpoint</span>
                </div>
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <span class="path">/{route}/swaig</span>
                    <span class="desc">SWAIG functions</span>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
'''

APP_JSON_TEMPLATE = '''{{
  "name": "{app_name}",
  "description": "SignalWire AI Agent",
  "keywords": ["signalwire", "ai", "agent", "python"],
  "env": {{
    "APP_ENV": {{
      "description": "Application environment",
      "value": "production"
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
        "name": "web check",
        "path": "/health"
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

DEPLOY_WORKFLOW_TEMPLATE = '''# Auto-deploy to Dokku on push
name: Deploy

on:
  workflow_dispatch:
  push:
    branches: [main, staging, develop]

concurrency:
  group: deploy-${{{{ github.ref }}}}
  cancel-in-progress: true

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: ${{{{ github.ref_name == 'main' && 'production' || github.ref_name == 'staging' && 'staging' || 'development' }}}}
    env:
      BASE_APP_NAME: ${{{{ github.event.repository.name }}}}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set variables
        id: vars
        run: |
          BRANCH="${{GITHUB_REF#refs/heads/}}"
          case "$BRANCH" in
            main)    APP="${{BASE_APP_NAME}}"; ENV="production" ;;
            staging) APP="${{BASE_APP_NAME}}-staging"; ENV="staging" ;;
            develop) APP="${{BASE_APP_NAME}}-dev"; ENV="development" ;;
            *)       APP="${{BASE_APP_NAME}}"; ENV="production" ;;
          esac
          echo "app_name=$APP" >> $GITHUB_OUTPUT
          echo "environment=$ENV" >> $GITHUB_OUTPUT

      - name: Setup SSH
        run: |
          mkdir -p ~/.ssh && chmod 700 ~/.ssh
          echo "${{{{ secrets.DOKKU_SSH_PRIVATE_KEY }}}}" > ~/.ssh/key && chmod 600 ~/.ssh/key
          ssh-keyscan -H ${{{{ secrets.DOKKU_HOST }}}} >> ~/.ssh/known_hosts
          echo -e "Host dokku\\n  HostName ${{{{ secrets.DOKKU_HOST }}}}\\n  User dokku\\n  IdentityFile ~/.ssh/key" > ~/.ssh/config

      - name: Create app
        run: |
          APP_NAME="${{{{ steps.vars.outputs.app_name }}}}"
          ssh dokku apps:exists $APP_NAME 2>/dev/null || ssh dokku apps:create $APP_NAME

      - name: Configure
        run: |
          APP_NAME="${{{{ steps.vars.outputs.app_name }}}}"
          DOMAIN="${{APP_NAME}}.${{{{ secrets.BASE_DOMAIN }}}}"
          ssh dokku config:set --no-restart $APP_NAME \\
            APP_ENV="${{{{ steps.vars.outputs.environment }}}}" \\
            APP_URL="https://${{DOMAIN}}"
          ssh dokku domains:clear $APP_NAME 2>/dev/null || true
          ssh dokku domains:add $APP_NAME $DOMAIN

      - name: Deploy
        run: |
          APP_NAME="${{{{ steps.vars.outputs.app_name }}}}"
          git remote add dokku dokku@${{{{ secrets.DOKKU_HOST }}}}:$APP_NAME 2>/dev/null || true
          GIT_SSH_COMMAND="ssh -i ~/.ssh/key" git push dokku HEAD:main -f

      - name: SSL
        run: |
          APP_NAME="${{{{ steps.vars.outputs.app_name }}}}"
          ssh dokku letsencrypt:active $APP_NAME 2>/dev/null || ssh dokku letsencrypt:enable $APP_NAME || true

      - name: Verify
        run: |
          APP_NAME="${{{{ steps.vars.outputs.app_name }}}}"
          DOMAIN="${{APP_NAME}}.${{{{ secrets.BASE_DOMAIN }}}}"
          sleep 10
          curl -sf "https://${{DOMAIN}}/health" && echo "Deployed: https://${{DOMAIN}}" || echo "Check logs: ssh dokku@${{{{ secrets.DOKKU_HOST }}}} logs $APP_NAME"
'''

PREVIEW_WORKFLOW_TEMPLATE = '''# Preview environments for pull requests
name: Preview

on:
  pull_request:
    types: [opened, synchronize, reopened, closed]

concurrency:
  group: preview-${{{{ github.event.pull_request.number }}}}

env:
  APP_NAME: ${{{{ github.event.repository.name }}}}-pr-${{{{ github.event.pull_request.number }}}}

jobs:
  deploy:
    if: github.event.action != 'closed'
    runs-on: ubuntu-latest
    environment: preview
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Setup SSH
        run: |
          mkdir -p ~/.ssh && chmod 700 ~/.ssh
          echo "${{{{ secrets.DOKKU_SSH_PRIVATE_KEY }}}}" > ~/.ssh/key && chmod 600 ~/.ssh/key
          ssh-keyscan -H ${{{{ secrets.DOKKU_HOST }}}} >> ~/.ssh/known_hosts

      - name: Deploy preview
        run: |
          DOMAIN="${{{{ env.APP_NAME }}}}.${{{{ secrets.BASE_DOMAIN }}}}"
          ssh -i ~/.ssh/key dokku@${{{{ secrets.DOKKU_HOST }}}} apps:exists $APP_NAME 2>/dev/null || \\
            ssh -i ~/.ssh/key dokku@${{{{ secrets.DOKKU_HOST }}}} apps:create $APP_NAME
          ssh -i ~/.ssh/key dokku@${{{{ secrets.DOKKU_HOST }}}} config:set --no-restart $APP_NAME APP_ENV=preview APP_URL="https://$DOMAIN"
          ssh -i ~/.ssh/key dokku@${{{{ secrets.DOKKU_HOST }}}} domains:clear $APP_NAME 2>/dev/null || true
          ssh -i ~/.ssh/key dokku@${{{{ secrets.DOKKU_HOST }}}} domains:add $APP_NAME $DOMAIN
          ssh -i ~/.ssh/key dokku@${{{{ secrets.DOKKU_HOST }}}} resource:limit $APP_NAME --memory 256m || true
          git remote add dokku dokku@${{{{ secrets.DOKKU_HOST }}}}:$APP_NAME 2>/dev/null || true
          GIT_SSH_COMMAND="ssh -i ~/.ssh/key" git push dokku HEAD:main -f
          ssh -i ~/.ssh/key dokku@${{{{ secrets.DOKKU_HOST }}}} letsencrypt:enable $APP_NAME || true

      - name: Comment URL
        uses: actions/github-script@v7
        with:
          script: |
            const url = `https://${{{{ env.APP_NAME }}}}.${{{{ secrets.BASE_DOMAIN }}}}`;
            const body = `## 🚀 Preview\\n\\n✅ Deployed: [${{url}}](${{url}})\\n\\n<sub>Auto-destroyed on PR close</sub>`;
            const comments = await github.rest.issues.listComments({{owner: context.repo.owner, repo: context.repo.repo, issue_number: context.issue.number}});
            const bot = comments.data.find(c => c.user.type === 'Bot' && c.body.includes('Preview'));
            if (bot) await github.rest.issues.updateComment({{owner: context.repo.owner, repo: context.repo.repo, comment_id: bot.id, body}});
            else await github.rest.issues.createComment({{owner: context.repo.owner, repo: context.repo.repo, issue_number: context.issue.number, body}});

  cleanup:
    if: github.event.action == 'closed'
    runs-on: ubuntu-latest
    steps:
      - name: Destroy preview
        run: |
          mkdir -p ~/.ssh
          echo "${{{{ secrets.DOKKU_SSH_PRIVATE_KEY }}}}" > ~/.ssh/key && chmod 600 ~/.ssh/key
          ssh-keyscan -H ${{{{ secrets.DOKKU_HOST }}}} >> ~/.ssh/known_hosts
          ssh -i ~/.ssh/key dokku@${{{{ secrets.DOKKU_HOST }}}} apps:destroy ${{{{ env.APP_NAME }}}} --force || true
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
