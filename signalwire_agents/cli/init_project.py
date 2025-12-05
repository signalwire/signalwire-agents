#!/usr/bin/env python3
"""
SignalWire Agent Project Generator

Interactive CLI tool to create new SignalWire agent projects with customizable features.

Usage:
    sw-agent-init                    # Interactive mode
    sw-agent-init myagent            # Quick mode with project name
    sw-agent-init myagent --type full --no-venv
"""

import os
import sys
import secrets
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Any


# Cloud platform options
CLOUD_PLATFORMS = {
    "local": "Local Agent (FastAPI/uvicorn server)",
    "aws": "AWS Lambda Function",
    "gcp": "Google Cloud Function",
    "azure": "Azure Function",
}

# Default regions per platform
DEFAULT_REGIONS = {
    "aws": "us-east-1",
    "gcp": "us-central1",
    "azure": "eastus",
}


# ANSI colors
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    NC = '\033[0m'  # No Color


def print_step(msg: str):
    print(f"{Colors.BLUE}==>{Colors.NC} {msg}")


def print_success(msg: str):
    print(f"{Colors.GREEN}✓{Colors.NC} {msg}")


def print_warning(msg: str):
    print(f"{Colors.YELLOW}!{Colors.NC} {msg}")


def print_error(msg: str):
    print(f"{Colors.RED}✗{Colors.NC} {msg}")


def prompt(question: str, default: str = "") -> str:
    """Prompt user for input with optional default."""
    if default:
        result = input(f"{question} [{default}]: ").strip()
        return result if result else default
    else:
        return input(f"{question}: ").strip()


def prompt_yes_no(question: str, default: bool = True) -> bool:
    """Prompt user for yes/no answer."""
    hint = "Y/n" if default else "y/N"
    result = input(f"{question} [{hint}]: ").strip().lower()
    if not result:
        return default
    return result in ('y', 'yes')


def prompt_select(question: str, options: List[str], default: int = 1) -> int:
    """Prompt user to select from numbered options. Returns 1-based index."""
    print(f"\n{question}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}) {opt}")
    while True:
        result = input(f"Select [{default}]: ").strip()
        if not result:
            return default
        try:
            idx = int(result)
            if 1 <= idx <= len(options):
                return idx
        except ValueError:
            pass
        print(f"Please enter a number between 1 and {len(options)}")


def prompt_multiselect(question: str, options: List[str], defaults: List[bool]) -> List[bool]:
    """Prompt user to toggle multiple options. Returns list of booleans."""
    selected = defaults.copy()

    while True:
        print(f"\n{question}")
        for i, (opt, sel) in enumerate(zip(options, selected), 1):
            marker = "x" if sel else " "
            print(f"  {i}) [{marker}] {opt}")
        print(f"  Enter number to toggle, or press Enter to continue")

        result = input("Toggle: ").strip()
        if not result:
            return selected
        try:
            idx = int(result)
            if 1 <= idx <= len(options):
                selected[idx - 1] = not selected[idx - 1]
        except ValueError:
            pass


def mask_token(token: str) -> str:
    """Mask a token showing only first 4 and last 3 characters."""
    if len(token) <= 10:
        return "*" * len(token)
    return f"{token[:4]}...{token[-3:]}"


def get_env_credentials() -> Dict[str, str]:
    """Get SignalWire credentials from environment variables."""
    return {
        'space': os.environ.get('SIGNALWIRE_SPACE_NAME', ''),
        'project': os.environ.get('SIGNALWIRE_PROJECT_ID', ''),
        'token': os.environ.get('SIGNALWIRE_TOKEN', ''),
    }


def generate_password(length: int = 32) -> str:
    """Generate a secure random password."""
    return secrets.token_urlsafe(length)[:length]


# =============================================================================
# Templates
# =============================================================================

TEMPLATE_AGENTS_INIT = '''from .main_agent import MainAgent

__all__ = ["MainAgent"]
'''

TEMPLATE_SKILLS_INIT = '''"""Skills module - Add reusable agent skills here."""
'''

TEMPLATE_TESTS_INIT = '''"""Test package."""
'''

TEMPLATE_GITIGNORE = '''# Environment
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
'''

TEMPLATE_ENV_EXAMPLE = '''# SignalWire Credentials
SIGNALWIRE_SPACE_NAME=your-space
SIGNALWIRE_PROJECT_ID=your-project-id
SIGNALWIRE_TOKEN=your-api-token

# Agent Server Configuration
HOST=0.0.0.0
PORT=5000

# Agent name (used for SWML handler - keeps the same handler across restarts)
AGENT_NAME=myagent

# Basic Auth for SWML webhooks (optional)
SWML_BASIC_AUTH_USER=signalwire
SWML_BASIC_AUTH_PASSWORD=your-secure-password

# Public URL (ngrok tunnel or production domain)
SWML_PROXY_URL_BASE=https://your-domain.ngrok.io

# Debug settings (0=off, 1=basic, 2=verbose)
DEBUG_WEBHOOK_LEVEL=1
'''

TEMPLATE_REQUIREMENTS = '''signalwire-agents>=1.0.10
python-dotenv>=1.0.0
requests>=2.28.0
pytest>=7.0.0
'''

# =============================================================================
# Cloud Function Templates
# =============================================================================

# AWS Lambda Templates
AWS_REQUIREMENTS_TEMPLATE = '''signalwire-agents>=1.0.10
h11>=0.13,<0.15
fastapi
mangum
uvicorn
'''

AWS_HANDLER_TEMPLATE = '''#!/usr/bin/env python3
"""AWS Lambda handler for {agent_name} agent.

This demonstrates deploying a SignalWire AI Agent to AWS Lambda
with SWAIG functions and SWML output.

Environment variables:
    SWML_BASIC_AUTH_USER: Basic auth username (optional)
    SWML_BASIC_AUTH_PASSWORD: Basic auth password (optional)
"""

import os
from signalwire_agents import AgentBase, SwaigFunctionResult


class {agent_class}(AgentBase):
    """{agent_name} agent for AWS Lambda deployment."""

    def __init__(self):
        super().__init__(name="{agent_name_slug}")

        self._configure_prompts()
        self.add_language("English", "en-US", "rime.spore")
        self._setup_functions()

    def _configure_prompts(self):
        self.prompt_add_section(
            "Role",
            "You are a helpful AI assistant deployed on AWS Lambda."
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

        @self.tool(description="Get AWS Lambda deployment information")
        def get_platform_info(args, raw_data):
            region = os.getenv("AWS_REGION", "unknown")
            function_name = os.getenv("AWS_LAMBDA_FUNCTION_NAME", "unknown")
            memory = os.getenv("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", "unknown")
            runtime = os.getenv("AWS_EXECUTION_ENV", "unknown")

            return SwaigFunctionResult(
                f"Running on AWS Lambda. "
                f"Function: {{function_name}}, Region: {{region}}, "
                f"Memory: {{memory}}MB, Runtime: {{runtime}}."
            )


# Create agent instance outside handler for warm starts
agent = {agent_class}()


def lambda_handler(event, context):
    """AWS Lambda entry point.

    Args:
        event: Lambda event (API Gateway request)
        context: Lambda context with runtime info

    Returns:
        API Gateway response dict
    """
    return agent.run(event, context)
'''

AWS_DEPLOY_TEMPLATE = '''#!/bin/bash
# AWS Lambda deployment script for {agent_name} agent
#
# Prerequisites:
#   - AWS CLI configured with appropriate credentials
#   - Docker installed and running (for building Lambda-compatible packages)
#
# Usage:
#   ./deploy.sh                    # Deploy with defaults
#   ./deploy.sh my-function        # Deploy with custom function name
#   ./deploy.sh my-function us-west-2  # Custom function and region

set -e

# Configuration
FUNCTION_NAME="${{1:-{function_name}}}"
REGION="${{2:-{region}}}"
RUNTIME="python3.11"
HANDLER="handler.lambda_handler"
MEMORY_SIZE=512
TIMEOUT=30
ROLE_NAME="${{FUNCTION_NAME}}-role"

# Default credentials (change these or set via environment)
AUTH_USER="${{SWML_BASIC_AUTH_USER:-{auth_user}}}"
AUTH_PASS="${{SWML_BASIC_AUTH_PASSWORD:-{auth_password}}}"

echo "=== {agent_name} - AWS Lambda Deployment ==="
echo "Function: $FUNCTION_NAME"
echo "Region: $REGION"
echo ""

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is required but not installed."
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "ERROR: Docker is not running. Please start Docker."
    exit 1
fi

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ROLE_ARN="arn:aws:iam::${{ACCOUNT_ID}}:role/${{ROLE_NAME}}"

# Step 1: Create IAM role if it doesn't exist
echo "Step 1: Setting up IAM role..."

TRUST_POLICY='{{
  "Version": "2012-10-17",
  "Statement": [
    {{
      "Effect": "Allow",
      "Principal": {{
        "Service": "lambda.amazonaws.com"
      }},
      "Action": "sts:AssumeRole"
    }}
  ]
}}'

if ! aws iam get-role --role-name "$ROLE_NAME" --region "$REGION" 2>/dev/null; then
    echo "Creating IAM role: $ROLE_NAME"
    aws iam create-role \\
        --role-name "$ROLE_NAME" \\
        --assume-role-policy-document "$TRUST_POLICY" \\
        --region "$REGION"

    # Attach basic Lambda execution policy
    aws iam attach-role-policy \\
        --role-name "$ROLE_NAME" \\
        --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole" \\
        --region "$REGION"

    echo "Waiting for role to propagate..."
    sleep 10
else
    echo "IAM role already exists: $ROLE_NAME"
fi

# Step 2: Package the function using Docker
echo ""
echo "Step 2: Packaging function with Docker (linux/amd64)..."

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
BUILD_DIR=$(mktemp -d)
PACKAGE_DIR="$BUILD_DIR/package"
ZIP_FILE="$BUILD_DIR/function.zip"

mkdir -p "$PACKAGE_DIR"

# Build dependencies using Lambda Python image for correct architecture
echo "Installing dependencies via Docker..."
docker run --rm \\
    --platform linux/amd64 \\
    --entrypoint "" \\
    -v "$SCRIPT_DIR:/var/task:ro" \\
    -v "$PACKAGE_DIR:/var/output" \\
    -w /var/task \\
    public.ecr.aws/lambda/python:3.11 \\
    bash -c "pip install -r requirements.txt -t /var/output --quiet && cp handler.py /var/output/"

# Create zip
echo "Creating deployment package..."
cd "$PACKAGE_DIR"
zip -r "$ZIP_FILE" . -q
cd - > /dev/null

PACKAGE_SIZE=$(du -h "$ZIP_FILE" | cut -f1)
echo "Package size: $PACKAGE_SIZE"

# Step 3: Create or update Lambda function
echo ""
echo "Step 3: Deploying Lambda function..."

if aws lambda get-function --function-name "$FUNCTION_NAME" --region "$REGION" 2>/dev/null; then
    echo "Updating existing function..."
    aws lambda update-function-code \\
        --function-name "$FUNCTION_NAME" \\
        --zip-file "fileb://$ZIP_FILE" \\
        --region "$REGION" \\
        --cli-read-timeout 300 \\
        --output text --query 'FunctionArn'
else
    echo "Creating new function..."
    aws lambda create-function \\
        --function-name "$FUNCTION_NAME" \\
        --runtime "$RUNTIME" \\
        --role "$ROLE_ARN" \\
        --handler "$HANDLER" \\
        --zip-file "fileb://$ZIP_FILE" \\
        --memory-size "$MEMORY_SIZE" \\
        --timeout "$TIMEOUT" \\
        --region "$REGION" \\
        --cli-read-timeout 300 \\
        --environment "Variables={{SWML_BASIC_AUTH_USER=$AUTH_USER,SWML_BASIC_AUTH_PASSWORD=$AUTH_PASS}}" \\
        --output text --query 'FunctionArn'
fi

# Wait for function to be active
echo "Waiting for function to be active..."
aws lambda wait function-active --function-name "$FUNCTION_NAME" --region "$REGION"

# Step 4: Create or get API Gateway
echo ""
echo "Step 4: Setting up API Gateway..."

API_NAME="${{FUNCTION_NAME}}-api"

# Check if API exists
API_ID=$(aws apigatewayv2 get-apis --region "$REGION" \\
    --query "Items[?Name=='$API_NAME'].ApiId" --output text)

if [ -z "$API_ID" ] || [ "$API_ID" == "None" ]; then
    echo "Creating HTTP API..."
    API_ID=$(aws apigatewayv2 create-api \\
        --name "$API_NAME" \\
        --protocol-type HTTP \\
        --region "$REGION" \\
        --output text --query 'ApiId')
fi

echo "API ID: $API_ID"

# Step 5: Create Lambda integration
echo ""
echo "Step 5: Creating Lambda integration..."

LAMBDA_ARN="arn:aws:lambda:${{REGION}}:${{ACCOUNT_ID}}:function:${{FUNCTION_NAME}}"

# Check for existing integration
INTEGRATION_ID=$(aws apigatewayv2 get-integrations \\
    --api-id "$API_ID" \\
    --region "$REGION" \\
    --query "Items[?IntegrationUri=='${{LAMBDA_ARN}}'].IntegrationId" \\
    --output text 2>/dev/null || echo "")

if [ -z "$INTEGRATION_ID" ] || [ "$INTEGRATION_ID" == "None" ]; then
    echo "Creating integration..."
    INTEGRATION_ID=$(aws apigatewayv2 create-integration \\
        --api-id "$API_ID" \\
        --integration-type AWS_PROXY \\
        --integration-uri "$LAMBDA_ARN" \\
        --payload-format-version "2.0" \\
        --region "$REGION" \\
        --output text --query 'IntegrationId')
fi

echo "Integration ID: $INTEGRATION_ID"

# Step 6: Create routes
echo ""
echo "Step 6: Creating routes..."

create_route() {{
    local route_key="$1"
    local existing=$(aws apigatewayv2 get-routes \\
        --api-id "$API_ID" \\
        --region "$REGION" \\
        --query "Items[?RouteKey=='$route_key'].RouteId" \\
        --output text 2>/dev/null || echo "")

    if [ -z "$existing" ] || [ "$existing" == "None" ]; then
        echo "Creating route: $route_key"
        aws apigatewayv2 create-route \\
            --api-id "$API_ID" \\
            --route-key "$route_key" \\
            --target "integrations/$INTEGRATION_ID" \\
            --region "$REGION" \\
            --output text --query 'RouteId'
    else
        echo "Route exists: $route_key"
    fi
}}

# Create routes for SWML and SWAIG
create_route "GET /"
create_route "POST /"
create_route "POST /swaig"
create_route "ANY /{{proxy+}}"

# Step 7: Create/update stage
echo ""
echo "Step 7: Deploying stage..."

STAGE_NAME="\\$default"

if ! aws apigatewayv2 get-stage --api-id "$API_ID" --stage-name "$STAGE_NAME" --region "$REGION" 2>/dev/null; then
    aws apigatewayv2 create-stage \\
        --api-id "$API_ID" \\
        --stage-name "$STAGE_NAME" \\
        --auto-deploy \\
        --region "$REGION" > /dev/null
fi

# Step 8: Add Lambda permission for API Gateway
echo ""
echo "Step 8: Configuring permissions..."

STATEMENT_ID="${{API_NAME}}-invoke"

# Remove existing permission if it exists (ignore errors)
aws lambda remove-permission \\
    --function-name "$FUNCTION_NAME" \\
    --statement-id "$STATEMENT_ID" \\
    --region "$REGION" 2>/dev/null || true

# Add permission
aws lambda add-permission \\
    --function-name "$FUNCTION_NAME" \\
    --statement-id "$STATEMENT_ID" \\
    --action lambda:InvokeFunction \\
    --principal apigateway.amazonaws.com \\
    --source-arn "arn:aws:execute-api:${{REGION}}:${{ACCOUNT_ID}}:${{API_ID}}/*" \\
    --region "$REGION" > /dev/null

# Get the endpoint URL
ENDPOINT="https://${{API_ID}}.execute-api.${{REGION}}.amazonaws.com"

# Cleanup
rm -rf "$BUILD_DIR"

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Endpoint URL: $ENDPOINT"
echo ""
echo "Authentication:"
echo "  Username: $AUTH_USER"
echo "  Password: $AUTH_PASS"
echo ""
echo "Test SWML output:"
echo "  curl -u $AUTH_USER:$AUTH_PASS $ENDPOINT/"
echo ""
echo "Test SWAIG function:"
echo "  curl -u $AUTH_USER:$AUTH_PASS -X POST $ENDPOINT/swaig \\\\"
echo "    -H 'Content-Type: application/json' \\\\"
echo "    -d '{{\\\"function\\\": \\\"get_info\\\", \\\"argument\\\": {{\\\"parsed\\\": [{{\\\"topic\\\": \\\"test\\\"}}]}}}}'"
echo ""
echo "Configure SignalWire:"
echo "  Set your phone number's SWML URL to: https://$AUTH_USER:$AUTH_PASS@${{API_ID}}.execute-api.${{REGION}}.amazonaws.com/"
echo ""
'''

# GCP Cloud Function Templates
GCP_REQUIREMENTS_TEMPLATE = '''signalwire-agents>=1.0.10
functions-framework>=3.0.0
'''

GCP_MAIN_TEMPLATE = '''#!/usr/bin/env python3
"""Google Cloud Functions handler for {agent_name} agent.

This demonstrates deploying a SignalWire AI Agent to Google Cloud Functions
with SWAIG functions and SWML output.

Environment variables:
    SWML_BASIC_AUTH_USER: Basic auth username (optional)
    SWML_BASIC_AUTH_PASSWORD: Basic auth password (optional)
"""

import os
from signalwire_agents import AgentBase, SwaigFunctionResult


class {agent_class}(AgentBase):
    """{agent_name} agent for Google Cloud Functions deployment."""

    def __init__(self):
        super().__init__(name="{agent_name_slug}")

        self._configure_prompts()
        self.add_language("English", "en-US", "rime.spore")
        self._setup_functions()

    def _configure_prompts(self):
        self.prompt_add_section(
            "Role",
            "You are a helpful AI assistant deployed on Google Cloud Functions."
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

        @self.tool(description="Get Google Cloud deployment information")
        def get_platform_info(args, raw_data):
            import urllib.request

            # Gen 2 Cloud Functions run on Cloud Run with these env vars
            service = os.getenv("K_SERVICE", "unknown")
            revision = os.getenv("K_REVISION", "unknown")

            # Query metadata server for project ID
            project = os.getenv("GOOGLE_CLOUD_PROJECT", "unknown")
            if project == "unknown":
                try:
                    req = urllib.request.Request(
                        "http://metadata.google.internal/computeMetadata/v1/project/project-id",
                        headers={{"Metadata-Flavor": "Google"}}
                    )
                    with urllib.request.urlopen(req, timeout=2) as resp:
                        project = resp.read().decode()
                except Exception:
                    pass

            return SwaigFunctionResult(
                f"Running on Google Cloud Functions Gen 2. "
                f"Service: {{service}}, Revision: {{revision}}, "
                f"Project: {{project}}."
            )


# Create agent instance outside handler for warm starts
agent = {agent_class}()


def main(request):
    """Google Cloud Functions entry point.

    Args:
        request: Flask request object

    Returns:
        Flask response
    """
    return agent.run(request)
'''

GCP_DEPLOY_TEMPLATE = '''#!/bin/bash
# Google Cloud Functions deployment script for {agent_name} agent
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - A Google Cloud project with Cloud Functions API enabled
#
# Usage:
#   ./deploy.sh                           # Deploy with defaults
#   ./deploy.sh my-function               # Custom function name
#   ./deploy.sh my-function us-central1   # Custom function and region

set -e

# Configuration
FUNCTION_NAME="${{1:-{function_name}}}"
REGION="${{2:-{region}}}"
RUNTIME="python311"
ENTRY_POINT="main"
MEMORY="512MB"
TIMEOUT="60s"
MIN_INSTANCES=0
MAX_INSTANCES=10

# Directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

echo "=== {agent_name} - Google Cloud Functions Deployment ==="
echo "Function: $FUNCTION_NAME"
echo "Region: $REGION"
echo ""

# Get current project
PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT" ]; then
    echo "Error: No project set. Run: gcloud config set project <project-id>"
    exit 1
fi
echo "Project: $PROJECT"
echo ""

# Step 1: Enable required APIs
echo "Step 1: Enabling required APIs..."
gcloud services enable cloudfunctions.googleapis.com --quiet
gcloud services enable cloudbuild.googleapis.com --quiet
gcloud services enable artifactregistry.googleapis.com --quiet

# Step 2: Create deployment package
echo ""
echo "Step 2: Creating deployment package..."

# Create a temporary deployment directory
DEPLOY_DIR=$(mktemp -d)
trap "rm -rf $DEPLOY_DIR" EXIT

# Copy the main files
cp "$SCRIPT_DIR/main.py" "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/requirements.txt" "$DEPLOY_DIR/"

echo "Deployment package contents:"
ls -la "$DEPLOY_DIR/"

# Step 3: Deploy function
echo ""
echo "Step 3: Deploying Cloud Function..."

# Check if function exists (Gen 2 vs Gen 1)
EXISTING_GEN2=$(gcloud functions describe "$FUNCTION_NAME" --region="$REGION" --gen2 2>/dev/null && echo "yes" || echo "no")

if [ "$EXISTING_GEN2" == "yes" ]; then
    echo "Updating existing Gen 2 function..."
else
    echo "Creating new Gen 2 function..."
fi

gcloud functions deploy "$FUNCTION_NAME" \\
    --gen2 \\
    --region="$REGION" \\
    --runtime="$RUNTIME" \\
    --source="$DEPLOY_DIR" \\
    --entry-point="$ENTRY_POINT" \\
    --trigger-http \\
    --allow-unauthenticated \\
    --memory="$MEMORY" \\
    --timeout="$TIMEOUT" \\
    --min-instances="$MIN_INSTANCES" \\
    --max-instances="$MAX_INSTANCES" \\
    --quiet

# Step 4: Get the endpoint URL
echo ""
echo "Step 4: Getting endpoint URL..."

ENDPOINT=$(gcloud functions describe "$FUNCTION_NAME" \\
    --region="$REGION" \\
    --gen2 \\
    --format="value(serviceConfig.uri)")

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Endpoint URL: $ENDPOINT"
echo ""
echo "Test SWML output:"
echo "  curl $ENDPOINT"
echo ""
echo "Test SWAIG function:"
echo "  curl -X POST $ENDPOINT/swaig \\\\"
echo "    -H 'Content-Type: application/json' \\\\"
echo "    -d '{{\\\"function\\\": \\\"get_info\\\", \\\"argument\\\": {{\\\"parsed\\\": [{{\\\"topic\\\": \\\"test\\\"}}]}}}}'"
echo ""
echo "Configure SignalWire:"
echo "  Set your phone number's SWML URL to: $ENDPOINT"
echo ""
echo "To set environment variables (optional):"
echo "  gcloud functions deploy $FUNCTION_NAME \\\\"
echo "    --region=$REGION \\\\"
echo "    --gen2 \\\\"
echo "    --update-env-vars SWML_BASIC_AUTH_USER=myuser,SWML_BASIC_AUTH_PASSWORD=mypass"
echo ""
'''

# Azure Function Templates
AZURE_REQUIREMENTS_TEMPLATE = '''azure-functions>=1.17.0
signalwire-agents>=1.0.10
'''

AZURE_INIT_TEMPLATE = '''#!/usr/bin/env python3
"""Azure Functions handler for {agent_name} agent.

This demonstrates deploying a SignalWire AI Agent to Azure Functions
with SWAIG functions and SWML output.

Environment variables:
    SWML_BASIC_AUTH_USER: Basic auth username (optional)
    SWML_BASIC_AUTH_PASSWORD: Basic auth password (optional)
"""

import os
import azure.functions as func
from signalwire_agents import AgentBase, SwaigFunctionResult


class {agent_class}(AgentBase):
    """{agent_name} agent for Azure Functions deployment."""

    def __init__(self):
        super().__init__(name="{agent_name_slug}")

        self._configure_prompts()
        self.add_language("English", "en-US", "rime.spore")
        self._setup_functions()

    def _configure_prompts(self):
        self.prompt_add_section(
            "Role",
            "You are a helpful AI assistant deployed on Azure Functions."
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

        @self.tool(description="Get Azure Functions deployment information")
        def get_platform_info(args, raw_data):
            function_name = os.getenv("WEBSITE_SITE_NAME", "unknown")
            region = os.getenv("REGION_NAME", "unknown")
            runtime = os.getenv("FUNCTIONS_WORKER_RUNTIME", "unknown")
            version = os.getenv("FUNCTIONS_EXTENSION_VERSION", "unknown")

            return SwaigFunctionResult(
                f"Running on Azure Functions. "
                f"App: {{function_name}}, Region: {{region}}, "
                f"Runtime: {{runtime}}, Version: {{version}}."
            )


# Create agent instance outside handler for warm starts
agent = {agent_class}()


def main(req: func.HttpRequest) -> func.HttpResponse:
    """Azure Functions entry point.

    Args:
        req: Azure Functions HTTP request object

    Returns:
        Azure Functions HTTP response
    """
    return agent.run(req)
'''

AZURE_FUNCTION_JSON_TEMPLATE = '''{{
    "scriptFile": "__init__.py",
    "bindings": [
        {{
            "authLevel": "anonymous",
            "type": "httpTrigger",
            "direction": "in",
            "name": "req",
            "methods": ["get", "post"],
            "route": "{{*path}}"
        }},
        {{
            "type": "http",
            "direction": "out",
            "name": "$return"
        }}
    ]
}}
'''

AZURE_HOST_JSON_TEMPLATE = '''{{
    "version": "2.0",
    "logging": {{
        "applicationInsights": {{
            "samplingSettings": {{
                "isEnabled": true,
                "excludedTypes": "Request"
            }}
        }}
    }},
    "extensionBundle": {{
        "id": "Microsoft.Azure.Functions.ExtensionBundle",
        "version": "[4.*, 5.0.0)"
    }},
    "extensions": {{
        "http": {{
            "routePrefix": ""
        }}
    }}
}}
'''

AZURE_LOCAL_SETTINGS_TEMPLATE = '''{{
    "IsEncrypted": false,
    "Values": {{
        "FUNCTIONS_WORKER_RUNTIME": "python",
        "AzureWebJobsStorage": ""
    }}
}}
'''

AZURE_DEPLOY_TEMPLATE = '''#!/bin/bash
# Azure Functions deployment script for {agent_name} agent
#
# Prerequisites:
#   - Azure CLI installed and authenticated (az login)
#   - Docker installed and running (for building correct architecture)
#
# Usage:
#   ./deploy.sh                              # Deploy with defaults
#   ./deploy.sh my-app                       # Custom app name
#   ./deploy.sh my-app eastus my-rg          # Custom app, region, and resource group

set -e

# Configuration
APP_NAME="${{1:-{function_name}}}"
LOCATION="${{2:-{region}}}"
RESOURCE_GROUP="${{3:-{resource_group}}}"
STORAGE_ACCOUNT="${{APP_NAME//-/}}storage"  # Remove hyphens for storage account
RUNTIME="python"
RUNTIME_VERSION="3.11"
FUNCTIONS_VERSION="4"

# Truncate storage account name to 24 chars (Azure limit)
STORAGE_ACCOUNT="${{STORAGE_ACCOUNT:0:24}}"

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

echo "=== {agent_name} - Azure Functions Deployment ==="
echo "App Name: $APP_NAME"
echo "Location: $LOCATION"
echo "Resource Group: $RESOURCE_GROUP"
echo "Storage Account: $STORAGE_ACCOUNT"
echo ""

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is required but not installed."
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "ERROR: Docker is not running. Please start Docker."
    exit 1
fi

# Step 1: Login check
echo "Step 1: Checking Azure login..."
if ! az account show &>/dev/null; then
    echo "Not logged in. Running: az login"
    az login
fi

SUBSCRIPTION=$(az account show --query name -o tsv)
echo "Subscription: $SUBSCRIPTION"
echo ""

# Step 2: Create resource group
echo "Step 2: Creating resource group..."
if ! az group show --name "$RESOURCE_GROUP" &>/dev/null; then
    az group create \\
        --name "$RESOURCE_GROUP" \\
        --location "$LOCATION" \\
        --output none
    echo "Created resource group: $RESOURCE_GROUP"
else
    echo "Resource group exists: $RESOURCE_GROUP"
fi

# Step 3: Create storage account
echo ""
echo "Step 3: Creating storage account..."
if ! az storage account show --name "$STORAGE_ACCOUNT" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
    az storage account create \\
        --name "$STORAGE_ACCOUNT" \\
        --resource-group "$RESOURCE_GROUP" \\
        --location "$LOCATION" \\
        --sku Standard_LRS \\
        --output none
    echo "Created storage account: $STORAGE_ACCOUNT"
    echo "Waiting for storage account to propagate..."
    sleep 10
else
    echo "Storage account exists: $STORAGE_ACCOUNT"
fi

# Step 4: Create Function App
echo ""
echo "Step 4: Creating Function App..."
if ! az functionapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" &>/dev/null; then
    az functionapp create \\
        --name "$APP_NAME" \\
        --resource-group "$RESOURCE_GROUP" \\
        --storage-account "$STORAGE_ACCOUNT" \\
        --consumption-plan-location "$LOCATION" \\
        --runtime "$RUNTIME" \\
        --runtime-version "$RUNTIME_VERSION" \\
        --functions-version "$FUNCTIONS_VERSION" \\
        --os-type Linux \\
        --output none
    echo "Created Function App: $APP_NAME"

    # Wait for app to be ready
    echo "Waiting for Function App to be ready..."
    sleep 30
else
    echo "Function App exists: $APP_NAME"
fi

# Step 5: Build and deploy the function using Docker
echo ""
echo "Step 5: Building function with Docker (linux/amd64)..."

DEPLOY_DIR=$(mktemp -d)

# Copy function files
cp -r "$SCRIPT_DIR/function_app" "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/host.json" "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/requirements.txt" "$DEPLOY_DIR/"
cp "$SCRIPT_DIR/local.settings.json" "$DEPLOY_DIR/" 2>/dev/null || true

# Build dependencies using Docker for correct architecture
echo "Installing dependencies via Docker..."
docker run --rm \\
    --platform linux/amd64 \\
    --entrypoint "" \\
    -v "$DEPLOY_DIR:/var/task" \\
    -w /var/task \\
    mcr.microsoft.com/azure-functions/python:4-python3.11 \\
    bash -c "pip install -r requirements.txt -t .python_packages/lib/site-packages --quiet"

# Create zip for deployment
echo "Creating deployment package..."
ZIP_FILE="$DEPLOY_DIR/deploy.zip"
cd "$DEPLOY_DIR"
zip -r "$ZIP_FILE" . -x "*.pyc" -q
cd - > /dev/null

PACKAGE_SIZE=$(du -h "$ZIP_FILE" | cut -f1)
echo "Package size: $PACKAGE_SIZE"

# Deploy using zip deployment
echo ""
echo "Step 6: Deploying to Azure..."
az functionapp deployment source config-zip \\
    --name "$APP_NAME" \\
    --resource-group "$RESOURCE_GROUP" \\
    --src "$ZIP_FILE" \\
    --output none

echo "Deployment complete"

# Cleanup
rm -rf "$DEPLOY_DIR"

# Step 7: Get the endpoint URL
echo ""
echo "Step 7: Getting endpoint URL..."

ENDPOINT="https://${{APP_NAME}}.azurewebsites.net"

# Verify deployment
echo "Waiting for deployment to propagate..."
sleep 10

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Endpoint URL: $ENDPOINT/api/function_app"
echo ""
echo "Test SWML output:"
echo "  curl $ENDPOINT/api/function_app"
echo ""
echo "Test SWAIG function:"
echo "  curl -X POST $ENDPOINT/api/function_app/swaig \\\\"
echo "    -H 'Content-Type: application/json' \\\\"
echo "    -d '{{\\\"function\\\": \\\"get_info\\\", \\\"argument\\\": {{\\\"parsed\\\": [{{\\\"topic\\\": \\\"test\\\"}}]}}}}'"
echo ""
echo "Configure SignalWire:"
echo "  Set your phone number's SWML URL to: $ENDPOINT/api/function_app"
echo ""
echo "To set environment variables (optional):"
echo "  az functionapp config appsettings set --name $APP_NAME --resource-group $RESOURCE_GROUP \\\\"
echo "    --settings SWML_BASIC_AUTH_USER=myuser SWML_BASIC_AUTH_PASSWORD=mypass"
echo ""
'''


def get_agent_template(agent_type: str, features: Dict[str, bool]) -> str:
    """Generate the main agent template based on type and features."""

    has_tool = features.get('example_tool', True)
    has_debug = features.get('debug_webhooks', False)
    has_auth = features.get('basic_auth', False)

    imports = ['from signalwire_agents import AgentBase']
    if has_tool:
        imports.append('from signalwire_agents import SwaigFunctionResult')

    imports_str = '\n'.join(imports)

    # Build the __init__ method
    init_parts = []

    if has_auth:
        init_parts.append('''
        # Set basic auth if configured
        user = os.getenv("SWML_BASIC_AUTH_USER")
        password = os.getenv("SWML_BASIC_AUTH_PASSWORD")
        if user and password:
            self.set_params({
                "swml_basic_auth_user": user,
                "swml_basic_auth_password": password,
            })''')

    init_parts.append('''
        self._configure_voice()
        self._configure_prompts()''')

    if has_debug:
        init_parts.append('''
        self._configure_debug_webhooks()''')

    init_body = ''.join(init_parts)

    # Build optional methods
    extra_methods = []

    if has_debug:
        extra_methods.append('''

    def _configure_debug_webhooks(self):
        """Set up debug and post-prompt webhooks."""
        proxy_url = os.getenv("SWML_PROXY_URL_BASE", "")
        debug_level = int(os.getenv("DEBUG_WEBHOOK_LEVEL", "1"))
        auth_user = os.getenv("SWML_BASIC_AUTH_USER", "")
        auth_pass = os.getenv("SWML_BASIC_AUTH_PASSWORD", "")

        if proxy_url and debug_level > 0:
            # Build URL with basic auth credentials
            if auth_user and auth_pass:
                if "://" in proxy_url:
                    scheme, rest = proxy_url.split("://", 1)
                    auth_proxy_url = f"{scheme}://{auth_user}:{auth_pass}@{rest}"
                else:
                    auth_proxy_url = f"{auth_user}:{auth_pass}@{proxy_url}"
            else:
                auth_proxy_url = proxy_url

            self.set_params({
                "debug_webhook_url": f"{auth_proxy_url}/debug",
                "debug_webhook_level": debug_level,
            })
            self.set_post_prompt(
                "Summarize the conversation including: "
                "the caller's main request, any actions taken, "
                "and the outcome of the call."
            )
            self.set_post_prompt_url(f"{auth_proxy_url}/post_prompt")

    def on_summary(self, summary):
        """Handle call summary."""
        print(f"Call summary: {summary}")
''')

    if has_tool:
        extra_methods.append('''

    @AgentBase.tool(
        name="get_info",
        description="Get information about a topic",
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to get information about"
                }
            },
            "required": ["topic"]
        }
    )
    def get_info(self, args, raw_data):
        """Get information about a topic."""
        topic = args.get("topic", "")
        # TODO: Implement your logic here
        return SwaigFunctionResult(f"Information about {topic}: This is a placeholder response.")''')

    extra_methods_str = ''.join(extra_methods)

    # Need os import if auth or debug
    os_import = 'import os\n' if (has_auth or has_debug) else ''

    return f'''#!/usr/bin/env python3
"""Main Agent - SignalWire AI Agent"""

{os_import}{imports_str}


class MainAgent(AgentBase):
    """Main voice AI agent."""

    def __init__(self):
        super().__init__(
            name="main-agent",
            route="/swml"
        )
{init_body}

    def _configure_voice(self):
        """Set up voice and language."""
        self.add_language("English", "en-US", "rime.spore")

        self.set_params({{
            "end_of_speech_timeout": 500,
            "attention_timeout": 15000,
        }})

    def _configure_prompts(self):
        """Set up AI prompts."""
        self.prompt_add_section(
            "Role",
            "You are a helpful AI assistant. "
            "Help callers with their questions and requests."
        )

        self.prompt_add_section(
            "Guidelines",
            body="Follow these guidelines:",
            bullets=[
                "Be professional and courteous",
                "Ask clarifying questions when needed",
                "Keep responses concise and helpful",
                "If you cannot help, offer to transfer to a human"
            ]
        )
{extra_methods_str}
'''


def get_app_template(features: Dict[str, bool]) -> str:
    """Generate the app.py template based on features."""

    has_debug = features.get('debug_webhooks', False)
    has_web_ui = features.get('web_ui', False)

    # Base imports
    imports = [
        'import os',
        'from pathlib import Path',
        'from dotenv import load_dotenv',
        '',
        '# Load environment variables from .env file',
        'load_dotenv()',
        '',
        'from signalwire_agents import AgentServer',
        'from agents import MainAgent',
    ]

    if has_debug:
        imports.insert(1, 'import sys')
        imports.insert(2, 'import json')
        imports.insert(3, 'from datetime import datetime')
        imports.insert(4, 'from starlette.requests import Request')

    imports_str = '\n'.join(imports)

    # Create agent at module level for swaig-test compatibility
    agent_instance = '''

# Create agent instance at module level for swaig-test compatibility
agent = MainAgent()
'''

    # Debug webhook code
    debug_code = ''
    if has_debug:
        debug_code = '''

# ANSI colors for console output
RESET = "\\033[0m"
BOLD = "\\033[1m"
DIM = "\\033[2m"
CYAN = "\\033[36m"
GREEN = "\\033[32m"
YELLOW = "\\033[33m"
MAGENTA = "\\033[35m"


def print_separator(char="-", width=80):
    print(f"{DIM}{char * width}{RESET}")


def print_debug_data(data):
    """Pretty print debug webhook data."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print()
    print_separator("=")
    print(f"{BOLD}{CYAN}> DEBUG WEBHOOK{RESET}")
    print(f"{DIM}{timestamp}{RESET}")
    print_separator()

    if isinstance(data, dict):
        event_type = data.get("event_type", data.get("type", "unknown"))
        print(f"{YELLOW}Event:{RESET} {event_type}")

        call_id = data.get("call_id", data.get("CallSid", ""))
        if call_id:
            print(f"{YELLOW}Call ID:{RESET} {call_id}")

        if "conversation" in data:
            print(f"\\n{BOLD}{YELLOW}Conversation:{RESET}")
            for msg in data.get("conversation", [])[-5:]:
                role = msg.get("role", "?")
                content = msg.get("content", "")[:100]
                color = GREEN if role == "assistant" else MAGENTA
                print(f"  {color}{role}:{RESET} {content}")

        debug_level = int(os.getenv("DEBUG_WEBHOOK_LEVEL", "1"))
        if debug_level >= 2:
            print(f"\\n{BOLD}{YELLOW}Full Data:{RESET}")
            formatted = json.dumps(data, indent=2)
            for line in formatted.split('\\n')[:50]:
                print(f"  {DIM}{line}{RESET}")
    else:
        print(f"{DIM}{data}{RESET}")

    print_separator("=")
    print()
    sys.stdout.flush()


def print_post_prompt_data(data):
    """Pretty print post-prompt summary data."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print()
    print_separator("=")
    print(f"{BOLD}{YELLOW}> POST-PROMPT SUMMARY{RESET}")
    print(f"{DIM}{timestamp}{RESET}")
    print_separator()

    if isinstance(data, dict):
        summary = data.get("post_prompt_data", {})
        if isinstance(summary, dict):
            for key, value in summary.items():
                print(f"{GREEN}{key}:{RESET} {value}")
        elif summary:
            print(f"{GREEN}Summary:{RESET} {summary}")

        raw = data.get("raw_response", data.get("response", ""))
        if raw:
            print(f"\\n{BOLD}{YELLOW}Response:{RESET}")
            print(f"  {MAGENTA}{raw}{RESET}")

        call_id = data.get("call_id", "")
        if call_id:
            print(f"\\n{DIM}Call ID: {call_id}{RESET}")
    else:
        print(f"{DIM}{data}{RESET}")

    print_separator("=")
    print()
    sys.stdout.flush()
'''

    # Main function
    main_body_parts = ['''
def main():
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))

    # Create server and register agent
    server = AgentServer(host=host, port=port)
    server.register(agent)
''']

    if has_web_ui:
        main_body_parts.append('''
    # Serve static files from web/ directory
    web_dir = Path(__file__).parent / "web"
    if web_dir.exists():
        server.serve_static_files(str(web_dir))
''')

    if has_debug:
        main_body_parts.append('''
    # Add debug webhook endpoint
    @server.app.post('/debug')
    async def debug_webhook(request: Request):
        """Receive and display debug webhook data."""
        try:
            data = await request.json()
        except:
            body = await request.body()
            data = body.decode('utf-8', errors='ignore')
        print_debug_data(data)
        return {'status': 'received'}

    # Add post-prompt webhook endpoint
    @server.app.post('/post_prompt')
    async def post_prompt_webhook(request: Request):
        """Receive and display post-prompt summary data."""
        try:
            data = await request.json()
        except:
            body = await request.body()
            data = body.decode('utf-8', errors='ignore')
        print_post_prompt_data(data)
        return {'status': 'received'}
''')

    main_body_parts.append('''
    # Print startup info
    print(f"\\nSignalWire Agent Server")
    print(f"SWML endpoint:  http://{host}:{port}/swml")
    print(f"SWAIG endpoint: http://{host}:{port}/swml/swaig/")
    print()

    server.run()


if __name__ == "__main__":
    main()
''')

    main_body = ''.join(main_body_parts)

    return f'''#!/usr/bin/env python3
"""Main entry point for the agent server."""

{imports_str}
{agent_instance}
{debug_code}
{main_body}'''


def get_test_template(has_tool: bool) -> str:
    """Generate test template."""

    tool_tests = ''
    if has_tool:
        tool_tests = '''

class TestFunctionExecution:
    """Test that SWAIG functions can be executed."""

    def test_get_info_function(self):
        """Test get_info function executes successfully."""
        returncode, stdout, stderr = run_swaig_test(
            "--exec", "get_info", "--topic", "SignalWire"
        )
        assert returncode == 0, f"Function execution failed: {stderr}"
        assert "SignalWire" in stdout, f"Expected 'SignalWire' in output: {stdout}"


class TestDirectImport:
    """Test direct Python import of the agent."""

    def test_agent_creation(self):
        """Test that agent can be instantiated."""
        from agents import MainAgent
        agent = MainAgent()
        assert agent is not None
        assert agent.name == "main-agent"

    def test_get_info_tool_direct(self):
        """Test the get_info tool via direct call."""
        from agents import MainAgent
        agent = MainAgent()
        result = agent.get_info({"topic": "test"}, {})
        assert "test" in result.response
'''

    tool_check = '''
    def test_agent_has_tools(self):
        """Test agent has expected tools defined."""
        tools = list_tools()
        assert "get_info" in tools, f"Missing get_info tool. Found: {tools}"
''' if has_tool else ''

    return f'''#!/usr/bin/env python3
"""
Tests for the main agent using swaig-test.
"""

import subprocess
import json
import sys
from pathlib import Path

import pytest


AGENT_FILE = Path(__file__).parent.parent / "agents" / "main_agent.py"


def run_swaig_test(*args) -> tuple:
    """Run swaig-test on the agent and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, "-m", "signalwire_agents.cli.swaig_test_wrapper", str(AGENT_FILE)] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result.returncode, result.stdout, result.stderr


def get_swml_json() -> dict:
    """Get SWML JSON output from the agent."""
    returncode, stdout, stderr = run_swaig_test("--dump-swml", "--raw")
    if returncode != 0:
        pytest.fail(f"swaig-test failed:\\nstderr: {{stderr}}\\nstdout: {{stdout}}")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        pytest.fail(f"Invalid JSON: {{e}}\\nOutput: {{stdout}}")


def list_tools() -> list:
    """List tools available in the agent."""
    returncode, stdout, stderr = run_swaig_test("--list-tools")
    if returncode != 0:
        return []
    tools = []
    for line in stdout.split('\\n'):
        line = line.strip()
        if ' - ' in line and not line.startswith('Parameters:'):
            parts = line.split(' - ')
            if parts:
                tool_name = parts[0].strip()
                if tool_name and not tool_name.startswith('('):
                    tools.append(tool_name)
    return tools


class TestAgentLoading:
    """Test that the agent can be loaded without errors."""

    def test_agent_loads(self):
        """Test agent file can be loaded by swaig-test."""
        returncode, stdout, stderr = run_swaig_test("--list-tools")
        assert returncode == 0, f"Failed to load agent: {{stderr}}"
{tool_check}

class TestSWMLGeneration:
    """Test that the agent generates valid SWML documents."""

    def test_swml_structure(self):
        """Test SWML has required structure."""
        swml = get_swml_json()
        assert "version" in swml, "SWML missing 'version'"
        assert "sections" in swml, "SWML missing 'sections'"
        assert "main" in swml["sections"], "SWML missing 'sections.main'"

    def test_swml_has_ai_section(self):
        """Test SWML has AI configuration."""
        swml = get_swml_json()
        main_section = swml.get("sections", {{}}).get("main", [])
        ai_found = any("ai" in verb for verb in main_section)
        assert ai_found, "SWML missing 'ai' verb"
{tool_tests}

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''


def get_readme_template(project_name: str, features: Dict[str, bool]) -> str:
    """Generate README template."""

    endpoints = [
        "| `/swml` | POST | Main SWML endpoint - point your SignalWire phone number here |",
    ]

    if features.get('debug_webhooks'):
        endpoints.append("| `/debug` | POST | Debug webhook - receives real-time call data |")
        endpoints.append("| `/post_prompt` | POST | Post-prompt webhook - receives call summaries |")

    if features.get('web_ui'):
        endpoints.append("| `/` | GET | Static files from `web/` directory |")

    endpoints_table = '\n'.join(endpoints)

    return f'''# {project_name}

A SignalWire AI Agent built with signalwire-agents.

## Quick Start

```bash
cd {project_name}
source .venv/bin/activate  # If using virtual environment
python app.py
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
{endpoints_table}

## Project Structure

```
{project_name}/
├── agents/
│   ├── __init__.py
│   └── main_agent.py    # Main agent implementation
├── skills/
│   └── __init__.py      # Reusable skills
├── tests/
│   └── test_agent.py    # Test suite
├── web/                  # Static files
├── app.py               # Entry point
├── .env                 # Environment configuration
└── requirements.txt     # Python dependencies
```

## Configuration

Edit `.env` to configure:

| Variable | Description |
|----------|-------------|
| `SIGNALWIRE_SPACE_NAME` | Your SignalWire space name |
| `SIGNALWIRE_PROJECT_ID` | Your SignalWire project ID |
| `SIGNALWIRE_TOKEN` | Your SignalWire API token |
| `HOST` | Server host (default: 0.0.0.0) |
| `PORT` | Server port (default: 5000) |

## Testing

```bash
pytest tests/ -v
```

## Adding Tools

Add new tools to your agent using the `@AgentBase.tool` decorator:

```python
@AgentBase.tool(
    name="my_tool",
    description="What this tool does",
    parameters={{
        "type": "object",
        "properties": {{
            "param1": {{"type": "string", "description": "Parameter description"}}
        }},
        "required": ["param1"]
    }}
)
def my_tool(self, args, raw_data):
    param1 = args.get("param1")
    return SwaigFunctionResult(f"Result: {{param1}}")
```
'''


def get_web_index_template() -> str:
    """Generate a simple web UI template."""
    return '''<!DOCTYPE html>
<html>
<head>
    <title>SignalWire Agent</title>
    <style>
        body {
            font-family: system-ui, -apple-system, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #f8f9fa;
            color: #333;
        }
        h1 {
            color: #044cf6;
            border-bottom: 3px solid #044cf6;
            padding-bottom: 10px;
        }
        .status {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .endpoint {
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
        }
        .endpoint h3 {
            margin-top: 0;
            color: #044cf6;
        }
        .method {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 12px;
            margin-right: 10px;
        }
        .method.post { background: #49cc90; color: white; }
        code {
            background: #e9ecef;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 14px;
        }
        pre {
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <h1>SignalWire Agent</h1>

    <div class="status">
        Your agent is running and ready to receive calls!
    </div>

    <h2>Endpoints</h2>

    <div class="endpoint">
        <h3><span class="method post">POST</span> /swml</h3>
        <p>Main SWML endpoint. Point your SignalWire phone number here.</p>
        <pre>curl -X POST http://localhost:5000/swml \\
  -H "Content-Type: application/json" \\
  -d '{}'</pre>
    </div>

    <div class="endpoint">
        <h3><span class="method post">POST</span> /swml/swaig/</h3>
        <p>SWAIG function endpoint for tool calls.</p>
    </div>

    <h2>Quick Start</h2>
    <pre># Test the SWML endpoint
curl -X POST http://localhost:5000/swml -H "Content-Type: application/json" -d '{}'

# Run tests
pytest tests/ -v</pre>

    <p>Powered by <a href="https://signalwire.com">SignalWire</a> and the
    <a href="https://github.com/signalwire/signalwire-agents">SignalWire Agents SDK</a></p>
</body>
</html>
'''


# =============================================================================
# Project Generator
# =============================================================================

class ProjectGenerator:
    """Generates a new SignalWire agent project."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project_dir = Path(config['project_dir'])
        self.project_name = config['project_name']
        self.features = config['features']
        self.credentials = config.get('credentials', {})
        self.platform = config.get('platform', 'local')
        self.cloud_config = config.get('cloud_config', {})

    def generate(self) -> bool:
        """Generate the project. Returns True on success."""
        try:
            if self.platform == 'local':
                return self._generate_local()
            elif self.platform == 'aws':
                return self._generate_aws()
            elif self.platform == 'gcp':
                return self._generate_gcp()
            elif self.platform == 'azure':
                return self._generate_azure()
            else:
                print_error(f"Unknown platform: {self.platform}")
                return False
        except Exception as e:
            print_error(f"Failed to generate project: {e}")
            return False

    def _generate_local(self) -> bool:
        """Generate a local agent project."""
        self._create_directories()
        self._create_agent_files()
        self._create_app_file()
        self._create_config_files()

        if self.features.get('tests'):
            self._create_test_files()

        if self.features.get('web_ui'):
            self._create_web_files()

        self._create_readme()

        if self.config.get('create_venv'):
            self._create_virtualenv()

        return True

    def _generate_aws(self) -> bool:
        """Generate an AWS Lambda project."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        print_success(f"Created {self.project_dir}/")

        # Get template variables
        template_vars = self._get_template_vars()

        # handler.py
        handler_code = AWS_HANDLER_TEMPLATE.format(**template_vars)
        (self.project_dir / 'handler.py').write_text(handler_code)
        print_success("Created handler.py")

        # requirements.txt
        (self.project_dir / 'requirements.txt').write_text(AWS_REQUIREMENTS_TEMPLATE)
        print_success("Created requirements.txt")

        # deploy.sh
        deploy_code = AWS_DEPLOY_TEMPLATE.format(**template_vars)
        deploy_path = self.project_dir / 'deploy.sh'
        deploy_path.write_text(deploy_code)
        deploy_path.chmod(0o755)
        print_success("Created deploy.sh")

        # .env.example
        self._create_cloud_env_example('aws')

        # .gitignore
        (self.project_dir / '.gitignore').write_text(TEMPLATE_GITIGNORE)
        print_success("Created .gitignore")

        # README.md
        self._create_cloud_readme('aws')

        return True

    def _generate_gcp(self) -> bool:
        """Generate a Google Cloud Function project."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        print_success(f"Created {self.project_dir}/")

        # Get template variables
        template_vars = self._get_template_vars()

        # main.py
        main_code = GCP_MAIN_TEMPLATE.format(**template_vars)
        (self.project_dir / 'main.py').write_text(main_code)
        print_success("Created main.py")

        # requirements.txt
        (self.project_dir / 'requirements.txt').write_text(GCP_REQUIREMENTS_TEMPLATE)
        print_success("Created requirements.txt")

        # deploy.sh
        deploy_code = GCP_DEPLOY_TEMPLATE.format(**template_vars)
        deploy_path = self.project_dir / 'deploy.sh'
        deploy_path.write_text(deploy_code)
        deploy_path.chmod(0o755)
        print_success("Created deploy.sh")

        # .env.example
        self._create_cloud_env_example('gcp')

        # .gitignore
        (self.project_dir / '.gitignore').write_text(TEMPLATE_GITIGNORE)
        print_success("Created .gitignore")

        # README.md
        self._create_cloud_readme('gcp')

        return True

    def _generate_azure(self) -> bool:
        """Generate an Azure Functions project."""
        self.project_dir.mkdir(parents=True, exist_ok=True)
        print_success(f"Created {self.project_dir}/")

        # Create function_app directory
        function_dir = self.project_dir / 'function_app'
        function_dir.mkdir(exist_ok=True)

        # Get template variables
        template_vars = self._get_template_vars()

        # function_app/__init__.py
        init_code = AZURE_INIT_TEMPLATE.format(**template_vars)
        (function_dir / '__init__.py').write_text(init_code)
        print_success("Created function_app/__init__.py")

        # function_app/function.json
        (function_dir / 'function.json').write_text(AZURE_FUNCTION_JSON_TEMPLATE)
        print_success("Created function_app/function.json")

        # host.json
        (self.project_dir / 'host.json').write_text(AZURE_HOST_JSON_TEMPLATE)
        print_success("Created host.json")

        # local.settings.json
        (self.project_dir / 'local.settings.json').write_text(AZURE_LOCAL_SETTINGS_TEMPLATE)
        print_success("Created local.settings.json")

        # requirements.txt
        (self.project_dir / 'requirements.txt').write_text(AZURE_REQUIREMENTS_TEMPLATE)
        print_success("Created requirements.txt")

        # deploy.sh
        deploy_code = AZURE_DEPLOY_TEMPLATE.format(**template_vars)
        deploy_path = self.project_dir / 'deploy.sh'
        deploy_path.write_text(deploy_code)
        deploy_path.chmod(0o755)
        print_success("Created deploy.sh")

        # .env.example
        self._create_cloud_env_example('azure')

        # .gitignore
        (self.project_dir / '.gitignore').write_text(TEMPLATE_GITIGNORE)
        print_success("Created .gitignore")

        # README.md
        self._create_cloud_readme('azure')

        return True

    def _get_template_vars(self) -> Dict[str, str]:
        """Get template variables for cloud function templates."""
        # Convert project name to various formats
        agent_name = self.project_name
        agent_name_slug = self.project_name.lower().replace(' ', '-').replace('_', '-')
        agent_class = ''.join(word.capitalize() for word in self.project_name.replace('-', ' ').replace('_', ' ').split()) + 'Agent'
        function_name = agent_name_slug

        # Auth credentials
        auth_user = 'admin'
        auth_password = generate_password(16)

        return {
            'agent_name': agent_name,
            'agent_name_slug': agent_name_slug,
            'agent_class': agent_class,
            'function_name': function_name,
            'region': self.cloud_config.get('region', DEFAULT_REGIONS.get(self.platform, '')),
            'resource_group': self.cloud_config.get('resource_group', f'{function_name}-rg'),
            'auth_user': auth_user,
            'auth_password': auth_password,
        }

    def _create_cloud_env_example(self, platform: str):
        """Create .env.example for cloud platforms."""
        env_content = '''# Optional: Basic authentication credentials
# If not set, the SDK will auto-generate secure credentials
SWML_BASIC_AUTH_USER=admin
SWML_BASIC_AUTH_PASSWORD=your-secure-password
'''
        (self.project_dir / '.env.example').write_text(env_content)
        print_success("Created .env.example")

    def _create_cloud_readme(self, platform: str):
        """Create README.md for cloud platforms."""
        template_vars = self._get_template_vars()
        platform_names = {'aws': 'AWS Lambda', 'gcp': 'Google Cloud Functions', 'azure': 'Azure Functions'}
        platform_name = platform_names.get(platform, platform)

        if platform == 'aws':
            readme = f'''# {self.project_name}

A SignalWire AI Agent deployed on {platform_name}.

## Prerequisites

- AWS CLI configured with appropriate credentials
- Docker installed and running

## Quick Start

```bash
cd {self.project_name}
./deploy.sh
```

## Deployment Options

```bash
# Deploy with defaults
./deploy.sh

# Custom function name
./deploy.sh my-function

# Custom function and region
./deploy.sh my-function us-west-2
```

## Project Structure

```
{self.project_name}/
├── handler.py          # Lambda handler with agent
├── requirements.txt    # Python dependencies
├── deploy.sh           # Deployment script
├── .env.example        # Environment template
└── README.md
```

## Testing

After deployment, test your function:

```bash
# Test SWML output
curl -u admin:password https://YOUR-API-ID.execute-api.REGION.amazonaws.com/

# Test SWAIG function
curl -u admin:password -X POST https://YOUR-API-ID.execute-api.REGION.amazonaws.com/swaig \\
  -H 'Content-Type: application/json' \\
  -d '{{"function": "get_info", "argument": {{"parsed": [{{"topic": "test"}}]}}}}'
```

## Configure SignalWire

Set your phone number's SWML URL to the endpoint URL shown after deployment.
'''

        elif platform == 'gcp':
            readme = f'''# {self.project_name}

A SignalWire AI Agent deployed on {platform_name}.

## Prerequisites

- gcloud CLI installed and authenticated
- A Google Cloud project with Cloud Functions API enabled

## Quick Start

```bash
cd {self.project_name}
./deploy.sh
```

## Deployment Options

```bash
# Deploy with defaults
./deploy.sh

# Custom function name
./deploy.sh my-function

# Custom function and region
./deploy.sh my-function us-central1
```

## Project Structure

```
{self.project_name}/
├── main.py             # Cloud Function entry point
├── requirements.txt    # Python dependencies
├── deploy.sh           # Deployment script
├── .env.example        # Environment template
└── README.md
```

## Testing

After deployment, test your function:

```bash
# Test SWML output
curl https://YOUR-FUNCTION-URL

# Test SWAIG function
curl -X POST https://YOUR-FUNCTION-URL/swaig \\
  -H 'Content-Type: application/json' \\
  -d '{{"function": "get_info", "argument": {{"parsed": [{{"topic": "test"}}]}}}}'
```

## Configure SignalWire

Set your phone number's SWML URL to the endpoint URL shown after deployment.
'''

        else:  # azure
            readme = f'''# {self.project_name}

A SignalWire AI Agent deployed on {platform_name}.

## Prerequisites

- Azure CLI installed and authenticated (`az login`)
- Docker installed and running

## Quick Start

```bash
cd {self.project_name}
./deploy.sh
```

## Deployment Options

```bash
# Deploy with defaults
./deploy.sh

# Custom app name
./deploy.sh my-app

# Custom app, region, and resource group
./deploy.sh my-app eastus my-resource-group
```

## Project Structure

```
{self.project_name}/
├── function_app/
│   ├── __init__.py     # Azure Function handler
│   └── function.json   # HTTP trigger config
├── host.json           # Host configuration
├── local.settings.json # Local dev settings
├── requirements.txt    # Python dependencies
├── deploy.sh           # Deployment script
├── .env.example        # Environment template
└── README.md
```

## Testing

After deployment, test your function:

```bash
# Test SWML output
curl https://YOUR-APP.azurewebsites.net/api/function_app

# Test SWAIG function
curl -X POST https://YOUR-APP.azurewebsites.net/api/function_app/swaig \\
  -H 'Content-Type: application/json' \\
  -d '{{"function": "get_info", "argument": {{"parsed": [{{"topic": "test"}}]}}}}'
```

## Configure SignalWire

Set your phone number's SWML URL to the endpoint URL shown after deployment.
'''

        (self.project_dir / 'README.md').write_text(readme)
        print_success("Created README.md")

    def _create_directories(self):
        """Create project directory structure."""
        dirs = ['agents', 'skills']
        if self.features.get('tests'):
            dirs.append('tests')
        if self.features.get('web_ui'):
            dirs.append('web')

        self.project_dir.mkdir(parents=True, exist_ok=True)
        print_success(f"Created {self.project_dir}/")

        for d in dirs:
            (self.project_dir / d).mkdir(exist_ok=True)

    def _create_agent_files(self):
        """Create agent module files."""
        agents_dir = self.project_dir / 'agents'

        # __init__.py
        (agents_dir / '__init__.py').write_text(TEMPLATE_AGENTS_INIT)
        print_success("Created agents/__init__.py")

        # main_agent.py
        agent_code = get_agent_template(self.config.get('agent_type', 'basic'), self.features)
        (agents_dir / 'main_agent.py').write_text(agent_code)
        print_success("Created agents/main_agent.py")

        # skills/__init__.py
        (self.project_dir / 'skills' / '__init__.py').write_text(TEMPLATE_SKILLS_INIT)
        print_success("Created skills/__init__.py")

    def _create_app_file(self):
        """Create main app.py entry point."""
        app_code = get_app_template(self.features)
        (self.project_dir / 'app.py').write_text(app_code)
        print_success("Created app.py")

    def _create_config_files(self):
        """Create configuration files."""
        # .env
        env_content = f'''# SignalWire Credentials
SIGNALWIRE_SPACE_NAME={self.credentials.get('space', '')}
SIGNALWIRE_PROJECT_ID={self.credentials.get('project', '')}
SIGNALWIRE_TOKEN={self.credentials.get('token', '')}

# Agent Server Configuration
HOST=0.0.0.0
PORT=5000

# Agent name
AGENT_NAME={self.project_name}
'''
        if self.features.get('basic_auth'):
            env_content += f'''
# Basic Auth for SWML webhooks
SWML_BASIC_AUTH_USER=signalwire
SWML_BASIC_AUTH_PASSWORD={generate_password()}
'''

        if self.features.get('debug_webhooks'):
            env_content += '''
# Public URL (ngrok tunnel or production domain)
SWML_PROXY_URL_BASE=

# Debug settings (0=off, 1=basic, 2=verbose)
DEBUG_WEBHOOK_LEVEL=1
'''

        (self.project_dir / '.env').write_text(env_content)
        print_success("Created .env")

        # .env.example
        (self.project_dir / '.env.example').write_text(TEMPLATE_ENV_EXAMPLE)
        print_success("Created .env.example")

        # .gitignore
        (self.project_dir / '.gitignore').write_text(TEMPLATE_GITIGNORE)
        print_success("Created .gitignore")

        # requirements.txt
        (self.project_dir / 'requirements.txt').write_text(TEMPLATE_REQUIREMENTS)
        print_success("Created requirements.txt")

    def _create_test_files(self):
        """Create test files."""
        tests_dir = self.project_dir / 'tests'

        (tests_dir / '__init__.py').write_text(TEMPLATE_TESTS_INIT)
        print_success("Created tests/__init__.py")

        test_code = get_test_template(self.features.get('example_tool', True))
        (tests_dir / 'test_agent.py').write_text(test_code)
        print_success("Created tests/test_agent.py")

    def _create_web_files(self):
        """Create web UI files."""
        web_dir = self.project_dir / 'web'

        (web_dir / 'index.html').write_text(get_web_index_template())
        print_success("Created web/index.html")

    def _create_readme(self):
        """Create README.md."""
        readme = get_readme_template(self.project_name, self.features)
        (self.project_dir / 'README.md').write_text(readme)
        print_success("Created README.md")

    def _create_virtualenv(self):
        """Create and set up virtual environment."""
        venv_dir = self.project_dir / '.venv'

        print_step("Creating virtual environment...")
        try:
            subprocess.run(
                [sys.executable, '-m', 'venv', str(venv_dir)],
                check=True,
                capture_output=True
            )
            print_success("Created virtual environment")

            # Install dependencies
            print_step("Installing dependencies...")
            pip_path = venv_dir / 'bin' / 'pip'
            if sys.platform == 'win32':
                pip_path = venv_dir / 'Scripts' / 'pip.exe'

            subprocess.run(
                [str(pip_path), 'install', '-q', '-r', str(self.project_dir / 'requirements.txt')],
                check=True,
                capture_output=True
            )
            print_success("Installed dependencies")

        except subprocess.CalledProcessError as e:
            print_warning(f"Failed to set up virtual environment: {e}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def run_interactive() -> Dict[str, Any]:
    """Run interactive prompts and return configuration."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}Welcome to SignalWire Agent Init!{Colors.NC}\n")

    # Platform selection
    platform_options = list(CLOUD_PLATFORMS.values())
    platform_idx = prompt_select("Target platform:", platform_options, default=1)
    platform = list(CLOUD_PLATFORMS.keys())[platform_idx - 1]

    # Project name
    default_name = "my-agent"
    project_name = prompt("Project name", default_name)

    # Project directory
    default_dir = f"./{project_name}"
    project_dir = prompt("Project directory", default_dir)
    project_dir = os.path.abspath(os.path.expanduser(project_dir))

    # Check if directory exists
    if os.path.exists(project_dir):
        if not prompt_yes_no(f"Directory {project_dir} exists. Overwrite?", default=False):
            print("Aborted.")
            sys.exit(0)

    # Cloud-specific configuration
    cloud_config = {}
    if platform != 'local':
        default_region = DEFAULT_REGIONS.get(platform, '')
        cloud_config['region'] = prompt("Region", default_region)

        if platform == 'azure':
            cloud_config['resource_group'] = prompt("Resource group", f"{project_name}-rg")

    # Agent type (for local) or simplified for cloud
    if platform == 'local':
        agent_types = [
            "Basic - Minimal agent with example tool",
            "Full - Debug webhooks, web UI, all features",
        ]
        agent_type_idx = prompt_select("Agent type:", agent_types, default=1)
        agent_type = 'basic' if agent_type_idx == 1 else 'full'

        # Set default features based on type
        if agent_type == 'full':
            default_features = [True, True, True, True, True, True]
        else:
            default_features = [False, False, False, True, True, False]

        # Feature selection
        feature_names = [
            "Debug webhooks (console output)",
            "Post-prompt summary",
            "Web UI",
            "Example SWAIG tool",
            "Test scaffolding (pytest)",
            "Basic authentication",
        ]
        selected = prompt_multiselect("Include features:", feature_names, default_features)

        features = {
            'debug_webhooks': selected[0],
            'post_prompt': selected[1],
            'web_ui': selected[2],
            'example_tool': selected[3],
            'tests': selected[4],
            'basic_auth': selected[5],
        }
    else:
        # Cloud platforms have simplified features
        agent_type = 'basic'
        features = {
            'debug_webhooks': False,
            'post_prompt': False,
            'web_ui': False,
            'example_tool': True,
            'tests': False,
            'basic_auth': prompt_yes_no("Enable basic authentication?", default=True),
        }

    # Credentials (only for local)
    credentials = {'space': '', 'project': '', 'token': ''}
    if platform == 'local':
        env_creds = get_env_credentials()

        if env_creds['space'] or env_creds['project'] or env_creds['token']:
            print(f"\n{Colors.GREEN}SignalWire credentials found in environment:{Colors.NC}")
            if env_creds['space']:
                print(f"  Space: {env_creds['space']}")
            if env_creds['project']:
                print(f"  Project: {env_creds['project'][:12]}...{env_creds['project'][-4:]}")
            if env_creds['token']:
                print(f"  Token: {mask_token(env_creds['token'])}")

            if prompt_yes_no("Use these credentials?", default=True):
                credentials = env_creds
            else:
                credentials['space'] = prompt("Space name", env_creds['space'])
                credentials['project'] = prompt("Project ID", env_creds['project'])
                credentials['token'] = prompt("Token", env_creds['token'])
        else:
            print(f"\n{Colors.YELLOW}No SignalWire credentials found in environment.{Colors.NC}")
            if prompt_yes_no("Enter credentials now?", default=False):
                credentials['space'] = prompt("Space name")
                credentials['project'] = prompt("Project ID")
                credentials['token'] = prompt("Token")

    # Virtual environment (only for local)
    create_venv = False
    if platform == 'local':
        create_venv = prompt_yes_no("\nCreate virtual environment?", default=True)

    return {
        'project_name': project_name,
        'project_dir': project_dir,
        'platform': platform,
        'agent_type': agent_type,
        'features': features,
        'credentials': credentials,
        'create_venv': create_venv,
        'cloud_config': cloud_config,
    }


def run_quick(project_name: str, args: Any) -> Dict[str, Any]:
    """Run in quick mode with minimal prompts."""
    project_dir = os.path.abspath(os.path.join('.', project_name))

    # Get platform from args
    platform = getattr(args, 'platform', 'local') or 'local'

    # Get region from args or use default
    region = getattr(args, 'region', None)
    if not region and platform != 'local':
        region = DEFAULT_REGIONS.get(platform, '')

    # Build cloud config
    cloud_config = {}
    if platform != 'local':
        cloud_config['region'] = region
        if platform == 'azure':
            cloud_config['resource_group'] = f"{project_name}-rg"

    # Determine features from args
    agent_type = getattr(args, 'type', 'basic') or 'basic'

    if platform == 'local':
        if agent_type == 'full':
            features = {
                'debug_webhooks': True,
                'post_prompt': True,
                'web_ui': True,
                'example_tool': True,
                'tests': True,
                'basic_auth': True,
            }
        else:
            features = {
                'debug_webhooks': False,
                'post_prompt': False,
                'web_ui': False,
                'example_tool': True,
                'tests': True,
                'basic_auth': False,
            }
    else:
        # Cloud platforms have simplified features
        features = {
            'debug_webhooks': False,
            'post_prompt': False,
            'web_ui': False,
            'example_tool': True,
            'tests': False,
            'basic_auth': True,
        }

    # Get credentials from environment (only for local)
    credentials = get_env_credentials() if platform == 'local' else {'space': '', 'project': '', 'token': ''}

    return {
        'project_name': project_name,
        'project_dir': project_dir,
        'platform': platform,
        'agent_type': agent_type,
        'features': features,
        'credentials': credentials,
        'create_venv': not getattr(args, 'no_venv', False) and platform == 'local',
        'cloud_config': cloud_config,
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Create a new SignalWire agent project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  sw-agent-init                        Interactive mode
  sw-agent-init myagent                Quick mode with defaults
  sw-agent-init myagent --type full    Full-featured local agent
  sw-agent-init myagent -p aws         AWS Lambda function
  sw-agent-init myagent -p gcp         Google Cloud Function
  sw-agent-init myagent -p azure       Azure Function
  sw-agent-init myagent -p aws -r us-west-2  Custom region
'''
    )
    parser.add_argument('name', nargs='?', help='Project name')
    parser.add_argument('--type', choices=['basic', 'full'], default='basic',
                        help='Agent type (default: basic)')
    parser.add_argument('--platform', '-p', choices=['local', 'aws', 'gcp', 'azure'],
                        default='local', help='Target platform (default: local)')
    parser.add_argument('--region', '-r', help='Cloud region (e.g., us-east-1, us-central1, eastus)')
    parser.add_argument('--no-venv', action='store_true',
                        help='Skip virtual environment creation')
    parser.add_argument('--dir', help='Parent directory for project')

    args = parser.parse_args()

    # Run interactive or quick mode
    if args.name:
        config = run_quick(args.name, args)
        if args.dir:
            config['project_dir'] = os.path.abspath(os.path.join(args.dir, args.name))
    else:
        config = run_interactive()

    # Generate project
    print(f"\n{Colors.BOLD}Creating project '{config['project_name']}'...{Colors.NC}\n")

    generator = ProjectGenerator(config)
    if generator.generate():
        print(f"\n{Colors.GREEN}{Colors.BOLD}Project created successfully!{Colors.NC}\n")
        print("To start your agent:\n")
        print(f"  cd {config['project_dir']}")
        if config.get('create_venv'):
            if sys.platform == 'win32':
                print("  .venv\\Scripts\\activate")
            else:
                print("  source .venv/bin/activate")
        print("  python app.py")
        print()
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
