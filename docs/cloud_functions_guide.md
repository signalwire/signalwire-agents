# SignalWire AI Agents - Cloud Functions Deployment Guide

This guide covers deploying SignalWire AI Agents to Google Cloud Functions and Azure Functions.

## Overview

SignalWire AI Agents now support deployment to major cloud function platforms:

- **Google Cloud Functions** - Serverless compute platform on Google Cloud
- **Azure Functions** - Serverless compute service on Microsoft Azure
- **AWS Lambda** - Already supported (see existing documentation)

## Google Cloud Functions

### Environment Detection

The agent automatically detects Google Cloud Functions environment using these variables:
- `FUNCTION_TARGET` - The function entry point
- `K_SERVICE` - Knative service name (Cloud Run/Functions)
- `GOOGLE_CLOUD_PROJECT` - Google Cloud project ID

### Deployment Steps

1. **Create your agent file** (`main.py`):
```python
import functions_framework
from your_agent_module import YourAgent

# Create agent instance
agent = YourAgent(
    name="my-agent",
    # Configure your agent parameters
)

@functions_framework.http
def agent_handler(request):
    """HTTP Cloud Function entry point"""
    return agent.handle_serverless_request(event=request)
```

2. **Create requirements.txt**:
```
functions-framework==3.*
signalwire-agents
# Add your other dependencies
```

3. **Deploy using gcloud**:
```bash
gcloud functions deploy my-agent \
    --runtime python39 \
    --trigger-http \
    --entry-point agent_handler \
    --allow-unauthenticated
```

### Environment Variables

Set these environment variables for your function:

```bash
# SignalWire credentials
export SIGNALWIRE_PROJECT_ID="your-project-id"
export SIGNALWIRE_TOKEN="your-token"

# Agent configuration
export AGENT_USERNAME="your-username"
export AGENT_PASSWORD="your-password"

# Optional: Custom region/project settings
export FUNCTION_REGION="us-central1"
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

### URL Format

Google Cloud Functions URLs follow this pattern:
```
https://{region}-{project-id}.cloudfunctions.net/{function-name}
```

With authentication:
```
https://username:password@{region}-{project-id}.cloudfunctions.net/{function-name}
```

## Azure Functions

### Environment Detection

The agent automatically detects Azure Functions environment using these variables:
- `AZURE_FUNCTIONS_ENVIRONMENT` - Azure Functions runtime environment
- `FUNCTIONS_WORKER_RUNTIME` - Runtime language (python, node, etc.)
- `AzureWebJobsStorage` - Azure storage connection string

### Deployment Steps

1. **Create your function app structure**:
```
my-agent-function/
├── __init__.py
├── function.json
└── requirements.txt
```

2. **Create `__init__.py`**:
```python
import azure.functions as func
from your_agent_module import YourAgent

# Create agent instance
agent = YourAgent(
    name="my-agent",
    # Configure your agent parameters
)

def main(req: func.HttpRequest) -> func.HttpResponse:
    """Azure Function entry point"""
    return agent.handle_serverless_request(event=req)
```

3. **Create `function.json`**:
```json
{
  "scriptFile": "__init__.py",
  "bindings": [
    {
      "authLevel": "anonymous",
      "type": "httpTrigger",
      "direction": "in",
      "name": "req",
      "methods": ["get", "post"]
    },
    {
      "type": "http",
      "direction": "out",
      "name": "$return"
    }
  ]
}
```

4. **Create `requirements.txt`**:
```
azure-functions
signalwire-agents
# Add your other dependencies
```

5. **Deploy using Azure CLI**:
```bash
# Create function app
az functionapp create \
    --resource-group myResourceGroup \
    --consumption-plan-location westus \
    --runtime python \
    --runtime-version 3.9 \
    --functions-version 4 \
    --name my-agent-function \
    --storage-account mystorageaccount

# Deploy code
func azure functionapp publish my-agent-function
```

### Environment Variables

Set these in your Azure Function App settings:

```bash
# SignalWire credentials
SIGNALWIRE_PROJECT_ID="your-project-id"
SIGNALWIRE_TOKEN="your-token"

# Agent configuration
AGENT_USERNAME="your-username"
AGENT_PASSWORD="your-password"

# Azure-specific (usually auto-set)
AZURE_FUNCTIONS_ENVIRONMENT="Development"
WEBSITE_SITE_NAME="my-agent-function"
```

### URL Format

Azure Functions URLs follow this pattern:
```
https://{function-app-name}.azurewebsites.net/api/{function-name}
```

With authentication:
```
https://username:password@{function-app-name}.azurewebsites.net/api/{function-name}
```

## Authentication

Both platforms support HTTP Basic Authentication:

### Automatic Authentication
The agent automatically validates credentials in cloud function environments:

```python
agent = YourAgent(
    name="my-agent",
    username="your-username",
    password="your-password"
)
```

### Authentication Flow
1. Client sends request with `Authorization: Basic <credentials>` header
2. Agent validates credentials against configured username/password
3. If invalid, returns 401 with `WWW-Authenticate` header
4. If valid, processes the request normally

## Testing

### SignalWire Agent Testing Tool

The SignalWire AI Agents SDK includes a powerful testing tool (`swaig-test`) that can simulate cloud function environments for comprehensive testing before deployment.

#### Cloud Function Environment Simulation

**Google Cloud Functions:**
```bash
# Test SWML generation in GCP environment
swaig-test examples/my_agent.py --simulate-serverless cloud_function --gcp-project my-project --dump-swml

# Test function execution
swaig-test examples/my_agent.py --simulate-serverless cloud_function --gcp-project my-project --exec my_function --param value

# With custom region and service
swaig-test examples/my_agent.py --simulate-serverless cloud_function \
  --gcp-project my-project \
  --gcp-region us-west1 \
  --gcp-service my-service \
  --dump-swml
```

**Azure Functions:**
```bash
# Test SWML generation in Azure environment
swaig-test examples/my_agent.py --simulate-serverless azure_function --dump-swml

# Test function execution
swaig-test examples/my_agent.py --simulate-serverless azure_function --exec my_function --param value

# With custom environment and URL
swaig-test examples/my_agent.py --simulate-serverless azure_function \
  --azure-env Production \
  --azure-function-url https://myapp.azurewebsites.net/api/myfunction \
  --dump-swml
```

#### Environment Variable Testing

Test with custom environment variables:
```bash
# Set individual environment variables
swaig-test examples/my_agent.py --simulate-serverless cloud_function \
  --env GOOGLE_CLOUD_PROJECT=my-project \
  --env DEBUG=1 \
  --exec my_function

# Load from environment file
swaig-test examples/my_agent.py --simulate-serverless azure_function \
  --env-file production.env \
  --dump-swml
```

#### Authentication Testing

Test authentication in cloud function environments:
```bash
# Test with authentication (uses agent's configured credentials)
swaig-test examples/my_agent.py --simulate-serverless cloud_function \
  --gcp-project my-project \
  --dump-swml --verbose

# The tool automatically tests:
# - Basic auth credential embedding in URLs
# - Authentication challenge responses
# - Platform-specific auth handling
```

#### URL Generation Testing

Verify that URLs are generated correctly for each platform:
```bash
# Check URL generation with verbose output
swaig-test examples/my_agent.py --simulate-serverless cloud_function \
  --gcp-project my-project \
  --dump-swml --verbose

# Extract webhook URLs from SWML
swaig-test examples/my_agent.py --simulate-serverless azure_function \
  --dump-swml --raw | jq '.sections.main[1].ai.SWAIG.functions[].web_hook_url'
```

#### Available Testing Options

**Platform Selection:**
- `--simulate-serverless cloud_function` - Google Cloud Functions
- `--simulate-serverless azure_function` - Azure Functions  
- `--simulate-serverless lambda` - AWS Lambda
- `--simulate-serverless cgi` - CGI environment

**Google Cloud Platform Options:**
- `--gcp-project PROJECT_ID` - Set Google Cloud project ID
- `--gcp-region REGION` - Set Google Cloud region (default: us-central1)
- `--gcp-service SERVICE` - Set service name
- `--gcp-function-url URL` - Override function URL

**Azure Functions Options:**
- `--azure-env ENVIRONMENT` - Set Azure environment (default: Development)
- `--azure-function-url URL` - Override Azure Function URL

**Environment Variables:**
- `--env KEY=value` - Set individual environment variables
- `--env-file FILE` - Load environment variables from file

**Output Options:**
- `--dump-swml` - Generate and display SWML document
- `--verbose` - Show detailed information
- `--raw` - Output raw JSON (useful for piping to jq)

#### Complete Testing Workflow

```bash
# 1. List available agents and tools
swaig-test examples/my_agent.py --list-agents
swaig-test examples/my_agent.py --list-tools

# 2. Test SWML generation for each platform
swaig-test examples/my_agent.py --simulate-serverless cloud_function --gcp-project test-project --dump-swml
swaig-test examples/my_agent.py --simulate-serverless azure_function --dump-swml

# 3. Test specific function execution
swaig-test examples/my_agent.py --simulate-serverless cloud_function --gcp-project test-project --exec search_knowledge --query "test"

# 4. Test with production-like environment
swaig-test examples/my_agent.py --simulate-serverless azure_function --env-file production.env --exec my_function --param value

# 5. Verify authentication and URL generation
swaig-test examples/my_agent.py --simulate-serverless cloud_function --gcp-project prod-project --dump-swml --verbose
```

### Local Testing

**Google Cloud Functions:**
```bash
# Install Functions Framework
pip install functions-framework

# Run locally
functions-framework --target=agent_handler --debug
```

**Azure Functions:**
```bash
# Install Azure Functions Core Tools
npm install -g azure-functions-core-tools@4

# Run locally
func start
```

### Testing Authentication

```bash
# Test without auth (should return 401)
curl https://your-function-url/

# Test with valid auth
curl -u username:password https://your-function-url/

# Test SWAIG function call
curl -u username:password \
  -H "Content-Type: application/json" \
  -d '{"call_id": "test", "argument": {"parsed": [{"param": "value"}]}}' \
  https://your-function-url/your_function_name
```

## Best Practices

### Performance
- Use connection pooling for database connections
- Implement proper caching strategies
- Minimize cold start times with smaller deployment packages

### Security
- Always use HTTPS endpoints
- Implement proper authentication
- Use environment variables for sensitive data
- Consider using cloud-native secret management

### Monitoring
- Enable cloud platform logging
- Monitor function execution times
- Set up alerts for errors and timeouts
- Use distributed tracing for complex workflows

### Cost Optimization
- Right-size memory allocation
- Implement proper timeout settings
- Use reserved capacity for predictable workloads
- Monitor and optimize function execution patterns

## Troubleshooting

### Common Issues

**Environment Detection:**
```python
# Check detected mode
from signalwire_agents.core.logging_config import get_execution_mode
print(f"Detected mode: {get_execution_mode()}")
```

**URL Generation:**
```python
# Check generated URLs
agent = YourAgent(name="test")
print(f"Base URL: {agent.get_full_url()}")
print(f"Auth URL: {agent.get_full_url(include_auth=True)}")
```

**Authentication Issues:**
- Verify username/password are set correctly
- Check that Authorization header is being sent
- Ensure credentials match exactly (case-sensitive)

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check environment variables:
```python
import os
for key, value in os.environ.items():
    if 'FUNCTION' in key or 'AZURE' in key or 'GOOGLE' in key:
        print(f"{key}: {value}")
```

## Migration from Other Platforms

### From AWS Lambda
- Update environment variable names
- Modify request/response handling if needed
- Update deployment scripts

### From Traditional Servers
- Add cloud function entry point
- Configure environment variables
- Update URL generation logic
- Test authentication flow

## Examples

See `examples/lambda_agent.py` for a complete AWS Lambda deployment example.

## Support

For issues specific to cloud function deployment:
1. Check the troubleshooting section above
2. Verify environment variables are set correctly
3. Test authentication flow manually
4. Check cloud platform logs for detailed error messages
5. Refer to platform-specific documentation for deployment issues 