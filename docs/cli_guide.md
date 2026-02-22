# CLI Guide

This guide covers the command-line tools included with the SignalWire Agents SDK.

## Overview

The `swaig-test` CLI tool provides a complete testing environment for:
- **SWAIG Functions**: Both webhook and DataMap functions with automatic type detection
- **SWML Generation**: Static and dynamic agent SWML document testing with realistic fake data
- **Mock Requests**: Complete FastAPI Request simulation for dynamic agent testing

The tool automatically detects function types, provides appropriate execution environments, and simulates the SignalWire platform locally while making real HTTP requests for DataMap functions.

## Key Features

- **`--exec` Syntax**: Modern CLI-style function arguments
- **Agent Auto-Selection**: Automatically chooses agent when only one exists in file
- **Agent Discovery**: Lists available agents when no arguments provided
- **Auto-Detection**: Automatically detects webhook vs DataMap functions - no manual flags needed
- **Complete DataMap Simulation**: Full processing including URL templates, responses, and fallbacks
- **SWML Testing**: Generate and test SWML documents with realistic fake call data
- **Dynamic Agent Support**: Test request-dependent SWML generation with mock request objects
- **Real HTTP Execution**: DataMap functions make actual HTTP requests to real APIs
- **Comprehensive Simulation**: Generate realistic post_data with all SignalWire metadata
- **Advanced Template Engine**: Supports all DataMap variable syntax (`${args.param}`, `${response.field}`, `${array[0].property}`)
- **Flexible CLI Syntax**: Support both `--exec` and JSON argument styles
- **Override System**: Precise control over test data with dot notation paths
- **Mock Request Objects**: Complete FastAPI Request simulation for dynamic agents
- **Verbose Debugging**: Detailed execution tracing for both function types
- **Flexible Data Modes**: Choose between minimal, comprehensive, or custom post_data
- **Serverless Environment Simulation**: Complete platform simulation for Lambda, CGI, Cloud Functions, and Azure Functions with environment variable management
- **Automatic Log Suppression**: Logs are suppressed by default unless `--verbose` flag is used
- **Enhanced Parameter Display**: Shows all JSON Schema constraints including enum values, min/max, patterns, and more

## Installation

Install as part of the signalwire_agents package:

```bash
pip install -e .
swaig-test --help
```

## Quick Start

### Agent Discovery

The tool can automatically discover agents in Python files:

```bash
# Discover all agents in a file (auto-runs when no other args provided)
swaig-test examples/joke_skill_demo.py

# Explicitly list available agents
swaig-test matti_and_sigmond/dual_agent_app.py --list-agents

# List agents with details
swaig-test matti_and_sigmond/dual_agent_app.py --list-agents --verbose
```

**Example Output:**
```
Available agents in matti_and_sigmond/dual_agent_app.py:

  MattiAgent
    Type: Ready instance
    Name: Matti
    Route: /matti-agent
    Description: Advanced agent with custom tools and weather integration

  SigmondAgent
    Type: Ready instance  
    Name: Sigmond
    Route: /sigmond-agent
    Description: Advanced conversational agent with data access

To use a specific agent with this tool:
  swaig-test matti_and_sigmond/dual_agent_app.py [tool_name] [args] --agent-class <AgentClassName>
  swaig-test matti_and_sigmond/dual_agent_app.py [tool_name] [args] --route <route_path>

Examples:
  swaig-test matti_and_sigmond/dual_agent_app.py --list-tools --agent-class MattiAgent
  swaig-test matti_and_sigmond/dual_agent_app.py --dump-swml --agent-class SigmondAgent
  swaig-test matti_and_sigmond/dual_agent_app.py --list-tools --route /matti-agent
  swaig-test matti_and_sigmond/dual_agent_app.py --dump-swml --route /sigmond-agent
```

### List Available Functions

```bash
# List functions in single-agent file (auto-selected)
swaig-test examples/joke_skill_demo.py --list-tools

# List functions for specific agent in multi-agent file
swaig-test matti_and_sigmond/dual_agent_app.py --agent-class MattiAgent --list-tools

# List functions using route selection
swaig-test matti_and_sigmond/dual_agent_app.py --route /matti-agent --list-tools

# Detailed function listing with schemas
swaig-test examples/joke_skill_demo.py --list-tools --verbose
```

**Example Output:**
```
Available SWAIG functions:
  get_joke - Get a random joke from API Ninjas (DataMap function - serverless)
    Parameters:
      type (string [options: jokes, dadjokes]) (required): Type of joke to get
    Config: {"data_map": {...}, "parameters": {...}}
      
  calculate - Perform mathematical calculations and return the result (LOCAL webhook)
    Parameters:
      expression (string) (required): Mathematical expression to evaluate
      precision (integer [min: 0, max: 10]): Number of decimal places (default: 2)
```

### Test SWML Generation

```bash
# Basic SWML generation with fake call data
swaig-test examples/my_agent.py --dump-swml

# Raw SWML JSON output for piping
swaig-test examples/my_agent.py --dump-swml --raw | jq '.'

# Verbose SWML testing with detailed fake data
swaig-test examples/my_agent.py --dump-swml --verbose

# Custom call types and scenarios
swaig-test examples/my_agent.py --dump-swml --call-type sip --call-direction outbound

# Test SWML in serverless environments
swaig-test examples/my_agent.py --simulate-serverless lambda --dump-swml
swaig-test examples/my_agent.py --simulate-serverless cgi --cgi-host example.com --dump-swml
```

## Logging and Output Control

By default, `swaig-test` suppresses agent logs to keep output clean. Use these options to control logging:

```bash
# Default behavior - logs are suppressed
swaig-test examples/my_agent.py --exec my_function --param value

# Enable logs with --verbose flag  
swaig-test examples/my_agent.py --verbose --exec my_function --param value

# Clean SWML output (logs always suppressed)
swaig-test examples/my_agent.py --dump-swml --raw
```

The tool automatically:
- Suppresses logs by default for cleaner output
- Shows logs when `--verbose` is specified
- Always suppresses output when using `--dump-swml` to ensure valid JSON

## Serverless Environment Simulation

The CLI tool provides comprehensive serverless platform simulation, allowing you to test your agents in Lambda, CGI, Cloud Functions, and Azure Functions environments locally without deployment.

### Quick Start with Serverless Simulation

```bash
# Test agent in Lambda environment
swaig-test examples/my_agent.py --simulate-serverless lambda --dump-swml

# Test function execution in Lambda context
swaig-test examples/my_agent.py --simulate-serverless lambda --exec my_function --param value

# Test with custom Lambda configuration
swaig-test examples/my_agent.py --simulate-serverless lambda --aws-function-name my-func --aws-region us-west-2 --exec my_function

# Test CGI environment with custom host
swaig-test examples/my_agent.py --simulate-serverless cgi --cgi-host example.com --dump-swml

# Test with environment variables
swaig-test examples/my_agent.py --simulate-serverless lambda --env DEBUG=1 --env TEST_MODE=cli --exec my_function
```

### Supported Serverless Platforms

| Platform | Simulation Flag | Key Features |
|----------|-----------------|--------------|
| **AWS Lambda** | `--simulate-serverless lambda` | Function URLs, API Gateway, environment detection |
| **CGI** | `--simulate-serverless cgi` | HTTP host, script paths, HTTPS simulation |
| **Google Cloud Functions** | `--simulate-serverless cloud_function` | Function URLs, project configuration |
| **Azure Functions** | `--simulate-serverless azure_function` | Function URLs, environment settings |

### Platform-Specific Configuration

#### AWS Lambda Simulation

```bash
# Default Lambda simulation with auto-generated URLs
swaig-test examples/my_agent.py --simulate-serverless lambda --dump-swml

# Custom Lambda function configuration
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --aws-function-name my-custom-function \
  --aws-function-url https://abc123.lambda-url.us-west-2.on.aws/ \
  --aws-region us-west-2 \
  --dump-swml

# API Gateway configuration
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --aws-api-gateway-id abc123def \
  --aws-region us-east-1 \
  --aws-stage prod \
  --exec my_function --param value

# Test function execution in Lambda context
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --exec get_weather --location "San Francisco" \
  --full-request
```

**Lambda Environment Variables Set:**
- `AWS_LAMBDA_FUNCTION_NAME`
- `AWS_LAMBDA_FUNCTION_URL` (if using Function URLs)
- `AWS_API_GATEWAY_ID` (if using API Gateway)
- `AWS_REGION`
- `_HANDLER`

#### CGI Simulation

```bash
# Basic CGI simulation
swaig-test examples/my_agent.py --simulate-serverless cgi --cgi-host example.com --dump-swml

# Custom CGI configuration
swaig-test examples/my_agent.py --simulate-serverless cgi \
  --cgi-host my-server.com \
  --cgi-script-name /cgi-bin/my-agent.cgi \
  --cgi-https \
  --cgi-path-info /custom/path \
  --exec my_function --param value

# Test CGI with specific environment
swaig-test examples/my_agent.py --simulate-serverless cgi \
  --cgi-host production.example.com \
  --cgi-https \
  --env REMOTE_USER=admin \
  --dump-swml
```

**CGI Environment Variables Set:**
- `GATEWAY_INTERFACE=CGI/1.1`
- `HTTP_HOST` (from --cgi-host)
- `SCRIPT_NAME` (from --cgi-script-name)
- `HTTPS=on` (if --cgi-https)
- `PATH_INFO` (from --cgi-path-info)

#### Google Cloud Functions Simulation

```bash
# Basic Cloud Function simulation
swaig-test examples/my_agent.py --simulate-serverless cloud_function --dump-swml

# Custom GCP configuration
swaig-test examples/my_agent.py --simulate-serverless cloud_function \
  --gcp-project my-project \
  --gcp-function-url https://my-function-abc123.cloudfunctions.net \
  --gcp-region us-central1 \
  --gcp-service my-service \
  --exec my_function --param value
```

**Cloud Function Environment Variables Set:**
- `GOOGLE_CLOUD_PROJECT`
- `FUNCTION_URL` (if provided)
- `GOOGLE_CLOUD_REGION`
- `K_SERVICE` (Knative service name)

#### Azure Functions Simulation

```bash
# Basic Azure Functions simulation
swaig-test examples/my_agent.py --simulate-serverless azure_function --dump-swml

# Custom Azure configuration
swaig-test examples/my_agent.py --simulate-serverless azure_function \
  --azure-env production \
  --azure-function-url https://my-function.azurewebsites.net \
  --exec my_function --param value
```

**Azure Functions Environment Variables Set:**
- `AZURE_FUNCTIONS_ENVIRONMENT`
- `WEBSITE_SITE_NAME`
- Custom function URL (if provided)

### Environment Variable Management

#### Manual Environment Variables

```bash
# Set custom environment variables
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --env API_KEY=secret123 \
  --env DEBUG=true \
  --env TIMEOUT=30 \
  --exec my_function

# Multiple environment variables
swaig-test examples/my_agent.py --simulate-serverless cgi \
  --env DB_HOST=localhost \
  --env DB_PORT=5432 \
  --env LOG_LEVEL=info \
  --cgi-host example.com \
  --dump-swml
```

#### Environment Files

Create environment files for reusable configurations:

```bash
# Create environment file
cat > lambda.env << EOF
AWS_LAMBDA_FUNCTION_NAME=my-production-function
AWS_REGION=us-west-2
API_KEY=prod_key_123
DEBUG=false
TIMEOUT=60
EOF

# Use environment file
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --env-file lambda.env \
  --exec my_function --param value

# Override specific variables from file
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --env-file lambda.env \
  --env DEBUG=true \
  --env AWS_REGION=us-east-1 \
  --dump-swml
```

### Webhook URL Generation

The serverless simulation automatically generates appropriate webhook URLs for each platform:

#### Platform-Specific URLs

| Platform | Example Webhook URL |
|----------|-------------------|
| **Lambda (Function URL)** | `https://abc123.lambda-url.us-east-1.on.aws/swaig/` |
| **Lambda (API Gateway)** | `https://api123.execute-api.us-east-1.amazonaws.com/prod/swaig/` |
| **CGI** | `https://example.com/cgi-bin/agent.cgi/swaig/` |
| **Cloud Functions** | `https://my-function-abc123.cloudfunctions.net/swaig/` |
| **Azure Functions** | `https://my-function.azurewebsites.net/swaig/` |

#### URL Generation Examples

```bash
# Lambda Function URL
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --aws-function-url https://custom123.lambda-url.us-west-2.on.aws/ \
  --dump-swml --format-json | jq '.sections.main[1].ai.SWAIG.defaults.web_hook_url'

# CGI with custom host
swaig-test examples/my_agent.py --simulate-serverless cgi \
  --cgi-host my-production-server.com \
  --cgi-https \
  --dump-swml --format-json | jq '.sections.main[1].ai.SWAIG.defaults.web_hook_url'

# Cloud Functions with custom URL
swaig-test examples/my_agent.py --simulate-serverless cloud_function \
  --gcp-function-url https://my-custom-function.cloudfunctions.net \
  --dump-swml --format-json | jq '.sections.main[1].ai.SWAIG.defaults.web_hook_url'
```

### Function Execution in Serverless Context

Test function execution with platform-specific request/response formats:

#### Lambda Function Execution

```bash
# Test function in Lambda context
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --exec get_weather --location "Miami" \
  --full-request

# Example output shows Lambda event format
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --exec calculate --expression "2+2" \
  --full-request --format-json
```

**Lambda Response Format:**
```json
{
  "statusCode": 200,
  "headers": {
    "Content-Type": "application/json"
  },
  "body": "{\"result\": 4, \"expression\": \"2+2\"}"
}
```

#### CGI Function Execution

```bash
# Test function in CGI context
swaig-test examples/my_agent.py --simulate-serverless cgi \
  --cgi-host example.com \
  --exec my_function --param value
```

### Advanced Serverless Features

#### Environment Presets

The tool includes built-in environment presets for each platform:

```bash
# Use default Lambda preset
swaig-test examples/my_agent.py --simulate-serverless lambda --dump-swml

# Override preset values
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --aws-function-name custom-name \
  --env CUSTOM_VAR=value \
  --dump-swml
```

#### Environment Conflict Resolution

The tool automatically clears conflicting environment variables between platforms:

```bash
# Switching platforms clears previous environment
export AWS_LAMBDA_FUNCTION_NAME=old-function

# This will clear AWS variables and set GCP variables
swaig-test examples/my_agent.py --simulate-serverless cloud_function \
  --gcp-project new-project \
  --dump-swml
```

#### Testing Multiple Platforms

```bash
# Test the same agent across multiple platforms
for platform in lambda cgi cloud_function azure_function; do
  echo "Testing $platform..."
  swaig-test examples/my_agent.py --simulate-serverless $platform \
    --exec my_function --param value
done

# Compare SWML generation across platforms
swaig-test examples/my_agent.py --simulate-serverless lambda --dump-swml > lambda.swml
swaig-test examples/my_agent.py --simulate-serverless cgi --cgi-host example.com --dump-swml > cgi.swml
diff lambda.swml cgi.swml
```

### Debugging Serverless Simulation

#### Verbose Mode

```bash
# See detailed environment setup
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --verbose \
  --dump-swml

# Debug function execution
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --verbose \
  --exec my_function --param value \
  --full-request
```

#### Environment Inspection

```bash
# Show environment variables being set
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --env DEBUG=1 \
  --exec get_status  # Use a function that returns environment info
```

#### Format Options

```bash
# Pretty-print JSON output
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --dump-swml --format-json

# Raw JSON for piping
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --dump-swml --format-json | jq '.sections.main[1].ai.SWAIG.functions[0]'
```

### Serverless Best Practices

#### Development Workflow

1. **Start with local testing**: Test your agent normally first
2. **Test each platform**: Use serverless simulation for target platforms
3. **Verify webhook URLs**: Check that URLs are generated correctly for your platform
4. **Test environment variables**: Ensure your agent works with platform-specific variables
5. **Test function execution**: Verify functions work in serverless context

#### Platform-Specific Testing

```bash
# Lambda development workflow
swaig-test examples/my_agent.py --list-tools  # First test locally
swaig-test examples/my_agent.py --simulate-serverless lambda --dump-swml  # Check SWML
swaig-test examples/my_agent.py --simulate-serverless lambda --exec my_function --param value  # Test functions

# Production-like testing
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --env-file production.env \
  --aws-function-name prod-my-agent \
  --aws-region us-east-1 \
  --exec critical_function --input "test"
```

#### Environment Management

```bash
# Development environment
cat > dev.env << EOF
DEBUG=true
LOG_LEVEL=debug
API_TIMEOUT=10
EOF

# Production environment  
cat > prod.env << EOF
DEBUG=false
LOG_LEVEL=info
API_TIMEOUT=30
EOF

# Test both environments
swaig-test examples/my_agent.py --simulate-serverless lambda --env-file dev.env --exec my_function
swaig-test examples/my_agent.py --simulate-serverless lambda --env-file prod.env --exec my_function
```

### Legacy Compatibility

The tool maintains backward compatibility with existing serverless parameters:

```bash
# Legacy syntax (still supported)
swaig-test examples/my_agent.py --serverless-mode lambda --function my_function --args '{"param":"value"}'

# New syntax (recommended)
swaig-test examples/my_agent.py --simulate-serverless lambda --exec my_function --param value
```

### CLI Syntax with --exec

The `--exec` syntax provides an intuitive way to test functions:

```bash
# --exec syntax (recommended) - CLI flags BEFORE --exec
swaig-test examples/joke_skill_demo.py --verbose --exec get_joke --type dadjokes
swaig-test examples/web_search_agent.py --exec web_search --query "AI agents" --limit 5
swaig-test matti_and_sigmond/dual_agent_app.py --agent-class MattiAgent --exec get_weather --location "New York"

# Multiple agents - specify which one to use
swaig-test matti_and_sigmond/dual_agent_app.py --verbose --agent-class SigmondAgent --exec search_knowledge --query "SignalWire"

# Auto-agent selection (when only one agent in file)
swaig-test examples/joke_skill_demo.py --exec get_joke --type jokes

# All CLI flags must come BEFORE --exec
swaig-test examples/agent.py --verbose --custom-data '{"test":"data"}' --exec my_function --param value
```

### JSON Syntax (Alternative)

```bash
# JSON syntax (alternative approach)
swaig-test examples/joke_skill_demo.py get_joke '{"type":"dadjokes"}'
swaig-test examples/web_search_agent.py web_search '{"query":"AI agents","limit":5}'
```

## CLI Argument Syntax

### Using --exec (Recommended)

The `--exec` syntax separates CLI flags from function arguments:

```bash
# Basic usage
swaig-test <file> [--cli-flags] --exec <function> [--function-args]

# All CLI flags must come BEFORE --exec
swaig-test examples/agent.py --verbose --agent-class MyAgent --exec search --query "test" --limit 10

# Function arguments come AFTER --exec function-name
swaig-test examples/joke_skill_demo.py --exec get_joke --type dadjokes
#                                      ^^^^                            ^^^^^^^^
#                                      Wrong place!                    Wrong place!

# Correct usage - CLI flags before --exec
swaig-test examples/joke_skill_demo.py --verbose --exec get_joke --type dadjokes
#                                      ^^^^^^^^         ^^^^^^^^ ^^^^^^^^^^^^^^
#                                      CLI flag         Function Function args
```

### Argument Type Handling

The tool automatically converts arguments based on function schema:

| Schema Type | CLI Input | Converted Value |
|-------------|-----------|-----------------|
| `string` | `--name "value"` | `"value"` |
| `integer` | `--count 42` | `42` |
| `number` | `--price 19.99` | `19.99` |
| `boolean` | `--verbose` or `--verbose true` | `true` |
| `array` | `--tags "tag1,tag2,tag3"` | `["tag1","tag2","tag3"]` |

### CLI Syntax Examples

```bash
# String parameters
swaig-test examples/agent.py --exec greet --name "Alice"

# Multiple parameters with type conversion
swaig-test examples/agent.py --exec search --query "AI" --limit 5 --include-metadata

# Boolean flags
swaig-test examples/agent.py --exec process --input "data" --verify --async false

# Array parameters (comma-separated)
swaig-test examples/agent.py --exec filter --categories "tech,science,health" --max-results 20

# Complex example with multiple agent support
swaig-test matti_and_sigmond/dual_agent_app.py --verbose --agent-class SigmondAgent --exec get_trivia --category science
```

## Multi-Agent Support

### Agent Auto-Selection

When a file contains only one agent, it's automatically selected:

```bash
# Auto-selects the single agent
swaig-test examples/joke_skill_demo.py --exec get_joke --type jokes
```

### Multi-Agent Files

For files with multiple agents, specify which one to use:

```bash
# Discover available agents
swaig-test matti_and_sigmond/dual_agent_app.py

# Use specific agent
swaig-test matti_and_sigmond/dual_agent_app.py --agent-class MattiAgent --exec get_weather --location "San Francisco"
swaig-test matti_and_sigmond/dual_agent_app.py --agent-class SigmondAgent --exec search_knowledge --query "AI"

# Different operations with different agents
swaig-test matti_and_sigmond/dual_agent_app.py --agent-class MattiAgent --list-tools
swaig-test matti_and_sigmond/dual_agent_app.py --agent-class SigmondAgent --dump-swml
```

## DataMap Function Testing

### Automatic DataMap Detection

DataMap functions are automatically detected and properly simulated:

```bash
# DataMap function - automatically detected and simulated
swaig-test examples/joke_skill_demo.py --verbose --exec get_joke --type dadjokes
```

**Complete DataMap Processing Pipeline:**
1. **URL Template Expansion**: `${args.type}` → `dadjokes`
2. **HTTP Request**: GET to `https://api.api-ninjas.com/v1/dadjokes`
3. **Response Processing**: Extract joke from API response array
4. **Output Template**: `${array[0].joke}` → actual joke text
5. **Fallback Handling**: If API fails, use fallback message

**Example Output:**
```
Executing DataMap function: get_joke
Arguments: {"type": "dadjokes"}
------------------------------------------------------------
Simple DataMap structure with 1 webhook(s)
Processing 1 webhook(s)...
  Webhook 1: GET https://api.api-ninjas.com/v1/${args.type}
    Original URL: https://api.api-ninjas.com/v1/${args.type}
    Template context: {'args': {'type': 'dadjokes'}, 'array': [], 'type': 'dadjokes'}
    Expanded URL: https://api.api-ninjas.com/v1/dadjokes
  ✓ Webhook succeeded: 200
    Response: [{"joke": "Why don't scientists trust atoms? Because they make up everything!"}]
    Processed output: {'response': "Here's a joke: Why don't scientists trust atoms? Because they make up everything!"}

RESULT:
Response: Here's a joke: Why don't scientists trust atoms? Because they make up everything!
```

### DataMap Template Expansion

The tool properly handles all DataMap template syntax:

- **Function Arguments**: `${args.type}`, `${args.location}`
- **Array Access**: `${array[0].joke}`, `${array[0].weather.temp}`
- **Nested Objects**: `${response.data.results[0].title}`
- **Fallback Values**: `${args.units || "metric"}`

### DataMap Error Handling

When APIs fail, DataMap functions gracefully fall back:

```bash
# Test with invalid parameters to see fallback
swaig-test examples/joke_skill_demo.py --verbose --exec get_joke --type invalid
```

**Fallback Output:**
```
  ✗ Webhook failed: 404
All webhooks failed, using fallback output...
Fallback result = {'response': 'Sorry, there is a problem with the joke service right now. Please try again later.'}

RESULT:
Response: Sorry, there is a problem with the joke service right now. Please try again later.
```

### Test Functions (Auto-Detection)

The tool automatically detects whether a function is a local webhook, external webhook, or DataMap function:

```bash
# Test local webhook function - auto-detected
swaig-test examples/datasphere_webhook_env_demo.py search_knowledge '{"query":"SignalWire"}'

# Test DataMap function - auto-detected  
swaig-test examples/datasphere_serverless_env_demo.py search_knowledge '{"query":"SignalWire"}'

# Test external webhook function - auto-detected
swaig-test examples/external_webhook_weather_agent.py getWeather '{"location":"New York"}' --verbose

# Test math skill function - auto-detected
swaig-test examples/datasphere_serverless_env_demo.py calculate '{"expression":"25 * 47"}'
```

#### External Webhook Function Testing

External webhook functions are automatically detected and tested by making HTTP requests to the external service URL:

```bash
# Test external webhook with verbose output
swaig-test examples/external_webhook_weather_agent.py getWeather '{"location":"San Francisco"}' --verbose

# List functions with their types (local vs external)
swaig-test examples/external_webhook_weather_agent.py --list-tools
```

**Example Output for External Webhook:**
```
Available SWAIG functions:
  getHelp - Get help information about using the weather service (LOCAL webhook)
  getWeather - Get current weather information for a specific location (EXTERNAL webhook)
    External URL: https://api.example-weather-service.com/webhook

Calling EXTERNAL webhook: getWeather
URL: https://api.example-weather-service.com/webhook
Arguments: {"location": "San Francisco"}

Sending payload: {
  "function": "getWeather",
  "argument": {
    "parsed": [{"location": "San Francisco"}],
    "raw": "{\"location\": \"San Francisco\"}"
  },
  "call_id": "test-call-123"
}
Making POST request to: https://api.example-weather-service.com/webhook
Response status: 200
✓ External webhook succeeded
```

**How External Webhook Testing Works:**

1. **Detection**: The CLI tool detects functions with `webhook_url` parameters as external webhooks
2. **HTTP Request**: Instead of calling the local function, it makes an HTTP POST to the external URL
3. **Payload Format**: Sends the same JSON payload that SignalWire would send:
   ```json
   {
     "function": "function_name",
     "argument": {
       "parsed": [{"param": "value"}],
       "raw": "{\"param\": \"value\"}"
     },
     "call_id": "generated-call-id",
     "call": { /* call information */ },
     "vars": { /* call variables */ }
   }
   ```
4. **Response Handling**: Processes the HTTP response and displays the result
5. **Error Handling**: Shows connection errors, timeouts, and HTTP error responses

**Testing Mixed Function Types:**

You can test agents that have both local and external webhook functions:

```bash
# Test local function
swaig-test examples/external_webhook_weather_agent.py getHelp '{}'

# Test external function  
swaig-test examples/external_webhook_weather_agent.py getWeather '{"location":"Tokyo"}'

# Show all function types
swaig-test examples/external_webhook_weather_agent.py --list-tools
```

## SWML Generation and Testing

### Realistic SWML Post Data

The tool automatically generates realistic fake SWML post_data that matches SignalWire's structure:

```bash
# Generate SWML with fake call data
swaig-test examples/my_agent.py --dump-swml --verbose
```

**Generated fake post_data structure:**
```json
{
  "call": {
    "call_id": "550e8400-e29b-41d4-a716-446655440000",
    "node_id": "test-node-a1b2c3d4",
    "segment_id": "550e8400-e29b-41d4-a716-446655440001", 
    "call_session_id": "550e8400-e29b-41d4-a716-446655440002",
    "tag": "550e8400-e29b-41d4-a716-446655440000",
    "state": "created",
    "direction": "inbound",
    "type": "webrtc",
    "from": "user-a1b2c3d4@test.domain",
    "to": "agent-e5f6g7h8@test.domain",
    "project_id": "550e8400-e29b-41d4-a716-446655440003",
    "space_id": "550e8400-e29b-41d4-a716-446655440004"
  },
  "vars": {
    "userVariables": {}
  },
  "envs": {}
}
```

### Call Type Simulation

Support for different call types with appropriate metadata:

```bash
# WebRTC call (default)
swaig-test examples/agent.py --dump-swml --call-type webrtc

# SIP call with phone numbers
swaig-test examples/agent.py --dump-swml --call-type sip
```

**SIP vs WebRTC differences:**
- **SIP**: Uses phone numbers (+15551234567), includes SIP headers
- **WebRTC**: Uses domain addresses (user@domain), includes WebRTC headers

### SWML Testing Options

| Option | Description | Example |
|--------|-------------|---------|
| `--dump-swml` | Generate SWML document with fake call data | `--dump-swml` |
| `--raw` | Output raw JSON only (pipeable) | `--dump-swml --raw \| jq '.'` |
| `--call-type` | SIP or WebRTC call simulation | `--call-type sip` |
| `--call-direction` | Inbound or outbound call | `--call-direction outbound` |
| `--call-state` | Call state (created, answered, etc.) | `--call-state answered` |
| `--call-id` | Override call_id | `--call-id my-test-call` |
| `--project-id` | Override project_id | `--project-id my-project` |
| `--space-id` | Override space_id | `--space-id my-space` |
| `--from-number` | Override from address | `--from-number +15551234567` |
| `--to-extension` | Override to address | `--to-extension +15559876543` |

### Data Override System

Precise control over fake data using dot notation paths:

```bash
# Simple value overrides
swaig-test examples/agent.py --dump-swml --override call.state=answered --override call.timeout=60

# JSON overrides for complex data
swaig-test examples/agent.py --dump-swml --override-json vars.userVariables='{"vip":true,"tier":"gold"}'

# User variables and environment variables
swaig-test examples/agent.py --dump-swml --user-vars '{"customer_id":"12345","tier":"premium"}'

# Query parameters (merged into userVariables)
swaig-test examples/agent.py --dump-swml --query-params '{"source":"api","debug":"true"}'
```

**Override Examples:**
```bash
# Multiple overrides
swaig-test examples/agent.py --dump-swml \
  --override call.project_id=my-project \
  --override call.direction=outbound \
  --override call.state=answered \
  --user-vars '{"vip_customer":true}'

# Complex JSON overrides
swaig-test examples/agent.py --dump-swml \
  --override-json call.headers='{"X-Custom":"value"}' \
  --override-json vars.userVariables='{"settings":{"theme":"dark","lang":"en"}}'
```

### Dynamic Agent Testing

Test dynamic agents that generate request-dependent SWML:

```bash
# Basic dynamic agent testing
swaig-test examples/dynamic_agent.py --dump-swml --header "Authorization=Bearer token"

# Custom request simulation
swaig-test examples/dynamic_agent.py --dump-swml \
  --method GET \
  --header "X-Source=api" \
  --header "Content-Type=application/json" \
  --query-params '{"source":"test","version":"v2"}' \
  --body '{"custom_data":"test"}'

# Combined dynamic testing
swaig-test examples/dynamic_agent.py --dump-swml \
  --call-type sip \
  --call-direction outbound \
  --header "X-Call-Source=external" \
  --user-vars '{"priority":"high"}' \
  --verbose
```

**Mock Request Features:**
- **Headers**: Case-insensitive HTTP headers
- **Query Parameters**: Case-sensitive query parameters  
- **Request Body**: JSON request body
- **HTTP Method**: GET, POST, PUT, etc.
- **URL**: Full URL with query string
- **Async Methods**: Compatible with FastAPI Request interface

### SWML Output Formats

```bash
# Standard output with headers
swaig-test examples/agent.py --dump-swml
# Output: Headers + formatted SWML + footers

# Raw JSON for automation
swaig-test examples/agent.py --dump-swml --raw
# Output: Raw JSON only

# Pipe to jq for processing
swaig-test examples/agent.py --dump-swml --raw | jq '.sections.main[1].ai.SWAIG.functions'

# Verbose with fake data details
swaig-test examples/agent.py --dump-swml --verbose
# Output: Fake data details + agent info + SWML
```

## Alternative CLI Argument Syntax

### Using --args Separator

Instead of JSON strings, use CLI-style arguments:

```bash
# Traditional JSON syntax
swaig-test examples/agent.py search_function '{"query":"test","limit":10,"verbose":true}'

# Alternative CLI syntax  
swaig-test examples/agent.py search_function --args --query "test" --limit 10 --verbose

# Schema-based type conversion
swaig-test examples/agent.py calculate --args --expression "25 * 47" --precision 2
```

### Argument Type Handling

The tool automatically converts arguments based on function schema:

| Schema Type | CLI Input | Converted Value |
|-------------|-----------|-----------------|
| `string` | `--name "value"` | `"value"` |
| `integer` | `--count 42` | `42` |
| `number` | `--price 19.99` | `19.99` |
| `boolean` | `--verbose` or `--verbose true` | `true` |
| `array` | `--tags "tag1,tag2,tag3"` | `["tag1","tag2","tag3"]` |

### CLI Syntax Examples

```bash
# Simple string parameter
swaig-test examples/agent.py greet --args --name "Alice"

# Multiple parameters with type conversion
swaig-test examples/agent.py search --args --query "AI" --limit 5 --include-metadata

# Boolean flags
swaig-test examples/agent.py process --args --input "data" --verify --async false

# Array parameters (comma-separated)
swaig-test examples/agent.py filter --args --categories "tech,science,health" --max-results 20
```

## DataMap Function Execution

### Complete Processing Pipeline

DataMap functions follow the SignalWire server-side processing pipeline:

1. **Expression Processing**: Pattern matching against function arguments
2. **Webhook Execution**: Sequential HTTP requests until one succeeds  
3. **Foreach Processing**: Array iteration with template expansion
4. **Output Generation**: Final result formatting using templates
5. **Fallback Handling**: Error recovery with fallback outputs

### Real API Execution Example

```bash
# Test DataSphere serverless search with verbose output
swaig-test examples/datasphere_serverless_env_demo.py search_knowledge '{"query":"SignalWire"}' --verbose
```

**Example Execution Flow:**
```
=== DataMap Function Execution ===
Config: { ... complete datamap configuration ... }

--- Processing Webhooks ---
=== Webhook 1/1 ===
Making POST request to: https://tony.signalwire.com/api/datasphere/documents/search
Headers: {
  "Content-Type": "application/json",
  "Authorization": "Basic ODQ2NTlmMjE..."
}
Request data: {
  "document_id": "b888a1cc-1707-4902-9573-aa201a0c1086", 
  "query_string": "SignalWire",
  "distance": "4.0",
  "count": "1"
}
Response status: 200
Webhook 1 succeeded!

--- Processing Webhook Foreach ---
Found array data in response.chunks: 1 items
Processed 1 items
Foreach result (formatted_results): === RESULT ===
SignalWire's competitive advantage comes from...

--- Processing Webhook Output ---
Set response = I found results for "SignalWire":

=== RESULT ===
SignalWire's competitive advantage comes from...

RESULT:
Response: I found results for "SignalWire": ...
```

### Template Expansion Support

The tool supports all DataMap template syntax with both `${}` and `%{}` variations:

| Syntax | Description | Example |
|--------|-------------|---------|
| `${args.param}` / `%{args.param}` | Function arguments | `${args.query}`, `%{args.type}` |
| `${response.field}` / `%{response.field}` | API response object | `${response.temperature}` |
| `${array[0].field}` / `%{array[0].field}` | API response array | `${array[0].joke}`, `%{array[0].text}` |
| `${this.property}` / `%{this.property}` | Current foreach item | `${this.title}`, `%{this.content}` |
| `${global_data.key}` / `%{global_data.key}` | Call-wide data store | `${global_data.customer_name}` |

**Array Response Handling**: When a webhook returns a nameless array (like `[{"joke": "..."}]`), it's automatically stored as the `array` key, making it accessible via `${array[0].property}` syntax.

**Template Expansion Examples:**
```json
{
  "url": "https://api.example.com/v1/%{args.type}",
  "output": {
    "response": "Here's a joke: ${array[0].joke}"
  }
}
```

### Foreach Processing

DataMap foreach loops concatenate strings from array elements:

```json
{
  "foreach": {
    "input_key": "chunks",
    "output_key": "formatted_results", 
    "max": 3,
    "append": "=== RESULT ===\n${this.text}\n====================\n\n"
  }
}
```

This processes each item in `response.chunks` and builds a single concatenated string in `formatted_results`.

## Webhook Function Testing

The CLI tool supports testing three types of webhook functions:

1. **Local Webhook Functions**: Traditional SWAIG functions handled by your local agent
2. **External Webhook Functions**: Functions that delegate to external HTTP services  
3. **DataMap Functions**: Server-side functions that don't require local webhook infrastructure

### External Webhook Functions

External webhook functions are automatically detected when a function has a `webhook_url` parameter and are tested by making HTTP requests to the external service:

```python
@AgentBase.tool(
    name="get_weather",
    description="Get weather from external service",
    parameters={"location": {"type": "string"}},
    webhook_url="https://weather-api.example.com/current"
)
def get_weather_external(self, args, raw_data):
    # This function body is never called for external webhooks
    pass
```

**Testing External Webhooks:**

```bash
# Test external webhook function
swaig-test examples/external_webhook_weather_agent.py getWeather '{"location":"Paris"}' --verbose

# Compare with local function  
swaig-test examples/external_webhook_weather_agent.py getHelp '{}' --verbose
```

**External Webhook Request Format:**

The CLI tool sends the same payload format that SignalWire uses:

```json
{
  "function": "getWeather",
  "argument": {
    "parsed": [{"location": "Paris"}],
    "raw": "{\"location\": \"Paris\"}"
  },
  "call_id": "test-call-uuid",
  "call": {
    "call_id": "test-call-uuid",
    "project_id": "project-uuid",
    "space_id": "space-uuid"
  },
  "vars": {
    "userVariables": {}
  }
}
```

**External Webhook Error Handling:**

```bash
# Test with unreachable external service
swaig-test examples/external_webhook_weather_agent.py testBrokenWebhook '{"message":"test"}' --verbose
```

Output shows connection errors and HTTP status codes:
```
✗ Could not connect to external webhook: HTTPSConnectionPool(host='nonexistent.example.com', port=443)
RESULT:
Dict: {
  "error": "Could not connect to external webhook: ...",
  "status_code": null
}
```

### Post Data Simulation Modes

#### 1. Default Mode (Minimal Data)
```bash
swaig-test my_agent.py my_function '{"param":"value"}'
```
**Includes**: `function`, `argument`, `call_id`, `meta_data`, `global_data`

#### 2. Comprehensive Mode (Full SignalWire Environment)
```bash
swaig-test my_agent.py my_function '{"param":"value"}' --fake-full-data
```

**Includes complete post_data with all SignalWire keys:**
- **Core identification**: `function`, `argument`, `call_id`, `call_session_id`, `node_id`
- **Metadata**: `meta_data_token`, `meta_data` (function-level shared data)
- **Global data**: `global_data` (agent configuration and state)
- **Conversation context**: `call_log`, `raw_call_log` (OpenAI conversation format)
- **SWML variables**: `prompt_vars` (includes SWML vars + global_data keys)
- **Permissions**: `swaig_allow_swml`, `swaig_post_conversation`, `swaig_post_swml_vars`
- **HTTP context**: `http_method`, `webhook_url`, `user_agent`, `request_headers`

#### 3. Custom Data Mode
```bash
swaig-test my_agent.py my_function '{"param":"value"}' --custom-data '{"call_id":"test-123","global_data":{"environment":"production"}}'
```

### Comprehensive Post Data Example

```json
{
  "function": "search_knowledge",
  "argument": {"query": "SignalWire"},
  "call_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "call_session_id": "session-uuid",
  "node_id": "test-node-001",
  "meta_data_token": "func_hash_token",
  "meta_data": {
    "test_mode": true,
    "function_name": "search_knowledge"
  },
  "global_data": {
    "app_name": "test_application",
    "environment": "test",
    "user_preferences": {"language": "en"}
  },
  "call_log": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant..."
    },
    {
      "role": "user",
      "content": "Please call the search_knowledge function"
    },
    {
      "role": "assistant",
      "content": "I'll call the search_knowledge function for you.",
      "tool_calls": [
        {
          "id": "call_12345678",
          "type": "function",
          "function": {
            "name": "search_knowledge",
            "arguments": "{\"query\":\"SignalWire\"}"
          }
        }
      ]
    }
  ],
  "raw_call_log": "... complete conversation history ...",
  "prompt_vars": {
    "ai_instructions": "You are a helpful assistant",
    "temperature": 0.7,
    "app_name": "test_application",
    "current_timestamp": "2024-01-15T10:30:00Z"
  },
  "swaig_allow_swml": true,
  "swaig_post_conversation": true,
  "swaig_post_swml_vars": true
}
```

### DataSphere Knowledge Search

```bash
# Test DataSphere serverless function
swaig-test examples/datasphere_serverless_env_demo.py search_knowledge '{"query":"AI agents"}' --verbose
```

**Expected Output:**
```
Executing DataMap function: search_knowledge
=== DataMap Function Execution ===

--- Processing Webhooks ---
Making POST request to: https://tony.signalwire.com/api/datasphere/documents/search
Response status: 200
Webhook 1 succeeded!

--- Processing Webhook Foreach ---
Found array data in response.chunks: 1 items
Processed 1 items

--- Processing Webhook Output ---
Set response = I found results for "AI agents":

=== RESULT ===
[Actual knowledge base content about AI agents...]

RESULT:
Response: I found results for "AI agents": ...
```

### Math Skill Function

```bash
# Test webhook-style math function
swaig-test examples/datasphere_serverless_env_demo.py calculate '{"expression":"25 * 47"}' --verbose
```

**Expected Output:**
```
Calling webhook function: calculate
Arguments: {"expression": "25 * 47"}
Function description: Perform mathematical calculations and return the result

RESULT:
SwaigFunctionResult: The result of 25 * 47 is 1175.
```

### DateTime Skill Function

```bash
# Test datetime function with comprehensive data
swaig-test examples/datasphere_serverless_env_demo.py get_datetime '{}' --fake-full-data
```

## Function Type Detection

The tool automatically detects function types:

- **DataMap Functions**: Stored as `dict` objects with `data_map` configuration
- **Webhook Functions**: Stored as `SWAIGFunction` objects with description and handler
- **Skill Functions**: Detected from loaded skills

**Detection Example:**
```bash
swaig-test my_agent.py --list-tools --verbose

Available SWAIG functions:
  search_knowledge - DataMap function (serverless)
    Config: {"webhooks": [...], "output": {...}}
  calculate - Perform mathematical calculations and return the result
    Function: <SWAIGFunction object>
```

## Command Line Options

### Core Options

| Option | Description |
|--------|-------------|
| `--exec FUNCTION` | Execute function with CLI-style arguments (recommended) |
| `--agent-class CLASS` | Specify agent class for multi-agent files |
| `--route ROUTE` | Select agent by route path (e.g., /matti-agent) |
| `--list-agents` | List all available agents in the file |
| `--list-tools` | List all available SWAIG functions and their types |
| `--verbose`, `-v` | Enable detailed execution tracing and debugging (also shows logs) |
| `--fake-full-data` | Generate comprehensive post_data with all SignalWire metadata |
| `--minimal` | Use minimal post_data (essential keys only) |
| `--custom-data` | JSON string with custom post_data overrides |

### SWML Testing Options

| Option | Description |
|--------|-------------|
| `--dump-swml` | Generate SWML document with fake call data |
| `--raw` | Output raw JSON only (no headers, pipeable) |
| `--call-type` | Call type: `sip` or `webrtc` (default: webrtc) |
| `--call-direction` | Call direction: `inbound` or `outbound` (default: inbound) |
| `--call-state` | Call state (default: created) |
| `--call-id` | Override call_id |
| `--project-id` | Override project_id |
| `--space-id` | Override space_id |
| `--from-number` | Override from address |
| `--to-extension` | Override to address |

### Data Override Options

| Option | Description |
|--------|-------------|
| `--user-vars` | JSON for vars.userVariables |
| `--query-params` | JSON for query parameters (merged into userVariables) |
| `--override` | Override values using dot notation (repeatable) |
| `--override-json` | Override with JSON values using dot notation (repeatable) |

### Mock Request Options

| Option | Description |
|--------|-------------|
| `--header` | Add HTTP headers for mock request (repeatable) |
| `--method` | HTTP method for mock request (default: POST) |
| `--body` | JSON string for mock request body |

### Alternative Syntax

| Option | Description |
|--------|-------------|
| `--args` | Separator for CLI-style function arguments |

## Real-World Examples

### Testing Joke Skill (DataMap)

```bash
# Test dad jokes with verbose output
swaig-test examples/joke_skill_demo.py --verbose --exec get_joke --type dadjokes

# Test regular jokes  
swaig-test examples/joke_skill_demo.py --exec get_joke --type jokes

# Test error handling with invalid type
swaig-test examples/joke_skill_demo.py --verbose --exec get_joke --type invalid
```

### Testing Multi-Agent Applications

```bash
# Discover available agents
swaig-test matti_and_sigmond/dual_agent_app.py

# Test MattiAgent functions
swaig-test matti_and_sigmond/dual_agent_app.py --agent-class MattiAgent --list-tools
swaig-test matti_and_sigmond/dual_agent_app.py --agent-class MattiAgent --exec get_weather --location "Tokyo"
swaig-test matti_and_sigmond/dual_agent_app.py --agent-class MattiAgent --exec transfer --name "support"

# Test SigmondAgent functions  
swaig-test matti_and_sigmond/dual_agent_app.py --agent-class SigmondAgent --list-tools
swaig-test matti_and_sigmond/dual_agent_app.py --agent-class SigmondAgent --exec search_knowledge --query "SignalWire"
swaig-test matti_and_sigmond/dual_agent_app.py --agent-class SigmondAgent --exec get_joke --type dadjokes

# Generate SWML for different agents
swaig-test matti_and_sigmond/dual_agent_app.py --agent-class MattiAgent --dump-swml
swaig-test matti_and_sigmond/dual_agent_app.py --agent-class SigmondAgent --dump-swml --raw | jq '.'
```

### Testing External Webhook Functions

```bash
# Test external webhook with verbose output
swaig-test examples/external_webhook_weather_agent.py --verbose --exec getWeather --location "San Francisco"

# List functions with their types (local vs external)
swaig-test examples/external_webhook_weather_agent.py --list-tools
```

### Advanced SWML Testing

```bash
# Test dynamic agent with custom headers and data
swaig-test examples/dynamic_agent.py --dump-swml \
  --header "Authorization=Bearer test-token" \
  --header "X-User-ID=12345" \
  --method POST \
  --body '{"source":"api","environment":"test"}' \
  --user-vars '{"customer_tier":"premium"}' \
  --verbose

# Test SIP vs WebRTC calls
swaig-test examples/agent.py --dump-swml --call-type sip --from-number "+15551234567"
swaig-test examples/agent.py --dump-swml --call-type webrtc --from-number "user@domain.com"

# Test with multi-agent file
swaig-test matti_and_sigmond/dual_agent_app.py --agent-class MattiAgent --dump-swml --call-type sip --verbose
```

### SWML Generation Examples

#### Basic Static Agent SWML

```bash
# Generate SWML for static agent
swaig-test examples/simple_agent.py --dump-swml
```

**Expected Output:**
```
Generating SWML document...
Agent: Simple Agent
Route: /swml

SWML Document:
==================================================
{"version":"1.0","sections":{"main":[{"ai":{"SWAIG":{"functions":[...]}}}]}}
==================================================
```

#### Dynamic Agent with Mock Request

```bash
# Test dynamic agent with custom headers and data
swaig-test examples/dynamic_agent.py --dump-swml \
  --header "Authorization=Bearer test-token" \
  --header "X-User-ID=12345" \
  --method POST \
  --body '{"source":"api","environment":"test"}' \
  --user-vars '{"customer_tier":"premium"}' \
  --verbose
```

**Expected Output:**
```
Generating SWML document...
Agent: Dynamic Agent
Route: /swml

Using fake SWML post_data:
{
  "call": {
    "call_id": "550e8400-e29b-41d4-a716-446655440000",
    ...
  },
  "vars": {
    "userVariables": {"customer_tier": "premium"}
  }
}

Mock request headers: {"authorization": "Bearer test-token", "x-user-id": "12345"}
Mock request method: POST
Dynamic agent modifications: {"ai_instructions": "Custom instructions for premium user"}

SWML Document:
==================================================
{"version":"1.0","sections":{"main":[{"ai":{"SWAIG":{"functions":[...]},"params":{"ai_instructions":"Custom instructions for premium user"}}}]}}
==================================================
```

#### Call Type Differentiation

```bash
# SIP call scenario
swaig-test examples/agent.py --dump-swml \
  --call-type sip \
  --call-direction outbound \
  --from-number "+15551234567" \
  --to-extension "+15559876543" \
  --verbose

# WebRTC call scenario  
swaig-test examples/agent.py --dump-swml \
  --call-type webrtc \
  --call-direction inbound \
  --from-number "customer@company.com" \
  --to-extension "support@myagent.com" \
  --header "Origin=https://company.com" \
  --verbose
```

#### Advanced Override Scenarios

```bash
# Complex call state testing
swaig-test examples/agent.py --dump-swml \
  --call-state answered \
  --override call.timeout=120 \
  --override call.max_duration=7200 \
  --override-json call.record='{"enabled":true,"format":"mp3"}' \
  --user-vars '{"call_reason":"support","priority":"high","customer_id":"CUST-12345"}' \
  --verbose

# Multi-environment testing
swaig-test examples/agent.py --dump-swml \
  --override call.project_id=prod-project-123 \
  --override call.space_id=prod-space-456 \
  --override-json vars.userVariables='{"environment":"production","region":"us-east-1","feature_flags":{"new_ui":true,"beta_features":false}}' \
  --override-json envs='{"DATABASE_URL":"prod-db","API_KEY":"prod-key"}'
```

### CLI Syntax Examples

#### DataMap Function with CLI Arguments

```bash
# Traditional JSON approach
swaig-test examples/datasphere_agent.py search_knowledge '{"query":"SignalWire features","count":"3","distance":"0.5"}'

# CLI syntax approach
swaig-test examples/datasphere_agent.py search_knowledge --args \
  --query "SignalWire features" \
  --count 3 \
  --distance 0.5
```

#### Math Function with Type Conversion

```bash
# CLI syntax with automatic type conversion
swaig-test examples/math_agent.py calculate --args \
  --expression "sqrt(144) + log(100)" \
  --precision 4 \
  --scientific-notation false
```

#### Complex Function with Mixed Types

```bash
# Function with string, number, boolean, and array parameters
swaig-test examples/complex_agent.py process_data --args \
  --input-text "Process this data" \
  --max-items 50 \
  --include-metadata \
  --categories "urgent,customer,support" \
  --confidence-threshold 0.85 \
  --async-processing false
```

## Advanced Usage

### SWML Testing Workflows

#### Testing Call Flow Scenarios

```bash
# Test inbound call flow
swaig-test examples/ivr_agent.py --dump-swml \
  --call-type sip \
  --call-direction inbound \
  --call-state created \
  --from-number "+15551234567" \
  --user-vars '{"caller_history":"first_time","language":"en"}' \
  --verbose

# Test transfer scenario
swaig-test examples/ivr_agent.py --dump-swml \
  --call-state answered \
  --override call.timeout=30 \
  --user-vars '{"transfer_reason":"escalation","agent_type":"supervisor"}' \
  --verbose

# Test callback scenario
swaig-test examples/callback_agent.py --dump-swml \
  --call-direction outbound \
  --override call.state=created \
  --user-vars '{"callback_scheduled":"2024-01-15T14:30:00Z","customer_id":"CUST-789"}' \
  --verbose
```

#### Testing Agent Variations

```bash
# Test with different project configurations
for project in test-proj staging-proj prod-proj; do
  echo "Testing project: $project"
  swaig-test examples/multi_tenant_agent.py --dump-swml \
    --project-id $project \
    --user-vars "{\"tenant\":\"$project\"}" \
    --raw | jq '.sections.main[0].ai.params.ai_instructions'
done

# Test with different user types
for tier in basic premium enterprise; do
  echo "Testing tier: $tier"
  swaig-test examples/tiered_agent.py --dump-swml \
    --user-vars "{\"customer_tier\":\"$tier\"}" \
    --verbose
done
```

#### Pipeline Testing with jq

```bash
# Extract specific SWML components
swaig-test examples/agent.py --dump-swml --raw | jq '.sections.main[0].ai.SWAIG.functions[].function'

# Test multiple agents and compare
for agent in examples/agent*.py; do
  echo "Agent: $agent"
  swaig-test $agent --dump-swml --raw | jq '.sections.main[0].ai.params.ai_instructions' 
done

# Validate SWML structure
swaig-test examples/agent.py --dump-swml --raw | jq 'has("version") and has("sections")'
```

### Mock Request Testing

#### Testing Request-Dependent Logic

```bash
# Test API key validation
swaig-test examples/api_agent.py --dump-swml \
  --header "Authorization=Bearer valid-token" \
  --body '{"api_version":"v2"}' \
  --verbose

# Test user authentication
swaig-test examples/auth_agent.py --dump-swml \
  --header "X-User-ID=user123" \
  --header "X-Session-Token=session456" \
  --query-params '{"authenticated":"true"}' \
  --verbose

# Test webhook validation
swaig-test examples/webhook_agent.py --dump-swml \
  --method POST \
  --header "X-Webhook-Signature=sha256=..." \
  --body '{"event":"call.created","data":{"call_id":"test"}}' \
  --verbose
```

#### Testing Different Request Patterns

```bash
# Test GET request handling
swaig-test examples/rest_agent.py --dump-swml \
  --method GET \
  --query-params '{"action":"get_config","format":"json"}' \
  --header "Accept=application/json"

# Test form data handling
swaig-test examples/form_agent.py --dump-swml \
  --method POST \
  --header "Content-Type=application/x-www-form-urlencoded" \
  --body '{"form_field":"value","submit":"true"}'

# Test file upload simulation
swaig-test examples/upload_agent.py --dump-swml \
  --method POST \
  --header "Content-Type=multipart/form-data" \
  --body '{"filename":"test.txt","content_type":"text/plain"}'
```

## Troubleshooting

### Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Agent Loading** | "No AgentBase instance found" | Ensure file has `agent` variable or AgentBase subclass |
| **Function Missing** | "Function 'X' not found" | Use `--list-tools` to verify function registration |
| **DataMap HTTP Error** | "Webhook request failed" | Check network connectivity and API credentials |
| **Template Expansion** | "MISSING:variable" in output | Verify template variable names match data structure |
| **JSON Parsing** | "Invalid JSON in args" | Check JSON syntax in function arguments |
| **SWML Generation** | "Error generating SWML" | Check agent initialization and SWML template syntax |
| **Dynamic Agent** | "Dynamic agent callback failed" | Verify on_swml_request method signature and mock request handling |
| **Override Syntax** | "Override path not found" | Use `--verbose` to see generated data structure and verify paths |
| **Wrong Argument Order** | CLI flags not working | Put all CLI flags BEFORE `--exec` |
| **Template Expansion** | "MISSING:variable" in output | Verify template variable names match data structure |
| **JSON Parsing** | "Invalid JSON in args" | Check JSON syntax or use `--exec` syntax |
| **Serverless URL Issues** | Wrong webhook URLs in SWML | Verify platform-specific configuration and environment variables |
| **Environment Conflicts** | Unexpected behavior in serverless mode | Clear conflicting environment variables or restart shell |
| **Platform Detection** | Wrong platform detected | Use `--simulate-serverless` explicitly instead of relying on auto-detection |

### Debug Strategies

1. **Use `--verbose`**: Shows complete execution flow including fake data generation
2. **Check function list**: Use `--list-tools --verbose` to see configurations  
3. **Test connectivity**: For DataMap functions, ensure API endpoints are reachable
4. **Validate JSON**: Use online JSON validators for complex arguments
5. **Check logs**: Agent initialization logs show skill loading status
6. **Test SWML incrementally**: Start with `--dump-swml` then add overrides gradually
7. **Verify mock requests**: Use `--verbose` to see mock request object details
8. **Pipeline with jq**: Use `--raw | jq '.'` to validate JSON structure

### SWML-Specific Debugging

For SWML generation issues:

```bash
# Check basic SWML generation
swaig-test my_agent.py --dump-swml --verbose

# Test with minimal overrides
swaig-test my_agent.py --dump-swml --override call.state=test --verbose

# Validate JSON structure
swaig-test my_agent.py --dump-swml --raw | python -m json.tool

# Check dynamic agent callback
swaig-test my_agent.py --dump-swml --header "test=value" --verbose
```

Look for:
- Fake post_data generation details
- Mock request object creation
- Dynamic agent callback execution  
- Override application order
- Final SWML document structure

### CLI Syntax Debugging

For `--args` parsing issues:

```bash
# Verify function schema
swaig-test my_agent.py --list-tools --verbose | grep -A 10 my_function

# Test with simple parameters first
swaig-test my_agent.py my_function --args --simple-param "value"

# Check type conversion
swaig-test my_agent.py my_function --args --number-param 42 --bool-param --verbose
```

Look for:
- Parameter schema definitions
- Type conversion results  
- Required vs optional parameters
- Parsed argument values

### Testing DataMap Error Handling

Test how DataMap functions handle API failures:

```bash
# Test with verbose output to see fallback processing
swaig-test my_agent.py my_datamap_func '{"input":"test"}' --verbose
```

If the primary webhook fails, you'll see:
```
Webhook 1 request failed: Connection timeout
--- Using DataMap Fallback Output ---
Fallback result = Sorry, the service is temporarily unavailable.
```

### Custom Environment Testing

Simulate different environments with custom data:

```bash
# Simulate production environment
swaig-test my_agent.py my_function '{"input":"test"}' --fake-full-data --custom-data '{
  "global_data": {
    "environment": "production", 
    "api_tier": "premium",
    "user_id": "prod-user-123"
  },
  "prompt_vars": {
    "ai_instructions": "You are a premium production assistant",
    "temperature": 0.3
  }
}'
```

### Testing Complex DataMap Configurations

For DataMap functions with multiple webhooks and complex foreach processing:

```bash
swaig-test my_agent.py complex_search '{"query":"test","filters":["type1","type2"]}' --verbose
```

This shows the complete processing pipeline:
- Template expansion in URLs and parameters
- Multiple webhook attempts with fallback
- Foreach processing of array responses
- Final output template expansion

### DataMap-Specific Debugging

For DataMap function issues:

```bash
# Enable verbose to see HTTP details
swaig-test my_agent.py --verbose --exec my_datamap_func --input test

# Check the complete configuration
swaig-test my_agent.py --list-tools --verbose | grep -A 20 my_datamap
```

Look for:
- Template expansion in request data
- HTTP response status and content
- Foreach processing details
- Output template expansion

## Integration with Development

### Pre-Deployment Testing

```bash
# Test all functions systematically
functions=$(swaig-test my_agent.py --list-tools | grep "  " | cut -d' ' -f3)
for func in $functions; do
    echo "Testing $func..."
    swaig-test my_agent.py $func '{"test":"data"}' --fake-full-data
done
```

### CI/CD Integration

The tool returns appropriate exit codes:
- `0`: Success
- `1`: Error (function failed, invalid arguments, network issues, etc.)

```yaml
# GitHub Actions example
- name: Test SWAIG Functions
  run: |
    swaig-test my_agent.py critical_function '{"input":"test"}' --fake-full-data
    if [ $? -ne 0 ]; then
      echo "Critical function test failed"
      exit 1
    fi
```

## Performance and Limitations

### Performance Considerations

- **DataMap HTTP Requests**: Real network latency applies
- **Large Responses**: Processing large API responses takes time
- **Verbose Output**: Can generate substantial debugging information
- **Memory Usage**: Comprehensive post_data mode uses more memory

### Current Limitations

1. **SignalWire Infrastructure**: Cannot perfectly replicate the serverless environment
2. **Network Dependencies**: DataMap testing requires internet connectivity
3. **Authentication**: Uses real API credentials (ensure proper security)
4. **State Isolation**: No persistence between separate test runs
5. **Concurrency**: Single-threaded execution only

### Best Practices

1. **Use minimal data mode** for basic function validation
2. **Enable verbose mode** when debugging issues
3. **Test DataMap functions** with real API credentials in secure environments
4. **Validate JSON arguments** before testing
5. **Check network connectivity** before testing DataMap functions

### Webhook Failure Detection

DataMap webhooks are considered failed when any of these conditions occur:

1. **HTTP Status Codes**: Status outside 200-299 range
2. **Explicit Error Keys**: `parse_error` or `protocol_error` in response
3. **Custom Error Keys**: Any keys specified in webhook `error_keys` configuration
4. **Network Errors**: Connection timeouts, DNS failures, etc.

When a webhook fails, the tool:
- Tries the next webhook in sequence (if any)
- Uses fallback output if all webhooks fail
- Provides detailed error information in verbose mode

## Troubleshooting

### Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Multiple Agents** | "Multiple agents found" | Use `--agent-class ClassName` to specify which agent |
| **Agent Loading** | "No AgentBase instance found" | Ensure file has agent instance or AgentBase subclass |
| **Function Missing** | "Function 'X' not found" | Use `--list-tools` to verify function registration |
| **DataMap HTTP Error** | "Webhook request failed" | Check network connectivity and API credentials |
| **Wrong Argument Order** | CLI flags not working | Put all CLI flags BEFORE `--exec` |
| **Template Expansion** | "MISSING:variable" in output | Verify template variable names match data structure |
| **JSON Parsing** | "Invalid JSON in args" | Check JSON syntax or use `--exec` syntax |
| **Serverless URL Issues** | Wrong webhook URLs in SWML | Verify platform-specific configuration and environment variables |
| **Environment Conflicts** | Unexpected behavior in serverless mode | Clear conflicting environment variables or restart shell |
| **Platform Detection** | Wrong platform detected | Use `--simulate-serverless` explicitly instead of relying on auto-detection |

### Debug Strategies

1. **Use `--verbose`**: Shows complete execution flow including agent selection and fake data generation
2. **Check agent discovery**: Use `--list-agents` to see available agents
3. **Check function list**: Use `--list-tools --verbose` to see configurations
4. **Test connectivity**: For DataMap functions, ensure API endpoints are reachable
5. **Check argument order**: CLI flags before `--exec`, function args after function name
6. **Validate syntax**: Use `--exec` syntax to avoid JSON parsing issues

### Serverless Debugging

#### Environment Variable Issues

```bash
# Debug environment variable setup
swaig-test my_agent.py --simulate-serverless lambda --verbose --exec get_status

# Check what environment variables are set
swaig-test my_agent.py --simulate-serverless lambda --env DEBUG=1 --exec debug_env

# Test environment file loading
swaig-test my_agent.py --simulate-serverless lambda --env-file my.env --verbose --dump-swml

# Clear conflicting variables
unset AWS_LAMBDA_FUNCTION_NAME GOOGLE_CLOUD_PROJECT
swaig-test my_agent.py --simulate-serverless cloud_function --verbose --dump-swml
```

#### Platform-Specific Debugging

```bash
# Debug Lambda configuration
swaig-test my_agent.py --simulate-serverless lambda \
  --aws-function-name my-function \
  --aws-region us-west-2 \
  --verbose --dump-swml

# Debug CGI configuration  
swaig-test my_agent.py --simulate-serverless cgi \
  --cgi-host example.com \
  --cgi-https \
  --verbose --dump-swml

# Debug webhook URL generation
swaig-test my_agent.py --simulate-serverless lambda \
  --dump-swml --format-json | jq '.sections.main[1].ai.SWAIG.defaults.web_hook_url'
```

#### Function Execution Debugging

```bash
# Debug function execution in serverless context
swaig-test my_agent.py --simulate-serverless lambda \
  --verbose \
  --exec my_function --param value \
  --full-request

# Compare responses across platforms
swaig-test my_agent.py --exec my_function --param value > local.json
swaig-test my_agent.py --simulate-serverless lambda --exec my_function --param value > lambda.json
diff local.json lambda.json
```

### Agent Discovery Debugging

```bash
# Debug agent discovery
swaig-test my_file.py --list-agents --verbose

# Check if agent is auto-selected
swaig-test my_file.py --verbose --exec my_function --param value

# Explicitly specify agent
swaig-test my_file.py --agent-class MyAgent --verbose --exec my_function --param value
```

### DataMap Debugging

```bash
# Enable verbose to see complete DataMap processing
swaig-test my_agent.py --verbose --exec my_datamap_func --input test

# Check URL template expansion
swaig-test my_agent.py --verbose --exec my_func --location "New York"
```

Look for:
- URL template expansion details
- HTTP request/response information
- Fallback processing when APIs fail
- Output template processing

### Joke Agent Examples

#### Working Joke API (Success Case)

```bash
# Test with valid API key - shows successful DataMap processing
API_NINJAS_KEY=your_api_key swaig-test examples/joke_skill_demo.py get_joke '{"type": "jokes"}' --verbose
```

**Expected Output:**
```
=== DataMap Function Execution ===
--- Processing Webhooks ---
Making GET request to: https://api.api-ninjas.com/v1/jokes
Response status: 200
Webhook 1 succeeded!
Array response: 1 items

--- Processing Webhook Output ---
Set response = Here's a joke: What do you call a bear with no teeth? A gummy bear!

RESULT:
Response: Here's a joke: What do you call a bear with no teeth? A gummy bear!
```

#### Invalid API Key (Failure Case)

```bash
# Test with invalid API key - shows fallback output processing
swaig-test examples/joke_agent.py get_joke '{"type": "jokes"}' --verbose
```

**Expected Output (when API key is invalid):**
```
=== DataMap Function Execution ===
--- Processing Webhooks ---
Making GET request to: https://api.api-ninjas.com/v1/jokes
Response status: 400
Response data: {"error": "Invalid API Key."}
Webhook failed: HTTP status 400 outside 200-299 range
Webhook 1 failed, trying next webhook...

--- Using DataMap Fallback Output ---
Fallback result = Tell the user that the joke service is not working right now and just make up a joke on your own

RESULT:
Response: Tell the user that the joke service is not working right now and just make up a joke on your own
```

This demonstrates both:
- **Successful webhook processing** with array response handling
- **Failure detection and fallback** when APIs return errors 

## Best Practices

### Development Workflow

1. **Start with discovery**: `swaig-test my_agent.py` to see available agents
2. **List functions**: `swaig-test my_agent.py --list-tools` to see available functions
3. **Test functions**: Use `--exec` syntax for cleaner testing
4. **Test SWML**: Use `--dump-swml` to verify agent configuration
5. **Use verbose mode**: Enable `--verbose` when debugging issues

### Multi-Agent Files

1. **Always use `--list-agents` first** to see what's available
2. **Use `--agent-class` consistently** for multi-agent files
3. **Test each agent separately** to isolate issues
4. **Use descriptive agent names** to make selection easier

### DataMap Functions

1. **Test with verbose mode** to see complete processing pipeline
2. **Verify API credentials** before testing DataMap functions
3. **Test error handling** with invalid parameters
4. **Check network connectivity** for external APIs

### CLI Syntax

1. **Prefer `--exec` syntax** for development
2. **Put CLI flags before `--exec`** for correct parsing
3. **Use `--verbose`** to see argument parsing results
4. **Use JSON syntax** when needed for complex argument structures

### Serverless Testing

1. **Test locally first** before using serverless simulation
2. **Use environment files** for consistent platform configuration
3. **Test all target platforms** to ensure compatibility
4. **Verify webhook URLs** are generated correctly for each platform
5. **Clear environment variables** between platform tests to avoid conflicts
6. **Use `--verbose`** to debug environment setup and URL generation

---

## sw-search - Build and Query Search Indexes

Build local search indexes from document collections for use with the native vector search skill.

```bash
sw-search <source_dir> [options]
```

### Building Indexes

**Arguments:**
- `source_dir` - Directory containing documents to index

**Options:**
- `--output FILE` - Output .swsearch file (default: `<source_dir>.swsearch`)
- `--chunk-size SIZE` - Chunk size in words (default: 50)
- `--chunk-overlap SIZE` - Overlap between chunks in words (default: 10)
- `--file-types TYPES` - Comma-separated file extensions (default: md,txt,rst)
- `--exclude PATTERNS` - Comma-separated glob patterns to exclude
- `--model MODEL` - Embedding model name (default: sentence-transformers/all-mpnet-base-v2)
- `--tags TAGS` - Comma-separated tags to add to all chunks
- `--verbose` - Show detailed progress information
- `--validate` - Validate the created index after building
- `--chunking-strategy STRATEGY` - Chunking strategy: sentence, sliding, paragraph, page, semantic, topic, qa (default: sentence)
- `--max-sentences-per-chunk NUM` - Maximum sentences per chunk (default: 3)
- `--semantic-threshold FLOAT` - Threshold for semantic chunking (default: 0.5)
- `--topic-threshold FLOAT` - Threshold for topic-based chunking (default: 0.3)
- `--index-nlp-backend BACKEND` - NLP backend for processing (default: basic)
- `--split-newlines` - Split on newlines in addition to sentence boundaries
- `--languages LANGS` - Comma-separated language codes (default: en)

### Validating Indexes

```bash
sw-search validate <index_file> [--verbose]
```

### Searching Indexes

```bash
sw-search search <index_file> <query> [options]
```

**Options:**
- `--count COUNT` - Number of results to return (default: 5)
- `--distance-threshold FLOAT` - Minimum similarity score (default: 0.0)
- `--tags TAGS` - Comma-separated tags to filter by
- `--verbose` - Show detailed information
- `--json` - Output results as JSON
- `--no-content` - Hide content in results (show only metadata)
- `--query-nlp-backend BACKEND` - NLP backend for query processing

### Remote Search

```bash
sw-search remote <endpoint> <query> [options]
```

**Options:**
- `--index-name NAME` - Name of the index to search (required)
- `--count COUNT` - Number of results to return (default: 5)
- `--distance-threshold FLOAT` - Minimum similarity score (default: 0.0)
- `--tags TAGS` - Comma-separated tags to filter by
- `--json` - Output results as JSON
- `--timeout SECONDS` - Request timeout in seconds (default: 30)

### Examples

```bash
# Build from a directory
sw-search docs --output docs.swsearch

# Build from multiple sources
sw-search docs examples README.md --output comprehensive.swsearch

# Validate an index
sw-search validate docs.swsearch

# Search within an index
sw-search search docs.swsearch "how to create an agent"
sw-search search docs.swsearch "API reference" --count 3 --verbose

# Search remote index via API
sw-search remote http://localhost:8001/search "query" --index-name docs
```

For complete documentation on the search system, see [Search Overview](search_overview.md).

---

## Installation

All CLI tools are included when you install the SignalWire Agents SDK:

```bash
pip install signalwire-agents

# For search functionality
pip install signalwire-agents[search]
```

## Getting Help

```bash
sw-search --help
swaig-test --help
```
