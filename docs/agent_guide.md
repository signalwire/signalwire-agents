# SignalWire AI Agent Guide

## Table of Contents
- [Introduction](#introduction)
- [Architecture Overview](#architecture-overview)
- [Creating an Agent](#creating-an-agent)
- [Prompt Building](#prompt-building)
- [SWAIG Functions](#swaig-functions)
- [Skills System](#skills-system)
- [Multilingual Support](#multilingual-support)
- [Agent Configuration](#agent-configuration)
- [Dynamic Agent Configuration](#dynamic-agent-configuration)
  - [Overview](#overview)
  - [Setting Up Dynamic Configuration](#setting-up-dynamic-configuration)
  - [Dynamic Configuration Methods](#dynamic-configuration-methods)
  - [Request Data Access](#request-data-access)
  - [Configuration Examples](#configuration-examples)
  - [Use Cases](#use-cases)
  - [Migration Guide](#migration-guide)
  - [Best Practices](#best-practices)
- [Advanced Features](#advanced-features)
  - [State Management](#state-management)
  - [SIP Routing](#sip-routing)
  - [Custom Routing](#custom-routing)
- [Prefab Agents](#prefab-agents)
- [API Reference](#api-reference)
- [Examples](#examples)

## Introduction

The `AgentBase` class provides the foundation for creating AI-powered agents using the SignalWire AI Agent SDK. It extends the `SWMLService` class, inheriting all its SWML document creation and serving capabilities, while adding AI-specific functionality.

Key features of `AgentBase` include:

- Structured prompt building with POM (Prompt Object Model)
- SWAIG function definitions for AI tool access
- Multilingual support
- Agent configuration (hint handling, pronunciation rules, etc.)
- State management for conversations

This guide explains how to create and customize your own AI agents, with examples based on the SDK's sample implementations.

## Architecture Overview

The Agent SDK architecture consists of several layers:

1. **SWMLService**: The base layer for SWML document creation and serving
2. **AgentBase**: Extends SWMLService with AI agent functionality
3. **Custom Agents**: Your specific agent implementations that extend AgentBase

Here's how these components relate to each other:

```
┌─────────────┐
│ Your Agent  │ (Extends AgentBase with your specific functionality)
└─────▲───────┘
      │
┌─────┴───────┐
│  AgentBase  │ (Adds AI functionality to SWMLService)
└─────▲───────┘
      │
┌─────┴───────┐
│ SWMLService │ (Provides SWML document creation and web service)
└─────────────┘
```

## Creating an Agent

To create an agent, extend the `AgentBase` class and define your agent's behavior:

```python
from signalwire_agents import AgentBase

class MyAgent(AgentBase):
    def __init__(self):
        super().__init__(
            name="my-agent",
            route="/agent",
            host="0.0.0.0",
            port=3000,
            use_pom=True  # Enable Prompt Object Model
        )
        
        # Define agent personality and behavior
        self.prompt_add_section("Personality", body="You are a helpful and friendly assistant.")
        self.prompt_add_section("Goal", body="Help users with their questions and tasks.")
        self.prompt_add_section("Instructions", bullets=[
            "Answer questions clearly and concisely",
            "If you don't know, say so",
            "Use the provided tools when appropriate"
        ])
        
        # Add a post-prompt for summary
        self.set_post_prompt("Please summarize the key points of this conversation.")
```

## Running Your Agent

The SignalWire AI Agent SDK provides a `run()` method that automatically detects the execution environment and configures the agent appropriately. This method works across all deployment modes:

### Deployment with `run()`

```python
def main():
    agent = MyAgent()
    
    print("Starting agent server...")
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    agent.run()  # Auto-detects environment

if __name__ == "__main__":
    main()
```

The `run()` method automatically detects and configures for:

- **HTTP Server**: When run directly, starts an HTTP server
- **CGI**: When CGI environment variables are detected, operates in CGI mode  
- **AWS Lambda**: When Lambda environment is detected, configures for serverless execution

### Deployment Modes

#### HTTP Server Mode
When run directly (e.g., `python my_agent.py`), the agent starts an HTTP server:

```python
# Automatically starts HTTP server when run directly
agent.run()
```

#### CGI Mode  
When CGI environment variables are present, operates in CGI mode with clean HTTP output:

```python
# Same code - automatically detects CGI environment
agent.run()
```

#### AWS Lambda Mode
When AWS Lambda environment is detected, configures for serverless execution:

```python
# Same code - automatically detects Lambda environment  
agent.run()
```

### Environment Detection

The SDK automatically detects the execution environment:

| Environment | Detection Method | Behavior |
|-------------|------------------|----------|
| **HTTP Server** | Default when no serverless environment detected | Starts FastAPI server on specified host/port |
| **CGI** | `GATEWAY_INTERFACE` environment variable present | Processes single CGI request and exits |
| **AWS Lambda** | `AWS_LAMBDA_FUNCTION_NAME` environment variable | Handles Lambda event/context |
| **Google Cloud** | `FUNCTION_NAME` or `K_SERVICE` variables | Processes Cloud Function request |
| **Azure Functions** | `AZURE_FUNCTIONS_*` variables | Handles Azure Function request |

### Logging Configuration

The SDK includes a central logging system that automatically configures based on the deployment environment:

```python
# Logging is automatically configured based on environment
# No manual setup required in most cases

# Optional: Override logging mode via environment variable
# SIGNALWIRE_LOG_MODE=off      # Disable all logging
# SIGNALWIRE_LOG_MODE=stderr   # Log to stderr
# SIGNALWIRE_LOG_MODE=default  # Use default logging
# SIGNALWIRE_LOG_MODE=auto     # Auto-detect (default)
```

The logging system automatically:
- **CGI Mode**: Sets logging to 'off' to avoid interfering with HTTP headers
- **Lambda Mode**: Configures appropriate logging for serverless environment
- **Server Mode**: Uses structured logging with timestamps and levels
- **Debug Mode**: Enhanced logging when debug flags are set

## Prompt Building

There are several ways to build prompts for your agent:

### 1. Using Prompt Sections (POM)

The Prompt Object Model (POM) provides a structured way to build prompts:

```python
# Add a section with just body text
self.prompt_add_section("Personality", body="You are a friendly assistant.")

# Add a section with bullet points
self.prompt_add_section("Instructions", bullets=[
    "Answer questions clearly",
    "Be helpful and polite",
    "Use functions when appropriate"
])

# Add a section with both body and bullets
self.prompt_add_section("Context", 
                       body="The user is calling about technical support.",
                       bullets=["They may need help with their account", 
                               "Check for existing tickets"])
```

For convenience, the SDK also provides wrapper methods that some users may prefer:

```python
# Convenience methods
self.setPersonality("You are a friendly assistant.") 
self.setGoal("Help users with their questions.")
self.setInstructions([
    "Answer questions clearly",
    "Be helpful and polite"
])
```

These convenience methods call `prompt_add_section()` internally with the appropriate section titles.

### 2. Using Raw Text Prompts

For simpler agents, you can set the prompt directly as text:

```python
self.set_prompt_text("""
You are a helpful assistant. Your goal is to provide clear and concise information
to the user. Answer their questions to the best of your ability.
""")
```

### 3. Setting a Post-Prompt

The post-prompt is sent to the AI after the conversation for summary or analysis:

```python
self.set_post_prompt("""
Analyze the conversation and extract:
1. Main topics discussed
2. Action items or follow-ups needed
3. Whether the user's questions were answered satisfactorily
""")
```

## SWAIG Functions

SWAIG functions allow the AI agent to perform actions and access external systems. There are two types of SWAIG functions you can define:

### 1. Local Webhook Functions (Standard)

These are the traditional SWAIG functions that are handled locally by your agent:

```python
from signalwire_agents.core.function_result import SwaigFunctionResult

@AgentBase.tool(
    name="get_weather",
    description="Get the current weather for a location",
    parameters={
        "location": {
            "type": "string",
            "description": "The city or location to get weather for"
        }
    },
    secure=True  # Optional, defaults to True
)
def get_weather(self, args, raw_data):
    # Extract the location parameter
    location = args.get("location", "Unknown location")
    
    # Here you would typically call a weather API
    # For this example, we'll return mock data
    weather_data = f"It's sunny and 72°F in {location}."
    
    # Return a SwaigFunctionResult
    return SwaigFunctionResult(weather_data)
```

### 2. External Webhook Functions

External webhook functions allow you to delegate function execution to external services instead of handling them locally. This is useful when you want to:
- Use existing web services or APIs directly
- Distribute function processing across multiple servers
- Integrate with third-party systems that provide their own endpoints

To create an external webhook function, add a `webhook_url` parameter to the decorator:

```python
@AgentBase.tool(
    name="get_weather_external",
    description="Get weather from external service",
    parameters={
        "location": {
            "type": "string",
            "description": "The city or location to get weather for"
        }
    },
    webhook_url="https://your-service.com/weather-endpoint"
)
def get_weather_external(self, args, raw_data):
    # This function will never be called locally when webhook_url is provided
    # The external service at webhook_url will receive the function call instead
    return SwaigFunctionResult("This should not be reached for external webhooks")
```

#### How External Webhooks Work

When you specify a `webhook_url`:

1. **Function Registration**: The function is registered with your agent as usual
2. **SWML Generation**: The generated SWML includes the external webhook URL instead of your local endpoint
3. **SignalWire Processing**: When the AI calls the function, SignalWire makes an HTTP POST request directly to your external URL
4. **Payload Format**: The external service receives a JSON payload with the function call data:

```json
{
    "function": "get_weather_external",
    "argument": {
        "parsed": [{"location": "New York"}],
        "raw": "{\"location\": \"New York\"}"
    },
    "call_id": "abc123-def456-ghi789",
    "call": { /* call information */ },
    "vars": { /* call variables */ }
}
```

5. **Response Handling**: Your external service should return a JSON response that SignalWire will process.

#### Mixing Local and External Functions

You can mix both types of functions in the same agent:

```python
class HybridAgent(AgentBase):
    def __init__(self):
        super().__init__(name="hybrid-agent", route="/hybrid")
    
    # Local function - handled by this agent
    @AgentBase.tool(
        name="get_help",
        description="Get help information",
        parameters={}
    )
    def get_help(self, args, raw_data):
        return SwaigFunctionResult("I can help you with weather and news!")
    
    # External function - handled by external service
    @AgentBase.tool(
        name="get_weather",
        description="Get current weather",
        parameters={
            "location": {"type": "string", "description": "City name"}
        },
        webhook_url="https://weather-service.com/api/weather"
    )
    def get_weather_external(self, args, raw_data):
        # This won't be called for external webhooks
        pass
    
    # Another external function - different service
    @AgentBase.tool(
        name="get_news",
        description="Get latest news",
        parameters={
            "topic": {"type": "string", "description": "News topic"}
        },
        webhook_url="https://news-service.com/api/news"
    )
    def get_news_external(self, args, raw_data):
        # This won't be called for external webhooks
        pass
```

#### Testing External Webhooks

You can test external webhook functions using the CLI tool:

```bash
# Test local function
swaig-test examples/my_agent.py --exec get_help

# Test external webhook function
swaig-test examples/my_agent.py --verbose --exec get_weather --location "New York"

# List all functions with their types
swaig-test examples/my_agent.py --list-tools
```

The CLI tool will automatically detect external webhook functions and make HTTP requests to the external services, simulating what SignalWire does in production.

### 3. Type-Hinted Functions

Instead of writing JSON Schema by hand, you can use Python type hints and the SDK will infer the schema automatically:

```python
from typing import Optional, Literal

@AgentBase.tool(name="get_weather")
def get_weather(self, city: str, units: Literal["celsius", "fahrenheit"] = "celsius"):
    """Get the weather forecast.

    Args:
        city: Name of the city to look up
        units: Temperature units to use
    """
    return SwaigFunctionResult(f"It's sunny in {city} (showing {units})")
```

The SDK automatically:
- Infers parameter types from type hints (`str` → `"string"`, `int` → `"integer"`, etc.)
- Marks parameters without defaults as required
- Extracts the tool description from the docstring's first line
- Extracts per-parameter descriptions from the `Args:` block
- Handles `Optional[X]` as a non-required parameter
- Converts `Literal["a", "b"]` to `enum` values

**Supported types:**

| Python Type | JSON Schema Type |
|---|---|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| `list` / `List[X]` | `"array"` (with `items` if parameterized) |
| `dict` | `"object"` |
| `Literal["a", "b"]` | `"string"` with `enum` |
| `Optional[X]` | type of `X`, not required |

**Rules:**
- If `parameters=` is provided explicitly, it always takes precedence over inference
- The `raw_data` parameter is special: if included in the signature, it receives the raw request data but is excluded from the schema
- Old-style `(self, args, raw_data)` handlers continue to work exactly as before

**Accessing raw_data in typed handlers:**

```python
@AgentBase.tool(name="check_call")
def check_call(self, query: str, raw_data: dict = None):
    """Check the current call."""
    call_id = raw_data.get("call_id", "unknown") if raw_data else "unknown"
    return SwaigFunctionResult(f"Call {call_id}: query={query}")
```

### Function Parameters

The parameters for a SWAIG function are defined using JSON Schema:

```python
parameters={
    "parameter_name": {
        "type": "string", # Can be string, number, integer, boolean, array, object
        "description": "Description of the parameter",
        # Optional attributes:
        "enum": ["option1", "option2"],  # For enumerated values
        "minimum": 0,  # For numeric types
        "maximum": 100,  # For numeric types
        "pattern": "^[A-Z]+$"  # For string validation
    }
}
```

### Function Results

To return results from a SWAIG function, use the `SwaigFunctionResult` class:

```python
# Basic result with just text
return SwaigFunctionResult("Here's the result")

# Result with a single action
return SwaigFunctionResult("Here's the result with an action")
       .add_action("say", "I found the information you requested.")

# Result with multiple actions using add_actions
return SwaigFunctionResult("Multiple actions example")
       .add_actions([
           {"playback_bg": {"file": "https://example.com/music.mp3"}},
           {"set_global_data": {"key": "value"}}
       ])

# Alternative way to add multiple actions sequentially
return (
    SwaigFunctionResult("Sequential actions example")
    .add_action("say", "I found the information you requested.")
    .add_action("playback_bg", {"file": "https://example.com/music.mp3"})
)
```

In the examples above:
- `add_action(name, data)` adds a single action with the given name and data
- `add_actions(actions)` adds multiple actions at once from a list of action objects

### Native Functions

The agent can use SignalWire's built-in functions:

```python
# Enable native functions
self.set_native_functions([
    "check_time",
    "wait_seconds"
])
```

### Function Includes

You can include functions from remote sources:

```python
# Include remote functions
self.add_function_include(
    url="https://api.example.com/functions",
    functions=["get_weather", "get_news"],
    meta_data={"session_id": "unique-session-123"}  # Use for session tracking, NOT credentials
)
```

### SWAIG Function Security

The SDK implements an automated security mechanism for SWAIG functions to ensure that only authorized calls can be made to your functions. This is important because SWAIG functions often provide access to sensitive operations or data.

#### Token-Based Security

By default, all SWAIG functions are marked as `secure=True`, which enables token-based security:

```python
@agent.tool(
    name="get_account_details",
    description="Get customer account details",
    parameters={"account_id": {"type": "string"}},
    secure=True  # This is the default, can be omitted
)
def get_account_details(self, args, raw_data):
    # Implementation
```

When a function is marked as secure:

1. The SDK automatically generates a secure token for each function when rendering the SWML document
2. The token is added to the function's URL as a query parameter: `?token=X2FiY2RlZmcuZ2V0X3RpbWUuMTcxOTMxNDI1...`
3. When the function is called, the token is validated before executing the function

These security tokens have important properties:
- **Completely stateless**: The system doesn't need to store tokens or track sessions
- **Self-contained**: Each token contains all information needed for validation
- **Function-specific**: A token for one function can't be used for another
- **Session-bound**: Tokens are tied to a specific call/session ID
- **Time-limited**: Tokens expire after a configurable duration (default: 60 minutes)
- **Cryptographically signed**: Tokens can't be tampered with or forged

This stateless design provides several benefits:
- **Server resilience**: Tokens remain valid even if the server restarts
- **No memory consumption**: No need to track sessions or store tokens in memory
- **High scalability**: Multiple servers can validate tokens without shared state
- **Load balancing**: Requests can be distributed across multiple servers freely

The token system secures both SWAIG functions and post-prompt endpoints:
- SWAIG function calls for interactive AI capabilities
- Post-prompt requests for receiving conversation summaries

You can disable token security for specific functions when appropriate:

```python
@agent.tool(
    name="get_public_information",
    description="Get public information that doesn't require security",
    parameters={},
    secure=False  # Disable token security for this function
)
def get_public_information(self, args, raw_data):
    # Implementation
```

#### Token Expiration

The default token expiration is 60 minutes (3600 seconds), but you can configure this when initializing your agent:

```python
agent = MyAgent(
    name="my_agent",
    token_expiry_secs=1800  # Set token expiration to 30 minutes
)
```

The expiration timer resets each time a function is successfully called, so as long as there is activity at least once within the expiration period, the tokens will remain valid throughout the entire conversation.

#### Custom Token Validation

You can override the default token validation by implementing your own `validate_tool_token` method in your custom agent class.

## Skills System

The Skills System allows you to extend your agents with powerful capabilities using simple one-liner calls. Skills are modular, reusable components that can be easily added to any agent and configured with parameters.

### Quick Start

```python
from signalwire_agents import AgentBase

class SkillfulAgent(AgentBase):
    def __init__(self):
        super().__init__(name="skillful-agent", route="/skillful")
        
        # Add skills with one-liners
        self.add_skill("web_search")    # Web search capability
        self.add_skill("datetime")      # Current date/time info
        self.add_skill("math")          # Mathematical calculations
        
        # Configure skills with parameters
        self.add_skill("web_search", {
            "num_results": 3,  # Get 3 search results instead of default 1
            "delay": 0.5       # Add delay between requests
        })
```

### Available Built-in Skills

#### Web Search Skill (`web_search`)
Provides web search capabilities using Google Custom Search API with web scraping.

**Requirements:**
- Packages: `beautifulsoup4`, `requests`

**Parameters:**
- `api_key` (required): Google Custom Search API key
- `search_engine_id` (required): Google Custom Search Engine ID
- `num_results` (default: 1): Number of search results to return
- `delay` (default: 0): Delay in seconds between requests
- `tool_name` (default: "web_search"): Custom name for the search tool
- `no_results_message` (default: "I couldn't find any results for '{query}'. This might be due to a very specific query or temporary issues. Try rephrasing your search or asking about a different topic."): Custom message to return when no search results are found. Use `{query}` as a placeholder for the search query.

**Multiple Instance Support:**
The web_search skill supports multiple instances with different search engines and tool names, allowing you to search different data sources:

**Example:**
```python
# Basic single instance
agent.add_skill("web_search", {
    "api_key": "your-google-api-key",
    "search_engine_id": "your-search-engine-id"
})
# Creates tool: web_search

# Fast single result (previous default)
agent.add_skill("web_search", {
    "api_key": "your-google-api-key",
    "search_engine_id": "your-search-engine-id",
    "num_results": 1,
    "delay": 0
})

# Multiple results with delay
agent.add_skill("web_search", {
    "api_key": "your-google-api-key",
    "search_engine_id": "your-search-engine-id",
    "num_results": 5,
    "delay": 1.0
})

# Multiple instances with different search engines
agent.add_skill("web_search", {
    "api_key": "your-google-api-key",
    "search_engine_id": "general-search-engine-id",
    "tool_name": "search_general",
    "num_results": 1
})
# Creates tool: search_general

agent.add_skill("web_search", {
    "api_key": "your-google-api-key",
    "search_engine_id": "news-search-engine-id",
    "tool_name": "search_news",
    "num_results": 3,
    "delay": 0.5
})
# Creates tool: search_news

# Custom no results message
agent.add_skill("web_search", {
    "api_key": "your-google-api-key",
    "search_engine_id": "your-search-engine-id",
    "no_results_message": "Sorry, I couldn't find information about '{query}'. Please try a different search term."
})
```

#### DateTime Skill (`datetime`)
Provides current date and time information with timezone support.

**Requirements:**
- Packages: `pytz`

**Tools Added:**
- `get_current_time`: Get current time with optional timezone
- `get_current_date`: Get current date with optional timezone

**Example:**
```python
agent.add_skill("datetime")
# Agent can now tell users the current time and date
```

#### Math Skill (`math`)
Provides safe mathematical expression evaluation.

**Requirements:**
- None (uses built-in Python functionality)

**Tools Added:**
- `calculate`: Evaluate mathematical expressions safely

**Example:**
```python
agent.add_skill("math")
# Agent can now perform calculations like "2 + 3 * 4"
```

#### DataSphere Skill (`datasphere`)
Provides knowledge search capabilities using SignalWire DataSphere RAG stack.

**Requirements:**
- Packages: `requests`

**Parameters:**
- `space_name` (required): SignalWire space name
- `project_id` (required): SignalWire project ID 
- `token` (required): SignalWire authentication token
- `document_id` (required): DataSphere document ID to search
- `count` (default: 1): Number of search results to return
- `distance` (default: 3.0): Distance threshold for search matching
- `tags` (optional): List of tags to filter search results
- `language` (optional): Language code to limit search
- `pos_to_expand` (optional): List of parts of speech for synonym expansion (e.g., ["NOUN", "VERB"])
- `max_synonyms` (optional): Maximum number of synonyms to use for each word
- `tool_name` (default: "search_knowledge"): Custom name for the search tool
- `no_results_message` (default: "I couldn't find any relevant information for '{query}' in the knowledge base. Try rephrasing your question or asking about a different topic."): Custom message when no results found

**Multiple Instance Support:**
The DataSphere skill supports multiple instances with different tool names, allowing you to search multiple knowledge bases:

**Example:**
```python
# Basic single instance
agent.add_skill("datasphere", {
    "space_name": "my-space",
    "project_id": "my-project",
    "token": "my-token",
    "document_id": "general-knowledge"
})
# Creates tool: search_knowledge

# Multiple instances for different knowledge bases
agent.add_skill("datasphere", {
    "space_name": "my-space",
    "project_id": "my-project", 
    "token": "my-token",
    "document_id": "product-docs",
    "tool_name": "search_products",
    "tags": ["Products", "Features"],
    "count": 3
})
# Creates tool: search_products

agent.add_skill("datasphere", {
    "space_name": "my-space",
    "project_id": "my-project",
    "token": "my-token", 
    "document_id": "support-kb",
    "tool_name": "search_support",
    "no_results_message": "I couldn't find support information about '{query}'. Try contacting our support team.",
    "distance": 5.0
})
# Creates tool: search_support
```

#### Native Vector Search Skill (`native_vector_search`)
Provides local document search capabilities using vector similarity and keyword search. This skill works entirely offline with local `.swsearch` index files or can connect to remote search servers.

**Requirements:**
- Packages: `sentence-transformers`, `scikit-learn`, `numpy` (install with `pip install signalwire-agents[search]`)

**Parameters:**
- `tool_name` (default: "search_knowledge"): Custom name for the search tool
- `description` (default: "Search the local knowledge base for information"): Tool description
- `index_file` (optional): Path to local `.swsearch` index file
- `remote_url` (optional): URL of remote search server (e.g., "http://localhost:8001")
- `index_name` (default: "default"): Index name on remote server (for remote mode)
- `build_index` (default: False): Auto-build index if missing
- `source_dir` (optional): Source directory for auto-building index
- `file_types` (default: ["md", "txt"]): File types to include when building index
- `count` (default: 3): Number of search results to return
- `distance_threshold` (default: 0.0): Minimum similarity score for results
- `tags` (optional): List of tags to filter search results
- `response_prefix` (optional): Text to prepend to all search responses
- `response_postfix` (optional): Text to append to all search responses
- `no_results_message` (default: "No information found for '{query}'"): Custom message when no results found

**Multiple Instance Support:**
The native vector search skill supports multiple instances with different indexes and tool names:

**Example:**
```python
# Local mode with auto-build
agent.add_skill("native_vector_search", {
    "tool_name": "search_docs",
    "description": "Search SDK concepts guide",
    "build_index": True,
    "source_dir": "./docs",
    "index_file": "concepts.swsearch",
    "count": 5
})
# Creates tool: search_docs

# Remote mode connecting to search server
agent.add_skill("native_vector_search", {
    "tool_name": "search_knowledge",
    "description": "Search the knowledge base",
    "remote_url": "http://localhost:8001",
    "index_name": "concepts",
    "count": 3
})
# Creates tool: search_knowledge

# Multiple local indexes
agent.add_skill("native_vector_search", {
    "tool_name": "search_examples",
    "description": "Search code examples",
    "index_file": "examples.swsearch",
    "response_prefix": "From the examples:"
})
# Creates tool: search_examples

# Voice-optimized responses using concepts guide
agent.add_skill("native_vector_search", {
    "tool_name": "search_docs",
    "index_file": "concepts.swsearch",
    "response_prefix": "Based on the comprehensive SDK guide:",
    "response_postfix": "Would you like more specific information?",
    "no_results_message": "I couldn't find information about '{query}' in the concepts guide."
})
```

**Building Search Indexes:**
Before using local mode, you need to build search indexes:

```bash
# Build index from documentation
python -m signalwire_agents.cli.build_search docs --output docs.swsearch

# Build with custom settings
python -m signalwire_agents.cli.build_search ./knowledge \
    --output knowledge.swsearch \
    --file-types md,txt,pdf \
    --chunk-size 500 \
    --verbose
```

For complete documentation on the search system, see [Search Overview](search_overview.md).

### Skill Management

```python
# Check what skills are loaded
loaded_skills = agent.list_skills()
print(f"Loaded skills: {', '.join(loaded_skills)}")

# Check if a specific skill is loaded
if agent.has_skill("web_search"):
    print("Web search is available")

# Remove a skill (if needed)
agent.remove_skill("math")
```

### Advanced Skill Configuration with swaig_fields

Skills support a special `swaig_fields` parameter that allows you to customize how SWAIG functions are registered. When you pass `swaig_fields` to a skill, they are automatically merged into all tool definitions created by that skill through the `SkillBase.define_tool()` wrapper method.

```python
# Add a skill with swaig_fields to customize SWAIG function properties
agent.add_skill("math", {
    "precision": 2,  # Regular skill parameter
    "swaig_fields": {  # Special fields merged into SWAIG function automatically
        "secure": False,  # Override default security requirement
        "fillers": {
            "en-US": ["Let me calculate that...", "Computing the result..."],
            "es-ES": ["Déjame calcular eso...", "Calculando el resultado..."]
        }
    }
})

# Add web search with custom security and fillers
agent.add_skill("web_search", {
    "num_results": 3,
    "delay": 0.5,
    "swaig_fields": {
        "secure": True,  # Require authentication
        "fillers": {
            "en-US": ["Searching the web...", "Looking that up...", "Finding information..."]
        }
    }
})
```

The `swaig_fields` can include any parameter accepted by `AgentBase.define_tool()`:
- `secure`: Boolean indicating if the function requires authentication
- `fillers`: Dictionary mapping language codes to arrays of filler phrases
- Any other fields supported by the SWAIG function system

**Implementation Note**: The `SkillBase` class provides a `define_tool()` wrapper method that automatically injects `swaig_fields` into all tool definitions. Skills should use `self.define_tool()` instead of `self.agent.define_tool()` to get automatic swaig_fields support without manual handling.

### Error Handling

The skills system provides detailed error messages for common issues:

```python
try:
    agent.add_skill("web_search")
except ValueError as e:
    print(f"Failed to load skill: {e}")
    # Output: "Failed to load skill 'web_search': Missing required environment variables: ['GOOGLE_SEARCH_API_KEY']"
```

### Creating Custom Skills

You can create your own skills by extending the `SkillBase` class:

```python
from signalwire_agents.core.skill_base import SkillBase
from signalwire_agents.core.function_result import SwaigFunctionResult

class WeatherSkill(SkillBase):
    """A custom skill for weather information"""
    
    SKILL_NAME = "weather"
    SKILL_DESCRIPTION = "Get weather information for locations"
    SKILL_VERSION = "1.0.0"
    REQUIRED_PACKAGES = ["requests"]
    REQUIRED_ENV_VARS = ["WEATHER_API_KEY"]
    
    def setup(self) -> bool:
        """Setup the skill - validate dependencies and initialize"""
        if not self.validate_env_vars() or not self.validate_packages():
            return False
        
        # Get configuration parameters
        self.default_units = self.params.get('units', 'fahrenheit')
        self.timeout = self.params.get('timeout', 10)
        
        return True
    
    def register_tools(self) -> None:
        """Register tools with the agent"""
        self.define_tool(
            name="get_weather",
            description="Get current weather for a location",
            parameters={
                "location": {
                    "type": "string",
                    "description": "City or location name"
                },
                "units": {
                    "type": "string",
                    "description": "Temperature units (fahrenheit or celsius)",
                    "enum": ["fahrenheit", "celsius"]
                }
            },
            handler=self._get_weather_handler
        )
    
    def _get_weather_handler(self, args, raw_data):
        """Handle weather requests"""
        location = args.get("location", "")
        units = args.get("units", self.default_units)
        
        if not location:
            return SwaigFunctionResult("Please provide a location")
        
        # Your weather API integration here
        weather_data = f"Weather for {location}: 72°F and sunny"
        return SwaigFunctionResult(weather_data)
    
    def get_hints(self) -> List[str]:
        """Return speech recognition hints"""
        return ["weather", "temperature", "forecast", "conditions"]
    
    def get_prompt_sections(self) -> List[Dict[str, Any]]:
        """Return prompt sections to add to agent"""
        return [
            {
                "title": "Weather Information",
                "body": "You can provide current weather information for any location.",
                "bullets": [
                    "Use get_weather tool when users ask about weather",
                    "Always specify the location clearly",
                    "Include temperature and conditions in your response"
                ]
            }
        ]
```

**Using the custom skill:**
```python
# Place the skill in signalwire_agents/skills/weather/skill.py
# Then use it in your agent:

agent.add_skill("weather", {
    "units": "celsius",
    "timeout": 15
})
```

### Skills with Dynamic Configuration

Skills work seamlessly with dynamic configuration:

```python
class DynamicSkillAgent(AgentBase):
    def __init__(self):
        super().__init__(name="dynamic-skill-agent")
        self.set_dynamic_config_callback(self.configure_per_request)
    
    def configure_per_request(self, query_params, body_params, headers, agent):
        # Add different skills based on request parameters
        tier = query_params.get('tier', 'basic')
        
        # Basic skills for all users
        agent.add_skill("datetime")
        agent.add_skill("math")
        
        # Premium skills for premium users
        if tier == 'premium':
            agent.add_skill("web_search", {
                "num_results": 5,
                "delay": 0.5
            })
        elif tier == 'basic':
            agent.add_skill("web_search", {
                "num_results": 1,
                "delay": 0
            })
```

### Best Practices

1. **Choose appropriate parameters**: Configure skills for your use case
   ```python
   # For speed (customer service)
   agent.add_skill("web_search", {"num_results": 1, "delay": 0})
   
   # For research (detailed analysis)
   agent.add_skill("web_search", {"num_results": 5, "delay": 1.0})
   ```

2. **Handle missing dependencies gracefully**:
   ```python
   try:
       agent.add_skill("web_search")
   except ValueError as e:
       self.logger.warning(f"Web search unavailable: {e}")
       # Continue without web search capability
   ```

3. **Document your custom skills**: Include clear descriptions and parameter documentation

4. **Test skills in isolation**: Create simple test scripts to verify skill functionality

For more detailed information about the skills system architecture and advanced customization, see the [Skills System README](SKILLS_SYSTEM_README.md).

## Multilingual Support

Agents can support multiple languages:

```python
# Add English language
self.add_language(
    name="English",
    code="en-US",
    voice="en-US-Neural2-F",
    speech_fillers=["Let me think...", "One moment please..."],
    function_fillers=["I'm looking that up...", "Let me check that..."]
)

# Add Spanish language
self.add_language(
    name="Spanish",
    code="es",
    voice="rime.spore:multilingual",
    speech_fillers=["Un momento por favor...", "Estoy pensando..."]
)
```

### Voice Formats

There are different ways to specify voices:

```python
# Simple format
self.add_language(name="English", code="en-US", voice="en-US-Neural2-F")

# Explicit parameters with engine and model
self.add_language(
    name="British English",
    code="en-GB",
    voice="spore",
    engine="rime",
    model="multilingual"
)

# Combined string format
self.add_language(
    name="Spanish",
    code="es",
    voice="rime.spore:multilingual"
)
```

## Agent Configuration

### Adding Hints

Hints help the AI understand certain terms better:

```python
# Simple hints (list of words)
self.add_hints(["SignalWire", "SWML", "SWAIG"])

# Pattern hint with replacement
self.add_pattern_hint(
    hint="AI Agent", 
    pattern="AI\\s+Agent", 
    replace="A.I. Agent", 
    ignore_case=True
)
```

### Adding Pronunciation Rules

Pronunciation rules help the AI speak certain terms correctly:

```python
# Add pronunciation rule
self.add_pronunciation("API", "A P I", ignore_case=False)
self.add_pronunciation("SIP", "sip", ignore_case=True)
```

### Setting AI Parameters

Configure various AI behavior parameters:

```python
# Set AI parameters
self.set_params({
    "wait_for_user": False,
    "end_of_speech_timeout": 1000,
    "ai_volume": 5,
    "languages_enabled": True,
    "local_tz": "America/Los_Angeles"
})
```

### Setting Global Data

Provide global data for the AI to reference:

```python
# Set global data
self.set_global_data({
    "company_name": "SignalWire",
    "product": "AI Agent SDK",
    "supported_features": [
        "Voice AI",
        "Telephone integration",
        "SWAIG functions"
    ]
})
```

### Customizing LLM Parameters

The SDK provides methods to fine-tune the Language Model parameters for both the main prompt and post-prompt, giving you precise control over the AI's behavior:

```python
# Set LLM parameters for the main prompt
# These parameters are passed to the server which validates them based on the model
self.set_prompt_llm_params(
    temperature=0.7,        # Controls randomness
    top_p=0.9,             # Nucleus sampling threshold
    barge_confidence=0.6,  # ASR confidence to interrupt
    presence_penalty=0.0,  # Penalizes token repetition
    frequency_penalty=0.0  # Penalizes frequent word usage
)

# Set different parameters for the post-prompt
self.set_post_prompt_llm_params(
    temperature=0.3,       # Lower temperature for consistent summaries
    top_p=0.95            # Slightly wider token selection
)
```

**Common Use Cases:**

- **Customer Service**: Low temperature (0.2-0.4) for consistent, professional responses
- **Creative Tasks**: Higher temperature (0.7-0.9) for varied, creative outputs
- **Technical Support**: Very low temperature (0.1-0.3) with high confidence for accuracy
- **General Assistant**: Medium temperature (0.5-0.7) for balanced interaction

For detailed information about each parameter and advanced tuning strategies, see [LLM Parameters Guide](llm_parameters.md).

## Dynamic Agent Configuration

Dynamic agent configuration allows you to configure agents per-request based on parameters from the HTTP request (query parameters, body data, headers). This enables powerful patterns like multi-tenant applications, A/B testing, personalization, and localization.

### Overview

There are two main approaches to agent configuration:

#### Static Configuration (Traditional)
```python
class StaticAgent(AgentBase):
    def __init__(self):
        super().__init__(name="static-agent")
        
        # Configuration happens once at startup
        self.add_language("English", "en-US", "rime.spore:mistv2")
        self.set_params({"end_of_speech_timeout": 500})
        self.prompt_add_section("Role", "You are a customer service agent.")
        self.set_global_data({"service_level": "standard"})
```

**Pros**: Simple, fast, predictable
**Cons**: Same behavior for all users, requires separate agents for different configurations

#### Dynamic Configuration (New)
```python
class DynamicAgent(AgentBase):
    def __init__(self):
        super().__init__(name="dynamic-agent")
        
        # No static configuration - set up dynamic callback instead
        self.set_dynamic_config_callback(self.configure_per_request)
    
    def configure_per_request(self, query_params, body_params, headers, agent):
        # Configuration happens fresh for each request
        tier = query_params.get('tier', 'standard')
        
        if tier == 'premium':
            agent.add_language("English", "en-US", "rime.spore:mistv2")
            agent.set_params({"end_of_speech_timeout": 300})  # Faster
            agent.prompt_add_section("Role", "You are a premium customer service agent.")
            agent.set_global_data({"service_level": "premium"})
        else:
            agent.add_language("English", "en-US", "rime.spore:mistv2")
            agent.set_params({"end_of_speech_timeout": 500})  # Standard
            agent.prompt_add_section("Role", "You are a customer service agent.")
            agent.set_global_data({"service_level": "standard"})
```

**Pros**: Highly flexible, single agent serves multiple configurations, enables advanced use cases
**Cons**: Slightly more complex, configuration overhead per request

### Setting Up Dynamic Configuration

Use the `set_dynamic_config_callback()` method to register a callback function that will be called for each request:

```python
class MyDynamicAgent(AgentBase):
    def __init__(self):
        super().__init__(name="my-agent", route="/agent")
        
        # Register the dynamic configuration callback
        self.set_dynamic_config_callback(self.configure_agent_dynamically)
    
    def configure_agent_dynamically(self, query_params, body_params, headers, agent):
        """
        This method is called for every request to configure the agent
        
        Args:
            query_params (dict): Query string parameters from the URL
            body_params (dict): Parsed JSON body from POST requests
            headers (dict): HTTP headers from the request
            agent (AgentBase): The agent instance to configure
        """
        # Your dynamic configuration logic here
        pass
```

The callback function receives four parameters:
- **query_params**: Dictionary of URL query parameters
- **body_params**: Dictionary of parsed JSON body (empty for GET requests)
- **headers**: Dictionary of HTTP headers
- **agent**: The agent instance to configure dynamically

### Dynamic Configuration Methods

The `agent` parameter in your callback is the actual agent instance, allowing you to use all the same configuration methods you would use during initialization:

#### Language Configuration
```python
# Add languages with voice configuration
agent.add_language("English", "en-US", "rime.spore:mistv2")
agent.add_language("Spanish", "es-ES", "rime.spore:mistv2")
```

#### Prompt Building
```python
# Add prompt sections
agent.prompt_add_section("Role", "You are a helpful assistant.")
agent.prompt_add_section("Guidelines", bullets=[
    "Be professional and courteous",
    "Provide accurate information",
    "Ask clarifying questions when needed"
])

# Set raw prompt text
agent.set_prompt_text("You are a specialized AI assistant...")

# Set post-prompt for summary
agent.set_post_prompt("Summarize the key points of this conversation.")
```

#### AI Parameters
```python
# Configure AI behavior
agent.set_params({
    "end_of_speech_timeout": 300,
    "attention_timeout": 20000,
    "background_file_volume": -30
})
```

#### Global Data
```python
# Set data available to the AI
agent.set_global_data({
    "customer_tier": "premium",
    "features_enabled": ["advanced_support", "priority_queue"],
    "session_info": {"start_time": "2024-01-01T00:00:00Z"}
})

# Update existing global data
agent.update_global_data({"additional_info": "value"})
```

#### Speech Recognition Hints
```python
# Add hints for better speech recognition
agent.add_hints(["SignalWire", "SWML", "API", "technical"])
agent.add_pronunciation("API", "A P I")
```

#### Function Configuration
```python
# Set native functions
agent.set_native_functions(["transfer", "hangup"])

# Add function includes
agent.add_function_include(
    url="https://api.example.com/functions",
    functions=["get_account_info", "update_profile"]
)
```

### Request Data Access

Your callback function receives detailed information about the incoming request:

#### Query Parameters
```python
def configure_agent_dynamically(self, query_params, body_params, headers, agent):
    # Extract query parameters
    tier = query_params.get('tier', 'standard')
    language = query_params.get('language', 'en')
    customer_id = query_params.get('customer_id')
    debug = query_params.get('debug', '').lower() == 'true'
    
    # Use parameters for configuration
    if tier == 'premium':
        agent.set_params({"end_of_speech_timeout": 300})
    
    if customer_id:
        agent.set_global_data({"customer_id": customer_id})

# Request: GET /agent?tier=premium&language=es&customer_id=12345&debug=true
```

#### POST Body Parameters
```python
def configure_agent_dynamically(self, query_params, body_params, headers, agent):
    # Extract from POST body
    user_profile = body_params.get('user_profile', {})
    preferences = body_params.get('preferences', {})
    
    # Configure based on profile
    if user_profile.get('language') == 'es':
        agent.add_language("Spanish", "es-ES", "rime.spore:mistv2")
    
    if preferences.get('voice_speed') == 'fast':
        agent.set_params({"end_of_speech_timeout": 200})

# Request: POST /agent with JSON body:
# {
#   "user_profile": {"language": "es", "region": "mx"},
#   "preferences": {"voice_speed": "fast", "tone": "formal"}
# }
```

#### HTTP Headers
```python
def configure_agent_dynamically(self, query_params, body_params, headers, agent):
    # Extract headers
    user_agent = headers.get('user-agent', '')
    auth_token = headers.get('authorization', '')
    locale = headers.get('accept-language', 'en-US')
    
    # Configure based on headers
    if 'mobile' in user_agent.lower():
        agent.set_params({"end_of_speech_timeout": 400})  # Longer for mobile
    
    if locale.startswith('es'):
        agent.add_language("Spanish", "es-ES", "rime.spore:mistv2")
```

### Configuration Examples

#### Simple Multi-Tenant Configuration
```python
def configure_agent_dynamically(self, query_params, body_params, headers, agent):
    tenant = query_params.get('tenant', 'default')
    
    # Tenant-specific configuration
    if tenant == 'healthcare':
        agent.add_language("English", "en-US", "rime.spore:mistv2")
        agent.prompt_add_section("Compliance", 
            "Follow HIPAA guidelines and maintain patient confidentiality.")
        agent.set_global_data({
            "industry": "healthcare",
            "compliance_level": "hipaa"
        })
    elif tenant == 'finance':
        agent.add_language("English", "en-US", "rime.spore:mistv2")
        agent.prompt_add_section("Compliance",
            "Follow financial regulations and protect sensitive data.")
        agent.set_global_data({
            "industry": "finance", 
            "compliance_level": "pci"
        })
```

#### Language and Localization
```python
def configure_agent_dynamically(self, query_params, body_params, headers, agent):
    language = query_params.get('language', 'en')
    region = query_params.get('region', 'us')
    
    # Configure language and voice
    if language == 'es':
        if region == 'mx':
            agent.add_language("Spanish (Mexico)", "es-MX", "rime.spore:mistv2")
        else:
            agent.add_language("Spanish", "es-ES", "rime.spore:mistv2")
        
        agent.prompt_add_section("Language", "Respond in Spanish.")
    elif language == 'fr':
        agent.add_language("French", "fr-FR", "rime.alois")
        agent.prompt_add_section("Language", "Respond in French.")
    else:
        agent.add_language("English", "en-US", "rime.spore:mistv2")
    
    # Regional customization
    agent.set_global_data({
        "language": language,
        "region": region,
        "currency": "USD" if region == "us" else "EUR" if region == "eu" else "MXN"
    })
```

#### A/B Testing Configuration
```python
def configure_agent_dynamically(self, query_params, body_params, headers, agent):
    # Determine test group (could be from query param, user ID hash, etc.)
    test_group = query_params.get('test_group', 'A')
    
    if test_group == 'A':
        # Control group - standard configuration
        agent.set_params({"end_of_speech_timeout": 500})
        agent.prompt_add_section("Style", "Use a standard conversational approach.")
        agent.set_global_data({"test_group": "A", "features": ["basic"]})
    else:
        # Test group B - experimental features
        agent.set_params({"end_of_speech_timeout": 300})
        agent.prompt_add_section("Style", 
            "Use an enhanced, more interactive conversational approach.")
        agent.set_global_data({"test_group": "B", "features": ["basic", "enhanced"]})
```

#### Customer Tier-Based Configuration
```python
def configure_agent_dynamically(self, query_params, body_params, headers, agent):
    customer_id = query_params.get('customer_id')
    tier = query_params.get('tier', 'standard')
    
    # Base configuration
    agent.add_language("English", "en-US", "rime.spore:mistv2")
    
    # Tier-specific configuration
    if tier == 'enterprise':
        agent.set_params({
            "end_of_speech_timeout": 200,  # Fastest response
            "attention_timeout": 30000     # Longest attention span
        })
        agent.prompt_add_section("Service Level",
            "You provide white-glove enterprise support with priority handling.")
        features = ["all_features", "dedicated_support", "custom_integration"]
    elif tier == 'premium':
        agent.set_params({
            "end_of_speech_timeout": 300,
            "attention_timeout": 20000
        })
        agent.prompt_add_section("Service Level",
            "You provide premium support with enhanced features.")
        features = ["premium_features", "priority_support"]
    else:
        agent.set_params({
            "end_of_speech_timeout": 500,
            "attention_timeout": 15000
        })
        agent.prompt_add_section("Service Level",
            "You provide standard customer support.")
        features = ["basic_features"]
    
    # Set global data
    global_data = {"tier": tier, "features": features}
    if customer_id:
        global_data["customer_id"] = customer_id
    agent.set_global_data(global_data)
```

### Use Cases

#### Multi-Tenant SaaS Applications
Perfect for SaaS platforms where each customer needs different agent behavior:

```python
# Different tenants get different capabilities
# /agent?tenant=acme&industry=healthcare
# /agent?tenant=globex&industry=finance
```

Benefits:
- Single agent deployment serves all customers
- Tenant-specific branding and behavior
- Industry-specific compliance and terminology
- Custom feature sets per subscription level

#### A/B Testing and Experimentation
Test different agent configurations with real users:

```python
# Split traffic between different configurations
# /agent?test_group=A  (control)
# /agent?test_group=B  (experimental)
```

Benefits:
- Compare agent performance metrics
- Test new features with subset of users
- Gradual rollout of improvements
- Data-driven optimization

#### Personalization and User Preferences
Adapt agent behavior to individual user preferences:

```python
# Personalized based on user profile
# /agent?user_id=123&voice_speed=fast&formality=casual
```

Benefits:
- Improved user experience
- Accessibility support (voice speed, etc.)
- Cultural and linguistic adaptation
- Learning from user interactions

#### Geographic and Cultural Localization
Adapt to different regions and cultures:

```python
# Location-based configuration
# /agent?country=mx&language=es&timezone=America/Mexico_City
```

Benefits:
- Local language and dialect support
- Cultural appropriateness
- Regional business practices
- Time zone aware responses

### Migration Guide

#### Converting Static Agents to Dynamic

**Step 1: Move Configuration to Callback**

Before (Static):
```python
class MyAgent(AgentBase):
    def __init__(self):
        super().__init__(name="my-agent")
        
        # Static configuration
        self.add_language("English", "en-US", "rime.spore:mistv2")
        self.set_params({"end_of_speech_timeout": 500})
        self.prompt_add_section("Role", "You are a helpful assistant.")
        self.set_global_data({"version": "1.0"})
```

After (Dynamic):
```python
class MyAgent(AgentBase):
    def __init__(self):
        super().__init__(name="my-agent")
        
        # Set up dynamic configuration
        self.set_dynamic_config_callback(self.configure_agent)
    
    def configure_agent(self, query_params, body_params, headers, agent):
        # Same configuration, but now dynamic
        agent.add_language("English", "en-US", "rime.spore:mistv2")
        agent.set_params({"end_of_speech_timeout": 500})
        agent.prompt_add_section("Role", "You are a helpful assistant.")
        agent.set_global_data({"version": "1.0"})
```

**Step 2: Add Parameter-Based Logic**

```python
def configure_agent(self, query_params, body_params, headers, agent):
    # Start with base configuration
    agent.add_language("English", "en-US", "rime.spore:mistv2")
    agent.prompt_add_section("Role", "You are a helpful assistant.")
    
    # Add parameter-based customization
    timeout = int(query_params.get('timeout', '500'))
    agent.set_params({"end_of_speech_timeout": timeout})
    
    version = query_params.get('version', '1.0')
    agent.set_global_data({"version": version})
```

**Step 3: Test Both Approaches**

You can support both static and dynamic patterns during migration:

```python
class MyAgent(AgentBase):
    def __init__(self, use_dynamic=False):
        super().__init__(name="my-agent")
        
        if use_dynamic:
            self.set_dynamic_config_callback(self.configure_agent)
        else:
            # Keep static configuration for backward compatibility
            self._setup_static_config()
    
    def _setup_static_config(self):
        # Original static configuration
        self.add_language("English", "en-US", "rime.spore:mistv2")
        # ... rest of static config
    
    def configure_agent(self, query_params, body_params, headers, agent):
        # New dynamic configuration
        # ... dynamic config logic
```

### Best Practices

#### Performance Considerations

1. **Keep Callbacks Lightweight**
```python
def configure_agent(self, query_params, body_params, headers, agent):
    # Good: Simple parameter extraction and configuration
    tier = query_params.get('tier', 'standard')
    agent.set_params(TIER_CONFIGS[tier])
    
    # Avoid: Heavy computation or external API calls
    # customer_data = expensive_api_call(customer_id)  # Don't do this
```

2. **Cache Configuration Data**
```python
class MyAgent(AgentBase):
    def __init__(self):
        super().__init__(name="my-agent")
        
        # Pre-compute configuration templates
        self.tier_configs = {
            'basic': {'end_of_speech_timeout': 500},
            'premium': {'end_of_speech_timeout': 300},
            'enterprise': {'end_of_speech_timeout': 200}
        }
        
        self.set_dynamic_config_callback(self.configure_agent)
    
    def configure_agent(self, query_params, body_params, headers, agent):
        tier = query_params.get('tier', 'basic')
        agent.set_params(self.tier_configs[tier])
```

3. **Use Default Values**
```python
def configure_agent(self, query_params, body_params, headers, agent):
    # Always provide defaults
    language = query_params.get('language', 'en')
    tier = query_params.get('tier', 'standard')
    
    # Handle invalid values gracefully
    if language not in ['en', 'es', 'fr']:
        language = 'en'
```

#### Security Considerations

1. **Validate Input Parameters**
```python
def configure_agent(self, query_params, body_params, headers, agent):
    # Validate and sanitize inputs
    tier = query_params.get('tier', 'standard')
    if tier not in ['basic', 'premium', 'enterprise']:
        tier = 'basic'  # Safe default
    
    # Validate numeric parameters
    try:
        timeout = int(query_params.get('timeout', '500'))
        timeout = max(100, min(timeout, 2000))  # Clamp to reasonable range
    except ValueError:
        timeout = 500  # Safe default
```

2. **Protect Sensitive Configuration**
```python
def configure_agent(self, query_params, body_params, headers, agent):
    # Don't expose internal configuration via parameters
    # Bad: agent.set_global_data({"api_key": query_params.get('api_key')})
    
    # Good: Use internal mapping for call-related data only
    customer_id = query_params.get('customer_id')
    if customer_id and self.is_valid_customer(customer_id):
        # Store call-related customer info, NOT sensitive credentials
        agent.set_global_data({
            "customer_id": customer_id,
            "customer_tier": self.get_customer_tier(customer_id),
            "account_type": "premium"
        })
```

3. **Rate Limiting for Complex Configurations**
```python
from functools import lru_cache

class MyAgent(AgentBase):
    @lru_cache(maxsize=100)
    def get_customer_config(self, customer_id):
        # Cache expensive lookups
        return self.database.get_customer_settings(customer_id)
    
    def configure_agent(self, query_params, body_params, headers, agent):
        customer_id = query_params.get('customer_id')
        if customer_id:
            config = self.get_customer_config(customer_id)
            agent.set_global_data(config)
```

#### Error Handling

1. **Graceful Degradation**
```python
def configure_agent(self, query_params, body_params, headers, agent):
    try:
        # Try custom configuration
        self.apply_custom_config(query_params, agent)
    except Exception as e:
        # Log error but don't fail the request
        self.log.error("config_error", error=str(e))
        
        # Fall back to default configuration
        self.apply_default_config(agent)
```

2. **Configuration Validation**
```python
def configure_agent(self, query_params, body_params, headers, agent):
    # Validate required parameters
    if not query_params.get('tenant'):
        agent.set_global_data({"error": "Missing tenant parameter"})
        return
    
    # Validate configuration makes sense
    language = query_params.get('language', 'en')
    region = query_params.get('region', 'us')
    
    if language == 'es' and region == 'us':
        # Adjust for Spanish speakers in US
        agent.add_language("Spanish (US)", "es-US", "rime.spore:mistv2")
```

Dynamic agent configuration is a powerful feature that enables sophisticated, multi-tenant AI applications while maintaining the familiar AgentBase API. Start with simple parameter-based configuration and gradually add more complex logic as your use cases evolve.

## Advanced Features

### Debug Events

The debug events system provides real-time visibility into what the AI module is doing during a call. When enabled, the module POSTs structured JSON events to your agent throughout the call lifecycle — session start/end, barge interruptions, LLM errors, step changes, and more.

#### Basic Setup

```python
agent = AgentBase("my_agent")
agent.enable_debug_events()  # That's it — events are auto-logged
agent.serve()
```

With just `enable_debug_events()`, every debug event is logged through the agent's structured logger. No other configuration is needed — the SDK automatically:
- Registers a `/debug_events` endpoint on the agent
- Sets `debug_webhook_url` and `debug_webhook_level` in the SWML params
- Logs each incoming event with its type and payload

#### Custom Event Handler

To act on specific events (alerting, metrics, custom logging), register a handler:

```python
agent = AgentBase("my_agent")
agent.enable_debug_events()

@agent.on_debug_event
def handle(event_type, data):
    call_id = data.get("call_id")

    if event_type == "barge":
        print(f"[{call_id}] Caller interrupted after {data.get('barge_elapsed_ms')}ms")

    elif event_type == "llm_error":
        print(f"[{call_id}] LLM error: {data.get('event')}")
        alert_ops_team(data)

    elif event_type == "session_end":
        duration = data.get("duration_ms", 0) / 1000
        print(f"[{call_id}] Call ended after {duration:.1f}s — reason: {data.get('reason')}")

agent.serve()
```

The handler is called for every event in addition to the default structured logging.

#### Verbosity Levels

- **Level 1** (default): High-level events — session start/end, barge, errors, step changes, hold, filler, gather flow, action processing
- **Level 2+**: Adds high-volume events — every LLM request/response, conversation history additions

```python
agent.enable_debug_events(level=2)  # Include LLM request/response events
```

For the complete list of event types and their payloads, see the [API Reference](api_reference.md#debug-events).

### Session Lifecycle Hooks

SignalWire provides special SWAIG functions that are automatically called at specific points during a voice session's lifecycle. These hooks enable you to perform initialization tasks when a call starts and cleanup tasks when a call ends.

#### Overview

Session lifecycle hooks are special SWAIG functions that SignalWire calls automatically:
- `startup_hook`: Called immediately when a new voice session begins
- `hangup_hook`: Called when a voice session ends (regardless of how it ended)

These hooks are particularly useful for:
- Initializing session state or resources
- Loading user preferences or history
- Logging session start/end events
- Cleaning up temporary resources
- Saving session data for analytics

#### Implementation

To implement lifecycle hooks, define them as regular SWAIG functions with these specific names:

```python
from signalwire_agents import AgentBase, SwaigFunctionResult

class MyAgent(AgentBase):
    def __init__(self):
        super().__init__(name="my-agent")
    
    @AgentBase.tool(
        name="startup_hook",
        description="Called when the voice session starts"
    )
    def startup_hook(self, args, raw_data):
        # Extract session information
        call_id = raw_data.get("call_id")
        from_number = raw_data.get("from_number")
        to_number = raw_data.get("to_number")
        
        # Initialize session state
        self.update_state(call_id, {
            "session_start": datetime.now().isoformat(),
            "from": from_number,
            "to": to_number,
            "interaction_count": 0
        })
        
        # Log session start
        print(f"Session started: {call_id} from {from_number}")
        
        # Return success (SignalWire expects a response)
        return SwaigFunctionResult("Session initialized successfully")
    
    @AgentBase.tool(
        name="hangup_hook",
        description="Called when the voice session ends"
    )
    def hangup_hook(self, args, raw_data):
        # Extract session information
        call_id = raw_data.get("call_id")
        
        # Retrieve session state
        state = self.get_state(call_id)
        
        if state:
            # Calculate session duration
            start_time = datetime.fromisoformat(state.get("session_start"))
            duration = (datetime.now() - start_time).total_seconds()
            
            # Log session metrics
            print(f"Session ended: {call_id}")
            print(f"Duration: {duration} seconds")
            print(f"Interactions: {state.get('interaction_count', 0)}")
            
            # Clean up state (optional - SignalWire will clean up automatically)
            self.delete_state(call_id)
        
        return SwaigFunctionResult("Session cleanup completed")
```

#### Common Use Cases

##### 1. User Preference Loading
```python
@AgentBase.tool(name="startup_hook", description="Called when the voice session starts", parameters={})
def startup_hook(self, args, raw_data):
    caller_id = raw_data.get("from_number")
    
    # Load user preferences from database
    preferences = self.load_user_preferences(caller_id)
    
    # Store in session state for quick access
    self.update_state(raw_data.get("call_id"), {
        "user_preferences": preferences,
        "language": preferences.get("language", "en-US"),
        "previous_orders": preferences.get("recent_orders", [])
    })
    
    return SwaigFunctionResult("User preferences loaded")
```

##### 2. Analytics and Logging
```python
@AgentBase.tool(name="hangup_hook", description="Called when the voice session ends", parameters={})
def hangup_hook(self, args, raw_data):
    call_id = raw_data.get("call_id")
    state = self.get_state(call_id)
    
    # Send analytics data
    analytics_data = {
        "call_id": call_id,
        "duration": state.get("duration"),
        "functions_called": state.get("functions_called", []),
        "outcome": state.get("outcome", "unknown")
    }
    
    # Post to analytics service
    self.send_to_analytics(analytics_data)
    
    return SwaigFunctionResult("Analytics data sent")
```

#### Important Notes

1. **Function Names**: The hooks must be named exactly `startup_hook` and `hangup_hook` for SignalWire to call them
2. **Error Handling**: Always implement proper error handling in hooks - failures shouldn't crash the voice session
3. **Timing**: `startup_hook` is called before the AI starts speaking to the caller
4. **Session Data**: Any data you need to persist across the session should be stored in external storage (Redis, database, etc.)
5. **Return Values**: Both hooks must return a `SwaigFunctionResult` object

### SIP Routing

SIP routing allows your agents to receive voice calls via SIP addresses. The SDK supports both individual agent-level routing and centralized server-level routing.

#### Individual Agent SIP Routing

Enable SIP routing on a single agent:

```python
# Enable SIP routing with automatic username mapping based on agent name
agent.enable_sip_routing(auto_map=True)

# Register additional SIP usernames for this agent
agent.register_sip_username("support_agent")
agent.register_sip_username("help_desk")
```

When `auto_map=True`, the agent automatically registers SIP usernames based on:
- The agent's name (e.g., `support@domain`)
- The agent's route path (e.g., `/support` becomes `support@domain`)
- Common variations (e.g., removing vowels for shorter dialing)

#### Server-Level SIP Routing (Multi-Agent)

For multi-agent setups, centralized routing is more efficient:

```python
# Create an AgentServer
server = AgentServer(host="0.0.0.0", port=3000)

# Register multiple agents
server.register(registration_agent)  # Route: /register
server.register(support_agent)       # Route: /support

# Set up central SIP routing
server.setup_sip_routing(route="/sip", auto_map=True)

# Register additional SIP username mappings
server.register_sip_username("signup", "/register")    # signup@domain → registration agent
server.register_sip_username("help", "/support")       # help@domain → support agent
```

With server-level routing:
- Each agent is reachable via its name (when `auto_map=True`)
- Additional SIP usernames can be mapped to specific agent routes
- All SIP routing is handled at a single endpoint (`/sip` by default)

#### How SIP Routing Works

1. A SIP call comes in with a username (e.g., `support@yourdomain`)
2. The SDK extracts the username part (`support`)
3. The system checks if this username is registered:
   - In individual routing: The current agent checks its own username list
   - In server routing: The server checks its central mapping table
4. If a match is found, the call is routed to the appropriate agent

### Custom Routing

You can dynamically handle requests to different paths using routing callbacks:

```python
# Enable custom routing in the constructor or anytime after initialization
self.register_routing_callback(self.handle_customer_route, path="/customer")
self.register_routing_callback(self.handle_product_route, path="/product")

# Define the routing handlers
def handle_customer_route(self, request, body):
    """
    Process customer-related requests
    
    Args:
        request: FastAPI Request object
        body: Parsed JSON body as dictionary
        
    Returns:
        Optional[str]: A URL to redirect to, or None to process normally
    """
    # Extract any relevant data
    customer_id = body.get("customer_id")
    
    # You can redirect to another agent/service if needed
    if customer_id and customer_id.startswith("vip-"):
        return f"/vip-handler/{customer_id}"
        
    # Or return None to process the request with on_swml_request
    return None
    
# Customize SWML based on the route in on_swml_request
def on_swml_request(self, request_data=None, callback_path=None):
    """
    Customize SWML based on the request and path
    
    Args:
        request_data: The request body data
        callback_path: The path that triggered the routing callback
    """
    if callback_path == "/customer":
        # Serve customer-specific content
        return {
            "sections": {
                "main": [
                    {"answer": {}},
                    {"play": {"url": "say:Welcome to customer service!"}}
                ]
            }
        }
    # Other path handling...
    return None
```

### Customizing SWML Requests

You can modify the SWML document based on request data by overriding the `on_swml_request` method:

```python
def on_swml_request(self, request_data=None, callback_path=None):
    """
    Customize the SWML document based on request data
    
    Args:
        request_data: The request data (body for POST or query params for GET)
        callback_path: The path that triggered the routing callback
        
    Returns:
        Optional dict with modifications to apply to the document
    """
    if request_data and "caller_type" in request_data:
        # Example: Return modifications to change the AI behavior based on caller type
        if request_data["caller_type"] == "vip":
            return {
                "sections": {
                    "main": [
                        # Keep the first verb (answer)
                        # Modify the AI verb parameters
                        {
                            "ai": {
                                "params": {
                                    "wait_for_user": False,
                                    "end_of_speech_timeout": 500  # More responsive
                                }
                            }
                        }
                    ]
                }
            }
            
    # You can also use the callback_path to serve different content based on the route
    if callback_path == "/customer":
        return {
            "sections": {
                "main": [
                    {"answer": {}},
                    {"play": {"url": "say:Welcome to our customer service line."}}
                ]
            }
        }
    
    # Return None to use the default document
    return None
```

### Conversation Summary Handling

Process conversation summaries:

```python
def on_summary(self, summary, raw_data=None):
    """
    Handle the conversation summary
    
    Args:
        summary: The summary object or None if no summary was found
        raw_data: The complete raw POST data from the request
    """
    if summary:
        # Log the summary
        self.log.info("conversation_summary", summary=summary)
        
        # Save the summary to a database, send notifications, etc.
        # ...
```

### Custom Webhook URLs

You can override the default webhook URLs for SWAIG functions and post-prompt delivery:

```python
# In your agent initialization or setup code:

# Override the webhook URL for all SWAIG functions
agent.set_web_hook_url("https://external-service.example.com/handle-swaig")

# Override the post-prompt delivery URL
agent.set_post_prompt_url("https://analytics.example.com/conversation-summaries")

# These methods allow you to:
# 1. Send function calls to external services instead of handling them locally
# 2. Send conversation summaries to analytics services or other systems
# 3. Use special URLs with pre-configured authentication
```

### External Input Checking

The SDK provides a check-for-input endpoint that allows agents to check for new input from external systems:

```python
# Example client code that checks for new input
import requests
import json

def check_for_new_input(agent_url, conversation_id, auth):
    """
    Check if there's any new input for a conversation
    
    Args:
        agent_url: Base URL for the agent
        conversation_id: ID of the conversation to check
        auth: (username, password) tuple for basic auth
    
    Returns:
        New messages if any, None otherwise
    """
    url = f"{agent_url}/check_for_input"
    response = requests.post(
        url,
        json={"conversation_id": conversation_id},
        auth=auth
    )
    
    if response.status_code == 200:
        data = response.json()
        if data.get("new_input", False):
            return data.get("messages", [])
    
    return None
```

By default, the check_for_input endpoint returns an empty response. To implement custom behavior, override the `_handle_check_for_input_request` method in your agent:

```python
async def _handle_check_for_input_request(self, request):
    # First do basic authentication check
    if not self._check_basic_auth(request):
        return Response(
            content=json.dumps({"error": "Unauthorized"}),
            status_code=401,
            headers={"WWW-Authenticate": "Basic"},
            media_type="application/json"
        )
    
    # Get conversation_id from request
    conversation_id = None
    if request.method == "POST":
        body = await request.json()
        conversation_id = body.get("conversation_id")
    else:
        conversation_id = request.query_params.get("conversation_id")
    
    if not conversation_id:
        return Response(
            content=json.dumps({"error": "Missing conversation_id"}),
            status_code=400,
            media_type="application/json"
        )
    
    # Custom logic to check for new input
    # For example, checking a database or external API
    messages = self._get_new_messages(conversation_id)
    
    return {
        "status": "success",
        "conversation_id": conversation_id,
        "new_input": len(messages) > 0,
        "messages": messages
    }
```

This endpoint is useful for implementing asynchronous conversations where users might send messages through different channels that need to be incorporated into the agent conversation.

## Prefab Agents

Prefab agents are pre-configured agent implementations designed for specific use cases. They provide ready-to-use functionality with customization options, saving development time and ensuring consistent patterns.

### Built-in Prefabs

The SDK includes several built-in prefab agents:

#### InfoGathererAgent

Collects structured information from users:

```python
from signalwire_agents.prefabs import InfoGathererAgent

agent = InfoGathererAgent(
    fields=[
        {"name": "full_name", "prompt": "What is your full name?"},
        {"name": "email", "prompt": "What is your email address?", 
         "validation": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"},
        {"name": "reason", "prompt": "How can I help you today?"}
    ],
    confirmation_template="Thanks {full_name}, I'll help you with {reason}. I'll send a confirmation to {email}.",
    name="info-gatherer",
    route="/info-gatherer"
)

agent.serve(host="0.0.0.0", port=8000)
```

#### FAQBotAgent

Answers questions based on a knowledge base:

```python
from signalwire_agents.prefabs import FAQBotAgent

agent = FAQBotAgent(
    knowledge_base_path="./docs",
    personality="I'm a product documentation assistant.",
    citation_style="inline",
    name="knowledge-base",
    route="/knowledge-base"
)

agent.serve(host="0.0.0.0", port=8000)
```

#### ConciergeAgent

Routes users to specialized agents:

```python
from signalwire_agents.prefabs import ConciergeAgent

agent = ConciergeAgent(
    routing_map={
        "technical_support": {
            "url": "http://tech-support-agent:8001",
            "criteria": ["error", "broken", "not working"]
        },
        "sales": {
            "url": "http://sales-agent:8002",
            "criteria": ["pricing", "purchase", "subscribe"]
        }
    },
    greeting="Welcome to SignalWire. How can I help you today?",
    name="concierge",
    route="/concierge"
)

agent.serve(host="0.0.0.0", port=8000)
```

#### SurveyAgent

Conducts structured surveys with different question types:

```python
from signalwire_agents.prefabs import SurveyAgent

agent = SurveyAgent(
    survey_name="Customer Satisfaction",
    introduction="We'd like to know about your recent experience with our product.",
    questions=[
        {
            "id": "satisfaction",
            "text": "How satisfied are you with our product?",
            "type": "rating",
            "scale": 5,
            "labels": {
                "1": "Very dissatisfied",
                "5": "Very satisfied"
            }
        },
        {
            "id": "feedback",
            "text": "Do you have any specific feedback about how we can improve?",
            "type": "text"
        }
    ],
    name="satisfaction-survey",
    route="/survey"
)

agent.serve(host="0.0.0.0", port=8000)
```

#### ReceptionistAgent

Handles call routing and department transfers:

```python
from signalwire_agents.prefabs import ReceptionistAgent

agent = ReceptionistAgent(
    departments=[
        {"name": "sales", "description": "For product inquiries and pricing", "number": "+15551235555"},
        {"name": "support", "description": "For technical assistance", "number": "+15551236666"},
        {"name": "billing", "description": "For payment and invoice questions", "number": "+15551237777"}
    ],
    greeting="Thank you for calling ACME Corp. How may I direct your call?",
    voice="rime.spore:mistv2",
    name="acme-receptionist",
    route="/reception"
)

agent.serve(host="0.0.0.0", port=8000)
```

### Creating Your Own Prefabs

You can create your own prefab agents by extending `AgentBase` or any existing prefab. Custom prefabs can be created directly within your project or packaged as reusable libraries.

#### Basic Prefab Structure

A well-designed prefab should:

1. Extend `AgentBase` or another prefab
2. Take configuration parameters in the constructor
3. Apply configuration to set up the agent
4. Provide appropriate default values
5. Include domain-specific tools

Example of a custom support agent prefab:

```python
from signalwire_agents import AgentBase
from signalwire_agents.core.function_result import SwaigFunctionResult

class CustomerSupportAgent(AgentBase):
    def __init__(
        self,
        product_name,
        knowledge_base_path=None,
        support_email=None,
        escalation_path=None,
        **kwargs
    ):
        # Pass standard params to parent
        super().__init__(**kwargs)
        
        # Store custom configuration
        self._product_name = product_name
        self._knowledge_base_path = knowledge_base_path
        self._support_email = support_email
        self._escalation_path = escalation_path
        
        # Configure prompt
        self.prompt_add_section("Personality", 
                               body=f"I am a customer support agent for {product_name}.")
        self.prompt_add_section("Goal", 
                               body="Help customers solve their problems effectively.")
        
        # Add standard instructions
        self._configure_instructions()
        
        # Register default tools
        self._register_default_tools()
    
    def _configure_instructions(self):
        """Configure standard instructions based on settings"""
        instructions = [
            "Be professional but friendly.",
            "Verify the customer's identity before sharing account details."
        ]
        
        if self._escalation_path:
            instructions.append(
                f"For complex issues, offer to escalate to {self._escalation_path}."
            )
            
        self.prompt_add_section("Instructions", bullets=instructions)
    
    def _register_default_tools(self):
        """Register default tools if appropriate paths are configured"""
        if self._knowledge_base_path:
            self.register_knowledge_base_tool()
    
    def register_knowledge_base_tool(self):
        """Register the knowledge base search tool if configured"""
        # Implementation...
        pass
    
    @AgentBase.tool(
        name="escalate_issue",
        description="Escalate a customer issue to a human agent",
        parameters={
            "issue_summary": {"type": "string", "description": "Brief summary of the issue"},
            "customer_email": {"type": "string", "description": "Customer's email address"}
        }
    )
    def escalate_issue(self, args, raw_data):
        # Implementation...
        return SwaigFunctionResult("Issue escalated successfully.")
    
    @AgentBase.tool(
        name="send_support_email",
        description="Send a follow-up email to the customer",
        parameters={
            "customer_email": {"type": "string"},
            "issue_summary": {"type": "string"},
            "resolution_steps": {"type": "string"}
        }
    )
    def send_support_email(self, args, raw_data):
        # Implementation...
        return SwaigFunctionResult("Follow-up email sent successfully.")
```

#### Using the Custom Prefab

```python
# Create an instance of the custom prefab
support_agent = CustomerSupportAgent(
    product_name="SignalWire Voice API",
    knowledge_base_path="./product_docs",
    support_email="support@example.com",
    escalation_path="tier 2 support",
    name="voice-support",
    route="/voice-support"
)

# Start the agent
support_agent.serve(host="0.0.0.0", port=8000)
```

#### Customizing Existing Prefabs

You can also extend and customize the built-in prefabs:

```python
from signalwire_agents.prefabs import InfoGathererAgent

class EnhancedGatherer(InfoGathererAgent):
    def __init__(self, fields, **kwargs):
        super().__init__(fields=fields, **kwargs)
        
        # Add an additional instruction
        self.prompt_add_section("Instructions", bullets=[
            "Verify all information carefully."
        ])
        
        # Add an additional custom tool
        
    @AgentBase.tool(
        name="check_customer", 
        description="Check customer status in database",
        parameters={"email": {"type": "string"}}
    )
    def check_customer(self, args, raw_data):
        # Implementation...
        return SwaigFunctionResult("Customer status: Active")
```

### Best Practices for Prefab Design

1. **Clear Documentation**: Document the purpose, parameters, and extension points
2. **Sensible Defaults**: Provide working defaults that make sense for the use case
3. **Error Handling**: Implement robust error handling with helpful messages
4. **Modular Design**: Keep prefabs focused on a specific use case
5. **Consistent Interface**: Maintain consistent patterns across related prefabs
6. **Extension Points**: Provide clear ways for others to extend your prefab
7. **Configuration Options**: Make all key behaviors configurable

### Making Prefabs Distributable

To create distributable prefabs that can be used across multiple projects:

1. **Package Structure**: Create a proper Python package
2. **Documentation**: Include clear usage examples 
3. **Configuration**: Support both code and file-based configuration
4. **Testing**: Include tests for your prefab
5. **Publishing**: Publish to PyPI or share via GitHub

Example package structure:

```
my-prefab-agents/
├── README.md
├── setup.py
├── examples/
│   └── support_agent_example.py
└── my_prefab_agents/
    ├── __init__.py
    ├── support.py
    ├── retail.py
    └── utils/
        ├── __init__.py
        └── knowledge_base.py
```

## API Reference

### Constructor Parameters

- `name`: Agent name/identifier (required)
- `route`: HTTP route path (default: "/")
- `host`: Host to bind to (default: "0.0.0.0")
- `port`: Port to bind to (default: 3000)
- `basic_auth`: Optional (username, password) tuple
- `use_pom`: Whether to use POM for prompts (default: True)
- `token_expiry_secs`: Security token expiry time (default: 3600)
- `auto_answer`: Auto-answer calls (default: True)
- `record_call`: Record calls (default: False)
- `schema_path`: Optional path to schema.json file
- `suppress_logs`: Whether to suppress structured logs (default: False)

### Prompt Methods

- `prompt_add_section(title, body=None, bullets=None, numbered=False, numbered_bullets=False)`
- `prompt_add_subsection(parent_title, title, body=None, bullets=None)`
- `prompt_add_to_section(title, body=None, bullet=None, bullets=None)`
- `set_prompt_text(prompt_text)` or `set_prompt(prompt_text)`
- `set_post_prompt(prompt_text)`
- `setPersonality(text)` - Convenience method that calls prompt_add_section
- `setGoal(text)` - Convenience method that calls prompt_add_section
- `setInstructions(bullets)` - Convenience method that calls prompt_add_section

### SWAIG Methods

- `@AgentBase.tool(name, description, parameters={}, secure=True, fillers=None)`
- `define_tool(name, description, parameters, handler, secure=True, fillers=None)`
- `set_native_functions(function_names)`
- `add_native_function(function_name)`
- `remove_native_function(function_name)`
- `add_function_include(url, functions, meta_data=None)`

### Configuration Methods

- `add_hint(hint)` and `add_hints(hints)`
- `add_pattern_hint(hint, pattern, replace, ignore_case=False)`
- `add_pronunciation(replace, with_text, ignore_case=False)`
- `add_language(name, code, voice, speech_fillers=None, function_fillers=None, engine=None, model=None)`
- `set_param(key, value)` and `set_params(params_dict)`
- `set_global_data(data_dict)` and `update_global_data(data_dict)`

### State Methods

- `get_state(call_id)`
- `set_state(call_id, data)` 
- `update_state(call_id, data)`
- `clear_state(call_id)`
- `cleanup_expired_state()`

### SIP Routing Methods

- `enable_sip_routing(auto_map=True, path="/sip")`: Enable SIP routing for an agent
- `register_sip_username(sip_username)`: Register a SIP username for an agent
- `auto_map_sip_usernames()`: Automatically register SIP usernames based on agent attributes

#### AgentServer SIP Methods

- `setup_sip_routing(route="/sip", auto_map=True)`: Set up central SIP routing for a server
- `register_sip_username(username, route)`: Map a SIP username to an agent route

### Service Methods

- `serve(host=None, port=None)`: Start the web server
- `as_router()`: Return a FastAPI router for this agent
- `on_swml_request(request_data=None, callback_path=None)`: Customize SWML based on request data and path
- `on_summary(summary, raw_data=None)`: Handle post-prompt summaries
- `on_function_call(name, args, raw_data=None)`: Process SWAIG function calls
- `register_routing_callback(callback_fn, path="/sip")`: Register a callback for custom path routing
- `set_web_hook_url(url)`: Override the default web_hook_url with a supplied URL string
- `set_post_prompt_url(url)`: Override the default post_prompt_url with a supplied URL string

### Endpoint Methods

The SDK provides several endpoints for different purposes:

- Root endpoint (`/`): Serves the main SWML document
- SWAIG endpoint (`/swaig`): Handles SWAIG function calls
- Post-prompt endpoint (`/post_prompt`): Processes conversation summaries
- Check-for-input endpoint (`/check_for_input`): Supports checking for new input from external systems
- Debug endpoint (`/debug`): Serves the SWML document with debug headers
- Debug events endpoint (`/debug_events`): Receives real-time debug events from the AI module (see [Debug Events](#debug-events))
- SIP routing endpoint (configurable, default `/sip`): Handles SIP routing requests

## Testing

The SignalWire AI Agent SDK provides comprehensive testing capabilities through the `swaig-test` CLI tool, which allows you to test agents locally and simulate serverless environments without deployment.

### Local Agent Testing

Test your agents locally before deployment:

```bash
# Discover agents in a file
swaig-test examples/my_agent.py

# List available functions
swaig-test examples/my_agent.py --list-tools

# Test SWAIG functions
swaig-test examples/my_agent.py --exec get_weather --location "New York"

# Generate SWML documents
swaig-test examples/my_agent.py --dump-swml
```

### Serverless Environment Simulation

Test your agents in simulated serverless environments to ensure they work correctly when deployed:

#### AWS Lambda Testing

```bash
# Basic Lambda environment simulation
swaig-test examples/my_agent.py --simulate-serverless lambda --dump-swml

# Test with custom Lambda configuration
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --aws-function-name my-production-function \
  --aws-region us-west-2 \
  --exec my_function --param value

# Test function execution in Lambda context
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --exec get_weather --location "Miami" \
  --full-request
```

#### CGI Environment Testing

```bash
# Test CGI environment
swaig-test examples/my_agent.py --simulate-serverless cgi \
  --cgi-host my-server.com \
  --cgi-https \
  --dump-swml

# Test function in CGI context
swaig-test examples/my_agent.py --simulate-serverless cgi \
  --cgi-host example.com \
  --exec my_function --param value
```

#### Google Cloud Functions Testing

```bash
# Test Cloud Functions environment
swaig-test examples/my_agent.py --simulate-serverless cloud_function \
  --gcp-project my-project \
  --gcp-function-url https://my-function.cloudfunctions.net \
  --dump-swml
```

#### Azure Functions Testing

```bash
# Test Azure Functions environment
swaig-test examples/my_agent.py --simulate-serverless azure_function \
  --azure-env production \
  --azure-function-url https://my-function.azurewebsites.net \
  --exec my_function
```

### Environment Variable Management

Use environment files for consistent testing across different platforms:

```bash
# Create environment file for production testing
cat > production.env << EOF
AWS_LAMBDA_FUNCTION_NAME=prod-my-agent
AWS_REGION=us-east-1
API_KEY=prod_api_key_123
DEBUG=false
TIMEOUT=60
EOF

# Test with environment file
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --env-file production.env \
  --exec critical_function --input "test"

# Override specific variables
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --env-file production.env \
  --env DEBUG=true \
  --dump-swml
```

### Cross-Platform Testing

Test the same agent across multiple platforms to ensure compatibility:

```bash
# Test across all platforms
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

### Webhook URL Verification

The serverless simulation automatically generates platform-appropriate webhook URLs:

| Platform | Example Webhook URL |
|----------|-------------------|
| Lambda (Function URL) | `https://abc123.lambda-url.us-east-1.on.aws/swaig/` |
| Lambda (API Gateway) | `https://api123.execute-api.us-east-1.amazonaws.com/prod/swaig/` |
| CGI | `https://example.com/cgi-bin/agent.cgi/swaig/` |
| Cloud Functions | `https://my-function-abc123.cloudfunctions.net/swaig/` |
| Azure Functions | `https://my-function.azurewebsites.net/swaig/` |

Verify webhook URLs are generated correctly:

```bash
# Check Lambda webhook URL
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --dump-swml --format-json | jq '.sections.main[1].ai.SWAIG.defaults.web_hook_url'

# Check CGI webhook URL
swaig-test examples/my_agent.py --simulate-serverless cgi \
  --cgi-host my-production-server.com \
  --dump-swml --format-json | jq '.sections.main[1].ai.SWAIG.defaults.web_hook_url'
```

### Testing Best Practices

1. **Test locally first**: Always test your agent in local mode before serverless simulation
2. **Test target platforms**: Test on all platforms where you plan to deploy
3. **Use environment files**: Create reusable environment configurations for different stages
4. **Verify webhook URLs**: Ensure URLs are generated correctly for your target platform
5. **Test function execution**: Verify that functions work correctly in serverless context
6. **Use verbose mode**: Enable `--verbose` for debugging environment setup and execution

### Multi-Agent Testing

For files with multiple agents, specify which agent to test:

```bash
# Discover available agents
swaig-test multi_agent_file.py --list-agents

# Test specific agent
swaig-test multi_agent_file.py --agent-class MyAgent --simulate-serverless lambda --dump-swml

# Test different agents across platforms
swaig-test multi_agent_file.py --agent-class AgentA --simulate-serverless lambda --exec function1
swaig-test multi_agent_file.py --agent-class AgentB --simulate-serverless cgi --cgi-host example.com --exec function2
```

For more detailed testing documentation, see the [CLI Guide](cli_guide.md).

## Examples

### Simple Question-Answering Agent

```python
from signalwire_agents import AgentBase
from signalwire_agents.core.function_result import SwaigFunctionResult
from datetime import datetime

class SimpleAgent(AgentBase):
    def __init__(self):
        super().__init__(
            name="simple",
            route="/simple",
            use_pom=True
        )
        
        # Configure agent personality
        self.prompt_add_section("Personality", body="You are a friendly and helpful assistant.")
        self.prompt_add_section("Goal", body="Help users with basic tasks and answer questions.")
        self.prompt_add_section("Instructions", bullets=[
            "Be concise and direct in your responses.",
            "If you don't know something, say so clearly.",
            "Use the get_time function when asked about the current time."
        ])
        
    @AgentBase.tool(
        name="get_time",
        description="Get the current time",
        parameters={}
    )
    def get_time(self, args, raw_data):
        """Get the current time"""
        now = datetime.now()
        formatted_time = now.strftime("%H:%M:%S")
        return SwaigFunctionResult(f"The current time is {formatted_time}")

def main():
    agent = SimpleAgent()
    print("Starting agent server...")
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    agent.run()

if __name__ == "__main__":
    main()
```

### Multi-Language Customer Service Agent

```python
class CustomerServiceAgent(AgentBase):
    def __init__(self):
        super().__init__(
            name="customer-service",
            route="/support",
            use_pom=True
        )
        
        # Configure agent personality
        self.prompt_add_section("Personality", 
                               body="You are a helpful customer service representative for SignalWire.")
        self.prompt_add_section("Knowledge", 
                               body="You can answer questions about SignalWire products and services.")
        self.prompt_add_section("Instructions", bullets=[
            "Greet customers politely",
            "Answer questions about SignalWire products",
            "Use check_account_status when customer asks about their account",
            "Use create_support_ticket for unresolved issues"
        ])
        
        # Add language support
        self.add_language(
            name="English",
            code="en-US",
            voice="en-US-Neural2-F",
            speech_fillers=["Let me think...", "One moment please..."],
            function_fillers=["I'm looking that up...", "Let me check that..."]
        )
        
        self.add_language(
            name="Spanish",
            code="es",
            voice="rime.spore:multilingual",
            speech_fillers=["Un momento por favor...", "Estoy pensando..."]
        )
        
        # Enable languages
        self.set_params({"languages_enabled": True})
        
        # Add company information
        self.set_global_data({
            "company_name": "SignalWire",
            "support_hours": "9am-5pm ET, Monday through Friday",
            "support_email": "support@signalwire.com"
        })
    
    @AgentBase.tool(
        name="check_account_status",
        description="Check the status of a customer's account",
        parameters={
            "account_id": {
                "type": "string",
                "description": "The customer's account ID"
            }
        }
    )
    def check_account_status(self, args, raw_data):
        account_id = args.get("account_id")
        # In a real implementation, this would query a database
        return SwaigFunctionResult(f"Account {account_id} is in good standing.")
    
    @AgentBase.tool(
        name="create_support_ticket",
        description="Create a support ticket for an unresolved issue",
        parameters={
            "issue": {
                "type": "string",
                "description": "Brief description of the issue"
            },
            "priority": {
                "type": "string",
                "description": "Ticket priority",
                "enum": ["low", "medium", "high", "critical"]
            }
        }
    )
    def create_support_ticket(self, args, raw_data):
        issue = args.get("issue", "")
        priority = args.get("priority", "medium")
        
        # Generate a ticket ID (in a real system, this would create a database entry)
        ticket_id = f"TICKET-{hash(issue) % 10000:04d}"
        
        return SwaigFunctionResult(
            f"Support ticket {ticket_id} has been created with {priority} priority. " +
            "A support representative will contact you shortly."
        )

def main():
    agent = CustomerServiceAgent()
    print("Starting customer service agent...")
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    agent.run()

if __name__ == "__main__":
    main()
```

### Dynamic Agent Configuration Examples

For working examples of dynamic agent configuration, see these files in the `examples` directory:

- **`simple_static_agent.py`**: Traditional static configuration approach
- **`simple_dynamic_agent.py`**: Same agent but using dynamic configuration
- **`simple_dynamic_enhanced.py`**: Enhanced version that actually uses request parameters
- **`comprehensive_dynamic_agent.py`**: Advanced multi-tier, multi-industry dynamic agent
- **`custom_path_agent.py`**: Dynamic agent with custom routing path
- **`multi_agent_server.py`**: Multiple specialized dynamic agents on one server

These examples demonstrate the progression from static to dynamic configuration and show real-world use cases like multi-tenant applications, A/B testing, and personalization.

For more examples, see the `examples` directory in the SignalWire AI Agent SDK repository. 

# Build index from the comprehensive concepts guide
sw-search docs/agent_guide.md --output concepts.swsearch

# Build from multiple sources
sw-search docs/agent_guide.md examples README.md --output comprehensive.swsearch

# Traditional directory approach with custom settings
sw-search ./knowledge \
    --output knowledge.swsearch \
    --file-types md,txt,pdf \
    --chunking-strategy sentence \
    --max-sentences-per-chunk 8 \
    --verbose