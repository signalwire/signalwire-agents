# DataMap Guide

The DataMap system allows you to create SWAIG tools that integrate directly with REST APIs without requiring custom webhook endpoints. DataMap tools execute on the SignalWire server, making them simpler to deploy and manage than traditional webhook-based tools.

## Table of Contents

- [Overview](#overview)
- [Basic Usage](#basic-usage)
- [DataMap Builder Pattern](#datamap-builder-pattern)
- [Processing Pipeline](#processing-pipeline)
- [Variable Expansion](#variable-expansion)
- [Webhook Configuration](#webhook-configuration)
- [Array Processing with Foreach](#array-processing-with-foreach)
- [Expression-Based Tools](#expression-based-tools)
- [Error Handling](#error-handling)
- [Helper Functions](#helper-functions)
- [Real-World Examples](#real-world-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

DataMap tools provide a declarative way to define API integrations that run on SignalWire's infrastructure. Instead of creating webhook endpoints, you describe the API call and response processing using JSON configuration that gets executed serverlessly.

### Key Benefits

- **No webhook infrastructure**: Tools run on SignalWire servers
- **Simplified deployment**: No need to expose endpoints  
- **Built-in authentication**: Support for API keys, Bearer tokens, Basic auth
- **Response processing**: Built-in JSON path traversal and array processing
- **Error handling**: Automatic error detection with `error_keys`
- **Pattern matching**: Expression-based responses without API calls

### When to Use DataMap vs Skills vs Custom Tools

- **DataMap**: Simple REST API integrations, no complex processing needed
- **Skills**: Reusable capabilities with complex logic or dependencies
- **Custom Tools**: Full control over webhook handling and processing

## Basic Usage

```python
from signalwire_agents.core.data_map import DataMap
from signalwire_agents.core.function_result import SwaigFunctionResult

class MyAgent(AgentBase):
    def setup_tools(self):
        # Simple weather API integration
        weather_tool = (DataMap('get_weather')
            .description('Get current weather information')
            .parameter('location', 'string', 'City name', required=True)
            .webhook('GET', 'https://api.weather.com/v1/current?key=YOUR_API_KEY&q=${args.location}')
            .output(SwaigFunctionResult('Weather in ${args.location}: ${response.current.condition.text}, ${response.current.temp_f}°F'))
            .error_keys(['error'])
        )
        
        # Register with agent
        self.register_swaig_function(weather_tool.to_swaig_function())
```

## DataMap Builder Pattern

DataMap uses a fluent interface where methods can be chained together:

```python
tool = (DataMap('function_name')
    .description('Tool description')
    .parameter('param', 'string', 'Parameter description', required=True)
    .webhook('POST', 'https://api.example.com/endpoint')
    .body({'data': '${args.param}'})
    .output(SwaigFunctionResult('Result: ${response.value}'))
)
```

## Processing Pipeline

DataMap tools follow this execution order:

1. **Expressions**: Pattern matching against arguments (if defined)
2. **Webhooks**: API calls (if expressions don't match or aren't defined)
3. **Foreach**: Array processing (if webhook returns array and foreach is configured)
4. **Output**: Final response generation

The pipeline stops at the first successful step.

### Webhook Output Structure

**Important**: Outputs are attached to individual webhooks, not at the top level. This allows for:

- **Per-webhook responses**: Each API can have its own output template
- **Sequential fallback**: Try multiple APIs until one succeeds
- **Error handling**: Per-webhook error detection

```python
# Correct: Output inside webhook
tool = (DataMap('get_data')
    .webhook('GET', 'https://api.primary.com/data')
    .output(SwaigFunctionResult('Primary: ${response.value}'))
    .error_keys(['error'])
)

# Multiple webhooks with fallback
tool = (DataMap('search_with_fallback')
    .webhook('GET', 'https://api.fast.com/search?q=${args.query}')
    .output(SwaigFunctionResult('Fast result: ${response.title}'))
    .webhook('GET', 'https://api.comprehensive.com/search?q=${args.query}')
    .output(SwaigFunctionResult('Comprehensive result: ${response.title}'))
    .fallback_output(SwaigFunctionResult('Sorry, all search services are unavailable'))
)
```

### Execution Flow

1. **Try first webhook**: If successful, use its output
2. **Try subsequent webhooks**: If first fails, try next webhook
3. **Fallback output**: If all webhooks fail, use top-level fallback (if defined)
4. **Generic error**: If no fallback defined, return generic error message

## Variable Expansion

DataMap supports powerful variable substitution using `${variable}` syntax:

### Data Store Usage

**global_data** - Call-wide data store that persists throughout the entire call:
- **Purpose**: Store user information, call state, preferences collected during conversation
- **Examples**: `${global_data.customer_name}`, `${global_data.account_type}`, `${global_data.preferred_language}`
- **Seeded by**: Initial SWML configuration, SWAIG actions during the call
- **Shared by**: All functions in the same call
- **NEVER use for**: API keys, passwords, secrets, or sensitive configuration

**meta_data** - Function-scoped data store:
- **Purpose**: Function-specific state and metadata
- **Examples**: `${meta_data.call_id}`, `${meta_data.session_id}`, `${meta_data.retry_count}`
- **Seeded by**: Function definition, SWAIG actions
- **Shared by**: Functions with the same meta_data_token
- **NEVER use for**: Credentials, API keys, or sensitive data

### Available Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `${args.param_name}` | Function arguments | `${args.location}` |
| `${array[0].field}` | API response data (array) | `${array[0].joke}` |
| `${response.field}` | API response data (object) | `${response.status}` |
| `${this.field}` | Current item in foreach | `${this.title}` |
| `${output_key}` | Built string from foreach | `${formatted_results}` |
| `${global_data.key}` | Agent global data | `${global_data.user_id}` |
| `${meta_data.call_id}` | Call metadata | `${meta_data.call_id}` |

### JSON Path Traversal

Variables support nested object access and array indexing:

```python
# Nested objects
'${response.current.condition.text}'

# Array indexing  
'${response.results[0].title}'

# Complex paths
'${response.data.users[2].profile.name}'
```

### Variable Scoping Rules

Understanding when to use different variable types:

| Context | Variable Type | Example | When to Use |
|---------|---------------|---------|-------------|
| **Array APIs** | `${array[0].field}` | `${array[0].joke}` | API returns JSON array `[{...}]` |
| **Object APIs** | `${response.field}` | `${response.temperature}` | API returns JSON object `{...}` |
| **Foreach processing** | `${this.field}` | `${this.title}` | Inside foreach append template |
| **Function args** | `${args.field}` | `${args.location}` | User-provided parameters |
| **Agent data** | `${global_data.field}` | `${global_data.api_key}` | Agent configuration |

```python
# Example: Weather API returns object
{
  "current": {"temp_f": 72, "condition": {"text": "Sunny"}},
  "location": {"name": "New York"}
}
# Use: ${response.current.temp_f}

# Example: Jokes API returns array  
[
  {"joke": "Why did the chicken cross the road?", "category": "classic"}
]
# Use: ${array[0].joke}

# Example: Search API with foreach
{
  "results": [
    {"title": "Result 1", "snippet": "..."},
    {"title": "Result 2", "snippet": "..."}
  ]
}
# Configure foreach and use: ${this.title} in append template
```

### Examples

```python
# URL parameter substitution
.webhook('GET', 'https://api.weather.com/v1/current?key=${global_data.api_key}&q=${args.location}')

# Request body templating
.body({
    'query': '${args.search_term}',
    'user_id': '${global_data.user_id}',
    'limit': 10
})

# Response formatting
.output(SwaigFunctionResult('Found ${response.total_results} results for "${args.query}"'))
```

## Webhook Configuration

### HTTP Methods

```python
# GET request
.webhook('GET', 'https://api.example.com/data?param=${args.value}')

# POST request with JSON body
.webhook('POST', 'https://api.example.com/create')
.body({'name': '${args.name}', 'value': '${args.value}'})

# PUT request with authentication
.webhook('PUT', 'https://api.example.com/update/${args.id}', 
         headers={'Authorization': 'Bearer ${global_data.token}'})
.body({'status': '${args.status}'})
```

### Authentication

```python
# API key in URL
.webhook('GET', 'https://api.service.com/data?key=${global_data.api_key}&q=${args.query}')

# Bearer token
.webhook('POST', 'https://api.service.com/search',
         headers={'Authorization': 'Bearer ${global_data.token}'})

# Basic auth
.webhook('GET', 'https://api.service.com/data',
         headers={'Authorization': 'Basic ${global_data.credentials}'})

# Custom headers
.webhook('GET', 'https://api.service.com/data',
         headers={
             'X-API-Key': '${global_data.api_key}',
             'X-Client-ID': '${global_data.client_id}',
             'Content-Type': 'application/json'
         })
```

## Array Processing with Foreach

The `foreach` mechanism processes arrays by building a concatenated string. It does **not** iterate like JavaScript forEach - instead it builds a single output string from array elements.

### How Foreach Works

1. **input_key**: Specifies which key in the API response contains the array
2. **output_key**: Names the string variable that gets built up  
3. **max**: Limits how many array items to process
4. **append**: Template string that gets evaluated for each item and concatenated

```python
search_tool = (DataMap('search_docs')
    .description('Search documentation')
    .parameter('query', 'string', 'Search query', required=True)
    .webhook('GET', 'https://api.docs.com/search?q=${args.query}')
    .foreach({
        "input_key": "results",           # API response key containing array
        "output_key": "formatted_results", # Name for the built string
        "max": 3,                         # Process max 3 items
        "append": "Document: ${this.title} - ${this.summary}\n"  # Template for each item
    })
    .output(SwaigFunctionResult('Search results for "${args.query}":\n\n${formatted_results}'))
)
```

### Array Response Example

If the API returns:
```json
{
  "results": [
    {"title": "Getting Started", "summary": "Basic setup"},
    {"title": "Advanced Features", "summary": "Complex workflows"}
  ]
}
```

The foreach will build a string in `formatted_results`:
```
Document: Getting Started - Basic setup
Document: Advanced Features - Complex workflows
```

### Foreach vs Direct Array Access

| Approach | Use Case | Example |
|----------|----------|---------|
| **Direct access** | Single array item | `${array[0].joke}` |
| **Foreach** | Multiple items formatted as string | `${formatted_results}` after foreach |

```python
# Direct access - single item from array response
joke_tool = (DataMap('get_joke')
    .webhook('GET', 'https://api.jokes.com/random')
    .output(SwaigFunctionResult('${array[0].joke}'))  # Just first joke
)

# Foreach - multiple items formatted
search_tool = (DataMap('search_all')
    .webhook('GET', 'https://api.search.com/query')
    .foreach({
        "input_key": "results",
        "output_key": "all_results", 
        "max": 5,
        "append": "- ${this.title}: ${this.description}\n"
    })
    .output(SwaigFunctionResult('Found multiple results:\n${all_results}'))
)
```

## Expression-Based Tools

For simple pattern matching without API calls, use expressions. The `expression()` method takes:
- `test_value`: The value to test (typically `'${args.parameter_name}'`)
- `pattern`: The regex pattern to match against
- `output`: The `SwaigFunctionResult` to return on match
- `nomatch_output` (optional): The result to return if pattern doesn't match

```python
file_control = (DataMap('file_control')
    .description('Control file playback')
    .parameter('command', 'string', 'Playback command')
    .parameter('filename', 'string', 'File to control', required=False)
    .expression('${args.command}', r'start.*', SwaigFunctionResult().add_action('start_playback', {'file': '${args.filename}'}))
    .expression('${args.command}', r'stop.*', SwaigFunctionResult().add_action('stop_playback', True))
    .expression('${args.command}', r'pause.*', SwaigFunctionResult().add_action('pause_playback', True))
)
```

### Expression Patterns

```python
# Exact match - test the 'input' argument against literal string
.expression('${args.input}', 'hello', SwaigFunctionResult('Hello response'))

# Case-insensitive regex
.expression('${args.query}', r'(?i)weather.*', SwaigFunctionResult('Weather info'))

# Multiple patterns using alternation
.expression('${args.command}', r'start|begin|play', SwaigFunctionResult().add_action('start', True))
.expression('${args.command}', r'stop|end|pause', SwaigFunctionResult().add_action('stop', True))
```

## Error Handling

Use `error_keys` to detect API errors:

```python
api_tool = (DataMap('check_status')
    .webhook('GET', 'https://api.service.com/status')
    .error_keys(['error', 'message', 'errors'])  # Check for these keys
    .output(SwaigFunctionResult('Status: ${response.status}'))
)
```

If the response contains any of the error keys, the tool will fail gracefully.

## Helper Functions

For common patterns, use convenience functions:

### Simple API Tool

```python
from signalwire_agents.core.data_map import create_simple_api_tool

weather = create_simple_api_tool(
    name='get_weather',
    url='https://api.weather.com/v1/current?key=API_KEY&q=${location}',
    response_template='Weather: ${response.current.condition.text}, ${response.current.temp_f}°F',
    parameters={
        'location': {
            'type': 'string', 
            'description': 'City name', 
            'required': True
        }
    },
    headers={'X-API-Key': 'your-api-key'},
    error_keys=['error']
)
```

### Expression Tool

```python
from signalwire_agents.core.data_map import create_expression_tool

control = create_expression_tool(
    name='media_control',
    patterns={
        r'start|play|begin': SwaigFunctionResult().add_action('start', True),
        r'stop|end|pause': SwaigFunctionResult().add_action('stop', True),
        r'next|skip': SwaigFunctionResult().add_action('next', True)
    },
    parameters={
        'command': {'type': 'string', 'description': 'Control command'}
    }
)
```

## Real-World Examples

### Weather Service

```python
weather_tool = (DataMap('get_weather')
    .description('Get current weather conditions')
    .parameter('location', 'string', 'City and state/country', required=True)
    .parameter('units', 'string', 'Temperature units', enum=['fahrenheit', 'celsius'])
    .webhook('GET', 'https://api.openweathermap.org/data/2.5/weather?q=${args.location}&appid=${global_data.api_key}&units=${"imperial" if args.units == "fahrenheit" else "metric"}')
    .error_keys(['cod', 'message'])
    .output(SwaigFunctionResult('Weather in ${args.location}: ${response.weather[0].description}, ${response.main.temp}°${"F" if args.units == "fahrenheit" else "C"}. Feels like ${response.main.feels_like}°.'))
)
```

### Knowledge Search with Foreach

```python
knowledge_tool = (DataMap('search_knowledge')
    .description('Search company knowledge base')
    .parameter('query', 'string', 'Search query', required=True)
    .parameter('category', 'string', 'Knowledge category', enum=['support', 'sales', 'technical'])
    .webhook('POST', 'https://api.company.com/knowledge/search',
             headers={'Authorization': 'Bearer ${global_data.knowledge_token}'})
    .body({
        'query': '${args.query}',
        'category': '${args.category}',
        'max_results': 5
    })
    .foreach({
        "input_key": "articles",
        "output_key": "article_summaries",
        "max": 3,
        "append": "Article: ${this.title}\nSummary: ${this.summary}\nRelevance: ${this.score}\n\n"
    })
    .output(SwaigFunctionResult('Found articles for "${args.query}":\n\n${article_summaries}'))
)
```

### Joke Service

```python
joke_tool = (DataMap('get_joke')
    .description('Get a random joke')
    .parameter('category', 'string', 'Joke category', 
               enum=['programming', 'dad', 'pun', 'random'])
    .webhook('GET', 'https://api.jokes.com/v1/joke?category=${args.category}&format=json')
    .output(SwaigFunctionResult('Here\'s a ${args.category} joke: ${response.setup} ... ${response.punchline}'))
    .error_keys(['error'])
    .fallback_output(SwaigFunctionResult('Sorry, the joke service is currently unavailable. Please try again later.'))
)
```

### API with Array Response

```python
# For APIs that return arrays, use ${array[0].field} syntax for single items
joke_ninja_tool = (DataMap('get_joke')
    .description('Get a random joke from API Ninjas')
    .parameter('type', 'string', 'Type of joke', enum=['jokes', 'dadjokes'])
    .webhook('GET', 'https://api.api-ninjas.com/v1/${args.type}',
             headers={'X-Api-Key': '${global_data.api_key}'})
    .output(SwaigFunctionResult('Here\'s a joke: ${array[0].joke}'))
    .error_keys(['error'])
    .fallback_output(SwaigFunctionResult('Sorry, there is a problem with the joke service right now. Please try again later.'))
)
```

### Multi-Step Fallback

```python
# Try multiple APIs with fallback
search_tool = (DataMap('web_search')
    .description('Search the web')
    .parameter('query', 'string', 'Search query', required=True)
    # Primary API
    .webhook('GET', 'https://api.primary.com/search?q=${args.query}&key=${global_data.primary_key}')
    # Fallback API
    .webhook('GET', 'https://api.fallback.com/search?query=${args.query}&token=${global_data.fallback_token}')
    .output(SwaigFunctionResult('Search results for "${args.query}": ${response.results[0].title} - ${response.results[0].snippet}'))
)
```

## Best Practices

### 1. Keep It Simple

DataMap is best for straightforward API integrations. For complex logic, use Skills or custom tools:

```python
# Good: Simple API call
.webhook('GET', 'https://api.service.com/data?id=${args.id}')

# Consider alternatives: Complex multi-step processing
# (Better handled by Skills or custom tools)
```

### 2. Use Global Data for Secrets

Store API keys and tokens in global data, not hardcoded:

```python
# Good
.webhook('GET', 'https://api.service.com/data?key=${global_data.api_key}')

# Bad
.webhook('GET', 'https://api.service.com/data?key=hardcoded-key')
```

### 3. Provide Clear Parameter Descriptions

```python
# Good
.parameter('location', 'string', 'City name or ZIP code (e.g., "New York" or "10001")', required=True)

# Bad  
.parameter('location', 'string', 'location', required=True)
```

### 4. Handle Errors Gracefully

```python
# Always include error handling
.error_keys(['error', 'message', 'status'])
.fallback_output(SwaigFunctionResult('Service temporarily unavailable'))
```

### 5. Use Foreach for Multiple Items

```python
# Good: Use foreach for multiple formatted results
.foreach({
    "input_key": "results",
    "output_key": "formatted_list", 
    "append": "- ${this.title}: ${this.summary}\n"
})

# Less optimal: Only showing first result
.output(SwaigFunctionResult('First result: ${response.results[0].title}'))
```

## Troubleshooting

### Common Issues

#### 1. Variable Not Expanding

**Problem**: Variables like `${args.location}` showing up as literal text.

**Solutions**:
- Check variable name matches parameter name exactly
- Ensure proper `${variable}` syntax 
- Verify the variable is in scope for that context

#### 2. Array Access Not Working

**Problem**: `${array[0].field}` returns undefined.

**Solutions**:
- Verify API actually returns an array `[{...}]`
- Check if API returns object with array property: use `${response.arrayfield[0].field}`
- For multiple items, consider using foreach instead

#### 3. Foreach Not Processing Arrays

**Problem**: Foreach output is empty or not working.

**Solutions**:
- Check `input_key` matches the actual API response structure
- Verify array exists and has items: `"input_key": "results"` for `{"results": [...]}`
- Ensure `append` template uses `${this.property}` syntax
- Check `max` value isn't zero

#### 4. Authentication Failures  

**Problem**: API returns 401/403 errors.

**Solutions**:
- Verify API key/token is correct in global_data
- Check header format matches API requirements
- Test API credentials outside of DataMap first

#### 5. Error Keys Not Working

**Problem**: Tool doesn't detect API errors properly.

**Solutions**:
- Check actual API error response structure
- Add all possible error field names to `error_keys`
- Use `fallback_output` for generic error handling

### Debug Tools

Enable debug mode to see variable expansion:

```python
debug_tool = (DataMap('debug_echo')
    .parameter('test', 'string', 'Test parameter')
    .output(SwaigFunctionResult('Input: ${args.test}, All args: ${args}'))
)

# Test variable expansion
test_variables = (DataMap('test_vars')
    .parameter('location', 'string', 'Location')
    .webhook('GET', 'https://httpbin.org/get?location=${args.location}')
    .output(SwaigFunctionResult('URL was: ${response.url}, Args: ${response.args}'))
)
```

This comprehensive guide should help you understand and effectively use the DataMap system for creating REST API integrations in your SignalWire agents.

---

## Related Documentation

- **[API Reference](signalwire_agents_api_reference.md)** - Complete DataMap class API reference
- **[SWAIG Function Result Methods](swaig_function_result_methods.md)** - All available result methods for DataMap outputs
- **[Agent Guide](agent_guide.md)** - General agent development including SWAIG functions
- **[DataMap SWML Complete Guide](datamap_swml_complete_guide.md)** - Extended DataMap and SWML integration

### Example Files

- `examples/data_map_demo.py` - Comprehensive DataMap patterns demo
- `examples/advanced_datamap_demo.py` - Advanced DataMap with webhooks and foreach
- `examples/datasphere_serverless_demo.py` - Serverless DataMap integration 