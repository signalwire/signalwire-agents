# DataMap Complete Guide: SWML Perspective and Development

A comprehensive guide to understanding, implementing, and testing DataMap configurations in SignalWire AI Agents from the SWML (SignalWire Markup Language) perspective.

## Table of Contents

### 1. Introduction to DataMap in SWML
- [1.1 What is DataMap](#11-what-is-datamap)
- [1.2 DataMap vs Traditional Webhooks](#12-datamap-vs-traditional-webhooks)
- [1.3 SWML Integration Overview](#13-swml-integration-overview)
- [1.4 When to Use DataMap](#14-when-to-use-datamap)

### 2. DataMap Architecture and Processing Pipeline
- [2.1 Server-Side Processing Flow](#21-server-side-processing-flow)
- [2.2 Processing Order: Expressions → Webhooks → Foreach → Output](#22-processing-order-expressions--webhooks--foreach--output)
- [2.3 Context and Variable Scope](#23-context-and-variable-scope)
- [2.4 Serverless Execution Model](#24-serverless-execution-model)

### 3. DataMap Configuration Structure
- [3.1 Basic DataMap Schema](#31-basic-datamap-schema)
- [3.2 Function-Level Configuration](#32-function-level-configuration)
- [3.3 Nested DataMap Objects](#33-nested-datamap-objects)
- [3.4 Parameter Validation](#34-parameter-validation)

### 4. Template Expansion System
- [4.1 Template Syntax Overview](#41-template-syntax-overview)
- [4.2 Variable Types and Sources](#42-variable-types-and-sources)
- [4.3 Array and Object Access Patterns](#43-array-and-object-access-patterns)
- [4.4 Context-Specific Variables](#44-context-specific-variables)
- [4.5 Template Expansion Examples](#45-template-expansion-examples)

### 5. Webhook Configuration and HTTP Processing
- [5.1 Webhook Structure](#51-webhook-structure)
- [5.2 HTTP Methods and Headers](#52-http-methods-and-headers)
- [5.3 Request Body Construction](#53-request-body-construction)
- [5.4 Sequential Webhook Processing](#54-sequential-webhook-processing)
- [5.5 Webhook Failure Detection](#55-webhook-failure-detection)

### 6. Response Processing and Data Handling
- [6.1 Response Data Structure](#61-response-data-structure)
- [6.2 Array vs Object Response Handling](#62-array-vs-object-response-handling)
- [6.3 Error Response Processing](#63-error-response-processing)
- [6.4 Custom Error Keys](#64-custom-error-keys)

### 7. Foreach Processing and Array Iteration
- [7.1 Foreach Configuration](#71-foreach-configuration)
- [7.2 Array Data Sources](#72-array-data-sources)
- [7.3 Template Expansion in Foreach](#73-template-expansion-in-foreach)
- [7.4 String Concatenation and Formatting](#74-string-concatenation-and-formatting)
- [7.5 Foreach Limitations and Best Practices](#75-foreach-limitations-and-best-practices)

### 8. Output Generation and Result Formatting
- [8.1 Webhook-Level Output](#81-webhook-level-output)
- [8.2 DataMap-Level Fallback Output](#82-datamap-level-fallback-output)
- [8.3 Response vs Action Outputs](#83-response-vs-action-outputs)
- [8.4 SWML Action Generation](#84-swml-action-generation)

### 9. Skills System Integration
- [9.1 DataMap Skills vs Raw Configuration](#91-datamap-skills-vs-raw-configuration)
- [9.2 Skill-Based DataMap Creation](#92-skill-based-datamap-creation)
- [9.3 Skill Configuration Patterns](#93-skill-configuration-patterns)
- [9.4 Multi-Instance Skill Usage](#94-multi-instance-skill-usage)

### 10. Practical Examples and Use Cases
- [10.1 API Integration Examples](#101-api-integration-examples)
- [10.2 Knowledge Base Search](#102-knowledge-base-search)
- [10.3 External Service Integration](#103-external-service-integration)
- [10.4 Multi-Step Processing Workflows](#104-multi-step-processing-workflows)

### 11. Development and Testing
- [11.1 Local Development Setup](#111-local-development-setup)
- [11.2 Environment Variable Configuration](#112-environment-variable-configuration)
- [11.3 CLI Testing Tools](#113-cli-testing-tools)
- [11.4 Debugging DataMap Execution](#114-debugging-datamap-execution)

### 12. Advanced Patterns and Best Practices
- [12.1 Multiple Webhook Fallback Chains](#121-multiple-webhook-fallback-chains)
- [12.2 Complex Template Expressions](#122-complex-template-expressions)
- [12.3 Dynamic API Endpoint Selection](#123-dynamic-api-endpoint-selection)
- [12.4 Response Transformation Patterns](#124-response-transformation-patterns)

### 13. Error Handling and Reliability
- [13.1 HTTP Error Codes and Handling](#131-http-error-codes-and-handling)
- [13.2 Network Timeout and Retry Logic](#132-network-timeout-and-retry-logic)
- [13.3 Graceful Degradation Strategies](#133-graceful-degradation-strategies)
- [13.4 Monitoring and Observability](#134-monitoring-and-observability)

### 14. Security and Best Practices
- [14.1 API Key Management](#141-api-key-management)
- [14.2 Secure Header Configuration](#142-secure-header-configuration)
- [14.3 Input Validation and Sanitization](#143-input-validation-and-sanitization)
- [14.4 Rate Limiting Considerations](#144-rate-limiting-considerations)

### 15. Performance Optimization
- [15.1 Request Optimization](#151-request-optimization)
- [15.2 Response Size Management](#152-response-size-management)
- [15.3 Caching Strategies](#153-caching-strategies)
- [15.4 Execution Time Considerations](#154-execution-time-considerations)

### 16. Migration and Upgrade Paths
- [16.1 From Webhook to DataMap Migration](#161-from-webhook-to-datamap-migration)
- [16.2 Legacy Configuration Support](#162-legacy-configuration-support)
- [16.3 Version Compatibility](#163-version-compatibility)
- [16.4 Gradual Migration Strategies](#164-gradual-migration-strategies)

---

*This guide provides comprehensive coverage of DataMap functionality within the SignalWire AI Agents framework, from basic concepts to advanced implementation patterns.*

## 1. Introduction to DataMap in SWML

### 1.1 What is DataMap

DataMap is a serverless function execution system within SignalWire AI Agents that enables integration with external APIs without the need for custom webhook endpoints. Unlike traditional webhook-based SWAIG functions that require you to host and maintain HTTP endpoints, DataMap functions are executed entirely within the SignalWire infrastructure.

**Key Characteristics:**
- **Serverless Architecture**: No need to host webhook endpoints
- **Built-in HTTP Client**: Native HTTP request capabilities
- **Template-Based Configuration**: Declarative API integration using template expansion
- **Sequential Processing**: Multiple webhook fallback support
- **Response Transformation**: Built-in data processing and formatting
- **Error Handling**: Automatic failure detection and fallback mechanisms

**DataMap Execution Flow:**
```
Function Call → Template Expansion → HTTP Request → Response Processing → Output Generation
```

### 1.2 DataMap vs Traditional Webhooks

| Aspect | Traditional Webhooks | DataMap |
|--------|---------------------|---------|
| **Infrastructure** | Requires hosted endpoints | Serverless execution |
| **Configuration** | Code-based handlers | Declarative JSON/YAML |
| **HTTP Requests** | Manual implementation | Built-in HTTP client |
| **Error Handling** | Custom error logic | Automatic failure detection |
| **Template Expansion** | Manual string formatting | Native template system |
| **Scalability** | Limited by hosting infrastructure | Auto-scaling serverless |
| **Maintenance** | Server maintenance required | Zero maintenance overhead |
| **Development Speed** | Slower (code + deploy) | Faster (configuration only) |

**Traditional Webhook Example:**
```python
def search_knowledge(args, post_data):
    # Custom HTTP request logic
    response = requests.post("https://api.example.com/search", 
                           json={"query": args["query"]})
    # Custom error handling
    if response.status_code != 200:
        return {"error": "API request failed"}
    # Custom response processing
    data = response.json()
    return {"response": f"Found: {data['results'][0]['text']}"}
```

**DataMap Equivalent:**
```json
{
  "function": "search_knowledge",
  "data_map": {
    "webhooks": [{
      "url": "https://api.example.com/search",
      "method": "POST",
      "headers": {"Content-Type": "application/json"},
      "params": {"query": "${args.query}"},
      "output": {"response": "Found: ${array[0].text}"},
      "error_keys": ["error"]
    }],
    "output": {"response": "Search service unavailable"}
  }
}
```

### 1.3 SWML Integration Overview

DataMap integrates with SWML (SignalWire Markup Language) through the AI verb's function calling mechanism. When an AI agent needs to call a function, SWML automatically detects whether it's a traditional webhook or DataMap function and routes the execution appropriately.

**SWML AI Verb Integration:**
```xml
<ai>
  <prompt>You can search knowledge using the search_knowledge function</prompt>
  <SWAIG>
    <function name="search_knowledge" data_map="..." />
  </SWAIG>
</ai>
```

**Execution Context:**
- DataMap functions have access to all SWML context variables
- Function arguments are automatically validated against parameter schemas
- Results can generate both response text and SWML actions
- Global data and prompt variables are available for template expansion

**Integration Benefits:**
- **Declarative Configuration**: Define API integrations using configuration, not code
- **Automatic Validation**: Parameter validation based on JSON schema
- **Context Awareness**: Access to conversation state and SWML variables
- **Action Generation**: Can produce SWML actions for call control
- **Error Recovery**: Built-in fallback mechanisms maintain conversation flow

### 1.4 When to Use DataMap

**Ideal Use Cases:**
- **External API Integration**: REST API calls to third-party services
- **Knowledge Base Queries**: Search operations against document stores
- **Data Transformation**: Simple data processing and formatting
- **Service Aggregation**: Combining data from multiple sources
- **Rapid Prototyping**: Quick API integration without infrastructure

**DataMap is Perfect For:**
- Simple to moderate API integration complexity
- Read-heavy operations (GET, POST with JSON)
- Services with predictable response formats
- Scenarios requiring fallback mechanisms
- Development teams without DevOps infrastructure

**Consider Traditional Webhooks When:**
- Complex business logic is required
- Advanced error handling and retry mechanisms needed
- Custom authentication schemes beyond headers
- Heavy computational processing required
- Integration with non-HTTP protocols
- Need for persistent state or caching
- Complex response transformation logic

**Hybrid Approach:**
Many applications benefit from using both DataMap and traditional webhooks:
- DataMap for simple API calls and data retrieval
- Traditional webhooks for complex processing and business logic
- DataMap for rapid prototyping, webhooks for production optimization

## 2. DataMap Architecture and Processing Pipeline

### 2.1 Server-Side Processing Flow

DataMap execution occurs entirely within the SignalWire infrastructure, following a deterministic processing pipeline implemented in the server-side `mod_openai.c` module. Understanding this flow is crucial for effective DataMap configuration.

**Server-Side Architecture:**
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   SWML Engine   │────│   DataMap        │────│   HTTP Client   │
│   Function Call │    │   Processor      │    │   Request       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                       ┌──────▼──────┐
                       │   Template   │
                       │   Expansion  │
                       │   Engine     │
                       └─────────────┘
```

**Processing Modules:**
- **Function Router**: Determines if function is DataMap or webhook
- **Context Builder**: Assembles variable context from function arguments and SWML state
- **Template Engine**: Expands variables in URLs, headers, and request bodies
- **HTTP Client**: Executes HTTP requests with timeout and error handling
- **Response Processor**: Parses and validates HTTP responses
- **Foreach Engine**: Processes array data with template expansion
- **Output Generator**: Formats final results for SWML consumption

### 2.2 Processing Order: Expressions → Webhooks → Foreach → Output

DataMap processing follows a strict sequential order that ensures deterministic execution and proper error handling:

**1. Expression Processing (Optional)**
```json
{
  "expressions": [
    {
      "pattern": "simple query",
      "output": {"response": "This is a simple response for: ${args.query}"}
    }
  ]
}
```
- Pattern matching against function arguments
- Early exit if pattern matches
- Bypasses HTTP requests for known cases

**2. Webhook Sequential Processing**
```json
{
  "webhooks": [
    {"url": "https://primary-api.com/search", "...": "..."},
    {"url": "https://fallback-api.com/search", "...": "..."}
  ]
}
```
- Process webhooks in array order
- Stop at first successful webhook
- Each webhook has independent configuration

**3. Foreach Processing (Per Successful Webhook)**
```json
{
  "foreach": {
    "input_key": "results",
    "output_key": "formatted_results",
    "max": 5,
    "append": "Result: ${this.title}\n"
  }
}
```
- Processes array data from successful webhook response
- Builds concatenated strings using template expansion
- Stores result in context for output templates

**4. Output Generation**
```json
{
  "output": {
    "response": "Found results: ${formatted_results}",
    "action": [{"SWML": {"version": "1.0.0", "...": "..."}}]
  }
}
```
- Webhook-level output (if webhook succeeds)
- DataMap-level fallback output (if all webhooks fail)

**Processing Flow Diagram:**
```
Function Call
     │
     ▼
┌─────────────┐     Yes    ┌──────────────┐
│ Expressions │────────────│ Return Early │
│   Match?    │            │   Output     │
└─────────────┘            └──────────────┘
     │ No
     ▼
┌─────────────┐
│  Webhook 1  │────┐
│   Success?  │    │
└─────────────┘    │
     │ No          │ Yes
     ▼             ▼
┌─────────────┐  ┌─────────────┐
│  Webhook 2  │  │   Foreach   │
│   Success?  │  │ Processing  │
└─────────────┘  └─────────────┘
     │ No          │
     ▼             ▼
┌─────────────┐  ┌─────────────┐
│  Fallback   │  │   Webhook   │
│   Output    │  │   Output    │
└─────────────┘  └─────────────┘
```

### 2.3 Context and Variable Scope

DataMap maintains a hierarchical context system that provides access to various data sources during template expansion:

**Context Hierarchy:**
```
┌────────────────────────────────────────┐
│                args                    │ ← Function arguments
│  ┌──────────────────────────────────┐  │
│  │            response              │  │ ← HTTP response object
│  │  ┌────────────────────────────┐  │  │
│  │  │          this             │  │  │ ← Current foreach item
│  │  │                           │  │  │
│  │  └────────────────────────────┘  │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘
```

**Variable Sources:**

1. **Function Arguments** (`args.*`)
   - Direct access to function call parameters
   - Available throughout entire execution
   - Example: `${args.query}`, `${args.filters}`

2. **HTTP Response Data** (`response.*` or `array.*`)
   - Response object for object responses
   - Array data for array responses
   - Available after successful webhook execution

3. **Global Data** (`global_data.*`)
   - Agent-level configuration and state
   - SWML prompt variables
   - Conversation context

4. **Foreach Context** (`this.*`)
   - Current item during foreach processing
   - Only available within foreach append templates
   - Dynamic scope based on array iteration

**Context Evolution:**
```javascript
// Initial context
{
  "args": {"query": "SignalWire", "count": 3}
}

// After webhook success (object response)
{
  "args": {"query": "SignalWire", "count": 3},
  "response": {"results": [...], "total": 25}
}

// After webhook success (array response)
{
  "args": {"query": "SignalWire", "count": 3},
  "array": [{"title": "...", "text": "..."}, ...]
}

// During foreach processing
{
  "args": {"query": "SignalWire", "count": 3},
  "array": [...],
  "this": {"title": "Current Item", "text": "Current content"}
}
```

### 2.4 Serverless Execution Model

DataMap functions execute in a serverless environment with specific characteristics and limitations:

**Execution Environment:**
- **Stateless**: No persistent memory between function calls
- **Isolated**: Each function execution is independent
- **Time-Limited**: HTTP requests have built-in timeouts
- **Resource-Constrained**: Optimized for typical API integration scenarios

**Execution Lifecycle:**
```
1. Function Call Received
   ├── Parse DataMap configuration
   ├── Validate function arguments
   └── Build initial context

2. Template Expansion
   ├── Expand webhook URLs and headers
   ├── Expand request body parameters
   └── Prepare HTTP request configuration

3. HTTP Request Execution
   ├── Make HTTP request with timeouts
   ├── Handle network errors
   └── Parse response (JSON/text)

4. Response Processing
   ├── Validate response structure
   ├── Check for error conditions
   └── Add response to context

5. Foreach Processing (if configured)
   ├── Extract array data
   ├── Iterate with template expansion
   └── Build concatenated result

6. Output Generation
   ├── Expand output templates
   ├── Format final response
   └── Return to SWML engine
```

**Performance Characteristics:**
- **Cold Start**: First execution may have slight latency
- **Warm Execution**: Subsequent calls are optimized
- **Concurrency**: Multiple functions can execute simultaneously
- **Scalability**: Automatic scaling based on demand

**Resource Limits:**
- HTTP request timeout: ~30 seconds
- Response size limits: Reasonable API response sizes
- Memory constraints: Optimized for typical API responses
- Concurrent execution: Platform-managed scaling

## 3. DataMap Configuration Structure

### 3.1 Basic DataMap Schema

DataMap configurations follow a specific JSON schema that defines how external APIs are integrated and how responses are processed. Understanding this schema is essential for creating effective DataMap functions.

**Complete DataMap Function Structure:**
```json
{
  "function": "function_name",
  "description": "Human-readable function description",
  "parameters": {
    "type": "object",
    "properties": {
      "param_name": {
        "type": "string|number|boolean|array|object",
        "description": "Parameter description",
        "required": true,
        "enum": ["optional", "enumeration", "values"]
      }
    },
    "required": ["param1", "param2"]
  },
  "data_map": {
    "expressions": [...],
    "webhooks": [...],
    "output": {...}
  }
}
```

**Core Schema Elements:**

1. **Function Metadata**
   - `function`: Unique function identifier
   - `description`: Human-readable description for AI understanding
   - `parameters`: JSON Schema for function arguments validation

2. **DataMap Configuration** (`data_map`)
   - `expressions`: Optional pattern-based early return logic
   - `webhooks`: Array of HTTP request configurations
   - `output`: Fallback output when all webhooks fail

**Minimal DataMap Example:**
```json
{
  "function": "simple_api_call",
  "description": "Call external API",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Search query"}
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [{
      "url": "https://api.example.com/search",
      "method": "GET",
      "headers": {"Authorization": "Bearer ${global_data.api_token}"},
      "params": {"q": "${args.query}"},
      "output": {"response": "Result: ${response.data}"}
    }]
  }
}
```

### 3.2 Function-Level Configuration

Function-level configuration defines the interface between the AI agent and the DataMap execution engine:

**Parameter Schema Definition:**
```json
{
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query text",
        "minLength": 1,
        "maxLength": 500
      },
      "category": {
        "type": "string",
        "description": "Content category filter",
        "enum": ["docs", "api", "tutorials", "blog"],
        "default": "docs"
      },
      "limit": {
        "type": "integer",
        "description": "Maximum number of results",
        "minimum": 1,
        "maximum": 20,
        "default": 5
      },
      "filters": {
        "type": "array",
        "description": "Additional search filters",
        "items": {"type": "string"},
        "maxItems": 10
      }
    },
    "required": ["query"],
    "additionalProperties": false
  }
}
```

**Validation Features:**
- **Type Validation**: Ensures correct data types
- **Range Validation**: Min/max values for numbers and arrays
- **Enumeration**: Restricts to specific allowed values
- **Required Fields**: Ensures essential parameters are provided
- **Default Values**: Automatic parameter population
- **Additional Properties**: Controls extra parameter handling

**AI Integration Benefits:**
```json
{
  "description": "Search the knowledge base for documentation and tutorials. Use specific keywords and categories for better results.",
  "parameters": {
    "properties": {
      "query": {
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides"
      }
    }
  }
}
```

### 3.3 Nested DataMap Objects

DataMap configurations can include nested objects and complex data structures for advanced use cases:

**Complex Webhook Configuration:**
```json
{
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

**Nested Structure Benefits:**
- **Complex API Integration**: Support for sophisticated API requirements
- **Conditional Logic**: Dynamic parameter construction based on arguments
- **Rich Response Processing**: Multiple output types and formatting
- **SWML Action Generation**: Create call control actions from API responses

### 3.4 Parameter Validation

Parameter validation ensures data integrity and provides clear error messages when function calls fail validation:

**Comprehensive Validation Example:**
```json
{
  "parameters": {
    "type": "object",
    "properties": {
      "email": {
        "type": "string",
        "description": "Email address to validate",
        "pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$",
        "maxLength": 254
      },
      "user_preferences": {
        "type": "object",
        "description": "User preference settings",
        "properties": {
          "language": {
            "type": "string",
            "enum": ["en", "es", "fr", "de"],
            "default": "en"
          },
          "notifications": {
            "type": "object",
            "properties": {
              "email": {"type": "boolean", "default": true},
              "sms": {"type": "boolean", "default": false},
              "push": {"type": "boolean", "default": true}
            }
          }
        }
      },
      "metadata": {
        "type": "object",
        "description": "Additional metadata",
        "additionalProperties": true,
        "maxProperties": 20
      }
    },
    "required": ["email"],
    "dependencies": {
      "user_preferences": {
        "properties": {
          "email": {"const": true}
        }
      }
    }
  }
}
```

**Validation Error Handling:**
When validation fails, the AI agent receives clear error messages:

```json
{
  "error": "Parameter validation failed",
  "details": [
    {
      "parameter": "email",
      "message": "Invalid email format",
      "received": "invalid-email"
    },
    {
      "parameter": "limit",
      "message": "Value must be between 1 and 20",
      "received": 50
    }
  ]
}
```

**Best Practices for Parameter Design:**
1. **Clear Descriptions**: Help the AI understand parameter purpose
2. **Appropriate Constraints**: Balance flexibility with validation
3. **Sensible Defaults**: Reduce required parameters where possible
4. **Enum Values**: Provide clear options for categorical parameters
5. **Nested Structure**: Organize complex parameters logically

## 4. Template Expansion System

### 4.1 Template Syntax Overview

Template expansion allows you to dynamically construct URLs, headers, and request bodies based on function arguments and context variables. Understanding the syntax and usage is essential for effective DataMap configuration.

**Template Syntax:**
```
${expression}
```

**Expression Types:**
- **Variable**: `${args.query}`, `${global_data.api_token}`
- **Function**: `${function_name(args)}`
- **Conditional**: `${if(condition, true_value, false_value)}`
- **Array Access**: `${array[index].property}`
- **Object Access**: `${object.property}`
- **Built-in Functions**: `${length(array)}`, `${now()}`

### 4.2 Variable Types and Sources

DataMap supports a variety of variable types and sources that can be accessed during template expansion:

**Variable Sources:**
- **Function Arguments** (`args.*`)
- **HTTP Response Data** (`response.*` or `array.*`)
- **Global Data** (`global_data.*`)
- **Foreach Context** (`this.*`)

### 4.3 Array and Object Access Patterns

DataMap provides flexible access patterns for array and object data:

**Array Access:**
```
${array[index].property}
```

**Object Access:**
```
${object.property}
```

### 4.4 Context-Specific Variables

DataMap provides context-specific variables that can be used in template expansion:

**Context-Specific Variables:**
- **Function Arguments**: `${args.query}`, `${args.filters}`
- **HTTP Response Data**: `${response.data}`, `${array[0].text}`
- **Global Data**: `${global_data.api_token}`, `${global_data.prompt_variables}`
- **Foreach Context**: `${this.title}`, `${this.text}`

### 4.5 Template Expansion Examples

**Simple Template Expansion:**
```
{
  "function": "simple_api_call",
  "description": "Call external API",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Search query"}
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [{
      "url": "https://api.example.com/search",
      "method": "GET",
      "headers": {"Authorization": "Bearer ${global_data.api_token}"},
      "params": {"q": "${args.query}"},
      "output": {"response": "Result: ${response.data}"}
    }]
  }
}
```

**Complex Template Expansion:**
```
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

## 5. Webhook Configuration and HTTP Processing

### 5.1 Webhook Structure

DataMap functions can be configured with multiple webhooks to handle different scenarios and provide fallback mechanisms:

**Webhook Configuration:**
```json
{
  "webhooks": [
    {"url": "https://primary-api.com/search", "...": "..."},
    {"url": "https://fallback-api.com/search", "...": "..."}
  ]
}
```

### 5.2 HTTP Methods and Headers

DataMap functions can use various HTTP methods and headers to customize request configurations:

**HTTP Method Examples:**
- **GET**: Retrieving data from a server
- **POST**: Sending data to a server for processing
- **PUT**: Updating existing data on a server
- **DELETE**: Removing data from a server

**HTTP Header Examples:**
- **Authorization**: Used for authentication and access control
- **Content-Type**: Specifies the format of the request body
- **Accept**: Specifies the format of the response body
- **X-Request-ID**: Used for request tracking and correlation

### 5.3 Request Body Construction

DataMap functions can construct request bodies dynamically based on function arguments and context variables:

**Request Body Examples:**
- **Simple Query**: `${args.query}`
- **Complex Query**: `${function_name(args)}`
- **JSON Object**: `${json_object}`
- **Formatted String**: `${formatted_string}`

### 5.4 Sequential Webhook Processing

DataMap functions can be configured to process multiple webhooks in sequence:

**Sequential Webhook Configuration:**
```json
{
  "webhooks": [
    {"url": "https://primary-api.com/search", "...": "..."},
    {"url": "https://fallback-api.com/search", "...": "..."}
  ]
}
```

### 5.5 Webhook Failure Detection

DataMap functions can be configured to handle webhook failures and provide fallback mechanisms:

**Webhook Failure Configuration:**
```json
{
  "webhooks": [
    {"url": "https://primary-api.com/search", "...": "..."},
    {"url": "https://fallback-api.com/search", "...": "..."}
  ]
}
```

## 6. Response Processing and Data Handling

### 6.1 Response Data Structure

DataMap functions can return various types of response data:

**Response Data Types:**
- **Text**: Simple text response
- **JSON**: Structured data in JSON format
- **Array**: List of data items
- **Object**: Key-value pairs

### 6.2 Array vs Object Response Handling

DataMap functions can handle both array and object responses:

**Array Response Example:**
```json
{
  "response": {
    "results": [{"title": "...", "text": "..."}, ...]
  }
}
```

**Object Response Example:**
```json
{
  "response": {
    "results": {"total": 25, "data": [...]}
  }
}
```

### 6.3 Error Response Processing

DataMap functions can handle errors and provide clear error messages:

**Error Handling Example:**
```json
{
  "error": "API request failed",
  "details": {
    "status_code": 400,
    "message": "Invalid request parameters"
  }
}
```

### 6.4 Custom Error Keys

DataMap functions can define custom error keys to provide more detailed error information:

**Custom Error Keys Example:**
```json
{
  "error": "API request failed",
  "details": {
    "status_code": 400,
    "message": "Invalid request parameters"
  }
}
```

## 7. Foreach Processing and Array Iteration

### 7.1 Foreach Configuration

DataMap functions can be configured to process array data:

**Foreach Configuration Example:**
```json
{
  "foreach": {
    "input_key": "results",
    "output_key": "formatted_results",
    "max": 5,
    "append": "Result: ${this.title}\n"
  }
}
```

### 7.2 Array Data Sources

DataMap functions can use various array data sources:

**Array Data Sources:**
- **Function Results**: `${response.results}`
- **Global Data**: `${global_data.array}`
- **Foreach Context**: `${this.array}`

### 7.3 Template Expansion in Foreach

DataMap functions can expand template variables within foreach append templates:

**Foreach Template Expansion Example:**
```json
{
  "foreach": {
    "input_key": "results",
    "output_key": "formatted_results",
    "max": 5,
    "append": "Result: ${this.title}\n"
  }
}
```

### 7.4 String Concatenation and Formatting

DataMap functions can concatenate and format string data:

**String Concatenation Example:**
```json
{
  "response": {
    "formatted_results": "Result: ${this.title}\n"
  }
}
```

### 7.5 Foreach Limitations and Best Practices

DataMap functions should be used cautiously when processing large arrays:

**Foreach Limitations:**
- **Performance**: Processing large arrays can be slow
- **Memory**: Large arrays can consume significant memory

**Best Practices:**
1. **Limit Array Size**: Use pagination or limit parameters
2. **Optimize Template Expansion**: Minimize array access in templates
3. **Use Foreach with Caution**: Only use when necessary

## 8. Output Generation and Result Formatting

### 8.1 Webhook-Level Output

DataMap functions can return output directly from webhook responses:

**Webhook Output Example:**
```json
{
  "response": {
    "formatted_results": "Result: ${this.title}\n"
  }
}
```

### 8.2 DataMap-Level Fallback Output

DataMap functions can provide a fallback output when all webhooks fail:

**Fallback Output Example:**
```json
{
  "output": {
    "response": "Search service unavailable"
  }
}
```

### 8.3 Response vs Action Outputs

DataMap functions can return both response text and SWML actions:

**Response vs Action Output Example:**
```json
{
  "response": {
    "formatted_results": "Result: ${this.title}\n"
  },
  "action": [{"SWML": {"version": "1.0.0", "...": "..."}}]
}
```

### 8.4 SWML Action Generation

DataMap functions can generate SWML actions for call control:

**SWML Action Generation Example:**
```json
{
  "action": [{"SWML": {"version": "1.0.0", "...": "..."}}]
}
```

## 9. Skills System Integration

### 9.1 DataMap Skills vs Raw Configuration

DataMap functions can be integrated with the skills system:

**Skills System Integration Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

### 9.2 Skill-Based DataMap Creation

DataMap functions can be created based on skills:

**Skill-Based DataMap Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

### 9.3 Skill Configuration Patterns

DataMap functions can be configured based on skill patterns:

**Skill Configuration Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

### 9.4 Multi-Instance Skill Usage

DataMap functions can be used across multiple instances:

**Multi-Instance Skill Usage Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

## 10. Practical Examples and Use Cases

### 10.1 API Integration Examples

DataMap functions can be used for various API integration scenarios:

**API Integration Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

### 10.2 Knowledge Base Search

DataMap functions can be used for knowledge base searches:

**Knowledge Base Search Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

### 10.3 External Service Integration

DataMap functions can be used for external service integrations:

**External Service Integration Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

### 10.4 Multi-Step Processing Workflows

DataMap functions can be used for multi-step processing workflows:

**Multi-Step Processing Workflow Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

## 11. Development and Testing

### 11.1 Local Development Setup

DataMap functions can be developed and tested locally:

**Local Development Setup Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

### 11.2 Environment Variable Configuration

DataMap functions can be configured with environment variables:

**Environment Variable Configuration Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

### 11.3 CLI Testing Tools

DataMap functions can be tested using command-line tools:

**CLI Testing Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

### 11.4 Debugging DataMap Execution

DataMap functions can be debugged using various tools:

**Debugging Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

## 12. Advanced Patterns and Techniques

### 12.0 Helper Functions

For common patterns, convenience functions simplify DataMap creation:

#### Simple API Tool

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

#### Expression Tool

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

### 12.1 Multiple Webhook Fallback Chains

DataMap functions can be configured with multiple webhooks to handle different scenarios and provide fallback mechanisms:

**Multiple Webhook Configuration Example:**
```json
{
  "webhooks": [
    {"url": "https://primary-api.com/search", "...": "..."},
    {"url": "https://fallback-api.com/search", "...": "..."}
  ]
}
```

### 12.2 Complex Template Expressions

DataMap functions can use complex template expressions to handle advanced scenarios:

**Complex Template Expression Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

### 12.3 Dynamic API Endpoint Selection

DataMap functions can dynamically select API endpoints based on function arguments:

**Dynamic API Endpoint Selection Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

### 12.4 Response Transformation Patterns

DataMap functions can transform response data based on function arguments:

**Response Transformation Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

## 13. Error Handling and Reliability

### 13.1 HTTP Error Codes and Handling

DataMap functions can handle various HTTP error codes:

**HTTP Error Handling Example:**
```json
{
  "error": "API request failed",
  "details": {
    "status_code": 400,
    "message": "Invalid request parameters"
  }
}
```

### 13.2 Network Timeout and Retry Logic

DataMap functions can handle network timeouts and implement retry logic:

**Network Timeout and Retry Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

### 13.3 Graceful Degradation Strategies

DataMap functions can implement graceful degradation strategies:

**Graceful Degradation Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

### 13.4 Monitoring and Observability

DataMap functions can be monitored and observed using various tools:

**Monitoring and Observability Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

## 14. Security and Best Practices

### 14.1 API Key Management

DataMap functions can be configured with API keys:

**API Key Management Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

### 14.2 Secure Header Configuration

DataMap functions can be configured with secure headers:

**Secure Header Configuration Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

### 14.3 Input Validation and Sanitization

DataMap functions should validate and sanitize input data:

**Input Validation Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

### 14.4 Rate Limiting Considerations

DataMap functions should consider rate limiting:

**Rate Limiting Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

## 15. Performance Optimization

### 15.1 Request Optimization

DataMap functions can optimize request configurations:

**Request Optimization Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

### 15.2 Response Size Management

DataMap functions can manage response sizes:

**Response Size Management Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

### 15.3 Caching Strategies

DataMap functions can implement caching strategies:

**Caching Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

### 15.4 Execution Time Considerations

DataMap functions should consider execution time:

**Execution Time Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

## 16. Migration and Upgrade Paths

### 16.1 From Webhook to DataMap Migration

DataMap functions can be migrated from traditional webhooks:

**Migration Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

### 16.2 Legacy Configuration Support

DataMap functions can support legacy configurations:

**Legacy Configuration Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

### 16.3 Version Compatibility

DataMap functions should be compatible with different versions:

**Version Compatibility Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

### 16.4 Gradual Migration Strategies

DataMap functions can implement gradual migration strategies:

**Gradual Migration Example:**
```json
{
  "function": "search_knowledge",
  "description": "Search the knowledge base for documentation and tutorials",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Specific search terms - be precise for better results"
      },
      "category": {
        "type": "string",
        "description": "Content type: 'docs' for documentation, 'api' for API references, 'tutorials' for guides",
        "enum": ["docs", "api", "tutorials"],
        "default": "docs"
      }
    },
    "required": ["query"]
  },
  "data_map": {
    "webhooks": [
      {
        "url": "https://api.primary.com/v2/search",
        "method": "POST",
        "headers": {
          "Content-Type": "application/json",
          "Authorization": "Bearer ${global_data.primary_token}",
          "X-Client-Version": "2.1",
          "X-Request-ID": "${args.request_id}"
        },
        "params": {
          "query": {
            "text": "${args.query}",
            "filters": {
              "category": "${args.category}",
              "date_range": {
                "start": "${args.start_date}",
                "end": "${args.end_date}"
              },
              "tags": "${args.tags}"
            },
            "options": {
              "highlight": true,
              "max_results": "${args.limit}",
              "include_metadata": true
            }
          }
        },
        "foreach": {
          "input_key": "results",
          "output_key": "formatted_results",
          "max": 10,
          "append": "## ${this.title}\n${this.excerpt}\n**Score:** ${this.relevance_score}\n\n"
        },
        "output": {
          "response": "Found ${response.total} results:\n\n${formatted_results}",
          "action": [
            {
              "SWML": {
                "version": "1.0.0",
                "sections": {
                  "main": [
                    {
                      "set": {
                        "last_search_query": "${args.query}",
                        "last_search_results": "${response.total}",
                        "search_timestamp": "${response.timestamp}"
                      }
                    }
                  ]
                }
              }
            }
          ]
        },
        "error_keys": ["error", "message", "detail"]
      }
    ],
    "output": {
      "response": "I'm sorry, the search service is currently unavailable. Please try again later."
    }
  }
}
```

---

*This guide provides comprehensive coverage of DataMap functionality within the SignalWire AI Agents framework, from basic concepts to advanced implementation patterns.*

## Related Documentation

- **[API Reference](api_reference.md)** - Complete DataMap class API reference
- **[SWAIG Reference](swaig_reference.md)** - SWAIG function results and actions
- **[Agent Guide](agent_guide.md)** - General agent development including SWAIG functions
- **[Contexts Guide](contexts_guide.md)** - Structured workflows with function restrictions