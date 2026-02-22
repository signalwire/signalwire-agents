# SignalWire SWML Service Guide

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Centralized Logging System](#centralized-logging-system)
- [SWML Document Creation](#swml-document-creation)
- [Verb Handling](#verb-handling)
- [Web Service Features](#web-service-features)
- [Custom Routing Callbacks](#custom-routing-callbacks)
- [Advanced Usage](#advanced-usage)
- [API Reference](#api-reference)
- [Examples](#examples)

## Introduction

The `SWMLService` class provides a foundation for creating and serving SignalWire Markup Language (SWML) documents. It serves as the base class for all SignalWire services, including AI Agents, and handles common tasks such as:

- SWML document creation and manipulation
- Schema validation
- Web service functionality
- Authentication
- Centralized logging

The class is designed to be extended for specific use cases, while providing a full set of capabilities out of the box.

## Installation

The `SWMLService` class is part of the SignalWire AI Agent SDK. Install it using pip:

```bash
pip install signalwire-agents
```

## Basic Usage

Here's a simple example of creating an SWML service:

```python
from signalwire_agents.core.swml_service import SWMLService

class SimpleVoiceService(SWMLService):
    def __init__(self, host="0.0.0.0", port=3000):
        super().__init__(
            name="voice-service",
            route="/voice",
            host=host,
            port=port
        )
        
        # Build the SWML document
        self.build_document()
    
    def build_document(self):
        # Reset the document to start fresh
        self.reset_document()
        
        # Add answer verb
        self.add_verb("answer", {})
        
        # Add play verb for greeting
        self.add_verb("play", {
            "url": "say:Hello, thank you for calling our service."
        })
        
        # Add hangup verb
        self.add_verb("hangup", {})

# Create and start the service
service = SimpleVoiceService()
service.run()
```

## Centralized Logging System

The `SWMLService` class includes a centralized logging system based on `structlog` that provides structured, JSON-formatted logs. This logging system is automatically set up when you import the module, so you don't need to configure it in each service or example.

### How It Works

1. When `swml_service.py` is imported, it configures `structlog` (if not already configured)
2. Each `SWMLService` instance gets a logger bound to its service name
3. All logs include contextual information like service name, timestamp, and log level
4. Logs are formatted as JSON for easy parsing and analysis

### Using the Logger

Every `SWMLService` instance has a `log` attribute that can be used for logging:

```python
# Basic logging
self.log.info("service_started")

# Logging with context
self.log.debug("document_created", size=len(document))

# Error logging
try:
    # Some operation
    pass
except Exception as e:
    self.log.error("operation_failed", error=str(e))
```

### Log Levels

The following log levels are available (in increasing order of severity):
- `debug`: Detailed information for debugging
- `info`: General information about operation
- `warning`: Warning about potential issues
- `error`: Error information when operations fail
- `critical`: Critical error that might cause the application to terminate

### Suppressing Logs

To suppress logs when running a service, you can set the log level:

```python
import logging
logging.getLogger().setLevel(logging.WARNING)  # Only show warnings and above
```

You can also pass `suppress_logs=True` when initializing an agent or service:

```python
service = SWMLService(
    name="my-service",
    suppress_logs=True
)
```

## SWML Document Creation

The `SWMLService` class provides methods for creating and manipulating SWML documents.

### Document Structure

SWML documents have the following basic structure:

```json
{
  "version": "1.0.0",
  "sections": {
    "main": [
      { "verb1": { /* configuration */ } },
      { "verb2": { /* configuration */ } }
    ],
    "section1": [
      { "verb3": { /* configuration */ } }
    ]
  }
}
```

### Document Methods

- `reset_document()`: Reset the document to an empty state
- `add_verb(verb_name, config)`: Add a verb to the main section
- `add_section(section_name)`: Add a new section
- `add_verb_to_section(section_name, verb_name, config)`: Add a verb to a specific section
- `get_document()`: Get the current document as a dictionary
- `render_document()`: Get the current document as a JSON string

### Common Verb Shortcuts

- `add_verb(verb_name, config)`: Add any SWML verb with configuration

## Verb Handling

The `SWMLService` class provides validation for SWML verbs using the SignalWire schema.

### Verb Validation

When adding a verb, the service validates it against the schema to ensure it has the correct structure and parameters.

```python
# This will validate the configuration against the schema
self.add_verb("play", {
    "url": "say:Hello, world!",
    "volume": 5
})

# This would fail validation (invalid parameter)
self.add_verb("play", {
    "invalid_param": "value"
})
```

### Custom Verb Handlers

You can register custom verb handlers for specialized verb processing:

```python
from signalwire_agents.core.swml_handler import SWMLVerbHandler

class CustomPlayHandler(SWMLVerbHandler):
    def __init__(self):
        super().__init__("play")
    
    def validate_config(self, config):
        # Custom validation logic
        return True, []
    
    def build_config(self, **kwargs):
        # Custom configuration building
        return kwargs

service.register_verb_handler(CustomPlayHandler())
```

## Web Service Features

The `SWMLService` class includes built-in web service capabilities for serving SWML documents.

### Endpoints

By default, a service provides the following endpoints:

- `GET /route`: Return the SWML document
- `POST /route`: Process request data and return the SWML document
- `GET /route/`: Same as above but with trailing slash
- `POST /route/`: Same as above but with trailing slash

Where `route` is the route path specified when creating the service.

### Authentication

Basic authentication is automatically set up for all endpoints. Credentials are generated if not provided, or can be specified:

```python
service = SWMLService(
    name="my-service",
    basic_auth=("username", "password")
)
```

You can also set credentials using environment variables:
- `SWML_BASIC_AUTH_USER`
- `SWML_BASIC_AUTH_PASSWORD`

### Dynamic SWML Generation

You can override the `on_swml_request` method to customize SWML documents based on request data:

```python
def on_swml_request(self, request_data=None):
    if not request_data:
        return None
        
    # Customize document based on request_data
    self.reset_document()
    self.add_answer_verb()
    
    # Add custom verbs based on request_data
    if request_data.get("caller_type") == "vip":
        self.add_verb("play", {
            "url": "say:Welcome VIP caller!"
        })
    else:
        self.add_verb("play", {
            "url": "say:Welcome caller!"
        })
    
    # Return modifications to the document
    # or None to use the document we've built without modifications
    return None
```

## Custom Routing Callbacks

The `SWMLService` class allows you to register custom routing callbacks that can examine incoming requests and determine where they should be routed.

### Registering a Routing Callback

You can use the `register_routing_callback` method to register a function that will be called to process requests to a specific path:

```python
def my_routing_callback(request, body):
    """
    Process incoming requests and determine routing
    
    Args:
        request: FastAPI Request object
        body: Parsed JSON body as a dictionary
        
    Returns:
        Optional[str]: If a string is returned, the request will be redirected to that URL.
                      If None is returned, the request will be processed normally.
    """
    # Example: Route based on a field in the request body
    if "customer_id" in body:
        customer_id = body["customer_id"]
        return f"/customer/{customer_id}"
    
    # Process request normally
    return None

# Register the callback for a specific path
service.register_routing_callback(my_routing_callback, path="/customer")
```

### How Routing Works

1. When a request is received at the registered path, the routing callback is executed
2. The callback inspects the request and can decide whether to redirect it
3. If the callback returns a URL string, the request is redirected with HTTP 307 (temporary redirect)
4. If the callback returns `None`, the request is processed normally by the `on_request` method

### Serving Different Content for Different Paths

You can use the `callback_path` parameter passed to `on_request` to serve different content for different paths:

```python
def on_request(self, request_data=None, callback_path=None):
    """
    Called when SWML is requested
    
    Args:
        request_data: Optional dictionary containing the parsed POST body
        callback_path: Optional callback path from the request
        
    Returns:
        Optional dict to modify/augment the SWML document
    """
    # Serve different content based on the callback path
    if callback_path == "/customer":
        return {
            "sections": {
                "main": [
                    {"answer": {}},
                    {"play": {"url": "say:Welcome to customer service!"}}
                ]
            }
        }
    elif callback_path == "/product":
        return {
            "sections": {
                "main": [
                    {"answer": {}},
                    {"play": {"url": "say:Welcome to product support!"}}
                ]
            }
        }
    
    # Default content
    return None
```

### Example: Multi-Section Service

Here's an example of a service that uses routing callbacks to handle different types of requests:

```python
from signalwire_agents.core.swml_service import SWMLService
from fastapi import Request
from typing import Dict, Any, Optional

class MultiSectionService(SWMLService):
    def __init__(self):
        super().__init__(
            name="multi-section",
            route="/main"
        )
        
        # Create the main document
        self.reset_document()
        self.add_verb("answer", {})
        self.add_verb("play", {"url": "say:Hello from the main service!"})
        self.add_verb("hangup", {})
        
        # Register customer and product routes
        self.register_customer_route()
        self.register_product_route()
    
    def register_customer_route(self):
        def customer_callback(request: Request, body: Dict[str, Any]) -> Optional[str]:
            # Check if we need to route to a specific customer ID
            if "customer_id" in body:
                customer_id = body["customer_id"]
                # In a real implementation, you might redirect to another service
                # Here we just log it and process normally
                print(f"Processing request for customer ID: {customer_id}")
            return None
            
        # Register the callback at the /customer path
        self.register_routing_callback(customer_callback, path="/customer")
        
        # Create the customer SWML section
        self.add_section("customer_section")
        self.add_verb_to_section("customer_section", "answer", {})
        self.add_verb_to_section("customer_section", "play", 
                                {"url": "say:Welcome to customer service!"})
        self.add_verb_to_section("customer_section", "hangup", {})
    
    def register_product_route(self):
        def product_callback(request: Request, body: Dict[str, Any]) -> Optional[str]:
            # Check if we need to route to a specific product ID
            if "product_id" in body:
                product_id = body["product_id"]
                print(f"Processing request for product ID: {product_id}")
            return None
            
        # Register the callback at the /product path
        self.register_routing_callback(product_callback, path="/product")
        
        # Create the product SWML section
        self.add_section("product_section")
        self.add_verb_to_section("product_section", "answer", {})
        self.add_verb_to_section("product_section", "play", 
                               {"url": "say:Welcome to product support!"})
        self.add_verb_to_section("product_section", "hangup", {})
    
    def on_request(self, request_data=None, callback_path=None):
        # Serve different content based on the callback path
        if callback_path == "/customer":
            return {
                "sections": {
                    "main": self.get_document()["sections"]["customer_section"]
                }
            }
        elif callback_path == "/product":
            return {
                "sections": {
                    "main": self.get_document()["sections"]["product_section"]
                }
            }
        return None
```

In this example:
1. The service registers two custom route paths: `/customer` and `/product`
2. Each path has its own callback function to handle routing decisions
3. The `on_request` method uses the `callback_path` to determine which content to serve
4. Different SWML sections are served for different paths

## Advanced Usage

### Creating a FastAPI Router

You can get a FastAPI router for the service to include in a larger application:

```python
from fastapi import FastAPI

app = FastAPI()
service = SWMLService(name="my-service")
router = service.as_router()
app.include_router(router, prefix="/voice")
```

### Schema Path Customization

You can specify a custom path to the schema file:

```python
service = SWMLService(
    name="my-service",
    schema_path="/path/to/schema.json"
)
```

## API Reference

### Constructor Parameters

- `name`: Service name/identifier (required)
- `route`: HTTP route path (default: "/")
- `host`: Host to bind to (default: "0.0.0.0")
- `port`: Port to bind to (default: 3000)
- `basic_auth`: Optional tuple of (username, password)
- `schema_path`: Optional path to schema.json
- `suppress_logs`: Whether to suppress structured logs (default: False)

### Document Methods

- `reset_document()`
- `add_verb(verb_name, config)`
- `add_section(section_name)`
- `add_verb_to_section(section_name, verb_name, config)`
- `get_document()`
- `render_document()`

### Service Methods

- `as_router()`: Get a FastAPI router for the service
- `run()`: Start the service
- `stop()`: Stop the service
- `get_basic_auth_credentials(include_source=False)`: Get the basic auth credentials
- `on_swml_request(request_data=None)`: Called when SWML is requested
- `register_routing_callback(callback_fn, path="/sip")`: Register a callback for request routing

### Verb Helper Methods

- `add_verb(verb_name, config)`: Add any SWML verb with configuration

## Examples

### Basic Voicemail Service

```python
from signalwire_agents.core.swml_service import SWMLService

class VoicemailService(SWMLService):
    def __init__(self, host="0.0.0.0", port=3000):
        super().__init__(
            name="voicemail",
            route="/voicemail",
            host=host,
            port=port
        )
        
        # Build the SWML document
        self.build_voicemail_document()
    
    def build_voicemail_document(self):
        """Build the voicemail SWML document"""
        # Reset the document
        self.reset_document()
        
        # Add answer verb
        self.add_verb("answer", {})
        
        # Add play verb for greeting
        self.add_verb("play", {
            "url": "say:Hello, you've reached the voicemail service. Please leave a message after the beep."
        })
        
        # Play a beep
        self.add_verb("play", {
            "url": "https://example.com/beep.wav"
        })
        
        # Record the message
        self.add_verb("record", {
            "format": "mp3",
            "stereo": False,
            "max_length": 120,  # 2 minutes max
            "terminators": "#"
        })
        
        # Thank the caller
        self.add_verb("play", {
            "url": "say:Thank you for your message. Goodbye!"
        })
        
        # Hang up
        self.add_verb("hangup", {})
        
        self.log.debug("voicemail_document_built")
```

### Dynamic Call Routing Service

```python
class CallRouterService(SWMLService):
    def on_swml_request(self, request_data=None):
        # If there's no request data, use default routing
        if not request_data:
            self.log.debug("no_request_data_using_default")
            return None
        
        # Create a new document
        self.reset_document()
        self.add_verb("answer", {})
        
        # Get routing parameters
        department = request_data.get("department", "").lower()
        
        # Add play verb for greeting
        self.add_verb("play", {
            "url": f"say:Thank you for calling our {department} department. Please hold."
        })
        
        # Route based on department
        phone_numbers = {
            "sales": "+15551112222",
            "support": "+15553334444",
            "billing": "+15555556666"
        }
        
        # Get the appropriate number or use default
        to_number = phone_numbers.get(department, "+15559990000")
        
        # Connect to the department
        self.add_verb("connect", {
            "to": to_number,
            "timeout": 30,
            "answer_on_bridge": True
        })
        
        # Add fallback message and hangup
        self.add_verb("play", {
            "url": "say:We're sorry, but all of our agents are currently busy. Please try again later."
        })
        self.add_verb("hangup", {})
        
        return None  # Use the document we've built
```

For more examples, see the `examples` directory in the SignalWire AI Agent SDK repository. 