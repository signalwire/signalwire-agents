# Lesson 4: Advanced Features and Best Practices

In this lesson, you'll master advanced features of the SignalWire Agents SDK and learn production deployment best practices. We'll cover custom SWAIG functions, error handling, debugging techniques, and deployment strategies.

## Table of Contents

1. [Custom SWAIG Functions](#custom-swaig-functions)
2. [Error Handling and Results](#error-handling-and-results)
3. [Logging and Debugging](#logging-and-debugging)
4. [Production Deployment](#production-deployment)
5. [Testing Strategies](#testing-strategies)
6. [Performance Optimization](#performance-optimization)
7. [Security Best Practices](#security-best-practices)
8. [Summary](#summary)

---

## Custom SWAIG Functions

SWAIG (SignalWire AI Gateway) functions allow your agent to perform actions beyond conversation. These can integrate with APIs, databases, or perform calculations.

### Basic Function Structure

```python
from signalwire_agents import AgentBase
from signalwire_agents.core.function_result import SwaigFunctionResult

class MyAgent(AgentBase):
    def __init__(self):
        super().__init__(name="My Agent", route="/")
        
        # Define a function using the decorator
        @self.tool(
            "calculate_price",
            description="Calculate total price with tax",
            parameters={
                "amount": {"type": "number", "description": "Base price"},
                "tax_rate": {"type": "number", "description": "Tax rate (default 8%)"}
            }
        )
        def calculate_price(self, args, raw_data):
            """Calculate total price including tax"""
            amount = args.get("amount", 0)
            tax_rate = args.get("tax_rate", 0.08)
            tax = amount * tax_rate
            total = amount + tax

            return SwaigFunctionResult(
                f"The total price is ${total:.2f} "
                f"(${amount:.2f} + ${tax:.2f} tax)"
            )
```

### Function Parameters

**Required Parameters:**

```python
@self.tool(
    "create_order",
    description="Create a new order",
    parameters={
        "customer_name": {"type": "string", "description": "Customer's full name"},
        "items": {"type": "string", "description": "Items to order"},
        "priority": {"type": "string", "description": "Order priority level"}
    },
    required=["customer_name", "items"]  # These params are required
)
def create_order(self, args, raw_data):
    customer_name = args.get("customer_name")
    items = args.get("items")
    priority = args.get("priority", "normal")
    # Implementation
```

**Parameter Types:**

```python
# Supported parameter types in the parameters dict
parameters={
    "text": {"type": "string", "description": "A text value"},
    "number": {"type": "integer", "description": "An integer value"},
    "decimal": {"type": "number", "description": "A decimal value"},
    "flag": {"type": "boolean", "description": "A true/false value"}
}
```

### Advanced Function Results

The `SwaigFunctionResult` class provides rich responses:

```python
@self.tool(
    "check_inventory",
    description="Check product availability",
    parameters={
        "product_id": {"type": "string", "description": "The product ID to check"}
    }
)
def check_inventory(self, args, raw_data):
    product_id = args.get("product_id")
    # Simulate inventory check
    in_stock = 5

    if in_stock > 0:
        result = SwaigFunctionResult(f"Product {product_id} is in stock ({in_stock} units)")

        # Add structured data
        result.add_data({
            "product_id": product_id,
            "quantity": in_stock,
            "warehouse": "main"
        })

        # Add action for the agent
        result.add_action("set_global_data", {
            "last_checked_product": product_id,
            "stock_level": in_stock
        })

        return result
    else:
        # Return error state
        return SwaigFunctionResult(
            f"Product {product_id} is out of stock",
            error=True
        )
```

### Async vs Sync Functions

Both patterns are supported:

```python
# Async function (recommended for I/O operations)
@self.tool(
    "fetch_data",
    description="Fetch data from API",
    parameters={
        "query": {"type": "string", "description": "Search query"}
    }
)
async def fetch_data(self, args, raw_data):
    query = args.get("query")
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/{query}") as resp:
            data = await resp.json()
            return SwaigFunctionResult(f"Found {len(data)} results")

# Sync function (for quick calculations)
@self.tool(
    "calculate",
    description="Perform calculation",
    parameters={
        "x": {"type": "integer", "description": "First number"},
        "y": {"type": "integer", "description": "Second number"}
    }
)
def calculate(self, args, raw_data):
    x = args.get("x", 0)
    y = args.get("y", 0)
    return SwaigFunctionResult(f"Result: {x + y}")
```

---

## Error Handling and Results

Proper error handling ensures your agent gracefully handles failures.

### Function Error Handling

```python
@self.tool(
    "process_order",
    description="Process customer order",
    parameters={
        "order_id": {"type": "string", "description": "The order ID to process"}
    }
)
def process_order(self, args, raw_data):
    order_id = args.get("order_id")
    try:
        # Validate input
        if not order_id or len(order_id) < 5:
            return SwaigFunctionResult(
                "Invalid order ID format",
                error=True
            )

        # Simulate processing
        if order_id.startswith("TEST"):
            raise ValueError("Test orders cannot be processed")

        # Success case
        return SwaigFunctionResult(f"Order {order_id} processed successfully")

    except ValueError as e:
        return SwaigFunctionResult(
            f"Order processing failed: {str(e)}",
            error=True
        )
    except Exception as e:
        # Log unexpected errors
        logger.error(f"Unexpected error processing order: {e}")
        return SwaigFunctionResult(
            "An unexpected error occurred. Please try again.",
            error=True
        )
```

### Agent-Level Error Handling

```python
class RobustAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Robust Agent", route="/")
        
        # Add error handling instructions to prompt
        self.prompt_add_section(
            "Error Handling",
            body="How to handle errors gracefully:",
            bullets=[
                "If a function returns an error, acknowledge it politely",
                "Offer alternative solutions when possible",
                "Never expose technical error details to customers",
                "Always maintain a helpful, professional tone"
            ]
        )
```

### Validation Patterns

```python
@self.tool(
    "update_customer",
    description="Update customer information",
    parameters={
        "customer_id": {"type": "string", "description": "The customer's ID"},
        "email": {"type": "string", "description": "Customer's email address"},
        "phone": {"type": "string", "description": "Customer's phone number"}
    }
)
def update_customer(self, args, raw_data):
    customer_id = args.get("customer_id")
    email = args.get("email")
    phone = args.get("phone")
    # Input validation
    errors = []

    if not customer_id:
        errors.append("Customer ID is required")

    if email and "@" not in email:
        errors.append("Invalid email format")

    if phone and len(phone) < 10:
        errors.append("Phone number must be at least 10 digits")

    if errors:
        return SwaigFunctionResult(
            f"Validation failed: {', '.join(errors)}",
            error=True
        )

    # Process valid input
    return SwaigFunctionResult("Customer updated successfully")
```

---

## Logging and Debugging

Effective logging is crucial for troubleshooting and monitoring.

### Using the Logger

```python
from signalwire_agents.core.logging_config import get_logger

logger = get_logger(__name__)

class DebugAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Debug Agent", route="/")
        logger.info("Initializing Debug Agent")

        @self.tool(
            "debug_function",
            description="Test function with logging",
            parameters={
                "param": {"type": "string", "description": "Parameter to process"}
            }
        )
        def debug_function(self, args, raw_data):
            param = args.get("param", "")
            logger.debug(f"Function called with param: {param}")

            try:
                # Some operation
                result = param.upper()
                logger.info(f"Operation successful: {result}")
                return SwaigFunctionResult(result)

            except Exception as e:
                logger.error(f"Operation failed: {e}", exc_info=True)
                return SwaigFunctionResult("Operation failed", error=True)
```

### Log Levels

```python
# Set log level when creating server
server = AgentServer(log_level="debug")

# Or via environment variable
export SWML_LOG_LEVEL=debug

# Available levels:
# - debug: Detailed information for debugging
# - info: General information (default)
# - warning: Warning messages
# - error: Error messages only
# - critical: Critical issues only
```

### Debugging Techniques

**1. Request Logging:**

```python
def configure_dynamic(self, query_params, body_params, headers, agent):
    logger.debug(f"Query params: {query_params}")
    logger.debug(f"Body params: {body_params}")
    logger.debug(f"Headers: {headers}")
```

**2. SWML Inspection:**

```bash
# Dump SWML without running
swaig-test agent.py --dump-swml

# Test specific functions
swaig-test agent.py --exec function_name --param value
```

**3. Interactive Debugging:**

```python
@self.tool(
    "debug_state",
    description="Debug agent state",
    parameters={}
)
def debug_state(self, args, raw_data):
    import json
    state = {
        "agent_name": self.get_name(),
        "functions": list(self._tools.keys()),
        "languages": self._languages
    }
    logger.info(f"Agent state: {json.dumps(state, indent=2)}")
    return SwaigFunctionResult("State logged to console")
```

---

## Production Deployment

### Environment Variables

```bash
# Core configuration
export SWML_AUTH_USER=produser
export SWML_AUTH_PASS=strongpassword
export SWML_LOG_LEVEL=info

# SSL configuration
export SWML_SSL_ENABLED=true
export SWML_SSL_CERT_PATH=/etc/ssl/certs/agent.crt
export SWML_SSL_KEY_PATH=/etc/ssl/private/agent.key
export SWML_DOMAIN=agents.example.com

# Performance tuning
export PYTORCH_DISABLE_AVX512=1  # For compatibility
export WORKERS=4  # Number of worker processes
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Run agent
CMD ["python", "agent.py"]
```

### Systemd Service

```ini
# /etc/systemd/system/signalwire-agent.service
[Unit]
Description=SignalWire AI Agent
After=network.target

[Service]
Type=simple
User=agent
WorkingDirectory=/opt/signalwire-agent
Environment="SWML_LOG_LEVEL=info"
Environment="SWML_SSL_ENABLED=true"
ExecStart=/usr/bin/python3 /opt/signalwire-agent/agent.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Health Monitoring

```python
# Add custom health checks
@server.app.get("/health/detailed")
async def detailed_health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents": {
            "triage": "active",
            "sales": "active",
            "support": "active"
        },
        "checks": {
            "database": check_database_connection(),
            "search_index": check_search_index(),
            "memory_usage": get_memory_usage()
        }
    }
```

---

## Testing Strategies

### Unit Testing Functions

```python
import pytest
from signalwire_agents.core.function_result import SwaigFunctionResult

@pytest.mark.asyncio
async def test_calculate_price():
    # Create agent instance
    agent = MyAgent()
    
    # Get the function
    calc_func = agent._tools["calculate_price"]["function"]
    
    # Test normal case
    result = await calc_func(amount=100.0, tax_rate=0.08)
    assert isinstance(result, SwaigFunctionResult)
    assert "108.00" in result.message
    
    # Test edge cases
    result = await calc_func(amount=0, tax_rate=0)
    assert "0.00" in result.message
```

### Integration Testing

```python
# test_integration.py
import requests
import json

def test_agent_swml_generation():
    """Test that agent generates valid SWML"""
    response = requests.get("http://localhost:3000/")
    assert response.status_code == 200
    
    swml = response.json()
    assert "ai" in swml
    assert "prompt" in swml["ai"]
    assert "voice" in swml["ai"]

def test_function_execution():
    """Test function execution via swaig-test"""
    import subprocess
    result = subprocess.run(
        ["swaig-test", "agent.py", "--exec", "calculate_price", "--amount", "100"],
        capture_output=True,
        text=True
    )
    assert "108.00" in result.stdout
```

### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 http://localhost:3000/

# Using curl in a loop
for i in {1..100}; do
    curl -s http://localhost:3000/ > /dev/null &
done
wait
```

---

## Performance Optimization

### Caching Strategies

```python
from functools import lru_cache
import asyncio

class OptimizedAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Optimized Agent", route="/")
        self._cache = {}

        @self.tool(
            "get_product_info",
            description="Get product information",
            parameters={
                "product_id": {"type": "string", "description": "The product ID to look up"}
            }
        )
        async def get_product_info(self, args, raw_data):
            product_id = args.get("product_id")
            # Check cache first
            if product_id in self._cache:
                logger.debug(f"Cache hit for {product_id}")
                return SwaigFunctionResult(self._cache[product_id])

            # Expensive operation
            info = await fetch_from_database(product_id)

            # Cache for 5 minutes
            self._cache[product_id] = info
            asyncio.create_task(self._expire_cache(product_id, 300))

            return SwaigFunctionResult(info)

    async def _expire_cache(self, key: str, seconds: int):
        await asyncio.sleep(seconds)
        self._cache.pop(key, None)
```

### Async Best Practices

```python
# Good: Concurrent operations
@self.tool(
    "get_full_info",
    description="Get complete information",
    parameters={
        "customer_id": {"type": "string", "description": "The customer ID"}
    }
)
async def get_full_info(self, args, raw_data):
    customer_id = args.get("customer_id")
    # Run multiple queries concurrently
    orders, profile, preferences = await asyncio.gather(
        get_orders(customer_id),
        get_profile(customer_id),
        get_preferences(customer_id)
    )

    return SwaigFunctionResult(f"Found {len(orders)} orders")

# Bad: Sequential operations (don't do this)
async def get_full_info_slow(customer_id):
    orders = await get_orders(customer_id)  # Waits
    profile = await get_profile(customer_id)  # Then waits
    preferences = await get_preferences(customer_id)  # Then waits
```

### Memory Management

```python
# Limit search results
self.add_skill("native_vector_search", {
    "tool_name": "search_knowledge",
    "index_file": "knowledge.swsearch",
    "count": 3  # Limit results to reduce memory
})

# Clean up large objects
@self.tool(
    "process_large_data",
    description="Process large dataset",
    parameters={
        "dataset_id": {"type": "string", "description": "The dataset ID to process"}
    }
)
async def process_large_data(self, args, raw_data):
    dataset_id = args.get("dataset_id")
    data = await load_large_dataset(dataset_id)
    result = process_data(data)

    # Explicitly clean up
    del data

    return SwaigFunctionResult(f"Processed {result['count']} items")
```

---

## Security Best Practices

### Input Sanitization

```python
import re

@self.tool(
    "safe_search",
    description="Search with sanitized input",
    parameters={
        "query": {"type": "string", "description": "Search query to sanitize and execute"}
    }
)
def safe_search(self, args, raw_data):
    query = args.get("query", "")
    # Sanitize input
    safe_query = re.sub(r'[^\w\s-]', '', query)
    safe_query = safe_query.strip()[:100]  # Limit length

    if not safe_query:
        return SwaigFunctionResult("Invalid search query", error=True)

    # Safe to use
    results = search_database(safe_query)
    return SwaigFunctionResult(f"Found {len(results)} results")
```

### Secrets Management

```python
import os
from typing import Optional

class SecureAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Secure Agent", route="/")

        # Load secrets from environment
        self._api_key = os.environ.get("API_KEY")
        if not self._api_key:
            logger.warning("API_KEY not set")

        @self.tool(
            "secure_api_call",
            description="Make secure API call",
            parameters={
                "endpoint": {"type": "string", "description": "API endpoint to call"}
            }
        )
        def secure_api_call(self, args, raw_data):
            endpoint = args.get("endpoint")
            if not self._api_key:
                return SwaigFunctionResult("API not configured", error=True)

            # Never log secrets
            logger.info(f"Calling API endpoint: {endpoint}")
            # logger.info(f"Using key: {self._api_key}")  # NEVER DO THIS

            headers = {"Authorization": f"Bearer {self._api_key}"}
            # Make API call...
```

### Rate Limiting

```python
from datetime import datetime, timedelta
from collections import defaultdict

class RateLimitedAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Rate Limited Agent", route="/")
        self._call_counts = defaultdict(list)

        @self.tool(
            "limited_function",
            description="Rate limited function",
            parameters={
                "user_id": {"type": "string", "description": "User ID for rate limiting"}
            }
        )
        def limited_function(self, args, raw_data):
            user_id = args.get("user_id")
            # Check rate limit (10 calls per minute)
            now = datetime.now()
            minute_ago = now - timedelta(minutes=1)

            # Clean old entries
            self._call_counts[user_id] = [
                t for t in self._call_counts[user_id]
                if t > minute_ago
            ]

            # Check limit
            if len(self._call_counts[user_id]) >= 10:
                return SwaigFunctionResult(
                    "Rate limit exceeded. Please try again later.",
                    error=True
                )
            
            # Record call
            self._call_counts[user_id].append(now)
            
            # Process normally
            return SwaigFunctionResult("Function executed successfully")
```

---

## Summary

You've mastered advanced SignalWire Agents features! You've learned:

**Technical Skills:**
- ✅ Creating custom SWAIG functions with parameters and validation
- ✅ Proper error handling with SwaigFunctionResult
- ✅ Logging and debugging techniques
- ✅ Production deployment patterns
- ✅ Testing strategies for reliability

**Best Practices:**
- ✅ Performance optimization techniques
- ✅ Security considerations
- ✅ Monitoring and health checks
- ✅ Scalable architecture patterns

**What's Next?**

In the final lesson, you'll learn how to extend agents with custom skills, create complex conversation flows, and integrate with external services.

### Practice Exercises

1. **Create a Calculator Agent**: Build an agent with math functions (add, subtract, multiply, divide) with proper error handling
2. **Add Caching**: Implement a caching layer for expensive operations
3. **Build Health Checks**: Create comprehensive health monitoring
4. **Security Audit**: Review an agent for security vulnerabilities

### Production Checklist

Before deploying to production:

- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Authentication enabled
- [ ] Logging configured appropriately
- [ ] Error handling comprehensive
- [ ] Health checks implemented
- [ ] Monitoring set up
- [ ] Load tested
- [ ] Security reviewed
- [ ] Documentation complete

---

[← Lesson 3: Building Multi-Agent Systems](lesson3_multi_agent_systems.md) | [Tutorial Overview](README.md) | [Lesson 5: Extending Your Agents →](lesson5_extending_agents.md)