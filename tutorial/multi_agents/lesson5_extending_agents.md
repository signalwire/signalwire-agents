# Lesson 5: Extending Your Agents

In this final lesson, you'll learn how to create custom skills, implement complex conversation flows, manage state between interactions, and integrate with external services. These advanced techniques will help you build sophisticated, production-ready agents.

## Table of Contents

1. [Creating Custom Skills](#creating-custom-skills)
2. [State Management](#state-management)
3. [Complex Conversation Flows](#complex-conversation-flows)
4. [External Service Integration](#external-service-integration)
5. [Custom Voice Personas](#custom-voice-personas)
6. [Advanced Prompt Engineering](#advanced-prompt-engineering)
7. [Common Patterns](#common-patterns)
8. [Summary](#summary)

---

## Creating Custom Skills

Skills are reusable modules that add capabilities to agents. They encapsulate functionality, configuration, and dependencies.

### Basic Skill Structure

```python
# my_custom_skill.py
from signalwire_agents.skills import SkillBase
from signalwire_agents.core.function_result import SwaigFunctionResult

class WeatherSkill(SkillBase):
    """Skill for weather information"""
    
    # Required attributes
    SKILL_NAME = "weather"
    SKILL_DESCRIPTION = "Provides weather information and forecasts"
    SKILL_VERSION = "1.0.0"
    
    # Dependencies
    REQUIRED_PACKAGES = ["aiohttp", "python-dateutil"]
    REQUIRED_ENV_VARS = ["WEATHER_API_KEY"]
    
    def setup(self) -> bool:
        """Initialize the skill"""
        import os
        
        # Get API key from environment
        self.api_key = os.environ.get("WEATHER_API_KEY")
        if not self.api_key:
            self.logger.error("WEATHER_API_KEY not set")
            return False
            
        # Initialize any resources
        self.base_url = "https://api.weather.com/v1"
        return True
    
    def register_tools(self):
        """Register skill functions with the agent"""
        
        @self.agent.tool(
            "get_weather",
            description="Get current weather for a location",
            parameters={
                "location": {"type": "string", "description": "City or location name"}
            }
        )
        async def get_weather(self, args, raw_data):
            location = args.get("location")
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/current"
                params = {"q": location, "key": self.api_key}
                
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return SwaigFunctionResult(
                            f"Current weather in {location}: "
                            f"{data['temp']}°F, {data['condition']}"
                        )
                    else:
                        return SwaigFunctionResult(
                            f"Could not get weather for {location}",
                            error=True
                        )
        
        @self.agent.tool(
            "get_forecast",
            description="Get weather forecast",
            parameters={
                "location": {"type": "string", "description": "City or location name"},
                "days": {"type": "integer", "description": "Number of days to forecast (default 3)"}
            }
        )
        async def get_forecast(self, args, raw_data):
            location = args.get("location")
            days = args.get("days", 3)
            # Implementation...
            pass
```

### Using Custom Skills

```python
from signalwire_agents import AgentBase

class WeatherAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Weather Agent", route="/")
        
        # Add the custom skill
        self.add_skill("weather", {
            "tool_name": "weather_info",
            "api_endpoint": "https://api.weather.com"
        })
        
        # Or load from a local skill
        self.add_skill("./skills/weather_skill.py")
```

### Skill Configuration

Skills can accept configuration parameters:

```python
# Configure skill with parameters
self.add_skill("database", {
    "connection_string": "postgresql://localhost/mydb",
    "pool_size": 10,
    "timeout": 30
})

# Multiple instances of same skill
self.add_skill("native_vector_search", {
    "tool_name": "search_products",
    "index_file": "products.swsearch"
})

self.add_skill("native_vector_search", {
    "tool_name": "search_docs",
    "index_file": "documentation.swsearch"
})
```

---

## State Management

Managing state allows agents to remember information across conversations and interactions.

### Session-Based State

```python
from signalwire_agents.core.session_manager import SessionManager

class StatefulAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Stateful Agent", route="/")
        
        # Initialize session manager
        self.session_manager = SessionManager()
        
        @self.tool(
            "remember_preference",
            description="Remember user preference",
            parameters={
                "session_id": {"type": "string", "description": "Session identifier"},
                "key": {"type": "string", "description": "Preference key"},
                "value": {"type": "string", "description": "Preference value"}
            }
        )
        async def remember_preference(self, args, raw_data):
            session_id = args.get("session_id")
            key = args.get("key")
            value = args.get("value")
            # Store in session
            session = self.session_manager.get_or_create(session_id)
            session[key] = value

            return SwaigFunctionResult(f"I'll remember that your {key} is {value}")
        
        @self.tool(
            "recall_preference",
            description="Recall user preference",
            parameters={
                "session_id": {"type": "string", "description": "Session identifier"},
                "key": {"type": "string", "description": "Preference key to recall"}
            }
        )
        async def recall_preference(self, args, raw_data):
            session_id = args.get("session_id")
            key = args.get("key")
            session = self.session_manager.get(session_id)

            if session and key in session:
                value = session[key]
                return SwaigFunctionResult(f"Your {key} is {value}")
            else:
                return SwaigFunctionResult(f"I don't have your {key} on record")
```

### Persistent State with Database

```python
import asyncpg
from datetime import datetime

class PersistentAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Persistent Agent", route="/")
        self.db_pool = None
        
    async def initialize(self):
        """Initialize database connection pool"""
        self.db_pool = await asyncpg.create_pool(
            "postgresql://localhost/agent_db",
            min_size=5,
            max_size=20
        )
    
        @self.tool(
            "save_interaction",
            description="Save interaction to database",
            parameters={
                "customer_id": {"type": "string", "description": "Customer identifier"},
                "interaction_type": {"type": "string", "description": "Type of interaction"},
                "details": {"type": "string", "description": "Interaction details"}
            }
        )
        async def save_interaction(self, args, raw_data):
            customer_id = args.get("customer_id")
            interaction_type = args.get("interaction_type")
            details = args.get("details")
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO interactions
                    (customer_id, type, details, timestamp)
                    VALUES ($1, $2, $3, $4)
                """, customer_id, interaction_type, details, datetime.now())

            return SwaigFunctionResult("Interaction saved successfully")
    
        @self.tool(
            "get_history",
            description="Get customer interaction history",
            parameters={
                "customer_id": {"type": "string", "description": "Customer identifier"},
                "limit": {"type": "integer", "description": "Max number of records (default 5)"}
            }
        )
        async def get_history(self, args, raw_data):
            customer_id = args.get("customer_id")
            limit = args.get("limit", 5)
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT type, details, timestamp
                    FROM interactions
                    WHERE customer_id = $1
                    ORDER BY timestamp DESC
                    LIMIT $2
                """, customer_id, limit)

            if rows:
                history = "\n".join([
                    f"- {row['type']} on {row['timestamp']}: {row['details']}"
                    for row in rows
                ])
                return SwaigFunctionResult(f"Previous interactions:\n{history}")
            else:
                return SwaigFunctionResult("No previous interactions found")
```

### Global Data Between Contexts

```python
# Setting global data that persists across contexts
@self.tool(
    "set_customer_data",
    description="Store customer data globally",
    parameters={
        "key": {"type": "string", "description": "Data key name"},
        "value": {"type": "string", "description": "Data value to store"}
    }
)
async def set_customer_data(self, args, raw_data):
    key = args.get("key")
    value = args.get("value")
    result = SwaigFunctionResult(f"Stored {key}")
    result.add_action("set_global_data", {
        f"customer_{key}": value
    })
    return result

# Accessing global data in prompts
self.prompt_add_section(
    "Customer Context",
    body="Customer information: ${customer_name}, ${customer_email}"
)
```

---

## Complex Conversation Flows

### Multi-Step Workflows

```python
class WorkflowAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Workflow Agent", route="/")
        
        # Define workflow states
        self.workflows = {}
        
        @self.tool(
            "start_order",
            description="Start new order workflow",
            parameters={
                "session_id": {"type": "string", "description": "Session identifier"}
            }
        )
        async def start_order(self, args, raw_data):
            session_id = args.get("session_id")
            # Initialize workflow
            self.workflows[session_id] = {
                "state": "awaiting_items",
                "items": [],
                "total": 0
            }

            return SwaigFunctionResult(
                "I'll help you create an order. What items would you like?"
            )
        
        @self.tool(
            "add_item",
            description="Add item to current order",
            parameters={
                "session_id": {"type": "string", "description": "Session identifier"},
                "item": {"type": "string", "description": "Item name to add"},
                "quantity": {"type": "integer", "description": "Quantity (default 1)"}
            }
        )
        async def add_item(self, args, raw_data):
            session_id = args.get("session_id")
            item = args.get("item")
            quantity = args.get("quantity", 1)
            if session_id not in self.workflows:
                return SwaigFunctionResult(
                    "No active order. Please start a new order first.",
                    error=True
                )

            workflow = self.workflows[session_id]

            # Add item
            workflow["items"].append({
                "name": item,
                "quantity": quantity,
                "price": await self.get_item_price(item)
            })

            # Update state
            if len(workflow["items"]) >= 1:
                workflow["state"] = "ready_to_confirm"

            items_list = ", ".join([
                f"{i['quantity']}x {i['name']}"
                for i in workflow["items"]
            ])

            return SwaigFunctionResult(
                f"Added {quantity}x {item}. Current order: {items_list}. "
                f"Add more items or confirm order."
            )
        
        @self.tool(
            "confirm_order",
            description="Confirm and submit order",
            parameters={
                "session_id": {"type": "string", "description": "Session identifier"}
            }
        )
        async def confirm_order(self, args, raw_data):
            session_id = args.get("session_id")
            if session_id not in self.workflows:
                return SwaigFunctionResult("No active order", error=True)

            workflow = self.workflows[session_id]

            if workflow["state"] != "ready_to_confirm":
                return SwaigFunctionResult(
                    "Please add items before confirming",
                    error=True
                )

            # Calculate total
            total = sum(i["price"] * i["quantity"] for i in workflow["items"])

            # Submit order
            order_id = await self.submit_order(workflow["items"])

            # Clean up
            del self.workflows[session_id]

            return SwaigFunctionResult(
                f"Order #{order_id} confirmed! Total: ${total:.2f}"
            )
```

### Conditional Flows

```python
class ConditionalAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Conditional Agent", route="/")
        
        # Add conditional instructions
        self.prompt_add_section(
            "Conditional Responses",
            body="Adapt your behavior based on context:",
            bullets=[
                "For new customers: Be extra welcoming and explain our services",
                "For VIP customers: Acknowledge their status and offer premium options",
                "For support issues: Show empathy and urgency",
                "For sales inquiries: Be enthusiastic and helpful"
            ]
        )
        
        @self.tool(
            "check_customer_status",
            description="Check customer status",
            parameters={
                "customer_id": {"type": "string", "description": "Customer identifier"}
            }
        )
        async def check_customer_status(self, args, raw_data):
            customer_id = args.get("customer_id")
            # Check database or API
            status = await self.get_customer_status(customer_id)

            result = SwaigFunctionResult(f"Customer status: {status}")

            # Set context for agent
            result.add_action("set_global_data", {
                "customer_status": status,
                "is_vip": status == "platinum"
            })

            # Add specific instructions based on status
            if status == "platinum":
                result.add_action("append_prompt", {
                    "section": "VIP Instructions",
                    "content": "This is a VIP customer - provide white glove service"
                })

            return result
```

---

## External Service Integration

### REST API Integration

```python
import aiohttp
from typing import Dict, Any

class APIIntegrationAgent(AgentBase):
    def __init__(self):
        super().__init__(name="API Agent", route="/")
        
        # Configure API endpoints
        self.apis = {
            "inventory": "https://api.company.com/inventory",
            "shipping": "https://api.company.com/shipping",
            "pricing": "https://api.company.com/pricing"
        }
        
        @self.tool(
            "check_availability",
            description="Check product availability",
            parameters={
                "product_sku": {"type": "string", "description": "Product SKU to check"}
            }
        )
        async def check_availability(self, args, raw_data):
            product_sku = args.get("product_sku")
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {os.environ['API_KEY']}"}

                async with session.get(
                    f"{self.apis['inventory']}/{product_sku}",
                    headers=headers
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()

                        if data["in_stock"]:
                            return SwaigFunctionResult(
                                f"Product {product_sku} is in stock. "
                                f"Available quantity: {data['quantity']}"
                            )
                        else:
                            return SwaigFunctionResult(
                                f"Product {product_sku} is currently out of stock. "
                                f"Expected restock: {data['restock_date']}"
                            )
                    else:
                        return SwaigFunctionResult(
                            "Unable to check availability",
                            error=True
                        )
```

### Webhook Integration

```python
from fastapi import BackgroundTasks

class WebhookAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Webhook Agent", route="/")
        
        # Add webhook endpoint to the agent's app
        @self.app.post("/webhook/order_update")
        async def handle_order_webhook(
            order_id: str,
            status: str,
            background_tasks: BackgroundTasks
        ):
            # Process webhook
            background_tasks.add_task(
                self.process_order_update,
                order_id,
                status
            )
            
            return {"status": "received"}
    
    async def process_order_update(self, order_id: str, status: str):
        """Process order updates asynchronously"""
        logger.info(f"Order {order_id} updated to {status}")
        
        # Update database
        await self.update_order_status(order_id, status)
        
        # Notify customer if needed
        if status in ["shipped", "delivered", "cancelled"]:
            await self.notify_customer(order_id, status)
```

### Message Queue Integration

```python
import asyncio
import aioredis

class QueueAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Queue Agent", route="/")
        self.redis = None
        
    async def initialize(self):
        """Initialize Redis connection"""
        self.redis = await aioredis.create_redis_pool(
            "redis://localhost",
            encoding="utf-8"
        )
        
        # Start background task processor
        asyncio.create_task(self.process_queue())
    
        @self.tool(
            "queue_task",
            description="Queue task for processing",
            parameters={
                "task_type": {"type": "string", "description": "Type of task to queue"},
                "data": {"type": "string", "description": "Task data payload"}
            }
        )
        async def queue_task(self, args, raw_data):
            task_type = args.get("task_type")
            data = args.get("data")
            # Add to queue
            await self.redis.rpush(
                "task_queue",
                json.dumps({
                    "type": task_type,
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                })
            )

            return SwaigFunctionResult(f"Task queued for processing")
    
    async def process_queue(self):
        """Background queue processor"""
        while True:
            try:
                # Get task from queue
                task_json = await self.redis.lpop("task_queue")
                
                if task_json:
                    task = json.loads(task_json)
                    await self.handle_task(task)
                else:
                    # No tasks, wait
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(5)
```

---

## Custom Voice Personas

### Creating Dynamic Personas

```python
class PersonaAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Persona Agent", route="/")
        
        # Define multiple personas
        self.personas = {
            "professional": {
                "voice": "rime.cove",
                "style": "formal and precise",
                "pace": "measured"
            },
            "friendly": {
                "voice": "rime.marsh",
                "style": "warm and conversational",
                "pace": "relaxed"
            },
            "energetic": {
                "voice": "rime.spore",
                "style": "enthusiastic and upbeat",
                "pace": "quick"
            }
        }
        
        # Default persona
        self.set_persona("friendly")
        
    def set_persona(self, persona_name: str):
        """Switch to a different persona"""
        if persona_name not in self.personas:
            return
            
        persona = self.personas[persona_name]
        
        # Update voice
        self.add_language(
            name="English",
            code="en-US",
            voice=persona["voice"]
        )
        
        # Update prompt
        self.prompt_add_section(
            "Voice Style",
            body=f"Speak in a {persona['style']} manner at a {persona['pace']} pace"
        )
```

### Multilingual Support

```python
class MultilingualAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Multilingual Agent", route="/")
        
        # Add multiple languages
        self.add_language(
            name="English",
            code="en-US",
            voice="rime.marsh"
        )
        
        self.add_language(
            name="Spanish",
            code="es-MX",
            voice="rime.marsh"
        )
        
        self.add_language(
            name="French",
            code="fr-FR",
            voice="rime.marsh"
        )
        
        # Language-specific prompts
        self.prompt_add_section(
            "Language Instructions",
            body="Adapt your communication style to the selected language:",
            bullets=[
                "English: Professional but friendly American business style",
                "Spanish: Warm and personable Latin American style",
                "French: Polite and formal French business etiquette"
            ]
        )
```

---

## Advanced Prompt Engineering

### Structured Reasoning

```python
class ReasoningAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Reasoning Agent", route="/")
        
        # Add reasoning framework
        self.prompt_add_section(
            "Reasoning Framework",
            body="Follow this structured approach for complex requests:",
            bullets=[
                "1. Understand: Clarify what the customer is asking for",
                "2. Analyze: Break down the request into components",
                "3. Plan: Determine the best approach to help",
                "4. Execute: Take action using available tools",
                "5. Verify: Confirm the solution meets their needs"
            ]
        )
        
        # Add decision criteria
        self.prompt_add_section(
            "Decision Criteria",
            body="When making recommendations, consider:",
            bullets=[
                "Customer's stated requirements",
                "Budget constraints",
                "Technical compatibility",
                "Future scalability",
                "Best value proposition"
            ]
        )
```

### Dynamic Prompt Injection

```python
@self.tool(
    "inject_context",
    description="Inject contextual instructions",
    parameters={
        "context_type": {"type": "string", "description": "Context type: technical, simple, or sales"}
    }
)
async def inject_context(self, args, raw_data):
    context_type = args.get("context_type")
    contexts = {
        "technical": {
            "section": "Technical Mode",
            "content": "Provide detailed technical explanations"
        },
        "simple": {
            "section": "Simple Mode",
            "content": "Explain everything in simple, non-technical terms"
        },
        "sales": {
            "section": "Sales Mode",
            "content": "Focus on benefits and value proposition"
        }
    }

    if context_type in contexts:
        result = SwaigFunctionResult(f"Switching to {context_type} mode")
        result.add_action("append_prompt", contexts[context_type])
        return result
    else:
        return SwaigFunctionResult("Unknown context type", error=True)
```

---

## Common Patterns

### Retry Pattern

```python
import asyncio
from typing import Optional

async def retry_with_backoff(
    func,
    max_attempts: int = 3,
    initial_delay: float = 1.0
):
    """Retry function with exponential backoff"""
    delay = initial_delay
    
    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            
            logger.warning(
                f"Attempt {attempt + 1} failed: {e}. "
                f"Retrying in {delay} seconds..."
            )
            await asyncio.sleep(delay)
            delay *= 2  # Exponential backoff
```

### Circuit Breaker Pattern

```python
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.is_open = False
    
    async def call(self, func):
        if self.is_open:
            if (datetime.now() - self.last_failure_time).seconds > self.reset_timeout:
                self.is_open = False
                self.failure_count = 0
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func()
            self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
                logger.error("Circuit breaker opened due to repeated failures")
            
            raise
```

### Factory Pattern for Agents

```python
class AgentFactory:
    """Factory for creating configured agents"""
    
    @staticmethod
    def create_agent(agent_type: str, config: Dict[str, Any]) -> AgentBase:
        agents = {
            "sales": SalesAgent,
            "support": SupportAgent,
            "triage": TriageAgent
        }
        
        if agent_type not in agents:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Create agent instance
        agent_class = agents[agent_type]
        agent = agent_class()
        
        # Apply configuration
        if "skills" in config:
            for skill_config in config["skills"]:
                agent.add_skill(skill_config["name"], skill_config["params"])
        
        if "languages" in config:
            for lang in config["languages"]:
                agent.add_language(**lang)
        
        return agent
```

---

## Summary

Congratulations! You've completed the SignalWire Agents SDK tutorial. You've learned:

**Core Skills Mastered:**
- ✅ Creating custom skills with dependencies and configuration
- ✅ Managing state across conversations
- ✅ Building complex multi-step workflows
- ✅ Integrating with external APIs and services
- ✅ Creating dynamic voice personas
- ✅ Advanced prompt engineering techniques
- ✅ Common patterns for robust agents

**You're Now Ready To:**
- Build production-ready AI voice agents
- Create custom skills for any use case
- Integrate agents with existing systems
- Handle complex conversational flows
- Deploy scalable multi-agent systems

### Next Steps

1. **Build Your Own Agent**: Start with a simple use case and expand
2. **Contribute**: Share your custom skills with the community
3. **Optimize**: Profile and improve your agent's performance
4. **Deploy**: Put your agents into production
5. **Iterate**: Gather feedback and continuously improve

### Resources for Continued Learning

- **Documentation**: [SignalWire Docs](https://docs.signalwire.com)
- **Community**: Join the SignalWire Discord
- **Examples**: Explore the examples directory
- **Support**: Open issues on GitHub

### Final Tips

**Remember:**
- Start simple and iterate
- Test thoroughly before deploying
- Monitor your agents in production
- Keep your dependencies updated
- Document your custom skills
- Share your learnings with others

Thank you for completing this tutorial! We can't wait to see what you build with the SignalWire Agents SDK.

---

[← Lesson 4: Advanced Features](lesson4_advanced_features.md) | [Tutorial Overview](README.md)