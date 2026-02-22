# Lesson 3: Building Multi-Agent Systems

In this lesson, you'll learn how to create sophisticated multi-agent systems where specialized agents work together. We'll build the complete PC Builder Pro system with three agents: Alex (triage), Morgan (sales), and Sam (support).

## Table of Contents

1. [Understanding Multi-Agent Architecture](#understanding-multi-agent-architecture)
2. [The AgentServer Class](#the-agentserver-class)
3. [Dynamic Configuration](#dynamic-configuration)
4. [Agent-to-Agent Transfers](#agent-to-agent-transfers)
5. [Building the Complete System](#building-the-complete-system)
6. [Testing Multi-Agent Flows](#testing-multi-agent-flows)
7. [Production Considerations](#production-considerations)
8. [Summary](#summary)

---

## Understanding Multi-Agent Architecture

Multi-agent systems allow you to create specialized agents that handle different aspects of customer interaction. This provides several benefits:

**Advantages:**

- **Specialization**: Each agent focuses on specific tasks
- **Maintainability**: Easier to update individual agents
- **Scalability**: Can add new agents without affecting others
- **Clear Workflows**: Natural handoffs between specialists

**Our System Architecture:**

```
Customer → Triage Agent (Alex) → ┬→ Sales Agent (Morgan)
                                 └→ Support Agent (Sam)
```

**Key Components:**

1. **AgentServer**: Hosts multiple agents on the same port
2. **Routes**: Each agent has a unique endpoint (/, /sales, /support)
3. **Transfer Mechanism**: swml_transfer skill for agent-to-agent handoffs
4. **Context Preservation**: Customer data flows between agents

---

## The AgentServer Class

The `AgentServer` class allows you to host multiple agents on a single port, each with its own route.

### Basic Usage

```python
from signalwire_agents import AgentServer, AgentBase

# Create the server
server = AgentServer(
    host="0.0.0.0",
    port=3001,
    log_level="info"
)

# Create and register agents
triage_agent = TriageAgent()
server.register(triage_agent, "/")

sales_agent = SalesAgent()
server.register(sales_agent, "/sales")

# Run the server
server.run()
```

### Server Features

**Automatic Features:**

- Health check endpoint at `/health`
- Shared authentication across all agents
- Unified logging
- Graceful shutdown handling

**Configuration Options:**

```python
server = AgentServer(
    host="0.0.0.0",        # Network interface
    port=3001,             # TCP port
    log_level="debug",     # Logging verbosity
    auth_user="custom",    # Override auto-generated auth
    auth_pass="secret"     # Override auto-generated password
)
```

---

## Dynamic Configuration

Dynamic configuration allows agents to adapt their behavior based on request parameters. This is crucial for:

- Building correct transfer URLs
- Detecting proxy/tunnel configurations
- Customizing behavior per request
- Multi-tenant scenarios

### How It Works

```python
def configure_transfer_tools(self, query_params, body_params, headers, agent):
    """
    Called for every request before processing
    
    Args:
        query_params: URL query parameters (dict)
        body_params: POST body parameters (dict)
        headers: HTTP headers (dict)
        agent: The agent instance to configure
    """
    # Access proxy-aware URL building
    base_url = agent.get_full_url(include_auth=True)
    
    # Configure agent based on request
    if query_params.get('transfer') == 'true':
        # This is a transfer call
        agent.prompt_add_section(...)
```

### Key Method: get_full_url()

This method intelligently builds URLs that work with:
- Direct connections
- Reverse proxies
- SignalWire's proxy tunnels
- Custom domains

```python
# Returns the correct URL for the current environment
url = agent.get_full_url(include_auth=True)
# Examples:
# Direct: https://user:pass@yourdomain.com:3001
# Proxy: https://user:pass@proxy.signalwire.com/agent-id
```

---

## Agent-to-Agent Transfers

The `swml_transfer` skill enables handoffs between agents while preserving context.

### Understanding swml_transfer

```python
agent.add_skill("swml_transfer", {
    "tool_name": "transfer_to_specialist",
    "description": "Transfer to sales or support specialist",
    "parameter_name": "specialist_type",
    "parameter_description": "The type of specialist (sales or support)",
    "required_fields": {
        "user_name": "The customer's name",
        "summary": "Summary of the conversation"
    },
    "transfers": {
        "/sales/i": {  # Regex pattern
            "url": sales_url,
            "message": "Transferring to sales...",
            "return_message": "Call complete."
        }
    }
})
```

### Key Features

**Required Fields:**
- Automatically prompts for missing information
- Ensures context is complete before transfer
- Makes data available via `${call_data.field_name}`

**Transfer Configuration:**
- Uses regex patterns for flexible matching
- Supports multiple transfer destinations
- Can return to original agent after transfer

### Accessing Transfer Data

In the receiving agent:

```python
"The customer's name is ${call_data.user_name}"
"They were transferred because: ${call_data.summary}"
```

---

## Building the Complete System

Let's examine the complete PC Builder Pro system in `pc_builder.py`:

### 1. Triage Agent (Alex)

```python
class TriageAgent(AgentBase):
    def __init__(self):
        super().__init__(
            name="PC Builder Triage Agent",
            route="/",  # Root route
            host="0.0.0.0",
            port=3001
        )
        
        # Configure prompt
        self._configure_prompt()
        
        # Set voice
        self.add_language(
            name="English",
            code="en-US",
            voice="rime.spore"  # Energetic voice
        )
        
        # Dynamic configuration for transfers
        self.set_dynamic_config_callback(self.configure_transfer_tools)
```

### 2. Dynamic Transfer Configuration

```python
def configure_transfer_tools(self, query_params, body_params, headers, agent):
    # Build URLs with proxy detection
    sales_url = agent.get_full_url(include_auth=True).rstrip('/') + "/sales?transfer=true"
    support_url = agent.get_full_url(include_auth=True).rstrip('/') + "/support?transfer=true"
    
    # Configure transfers
    agent.add_skill("swml_transfer", {
        "tool_name": "transfer_to_specialist",
        "required_fields": {
            "user_name": "The customer's name",
            "summary": "A comprehensive summary of the conversation"
        },
        "transfers": {
            "/sales/i": {
                "url": sales_url,
                "message": "Perfect! Let me transfer you to our sales specialist.",
                "post_process": True
            },
            "/support/i": {
                "url": support_url,
                "message": "I'll connect you with technical support.",
                "post_process": True
            }
        }
    })
```

### 3. Sales Agent with Transfer Detection

```python
class SalesAgent(AgentBase):
    def __init__(self):
        # ... initialization ...
        
        # Dynamic prompt configuration
        self.set_dynamic_config_callback(self.configure_dynamic_prompt)
    
    def configure_dynamic_prompt(self, query_params, body_params, headers, agent):
        if query_params.get('transfer') == 'true':
            # This is a transfer - add context
            agent.prompt_add_section(
                "Call Transfer Information",
                body="This call has been transferred from triage.",
                bullets=[
                    "Customer name: ${call_data.user_name}",
                    "Transfer reason: ${call_data.summary}",
                    "Greet them by name and acknowledge the transfer"
                ]
            )
        else:
            # Direct call - standard greeting
            agent.prompt_add_section(
                "Initial Greeting",
                body="This is a direct call to sales.",
                bullets=["Greet warmly", "Ask for name", "Understand needs"]
            )
```

### 4. Creating the Server

```python
def create_pc_builder_app(host="0.0.0.0", port=3001):
    # Create server
    server = AgentServer(host=host, port=port)
    
    # Create and register agents
    triage = TriageAgent()
    server.register(triage, "/")
    
    sales = SalesAgent()
    server.register(sales, "/sales")
    
    support = SupportAgent()
    server.register(support, "/support")
    
    # Add info endpoint
    @server.app.get("/info")
    async def info():
        return {
            "agents": {
                "triage": {"endpoint": "/"},
                "sales": {"endpoint": "/sales"},
                "support": {"endpoint": "/support"}
            }
        }
    
    return server
```

---

## Testing Multi-Agent Flows

### Starting the System

```bash
# Run the complete system
python tutorial/pc_builder.py

# You'll see:
# Triage Agent (Alex): http://localhost:3001/
# Sales Agent (Morgan): http://localhost:3001/sales
# Support Agent (Sam): http://localhost:3001/support
```

### Testing Individual Agents

```bash
# Test triage agent
curl http://localhost:3001/

# Test sales agent directly
curl http://localhost:3001/sales

# Test with transfer flag
curl "http://localhost:3001/sales?transfer=true"
```

### Testing Transfer Flow

1. **Call Triage Agent**: Start with Alex
2. **Provide Information**: Give name and describe needs
3. **Request Transfer**: Say "I want to buy a gaming PC"
4. **Observe Handoff**: See context preserved in sales

### Monitoring Transfers

Look for these in the logs:
- Transfer skill activation
- Required fields collection
- URL construction with proxy detection
- Context passing to new agent

---

## Production Considerations

### Security

**Authentication:**
```bash
# Set custom credentials
export SWML_AUTH_USER=myuser
export SWML_AUTH_PASS=mypassword

# Or let the system generate them (check logs)
```

**SSL/HTTPS:**
```bash
SWML_SSL_ENABLED=true \
SWML_SSL_CERT_PATH=/path/to/cert.pem \
SWML_SSL_KEY_PATH=/path/to/key.pem \
python tutorial/pc_builder.py
```

### Deployment Patterns

**1. Single Server**:
- All agents on one AgentServer
- Shared resources and authentication
- Best for small to medium deployments

**2. Distributed Agents**:
- Each agent on separate servers
- Independent scaling
- More complex configuration

**3. Lambda/Serverless**:
```python
def lambda_handler(event, context):
    server = create_pc_builder_app()
    return server.run(event, context)
```

### Monitoring

**Health Checks:**
```bash
# Built-in health endpoint
curl http://localhost:3001/health
```

**Logging:**
```python
# Set appropriate log level
server = AgentServer(log_level="info")  # or "debug" for more detail
```

**Metrics to Track:**
- Transfer success rates
- Agent response times
- Error rates by agent
- Context preservation success

---

## Summary

You've built a complete multi-agent system! You've mastered:

**Core Concepts:**
- ✅ Using AgentServer to host multiple agents
- ✅ Dynamic configuration for request-time adaptation
- ✅ Proxy-aware URL building with get_full_url()
- ✅ Agent transfers with context preservation
- ✅ Handling both direct calls and transfers

**Architecture Patterns:**
- ✅ Specialized agents for different roles
- ✅ Agent-to-agent handoffs with context
- ✅ Context flowing through the system
- ✅ Production-ready security and monitoring

**What's Next?**

In the next lesson, you'll learn advanced features including custom SWAIG functions, error handling, and production deployment strategies.

### Practice Exercises

1. **Add a Fourth Agent**: Create a billing agent and add transfer routes
2. **Custom Transfer Messages**: Personalize transfer messages based on context
3. **Transfer Validation**: Add logic to prevent invalid transfers
4. **Multi-Language**: Add Spanish versions of each agent

### Troubleshooting

**Common Issues:**

- **Transfer URLs Wrong**: Check get_full_url() is being used
- **Context Not Passing**: Verify required_fields are defined
- **Port Conflicts**: Ensure only one server runs on each port
- **Auth Issues**: Check credentials in logs or set custom ones

---

[← Lesson 2: Adding Intelligence with Knowledge Bases](lesson2_knowledge_bases.md) | [Tutorial Overview](README.md) | [Lesson 4: Advanced Features →](lesson4_advanced_features.md)