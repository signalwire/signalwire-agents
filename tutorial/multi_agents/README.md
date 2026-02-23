# SignalWire Agents SDK Tutorial

Welcome to the comprehensive tutorial for building AI-powered voice agents using the SignalWire Agents SDK. This tutorial will take you from creating your first simple agent to building sophisticated multi-agent systems with knowledge base integration.

## Table of Contents

1. [What You'll Learn](#what-youll-learn)
2. [Prerequisites](#prerequisites)
3. [Tutorial Structure](#tutorial-structure)
4. [Getting Started](#getting-started)
5. [Additional Resources](#additional-resources)

---

## What You'll Learn

This tutorial covers everything you need to know to build production-ready AI voice agents:

**Core Concepts:**

- Creating agents that inherit from `AgentBase`
- Using the Prompt Object Model (POM) for structured prompts
- Configuring voice personas and languages
- Running agents locally and with SSL/HTTPS

**Advanced Features:**

- Integrating searchable knowledge bases with vector search
- Building multi-agent systems with intelligent routing
- Implementing custom SWAIG (SignalWire AI Gateway) functions
- Managing agent-to-agent context sharing
- Deploying to production environments

**Real-World Skills:**

- Best practices for prompt engineering
- Error handling and debugging techniques
- Performance optimization strategies
- Security considerations

---

## Prerequisites

Before starting this tutorial, ensure you have:

**Required:**

- Python 3.8 or higher
- pip package manager
- Basic Python programming knowledge
- A text editor or IDE

**Recommended:**

- SignalWire account (for testing with real calls)
- Basic understanding of REST APIs
- Familiarity with async/await in Python

---

## Tutorial Structure

This tutorial is organized into progressive lessons, each building on the previous:

### Lesson 1: Creating Your First Agent
[📖 Read Lesson 1](lesson1_first_agent.md)

Learn the fundamentals by building a simple sales agent named Morgan. You'll understand:
- Basic agent structure and configuration
- Prompt engineering with POM
- Running agents locally and with SSL

### Lesson 2: Adding Intelligence with Knowledge Bases
[📖 Read Lesson 2](lesson2_knowledge_bases.md)

Enhance your agent with searchable knowledge using vector search. You'll learn:
- Installing search dependencies
- Converting documents to searchable format
- Integrating knowledge bases into agents

### Lesson 3: Building Multi-Agent Systems
[📖 Read Lesson 3](lesson3_multi_agent_systems.md)

Create a complete multi-agent system with routing and context sharing. You'll master:
- AgentServer for hosting multiple agents
- Dynamic configuration callbacks
- Agent-to-agent transfers with context preservation

### Lesson 4: Advanced Features and Best Practices
[📖 Read Lesson 4](lesson4_advanced_features.md)

Master advanced concepts for production deployment. Topics include:
- Custom SWAIG functions
- Error handling strategies
- Production deployment patterns
- Performance optimization

### Lesson 5: Extending Your Agents
[📖 Read Lesson 5](lesson5_extending_agents.md)

Learn to add custom functionality and create sophisticated conversational flows:
- Creating custom skills
- State management
- Complex conversation patterns
- Integration with external services

---

## Getting Started

### Quick Installation

```bash
# Clone the repository (if not already done)
git clone https://github.com/signalwire/signalwire-agents.git
cd signalwire-agents

# Install the base SDK
pip install -e .

# Or install with search capabilities
pip install -e .[search]
```

### Running Your First Agent

After completing Lesson 1, you can run your first agent:

```bash
# Run the simple sales agent
python tutorial/sales_agent.py

# The agent will be available at http://localhost:3000/
```

### Testing with SWML

You can test your agent using curl:

```bash
curl -X POST http://localhost:3000/ \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

## Additional Resources

### Quick Reference

**Common Commands:**

```bash
# Install with search support
pip install -e .[search]

# Build a search index
sw-search ./docs --output knowledge.swsearch

# Test SWAIG functions locally
swaig-test agent.py --list-tools
swaig-test agent.py --exec tool_name --param value

# Run with SSL enabled
SWML_SSL_ENABLED=true \
SWML_SSL_CERT_PATH=/path/to/cert.pem \
SWML_SSL_KEY_PATH=/path/to/key.pem \
SWML_DOMAIN=yourdomain.com \
python agent.py
```

**Environment Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `SWML_SSL_ENABLED` | Enable HTTPS | `false` |
| `SWML_SSL_CERT_PATH` | Path to SSL certificate | None |
| `SWML_SSL_KEY_PATH` | Path to SSL private key | None |
| `SWML_DOMAIN` | Domain for SSL | None |
| `SWML_AUTH_USER` | Basic auth username | Auto-generated |
| `SWML_AUTH_PASS` | Basic auth password | Auto-generated |
| `PYTORCH_DISABLE_AVX512` | Disable AVX512 for compatibility | `0` |

### Code Examples

All code examples from this tutorial are available in this directory:

- `sales_agent.py` - Basic sales agent (Lesson 1)
- `sales_agent_with_search.py` - Agent with knowledge base (Lesson 2)
- `pc_builder.py` - Complete multi-agent system (Lesson 3)
- `sales_knowledge.md` - Sales knowledge base
- `support_knowledge.md` - Support knowledge base

### Troubleshooting

**Common Issues:**

1. **Port already in use:**
   ```bash
   # Change the port in your agent configuration
   port=3001  # or any available port
   ```

2. **Search module not found:**
   ```bash
   # Install with search support
   pip install -e .[search]
   ```

3. **PyTorch compatibility issues:**
   ```bash
   # Disable AVX512 instructions
   export PYTORCH_DISABLE_AVX512=1
   ```

### Getting Help

**Resources:**

- [SignalWire Documentation](https://docs.signalwire.com)
- [GitHub Issues](https://github.com/signalwire/signalwire-agents/issues)
- [Community Discord](https://discord.gg/signalwire)

**Debugging Tips:**

- Use `--verbose` flag for detailed logging
- Check agent logs for error messages
- Test SWAIG functions with `swaig-test` before deployment
- Use `curl` to test SWML endpoints directly

### Best Practices Checklist

**Agent Development:**

- ✅ Always inherit from `AgentBase`
- ✅ Use POM for structured prompts
- ✅ Handle errors gracefully with `SwaigFunctionResult`
- ✅ Test locally before deployment
- ✅ Use meaningful function and parameter names

**Production Deployment:**

- ✅ Enable SSL/HTTPS in production
- ✅ Use environment variables for configuration
- ✅ Implement proper logging
- ✅ Set up health checks
- ✅ Monitor agent performance

**Security:**

- ✅ Use strong authentication
- ✅ Validate all inputs
- ✅ Don't expose sensitive data in prompts
- ✅ Use HTTPS for all communications
- ✅ Regularly update dependencies

---

## Ready to Begin?

Start with [Lesson 1: Creating Your First Agent](lesson1_first_agent.md) to begin your journey into building AI-powered voice agents with SignalWire!

*This tutorial is part of the SignalWire Agents SDK. For the latest updates and more information, visit the [official repository](https://github.com/signalwire/signalwire-agents).*