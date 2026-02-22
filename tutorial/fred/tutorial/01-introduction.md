# Lesson 1: Introduction to SignalWire Agents

Welcome to your journey of building Fred, a friendly Wikipedia-powered AI assistant! In this first lesson, we'll explore what SignalWire AI Agents are and how they work for building voice applications.

## Table of Contents

1. [What is SignalWire?](#what-is-signalwire)
2. [Understanding AI Agents](#understanding-ai-agents)
3. [The SDK Architecture](#the-sdk-architecture)
4. [What We're Building](#what-were-building)
5. [Key Concepts](#key-concepts)

---

## What is SignalWire?

SignalWire is a communications platform that enables developers to build voice, video, and messaging applications. Think of it as the infrastructure that connects your AI agent to phone calls, allowing your bot to have real conversations with people.

**Key Features:**

- Voice calling capabilities
- Real-time communication
- AI integration
- Scalable infrastructure

## Understanding AI Agents

An AI Agent in the SignalWire context is a Python application that:

1. **Listens for incoming calls** via HTTP endpoints
2. **Generates SWML documents** (SignalWire Markup Language) that define behavior
3. **Processes voice input** and generates appropriate responses
4. **Executes functions** based on user requests

### The Communication Flow

```
User calls → SignalWire Platform → Your Agent → SWML Response → AI Conversation
```

## The SDK Architecture

The SignalWire AI Agents SDK provides a clean abstraction layer:

```python
from signalwire_agents import AgentBase

class MyAgent(AgentBase):
    def __init__(self):
        super().__init__(name="My Agent", route="/agent")
```

**Core Components:**

1. **AgentBase Class**: The foundation all agents inherit from
2. **Skills System**: Modular capabilities you can add with one line
3. **SWAIG Functions**: Tools the AI can call during conversations
4. **Prompt Object Model (POM)**: Structured way to define agent behavior

## What We're Building

Fred is a voice-enabled Wikipedia assistant that demonstrates:

- **Personality Design**: Making AI agents feel human and approachable
- **Skill Integration**: Using the Wikipedia search skill
- **Custom Functions**: Adding unique capabilities
- **Voice Configuration**: Setting up natural-sounding speech

### Fred's Capabilities

By the end of this tutorial, Fred will be able to:

1. **Search Wikipedia** for any topic
2. **Share fun facts** about Wikipedia itself
3. **Engage naturally** in educational conversations
4. **Handle voice calls** through SignalWire

## Key Concepts

Before we start coding, let's understand these essential concepts:

### 1. SWML (SignalWire Markup Language)

SWML is a JSON document that tells SignalWire how your agent should behave. It includes:
- AI personality and instructions
- Available functions
- Voice settings
- Language configuration

### 2. SWAIG (SignalWire AI Gateway)

SWAIG enables your agent to execute functions during conversations. When a user asks Fred to search Wikipedia, SWAIG handles the function call and returns results.

### 3. Skills

Skills are pre-built, reusable modules that add capabilities to your agent. Instead of writing Wikipedia search from scratch, we'll use the existing skill:

```python
agent.add_skill("wikipedia_search")
```

### 4. HTTP Endpoints

Your agent exposes HTTP endpoints that SignalWire calls:
- `GET /agent` - Returns the SWML configuration
- `POST /agent/swaig/` - Handles function execution

### 5. Authentication

Agents use HTTP Basic Authentication for security. The SDK handles this automatically, generating credentials or using environment variables.

## Why This Architecture?

**Benefits:**

1. **Separation of Concerns**: Your code focuses on logic, not telephony
2. **Scalability**: Agents can handle multiple concurrent calls
3. **Flexibility**: Easy to add new capabilities
4. **Testing**: Can test locally without phone infrastructure

## Next Steps

Now that you understand the concepts, let's set up your development environment!

➡️ Continue to [Lesson 2: Setting Up Your Environment](02-setup.md)

---

**Review Questions:**

1. What is the purpose of SWML?
2. How do skills enhance an agent's capabilities?
3. What are the two main HTTP endpoints an agent exposes?

**Answers:**
1. SWML defines how the agent behaves during calls (personality, functions, voice settings)
2. Skills provide pre-built, tested functionality that can be added with one line of code
3. GET endpoint for SWML configuration, POST endpoint for SWAIG function execution

---

[← Back to Overview](README.md) | [Next: Environment Setup →](02-setup.md)