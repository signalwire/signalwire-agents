# BedrockAgent Documentation

## Overview

BedrockAgent is a specialized agent implementation that integrates Amazon Bedrock's voice-to-voice model with SignalWire's agent ecosystem. It extends AgentBase to provide full compatibility with all SignalWire agent features while generating SWML documents with the `amazon_bedrock` verb instead of the standard `ai` verb.

## Key Features

- **Full Agent Compatibility**: Inherits all capabilities from AgentBase including skills, tools, POM, and SWAIG functions
- **Bedrock Integration**: Generates SWML with `amazon_bedrock` verb for native Bedrock support
- **Voice-to-Voice Model**: Uses Amazon Bedrock's fixed voice-to-voice model
- **Drop-in Migration**: Drop-in replacement for standard agents when using Bedrock

## Installation

BedrockAgent is included in the signalwire-agents package:

```python
from signalwire_agents import BedrockAgent
```

## Basic Usage

### Creating a BedrockAgent

```python
from signalwire_agents import BedrockAgent

# Create a basic Bedrock agent
agent = BedrockAgent(
    name="my_bedrock_agent",
    system_prompt="You are a helpful AI assistant.",
    voice_id="joanna",
    temperature=0.7
)

# Run as SWML server
agent.run()
```

### With Skills and Tools

```python
from signalwire_agents import BedrockAgent

agent = BedrockAgent(
    name="advanced_bedrock",
    system_prompt="You are an AI assistant with various capabilities.",
    voice_id="matthew",
    temperature=0.8,
    top_p=0.9,
    max_tokens=1024
)

# Add skills
agent.add_skill("datetime")
agent.add_skill("weather_api", {"api_key": "your_api_key"})

# Add custom tools
@agent.tool
def calculate_sum(a: int, b: int):
    """Calculate the sum of two numbers"""
    return f"The sum of {a} and {b} is {a + b}"

# Run the agent
if __name__ == "__main__":
    agent.run()
```

## Constructor Parameters

```python
BedrockAgent(
    name: str = "bedrock_agent",      # Agent name
    route: str = "/bedrock",          # HTTP route for the agent
    system_prompt: Optional[str] = None,  # Initial system prompt
    voice_id: str = "matthew",        # Bedrock voice ID
    temperature: float = 0.7,         # Generation temperature (0-1)
    top_p: float = 0.9,              # Nucleus sampling parameter (0-1)
    max_tokens: int = 1024,          # Maximum tokens to generate
    **kwargs                         # Additional arguments passed to AgentBase
)
```

### Available Voice IDs

Common Bedrock voice IDs include:
- `matthew` - Male voice
- `joanna` - Female voice
- `ivy` - Female voice
- `justin` - Male voice
- `kendra` - Female voice
- `kimberly` - Female voice
- `salli` - Female voice
- `joey` - Male voice

## Available Methods

### All AgentBase Methods

BedrockAgent inherits all methods from AgentBase, including:

#### Prompt Management
- `set_prompt_text(text)` - Set the prompt as raw text
- `set_prompt_pom(pom)` - Set the prompt as a POM dictionary
- `prompt_add_section(title, body, bullets)` - Add sections to the prompt
- `set_post_prompt(text)` - Set post-prompt for summary generation

#### Skills
- `add_skill(skill_name, params)` - Add a skill to the agent
- `remove_skill(skill_name)` - Remove a skill
- `list_skills()` - List loaded skills
- `has_skill(skill_name)` - Check if a skill is loaded

#### Tools/Functions
- `@agent.tool` - Decorator for defining SWAIG tools
- `define_tool(name, description, parameters, handler)` - Define a tool programmatically
- `register_swaig_function(function_dict)` - Register a raw SWAIG function

#### Configuration
- `add_hint(hint)` - Add hints for the AI
- `add_language(name, code, voice)` - Add language configuration
- `set_params(params)` - Set AI parameters
- `set_global_data(data)` - Set global data available to AI

#### Web Server & Routing
- `run()` - Start the agent (auto-detects environment)
- `serve(host, port)` - Start web server explicitly
- `enable_sip_routing()` - Enable SIP-based routing
- `register_sip_username(username)` - Register SIP username

### BedrockAgent-Specific Methods

#### Voice Configuration
```python
agent.set_voice("joanna")  # Change the Bedrock voice
```

#### Inference Parameters
```python
# Update inference parameters
agent.set_inference_params(
    temperature=0.8,
    top_p=0.95,
    max_tokens=2048
)
```

### Disabled/Modified Methods

The following methods have modified behavior in BedrockAgent:

1. **`set_llm_model(model)`** - Logs warning and does nothing (Bedrock uses fixed model)
2. **`set_llm_temperature(temperature)`** - Redirects to `set_inference_params()`
3. **`set_post_prompt_llm_params(**params)`** - Logs warning (post-prompt uses OpenAI)
4. **`set_prompt_llm_params(**params)`** - Logs warning, use `set_inference_params()` instead

## SWML Output Structure

BedrockAgent generates SWML with the `amazon_bedrock` verb:

```json
{
  "version": "1.0.0",
  "sections": {
    "main": [
      {
        "answer": {}
      },
      {
        "amazon_bedrock": {
          "prompt": {
            "text": "Your system prompt here",
            "voice_id": "joanna",
            "temperature": 0.3,
            "top_p": 1.0,
            "barge_confidence": 0.0,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.1
          },
          "SWAIG": {
            "functions": [...],
            "defaults": {
              "web_hook_url": "http://..."
            }
          },
          "params": {
            "temperature": 0.8,
            "top_p": 0.9,
            "max_tokens": 1024
          },
          "global_data": {},
          "languages": [...],
          "hints": [...],
          "pronounce": [...]
        }
      }
    ]
  }
}
```

## Key Differences from Standard Agent

1. **Verb Name**: Generates `amazon_bedrock` verb instead of `ai`
2. **Voice Configuration**: Voice ID is included in the prompt object (Bedrock-specific)
3. **Fixed Model**: Cannot change the underlying model (always uses Bedrock voice-to-voice)
4. **Parameter Handling**: Simplified parameter configuration focused on inference settings
5. **Post-Prompt**: Post-prompt summarization still uses OpenAI

## Testing with swaig-test

BedrockAgent is fully compatible with swaig-test:

```bash
# List functions
swaig-test examples/bedrock_with_skills.py --list

# Dump SWML
swaig-test examples/bedrock_with_skills.py --dump

# Test a function
swaig-test examples/bedrock_with_skills.py --function calculate_sum --args '{"a": 5, "b": 3}'
```

## Complete Example

```python
#!/usr/bin/env python3
from signalwire_agents import BedrockAgent
import os

# Create a BedrockAgent with full configuration
agent = BedrockAgent(
    name="production_bedrock",
    route="/api/bedrock",
    system_prompt="You are a professional AI assistant.",
    voice_id="matthew",
    temperature=0.7,
    top_p=0.9,
    max_tokens=1024
)

# Configure prompt with POM
agent.prompt_add_section(
    "Role",
    "You are a helpful customer service agent.",
    bullets=[
        "Be professional and courteous",
        "Provide accurate information",
        "Ask clarifying questions when needed"
    ]
)

# Add languages
agent.add_language(
    name="Spanish",
    code="es",
    voice="miguel"  # Bedrock will map this appropriately
)

# Add skills
agent.add_skill("datetime")
if os.environ.get("WEATHER_API_KEY"):
    agent.add_skill("weather_api", {
        "api_key": os.environ["WEATHER_API_KEY"]
    })

# Add custom tools
@agent.tool
def transfer_to_support():
    """Transfer the call to support department"""
    return "transfer:support"

@agent.tool
def check_order_status(order_id: str):
    """Check the status of an order"""
    # Simulate order lookup
    return f"Order {order_id} is currently being processed and will ship tomorrow."

# Enable SIP routing
agent.enable_sip_routing()
agent.register_sip_username("bedrock-support")

# Run the agent
if __name__ == "__main__":
    print(f"Starting BedrockAgent on {agent.get_full_url()}")
    agent.run()
```

## Deployment Considerations

1. **Environment Variables**: Set API keys for skills that require them
2. **Authentication**: Uses same auth mechanism as standard agents (dev:w00t by default)
3. **Port Configuration**: Default port 3000, configurable via constructor or environment
4. **Production**: Use proper authentication and HTTPS in production environments

## Migration from Standard Agent

Migrating from a standard Agent to BedrockAgent is straightforward:

```python
# Before
from signalwire_agents import Agent
agent = Agent(name="my_agent")

# After
from signalwire_agents import BedrockAgent
agent = BedrockAgent(name="my_agent", voice_id="matthew")
```

Most code will work without modification. Only adjust:
- Voice configuration (use Bedrock voice IDs)
- Model-specific parameters (use `set_inference_params()`)
- Remove any `set_llm_model()` calls

## Troubleshooting

### Common Issues

1. **Voice not changing**: Ensure you're using valid Bedrock voice IDs
2. **Parameters not applying**: Use `set_inference_params()` instead of `set_prompt_llm_params()`
3. **Skills not loading**: Check API keys are properly configured
4. **SWML not generating**: Verify the agent is running and accessible

### Debug Mode

Enable debug logging to troubleshoot:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

agent = BedrockAgent(name="debug_agent")
agent.enable_debug_routes()
```

## Best Practices

1. **Voice Selection**: Choose appropriate voices for your use case and language
2. **Temperature Settings**: Lower values (0.3-0.5) for factual responses, higher (0.7-0.9) for creative tasks
3. **Skill Loading**: Load only necessary skills to minimize token usage
4. **Error Handling**: Implement proper error handling in custom tools
5. **Testing**: Use swaig-test extensively before deployment

## Limitations

- Cannot change the underlying AI model (Bedrock uses a fixed voice-to-voice model)
- Post-prompt summarization uses OpenAI for compatibility with existing integrations
- Text-specific features like hints and pronunciation rules don't apply to voice models
- Voice options limited to Bedrock's available voices

## See Also

- [AgentBase Documentation](agent_base.md)
- [Skills Documentation](skills.md)
- [SWAIG Functions Documentation](swaig_functions.md)
- [SignalWire AI Gateway Documentation](https://docs.signalwire.com/topics/ai-gateway/)

## Amazon Bedrock Verb Keys

The `amazon_bedrock` verb in SWML supports the following keys:

**Top-level keys:**
- `prompt` - Prompt configuration with text/POM and voice settings
- `SWAIG` - Function definitions and webhook configuration
- `params` - Inference parameters for the model
- `global_data` - Global data available to all functions
- `post_prompt` - Post-prompt text for summary generation
- `post_prompt_url` - URL for posting conversation summaries

**Within nested objects:**
- **prompt**: `text`, `pom`, `voice_id`, `temperature`, `top_p`
- **SWAIG**: `functions`, `defaults`
- **params**: `temperature`, `top_p`, `max_tokens`
- **global_data**: (any custom key-value pairs)

**Features not applicable to voice-to-voice models:**
- `languages` - Language configuration (voice models handle languages natively through voice)
- `hints` - AI hints (voice models process audio directly without text hints)
- `pronounce` - Pronunciation rules (not needed as voice input preserves pronunciation)
- `barge_confidence` - ASR confidence threshold (not applicable to voice-to-voice)
- `presence_penalty` - Topic diversity control (text model parameter)
- `frequency_penalty` - Repetition control (text model parameter)

These features are designed for text-based AI models and don't apply to Bedrock's voice-to-voice architecture.