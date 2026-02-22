#!/usr/bin/env python3
"""
Test BedrockAgent with skills

This example demonstrates how BedrockAgent works exactly like a regular agent
but outputs SWML with the amazon_bedrock verb.
"""

from signalwire_agents import BedrockAgent

# Create a Bedrock agent
agent = BedrockAgent(
    name="test_bedrock",
    route="/bedrock_test",
    system_prompt="You are a helpful AI assistant powered by Amazon Bedrock.",
    voice_id="inworld.Mark",
    temperature=0.7
)

# Add a simple function
@agent.tool("Get the current weather for a location")
def get_weather(location: str, unit: str = "celsius"):
    """Get weather for a location"""
    # Mock weather data
    return {
        "location": location,
        "temperature": 22,
        "unit": unit,
        "conditions": "sunny"
    }

# Skills can be loaded if needed
# agent.add_skill("datetime")

# Export the agent for testing
test_agent = agent

if __name__ == "__main__":
    print(f"BedrockAgent created: {agent}")
    print(f"Name: {agent.get_name()}")
    print(f"Route: {agent.route}")
    print(f"Functions registered: 1 (get_weather)")