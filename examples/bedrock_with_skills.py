#!/usr/bin/env python3
"""
Test BedrockAgent with skills
"""

import os
from signalwire_agents import BedrockAgent

# Create a Bedrock agent
agent = BedrockAgent(
    name="bedrock_skills_test",
    voice_id="inworld.Mark",
    temperature=0.8
)

# Set up the agent's personality using POM (Prompt Object Model)
agent.prompt_add_section(
    "Personality",
    "You are a helpful AI assistant with access to various skills including real-time information lookup, weather data, and web search capabilities."
)

agent.prompt_add_section(
    "Instructions",
    bullets=[
        "Use the current_time function when users ask about the time or date",
        "Use the getWeather function when users ask about weather conditions",
        "Use the web_search function when users need current information from the internet",
        "Always provide accurate and helpful responses",
        "Be conversational and friendly"
    ]
)

# Add skills
agent.add_skill("datetime")
print("✓ datetime skill added")

# Add weather_api skill with API key
agent.add_skill("weather_api", {
    "api_key": os.environ.get("WEATHER_API_KEY", "")
})
print("✓ weather_api skill added")


# Add web_search skill with Google Custom Search API credentials
google_api_key = os.environ.get("GOOGLE_SEARCH_API_KEY", "")
google_search_engine_id = os.environ.get("GOOGLE_SEARCH_ENGINE_ID", "")

if google_api_key and google_search_engine_id:
    agent.add_skill("web_search", {
        "api_key": google_api_key,
        "search_engine_id": google_search_engine_id
    })
    print("✓ web_search skill added")
else:
    print("⚠ web_search skill skipped - Set GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables")

# Note: joke skill has a prompt configuration issue
# agent.add_skill("joke", {"api_key": os.environ.get("API_NINJAS_KEY", "")})

# List functions
if hasattr(agent, '_tool_registry'):
    functions = agent._tool_registry.get_all_functions()
    print(f"\nRegistered functions: {list(functions.keys())}")

# Print agent info
print(f"\nAgent: {agent}")

if __name__ == "__main__":
    # Export for testing
    test_agent = agent
    
    # Run as SWML server
    agent.run()
