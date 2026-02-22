#!/usr/bin/env python3
"""
Test BedrockAgent as a SWML server
"""

import os
from signalwire_agents import BedrockAgent

# Create a Bedrock agent
agent = BedrockAgent(
    name="bedrock_server",
    system_prompt="You are a helpful AI assistant.",
    voice_id="inworld.Mark"
)

# Add a simple tool
@agent.tool
def hello_world(name: str = "World"):
    """Say hello to someone"""
    return f"Hello, {name}!"

print(f"Starting BedrockAgent server...")
print(f"Route: /bedrock")
print(f"URL: http://localhost:3000/bedrock")
print(f"Auth: dev:w00t")
print("\nTo test:")
print("  curl -u dev:w00t http://localhost:3000/bedrock")
print("  swaig-test examples/bedrock_server_test.py --list")
print("\nPress Ctrl+C to stop")

if __name__ == "__main__":
    agent.run()