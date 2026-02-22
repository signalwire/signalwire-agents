#!/usr/bin/env python3
"""
Test BedrockAgent with standard agent run
"""

from signalwire_agents import BedrockAgent, run_agent

# Create a Bedrock agent
agent = BedrockAgent(
    name="bedrock_run_test",
    system_prompt="You are a helpful AI assistant powered by Amazon Bedrock.",
    voice_id="inworld.Mark",
    temperature=0.7
)

# Add a function
@agent.tool("Get the current time")
def get_time():
    """Get the current time"""
    from datetime import datetime
    return {"time": datetime.now().strftime("%H:%M:%S")}

# Run the agent
if __name__ == "__main__":
    print("Starting BedrockAgent with run_agent()...")
    run_agent(agent)