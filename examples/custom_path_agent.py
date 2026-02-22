#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""


"""
Custom Path Agent Example

This example demonstrates how to create an agent with a custom route/path.
Instead of the default "/" route, this agent will be available at "/chat".

This is useful for:
- Running multiple agents on the same server at different paths
- Creating semantic URLs that describe the agent's purpose
- Organizing agents by department or function
- Hosting agents behind reverse proxies with path-based routing

Usage:
    python custom_path_agent.py

The agent will be available at:
    http://localhost:3000/chat

Try these requests:
    curl "http://localhost:3000/chat"
    curl "http://localhost:3000/chat?user_name=Alice&topic=AI"
    curl "http://localhost:3000/chat/debug"
"""

from signalwire_agents import AgentBase

class ChatAgent(AgentBase):
    def __init__(self, route="/chat"):
        super().__init__(
            name="Chat Assistant",
            route=route,  # Custom path for this agent
            auto_answer=True,
            record_call=True
        )
        
        # Set up dynamic configuration based on query parameters
        self.set_dynamic_config_callback(self.configure_chat_agent)
        
        # Base prompt
        self.prompt_add_section(
            "Role",
            "You are a friendly chat assistant ready to help with any questions or conversations."
        )

    def configure_chat_agent(self, query_params, body_params, headers, agent):
        """Configure the agent based on request parameters"""
        user_name = query_params.get('user_name', 'friend')
        topic = query_params.get('topic', 'general conversation')
        mood = query_params.get('mood', 'friendly').lower()
        
        # Personalize the greeting
        agent.prompt_add_section(
            "Personalization",
            f"The user's name is {user_name}. They're interested in discussing {topic}."
        )
        
        # Adjust voice and tone based on mood
        voice_model = "inworld.Mark"
        agent.add_language("English", "en-US", voice_model)
        
        if mood == 'professional':
            agent.prompt_add_section(
                "Communication Style",
                "Maintain a professional, business-appropriate tone in all interactions."
            )
        elif mood == 'casual':
            agent.prompt_add_section(
                "Communication Style", 
                "Use a casual, relaxed conversational style. Feel free to use informal language."
            )
        else:  # friendly (default)
            agent.prompt_add_section(
                "Communication Style",
                "Be warm, friendly, and approachable in your responses."
            )
        
        # Set global data for the conversation
        agent.set_global_data({
            "user_name": user_name,
            "topic": topic,
            "mood": mood,
            "session_type": "chat"
        })
        
        # Add speech recognition hints for common chat terms
        agent.add_hints([
            "chat",
            "assistant",
            "help", 
            "conversation",
            "question"
        ])


if __name__ == "__main__":
    # Create and start the agent
    agent = ChatAgent("/chat")
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    agent.run() 