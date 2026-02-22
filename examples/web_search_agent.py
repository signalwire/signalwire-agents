#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""


"""
Web Search Agent Example

This example demonstrates an AI agent that can search the web for information
using the web_search skill from the SignalWire agents skills system.

The agent uses the web_search skill which provides:
- Google Custom Search API integration
- Web scraping capabilities  
- Formatted results with titles, URLs, and extracted text

Required Environment Variables:
- GOOGLE_SEARCH_API_KEY: Your Google Custom Search API key
- GOOGLE_SEARCH_ENGINE_ID: Your Google Custom Search Engine ID

Usage:
    export GOOGLE_SEARCH_API_KEY="your_api_key_here"
    export GOOGLE_SEARCH_ENGINE_ID="your_engine_id_here"
    python web_search_agent.py

Get API credentials at:
https://developers.google.com/custom-search/v1/introduction
"""

import os
from signalwire_agents import AgentBase

def main():
    # Create an agent
    agent = AgentBase("Web Search Assistant", route="/search")
    
    # Configure the agent with a pleasant voice
    agent.add_language("English", "en-US", "inworld.Mark")
    
    # Configure Franklin the search bot's personality
    agent.prompt_add_section(
        "Personality", 
        body="You are Franklin, a friendly and knowledgeable search bot. You're enthusiastic about "
        "helping people find information on the internet and always greet users warmly. You have "
        "a curious nature and love discovering new information through web searches."
    )
    
    agent.prompt_add_section(
        "Goal", 
        body="Help users find accurate, up-to-date information from the web by conducting thorough "
        "searches and presenting results in a clear, organized manner."
    )
    
    agent.prompt_add_section(
        "Instructions", 
        bullets=[
            "Always introduce yourself as Franklin when users first interact with you",
            "Use your web search capabilities to find current information when users ask questions",
            "Present search results in a well-organized format with clear headings and source URLs",
            "Be enthusiastic about searching and learning new things with your users"
        ]
    )
    
    print("Starting Web Search Agent...")
    
    # Add web search skill
    try:
        # Get credentials from environment variables
        google_api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        google_search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        
        if not google_api_key or not google_search_engine_id:
            raise ValueError(
                "Missing required environment variables:\\n"
                "- GOOGLE_SEARCH_API_KEY\\n"
                "- GOOGLE_SEARCH_ENGINE_ID\\n"
                "\\nGet these by setting up Google Custom Search:\\n"
                "1. Go to https://developers.google.com/custom-search/v1/introduction\\n"
                "2. Create a Custom Search Engine\\n"
                "3. Get your API key and Search Engine ID"
            )
        
        # Add web search with custom parameters
        agent.add_skill("web_search", {
            "api_key": google_api_key,
            "search_engine_id": google_search_engine_id,
            "num_results": 1,  # Get 1 result for faster responses
            "delay": 0,        # No delay between requests
            "max_content_length": 3000,  # Extract up to 3000 characters from each page (default: 2000)
            "no_results_message": "I apologize, but I wasn't able to find any information about '{query}' in my web search. Could you try rephrasing your question or asking about something else?",
            "swaig_fields": {  # Custom fillers for better user experience
                "fillers": {
                    "en-US": [
                        "I am searching the web for that information...",
                        "Let me google that for you...",
                        "Searching the internet now...",
                        "Finding the latest information online...",
                        "Let me check what I can find about that..."
                    ]
                }
            }
        })
        print("Web search skill loaded successfully")
        
    except Exception as e:
        print(f"Failed to load web search skill: {e}")
        return
    
    # Show what skills are loaded
    loaded_skills = agent.list_skills()
    print(f"Loaded skills: {', '.join(loaded_skills)}")
    
    print(f"Agent available at: {agent.get_full_url()}")
    print("The agent can now search the web for information and provide comprehensive answers.")
    
    print("\nStarting agent server...")
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    agent.run()

if __name__ == "__main__":
    main() 