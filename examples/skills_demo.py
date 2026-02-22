#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Skills System Demo

This example demonstrates the new modular skills system for SignalWire agents.
Skills are automatically discovered and can be added with simple one-liner calls.

Features demonstrated:
- Basic skill loading with agent.add_skill()
- Skill parameter configuration (num_results, delay)
- swaig_fields for customizing SWAIG function properties (fillers, security)

To use the web_search skill, you'll need:
- GOOGLE_SEARCH_API_KEY environment variable
- GOOGLE_SEARCH_ENGINE_ID environment variable

The datetime and math skills work without any additional setup.
"""

import os
from signalwire_agents import AgentBase

def main():
    # Create an agent
    agent = AgentBase("Multi-Skill Assistant", route="/assistant")
    
    # Configure the agent with inworld.Mark voice
    agent.add_language("English", "en-US", "inworld.Mark")
    
    print("Creating agent with multiple skills...")
    
    # Add skills using the new system - these are one-liners!
    try:
        agent.add_skill("datetime")
        print("Added datetime skill")
    except Exception as e:
        print(f"Failed to add datetime skill: {e}")
    
    try:
        agent.add_skill("math")
        print("Added math skill")
    except Exception as e:
        print(f"Failed to add math skill: {e}")
    
    try:
        # Get credentials from environment variables
        google_api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        google_search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
        
        if not google_api_key or not google_search_engine_id:
            raise ValueError("Missing GOOGLE_SEARCH_API_KEY or GOOGLE_SEARCH_ENGINE_ID environment variables")
        
        # Add web search with custom parameters and swaig_fields for fillers
        # Pass API credentials as parameters instead of using env vars
        agent.add_skill("web_search", {
            "api_key": google_api_key,
            "search_engine_id": google_search_engine_id,
            "num_results": 1,  # Just get one result for faster responses
            "delay": 0,        # No delay between requests for minimal latency
            "no_results_message": "I apologize, but I wasn't able to find any information about '{query}' in my web search. Could you try rephrasing your question or asking about something else?",
            "swaig_fields": {  # Special fields merged into SWAIG function definition
                "fillers": {
                    "en-US": [
                        "I am searching the web for that information...",
                        "Let me google that for you...",
                        "Searching the internet now...",
                        "Looking that up on the web...",
                        "Finding the latest information online..."
                    ]
                }
            }
        })
        print("Added web search skill with optimized parameters and custom fillers")
    except Exception as e:
        print(f"Failed to add web search skill: {e}")
        print("   Note: Web search requires GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID environment variables")
    
    # Show what skills are loaded
    loaded_skills = agent.list_skills()
    print(f"\nLoaded skills: {', '.join(loaded_skills)}")
    
    # Show available skills from registry
    try:
        from signalwire_agents.skills.registry import skill_registry
        available_skills = skill_registry.list_skills()
        print(f"\nAvailable skills in registry:")
        for skill in available_skills:
            print(f"  - {skill['name']}: {skill['description']}")
            if skill['required_env_vars']:
                print(f"    Requires env vars: {', '.join(skill['required_env_vars'])}")
            if skill['required_packages']:
                print(f"    Requires packages: {', '.join(skill['required_packages'])}")
            if skill['supports_multiple_instances']:
                print(f"    Supports multiple instances: Yes")
            else:
                print(f"    Supports multiple instances: No")
    except Exception as e:
        print(f"Failed to list available skills: {e}")
    
    print(f"\nAgent available at: {agent.get_full_url()}")
    print("The agent now has enhanced capabilities:")
    print("   - Can tell current date/time")
    print("   - Can perform mathematical calculations")
    if "web_search" in loaded_skills:
        print("   - Can search the web for information with custom fillers ('Let me google that...')")
        print("     (optimized for speed: 1 result, no delay)")
    
    print("\nSkill Parameter Examples:")
    print("   # Default web search (requires API credentials)")
    print("   agent.add_skill('web_search', {")
    print("       'api_key': 'your-google-api-key',")
    print("       'search_engine_id': 'your-search-engine-id'")
    print("   })")
    print("   ")
    print("   # Custom web search (3 results, 0.5s delay)")
    print("   agent.add_skill('web_search', {")
    print("       'api_key': 'your-api-key',")
    print("       'search_engine_id': 'your-engine-id',")
    print("       'num_results': 3,")
    print("       'delay': 0.5")
    print("   })")
    print("   ")
    print("   # Multiple web search instances with different engines")
    print("   agent.add_skill('web_search', {")
    print("       'api_key': 'your-api-key',")
    print("       'search_engine_id': 'general-engine-id',")
    print("       'tool_name': 'search_general',")
    print("       'num_results': 1")
    print("   })")
    print("   agent.add_skill('web_search', {")
    print("       'api_key': 'your-api-key',")
    print("       'search_engine_id': 'news-engine-id',") 
    print("       'tool_name': 'search_news',")
    print("       'num_results': 3")
    print("   })")
    print("   ")
    print("   # Web search with custom no results message")
    print("   agent.add_skill('web_search', {")
    print("       'api_key': 'your-api-key',")
    print("       'search_engine_id': 'your-engine-id',")
    print("       'no_results_message': 'Sorry, no results found for \"{query}\". Please try a different search.'")
    print("   })")
    print("   ")
    print("   # Web search with custom fillers using swaig_fields")
    print("   agent.add_skill('web_search', {")
    print("       'api_key': 'your-api-key',")
    print("       'search_engine_id': 'your-engine-id',")
    print("       'swaig_fields': {")
    print("           'fillers': {'en-US': ['Let me google that...', 'Searching now...']}") 
    print("       }")
    print("   })")
    
    print("\nStarting agent server...")
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    agent.run()

if __name__ == "__main__":
    main() 