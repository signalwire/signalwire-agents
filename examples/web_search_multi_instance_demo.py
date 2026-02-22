#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Web Search Multiple Instance Demo

This example demonstrates the enhanced web search skill with multiple instance support.
You can load the same skill multiple times with different configurations and tool names.

Features demonstrated:
- Multiple instances of the same skill (web_search)
- Custom tool names for each instance
- Different search engines and configurations per instance
- Custom no_results_message per instance
- Wikipedia search capability

To use this demo, you'll need:
- Google Custom Search API key (GOOGLE_SEARCH_API_KEY)
- Google Custom Search Engine IDs (GOOGLE_SEARCH_ENGINE_ID)
"""

import os
from signalwire_agents import AgentBase

def main():
    # Create an agent
    agent = AgentBase("Multi-Search Assistant", route="/search-demo")
    
    # Configure the agent with inworld.Mark voice
    agent.add_language("English", "en-US", "inworld.Mark")
    
    print("Creating agent with multiple web search skill instances...")
    
    # Add datetime and math skills for basic functionality
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
    
    # Add Wikipedia search skill
    try:
        agent.add_skill("wikipedia_search", {
            "num_results": 2,  # Get 2 Wikipedia articles by default
            "no_results_message": "I couldn't find any Wikipedia articles about '{query}'. Try using different keywords or check the spelling.",
            "swaig_fields": {
                "fillers": {
                    "en-US": [
                        "Let me check Wikipedia for that...",
                        "Searching Wikipedia...",
                        "Looking that up on Wikipedia..."
                    ]
                }
            }
        })
        print("Added Wikipedia search (tool: search_wiki)")
    except Exception as e:
        print(f"Failed to add Wikipedia skill: {e}")
    
    # Get credentials from environment variables
    google_api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
    google_search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
    
    if not google_api_key or not google_search_engine_id:
        print("Warning: Missing GOOGLE_SEARCH_API_KEY or GOOGLE_SEARCH_ENGINE_ID environment variables")
        print("Web search instances will not be available.")
        print("Set these environment variables to enable web search functionality.")
    else:
        # Instance 1: General web search (default tool name)
        try:
            agent.add_skill("web_search", {
                "api_key": google_api_key,
                "search_engine_id": google_search_engine_id,
                # No tool_name specified, so it will use default "web_search"
                "num_results": 1,
                "delay": 0,
                "no_results_message": "I couldn't find any information about '{query}' in my web search. Please try rephrasing your question.",
                "swaig_fields": {
                    "fillers": {
                        "en-US": [
                            "Let me search the web for that...",
                            "Searching the internet now...",
                            "Looking that up online..."
                        ]
                    }
                }
            })
            print("Added general web search (tool: web_search - default)")
        except Exception as e:
            print(f"Failed to add general web search skill: {e}")
        
        # Instance 2: News search (assuming you have a news-specific search engine)
        try:
            agent.add_skill("web_search", {
                "api_key": google_api_key,
                "search_engine_id": google_search_engine_id,  # You could use a different engine ID for news
                "tool_name": "search_news",
                "num_results": 3,  # More results for news
                "delay": 0.5,      # Small delay for news searches
                "no_results_message": "I couldn't find any recent news about '{query}'. Try searching for a different topic or current event.",
                "swaig_fields": {
                    "fillers": {
                        "en-US": [
                            "Checking the latest news...",
                            "Searching for recent articles...",
                            "Looking up current news stories..."
                        ]
                    }
                }
            })
            print("Added news search (tool: search_news)")
        except Exception as e:
            print(f"Failed to add news search skill: {e}")
        
        # Instance 3: Fast search for quick answers
        try:
            agent.add_skill("web_search", {
                "api_key": google_api_key,
                "search_engine_id": google_search_engine_id,
                "tool_name": "quick_search",
                "num_results": 1,  # Just one result for speed
                "delay": 0,        # No delay for maximum speed
                "no_results_message": "Quick search didn't find anything for '{query}'. Try using the regular web search for more comprehensive results.",
                "swaig_fields": {
                    "fillers": {
                        "en-US": [
                            "Quick search...",
                            "Fast lookup...",
                            "Rapid search..."
                        ]
                    }
                }
            })
            print("Added quick search (tool: quick_search)")
        except Exception as e:
            print(f"Failed to add quick search skill: {e}")
    
    # Show what skills/instances are loaded
    loaded_skills = agent.list_skills()
    print(f"\nLoaded skill instances: {', '.join(loaded_skills)}")
    
    # Show available skills from registry
    try:
        from signalwire_agents.skills.registry import skill_registry
        available_skills = skill_registry.list_skills()
        print(f"\nAvailable skills in registry:")
        for skill in available_skills:
            print(f"  - {skill['name']}: {skill['description']}")
            if skill['name'] == 'web_search':
                print(f"    Supports multiple instances: Yes")
            else:
                print(f"    Supports multiple instances: {skill.get('supports_multiple_instances', 'No')}")
    except Exception as e:
        print(f"Failed to list available skills: {e}")
    
    print(f"\nAgent available at: {agent.get_full_url()}")
    print("The agent now has enhanced search capabilities:")
    print("   - Can tell current date/time")
    print("   - Can perform mathematical calculations")
    print("   - Can search Wikipedia for factual information")
    
    # Count how many web search instances we have
    websearch_instances = [skill for skill in loaded_skills if skill.startswith('web_search_')]
    if websearch_instances:
        print(f"   - Has {len(websearch_instances)} web search capabilities:")
        for instance in websearch_instances:
            # Extract tool name from instance key (format: web_search_engineid_toolname)
            parts = instance.split('_')
            if len(parts) >= 3:
                tool_name = '_'.join(parts[2:])  # Join all parts after the second underscore
                print(f"     * {tool_name}")
    
    print("\nWeb Search Multiple Instance Examples:")
    print("   # Basic usage with default tool name")
    print("   agent.add_skill('web_search', {")
    print("       'api_key': 'your-api-key',")
    print("       'search_engine_id': 'your-engine-id'")
    print("   })")
    print("   # Creates tool: web_search")
    print("   ")
    print("   # Custom tool name for news search")
    print("   agent.add_skill('web_search', {")
    print("       'api_key': 'your-api-key',")
    print("       'search_engine_id': 'news-engine-id',")
    print("       'tool_name': 'search_news',")
    print("       'num_results': 3,")
    print("       'delay': 0.5")
    print("   })")
    print("   # Creates tool: search_news")
    print("   ")
    print("   # Wikipedia search (single instance only)")
    print("   agent.add_skill('wikipedia_search', {")
    print("       'num_results': 2,")
    print("       'no_results_message': 'No Wikipedia articles found for {query}'")
    print("   })")
    print("   # Creates tool: search_wiki")
    print("   ")
    print("   # Multiple instances with different configurations")
    print("   agent.add_skill('web_search', {..., 'tool_name': 'search_products'})")
    print("   agent.add_skill('web_search', {..., 'tool_name': 'search_support'})")
    print("   agent.add_skill('web_search', {..., 'tool_name': 'quick_search'})")
    
    print("\nStarting agent server...")
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    agent.run()

if __name__ == "__main__":
    main() 