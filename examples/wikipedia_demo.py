#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Wikipedia Search Skill Demo

This example demonstrates the Wikipedia search skill for factual information retrieval.

Features demonstrated:
- Wikipedia article search and summaries
- Custom no_results_message with query placeholder
- Multiple result configuration
- SWAIG fillers for better user experience
"""

from signalwire_agents import AgentBase

def main():
    # Create an agent focused on Wikipedia search
    agent = AgentBase("Wikipedia Assistant", route="/wiki-demo")
    
    # Configure the agent with a clear voice
    agent.add_language("English", "en-US", "inworld.Mark")
    
    print("Creating Wikipedia search assistant...")
    
    # Add basic skills
    try:
        agent.add_skill("datetime")
        print("Added datetime skill")
    except Exception as e:
        print(f"Failed to add datetime skill: {e}")
    
    # Add Wikipedia search skill with custom configuration
    try:
        agent.add_skill("wikipedia_search", {
            "num_results": 2,  # Get up to 2 articles for broader coverage
            "no_results_message": "I couldn't find any Wikipedia articles about '{query}'. You might want to try different keywords, check the spelling, or ask about a related topic.",
            "swaig_fields": {
                "fillers": {
                    "en-US": [
                        "Let me search Wikipedia for that information...",
                        "Checking Wikipedia's knowledge base...",
                        "Looking that up in the encyclopedia...",
                        "Searching for factual information...",
                        "Let me find that on Wikipedia..."
                    ]
                }
            }
        })
        print("Added Wikipedia search (tool: search_wiki)")
    except Exception as e:
        print(f"Failed to add Wikipedia skill: {e}")
        return
    
    # Show loaded skills
    loaded_skills = agent.list_skills()
    print(f"\nLoaded skills: {', '.join(loaded_skills)}")
    
    print(f"\nWikipedia Assistant available at: {agent.get_full_url()}")
    print("\nThis agent specializes in:")
    print("   - Searching Wikipedia for factual information")
    print("   - Providing article summaries and encyclopedic knowledge")
    print("   - Answering questions about people, places, concepts, and history")
    print("   - Current date and time information")
    
    print("\nExample queries to try:")
    print("   'Tell me about Albert Einstein'")
    print("   'What is quantum physics?'")
    print("   'Who was Marie Curie?'")
    print("   'Search for information about the Roman Empire'")
    print("   'Look up Python programming language'")
    print("   'What is artificial intelligence?'")
    
    print("\nWikipedia Skill Configuration:")
    print("   - Returns up to 2 article summaries per search")
    print("   - Custom no-results message with helpful suggestions")
    print("   - Enhanced fillers for better user experience")
    print("   - Tool name: search_wiki")
    
    print("\nStarting Wikipedia Assistant...")
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    agent.run()

if __name__ == "__main__":
    main() 