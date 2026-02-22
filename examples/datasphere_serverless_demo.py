#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
DataSphere Serverless Demo

This example demonstrates the new DataSphere Serverless skill using DataMap.
It provides the same functionality as the standard datasphere skill but executes
on SignalWire servers rather than the agent server.

Features demonstrated:
- Serverless execution via DataMap
- Same parameters as standard datasphere skill
- Multiple instances with different configurations
- No webhook infrastructure required

To use this demo, you'll need:
- SignalWire DataSphere setup with documents
- Valid space_name, project_id, token, and document_id values
"""

import os
from signalwire_agents import AgentBase

def main():
    # Create an agent
    agent = AgentBase("DataSphere Serverless Assistant", route="/datasphere-serverless-demo")
    
    # Configure the agent with inworld.Mark voice
    agent.add_language("English", "en-US", "inworld.Mark")
    
    print("Creating agent with DataSphere Serverless skill (DataMap implementation)...")
    
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
    
    # Example configuration - replace with your actual DataSphere details
    example_config = {
        'space_name': 'your-space',
        'project_id': 'your-project-id', 
        'token': 'your-token'
    }
    
    # Instance 1: Drinks knowledge base (serverless)
    try:
        agent.add_skill("datasphere_serverless", {
            **example_config,
            "document_id": "drinks-doc-123",
            "tool_name": "search_drinks_serverless",
            "tags": ["Drinks", "Bar", "Cocktails"],
            "count": 2,
            "distance": 5.0,
            "no_results_message": "I couldn't find any drink recipes or information about '{query}' using serverless search. Try asking about a different cocktail or beverage.",
            "swaig_fields": {
                "fillers": {
                    "en-US": [
                        "Searching drink recipes on SignalWire servers...",
                        "Looking up cocktail information serverlessly...",
                        "Checking our drink database via DataMap..."
                    ]
                }
            }
        })
        print("Added serverless drinks knowledge search (tool: search_drinks_serverless)")
    except Exception as e:
        print(f"Failed to add drinks DataSphere Serverless skill: {e}")
        print("   Note: Replace example_config with your actual DataSphere credentials")
    
    # Instance 2: Food knowledge base (serverless)
    try:
        agent.add_skill("datasphere_serverless", {
            **example_config,
            "document_id": "food-doc-456",
            "tool_name": "search_food_serverless", 
            "tags": ["Food", "Recipes", "Cooking"],
            "count": 3,
            "distance": 4.0,
            "language": "en",
            "no_results_message": "I couldn't find any recipes or cooking information about '{query}' using serverless search. Try asking about a different dish or ingredient.",
            "swaig_fields": {
                "fillers": {
                    "en-US": [
                        "Searching recipes on SignalWire infrastructure...",
                        "Looking up cooking instructions serverlessly...",
                        "Checking our food database via DataMap..."
                    ]
                }
            }
        })
        print("Added serverless food knowledge search (tool: search_food_serverless)")
    except Exception as e:
        print(f"Failed to add food DataSphere Serverless skill: {e}")
        print("   Note: Replace example_config with your actual DataSphere credentials")
    
    # Instance 3: General knowledge base with default tool name (serverless)
    try:
        agent.add_skill("datasphere_serverless", {
            **example_config,
            "document_id": "general-doc-789",
            # No tool_name specified, so it will use default "search_knowledge"
            "count": 1,
            "distance": 3.0,
            "no_results_message": "I couldn't find information about '{query}' in our general knowledge base using serverless search. Please try rephrasing your question."
        })
        print("Added serverless general knowledge search (tool: search_knowledge - default)")
    except Exception as e:
        print(f"Failed to add general DataSphere Serverless skill: {e}")
        print("   Note: Replace example_config with your actual DataSphere credentials")
    
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
            if skill['name'] in ['datasphere', 'datasphere_serverless']:
                print(f"    Supports multiple instances: Yes")
                if skill['name'] == 'datasphere_serverless':
                    print(f"    Execution: Serverless (DataMap)")
                else:
                    print(f"    Execution: Agent server")
            else:
                print(f"    Supports multiple instances: No")
    except Exception as e:
        print(f"Failed to list available skills: {e}")
    
    print(f"\nAgent available at: {agent.get_full_url()}")
    print("The agent now has enhanced capabilities:")
    print("   - Can tell current date/time")
    print("   - Can perform mathematical calculations") 
    
    # Count how many DataSphere instances we have
    datasphere_instances = [skill for skill in loaded_skills if skill.startswith('datasphere_')]
    if datasphere_instances:
        print(f"   - Has {len(datasphere_instances)} knowledge search capabilities:")
        for instance in datasphere_instances:
            tool_name = instance.split('_', 1)[1] if '_' in instance else 'search_knowledge'
            execution_type = "Serverless" if "serverless" in instance else "Standard"
            print(f"     * {tool_name} ({execution_type})")
    
    print("\nDataSphere Serverless vs Standard Comparison:")
    print("   ┌─────────────────────┬─────────────────────┬─────────────────────┐")
    print("   │ Feature             │ Standard DataSphere │ Serverless DataMap  │")
    print("   ├─────────────────────┼─────────────────────┼─────────────────────┤")
    print("   │ Execution           │ Agent server        │ SignalWire servers  │")
    print("   │ Parameters          │ Identical           │ Identical           │")
    print("   │ Webhook endpoints   │ Required            │ Not required        │")
    print("   │ Performance impact  │ Uses agent resources│ No agent impact     │")
    print("   │ Response formatting │ Custom Python code  │ DataMap templates   │")
    print("   │ Error handling      │ Granular exceptions │ DataMap error keys  │")
    print("   │ Multiple instances  │ Yes                 │ Yes                 │")
    print("   └─────────────────────┴─────────────────────┴─────────────────────┘")
    
    print("\nDataSphere Serverless Examples:")
    print("   # Basic usage - identical API to standard skill")
    print("   agent.add_skill('datasphere_serverless', {")
    print("       'space_name': 'my-space',")
    print("       'project_id': 'my-project',") 
    print("       'token': 'my-token',")
    print("       'document_id': 'my-doc-id'")
    print("   })")
    print("   # Executes on SignalWire servers, no webhook infrastructure needed")
    print("   ")
    print("   # Multiple instances with custom tool names")
    print("   agent.add_skill('datasphere_serverless', {")
    print("       'space_name': 'my-space',")
    print("       'project_id': 'my-project',")
    print("       'token': 'my-token',")
    print("       'document_id': 'drinks-doc',")
    print("       'tool_name': 'search_drinks_serverless',")
    print("       'tags': ['Drinks', 'Cocktails'],")
    print("       'count': 3")
    print("   })")
    print("   # Creates serverless tool: search_drinks_serverless")
    print("   ")
    print("   # Benefits:")
    print("   # - No HTTP endpoints to expose")
    print("   # - Executes on SignalWire infrastructure") 
    print("   # - Same functionality as standard skill")
    print("   # - Simplified deployment")
    
    print("\nStarting agent server...")
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    agent.run()

if __name__ == "__main__":
    main() 