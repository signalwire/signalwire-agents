#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
DataSphere Multiple Instance Demo

This example demonstrates the new DataSphere skill with multiple instance support.
You can load the same skill multiple times with different configurations and tool names.

Features demonstrated:
- Multiple instances of the same skill (datasphere)
- Custom tool names for each instance
- Different configurations per instance
- Custom no_results_message per instance

To use this demo, you'll need:
- SignalWire DataSphere setup with documents
- Valid space_name, project_id, token, and document_id values
"""

import os
from signalwire_agents import AgentBase

def main():
    # Create an agent
    agent = AgentBase("Multi-DataSphere Assistant", route="/datasphere-demo")
    
    # Configure the agent with inworld.Mark voice
    agent.add_language("English", "en-US", "inworld.Mark")
    
    print("Creating agent with multiple DataSphere skill instances...")
    
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
    
    # Instance 1: Drinks knowledge base
    try:
        agent.add_skill("datasphere", {
            **example_config,
            "document_id": "drinks-doc-123",
            "tool_name": "search_drinks_knowledge",
            "tags": ["Drinks", "Bar", "Cocktails"],
            "count": 2,
            "distance": 5.0,
            "no_results_message": "I couldn't find any drink recipes or information about '{query}'. Try asking about a different cocktail or beverage.",
            "swaig_fields": {
                "fillers": {
                    "en-US": [
                        "Let me check our drink recipes...",
                        "Searching our cocktail database...",
                        "Looking up drink information..."
                    ]
                }
            }
        })
        print("Added drinks knowledge search (tool: search_drinks_knowledge)")
    except Exception as e:
        print(f"Failed to add drinks DataSphere skill: {e}")
        print("   Note: Replace example_config with your actual DataSphere credentials")
    
    # Instance 2: Food knowledge base
    try:
        agent.add_skill("datasphere", {
            **example_config,
            "document_id": "food-doc-456",
            "tool_name": "search_food_knowledge", 
            "tags": ["Food", "Recipes", "Cooking"],
            "count": 3,
            "distance": 4.0,
            "language": "en",
            "no_results_message": "I couldn't find any recipes or cooking information about '{query}'. Try asking about a different dish or ingredient.",
            "swaig_fields": {
                "fillers": {
                    "en-US": [
                        "Checking our recipe collection...",
                        "Searching cooking instructions...",
                        "Looking through our food database..."
                    ]
                }
            }
        })
        print("Added food knowledge search (tool: search_food_knowledge)")
    except Exception as e:
        print(f"Failed to add food DataSphere skill: {e}")
        print("   Note: Replace example_config with your actual DataSphere credentials")
    
    # Instance 3: General knowledge base with default tool name
    try:
        agent.add_skill("datasphere", {
            **example_config,
            "document_id": "general-doc-789",
            # No tool_name specified, so it will use default "search_knowledge"
            "count": 1,
            "distance": 3.0,
            "no_results_message": "I couldn't find information about '{query}' in our general knowledge base. Please try rephrasing your question."
        })
        print("Added general knowledge search (tool: search_knowledge - default)")
    except Exception as e:
        print(f"Failed to add general DataSphere skill: {e}")
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
            if skill['name'] == 'datasphere':
                print(f"    Supports multiple instances: Yes")
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
            print(f"     * {tool_name}")
    
    print("\nDataSphere Multiple Instance Examples:")
    print("   # Basic usage with default tool name")
    print("   agent.add_skill('datasphere', {")
    print("       'space_name': 'my-space',")
    print("       'project_id': 'my-project',") 
    print("       'token': 'my-token',")
    print("       'document_id': 'my-doc-id'")
    print("   })")
    print("   # Creates tool: search_knowledge")
    print("   ")
    print("   # Custom tool name for drinks knowledge")
    print("   agent.add_skill('datasphere', {")
    print("       'space_name': 'my-space',")
    print("       'project_id': 'my-project',")
    print("       'token': 'my-token',")
    print("       'document_id': 'drinks-doc',")
    print("       'tool_name': 'search_drinks',")
    print("       'tags': ['Drinks', 'Cocktails'],")
    print("       'count': 3")
    print("   })")
    print("   # Creates tool: search_drinks")
    print("   ")
    print("   # Multiple instances with different configurations")
    print("   agent.add_skill('datasphere', {..., 'tool_name': 'search_products'})")
    print("   agent.add_skill('datasphere', {..., 'tool_name': 'search_support'})")
    print("   agent.add_skill('datasphere', {..., 'tool_name': 'search_policies'})")
    
    print("\nStarting agent server...")
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    agent.run()

if __name__ == "__main__":
    main() 