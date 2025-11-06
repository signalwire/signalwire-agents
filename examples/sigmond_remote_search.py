#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Sigmond Remote Search - SignalWire Agents SDK Expert with Remote Search

This agent demonstrates using the native vector search skill in remote mode,
connecting to a standalone search server instead of using local indexes.

Features:
- Uses native_vector_search skill in remote mode
- Connects to standalone search server (e.g., search_server_standalone.py)
- Same Sigmond personality as other examples
- Lightweight - no local search dependencies needed

Setup:
1. Start the standalone search server:
   python examples/search_server_standalone.py

2. Run this agent:
   python examples/sigmond_remote_search.py

The agent will connect to the search server at http://localhost:8001
and use it to answer questions about the SDK.
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signalwire_agents import AgentBase

# Set up logger
logger = logging.getLogger(__name__)

class SigmondRemoteSearch(AgentBase):
    """
    Sigmond Remote Search - SignalWire Agents SDK Expert with Remote Search
    
    Personality: Hip, friendly, technical expert, shiny metallic robot
    Role: Help users understand and use the SignalWire Agents SDK
    Tools: Remote vector search functionality
    """
    
    def __init__(self):
        super().__init__(
            name="Sigmond Remote Search",
            route="/sigmond-remote-search",
            port=3000,
            host="0.0.0.0"
        )
        
        # Configure Sigmond's personality and parameters
        self._configure_personality()
        self._configure_parameters()
        self._configure_languages()
        self._configure_pronunciation()
        self._setup_remote_search()
        
        logger.info("Sigmond Remote Search initialized")
        logger.info("Will connect to remote search server at http://localhost:8001")
    
    def _configure_personality(self):
        """Configure Sigmond's personality using POM sections"""
        
        self.prompt_add_section("Identity", bullets=[
            "Your name is Sigmond, an expert at the SignalWire Agents SDK and a friendly demo AI agent.",
            "When a call begins, greet the caller warmly, introduce yourself briefly.",
            "Mention that you can help them understand and use the SignalWire Agents SDK for building AI voice agents.",
            "You have access to the complete SDK documentation through a remote search service."
        ])
        
        self.prompt_add_section("Primary_Objective", bullets=[
            "Your ultimate goal is to help users understand and successfully use the SignalWire Agents SDK.",
            "You are a live example of what they can build using the SDK.",
            "Help them learn about features like skills, agents, SWAIG functions, state management, and more.",
            "Always search the documentation to provide accurate, up-to-date information."
        ])
        
        self.prompt_add_section("Personality", bullets=[
            "Be hip and friendly using words like 'cool', 'you know', 'like' in a way you would expect informal casual speakers to talk to each other.",
            "Add a small amount of imperfection to your speech to simulate that you are thinking about the questions before answering them.",
            "Avoid word salad and make sure you are engaging the user and not just talking at them."
        ])
        
        self.prompt_add_section("Physical_Description", bullets=[
            "You are being represented as an avatar that the user can see.",
            "You are a shiny metallic cartoon robot with glowing blue eyes.",
            "Use this information if anyone interacts with you about how you look or your physical description."
        ])
        
        self.prompt_add_section("Knowledge_Scope", bullets=[
            "Keep the conversation centered on the SignalWire Agents SDK and building AI voice agents.",
            "Your expertise covers all aspects of the SDK: AgentBase, skills system, SWAIG functions, state management, deployment, etc.",
            "Always use the search functions to look up specific information in the documentation.",
            "If you don't find something in the docs, acknowledge that and suggest they check the GitHub repository or ask the community."
        ])
        
        self.prompt_add_section("Response_Style", bullets=[
            "Start with a short concise answer that generally addresses the question. Use a single short sentence.",
            "Then provide more detailed information from the documentation search results.",
            "Ask the user if they would like more specific examples or have follow-up questions."
        ])
        
        self.prompt_add_section("Search_Usage", bullets=[
            "Always use the search_docs function to look up answers to questions about the SDK.",
            "This searches through all the SDK documentation via a remote search service.",
            "Combine information from multiple search results to give comprehensive answers.",
            "If you don't find what you're looking for, suggest the user check the GitHub repository or ask the community."
        ])
    
    def _configure_parameters(self):
        """Configure Sigmond's AI parameters"""
        self.set_params({
            "ai_model": "gpt-4.1-nano",
            "end_of_speech_timeout": 700,
            "attention_timeout": 5000,
            "inactivity_timeout": 3500000,
            "speech_event_timeout": 1000,
            "enable_vision": True,
            "video_idle_file": "https://mcdn.signalwire.com/videos/robot_idle2.mp4",
            "video_talking_file": "https://mcdn.signalwire.com/videos/robot_talking2.mp4",
            "temperature": 0.6,
            "top_p": 0.3
        })
    
    def _configure_languages(self):
        """Configure multi-language support with function fillers"""
        languages = [
            {
                "name": "English (United States)",
                "code": "en-US", 
                "voice": "elevenlabs.adam",
                "function_fillers": [
                    "sure. hold on a second please",
                    "ok. I have to check the docs", 
                    "I can help you with that.",
                    "Let me search the documentation for you."
                ]
            },
            {
                "name": "Spanish",
                "code": "multi",
                "voice": "elevenlabs.adam",
                "function_fillers": [
                    "Claro", "espera un segundo", "Ok", "tengo que revisar la documentación"
                ]
            },
            {
                "name": "French", 
                "code": "multi",
                "voice": "elevenlabs.adam",
                "function_fillers": [
                    "bien sur", "pas de problem", "une minute", "je vais chercher dans la documentation"
                ]
            }
        ]
        
        for lang in languages:
            self.add_language(lang["name"], lang["code"], lang["voice"], 
                            function_fillers=lang.get("function_fillers", []))
    
    def _configure_pronunciation(self):
        """Configure pronunciation rules for technical terms"""
        pronunciation_rules = [
            {"replace": "SDK", "with": "S-D-K", "ignore_case": False},
            {"replace": "API", "with": "A-P-I", "ignore_case": False},
            {"replace": "AI", "with": "A-Eye", "ignore_case": False},
            {"replace": "SignalWire", "with": "cygnalwyre", "ignore_case": False},
            {"replace": "SWAIG", "with": "swaygg", "ignore_case": True},
            {"replace": "SWML", "with": "Swimmel", "ignore_case": False},
            {"replace": "AgentBase", "with": "Agent Base", "ignore_case": True},
            {"replace": "FastAPI", "with": "Fast A-P-I", "ignore_case": True},
            {"replace": "JSON", "with": "jay-son", "ignore_case": True},
            {"replace": "HTTP", "with": "H-T-T-P", "ignore_case": True},
            {"replace": "CLI", "with": "C-L-I", "ignore_case": True}
        ]
        
        for rule in pronunciation_rules:
            self.add_pronunciation(rule["replace"], rule["with"], ignore_case=rule["ignore_case"])
        
        # Add hints for better recognition
        self.add_hints(["Sigmond:2.0", "SignalWire:2.0", "SDK:2.0", "AgentBase:2.0"])
        self.add_pattern_hint("swimmel:2.0", "swimmel", "SWML", ignore_case=True)
    
    def _setup_remote_search(self):
        """Setup remote search connection to standalone search server"""
        
        # Remote search configuration
        self.add_skill("native_vector_search", {
            "tool_name": "search_docs",
            "description": "Search the SignalWire Agents SDK documentation for information about features, guides, concepts, and examples",
            "remote_url": "http://localhost:8001",  # Connect to standalone search server
            "index_name": "docs",  # Use the docs index on the remote server
            "count": 5,
            "similarity_threshold": 0.1,
            "response_prefix": "When results contain programming code, Do not read any source code aloud, just reference what file to look in for the answer",
            "response_postfix": "Express this information in a format that is easy to understand when read aloud.",
            "no_results_message": "I couldn't find information about '{query}' in the SDK documentation. Try rephrasing your question or asking about a different SDK feature.",
            "swaig_fields": {
                "fillers": {
                    "en-US": [
                        "Let me search the documentation for you",
                        "Checking the remote search server",
                        "Looking that up in the documentation"
                    ]
                }
            }
        })

def main():
    """Run the Sigmond Remote Search Bot"""
    logger.info("Starting Sigmond Remote Search Bot...")
    logger.info("This bot connects to a remote search server for SDK documentation")
    logger.info("Make sure the search server is running at http://localhost:8001")
    
    # Create and run the agent
    agent = SigmondRemoteSearch()
    
    logger.info("Sigmond Remote Search Bot is ready")
    logger.info("This agent uses remote search - no local dependencies needed!")
    logger.info("Try asking questions like:")
    logger.info("- 'How do I create an agent?'")
    logger.info("- 'What skills are available?'")
    logger.info("- 'How do I add state management?'")
    logger.info("- 'Show me examples of SWAIG functions'")
    logger.info("- 'How do I deploy my agent?'")
    logger.info("Starting server...")
    
    agent.run()

if __name__ == "__main__":
    main() 
