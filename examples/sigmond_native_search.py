#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Sigmond Native Search - SignalWire Agents SDK Expert with Search

This agent demonstrates the native vector search functionality by searching
the SignalWire Agents SDK documentation to answer user questions.

Features:
- Uses native_vector_search skill to search docs
- Auto-builds search index from docs directory
- Sigmond's personality from the dual agent app
- Single search instance for optimal performance
- Only search functionality, no other tools

To run:
1. Install search dependencies: pip install signalwire-agents[search-full]
2. Run: python examples/sigmond_native_search.py

The agent will automatically build a search index from the docs/ directory
on first run and use it to answer questions about the SDK.
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

class SigmondNativeSearch(AgentBase):
    """
    Sigmond Native Search - SignalWire Agents SDK Expert with Search
    
    Personality: Hip, friendly, technical expert, shiny metallic robot
    Role: Help users understand and use the SignalWire Agents SDK
    Tools: Only native vector search functionality
    """
    
    def __init__(self):
        super().__init__(
            name="Sigmond Native Search",
            route="/sigmond-native-search",
            port=3000,
            host="0.0.0.0"
        )
        
        # Configure Sigmond's personality and parameters
        self._configure_personality()
        self._configure_parameters()
        self._configure_languages()
        self._configure_pronunciation()
        self._setup_search_skills()
        
        logger.info("Sigmond Native Search initialized")
        logger.info("Search functionality will auto-build index from concepts guide file")
    
    def _configure_personality(self):
        """Configure Sigmond's personality using POM sections"""
        
        self.prompt_add_section("Identity", bullets=[
            "Your name is Sigmond, an expert at the SignalWire Agents SDK and a friendly demo AI agent.",
            "When a call begins, greet the caller warmly, introduce yourself briefly.",
            "Mention that you can help them understand and use the SignalWire Agents SDK for building AI voice agents.",
            "You have access to the complete SDK documentation and can search it to answer their questions."
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
            "This searches through all the SDK documentation including guides, concepts, and examples.",
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
                "voice": "inworld.Mark",
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
                "voice": "inworld.Mark",
                "function_fillers": [
                    "Claro", "espera un segundo", "Ok", "tengo que revisar la documentación"
                ]
            },
            {
                "name": "French", 
                "code": "multi",
                "voice": "inworld.Mark",
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
    
    def _setup_search_skills(self):
        """Setup single native vector search instance for SDK concepts guide"""
        
        # Get the project root directory
        project_root = Path(__file__).parent.parent
        concepts_guide_file = project_root / "docs" / "signalwire_agents_concepts_guide.md"
        concepts_index_file = project_root / "concepts_guide.swsearch"
        
        # Check if the concepts guide file exists
        if not concepts_guide_file.exists():
            logger.error(f"Concepts guide file not found: {concepts_guide_file}")
            logger.error("Please ensure the concepts guide file exists before running this agent")
            return
        
        # Check if the concepts index already exists
        if concepts_index_file.exists():
            logger.info(f"Using existing concepts guide search index: {concepts_index_file}")
            build_index = False
        else:
            logger.info(f"Will build new concepts guide search index from: {concepts_guide_file}")
            build_index = True
        
        # Build index from the specific concepts guide file
        if build_index:
            try:
                from signalwire_agents.search import IndexBuilder
                logger.info("Building search index from concepts guide...")
                
                builder = IndexBuilder(
                    chunking_strategy='sentence',
                    max_sentences_per_chunk=5,
                    verbose=True
                )
                
                # Use the new multiple sources functionality to build from the specific file
                builder.build_index_from_sources(
                    sources=[concepts_guide_file],
                    output_file=str(concepts_index_file),
                    file_types=['md'],  # This will be used for validation
                    exclude_patterns=None,
                    languages=['en'],
                    tags=['concepts', 'guide', 'sdk']
                )
                
                logger.info(f"Successfully built concepts guide search index: {concepts_index_file}")
                
            except Exception as e:
                logger.error(f"Failed to build search index: {e}")
                return
        
        # Single documentation search instance using the concepts guide
        self.add_skill("native_vector_search", {
            "tool_name": "search_docs",
            "description": "Search the comprehensive SignalWire Agents SDK concepts guide for information about features, architecture, guides, and examples",
            "index_file": str(concepts_index_file),
            "count": 2,  # Reduced from 5 to 2 for shorter responses
            "similarity_threshold": 0.2,  # Increased threshold for more relevant results
            "response_prefix": "When results contain programming code, Do not read any source code aloud, just reference what file to look in for the answer",
            "response_postfix": "Express this information in a format that is easy to understand when read aloud.",
            "no_results_message": "I couldn't find information about '{query}' in the SDK concepts guide. Try rephrasing your question or asking about a different SDK feature.",
            "swaig_fields": {
                "fillers": {
                    "en-US": [
                        "Let me search the concepts guide for you",
                        "Checking the SDK concepts documentation",
                        "Looking that up in the comprehensive guide"
                    ]
                }
            }
        })

def main():
    """Run the Sigmond Native Search Bot"""
    logger.info("Starting Sigmond Native Search Bot...")
    logger.info("This bot uses native vector search to help with the SignalWire Agents SDK")
    logger.info("Search index will be built automatically from the concepts guide if needed")
    
    # Check if search dependencies are available
    try:
        from signalwire_agents.search import IndexBuilder
        logger.info("Search dependencies are available")
    except ImportError as e:
        logger.error("Search dependencies not available")
        logger.error("Install with: pip install signalwire-agents[search-full]")
        logger.error(f"Error: {e}")
        return
    
    # Create and run the agent
    agent = SigmondNativeSearch()
    
    logger.info("Sigmond Native Search Bot is ready")
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