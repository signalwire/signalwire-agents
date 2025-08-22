#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

# -*- coding: utf-8 -*-
"""
Simple example of using the SignalWire AI Agent SDK

This example demonstrates creating an agent using explicit methods
to manipulate the POM (Prompt Object Model) structure directly.

This uses the refactored AgentBase class that internally uses SWMLService.
"""

from datetime import datetime
import os
import logging
import sys
import json
import argparse
import fastapi

# Import structlog for proper structured logging
import structlog

from signalwire_agents import AgentBase
from signalwire_agents.core.function_result import SwaigFunctionResult

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Set up the root logger with structlog
logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)

# Create structured logger
logger = structlog.get_logger("simple_agent")

class SimpleAgent(AgentBase):
    """
    A simple agent that demonstrates using explicit methods
    to manipulate the POM structure directly
    
    This example shows:
    
    1. How to create an agent with a structured prompt using POM
    2. How to define SWAIG functions that the AI can call
    3. How to return results from SWAIG functions
    4. How to configure various AI verb parameters
    5. How to configure SWAIG with native functions and remote includes
    """
    
    def __init__(self, suppress_logs=False):
        # Find schema.json in the current directory or parent directory
        # The schema.json file defines the valid structure for SWML documents
        # and is used for validation during document creation
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        # Try to find schema.json in several locations
        # This allows the example to work regardless of where it's run from
        schema_locations = [
            os.path.join(current_dir, "schema.json"),
            os.path.join(parent_dir, "schema.json"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "schema.json")
        ]
        
        schema_path = None
        for loc in schema_locations:
            if os.path.exists(loc):
                schema_path = loc
                logger.info("schema_found", path=schema_path)
                break
                
        if not schema_path:
            logger.warning("schema_not_found", locations=schema_locations)
            
        # Initialize the agent with a name and route
        # The name is used for identification and logging
        # The route defines the HTTP endpoint path for the agent (e.g., /simple)
        super().__init__(
            name="simple",
            route="/simple",
            host="0.0.0.0",  # Bind to all interfaces
            port=3000,       # Listen on port 3000
            use_pom=True,    # Enable Prompt Object Model for structured prompts
            schema_path=schema_path,  # Pass the explicit schema path for validation
            suppress_logs=suppress_logs  # Option to reduce log output
        )
        
        #------------------------------------------------------------------------
        # PROMPT CONFIGURATION
        # Set up the AI's personality, goals, and instructions using POM structure
        #------------------------------------------------------------------------
        
        # Initialize POM sections using convenience methods
        # These methods call prompt_add_section() internally with appropriate section titles
        self.setPersonality("You are a friendly and helpful assistant.")
        self.setGoal("Help users with basic tasks and answer questions.")
        self.setInstructions([
            "Be concise and direct in your responses.",
            "If you don't know something, say so clearly.",
            "Use the get_time function when asked about the current time.",
            "Use the get_weather function when asked about the weather."
        ])


        # Optional: Set LLM parameters for fine-tuned control
        # These parameters are passed to the server which validates them based on the model
        self.set_prompt_llm_params(
            temperature=0.3,        # Low temperature for more consistent responses
            top_p=0.9,             # Slightly reduced for focused responses
            barge_confidence=0.7,   # Higher confidence threshold
            presence_penalty=0.1,   # Slight penalty for repetition
            frequency_penalty=0.2   # Encourage varied vocabulary
        )

        
        
        # Add a post-prompt for summary generation
        # This is processed after the conversation ends to generate a summary
        # The summary is received in the on_summary method
        self.set_post_prompt("""
        Return a JSON summary of the conversation:
        {
            "topic": "MAIN_TOPIC",
            "satisfied": true/false,
            "follow_up_needed": true/false
        }
        """)
        
        #------------------------------------------------------------------------
        # PRONUNCIATION AND HINTS
        # Help the AI understand and pronounce certain terms correctly
        #------------------------------------------------------------------------
        
        # Configure hints to help the AI understand certain words
        # Hints are used to improve entity recognition in the conversation
        self.add_hints([
            "SignalWire", 
            "SWML", 
            "SWAIG"
        ])
        
        # Add a pattern hint for pronunciation
        # Pattern hints use regex to match text patterns and provide replacements
        # This helps the AI pronounce terms more naturally
        self.add_pattern_hint(
            hint="AI Agent",           # Term to help with
            pattern="AI\\s+Agent",     # Regex pattern to match
            replace="A.I. Agent",      # Replacement for pronunciation
            ignore_case=True           # Case-insensitive matching
        )
        
        # Add simple pronunciation rules
        # These map exact terms to the phonetic way they should be pronounced
        self.add_pronunciation("API", "A P I", ignore_case=False)  # Spell out A-P-I
        self.add_pronunciation("SIP", "sip", ignore_case=True)     # Pronounce as word
        
        #------------------------------------------------------------------------
        # MULTILINGUAL SUPPORT
        # Configure multiple languages with different voice options
        #------------------------------------------------------------------------
        
        # Example 1: Simple format with ElevenLabs voice
        self.add_language(
            name="English",    # Display name for the language
            code="en-US",      # ISO language code
            voice="rime.spore",  # Voice ID with provider prefix
            # Phrases to use when the AI needs time to think
            speech_fillers=["Let me think about that...", "One moment please..."],
            # Phrases to use when the AI is calling a function
            function_fillers=["I'm looking that up for you...", "Let me check that..."]
        )

        # Example 2: Explicit parameters with engine and model
        self.add_language(
            name="British English",
            code="en-GB",
            voice="spore",        # Voice ID without provider prefix
            engine="rime", # The TTS engine to use
            model="multilingual",  # Specific model for the voice
            speech_fillers=["Just a moment...", "Thinking..."]
        )

        # Example 3: Combined string format (provider.voice:model)
        self.add_language(
            name="Spanish",
            code="es",
            # Format: provider.voice:model
            voice="rime.pablo",
            speech_fillers=["Un momento por favor...", "Estoy pensando..."],
            function_fillers=["Estoy buscando esa información...", "Déjame verificar..."]
        )

        # Example 4: Alternative engine (Rime) with explicit parameters
        self.add_language(
            name="French",
            code="fr-FR",
            voice="rime.alois"
        )

        # Example 5: Alternative engine with combined format
        self.add_language(
            name="German",
            code="de-DE",
            voice="rime.hans:arcana"  # Format: provider.voice:model
        )
        
        #------------------------------------------------------------------------
        # AI BEHAVIOR PARAMETERS
        # Configure how the AI interacts with users
        #------------------------------------------------------------------------
        
        # Set AI behavior parameters that control conversation flow
        self.set_params({
            "ai_model": "gpt-4.1-nano",
            "wait_for_user": False,          # Start speaking immediately rather than waiting
            "end_of_speech_timeout": 1000,   # Milliseconds of silence before assuming speech ended
            "ai_volume": 5,                  # Voice volume level
            "languages_enabled": True,       # Enable multilingual support
            "local_tz": "America/Los_Angeles" # Default timezone for time-related functions
        })
        
        # Add global data available to the AI
        # This provides context that the AI can reference during conversations
        self.set_global_data({
            "company_name": "SignalWire",
            "product": "AI Agent SDK",
            "supported_features": [
                "Voice AI",
                "Telephone integration",
                "SWAIG functions"
            ]
        })
        
        #------------------------------------------------------------------------
        # FUNCTION CONFIGURATION
        # Set up built-in and remote functions the AI can call
        #------------------------------------------------------------------------
        
        # Configure native functions (built-in functions provided by SignalWire)
        # These don't require implementation in the agent class
        self.set_native_functions([
            "check_time",     # Get the current time in various formats
            "wait_seconds"    # Pause the conversation for a specified duration
        ])
        
        # Add remote function includes
        # These are functions hosted at external URLs that the AI can call
        # First remote endpoint with metadata
        self.add_function_include(
            url="https://api.example.com/remote-functions",  # API endpoint
            functions=[  # List of available functions at this endpoint
                "get_weather_extended", 
                "get_traffic", 
                "get_news"
            ],
            meta_data={  # Additional data for the API call
                "auth_type": "bearer",
                "region": "us-west"
            }
        )
        
        # Add another remote function include with a different URL
        # This one doesn't include metadata, using default settings
        self.add_function_include(
            url="https://ai-tools.example.org/functions",
            functions=["translate_text", "summarize_document"]
        )
        
        #------------------------------------------------------------------------
        # SIP ROUTING CONFIGURATION
        # Enable the agent to be reachable via SIP for voice calls
        #------------------------------------------------------------------------
        
        # Enable SIP routing for this agent with auto_map=True
        # This allows the agent to be contacted through SIP protocol
        # auto_map=True creates a default SIP mapping using the agent's name
        # (e.g., simple@your-sip-domain)
        self.enable_sip_routing(auto_map=True)
        
        # Register additional SIP usernames for this agent
        # These provide alternative ways to reach the same agent
        # (e.g., simple_agent@your-sip-domain, assistant@your-sip-domain)
        self.register_sip_username("simple_agent")
        self.register_sip_username("assistant")
        
        # Log that the agent has been fully initialized
        logger.info("agent_initialized", agent_name=self.name, route=self.route)
    
    def setPersonality(self, personality_text):
        """
        Set the AI personality description
        
        This is a convenience method that adds a Personality section to the prompt.
        The personality defines the AI's character, tone, and style of interaction.
        
        Args:
            personality_text: The personality description as a string
            
        Returns:
            self: For method chaining
        """
        self.prompt_add_section("Personality", body=personality_text)
        return self
    
    def setGoal(self, goal_text):
        """
        Set the primary goal for the AI agent
        
        This is a convenience method that adds a Goal section to the prompt.
        The goal defines the AI's primary purpose and objective.
        
        Args:
            goal_text: The goal description as a string
            
        Returns:
            self: For method chaining
        """
        self.prompt_add_section("Goal", body=goal_text)
        return self
    
    def setInstructions(self, instructions_list):
        """
        Set the list of instructions for the AI agent
        
        This is a convenience method that adds an Instructions section to the prompt.
        Instructions provide specific guidance for the AI's behavior.
        
        Args:
            instructions_list: List of instruction strings
            
        Returns:
            self: For method chaining
        """
        if instructions_list:
            self.prompt_add_section("Instructions", bullets=instructions_list)
        return self
    
    @AgentBase.tool(
        name="get_time",
        description="Get the current time",
        parameters={}  # No parameters needed for this function
    )
    def get_time(self, args, raw_data):
        """
        Get the current time
        
        This SWAIG function is called by the AI when a user asks about the current time.
        It returns the current time in HH:MM:SS format.
        
        Args:
            args: Dictionary containing parsed parameters (empty for this function)
            raw_data: Complete request data including call_id and other metadata
            
        Returns:
            SwaigFunctionResult containing the current time
        """
        now = datetime.now()
        formatted_time = now.strftime("%H:%M:%S")
        logger.debug("get_time_called", time=formatted_time)
        return SwaigFunctionResult(f"The current time is {formatted_time}")
    
    @AgentBase.tool(
        name="get_weather",
        description="Get the current weather for a location",
        parameters={
            "location": {
                "type": "string",
                "description": "The city or location to get weather for"
            }
        }
    )
    def get_weather(self, args, raw_data):
        """
        Get the current weather for a location
        
        This SWAIG function is called by the AI when a user asks about weather.
        Parameters are passed in the args dictionary, while the complete request
        data is in raw_data.
        
        Args:
            args: Dictionary containing parsed parameters (location)
            raw_data: Complete request data including call_id and other metadata
            
        Returns:
            SwaigFunctionResult containing the response text and optional actions
        """
        # Extract location from the args dictionary 
        location = args.get("location", "Unknown location")
        
        # Log the function call with structured data
        logger.debug("get_weather_called", location=location)
        
        # Create the result with a response and multiple actions using add_actions
        # In a real implementation, this would call a weather API
        # For this example, we return mock data with multiple actions
        result = SwaigFunctionResult(f"It's sunny and 72°F in {location}.")
        
        # Example 1: Add a single action using add_action
        result.add_action("set_global_data", {"weather_location": location})
        
        # Example 2: Add multiple actions at once using add_actions
        result.add_actions([
            {"playback_bg": {"file": "https://example.com/weather_sounds.mp3"}},
            {"log": {"message": f"Weather requested for {location}"}}
        ])
        
        return result
    
    def on_summary(self, summary, raw_data=None):
        """
        Handle the conversation summary
        
        This method is called after the conversation ends when a post_prompt
        was configured. It processes the summary generated by the AI based on
        the post_prompt instructions.
        
        Args:
            summary: The summary object or None if no summary was found
            raw_data: The complete raw POST data from the request
        """
        # Print summary as properly formatted JSON (not Python dict representation)
        if summary:
            if isinstance(summary, (dict, list)):
                print("SUMMARY: " + json.dumps(summary))
            else:
                print(f"SUMMARY: {summary}")
        
        # Also directly print parsed array if available
        # The post_prompt_data contains both raw and parsed versions of the summary
        if raw_data and 'post_prompt_data' in raw_data:
            post_prompt_data = raw_data.get('post_prompt_data')
            
            # Print parsed summary data if available (usually a JSON object)
            if isinstance(post_prompt_data, dict) and 'parsed' in post_prompt_data:
                parsed = post_prompt_data.get('parsed')
                if parsed and len(parsed) > 0:
                    print("PARSED_SUMMARY: " + json.dumps(parsed[0]))
            
            # Print raw summary if available (this is the unprocessed text)
            if isinstance(post_prompt_data, dict) and 'raw' in post_prompt_data:
                raw = post_prompt_data.get('raw')
                if isinstance(raw, str):
                    print(f"RAW_SUMMARY: {raw}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SimpleAgent")
    parser.add_argument("--suppress-logs", action="store_true", help="Suppress extra logs")
    args = parser.parse_args()
    
    # Create an agent instance with log suppression if requested
    agent = SimpleAgent(suppress_logs=args.suppress_logs)
    
    # Get and print the authentication credentials
    # include_source=True also returns where the credentials came from
    # (generated, environment variables, or explicitly provided)
    username, password, source = agent.get_basic_auth_credentials(include_source=True)
    
    # Log agent startup with structured data
    logger.info("starting_agent", 
               url=f"http://localhost:3000/simple", 
               username=username, 
               password_length=len(password),
               auth_source=source)
    
    # Print user-friendly startup message with access details
    print("Starting the agent. Press Ctrl+C to stop.")
    print(f"Agent 'simple' is available at:")
    print(f"URL: http://localhost:3000/simple")
    print(f"Basic Auth: {username}:{password}")
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    
    try:
        # Start the agent's server using the built-in serve method
        # This starts a uvicorn server with the FastAPI app
        agent.run()
    except KeyboardInterrupt:
        # Handle clean shutdown on Ctrl+C
        logger.info("server_shutdown")
        print("\nStopping the agent.") 
