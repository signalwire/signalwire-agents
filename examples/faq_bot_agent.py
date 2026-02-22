#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
FAQ Bot Agent Example

This example demonstrates how to create a specialized agent 
for answering frequently asked questions from a knowledge base.

Key concepts demonstrated:
1. Creating a domain-specific agent for FAQs
2. Structuring knowledge in the prompt
3. Using declarative prompt templates 
4. Customizing agent behavior at runtime
"""

import os
import sys
from typing import Dict, List, Optional

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signalwire_agents import AgentBase
from signalwire_agents.core.function_result import SwaigFunctionResult


class FAQBotAgent(AgentBase):
    """
    A specialized agent for answering frequently asked questions (FAQs)
    from a predefined knowledge base.
    
    This agent demonstrates:
    1. Using a class-level template for prompt sections
    2. Dynamically formatting prompt sections with instance variables
    3. Building a structured knowledge base at initialization time
    4. Processing conversation summaries
    
    The FAQ bot is designed to only provide information contained in its
    knowledge base and to clearly indicate when it doesn't have an answer.
    """
    
    # Define the prompt sections declaratively with placeholders
    # This demonstrates creating reusable prompt templates that can be
    # customized for different instances of the same agent type
    PROMPT_SECTIONS = {
        "Personality": "You are a helpful FAQ assistant for {company_name}.",
        "Goal": "Answer customer questions using only the provided FAQ knowledge base.",
        "Instructions": [
            "Only answer questions if the information is in the FAQ knowledge base.",
            "If you don't know the answer, politely say so and offer to help with something else.",
            "Be concise and direct in your responses.",
            "If the answer is in the knowledge base, cite the relevant FAQ item."
        ],
        "Knowledge Base": ""  # Will be populated during initialization
    }
    
    def __init__(
        self, 
        name: str = "faq_bot",
        route: str = "/faq",
        host: str = "0.0.0.0", 
        port: int = 3000,
        company_name: str = "Our Company",
        faqs: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the FAQ Bot Agent
        
        Args:
            name: Agent identifier used for logging and management
            route: HTTP endpoint path for this agent
            host: Host to bind the HTTP server to
            port: Port to bind the HTTP server to
            company_name: Company name to include in the personality section
            faqs: Dictionary of questions and answers to populate the knowledge base
        """
        #------------------------------------------------------------------------
        # BASE INITIALIZATION
        # Set up the core agent capabilities
        #------------------------------------------------------------------------
        
        # Initialize the base agent
        # This creates the HTTP server and sets up the basic agent functionality
        super().__init__(
            name=name,
            route=route,
            host=host,
            port=port
        )
        
        #------------------------------------------------------------------------
        # KNOWLEDGE BASE SETUP
        # Build the FAQ knowledge structure
        #------------------------------------------------------------------------
        
        # Store the company name for use in prompt sections
        self.company_name = company_name
        
        # Default FAQs if none provided
        # This allows the agent to work out-of-the-box with sample data
        if faqs is None:
            faqs = {
                "What are your hours?": "We are open Monday through Friday, 9am to 5pm.",
                "How do I reset my password?": "You can reset your password by clicking on the 'Forgot Password' link on the login page.",
                "Do you offer refunds?": "Yes, we offer refunds within 30 days of purchase if you're not satisfied with your product."
            }
        
        # Store FAQs for potential runtime access
        self.faqs = faqs
        
        #------------------------------------------------------------------------
        # PROMPT CONFIGURATION
        # Set up the AI's instructions and knowledge
        #------------------------------------------------------------------------
        
        # Generate Knowledge Base content with a clear Q&A format
        # This structures the knowledge in a way that's easy for the AI to reference
        kb_content = "Frequently Asked Questions:\n\n"
        for question, answer in faqs.items():
            kb_content += f"Q: {question}\n"
            kb_content += f"A: {answer}\n\n"
        
        # Add the Knowledge Base prompt section using the public API
        # This adds the formatted FAQs to the prompt
        self.prompt_add_section("Knowledge Base", body=kb_content)

        # Add the Personality section with company name
        # This demonstrates dynamically formatting template strings with instance variables
        personality_text = self.PROMPT_SECTIONS["Personality"].format(company_name=company_name)
        self.prompt_add_section("Personality", body=personality_text)

        # Add the Goal section
        self.prompt_add_section("Goal", body=self.PROMPT_SECTIONS["Goal"])

        # Add the Instructions section with bullet points
        self.prompt_add_section("Instructions", bullets=self.PROMPT_SECTIONS["Instructions"])
        
        # Set up a post-prompt for conversation summary
        # This defines the structure of data we want to capture after the conversation
        self.set_post_prompt("""
        Provide a JSON summary of the interaction:
        {
            "question_type": "CATEGORY_OF_QUESTION",
            "answered_from_kb": true/false,
            "follow_up_needed": true/false
        }
        """)
        
    def on_summary(self, summary: Dict, raw_data=None):
        """
        Handle the conversation summary generated by the post-prompt
        
        This method is called after a conversation has completed and
        the AI has generated the summary JSON structure.
        
        Args:
            summary: Dictionary containing the structured conversation summary
            raw_data: Raw request data (optional)
        """
        print(f"FAQ Bot conversation summary: {summary}")
        
        # Extract useful information from the summary
        question_type = summary.get("question_type", "unknown")
        answered = summary.get("answered_from_kb", False)
        needs_followup = summary.get("follow_up_needed", False)
        
        # Log the key metrics
        print(f"Question Type: {question_type}")
        print(f"Successfully Answered: {answered}")
        print(f"Needs Follow-up: {needs_followup}")
        
        # In a real implementation, you might:
        # - Log this information to a database for analytics
        # - Trigger follow-up actions for unanswered questions
        # - Update the FAQ database with commonly asked but missing questions
        # - Route to a human agent if follow-up is needed


if __name__ == "__main__":
    #------------------------------------------------------------------------
    # AGENT CREATION AND STARTUP
    # Create and run a customized FAQ bot
    #------------------------------------------------------------------------
    
    # Create a custom FAQ bot with specific SignalWire FAQs
    # This demonstrates how to create a domain-specific instance
    # with custom knowledge
    custom_faqs = {
        "What is SignalWire?": "SignalWire is a communications platform that provides APIs for voice, video, and messaging.",
        "How do I create an AI Agent?": "You can create an AI Agent using the SignalWire AI Agent SDK, which provides a simple way to build and deploy conversational AI agents.",
        "What is SWML?": "SWML (SignalWire Markup Language) is a markup language for defining communications workflows, including AI interactions."
    }
    
    # Initialize the agent with custom configuration
    agent = FAQBotAgent(
        name="signalwire_faq",       # Custom name for the agent
        company_name="SignalWire",   # Company name to include in personality
        faqs=custom_faqs             # Custom knowledge base
    )
    
    print("Starting the FAQ Bot. Press Ctrl+C to stop.")
    print(f"Agent is accessible at: http://localhost:3000/faq")
    print("Example usage: curl -X POST -H \"Content-Type: application/json\" -d '{\"message\": \"What is SignalWire?\"}' http://localhost:3000/faq")
    
    # Start the agent's web server
    try:
        print("Note: Works in any deployment mode (server/CGI/Lambda)")
        agent.run()
    except KeyboardInterrupt:
        print("\nStopping the FAQ Bot.")
        agent.stop() 