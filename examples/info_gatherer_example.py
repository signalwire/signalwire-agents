#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

# -*- coding: utf-8 -*-
"""
Example of using the redesigned InfoGathererAgent to collect answers to questions
"""

import os
import sys
from signalwire_agents.prefabs import InfoGathererAgent

def main():
    """Run the InfoGathererAgent example"""
    
    print("Static InfoGatherer Agent - questions are fixed at startup")
    print("For dynamic configuration, see dynamic_info_gatherer_example.py")
    print()
    
    # Create an agent with a list of questions (STATIC MODE)
    agent = InfoGathererAgent(
        questions=[
            {"key_name": "name", "question_text": "What is your full name?"},
            {"key_name": "phone", "question_text": "What is your phone number?", "confirm": True},
            {"key_name": "age", "question_text": "What is your age?"},
            {"key_name": "reason", "question_text": "What are you contacting us about today?"}
        ],
        name="contact-form",
        route="/contact"
    )

    # Set voice
    agent.add_language(
        name="English",    # Display name for the language
        code="en-US",      # ISO language code
        voice="inworld.Mark"  # Voice ID with provider prefix
    )

    # Customize the agent with additional prompt sections if desired
    agent.prompt_add_section(
        "Introduction", 
        body="I'm here to help you fill out our contact form. "
             "This information helps us better serve you."
    )

    # Set the post prompt to summarize the questions and answers in a concise manner
    
    agent.set_post_prompt("Summarize the questions and answers in a concise manner.")

    # you have to set a post prompt to be able to use either of the next two options

    # Set the post prompt URL to the remote server, even if you define the on_summary method it will be ignored if you set the post prompt URL
    agent.set_post_prompt_url("https://user:password@example.com/ai/post.cgi")
    
    # Get basic auth credentials for display
    username, password = agent.get_basic_auth_credentials()
    
    # Print information about the agent
    print("Starting the Information Gathering Agent")
    print("----------------------------------------")
    print(f"URL: http://localhost:3000{agent.route}")
    print(f"Basic Auth: {username}:{password}")
    print("----------------------------------------")
    print("Press Ctrl+C to stop the agent")
    
    # Start the agent server
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    agent.run()

if __name__ == "__main__":
    main() 
