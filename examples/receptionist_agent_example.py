#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

# -*- coding: utf-8 -*-
"""
Example of using the ReceptionistAgent prefab to create a call routing system
"""

import os
import sys
import json
import argparse

# Import the ReceptionistAgent prefab
from signalwire_agents.prefabs import ReceptionistAgent
from signalwire_agents.core.function_result import SwaigFunctionResult

class CustomReceptionistAgent(ReceptionistAgent):
    """
    Extending the ReceptionistAgent prefab with custom functionality
    
    This example demonstrates:
    1. Using the prefab with minimal configuration
    2. Adding custom voice and greeting
    3. Overriding the on_summary method to log call data
    """
    
    def __init__(self, **kwargs):
        # Define departments with their descriptions and transfer numbers
        departments = [
            {
                "name": "sales", 
                "description": "For product inquiries, pricing, and purchasing", 
                "number": "+15551235555"
            },
            {
                "name": "support", 
                "description": "For technical assistance, troubleshooting, and bug reports", 
                "number": "+15551236666"
            },
            {
                "name": "billing", 
                "description": "For payment questions, invoices, and subscription changes", 
                "number": "+15551237777"
            },
            {
                "name": "general", 
                "description": "For all other inquiries or if you're not sure which department you need", 
                "number": "+15551238888"
            }
        ]
        
        # Custom greeting message
        greeting = "Hello, thank you for calling ACME Corporation. How may I direct your call today?"
        
        # Initialize the base ReceptionistAgent with our configuration
        super().__init__(
            departments=departments,
            name="acme-receptionist",
            route="/reception",
            greeting=greeting,
            voice="rime.spore",  # Using proper voice format
            **kwargs
        )
        
        # Add additional custom prompt section if needed
        self.prompt_add_section(
            "Company Information",
            body="ACME Corporation is a leading provider of innovative solutions. Our business hours are Monday through Friday, 9 AM to 5 PM Eastern Time."
        )
        
        # Add a prompt section that clarifies available departments for transfers
        departments_text = "Available departments for transfer:\n"
        for dept in departments:
            departments_text += f"- {dept['name'].title()}: {dept['description']}\n"
        
        self.prompt_add_section(
            "Transfer Options",
            body=departments_text
        )
    
    def on_summary(self, summary, raw_data=None):
        """
        Process the conversation summary with custom handling
        
        Args:
            summary: Summary data from the conversation
            raw_data: The complete raw POST data from the request
        """
        # Log the call summary for reporting
        if summary:
            # In a real application, you might:
            # - Store in a database
            # - Send to a CRM
            # - Generate analytics
            print(f"Call Summary: {json.dumps(summary, indent=2)}")
            
            # You could trigger follow-up actions based on satisfaction
            satisfaction = summary.get("satisfaction", "")
            if satisfaction == "low":
                # In a real system, this might trigger an alert or follow-up task
                print("ALERT: Caller had low satisfaction. Schedule follow-up.")


def main():
    """Run the ReceptionistAgent example"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="ReceptionistAgent Example")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=3000, help="Port to bind the server to")
    args, _ = parser.parse_known_args()
    
    # Create our custom receptionist agent
    agent = CustomReceptionistAgent(host=args.host, port=args.port)
    return agent


if __name__ == "__main__":
    agent = main()

    # Get basic auth credentials for display
    username, password = agent.get_basic_auth_credentials()

    # Print information about the agent
    print("Starting the Custom Receptionist Agent")
    print("----------------------------------------")
    print(f"URL: http://0.0.0.0:3000{agent.route}")
    print(f"Basic Auth: {username}:{password}")
    print("----------------------------------------")
    print("Press Ctrl+C to stop the agent")

    print("\nStarting agent server...")
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    agent.run() 