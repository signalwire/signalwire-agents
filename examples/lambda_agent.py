#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
AWS Lambda deployment example for SignalWire AI Agents

This example shows how to deploy a SignalWire AI Agent to AWS Lambda using Mangum.
Mangum is included in requirements.txt, so no additional installation is needed.

Requirements:
    - signalwire-agents (includes mangum>=0.19.0)

Usage:
    1. Deploy this file to AWS Lambda
    2. Configure API Gateway to route all requests to this function
    3. Set environment variables:
       - SWML_BASIC_AUTH_USER (optional, defaults to 'dev')
       - SWML_BASIC_AUTH_PASSWORD (optional, defaults to 'w00t')

Features:
    - Full SignalWire AI agent functionality in serverless
    - Same routing, authentication, SWAIG functions as regular deployment
    - Health checks work (/health, /ready)
    - Structured logging compatible with CloudWatch
    - Environment-based configuration
"""

import os
import sys
from datetime import datetime

# Add the signalwire_agents module to the path if needed
# (not needed if installed via pip)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from signalwire_agents import AgentBase
from signalwire_agents.core.function_result import SwaigFunctionResult

# Import Mangum for Lambda/API Gateway integration
try:
    from mangum import Mangum
except ImportError:
    print("ERROR: Mangum not installed. Run: pip install mangum")
    sys.exit(1)


class HealthCheckAgent(AgentBase):
    """
    Example agent for Lambda deployment

    This agent demonstrates all the standard AgentBase features:
    - SWAIG functions with SwaigFunctionResult
    - Prompt configuration via prompt_add_section
    - Voice configuration via add_language
    - Basic auth and health endpoints
    """

    def __init__(self, name="lambda-agent", route="/", **kwargs):
        super().__init__(name=name, route=route, **kwargs)

        # Configure voice
        self.add_language(name="English", code="en-US", voice="inworld.Mark")

        # Configure prompt using prompt_add_section (the public API)
        self.prompt_add_section(
            "Role",
            body="You are a helpful AI assistant running in AWS Lambda."
        )
        self.prompt_add_section(
            "Instructions",
            bullets=[
                "Greet users warmly and offer help",
                "Use the greet_user function when asked to greet someone",
                "Use the get_time function when asked about the current time",
            ]
        )

    @AgentBase.tool("Greet a user by name")
    def greet_user(self, name: str = "friend"):
        """Greet a user by name"""
        return SwaigFunctionResult(f"Hello {name}! I'm running in AWS Lambda!")

    @AgentBase.tool("Get the current time")
    def get_time(self):
        """Get the current time"""
        return SwaigFunctionResult(f"Current time: {datetime.now().isoformat()}")


# Create the agent instance
# This works exactly the same as local development!
agent = HealthCheckAgent(
    name="lambda-agent",
    route="/",  # Lambda usually serves from root
)

# Get the FastAPI app from the agent
app = agent.get_app()

# Create the Lambda handler using Mangum
# This handles all the API Gateway <-> FastAPI translation
handler = Mangum(app)

def lambda_handler(event, context):
    """
    AWS Lambda entry point

    This function receives API Gateway events and returns responses.
    Mangum handles all the translation between Lambda/API Gateway format
    and FastAPI's expected format.

    Args:
        event: API Gateway event
        context: Lambda context

    Returns:
        dict: API Gateway response format
    """
    return handler(event, context)


# For local testing (optional)
def main():
    print("\nStarting agent server...")
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    agent.run()

if __name__ == "__main__":
    main()
