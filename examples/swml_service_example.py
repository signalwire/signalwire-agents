#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
SWML Service Example

This example demonstrates creating a simple SWML service using the new architecture.
It shows three different approaches to building SWML documents:

1. Creating a basic SWML document with direct verb manipulation
2. Using the fluent SWMLBuilder API for more readable code
3. Using the AI verb for creating conversational experiences

Each approach creates a standalone SWML service that can be run independently.
"""

import os
import sys

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signalwire_agents.core.swml_service import SWMLService
from signalwire_agents.core.swml_builder import SWMLBuilder


def example_using_service():
    """
    Example using SWMLService's direct document manipulation
    
    This approach adds verbs directly to the SWML document by calling
    methods on the SWMLService. It provides the most control but requires
    more detailed knowledge of the SWML document structure.
    
    Returns:
        SWMLService: Configured service instance
    """
    print("=== Example using SWMLService directly ===")
    
    #------------------------------------------------------------------------
    # SERVICE SETUP
    # Create a basic SWML service with HTTP server configuration
    #------------------------------------------------------------------------
    
    # Create a simple SWML service with HTTP server settings
    # The route defines the HTTP path where this service will be available
    service = SWMLService(
        name="simple-swml-service",  # Identifier for the service
        route="/simple",             # HTTP endpoint path
        host="0.0.0.0",              # Listen on all network interfaces
        port=3001                    # Port for HTTP server
    )
    
    #------------------------------------------------------------------------
    # DOCUMENT CREATION
    # Build the SWML document by adding verbs directly
    #------------------------------------------------------------------------
    
    # Reset the document to start fresh (clears any existing verbs)
    service.reset_document()
    
    # Add verbs to the document in sequence
    # add_verb takes a verb name and parameters dictionary
    service.add_verb("answer", {})  # Answer the call when it arrives
    
    # This plays a text-to-speech message to the caller
    service.add_verb("play", {"url": "say:Hello, world!"})
    
    # Add a hangup verb to end the call
    service.add_verb("hangup", {})  # End the call
    
    # Print the rendered document as JSON for inspection
    print(service.render_document())
    print()
    
    return service


def example_using_builder():
    """
    Example using SWMLBuilder fluent API
    
    This approach uses the builder pattern with method chaining
    to create a more readable SWML document construction process.
    
    Returns:
        SWMLService: Configured service instance
    """
    print("=== Example using SWMLBuilder fluent API ===")
    
    #------------------------------------------------------------------------
    # SERVICE SETUP
    # Create a basic SWML service on a different port
    #------------------------------------------------------------------------
    
    # Create a simple SWML service
    # Using a different port than the first example
    service = SWMLService(
        name="builder-swml-service",  # Unique service name
        route="/builder",             # HTTP endpoint path
        host="0.0.0.0",               # Listen on all networks
        port=3002                     # Different port than first example
    )
    
    #------------------------------------------------------------------------
    # DOCUMENT CREATION USING BUILDER
    # Use the fluent API for more readable code
    #------------------------------------------------------------------------
    
    # Create a builder attached to our service
    # The builder provides an alternative, more fluent syntax
    # for building SWML documents
    builder = SWMLBuilder(service)
    
    # Build the document using the fluent API with method chaining
    # This style is more readable than direct document manipulation
    # Start with a fresh document
    builder.reset()
    # Answer the incoming call
    builder.answer()
    # First TTS message
    builder.say("Hello from the SWML Builder API!")
    # Second message with options
    builder.say(
        "Isn't this easier than assembling JSON?",
        voice="inworld.Mark",          # Use a specific voice
        language="en-US"                # Specify language for TTS
    )
    # End the call
    builder.hangup()                    
    
    # Print the rendered document as JSON for inspection
    print(builder.render())
    print()
    
    return service


def example_using_ai():
    """
    Example using AI verb for conversational interfaces
    
    This approach demonstrates using the AI verb to create a
    conversational agent that can interact with callers naturally.
    
    Returns:
        SWMLService: Configured service instance
    """
    print("=== Example using AI verb ===")
    
    #------------------------------------------------------------------------
    # SERVICE SETUP
    # Create a third SWML service for AI capabilities
    #------------------------------------------------------------------------
    
    # Create a simple SWML service for AI
    # Each service runs on its own port
    service = SWMLService(
        name="ai-swml-service",     # Unique service name
        route="/ai",                # HTTP endpoint path
        host="0.0.0.0",             # Listen on all networks
        port=3003                   # Third unique port
    )
    
    #------------------------------------------------------------------------
    # AI DOCUMENT CREATION
    # Build an AI-powered conversational interface
    #------------------------------------------------------------------------
    
    # Create a builder for the service
    builder = SWMLBuilder(service)
    
    # Build the document using the fluent API
    # Start with a fresh document
    builder.reset()
    # Answer the incoming call
    builder.answer()
    # Add the AI conversation verb
    builder.ai(
        # The prompt configures the AI's personality and capabilities
        prompt_text="You are a helpful assistant. Answer user questions concisely.",
        # The post-prompt generates a summary after the conversation ends
        post_prompt="Summarize the conversation in 1-2 sentences."
    )
    # End the call after AI interaction completes
    builder.hangup()                     
    
    # Print the rendered document as JSON for inspection
    print(builder.render())
    print()
    
    return service


def main():
    #------------------------------------------------------------------------
    # RUN EXAMPLES AND INTERACTIVE SELECTION
    # Create all three services and let the user choose which to run
    #------------------------------------------------------------------------
    
    # Run the examples to create the configured services
    # Each example prints its SWML document for inspection
    service1 = example_using_service()    # Basic service with direct verb manipulation
    service2 = example_using_builder()    # Service using fluent builder API
    service3 = example_using_ai()         # Service with AI conversational capabilities
    
    # Ask which example to serve
    # Only one can be run at a time since they would conflict on host/port
    print("Choose a service to start:")
    print("1. Simple SWML service (direct verb manipulation)")
    print("2. Builder SWML service (fluent API)")
    print("3. AI SWML service (conversational interface)")
    choice = input("Enter choice (1-3) or any other key to exit: ")
    
    # Start the selected service based on user input
    if choice == "1":
        print("Starting simple SWML service on http://localhost:3001/simple")
        print("To test: curl http://localhost:3001/simple")
        service1.run()
    elif choice == "2":
        print("Starting builder SWML service on http://localhost:3002/builder")
        print("To test: curl http://localhost:3002/builder")
        service2.run()
    elif choice == "3":
        print("Starting AI SWML service on http://localhost:3003/ai")
        print("To test: curl http://localhost:3003/ai")
        service3.run()
    else:
        print("Exiting.")


if __name__ == "__main__":
    main() 