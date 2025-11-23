#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Concierge Agent Example - Demonstrating the ConciergeAgent prefab

This example shows how to use the ConciergeAgent prefab for providing
virtual concierge services with information about amenities and services.

Features demonstrated:
1. Initializing a ConciergeAgent with venue information
2. Configuring amenities and services
3. Running the agent server
"""

import os
import logging
import sys
import argparse

# Import the ConciergeAgent prefab
from signalwire_agents.prefabs import ConciergeAgent

# Basic logging setup
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)

# Create logger
logger = logging.getLogger("concierge_example")


def main():
    parser = argparse.ArgumentParser(description="Run the ConciergeAgent Example")
    parser.add_argument("--port", type=int, default=3000, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    args, _ = parser.parse_known_args()
    
    # Find schema.json in the current directory or parent directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    schema_locations = [
        os.path.join(current_dir, "schema.json"),
        os.path.join(parent_dir, "schema.json")
    ]
    
    schema_path = None
    for loc in schema_locations:
        if os.path.exists(loc):
            schema_path = loc
            logger.info(f"Found schema.json at: {schema_path}")
            break
            
    if not schema_path:
        logger.warning(f"Could not find schema.json in: {schema_locations}")
    
    # Define a luxury resort concierge
    venue_name = "Oceanview Resort"
    
    # Define services
    services = [
        "room service",
        "spa bookings",
        "restaurant reservations",
        "activity bookings",
        "airport shuttle",
        "valet parking",
        "concierge assistance"
    ]
    
    # Define amenities with details
    amenities = {
        "infinity pool": {
            "hours": "7:00 AM - 10:00 PM",
            "location": "Main Level, Ocean View",
            "description": "Heated infinity pool overlooking the ocean with poolside service.",
            "features": "Cabanas, hot tub, kids' splash area"
        },
        "spa": {
            "hours": "9:00 AM - 8:00 PM",
            "location": "Lower Level, East Wing",
            "description": "Full-service luxury spa offering massages, facials, and body treatments.",
            "reservation": "Required"
        },
        "fitness center": {
            "hours": "24 hours",
            "location": "2nd Floor, North Wing",
            "description": "State-of-the-art fitness center with cardio equipment, weights, and yoga studio.",
            "features": "Personal trainers available by appointment"
        },
        "beach access": {
            "hours": "Dawn to Dusk",
            "location": "Southern Pathway",
            "description": "Private beach access with complimentary chairs, umbrellas, and towels.",
            "services": "Beach attendants, food and beverage service"
        }
    }
    
    # Define hours of operation
    hours = {
        "check-in": "3:00 PM",
        "check-out": "11:00 AM",
        "front desk": "24 hours",
        "concierge": "7:00 AM - 11:00 PM",
        "room service": "24 hours"
    }
    
    # Special instructions for the concierge
    special_instructions = [
        "Always greet guests by name when possible.",
        "Offer to make reservations for guests at local attractions.",
        "Provide weather updates when discussing outdoor activities.",
        "Inform guests about the daily resort activities and events."
    ]
    
    # Welcome message
    welcome_message = (
        "Welcome to Oceanview Resort, where luxury meets comfort. "
        "I'm your virtual concierge, ready to assist with any requests "
        "or answer questions about our amenities and services. "
        "How may I help you today?"
    )
    
    # Create the concierge agent
    agent = ConciergeAgent(
        venue_name=venue_name,
        services=services,
        amenities=amenities,
        hours_of_operation=hours,
        special_instructions=special_instructions,
        welcome_message=welcome_message,
        schema_path=schema_path,
        host=args.host,
        port=args.port
    )
    
    return agent


if __name__ == "__main__":
    agent = main()

    # Print credentials
    username, password, source = agent.get_basic_auth_credentials(include_source=True)

    logger.info(f"Starting Concierge Agent for Oceanview Resort")

    print("\nStarting agent server...")
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    agent.run() 