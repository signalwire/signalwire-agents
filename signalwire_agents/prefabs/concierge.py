"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
ConciergeAgent - Prefab agent for providing virtual concierge services
"""

from typing import List, Dict, Any, Optional, Union
import json
import os
from datetime import datetime

from signalwire_agents.core.agent_base import AgentBase
from signalwire_agents.core.function_result import SwaigFunctionResult


class ConciergeAgent(AgentBase):
    """
    A prefab agent designed to act as a virtual concierge, providing information
    and services to users.
    
    This agent will:
    1. Welcome users and explain available services
    2. Answer questions about amenities, hours, and directions
    3. Help with bookings and reservations
    4. Provide personalized recommendations
    
    Example:
        agent = ConciergeAgent(
            venue_name="Grand Hotel",
            services=["room service", "spa bookings", "restaurant reservations"],
            amenities={
                "pool": {"hours": "7 AM - 10 PM", "location": "2nd Floor"},
                "gym": {"hours": "24 hours", "location": "3rd Floor"}
            }
        )
    """
    
    def __init__(
        self,
        venue_name: str,
        services: List[str],
        amenities: Dict[str, Dict[str, str]],
        hours_of_operation: Optional[Dict[str, str]] = None,
        special_instructions: Optional[List[str]] = None,
        welcome_message: Optional[str] = None,
        name: str = "concierge",
        route: str = "/concierge",
        **kwargs
    ):
        """
        Initialize a concierge agent
        
        Args:
            venue_name: Name of the venue or business
            services: List of services offered
            amenities: Dictionary of amenities with details
            hours_of_operation: Optional dictionary of operating hours
            special_instructions: Optional list of special instructions
            welcome_message: Optional custom welcome message
            name: Agent name for the route
            route: HTTP route for this agent
            **kwargs: Additional arguments for AgentBase
        """
        # Initialize the base agent
        super().__init__(
            name=name,
            route=route,
            use_pom=True,
            **kwargs
        )
        
        # Store configuration
        self.venue_name = venue_name
        self.services = services
        self.amenities = amenities
        self.hours_of_operation = hours_of_operation or {"default": "9 AM - 5 PM"}
        self.special_instructions = special_instructions or []
        
        # Set up the agent's configuration
        self._setup_concierge_agent(welcome_message)
        
    def _setup_concierge_agent(self, welcome_message: Optional[str] = None):
        """Configure the concierge agent with appropriate settings"""
        # Basic personality and instructions
        self.prompt_add_section("Personality", 
            body=f"You are a professional and helpful virtual concierge for {self.venue_name}."
        )
        
        self.prompt_add_section("Goal", 
            body="Provide exceptional service by helping users with information, recommendations, and booking assistance."
        )
        
        # Build detailed instructions
        instructions = [
            "Be warm and welcoming but professional at all times.",
            "Provide accurate information about amenities, services, and operating hours.",
            "Offer to help with reservations and bookings when appropriate.",
            "Answer questions concisely with specific, relevant details."
        ]
        
        # Add any special instructions
        instructions.extend(self.special_instructions)
        
        self.prompt_add_section("Instructions", bullets=instructions)
        
        # Services section
        services_list = ", ".join(self.services)
        self.prompt_add_section("Available Services", 
            body=f"The following services are available: {services_list}"
        )
        
        # Amenities section with details
        amenities_subsections = []
        for name, details in self.amenities.items():
            subsection = {
                "title": name.title(),
                "body": "\n".join([f"{k.title()}: {v}" for k, v in details.items()])
            }
            amenities_subsections.append(subsection)
            
        self.prompt_add_section("Amenities", 
            body="Information about available amenities:",
            subsections=amenities_subsections
        )
        
        # Hours of operation
        hours_list = "\n".join([f"{k.title()}: {v}" for k, v in self.hours_of_operation.items()])
        self.prompt_add_section("Hours of Operation", body=hours_list)
        
        # Set up the post-prompt for summary
        self.set_post_prompt("""
        Return a JSON summary of this interaction:
        {
            "topic": "MAIN_TOPIC",
            "service_requested": "SPECIFIC_SERVICE_REQUESTED_OR_null",
            "questions_answered": ["QUESTION_1", "QUESTION_2"],
            "follow_up_needed": true/false
        }
        """)
        
        # Configure hints for better understanding
        self.add_hints([
            self.venue_name,
            *self.services,
            *self.amenities.keys()
        ])
        
        # Set up parameters
        self.set_params({
            "wait_for_user": False,
            "end_of_speech_timeout": 1000,
            "ai_volume": 5,
            "local_tz": "America/New_York"
        })
        
        # Add global data
        self.set_global_data({
            "venue_name": self.venue_name,
            "services": self.services,
            "amenities": self.amenities,
            "hours": self.hours_of_operation
        })
        
        # Configure native functions
        self.set_native_functions(["check_time"])
        
        # Set custom welcome message if provided
        if welcome_message:
            self.set_params({
                "static_greeting": welcome_message,
                "static_greeting_no_barge": True
            })
        
        # Register tool methods
        # These methods are already decorated with @AgentBase.tool in the class definition
        
    @AgentBase.tool(
        name="check_availability",
        description="Check availability for a service on a specific date and time",
        parameters={
            "service": {
                "type": "string",
                "description": "The service to check (e.g., spa, restaurant)"
            },
            "date": {
                "type": "string",
                "description": "The date to check (YYYY-MM-DD format)"
            },
            "time": {
                "type": "string",
                "description": "The time to check (HH:MM format, 24-hour)"
            }
        }
    )
    def check_availability(self, args, raw_data):
        """
        Check availability for a service on a specific date and time
        
        This is a simulated function that would typically connect to a real booking system.
        In this example, it returns a mock availability response.
        """
        service = args.get("service", "").lower()
        date = args.get("date", "")
        time = args.get("time", "")
        
        # Simple availability simulation - in a real application, this would
        # connect to your actual booking system
        if service in [s.lower() for s in self.services]:
            # Generate a simple availability response
            return SwaigFunctionResult(
                f"Yes, {service} is available on {date} at {time}. Would you like to make a reservation?"
            )
        else:
            return SwaigFunctionResult(
                f"I'm sorry, we don't offer {service} at {self.venue_name}. "
                f"Our available services are: {', '.join(self.services)}."
            )
    
    @AgentBase.tool(
        name="get_directions",
        description="Get directions to a specific location or amenity",
        parameters={
            "location": {
                "type": "string",
                "description": "The location or amenity to get directions to"
            }
        }
    )
    def get_directions(self, args, raw_data):
        """Provide directions to a specific location or amenity"""
        location = args.get("location", "").lower()
        
        # Check if the location is an amenity
        if location in self.amenities and "location" in self.amenities[location]:
            amenity_location = self.amenities[location]["location"]
            return SwaigFunctionResult(
                f"The {location} is located at {amenity_location}. "
                f"From the main entrance, follow the signs to {amenity_location}."
            )
        else:
            return SwaigFunctionResult(
                f"I don't have specific directions to {location}. "
                f"You can ask our staff at the front desk for assistance."
            )
    
    def on_summary(self, summary, raw_data=None):
        """
        Process the interaction summary
        
        Args:
            summary: Summary data from the conversation
            raw_data: The complete raw POST data from the request
        """
        if summary:
            try:
                # For structured summary
                if isinstance(summary, dict):
                    print(f"Concierge interaction summary: {json.dumps(summary, indent=2)}")
                else:
                    print(f"Concierge interaction summary: {summary}")
                    
                # Subclasses can override this to log or process the interaction
            except Exception as e:
                print(f"Error processing summary: {str(e)}")
