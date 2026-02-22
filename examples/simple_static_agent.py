#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""


"""
Simple Static Agent Example (Traditional Way)

This example demonstrates the traditional way of configuring an agent with static settings.
All configuration is done during initialization and remains the same for every request.

This agent:
- Uses a professional voice (inworld.Mark)
- Has a 500ms speech timeout
- Includes helpful hints
- Sets up global data with session info
- Has a customer service focused prompt

Usage:
    python simple_static_agent.py

Available at: http://localhost:3000/

Try these requests:
    curl "http://localhost:3000"
    curl "http://localhost:3000/debug"
"""

from signalwire_agents import AgentBase

class SimpleStaticAgent(AgentBase):
    def __init__(self):
        super().__init__(
            name="Simple Customer Service Agent",
            auto_answer=True,
            record_call=True
        )
        
        # STATIC CONFIGURATION - Set once during initialization
        
        # Voice and language (never changes)
        self.add_language("English", "en-US", "inworld.Mark")
        
        # AI parameters (never changes)
        self.set_params({
            "ai_model": "gpt-4.1-nano",
            "end_of_speech_timeout": 500,
            "attention_timeout": 15000,
            "background_file_volume": -20
        })
        
        # Hints for speech recognition (never changes)
        self.add_hints([
            "SignalWire",
            "SWML", 
            "API",
            "webhook",
            "SIP"
        ])
        
        # Global data (same for every call)
        self.set_global_data({
            "agent_type": "customer_service",
            "service_level": "standard",
            "features_enabled": ["basic_conversation", "help_desk"],
            "session_info": {
                "environment": "production",
                "version": "1.0"
            }
        })
        
        # Prompt sections (never changes)
        self.prompt_add_section(
            "Role and Purpose",
            "You are a professional customer service representative. Your goal is to help "
            "customers with their questions and provide excellent service."
        )
        
        self.prompt_add_section(
            "Guidelines",
            "Follow these customer service principles:",
            bullets=[
                "Listen carefully to customer needs",
                "Provide accurate and helpful information", 
                "Maintain a professional and friendly tone",
                "Escalate complex issues when appropriate",
                "Always confirm understanding before ending"
            ]
        )
        
        self.prompt_add_section(
            "Available Services",
            "You can help customers with:",
            bullets=[
                "General product information",
                "Account questions and support",
                "Technical troubleshooting guidance",
                "Billing and payment inquiries",
                "Service status and updates"
            ]
        )


if __name__ == "__main__":
    agent = SimpleStaticAgent()
    
    print("Starting Simple Static Agent")
    print("\nConfiguration: STATIC (set once at startup)")
    print("- Voice: inworld.Mark (professional)")
    print("- Service Level: standard") 
    print("- Speech Timeout: 500ms")
    print("- Features: basic conversation, help desk")
    print("\nAgent available at: http://localhost:3000/")
    print("\nTry these requests:")
    print("curl 'http://localhost:3000'")
    print("curl 'http://localhost:3000/debug'")
    print("\nNote: Configuration never changes regardless of request parameters")
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    print()
    
    agent.run() 