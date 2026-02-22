#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""


"""
Simple Dynamic Agent Example (New Way)

This example demonstrates the NEW dynamic configuration pattern.
The exact same agent as simple_static_agent.py, but configured dynamically per-request.

This agent does THE SAME THING as the static version:
- Uses a professional voice (inworld.Mark)
- Has a 500ms speech timeout
- Includes helpful hints
- Sets up global data with session info
- Has a customer service focused prompt

BUT... it's configured dynamically, which means:
- We COULD change any setting based on request parameters
- Configuration happens fresh for each request
- Easy to extend with parameter-based customization

Usage:
    python simple_dynamic_agent.py

Available at: http://localhost:3000/

Try these requests:
    curl "http://localhost:3000"
    curl "http://localhost:3000/debug"
    
Compare the SWML output - it will be IDENTICAL to the static version!
"""

from signalwire_agents import AgentBase

class SimpleDynamicAgent(AgentBase):
    def __init__(self):
        super().__init__(
            name="Simple Customer Service Agent (Dynamic)",
            auto_answer=True,
            record_call=True
        )
        
        # NO STATIC CONFIGURATION HERE!
        # Instead, we set up a dynamic configuration callback
        self.set_dynamic_config_callback(self.configure_agent_dynamically)
    
    def configure_agent_dynamically(self, query_params, body_params, headers, agent):
        """
        DYNAMIC CONFIGURATION - Called fresh for every request
        
        This method receives the agent instance directly, allowing you to
        configure it dynamically based on the request data.
        
        Args:
            query_params: Query string parameters from the request
            body_params: POST body parameters (empty for GET requests)
            headers: HTTP headers from the request
            agent: The agent instance to configure
        """
        
        # EXACT SAME CONFIGURATION as the static version
        # But now it happens fresh for every request!
        
        # Voice and language (same as static version)
        agent.add_language("English", "en-US", "inworld.Mark")
        
        # AI parameters (same as static version)
        agent.set_params({
            "ai_model": "gpt-4.1-nano",
            "end_of_speech_timeout": 500,
            "attention_timeout": 15000,
            "background_file_volume": -20
        })
        
        # Hints for speech recognition (same as static version)
        agent.add_hints([
            "SignalWire",
            "SWML",
            "API", 
            "webhook",
            "SIP"
        ])
        
        # Global data (same as static version)
        agent.set_global_data({
            "agent_type": "customer_service",
            "service_level": "standard",
            "features_enabled": ["basic_conversation", "help_desk"],
            "session_info": {
                "environment": "production",
                "version": "1.0"
            }
        })
        
        # Prompt sections (same as static version)
        agent.prompt_add_section(
            "Role and Purpose",
            "You are a professional customer service representative. Your goal is to help "
            "customers with their questions and provide excellent service."
        )
        
        agent.prompt_add_section(
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
        
        agent.prompt_add_section(
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
        
        # BONUS: We could easily add parameter-based customization here!
        # For example (commented out for this basic demo):
        #
        # if query_params.get('vip') == 'true':
        #     agent.add_language("English", "en-US", "inworld.Mark")  # Premium voice
        #     agent.set_params({"end_of_speech_timeout": 300})  # Faster response
        #     agent.update_global_data({"service_level": "vip"})
        #
        # customer_id = query_params.get('customer_id')
        # if customer_id:
        #     agent.update_global_data({"customer_id": customer_id})


if __name__ == "__main__":
    # Create and start the agent
    print("Starting Simple Dynamic Agent - configuration changes based on requests")
    print("Available at: http://localhost:3000/")
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    
    agent = SimpleDynamicAgent()
    agent.run() 