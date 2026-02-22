#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""


"""
Simple Dynamic Agent - Enhanced Example

This example shows the REAL POWER of dynamic configuration by actually using
request parameters to customize the agent behavior.

Same base agent as the previous examples, but now it adapts based on:
- vip: true/false (changes voice and response speed)
- department: sales/support/billing (changes expertise and prompts)
- customer_id: any string (personalizes the experience)
- language: en/es (changes language and voice)

Usage:
    python simple_dynamic_enhanced.py

Available at: http://localhost:3000/

Try these requests to see the differences:

1. Basic request (default configuration):
   curl "http://localhost:3000"

2. VIP customer:
   curl "http://localhost:3000?vip=true&customer_id=CUST123"

3. Sales department with Spanish:
   curl "http://localhost:3000?department=sales&language=es&customer_id=Maria"

4. Billing support:
   curl "http://localhost:3000?department=billing&vip=true"

5. Compare the /debug endpoint:
   curl "http://localhost:3000/debug?vip=true&department=sales&customer_id=CUST456"
"""

from signalwire_agents import AgentBase

class SimpleDynamicEnhanced(AgentBase):
    def __init__(self):
        super().__init__(
            name="Enhanced Dynamic Customer Service Agent",
            auto_answer=True,
            record_call=True
        )
        
        # Set up dynamic configuration callback
        self.set_dynamic_config_callback(self.configure_agent_dynamically)
    
    def configure_agent_dynamically(self, query_params, body_params, headers, agent):
        """
        ENHANCED DYNAMIC CONFIGURATION - Adapts based on request parameters
        
        This shows the real power of the dynamic pattern!
        """
        
        # Extract parameters from the request
        is_vip = query_params.get('vip', '').lower() == 'true'
        department = query_params.get('department', 'general').lower()
        customer_id = query_params.get('customer_id', '')
        language = query_params.get('language', 'en').lower()
        
        # === VOICE AND LANGUAGE CONFIGURATION ===
        if language == 'es':
            if is_vip:
                agent.add_language("Spanish", "es-ES", "inworld.Sarah")  # Premium for VIP
            else:
                agent.add_language("Spanish", "es-ES", "inworld.Mark")  # Standard Spanish
        else:  # English
            if is_vip:
                agent.add_language("English", "en-US", "inworld.Sarah")  # Premium voice for VIP
            else:
                agent.add_language("English", "en-US", "inworld.Mark")  # Standard professional voice
        
        # === AI PARAMETERS (VIP gets faster response) ===
        if is_vip:
            agent.set_params({
                "ai_model": "gpt-4.1-nano",
                "end_of_speech_timeout": 300,  # Faster for VIP
                "attention_timeout": 20000,     # Longer attention for VIP
                "background_file_volume": -30   # Better audio quality
            })
        else:
            agent.set_params({
                "ai_model": "gpt-4.1-nano",
                "end_of_speech_timeout": 500,  # Standard timing
                "attention_timeout": 15000,     # Standard attention span
                "background_file_volume": -20   # Standard audio
            })
        
        # === HINTS (Speech recognition aids based on department) ===
        base_hints = [
            "SignalWire",
            "SWML",
            "API",
            "webhook",
            "SIP"
        ]
        
        if department == 'sales':
            base_hints.extend([
                "pricing",
                "enterprise",
                "premium",
                "subscription",
                "upgrade"
            ])
        elif department == 'billing':
            base_hints.extend([
                "invoice",
                "payment",
                "billing",
                "account",
                "charges"
            ])
        else:  # general support
            base_hints.extend([
                "support",
                "technical",
                "troubleshoot",
                "configuration",
                "integration"
            ])
        
        agent.add_hints(base_hints)
        
        # === GLOBAL DATA (Customer-specific information) ===
        global_data = {
            "agent_type": "customer_service",
            "service_level": "vip" if is_vip else "standard",
            "department": department,
            "language": language,
            "features_enabled": ["basic_conversation", "help_desk"],
            "session_info": {
                "environment": "production",
                "version": "1.0",
                "request_timestamp": "2024-01-01T00:00:00Z"  # Would be actual timestamp
            }
        }
        
        if customer_id:
            global_data["customer_id"] = customer_id
            global_data["personalized"] = True
        
        if is_vip:
            global_data["features_enabled"].extend(["priority_support", "premium_features"])
        
        agent.set_global_data(global_data)
        
        # === DYNAMIC PROMPTS (Department-specific) ===
        
        # Base role (personalized if we have customer_id)
        if customer_id:
            role_text = f"You are a professional customer service representative helping customer {customer_id}. "
        else:
            role_text = "You are a professional customer service representative. "
            
        if is_vip:
            role_text += "This is a VIP customer who receives priority service and premium support."
        else:
            role_text += "Your goal is to help customers with their questions and provide excellent service."
        
        agent.prompt_add_section("Role and Purpose", role_text)
        
        # Department-specific expertise
        if department == 'sales':
            agent.prompt_add_section(
                "Sales Expertise",
                "You specialize in sales and product consultation:",
                bullets=[
                    "Present product features and benefits clearly",
                    "Understand customer needs and match solutions",
                    "Handle pricing questions and special offers",
                    "Process orders and upgrades",
                    "Provide product comparisons and recommendations"
                ]
            )
        elif department == 'billing':
            agent.prompt_add_section(
                "Billing Expertise", 
                "You specialize in billing and account management:",
                bullets=[
                    "Explain billing statements and charges",
                    "Process payment arrangements and updates",
                    "Handle dispute resolution professionally",
                    "Verify account information securely",
                    "Provide payment history and projections"
                ]
            )
        else:  # general support
            agent.prompt_add_section(
                "General Support Guidelines",
                "Follow these customer service principles:",
                bullets=[
                    "Listen carefully to customer needs",
                    "Provide accurate and helpful information",
                    "Maintain a professional and friendly tone",
                    "Escalate complex issues when appropriate",
                    "Always confirm understanding before ending"
                ]
            )
        
        # VIP-specific service standards
        if is_vip:
            agent.prompt_add_section(
                "VIP Service Standards",
                "This customer receives premium service:",
                bullets=[
                    "Provide immediate attention and priority handling",
                    "Offer premium solutions and exclusive options",
                    "Ensure complete satisfaction before concluding",
                    "Document all interactions for continuity",
                    "Proactively suggest additional value-added services"
                ]
            )
        
        # Available services (department-specific)
        if department == 'sales':
            services = [
                "Product demonstrations and information",
                "Custom solution design",
                "Pricing and proposal generation",
                "Order processing and tracking",
                "Upgrade and expansion planning"
            ]
        elif department == 'billing':
            services = [
                "Billing explanation and breakdown",
                "Payment processing and plans",
                "Account balance and history",
                "Dispute investigation and resolution",
                "Billing preference updates"
            ]
        else:
            services = [
                "General product information",
                "Account questions and support",
                "Technical troubleshooting guidance",
                "Service status and updates",
                "General inquiries and assistance"
            ]
        
        agent.prompt_add_section("Available Services", f"You can help customers with:", bullets=services)


def main():
    agent = SimpleDynamicEnhanced()
    
    print("Starting Enhanced Dynamic Agent")
    print("\nConfiguration: ENHANCED DYNAMIC (adapts to request parameters)")
    print("\nSupported parameters:")
    print("- vip=true/false (premium voice, faster response)")
    print("- department=sales/support/billing (specialized expertise)")
    print("- customer_id=<string> (personalized experience)")
    print("- language=en/es (language and voice selection)")
    print("\nAgent available at: http://localhost:3000/")
    print("\nFEATURES: Try these requests to see the differences:")
    print("curl 'http://localhost:3000'  # Basic")
    print("curl 'http://localhost:3000?vip=true&customer_id=CUST123'  # VIP")
    print("curl 'http://localhost:3000?department=sales&language=es'  # Spanish Sales")
    print("curl 'http://localhost:3000?department=billing&vip=true'  # VIP Billing")
    print("curl 'http://localhost:3000/debug?vip=true&department=sales'  # Debug mode")
    print("\nCOMPARE: Compare the SWML output - each request creates different configuration!")
    print()
    
    print("\nStarting agent server...")
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    agent.run()


if __name__ == "__main__":
    main() 