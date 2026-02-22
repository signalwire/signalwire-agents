#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Multi-Agent Server Example

This example demonstrates how to run multiple dynamic agents on the same server,
each with different paths and configurations.

This shows the power of the route parameter - you can host multiple specialized
agents on the same domain, each optimized for different use cases.

Available Agents:
- /healthcare - Healthcare-focused agent with HIPAA compliance
- /finance - Finance-focused agent with regulatory compliance  
- /retail - Retail/customer service agent with sales focus
- /dynamic - General dynamic agent that adapts based on parameters

Usage examples:

1. Healthcare Agent:
   curl "http://localhost:3000/healthcare?customer_id=patient123&urgency=high"

2. Finance Agent:
   curl "http://localhost:3000/finance?account_type=premium&service=investment"

3. Retail Agent:
   curl "http://localhost:3000/retail?department=electronics&customer_tier=vip"

4. Dynamic Agent:
   curl "http://localhost:3000/dynamic?tier=enterprise&industry=tech&voice=inworld.Mark"
"""

import os
import sys
import json
from datetime import datetime
from fastapi import FastAPI
from signalwire_agents import AgentBase
from comprehensive_dynamic_agent import ComprehensiveDynamicAgent

# Add the parent directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signalwire_agents import AgentServer
from signalwire_agents.prefabs import InfoGathererAgent
from signalwire_agents.core.function_result import SwaigFunctionResult


class CustomInfoGatherer(InfoGathererAgent):
    """
    Custom information gatherer that adds a save_info tool
    and overrides on_summary to do something with the collected data
    
    This demonstrates how to extend a prefab agent with:
    1. Custom SWAIG functions to perform additional actions
    2. Custom summary handling to process collected information
    3. Action integration (SMS sending) as part of the agent response
    """
    
    def __init__(self):
        #------------------------------------------------------------------------
        # PREFAB CONFIGURATION
        # Configure the base InfoGathererAgent with our specific requirements
        #------------------------------------------------------------------------
        
        # Initialize with field definitions
        # The InfoGathererAgent prefab will automatically create an agent that:
        # - Collects these specific pieces of information
        # - Validates that the data matches required formats
        # - Handles clarifications and corrections
        # - Provides a confirmation once all data is collected
        super().__init__(
            name="registration",           # Agent identifier and logging name
            route="/register",             # HTTP endpoint path
            fields=[                       # Information to collect
                {"name": "full_name", "prompt": "What is your full name?"},
                {"name": "email", "prompt": "What is your email address?"},
                {"name": "phone", "prompt": "What is your phone number?"}
            ],
            # Template uses {field_name} to insert collected data
            confirmation_template="Thanks {full_name}! We've recorded your contact info: {email} and {phone}."
        )
        
        #------------------------------------------------------------------------
        # CUSTOM TOOL DEFINITION
        # Add additional capabilities not included in the base prefab
        #------------------------------------------------------------------------
        
        # Add a tool for saving info to CRM
        # This is a custom extension not part of the base InfoGathererAgent
        self.define_tool(
            name="save_to_crm",           # Function name the AI will use
            description="Save customer information to the CRM system",  # Description for the AI
            parameters={                   # Parameter schema
                "name": {"type": "string", "description": "Customer name"},
                "email": {"type": "string", "description": "Customer email"},
                "phone": {"type": "string", "description": "Customer phone"}
            },
            handler=self.save_to_crm       # Method to call when the function is invoked
        )
    
    def save_to_crm(self, args, raw_data):
        """
        Tool handler for saving info to CRM
        
        This function demonstrates how to:
        1. Process collected information
        2. Integrate with external systems
        3. Return actions that trigger additional behaviors
        
        Args:
            args: Dictionary with name, email, and phone parameters
            raw_data: Complete request data
            
        Returns:
            SwaigFunctionResult with confirmation and SMS action
        """
        # Extract parameters from the args dictionary
        name = args.get("name", "")
        email = args.get("email", "")
        phone = args.get("phone", "")
        
        print(f"Saving to CRM: {name}, {email}, {phone}")
        
        # Simulate CRM save by writing to a local file
        # In a real implementation, this would call a CRM API
        with open("customer_data.json", "a") as f:
            f.write(json.dumps({
                "name": name,
                "email": email,
                "phone": phone,
                "timestamp": datetime.now().isoformat()
            }) + "\n")
        
        # Return success response with an additional action
        # This demonstrates how to trigger SMS sending as part of the function result
        return (
            SwaigFunctionResult("I've saved your information to our system.")
            .add_action("send_sms", 
                       {"to": phone,                # Phone number to send to
                       "message": f"Thanks {name} for registering!"})  # SMS content
        )
    
    def on_summary(self, summary, raw_data=None):
        """
        Process the conversation summary data
        
        This method is called after all information has been collected and the post-prompt
        has generated a summary. It allows for final processing of the data.
        
        Args:
            summary: JSON structure containing the collected information
            raw_data: Complete request data
        """
        print(f"Registration completed: {json.dumps(summary, indent=2)}")
        
        # Additional processing could happen here
        # For example:
        # - Saving to a database
        # - Sending confirmation emails
        # - Triggering workflow events


class SupportInfoGatherer(InfoGathererAgent):
    """
    Support ticket information gatherer
    
    This agent demonstrates:
    1. Using validation rules in field definitions
    2. Customizing the structure of the summary data
    3. Simple processing of submitted support tickets
    """
    
    def __init__(self):
        #------------------------------------------------------------------------
        # PREFAB CONFIGURATION
        # Configure the InfoGathererAgent for support ticket collection
        #------------------------------------------------------------------------
        
        # Initialize with ticket field definitions
        super().__init__(
            name="support",
            route="/support",
            fields=[
                {"name": "name", "prompt": "What is your name?"},
                {"name": "issue", "prompt": "Please describe the issue you're experiencing."},
                # This field includes validation to ensure the value is a number between 1-5
                {"name": "urgency", "prompt": "On a scale of 1-5, how urgent is this issue?", 
                 "validation": "Must be a number between 1 and 5"}
            ],
            confirmation_template="Thanks {name}. We've recorded your {urgency}-priority issue and will respond soon.",
            # Define a custom structure for the summary data
            # The %{field_name} syntax inserts the collected data into the specified structure
            summary_format={
                "customer": {
                    "name": "%{name}"
                },
                "ticket": {
                    "description": "%{issue}",
                    "priority": "%{urgency}"
                }
            }
        )
    
    def on_summary(self, summary, raw_data=None):
        """
        Process the ticket data after collection is complete
        
        This method is called when the conversation has ended
        and all required information has been collected.
        
        Args:
            summary: The structured ticket data as configured in summary_format
            raw_data: Complete request data
        """
        print(f"Support ticket created: {json.dumps(summary, indent=2)}")
        
        # In a real implementation, you might:
        # - Create a ticket in a ticketing system
        # - Send a notification to support staff
        # - Add the ticket to a queue based on priority


class HealthcareAgent(AgentBase):
    def __init__(self):
        super().__init__(
            name="Healthcare AI Assistant",
            route="/healthcare",
            auto_answer=True,
            record_call=True
        )
        
        # Set up healthcare-specific dynamic configuration
        self.set_dynamic_config_callback(self.configure_healthcare_agent)
        
        # Base healthcare configuration
        self.prompt_add_section(
            "Healthcare Role",
            "You are a HIPAA-compliant healthcare AI assistant. You help patients and "
            "healthcare providers with information, scheduling, and basic guidance."
        )
        
        self.prompt_add_section(
            "Compliance Guidelines",
            "Always maintain patient privacy and confidentiality:",
            bullets=[
                "Never share patient information with unauthorized parties",
                "Direct medical diagnoses to qualified healthcare providers",
                "Use appropriate medical terminology",
                "Maintain professional, caring communication"
            ]
        )

    def configure_healthcare_agent(self, query_params, body_params, headers, agent):
        """Configure agent based on healthcare-specific parameters"""
        customer_id = query_params.get('customer_id', '')
        urgency = query_params.get('urgency', 'normal').lower()
        department = query_params.get('department', 'general').lower()
        
        # Configure voice based on urgency
        if urgency == 'high':
            agent.add_language("English", "en-US", "inworld.Sarah")  # Clear, professional voice
            agent.set_params({"ai_model": "gpt-4.1-nano", "end_of_speech_timeout": 300})  # Faster response
        else:
            agent.add_language("English", "en-US", "inworld.Mark")  # Professional voice
            agent.set_params({"ai_model": "gpt-4.1-nano", "end_of_speech_timeout": 500})  # Normal response
        
        # Department-specific configuration
        if department == 'emergency':
            agent.prompt_add_section(
                "Emergency Protocol",
                "Emergency department protocols are in effect:",
                bullets=[
                    "Prioritize urgent medical situations",
                    "Escalate immediately to medical staff when needed",
                    "Provide clear, concise information",
                    "Maintain calm, professional demeanor"
                ]
            )
        
        # Set global data
        agent.set_global_data({
            "customer_id": customer_id,
            "urgency_level": urgency,
            "department": department,
            "compliance_level": "hipaa",
            "session_type": "healthcare"
        })


class FinanceAgent(AgentBase):
    def __init__(self):
        super().__init__(
            name="Financial Services AI",
            route="/finance",
            auto_answer=True,
            record_call=True
        )
        
        self.set_dynamic_config_callback(self.configure_finance_agent)
        
        self.prompt_add_section(
            "Financial Services Role",
            "You are a financial services AI assistant specializing in banking, "
            "investments, and financial planning guidance."
        )
        
        self.prompt_add_section(
            "Regulatory Compliance",
            "Adhere to financial industry regulations:",
            bullets=[
                "Protect sensitive financial information",
                "Never provide specific investment advice without disclaimers",
                "Refer complex matters to licensed financial advisors",
                "Maintain accurate, professional communication"
            ]
        )

    def configure_finance_agent(self, query_params, body_params, headers, agent):
        """Configure agent based on finance-specific parameters"""
        account_type = query_params.get('account_type', 'standard').lower()
        service = query_params.get('service', 'general').lower()
        customer_id = query_params.get('customer_id', '')
        
        # Voice and parameters based on account type
        if account_type == 'premium':
            agent.add_language("English", "en-US", "inworld.Sarah")
            agent.set_params({
                "ai_model": "gpt-4.1-nano",
                "end_of_speech_timeout": 600,
                "attention_timeout": 20000
            })
        elif account_type == 'wealth':
            agent.add_language("English", "en-US", "inworld.Hanna")
            agent.set_params({
                "ai_model": "gpt-4.1-nano",
                "end_of_speech_timeout": 800,
                "attention_timeout": 25000
            })
        else:
            agent.add_language("English", "en-US", "inworld.Mark")
            agent.set_params({
                "ai_model": "gpt-4.1-nano",
                "end_of_speech_timeout": 400,
                "attention_timeout": 15000
            })
        
        # Service-specific prompts
        if service == 'investment':
            agent.prompt_add_section(
                "Investment Services",
                "Investment consultation services:",
                bullets=[
                    "Provide educational information about investment options",
                    "Explain market trends and analysis",
                    "Connect with licensed investment advisors for specific advice",
                    "Discuss risk tolerance and financial goals"
                ]
            )
        elif service == 'lending':
            agent.prompt_add_section(
                "Lending Services",
                "Loan and credit services:",
                bullets=[
                    "Explain loan products and terms",
                    "Assist with application processes",
                    "Provide credit education and guidance",
                    "Connect with loan specialists as needed"
                ]
            )
        
        agent.set_global_data({
            "customer_id": customer_id,
            "account_type": account_type,
            "service_area": service,
            "compliance_level": "financial",
            "session_type": "finance"
        })


class RetailAgent(AgentBase):
    def __init__(self):
        super().__init__(
            name="Retail Customer Service AI",
            route="/retail",
            auto_answer=True,
            record_call=True
        )
        
        self.set_dynamic_config_callback(self.configure_retail_agent)
        
        self.prompt_add_section(
            "Customer Service Role",
            "You are a friendly retail customer service AI assistant focused on "
            "providing excellent customer experiences and sales support."
        )
        
        self.prompt_add_section(
            "Service Excellence",
            "Customer service principles:",
            bullets=[
                "Maintain friendly, helpful demeanor",
                "Listen actively to customer needs",
                "Provide accurate product information",
                "Look for opportunities to enhance the shopping experience"
            ]
        )

    def configure_retail_agent(self, query_params, body_params, headers, agent):
        """Configure agent based on retail-specific parameters"""
        department = query_params.get('department', 'general').lower()
        customer_tier = query_params.get('customer_tier', 'standard').lower()
        customer_id = query_params.get('customer_id', '')
        
        # Voice based on customer tier
        if customer_tier == 'vip':
            agent.add_language("English", "en-US", "inworld.Sarah")
            agent.set_params({"ai_model": "gpt-4.1-nano", "end_of_speech_timeout": 600})
        else:
            agent.add_language("English", "en-US", "inworld.Mark")
            agent.set_params({"ai_model": "gpt-4.1-nano", "end_of_speech_timeout": 400})
        
        # Department-specific knowledge
        if department == 'electronics':
            agent.prompt_add_section(
                "Electronics Expertise",
                "Electronics department specialization:",
                bullets=[
                    "Detailed knowledge of electronic products",
                    "Technical specifications and compatibility",
                    "Warranty and service information",
                    "Installation and setup guidance"
                ]
            )
        elif department == 'clothing':
            agent.prompt_add_section(
                "Fashion and Apparel",
                "Clothing department specialization:",
                bullets=[
                    "Style and fashion guidance",
                    "Size and fit recommendations",
                    "Care and maintenance instructions",
                    "Return and exchange policies"
                ]
            )
        
        agent.set_global_data({
            "customer_id": customer_id,
            "department": department,
            "customer_tier": customer_tier,
            "session_type": "retail"
        })


def create_multi_agent_app():
    """Create a FastAPI app with multiple agents"""
    app = FastAPI(title="Multi-Agent AI Server", redirect_slashes=False)
    
    # Create all agents
    healthcare_agent = HealthcareAgent()
    finance_agent = FinanceAgent()
    retail_agent = RetailAgent()
    dynamic_agent = ComprehensiveDynamicAgent("/dynamic")
    
    # Add all agents to the app
    app.include_router(healthcare_agent.as_router())
    app.include_router(finance_agent.as_router())
    app.include_router(retail_agent.as_router())
    app.include_router(dynamic_agent.as_router())
    
    # Add a root endpoint that lists all available agents
    @app.get("/")
    async def list_agents():
        return {
            "message": "Multi-Agent AI Server",
            "available_agents": {
                "/healthcare": "Healthcare AI Assistant - HIPAA compliant medical support",
                "/finance": "Financial Services AI - Banking and investment guidance", 
                "/retail": "Retail Customer Service AI - Shopping and product support",
                "/dynamic": "Dynamic AI Agent - Adapts based on request parameters"
            },
            "usage": "Send GET or POST requests to any agent path with appropriate query parameters"
        }
    
    return app


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Multi-Agent AI Server")
    print("\nAvailable agents:")
    print("- http://localhost:3000/healthcare - Healthcare AI (HIPAA compliant)")
    print("- http://localhost:3000/finance - Financial Services AI")
    print("- http://localhost:3000/retail - Retail Customer Service AI")
    print("- http://localhost:3000/dynamic - Dynamic AI (adapts to parameters)")
    print("\nExample requests:")
    print("curl 'http://localhost:3000/healthcare?customer_id=patient123&urgency=high'")
    print("curl 'http://localhost:3000/finance?account_type=premium&service=investment'")
    print("curl 'http://localhost:3000/retail?department=electronics&customer_tier=vip'")
    print("curl 'http://localhost:3000/dynamic?tier=enterprise&industry=tech&voice=inworld.Mark'")
    print()
    
    app = create_multi_agent_app()
    uvicorn.run(app, host="0.0.0.0", port=3000) 