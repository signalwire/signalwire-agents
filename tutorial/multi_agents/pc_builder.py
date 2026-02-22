#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
PC Builder Demo Service - Multiple Agent Architecture

This service provides specialized PC building assistance through three dedicated agents:
- Triage Agent (/) - Routes customers to appropriate specialists
- Sales Agent (/sales) - Handles product recommendations and purchases  
- Support Agent (/support) - Provides technical support and troubleshooting

Key Feature: Uses SignalWire's swml_transfer skill with required_fields for automatic
context preservation. The skill ensures both the customer's name and a comprehensive summary
are collected before transferring and makes them available via ${call_data.user_name}
and ${call_data.summary} to the receiving agent.
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional
from signalwire_agents import AgentBase, AgentServer
from signalwire_agents.core.function_result import SwaigFunctionResult
from signalwire_agents.core.logging_config import get_logger

# Set up logger for this module
logger = get_logger(__name__)

# Define the Triage Agent (root route)
class TriageAgent(AgentBase):
    def __init__(self):
        super().__init__(
            name="PC Builder Triage Agent",
            route="/",  # Root route
            host="0.0.0.0",
            port=3001
        )
        
        # Configure prompt using POM
        self._configure_prompt()
        
        # Configure language and voice for Alex persona
        self.add_language(
            name="English",
            code="en-US",
            voice="rime.spore"
        )
        
        # Set up dynamic configuration for URL-dependent tools
        self.set_dynamic_config_callback(self.configure_transfer_tools)
    
    def configure_transfer_tools(self, query_params, body_params, headers, agent):
        """
        DYNAMIC CONFIGURATION - Called fresh for every request
        
        This uses the swml_transfer skill with correct URLs after proxy detection is available.
        
        Args:
            query_params: Query string parameters from the request
            body_params: POST body parameters (empty for GET requests)
            headers: HTTP headers from the request
            agent: The agent instance to configure
        """
        # Build URLs with proper proxy detection
        sales_url = agent.get_full_url(include_auth=True).rstrip('/') + "/sales?transfer=true"
        support_url = agent.get_full_url(include_auth=True).rstrip('/') + "/support?transfer=true"
        
        # Configure transfers using swml_transfer skill
        agent.add_skill("swml_transfer", {
            "tool_name": "transfer_to_specialist",
            "description": "Transfer to sales or support specialist with conversation summary",
            "parameter_name": "specialist_type",
            "parameter_description": "The type of specialist to transfer to (sales or support)",
            "required_fields": {
                "user_name": "The customer's name",
                "summary": "A comprehensive summary of the conversation so far, including what the customer needs help with"
            },
            "transfers": {
                "/sales/i": {
                    "url": sales_url,
                    "message": "Perfect! Let me transfer you to our sales specialist right away.",
                    "return_message": "The call with the sales specialist is complete. How else can I help you?"
                },
                "/support/i": {
                    "url": support_url,
                    "message": "I'll connect you with our technical support specialist right away.",
                    "return_message": "The call with the support specialist is complete. How else can I help you?"
                }
            },
            "default_message": "I can transfer you to either our sales or support specialist. Which would you prefer?"
        })
    
    def _configure_prompt(self):
        """Configure the prompt for the triage agent using POM"""
        self.prompt_add_section(
            "AI Role",
            body=(
                "You are Alex, the friendly front desk assistant at PC Builder Pro. "
                "You're enthusiastic about technology and love helping customers find "
                "the right specialist for their needs. Introduce yourself by name when "
                "greeting customers."
            )
        )
        
        self.prompt_add_section(
            "Your Tasks",
            body="Guide customers through the initial triage process with enthusiasm and energy.",
            bullets=[
                "Greet the customer warmly with your signature enthusiasm",
                "Ask for their name in a friendly way",
                "Determine if they need sales (buying/building) or support (technical issues)",
                "Get a brief description of what they need help with",
                "Prepare a comprehensive summary before transferring",
                "Use transfer_to_specialist with both the destination and summary"
            ]
        )
        
        self.prompt_add_section(
            "Voice Instructions",
            body="Speak with energy and enthusiasm about technology."
        )
        
        self.prompt_add_section(
            "Important",
            body="Follow these key guidelines for effective triage:",
            bullets=[
                "Always get the customer's name first",
                "Ask clarifying questions to determine sales vs support",
                "The transfer_to_specialist function requires both specialist_type AND summary",
                "Include customer name, their needs, and reason for transfer in the summary"
            ]
        )
        
        self.prompt_add_section(
            "Summary Example",
            body=(
                "When transferring, provide a summary like: 'Customer John Smith is "
                "interested in building a gaming PC with a budget of $2000. He needs "
                "help selecting compatible components and wants recommendations for "
                "the best performance within his budget.'"
            )
        )


# Define the Sales Agent
class SalesAgent(AgentBase):
    def __init__(self):
        super().__init__(
            name="PC Builder Sales Specialist",
            route="/sales",
            host="0.0.0.0", 
            port=3001
        )
        
        # Set up dynamic configuration to check for transfer
        self.set_dynamic_config_callback(self.configure_dynamic_prompt)
        
        # Configure prompt using POM
        self._configure_prompt()
        
        # Configure language and voice for Morgan persona
        self.add_language(
            name="English",
            code="en-US",
            voice="rime.marsh"
        )
        
        # Add search capability for sales knowledge
        self.add_skill("native_vector_search", {
            "tool_name": "search_sales_knowledge",
            "description": "Search sales and product information",
            "index_file": "sales_knowledge.swsearch",
            "count": 3
        })
        
        # Define sales-specific functions
        @self.tool("create_build_recommendation", description="Create a custom PC build recommendation")
        async def create_build_recommendation(budget: str, use_case: str, preferences: str):
            return SwaigFunctionResult(
                f"Based on your ${budget} budget for {use_case}, I recommend: "
                f"[Custom build details would be generated here based on current "
                f"market data and your preferences: {preferences}]"
            )
        
        @self.tool("check_component_compatibility", description="Check if PC components are compatible")
        async def check_component_compatibility(components: str):
            return SwaigFunctionResult(
                f"Compatibility check for: {components} - "
                f"[Detailed compatibility analysis would be performed here]"
            )
    
    def configure_dynamic_prompt(self, query_params, body_params, headers, agent):
        """
        Dynamic configuration callback to add transfer information when transfer=true
        
        Args:
            query_params: Query string parameters from the request
            body_params: POST body parameters (empty for GET requests)
            headers: HTTP headers from the request
            agent: The agent instance to configure
        """
        # Check if this is a transfer call
        if query_params.get('transfer') == 'true':
            agent.prompt_add_section(
                "Call Transfer Information",
                body="This call has been transferred to you from the triage agent.",
                bullets=[
                    "The customer's name is ${call_data.user_name} - greet them by name",
                    "They were transferred because: ${call_data.summary}",
                    "Start by greeting them by name and acknowledging why they were transferred",
                    (
                        "Example: 'Hi ${call_data.user_name}, I'm Morgan! I understand "
                        "you're looking to build a gaming PC with a $2000 budget. I'm "
                        "excited to help you build the perfect system!'"
                    )
                ]
            )
        else:
            agent.prompt_add_section(
                "Initial Greeting",
                body="This is a direct call to the sales department.",
                bullets=[
                    "Greet the customer warmly and professionally",
                    "Introduce yourself as a PC building sales specialist",
                    "Ask for their name",
                    "Ask how you can help them today",
                    (
                        "Example: 'Hello! Welcome to PC Builder Pro sales. I'm Morgan, "
                        "your PC building specialist. May I have your name, and how can "
                        "I help you build something amazing today?'"
                    )
                ]
            )
    
    def _configure_prompt(self):
        """Configure the prompt for the sales agent using POM"""
        self.prompt_add_section(
            "AI Role",
            body=(
                "You are Morgan, a passionate PC building expert and sales specialist "
                "at PC Builder Pro. You're known for your deep knowledge of components "
                "and your ability to match customers with their perfect build. You get "
                "excited about the latest hardware and love sharing that enthusiasm. "
                "Always introduce yourself by name."
            )
        )
        
        self.prompt_add_section(
            "Your Expertise",
            body="Areas of specialization:",
            bullets=[
                "Custom PC builds for all budgets",
                "Component compatibility and optimization",
                "Performance recommendations",
                "Price/performance analysis",
                "Current market trends"
            ]
        )
        
        self.prompt_add_section(
            "Your Tasks",
            body="Complete sales process workflow with passion and expertise:",
            bullets=[
                "Understand their specific PC building requirements with genuine interest",
                "Ask about budget, intended use, and preferences enthusiastically",
                "Search knowledge base for current product info",
                "Create customized build recommendations with excitement about the possibilities",
                "Help with component selection and compatibility while sharing your expertise"
            ]
        )
        
        self.prompt_add_section(
            "Voice Instructions",
            body=(
                "Share your passion for PC building and get excited about "
                "helping customers create their perfect system."
            )
        )
        
        self.prompt_add_section(
            "Tools Available",
            body="Use these tools to assist customers:",
            bullets=[
                "search_sales_knowledge: Find current product information",
                "create_build_recommendation: Generate custom build suggestions",
                "check_component_compatibility: Verify component compatibility"
            ]
        )
        
        self.prompt_add_section(
            "Important",
            body="Key guidelines for sales interactions:",
            bullets=[
                "Ask clarifying questions about their specific requirements",
                "Use search to get current pricing and availability",
                "Provide detailed explanations for recommendations"
            ]
        )


# Define the Support Agent  
class SupportAgent(AgentBase):
    def __init__(self):
        super().__init__(
            name="PC Builder Support Specialist",
            route="/support",
            host="0.0.0.0",
            port=3001
        )

        # Set up dynamic configuration to check for transfer
        self.set_dynamic_config_callback(self.configure_dynamic_prompt)
        
        # Configure prompt using POM
        self._configure_prompt()
        
        # Configure language and voice for Sam persona
        self.add_language(
            name="English",
            code="en-US",
            voice="rime.cove"
        )
        
        # Add search capability for support knowledge
        self.add_skill("native_vector_search", {
            "tool_name": "search_support_knowledge", 
            "description": "Search technical support and troubleshooting information",
            "index_file": "support_knowledge.swsearch",
            "count": 3
        })
        
        # Define support-specific functions
        @self.tool("diagnose_hardware_issue", description="Help diagnose PC hardware problems")
        async def diagnose_hardware_issue(symptoms: str, system_specs: str):
            return SwaigFunctionResult(
                f"For symptoms '{symptoms}' on system '{system_specs}': "
                f"[Diagnostic steps and potential solutions would be provided here]"
            )
        
        @self.tool("create_support_ticket", description="Create a support ticket for complex issues")
        async def create_support_ticket(issue_description: str, customer_info: str, priority: str):
            ticket_id = f"SUP-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            return SwaigFunctionResult(
                f"Support ticket {ticket_id} created for: {issue_description}. "
                f"Priority: {priority}. We'll follow up within 24 hours."
            )

    def configure_dynamic_prompt(self, query_params, body_params, headers, agent):
        """
        Dynamic configuration callback to add transfer information when transfer=true
        
        Args:
            query_params: Query string parameters from the request
            body_params: POST body parameters (empty for GET requests)
            headers: HTTP headers from the request
            agent: The agent instance to configure
        """
        # Check if this is a transfer call
        if query_params.get('transfer') == 'true':
            agent.prompt_add_section(
                "Call Transfer Information",
                body="This call has been transferred to you from the triage agent.",
                bullets=[
                    "The customer's name is ${call_data.user_name} - greet them by name",
                    "They were transferred because: ${call_data.summary}",
                    "Start by greeting them by name and acknowledging their technical issue",
                    (
                        "Example: 'Hi ${call_data.user_name}, I'm Sam. I understand you're "
                        "experiencing issues with your PC not booting. Let's work through "
                        "this together and get your system back up and running.'"
                    )
                ]
            )
        else:
            agent.prompt_add_section(
                "Initial Greeting",
                body="This is a direct call to the support department.",
                bullets=[
                    "Greet the customer warmly and professionally",
                    "Introduce yourself as a technical support specialist",
                    "Ask for their name",
                    "Ask what technical issue they're experiencing",
                    (
                        "Example: 'Hello! Welcome to PC Builder Pro technical support. "
                        "I'm Sam, and I'm here to help solve any technical issues you're "
                        "facing. May I have your name, and what can I help you troubleshoot today?'"
                    )
                ]
            )
        
    def _configure_prompt(self):
        """Configure the prompt for the support agent using POM"""
        self.prompt_add_section(
            "AI Role",
            body=(
                "You are Sam, a patient and methodical technical support specialist at "
                "PC Builder Pro. You have a calming presence and excel at breaking down "
                "complex technical problems into simple steps. You're known for never "
                "giving up on a problem until it's solved. Always introduce yourself by name."
            )
        )
        
        
        self.prompt_add_section(
            "Your Expertise",
            body="Areas of technical specialization:",
            bullets=[
                "Hardware troubleshooting and diagnostics",
                "Software compatibility issues",
                "System optimization and performance",
                "Component failure analysis",
                "Warranty and repair processes"
            ]
        )
        
        self.prompt_add_section(
            "Your Tasks",
            body="Complete support process workflow with patience and thoroughness:",
            bullets=[
                "Understand their specific technical problems with careful listening",
                "Search knowledge base for solutions methodically",
                "Guide through diagnostic steps one at a time",
                "Provide troubleshooting solutions with clear explanations",
                "Create support tickets for complex issues when needed"
            ]
        )
        
        self.prompt_add_section(
            "Voice Instructions",
            body=(
                "Maintain a calm, patient tone that reassures customers "
                "their problems will be solved."
            )
        )
        
        self.prompt_add_section(
            "Tools Available",
            body="Use these tools to resolve issues:",
            bullets=[
                "search_support_knowledge: Find technical solutions",
                "diagnose_hardware_issue: Analyze hardware problems",
                "create_support_ticket: Escalate complex issues"
            ]
        )
        
        self.prompt_add_section(
            "Important",
            body="Key guidelines for support interactions:",
            bullets=[
                "Ask detailed questions about the problem",
                "Use search to find known solutions",
                "Guide step-by-step through troubleshooting",
                "Be patient and thorough"
            ]
        )


def create_pc_builder_app(host: str = "0.0.0.0", port: int = 3001, log_level: str = "info") -> AgentServer:
    """
    Create and configure the PC Builder application with three specialized agents
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to  
        log_level: Logging level (debug, info, warning, error, critical)
    
    Returns:
        Configured AgentServer with all three agents registered
    """
    # Create the server
    server = AgentServer(host=host, port=port, log_level=log_level)
    
    # Create and register Triage Agent (root)
    triage = TriageAgent()
    server.register(triage, "/")
    
    # Create and register Sales Agent
    sales = SalesAgent()
    server.register(sales, "/sales")
    
    # Create and register Support Agent
    support = SupportAgent()
    server.register(support, "/support")
    
    # Add a root endpoint to show available agents
    @server.app.get("/info")
    async def info():
        return {
            "message": "PC Builder Pro - Multi-Agent Service",
            "agents": {
                "triage": {
                    "endpoint": "/",
                    "description": (
                        "Greets customers and routes to specialists with "
                        "automatic context collection"
                    )
                },
                "sales": {
                    "endpoint": "/sales",
                    "description": "PC building sales and recommendations specialist"
                },
                "support": {
                    "endpoint": "/support", 
                    "description": "Technical support and troubleshooting specialist"
                }
            },
            "features": {
                "context_sharing": (
                    "Uses swml_transfer skill with user_name and summary requirements"
                ),
                "pom_prompts": "Structured prompts using Prompt Object Model",
                "summary_access": (
                    "Transfer context available via ${call_data.user_name} and "
                    "${call_data.summary}"
                ),
                "multi_agent": "Three specialized agents working together"
            },
            "usage": {
                "triage_swml": f"GET/POST http://{host}:{port}/",
                "sales_swml": f"GET/POST http://{host}:{port}/sales",
                "support_swml": f"GET/POST http://{host}:{port}/support"
            }
        }
    
    return server


def lambda_handler(event, context):
    """AWS Lambda entry point - delegates to universal server run method"""
    server = create_pc_builder_app()
    return server.run(event, context)


if __name__ == "__main__":
    logger.info("Starting PC Builder Pro Multi-Agent Service")
    logger.info("=" * 60)
    logger.info("Triage Agent (Alex): http://localhost:3001/")
    logger.info("  - Enthusiastic front desk assistant with rime.spore voice")
    logger.info("  - Greets customers and routes to specialists")
    logger.info("  - Requires customer name and comprehensive summary before transfer")
    logger.info("  - Uses swml_transfer skill to hand off calls with context")
    logger.info("")
    logger.info("Sales Agent (Morgan): http://localhost:3001/sales") 
    logger.info("  - Passionate PC building expert with rime.marsh voice")
    logger.info("  - Custom PC build recommendations")
    logger.info("  - Component compatibility checking")
    logger.info("  - Pricing and performance analysis")
    logger.info("  - Accesses customer name via ${call_data.user_name}")
    logger.info("  - Accesses transfer summary via ${call_data.summary}")
    logger.info("")
    logger.info("Support Agent (Sam): http://localhost:3001/support")
    logger.info("  - Patient technical specialist with rime.cove voice")
    logger.info("  - Technical troubleshooting and diagnostics")
    logger.info("  - Hardware issue resolution")
    logger.info("  - Support ticket creation")
    logger.info("  - Accesses customer name via ${call_data.user_name}")
    logger.info("  - Accesses transfer summary via ${call_data.summary}")
    logger.info("")
    logger.info("Service Info: http://localhost:3001/info")
    logger.info("=" * 60)
    
    logger.info("Features:")
    logger.info("✔ Multi-agent architecture with automatic context sharing")
    logger.info("✔ swml_transfer skill with required user_name and summary fields")
    logger.info("✔ Native vector search for knowledge bases")
    logger.info("✔ POM-style prompts for better structure and maintainability")
    logger.info("✔ Automatic name and summary preservation across transfers")
    logger.info("✔ Specialized expertise per agent")
    logger.info("")
    logger.info("How it works:")
    logger.info("1. Triage agent uses swml_transfer skill with user_name and summary requirements")
    logger.info("2. The skill automatically prompts for customer name and comprehensive summary before transfer") 
    logger.info("3. Name and summary are automatically available to receiving agent")
    logger.info("4. Sales/Support agents access context via ${call_data.user_name} and ${call_data.summary}")
    
    # Create and run the server
    server = create_pc_builder_app()
    
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Shutting down PC Builder Pro service...") 
