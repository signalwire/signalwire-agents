#!/usr/bin/env python3
"""
Example demonstrating LLM parameter customization for different agent personalities.

This example shows how to use set_prompt_llm_params() and set_post_prompt_llm_params()
to create agents with different response characteristics.
"""

from signalwire_agents import AgentBase
from signalwire_agents.core.function_result import SwaigFunctionResult
import random


class PreciseAssistant(AgentBase):
    """An agent configured for precise, consistent responses"""
    
    def __init__(self):
        super().__init__(name="precise-assistant", route="/precise")
        
        # Configure personality
        self.prompt_add_section("Role", "You are a precise technical assistant.")
        self.prompt_add_section("Instructions", bullets=[
            "Provide accurate, factual information",
            "Be concise and direct",
            "Avoid speculation or guessing",
            "If uncertain, say so clearly"
        ])
        
        # Configure LLM for consistency and precision
        self.set_prompt_llm_params(
            temperature=0.2,        # Very low for consistent responses
            top_p=0.85,            # More focused token selection
            barge_confidence=0.8,  # Hard to interrupt - let it finish technical info
            presence_penalty=0.0,  # No penalty - technical terms may repeat
            frequency_penalty=0.1  # Slight penalty for word variety
        )
        
        # Post-prompt for technical summaries
        self.set_post_prompt("Provide a brief technical summary of the key points discussed.")
        self.set_post_prompt_llm_params(
            temperature=0.1       # Extremely consistent summaries
        )
    
    @AgentBase.tool(
        name="get_system_info",
        description="Get technical system information",
        parameters={}
    )
    def get_system_info(self, args, raw_data):
        """Simulated system information"""
        info = {
            "cpu_usage": f"{random.randint(10, 90)}%",
            "memory_available": f"{random.randint(1, 16)}GB",
            "disk_space": f"{random.randint(50, 500)}GB free",
            "uptime": f"{random.randint(1, 30)} days"
        }
        return SwaigFunctionResult(
            f"System Status: CPU {info['cpu_usage']}, "
            f"Memory {info['memory_available']}, "
            f"Disk {info['disk_space']}, "
            f"Uptime {info['uptime']}"
        )


class CreativeAssistant(AgentBase):
    """An agent configured for creative, varied responses"""
    
    def __init__(self):
        super().__init__(name="creative-assistant", route="/creative")
        
        # Configure personality
        self.prompt_add_section("Role", "You are a creative writing assistant.")
        self.prompt_add_section("Instructions", bullets=[
            "Be imaginative and creative",
            "Use varied vocabulary and expressions",
            "Encourage creative thinking",
            "Suggest unique perspectives"
        ])
        
        # Configure LLM for creativity
        self.set_prompt_llm_params(
            temperature=0.8,        # High for creative variety
            top_p=0.95,            # Wide token selection
            barge_confidence=0.5,  # Easy to interrupt for collaboration
            presence_penalty=0.2,  # Encourage new topics
            frequency_penalty=0.3  # Strong vocabulary variety
        )
        
        # Post-prompt for creative summaries
        self.set_post_prompt("Create an artistic summary of our conversation.")
        self.set_post_prompt_llm_params(
            temperature=0.7       # Still creative for summaries
        )
    
    @AgentBase.tool(
        name="generate_story_prompt",
        description="Generate a creative story prompt",
        parameters={"theme": {"type": "string", "description": "Story theme"}}
    )
    def generate_story_prompt(self, args, raw_data):
        """Generate creative story prompts"""
        theme = args.get("theme", "adventure")
        prompts = {
            "adventure": [
                "A map that only appears during thunderstorms",
                "A compass that points to what you need most",
                "A backpack that's bigger on the inside"
            ],
            "mystery": [
                "A photograph where people keep disappearing",
                "A library book that writes itself",
                "A mirror that shows different reflections"
            ],
            "default": [
                "An ordinary object with extraordinary powers",
                "A door that leads somewhere different each time",
                "A message from your future self"
            ]
        }
        
        prompt_list = prompts.get(theme.lower(), prompts["default"])
        chosen_prompt = random.choice(prompt_list)
        
        return SwaigFunctionResult(
            f"Story prompt for {theme}: {chosen_prompt}"
        )


class CustomerServiceAgent(AgentBase):
    """An agent configured for professional customer service"""
    
    def __init__(self):
        super().__init__(name="customer-service", route="/support")
        
        # Configure personality
        self.prompt_add_section("Role", "You are a professional customer service representative.")
        self.prompt_add_section("Guidelines", bullets=[
            "Always be polite and empathetic",
            "Listen carefully to customer concerns",
            "Provide clear, helpful solutions",
            "Follow company policies"
        ])
        
        # Configure LLM for professional consistency
        self.set_prompt_llm_params(
            temperature=0.4,        # Balanced consistency
            top_p=0.9,             # Standard token selection
            barge_confidence=0.7,  # Moderate interruption threshold
            presence_penalty=0.1,  # Slight penalty for repetition
            frequency_penalty=0.1  # Encourage natural variety
        )
        
        # Post-prompt for ticket summaries
        self.set_post_prompt("Summarize the customer's issue and resolution for the ticket system.")
        self.set_post_prompt_llm_params(
            temperature=0.3       # Consistent ticket summaries
        )
    
    @AgentBase.tool(
        name="check_order_status",
        description="Check the status of a customer order",
        parameters={"order_id": {"type": "string", "description": "Order ID"}}
    )
    def check_order_status(self, args, raw_data):
        """Simulated order status check"""
        order_id = args.get("order_id", "unknown")
        
        # Simulate order statuses
        statuses = [
            "Processing - Expected to ship within 24 hours",
            "Shipped - Tracking number: TRK" + str(random.randint(100000, 999999)),
            "Out for delivery - Expected today by 6 PM",
            "Delivered - Left at front door"
        ]
        
        status = random.choice(statuses)
        return SwaigFunctionResult(f"Order {order_id} status: {status}")


def main():
    """Run the demo agent based on command line argument"""
    import sys
    
    if len(sys.argv) > 1:
        agent_type = sys.argv[1].lower()
        
        if agent_type == "precise":
            agent = PreciseAssistant()
            print("Starting Precise Assistant (low temperature, hard to interrupt)...")
        elif agent_type == "creative":
            agent = CreativeAssistant()
            print("Starting Creative Assistant (high temperature, easy to interrupt)...")
        elif agent_type == "support":
            agent = CustomerServiceAgent()
            print("Starting Customer Service Agent (balanced parameters)...")
        else:
            print(f"Unknown agent type: {agent_type}")
            print("Usage: python llm_params_demo.py [precise|creative|support]")
            sys.exit(1)
    else:
        # Default to customer service
        agent = CustomerServiceAgent()
        print("Starting Customer Service Agent (default)...")
        print("Try: python llm_params_demo.py [precise|creative|support]")
    
    agent.serve(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()