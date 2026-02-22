#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Advanced Contexts and Steps Demo Agent

This agent demonstrates the complete contexts system including:
- Context entry parameters (system_prompt, consolidate, full_reset, user_prompt)
- Context prompts (structured POM format)
- Step-to-context navigation with context switching
- Multi-persona experience (Franklin, Rachael, Dwight)
"""

from signalwire_agents import AgentBase


class AdvancedContextsDemoAgent(AgentBase):
    """Advanced Computer Sales Agent demonstrating full contexts system"""
    
    def __init__(self):
        super().__init__(
            name="Advanced Computer Sales Agent",
            route="/advanced-contexts-demo"
        )
        
        # Set base prompt (required even when using contexts)

        self.prompt_add_section(
            "Instructions",
            "Follow the structured sales workflow to guide customers through their computer purchase decision. Follow the newest role as defined.8",
            bullets=[
                "Complete each step's specific criteria before advancing",
                "Ask focused questions to gather the exact information needed",
                "Be helpful and consultative, not pushy"
            ]
        )
        
        # Define contexts and steps AFTER setting base prompt
        contexts = self.define_contexts()
        
        # Context 1: Sales workflow (workflow container with context prompts)
        sales_context = contexts.add_context("sales") \
            .set_isolated(True) \
            .add_section("Role", "You are Franklin, a helpful computer sales agent. Introduce yourself by name and welcome customers to help them find the perfect computer through your systematic approach.") \
            .add_section("Voice Instructions", "Use the English-Franklin language for natural conversation flow.") \
            .add_bullets("When to use change_context tool", [
                "Customer asks for manager/supervisor/store manager - change_context to 'manager'",
                "Customer asks technical questions - change_context to 'tech_support'",
                "Customer frustrated/confused - change_context to 'manager'"
            ])
        
        # Step 1: Determine use case
        sales_context.add_step("determine_use_case") \
            .add_section("Current Task", "Identify the customer's primary computer use case") \
            .add_bullets("Required Information to Collect", [
                "What will they primarily use the computer for?",
                "Do they play video games? If so, what types and how often?",
                "Do they need it for work? What kind of work applications?",
                "Do they do creative work like video editing, design, or programming?",
                "Help them categorize as: GAMING, WORK, or BALANCED"
            ]) \
            .set_step_criteria("Customer has clearly stated their use case as one of: GAMING, WORK, or BALANCED") \
            .set_valid_steps(["determine_form_factor"]) \
            .set_valid_contexts(["tech_support", "manager"])
        
        # Step 2: Form factor preference
        sales_context.add_step("determine_form_factor") \
            .add_section("Current Task", "Determine if customer wants a laptop or desktop computer") \
            .add_bullets("Decision-Making Questions", [
                "Ask directly: 'Are you looking for a laptop or desktop computer or if they need help deciding?'",
                "If they're unsure, help them decide by asking about portability vs performance needs"
            ]) \
            .set_step_criteria("Customer has explicitly stated they want either a LAPTOP or DESKTOP") \
            .set_valid_steps(["make_recommendation"]) \
            .set_valid_contexts(["tech_support", "manager"])
        
        # Step 3: Recommendation with context switching options
        sales_context.add_step("make_recommendation") \
            .add_section("Current Task", "Provide specific computer recommendation based on gathered information") \
            .add_bullets("Recommendation Requirements", [
                "Recommend a specific computer type based on their use case and form factor",
                "Explain why this recommendation fits their needs",
                "Mention key specs that matter for their use case",
                "Provide a rough price range they should expect",
                "Ask if they want technical support or need to speak with a manager"
            ]) \
            .set_step_criteria("Customer has received a specific recommendation with explanation and pricing guidance") \
            .set_valid_contexts(["tech_support", "manager"])
        
        # Context 2: Technical Support (context switch with entry parameters)
        tech_context = contexts.add_context("tech_support") \
            .set_isolated(True)
        
        # Configure context switching behavior (when entering this context)
        
        # Add context prompt for technical support with Rachael persona
        tech_context.add_section("Role", "You are Rachael, a technical support specialist. Introduce yourself by name and offer to help with technical questions.") \
            .add_section("Voice Instructions", "Use the English-Rachael language for helpful and knowledgeable technical assistance.") \
            .add_section("Your Expertise", "Provide detailed technical assistance for computer-related questions and troubleshooting.") \
            .add_bullets("When to use change_context tool", [
                "Customer wants to continue shopping - change_context to 'sales'",
                "Customer asks for manager/store manager/pricing issues - change_context to 'manager'",
                "Technical question resolved, wants to buy - change_context to 'sales'"
            ])
        
        # Technical support step with return to sales
        tech_context.add_step("provide_support") \
            .add_section("Current Task", "Help with technical questions about computers") \
            .add_bullets("Areas of Support", [
                "Specifications and compatibility questions",
                "Setup and installation guidance", 
                "Performance optimization tips",
                "Troubleshooting common issues"
            ]) \
            .set_step_criteria("Customer's technical question has been answered satisfactorily") \
            .set_valid_contexts(["sales", "manager"])
        
        # Context 3: Manager Escalation (full context reset)
        manager_context = contexts.add_context("manager") \
            .set_isolated(True) \
            .add_enter_filler("en-US", [
                "Let me connect you with our store manager right away...",
                "I'll get the manager for you immediately...",
                "One moment while I bring in the store manager to assist you...",
                "The manager will be right with you to help resolve this..."
            ]) \
            .add_enter_filler("default", [
                "Transferring to the manager...",
                "Getting the manager for you..."
            ])
        
        # Configure context entry
        
        # Add manager context prompt with introduction
        manager_context.add_section("Role", "You are Dwight, the store manager. Introduce yourself by name and acknowledge that the customer needs additional assistance with their computer purchase.") \
            .add_section("Voice Instructions", "Use the English-Dwight language for authoritative but friendly communication.") \
            .add_section("Authority Level", "You are a senior store manager with decision-making power") \
            .add_bullets("Your Capabilities", [
                "Authorize special pricing and discounts",
                "Handle complex customer requirements", 
                "Resolve escalated issues",
                "Make final purchasing decisions"
            ]) \
            .add_section("Approach", "Express commitment to finding the right solution and ask how you can help resolve their concerns.") \
            .add_bullets("When to use change_context tool", [
                "Customer needs technical help - change_context to 'tech_support'",
                "Issue resolved, wants to continue shopping - change_context to 'sales'",
                "Customer wants to talk to sales team - change_context to 'sales'"
            ])

        manager_context.add_step("handle_escalation") \
            .add_section("Current Task", "Understand the customer's specific concerns and work to resolve their issues") \
            .add_bullets("Manager Approach", [
                "Ask what specific concerns they have about their computer purchase",
                "Listen carefully to their issues",
                "Use your authority to provide solutions",
                "Offer appropriate assistance based on their needs"
            ]) \
            .set_step_criteria("Customer issue has been resolved by manager") \
            .set_valid_contexts(["sales", "tech_support"])
        
        # Add language support
        self.add_language(
            name="English-Franklin",
            code="en-US",
            voice="inworld.Mark"
        )

        self.add_language(
            name="English-Dwight",
            code="en-US",
            voice="inworld.Blake"
        )

        self.add_language(
            name="English-Rachael",
            code="en-US",
            voice="inworld.Sarah"
        )
        
        # Add internal fillers for context switching
        self.add_internal_filler("next_step", "en-US", [
            "Great! Let's move to the next step...",
            "Perfect! Moving forward in our conversation...",
            "Excellent! Let's continue to the next part...",
            "Wonderful! Progressing to the next step..."
        ])
        
        self.add_internal_filler("change_context", "en-US", [
            "Let me connect you with the right department...",
            "I'm transferring you to a specialist...",
            "One moment while I get the right person...",
            "Let me get someone who can help with that..."
        ])


def main():
    """Main function to run the advanced contexts demo agent"""
    print("=" * 80)
    print("ADVANCED CONTEXTS AND STEPS DEMO")
    print("=" * 80)
    print()
    print("This agent demonstrates the complete contexts system including:")
    print()
    print("Context Features:")
    print("  • Context entry parameters (system_prompt, consolidate, full_reset, user_prompt)")
    print("  • Context prompts (structured POM format)")
    print("  • Post prompt overrides per context")
    print("  • Enter fillers for smooth context transitions (manager context demo)")
    print()
    print("Step Features:")
    print("  • Step-to-step navigation (set_valid_steps)")
    print("  • Step-to-context navigation (set_valid_contexts)")
    print("  • Direct context switching instructions")
    print()
    print("Navigation Flow:")
    print("  1. Sales workflow (determine use case → form factor → recommendation)")
    print("  2. Optional tech support (context switch)")
    print("  3. Optional manager escalation (full reset)")
    print("  4. Return to sales from any department")
    print()
    print("Personas:")
    print("  • Franklin (Sales) - English-Franklin with inworld.Mark voice")
    print("  • Rachael (Tech Support) - English-Rachael with inworld.Sarah voice")
    print("  • Dwight (Manager) - English-Dwight with inworld.Blake voice")
    print()
    print("Test the agent at: http://localhost:3000/advanced-contexts-demo")
    print()
    
    agent = AdvancedContextsDemoAgent()
    
    print("Agent configuration:")
    print(f"  Name: {agent.get_name()}")
    print(f"  Route: /advanced-contexts-demo")
    print(f"  Contexts: 3 (sales, tech_support, manager)")
    print(f"  Features: Context entry params + Context switching")
    print()
    
    try:
        agent.run()
    except KeyboardInterrupt:
        print("\nShutting down agent...")


if __name__ == "__main__":
    main() 
