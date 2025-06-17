# Contexts and Steps Guide

## Table of Contents

- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Navigation and Flow Control](#navigation-and-flow-control)
- [Function Restrictions](#function-restrictions)
- [Real-World Examples](#real-world-examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Migration from POM](#migration-from-pom)

## Overview

The **Contexts and Steps** system enhances traditional Prompt Object Model (POM) prompts in SignalWire AI agents by adding structured workflows on top of your base prompt. Instead of just defining a single prompt, you create workflows with explicit steps, navigation rules, and completion criteria.

### Key Benefits

- **Structured Workflows**: Define clear, step-by-step processes
- **Explicit Navigation**: Control exactly where users can go next
- **Function Restrictions**: Limit AI tool access per step
- **Completion Criteria**: Define clear progression requirements
- **Context Isolation**: Separate different conversation flows
- **Debugging**: Easier to trace and debug complex interactions

### When to Use Contexts vs Traditional Prompts

**Use Contexts and Steps when:**
- Building multi-step workflows (onboarding, support tickets, applications)
- Need explicit navigation control between conversation states
- Want to restrict function access based on conversation stage
- Building complex customer service or troubleshooting flows
- Creating guided experiences with clear progression

**Use Traditional Prompts when:**
- Building simple, freeform conversational agents
- Want maximum flexibility in conversation flow
- Creating general-purpose assistants
- Prototyping or building simple proof-of-concepts

## Core Concepts

### Contexts

A **Context** represents a conversation state or workflow area. Contexts can be:

- **Workflow Container**: Simple step organization without state changes
- **Context Switch**: Triggers conversation state changes when entered

Each context can define:

- **Steps**: Individual workflow stages within the context
- **Context Prompts**: Guidance that applies to all steps in the context  
- **Entry Parameters**: Control conversation state when context is entered
- **Navigation Rules**: Which other contexts can be accessed

### Context Entry Parameters

When entering a context, these parameters control conversation behavior:

- **`post_prompt`**: Override the agent's post prompt for this context
- **`system_prompt`**: Trigger conversation reset with new instructions
- **`consolidate`**: Summarize previous conversation in new prompt
- **`full_reset`**: Complete system prompt replacement vs injection
- **`user_prompt`**: Inject user message for context establishment

**Important**: If `system_prompt` is present, the context becomes a "Context Switch Context" that processes entry parameters like a `context_switch` SWAIG action. Without `system_prompt`, it's a "Workflow Container Context" that only organizes steps.

### Context Prompts

Contexts can have their own prompts (separate from entry parameters):

```python
# Simple string prompt
context.set_prompt("Context-specific guidance")

# POM-style sections  
context.add_section("Department", "Billing Department")
context.add_bullets("Services", ["Payments", "Refunds", "Account inquiries"])
```

Context prompts provide guidance that applies to all steps within that context, creating a prompt hierarchy: Base Agent Prompt → Context Prompt → Step Prompt.

### Steps

A **Step** is a specific stage within a context. Each step defines:

- **Prompt Content**: What the AI says/does (text or POM sections)
- **Completion Criteria**: When the step is considered complete
- **Navigation Rules**: Where the user can go next
- **Function Access**: Which AI tools are available

### Navigation Control

The system provides fine-grained control over conversation flow:

- **Valid Steps**: Control movement within a context
- **Valid Contexts**: Control switching between contexts  
- **Implicit Navigation**: Automatic "next" step progression
- **Explicit Navigation**: User must explicitly choose next step

## Getting Started

### Basic Single-Context Workflow

```python
from signalwire_agents import AgentBase

class OnboardingAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Onboarding Assistant", route="/onboarding")
        
        # Define contexts (replaces traditional prompt setup)
        contexts = self.define_contexts()
        
        # Single context must be named "default"
        workflow = contexts.add_context("default")
        
        # Step 1: Welcome
        workflow.add_step("welcome") \
            .set_text("Welcome to our service! Let's get you set up. What's your name?") \
            .set_step_criteria("User has provided their name") \
            .set_valid_steps(["collect_email"])
        
        # Step 2: Collect Email
        workflow.add_step("collect_email") \
            .set_text("Thanks! Now I need your email address to create your account.") \
            .set_step_criteria("Valid email address has been provided") \
            .set_valid_steps(["confirm_details"])
        
        # Step 3: Confirmation
        workflow.add_step("confirm_details") \
            .set_text("Perfect! Let me confirm your details before we proceed.") \
            .set_step_criteria("User has confirmed their information") \
            .set_valid_steps(["complete"])
        
        # Step 4: Completion
        workflow.add_step("complete") \
            .set_text("All set! Your account has been created successfully.")
            # No valid_steps = end of workflow

agent = OnboardingAgent()
agent.run()

if __name__ == "__main__":
    main()
```

### Multi-Context Workflow

```python
class CustomerServiceAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Customer Service", route="/service")
        
        # Add skills for enhanced capabilities
        self.add_skill("datetime")
        self.add_skill("web_search", {
            "api_key": "your-api-key",
            "search_engine_id": "your-engine-id"
        })
        
        contexts = self.define_contexts()
        
        # Main triage context
        triage = contexts.add_context("triage")
        triage.add_step("greeting") \
            .add_section("Current Task", "Understand the customer's need and route appropriately") \
            .add_bullets("Required Information", [
                "Type of issue they're experiencing",
                "Urgency level of the problem", 
                "Previous troubleshooting attempts"
            ]) \
            .set_step_criteria("Customer's need has been identified") \
            .set_valid_contexts(["technical", "billing", "general"])
        
        # Technical support context
        tech = contexts.add_context("technical")
        tech.add_step("technical_help") \
            .add_section("Current Task", "Help diagnose and resolve technical issues") \
            .add_section("Available Tools", "Use web search and datetime functions for technical solutions") \
            .set_functions(["web_search", "datetime"]) \
            .set_step_criteria("Issue is resolved or escalated") \
            .set_valid_contexts(["triage"])
        
        # Billing context (restricted functions for security)
        billing = contexts.add_context("billing")
        billing.add_step("billing_help") \
            .set_text("I'll help with your billing question. For security, please provide your account verification.") \
            .set_functions("none") \
            .set_step_criteria("Billing issue is addressed") \
            .set_valid_contexts(["triage"])
        
        # General inquiries context
        general = contexts.add_context("general")
        general.add_step("general_help") \
            .set_text("I'm here to help with general questions. What can I assist you with?") \
            .set_functions(["web_search", "datetime"]) \
            .set_step_criteria("Question has been answered") \
            .set_valid_contexts(["triage"])

agent = CustomerServiceAgent()
agent.run()

if __name__ == "__main__":
    main()
```

## API Reference

### ContextBuilder

The main entry point for defining contexts and steps.

```python
# Get the builder
contexts = self.define_contexts()

# Create contexts
context = contexts.add_context(name: str) -> Context
```

### Context

Represents a conversation context or workflow state.

```python
class Context:
    def add_step(self, name: str) -> Step
        """Create a new step in this context"""
    
    def set_valid_contexts(self, contexts: List[str]) -> Context
        """Set which contexts can be accessed from this context"""
        
    # Context entry parameters
    def set_post_prompt(self, post_prompt: str) -> Context
        """Override post prompt for this context"""
    
    def set_system_prompt(self, system_prompt: str) -> Context
        """Trigger context switch with new system prompt"""
        
    def set_consolidate(self, consolidate: bool) -> Context
        """Consolidate conversation history when entering"""
        
    def set_full_reset(self, full_reset: bool) -> Context
        """Full system prompt replacement vs injection"""
        
    def set_user_prompt(self, user_prompt: str) -> Context
        """Inject user message for context"""
    
    # Context prompts
    def set_prompt(self, prompt: str) -> Context
        """Set simple string prompt for context"""
        
    def add_section(self, title: str, body: str) -> Context
        """Add POM section to context prompt"""
        
    def add_bullets(self, title: str, bullets: List[str]) -> Context
        """Add POM bullet section to context prompt"""
```

#### Methods

- `add_step(name)`: Create and return a new Step
- `set_valid_contexts(contexts)`: Allow navigation to specified contexts
- `set_post_prompt(prompt)`: Override agent's post prompt for this context
- `set_system_prompt(prompt)`: Trigger context switch behavior (makes this a Context Switch Context)
- `set_consolidate(bool)`: Whether to consolidate conversation when entering
- `set_full_reset(bool)`: Complete vs partial context reset
- `set_user_prompt(prompt)`: User message to inject when entering context
- `set_prompt(text)`: Simple string prompt for context
- `add_section(title, body)`: Add POM section to context prompt
- `add_bullets(title, list)`: Add POM bullet section to context prompt

### Step

Represents a single step within a context workflow.

```python
class Step:
    # Content definition (choose one approach)
    def set_text(self, text: str) -> Step
        """Set direct text prompt (mutually exclusive with POM sections)"""
    
    def add_section(self, title: str, body: str = "") -> Step
        """Add a POM-style section (mutually exclusive with set_text)"""
    
    def add_bullets(self, bullets: List[str], numbered: bool = False) -> Step
        """Add bullets to the current or most recent section"""
    
    # Flow control
    def set_step_criteria(self, criteria: str) -> Step
        """Define completion criteria for this step"""
    
    def set_valid_steps(self, steps: List[str]) -> Step
        """Set which steps can be accessed next in same context"""
    
    def set_valid_contexts(self, contexts: List[str]) -> Step
        """Set which contexts can be accessed from this step"""
    
    # Function restrictions
    def set_functions(self, functions: Union[List[str], str]) -> Step
        """Restrict available functions ('none' or list of function names)"""
```

#### Content Methods

**Option 1: Direct Text**
```python
step.set_text("Direct prompt text for the AI")
```

**Option 2: POM-Style Sections**
```python
step.add_section("Role", "You are a helpful assistant") \
    .add_section("Instructions", "Help users with their questions") \
    .add_bullets(["Be friendly", "Ask clarifying questions"])
```

**Note**: You cannot mix `set_text()` with `add_section()` in the same step.

#### Navigation Methods

```python
# Control step progression within context
step.set_valid_steps(["step1", "step2"])  # Can go to step1 or step2
step.set_valid_steps([])                   # Cannot progress (dead end)
# No set_valid_steps() call = implicit "next" step

# Control context switching
step.set_valid_contexts(["context1", "context2"])  # Can switch contexts
step.set_valid_contexts([])                         # Trapped in current context
# No set_valid_contexts() call = inherit from context level
```

#### Function Restriction Methods

```python
# Allow specific functions only
step.set_functions(["datetime", "math"])

# Block all functions
step.set_functions("none")

# No restriction (default - all agent functions available)
# step.set_functions()  # Don't call this method
```

## Navigation and Flow Control

### Step Navigation Rules

The `set_valid_steps()` method controls movement within a context:

```python
# Explicit step list - can only go to these steps
step.set_valid_steps(["review", "edit", "cancel"])

# Empty list - dead end, cannot progress
step.set_valid_steps([])

# Not called - implicit "next" step progression
# (will go to the next step defined in the context)
```

### Context Navigation Rules

The `set_valid_contexts()` method controls switching between contexts:

```python
# Can switch to these contexts
step.set_valid_contexts(["billing", "technical", "general"])

# Trapped in current context
step.set_valid_contexts([])

# Not called - inherit from context-level settings
```

### Navigation Inheritance

Context-level navigation settings are inherited by steps:

```python
# Set at context level
context.set_valid_contexts(["main", "help"])

# All steps in this context can access main and help contexts
# unless overridden at step level
step.set_valid_contexts(["main"])  # Override - only main allowed
```

### Complete Navigation Example

```python
contexts = self.define_contexts()

# Main context
main = contexts.add_context("main")
main.set_valid_contexts(["help", "settings"])  # Context-level setting

main.add_step("welcome") \
    .set_text("Welcome! How can I help you?") \
    .set_valid_steps(["menu"])  # Must go to menu
    # Inherits context-level valid_contexts

main.add_step("menu") \
    .set_text("Choose an option: 1) Help 2) Settings 3) Continue") \
    .set_valid_contexts(["help", "settings", "main"])  # Override context setting
    # No valid_steps = this is a branching point

# Help context  
help_ctx = contexts.add_context("help")
help_ctx.add_step("help_info") \
    .set_text("Here's how to use the system...") \
    .set_valid_contexts(["main"])  # Can return to main

# Settings context
settings = contexts.add_context("settings")
settings.add_step("settings_menu") \
    .set_text("Choose a setting to modify...") \
    .set_valid_contexts(["main"])  # Can return to main
```

## Function Restrictions

Control which AI tools/functions are available in each step for enhanced security and user experience.

### Function Restriction Levels

```python
# No restrictions (default) - all agent functions available
step  # Don't call set_functions()

# Allow specific functions only
step.set_functions(["datetime", "math", "web_search"])

# Block all functions
step.set_functions("none")
```

### Security-Focused Example

```python
class SecureBankingAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Banking Assistant", route="/banking")
        
        # Add potentially sensitive functions
        self.add_skill("web_search", {"api_key": "key", "search_engine_id": "id"})
        self.add_skill("datetime")
        
        contexts = self.define_contexts()
        
        # Public context - full access
        public = contexts.add_context("public")
        public.add_step("welcome") \
            .set_text("Welcome to banking support. Are you an existing customer?") \
            .set_functions(["datetime", "web_search"])  # Safe functions only \
            .set_valid_contexts(["authenticated", "public"])
        
        # Authenticated context - restricted for security
        auth = contexts.add_context("authenticated")
        auth.add_step("account_access") \
            .set_text("I can help with your account. What do you need assistance with?") \
            .set_functions("none")  # No external functions for account data \
            .set_valid_contexts(["public"])  # Can log out
```

### Function Access Patterns

```python
# Progressive function access based on trust level
contexts = self.define_contexts()

# Low trust - limited functions
public = contexts.add_context("public")
public.add_step("initial_contact") \
    .set_functions(["datetime"])  # Only safe functions

# Medium trust - more functions  
verified = contexts.add_context("verified")
verified.add_step("verified_user") \
    .set_functions(["datetime", "web_search"])  # Add search capability

# High trust - full access
authenticated = contexts.add_context("authenticated")
authenticated.add_step("full_access") \
    # No set_functions() call = all functions available
```

## Real-World Examples

### Example 1: Technical Support Troubleshooting

```python
class TechnicalSupportAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Tech Support", route="/tech-support")
        
        # Add diagnostic tools
        self.add_skill("web_search", {"api_key": "key", "search_engine_id": "id"})
        self.add_skill("datetime")
        
        contexts = self.define_contexts()
        
        # Initial triage
        triage = contexts.add_context("triage")
        triage.add_step("problem_identification") \
            .add_section("Current Task", "Identify the type of technical issue") \
            .add_bullets("Information to Gather", [
                "Description of the specific problem",
                "When did the issue start occurring?",
                "What steps has the customer already tried?",
                "Rate the severity level (critical/high/medium/low)"
            ]) \
            .set_step_criteria("Issue type and severity determined") \
            .set_valid_contexts(["hardware", "software", "network"])
        
        # Hardware troubleshooting
        hardware = contexts.add_context("hardware")
        hardware.add_step("hardware_diagnosis") \
            .add_section("Current Task", "Guide user through hardware diagnostics") \
            .add_section("Available Tools", "Use web search to find hardware specifications and troubleshooting guides") \
            .set_functions(["web_search"])  # Can search for hardware info \
            .set_step_criteria("Hardware issue diagnosed") \
            .set_valid_steps(["hardware_solution"])
        
        hardware.add_step("hardware_solution") \
            .set_text("Based on the diagnosis, here's how to resolve the hardware issue...") \
            .set_step_criteria("Solution provided and tested") \
            .set_valid_contexts(["triage"])  # Can start over if needed
        
        # Software troubleshooting
        software = contexts.add_context("software")
        software.add_step("software_diagnosis") \
            .add_section("Current Task", "Diagnose software-related issues") \
            .add_section("Available Tools", "Use web search for software updates and datetime to check for recent changes") \
            .set_functions(["web_search", "datetime"])  # Can check for updates \
            .set_step_criteria("Software issue identified") \
            .set_valid_steps(["software_fix", "escalation"])
        
        software.add_step("software_fix") \
            .set_text("Let's try these software troubleshooting steps...") \
            .set_step_criteria("Fix attempted and result confirmed") \
            .set_valid_steps(["escalation", "resolution"])
        
        software.add_step("escalation") \
            .set_text("I'll escalate this to our specialist team.") \
            .set_functions("none")  # No tools needed for escalation \
            .set_step_criteria("Escalation ticket created")
        
        software.add_step("resolution") \
            .set_text("Great! The issue has been resolved.") \
            .set_step_criteria("Customer confirms resolution") \
            .set_valid_contexts(["triage"])
        
        # Network troubleshooting
        network = contexts.add_context("network")
        network.add_step("network_diagnosis") \
            .add_section("Current Task", "Diagnose network and connectivity issues") \
            .add_section("Available Tools", "Use web search to check service status and datetime for outage windows") \
            .set_functions(["web_search", "datetime"])  # Check service status \
            .set_step_criteria("Network issue diagnosed") \
            .set_valid_steps(["network_fix"])
        
        network.add_step("network_fix") \
            .set_text("Let's resolve your connectivity issue with these steps...") \
            .set_step_criteria("Network connectivity restored") \
            .set_valid_contexts(["triage"])

agent = TechnicalSupportAgent()
agent.run()

if __name__ == "__main__":
    main()
```

### Example 2: Multi-Step Application Process

```python
class LoanApplicationAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Loan Application", route="/loan-app")
        
        # Add verification tools
        self.add_skill("datetime")  # For date validation
        
        contexts = self.define_contexts()
        
        # Single workflow context
        application = contexts.add_context("default")
        
        # Step 1: Introduction and eligibility
        application.add_step("introduction") \
            .add_section("Current Task", "Guide customers through the loan application process") \
            .add_bullets("Information to Provide", [
                "Explain the process clearly",
                "Outline what information will be needed",
                "Set expectations for timeline and next steps"
            ]) \
            .set_step_criteria("Customer understands process and wants to continue") \
            .set_valid_steps(["personal_info"])
        
        # Step 2: Personal information
        application.add_step("personal_info") \
            .add_section("Instructions", "Collect personal information") \
            .add_bullets([
                "Full legal name",
                "Date of birth",
                "Social Security Number",
                "Phone number and email"
            ]) \
            .set_functions(["datetime"])  # Can validate dates \
            .set_step_criteria("All personal information collected and verified") \
            .set_valid_steps(["employment_info", "personal_info"])  # Can review/edit
        
        # Step 3: Employment information  
        application.add_step("employment_info") \
            .set_text("Now I need information about your employment and income.") \
            .set_step_criteria("Employment and income information complete") \
            .set_valid_steps(["financial_info", "personal_info"])  # Can go back
        
        # Step 4: Financial information
        application.add_step("financial_info") \
            .set_text("Let's review your financial situation including assets and debts.") \
            .set_step_criteria("Financial information complete") \
            .set_valid_steps(["review", "employment_info"])  # Can go back
        
        # Step 5: Review all information
        application.add_step("review") \
            .add_section("Instructions", "Review all collected information") \
            .add_bullets([
                "Confirm personal details",
                "Verify employment information", 
                "Review financial data",
                "Ensure accuracy before submission"
            ]) \
            .set_step_criteria("Customer has reviewed and confirmed all information") \
            .set_valid_steps(["submit", "personal_info", "employment_info", "financial_info"])
        
        # Step 6: Submission
        application.add_step("submit") \
            .set_text("Thank you! Your loan application has been submitted successfully. You'll receive a decision within 2-3 business days.") \
            .set_functions("none")  # No tools needed for final message \
            .set_step_criteria("Application submitted and confirmation provided")
            # No valid_steps = end of process

agent = LoanApplicationAgent()
agent.run()

if __name__ == "__main__":
    main()
```

### Example 3: E-commerce Customer Service

```python
class EcommerceServiceAgent(AgentBase):
    def __init__(self):
        super().__init__(name="E-commerce Support", route="/ecommerce")
        
        # Add tools for order management
        self.add_skill("web_search", {"api_key": "key", "search_engine_id": "id"})
        self.add_skill("datetime")
        
        contexts = self.define_contexts()
        
        # Main service menu
        main = contexts.add_context("main")
        main.add_step("service_menu") \
            .add_section("Current Task", "Help customers with their orders and questions") \
            .add_bullets("Service Areas Available", [
                "Order status, modifications, and tracking",
                "Returns and refunds",
                "Product information and specifications",
                "Account-related questions"
            ]) \
            .set_step_criteria("Customer's need has been identified") \
            .set_valid_contexts(["orders", "returns", "products", "account"])
        
        # Order management context
        orders = contexts.add_context("orders")
        orders.add_step("order_assistance") \
            .add_section("Current Task", "Help with order status, modifications, and tracking") \
            .add_section("Available Tools", "Use datetime to check delivery dates and processing times") \
            .set_functions(["datetime"])  # Can check delivery dates \
            .set_step_criteria("Order issue resolved or escalated") \
            .set_valid_contexts(["main"])
        
        # Returns and refunds context
        returns = contexts.add_context("returns")
        returns.add_step("return_process") \
            .add_section("Current Task", "Guide customers through return process") \
            .add_bullets("Return Process Steps", [
                "Verify return eligibility",
                "Explain return policy", 
                "Provide return instructions",
                "Process refund if applicable"
            ]) \
            .set_functions("none")  # Sensitive financial operations \
            .set_step_criteria("Return request processed") \
            .set_valid_contexts(["main"])
        
        # Product information context
        products = contexts.add_context("products")
        products.add_step("product_help") \
            .add_section("Current Task", "Help customers with product questions") \
            .add_section("Available Tools", "Use web search to find detailed product information and specifications") \
            .set_functions(["web_search"])  # Can search for product info \
            .set_step_criteria("Product question answered") \
            .set_valid_contexts(["main"])
        
        # Account management context
        account = contexts.add_context("account")
        account.add_step("account_help") \
            .set_text("I can help with account-related questions. Please verify your identity first.") \
            .set_functions("none")  # Security-sensitive context \
            .set_step_criteria("Account issue resolved") \
            .set_valid_contexts(["main"])

agent = EcommerceServiceAgent()
agent.run()

if __name__ == "__main__":
    main()
```

## Best Practices

### 1. Clear Step Naming

Use descriptive step names that indicate purpose:

```python
# Good
.add_step("collect_shipping_address")
.add_step("verify_payment_method")
.add_step("confirm_order_details")

# Avoid
.add_step("step1")
.add_step("next")
.add_step("continue")
```

### 2. Meaningful Completion Criteria

Define clear, testable completion criteria:

```python
# Good - specific and measurable
.set_step_criteria("User has provided valid email address and confirmed subscription preferences")
.set_step_criteria("All required fields completed and payment method verified")

# Avoid - vague or subjective
.set_step_criteria("User is ready")
.set_step_criteria("Everything is good")
```

### 3. Logical Navigation Flow

Design intuitive navigation that matches user expectations:

```python
# Allow users to go back and review
.set_valid_steps(["review_info", "edit_details", "confirm_submission"])

# Provide escape routes
.set_valid_contexts(["main_menu", "help"])

# Consider dead ends carefully
.set_valid_steps([])  # Only if this is truly the end
```

### 4. Progressive Function Access

Restrict functions based on security and context needs:

```python
# Public areas - limited functions
public_step.set_functions(["datetime", "web_search"])

# Authenticated areas - more functions allowed
auth_step.set_functions(["datetime", "web_search", "user_profile"])

# Sensitive operations - minimal functions
billing_step.set_functions("none")
```

### 5. Context Organization

Organize contexts by functional area or user journey:

```python
# By functional area
contexts = ["triage", "technical_support", "billing", "account_management"]

# By user journey stage  
contexts = ["onboarding", "verification", "configuration", "completion"]

# By security level
contexts = ["public", "authenticated", "admin"]
```

### 6. Error Handling and Recovery

Provide recovery paths for common issues:

```python
# Allow users to retry failed steps
.set_valid_steps(["retry_payment", "choose_different_method", "contact_support"])

# Provide help context access
.set_valid_contexts(["help", "main"])

# Include validation steps
verification_step.add_step("validation") \
    .set_step_criteria("Data validation passed") \
    .set_valid_steps(["proceed", "edit_data"])
```

### 7. Content Strategy

Choose the right content approach for each step:

```python
# Use set_text() for simple, direct instructions
step.set_text("Please provide your email address")

# Use POM sections for complex, structured content
step.add_section("Role", "You are a technical specialist") \
    .add_section("Context", "Customer is experiencing network issues") \
    .add_section("Instructions", "Follow diagnostic protocol") \
    .add_bullets(["Check connectivity", "Test speed", "Verify settings"])
```

## Troubleshooting

### Common Issues

#### 1. "Single context must be named 'default'"

**Error**: When using a single context with a name other than "default"

```python
# Wrong
context = contexts.add_context("main")  # Error!

# Correct
context = contexts.add_context("default")
```

#### 2. "Cannot mix set_text with add_section"

**Error**: Using both direct text and POM sections in the same step

```python
# Wrong
step.set_text("Welcome!") \
    .add_section("Role", "Assistant")  # Error!

# Correct - choose one approach
step.set_text("Welcome! I'm your assistant.")
# OR
step.add_section("Role", "Assistant") \
    .add_section("Message", "Welcome!")
```

#### 3. Navigation Issues

**Problem**: Users getting stuck or unable to navigate

```python
# Check your navigation rules
step.set_valid_steps([])  # Dead end - is this intended?
step.set_valid_contexts([])  # Trapped in context - is this intended?

# Add appropriate navigation
step.set_valid_steps(["next_step", "previous_step"])
step.set_valid_contexts(["main", "help"])
```

#### 4. Function Access Problems

**Problem**: Functions not available when expected

```python
# Check function restrictions
step.set_functions("none")  # All functions blocked
step.set_functions(["datetime"])  # Only datetime allowed

# Verify function names match your agent's functions
self.add_skill("web_search")  # Function name is "web_search"
step.set_functions(["web_search"])  # Must match exactly
```

### Debugging Tips

#### 1. Trace Navigation Flow

Add logging to understand flow:

```python
def create_step_with_logging(self, name):
    step = context.add_step(name)
    print(f"Created step: {name}")
    return step
```

#### 2. Validate Navigation Rules

Check that all referenced steps/contexts exist:

```python
# Ensure referenced steps exist
.set_valid_steps(["review", "edit"])  # Both "review" and "edit" steps must exist

# Ensure referenced contexts exist  
.set_valid_contexts(["main", "help"])  # Both "main" and "help" contexts must exist
```

#### 3. Test Function Restrictions

Verify functions are properly restricted:

```python
# Test with all functions
# step  # No set_functions() call

# Test with restrictions
step.set_functions(["datetime"])

# Test with no functions
step.set_functions("none")
```

## Migration from POM

### Converting Traditional Prompts

**Before (Traditional POM):**
```python
class TraditionalAgent(AgentBase):
    def __init__(self):
        super().__init__(name="assistant", route="/assistant")
        
        self.prompt_add_section("Role", "You are a helpful assistant")
        self.prompt_add_section("Instructions", "Help users with questions")
        self.prompt_add_section("Guidelines", bullets=[
            "Be friendly",
            "Ask clarifying questions",
            "Provide accurate information"
        ])
```

**After (Contexts and Steps):**
```python
class ContextsAgent(AgentBase):
    def __init__(self):
        super().__init__(name="assistant", route="/assistant")
        
        contexts = self.define_contexts()
        main = contexts.add_context("default")
        
        main.add_step("assistance") \
            .add_section("Role", "You are a helpful assistant") \
            .add_section("Instructions", "Help users with questions") \
            .add_section("Guidelines", bullets=[
                "Be friendly",
                "Ask clarifying questions", 
                "Provide accurate information"
            ]) \
            .set_step_criteria("User's question has been answered")
```

### Hybrid Approach

You can use both traditional prompts and contexts in the same agent:

```python
class HybridAgent(AgentBase):
    def __init__(self):
        super().__init__(name="hybrid", route="/hybrid")
        
        # Traditional prompt sections (from skills, global settings, etc.)
        # These will coexist with contexts
        
        # Define contexts for structured workflows
        contexts = self.define_contexts()
        workflow = contexts.add_context("default")
        
        workflow.add_step("structured_process") \
            .set_text("Following the structured workflow...") \
            .set_step_criteria("Workflow complete")
```

### Migration Strategy

1. **Start Simple**: Convert one workflow at a time
2. **Preserve Existing**: Keep traditional prompts for simple interactions
3. **Add Structure**: Use contexts for complex, multi-step processes
4. **Test Thoroughly**: Verify navigation and function access work as expected
5. **Iterate**: Refine step criteria and navigation based on testing

---

## Conclusion

The Contexts and Steps system provides powerful workflow control for building sophisticated AI agents. By combining structured navigation, function restrictions, and clear completion criteria, you can create predictable, user-friendly agent experiences that guide users through complex processes while maintaining security and control.

Start with simple single-context workflows and gradually build more complex multi-context systems as your requirements grow. The system is designed to be flexible and scalable, supporting both simple linear workflows and complex branching conversation trees. 

### Context Inheritance

Contexts can inherit from other contexts to create hierarchical structures:

```python
from signalwire_agents import AgentBase
from signalwire_agents.core.context import Context

class CustomerServiceAgent(AgentBase):
    def __init__(self):
        super().__init__(name="customer-service", route="/support")
        
        # Base context for all customer interactions
        base_context = Context(
            name="customer_base",
            description="Base context for all customer interactions",
            instructions=[
                "Always be polite and professional",
                "Verify customer identity before accessing account information",
                "Document all interactions in the customer record"
            ]
        )
        
        # Billing context inherits from base
        billing_context = Context(
            name="billing_support",
            description="Handle billing inquiries and payment issues",
            parent=base_context,  # Inherits from base_context
            instructions=[
                "Check payment history before suggesting solutions",
                "Offer payment plan options for overdue accounts",
                "Escalate disputes over $500 to billing manager"
            ]
        )
        
        # Technical support context also inherits from base
        tech_context = Context(
            name="technical_support", 
            description="Provide technical assistance and troubleshooting",
            parent=base_context,  # Also inherits from base_context
            instructions=[
                "Start with basic troubleshooting steps",
                "Document error messages and symptoms",
                "Create support tickets for unresolved issues"
            ]
        )
        
        self.add_context(base_context)
        self.add_context(billing_context)
        self.add_context(tech_context)

def main():
    agent = CustomerServiceAgent()
    agent.run()

if __name__ == "__main__":
    main()
```

## Dynamic Context Switching

Contexts can be switched dynamically during conversations based on user input or business logic:

```python
from signalwire_agents import AgentBase
from signalwire_agents.core.context import Context
from signalwire_agents.core.function_result import SwaigFunctionResult

class AdaptiveAgent(AgentBase):
    def __init__(self):
        super().__init__(name="adaptive", route="/adaptive")
        
        # Define multiple contexts
        self.setup_contexts()
        
        # Start with general context
        self.set_active_context("general")
    
    def setup_contexts(self):
        general = Context(
            name="general",
            description="General conversation and routing",
            instructions=[
                "Determine what the user needs help with",
                "Route to appropriate specialized context",
                "Be helpful and friendly"
            ]
        )
        
        sales = Context(
            name="sales",
            description="Sales and product information",
            instructions=[
                "Focus on product benefits and features",
                "Understand customer needs and budget",
                "Provide pricing and availability information"
            ]
        )
        
        support = Context(
            name="support", 
            description="Technical support and troubleshooting",
            instructions=[
                "Diagnose technical issues systematically",
                "Provide step-by-step solutions",
                "Escalate complex problems to specialists"
            ]
        )
        
        self.add_context(general)
        self.add_context(sales)
        self.add_context(support)
    
    @AgentBase.tool(
        name="switch_to_sales",
        description="Switch to sales context for product inquiries",
        parameters={}
    )
    def switch_to_sales(self, args, raw_data):
        self.set_active_context("sales")
        return SwaigFunctionResult("Switching to sales mode. How can I help you with our products?")
    
    @AgentBase.tool(
        name="switch_to_support", 
        description="Switch to technical support context",
        parameters={}
    )
    def switch_to_support(self, args, raw_data):
        self.set_active_context("support")
        return SwaigFunctionResult("Switching to technical support. What issue are you experiencing?")

def main():
    agent = AdaptiveAgent()
    agent.run()

if __name__ == "__main__":
    main()
```

## Context-Aware Function Behavior

Functions can behave differently based on the active context:

```python
from signalwire_agents import AgentBase
from signalwire_agents.core.context import Context
from signalwire_agents.core.function_result import SwaigFunctionResult

class ContextAwareAgent(AgentBase):
    def __init__(self):
        super().__init__(name="context-aware", route="/context")
        
        # Setup contexts
        self.setup_contexts()
        self.set_active_context("customer")
    
    def setup_contexts(self):
        customer = Context(
            name="customer",
            description="Customer-facing interactions",
            instructions=["Be friendly and helpful", "Use simple language"]
        )
        
        internal = Context(
            name="internal", 
            description="Internal staff interactions",
            instructions=["Be direct and technical", "Include detailed information"]
        )
        
        self.add_context(customer)
        self.add_context(internal)
    
    @AgentBase.tool(
        name="get_account_info",
        description="Get account information",
        parameters={
            "account_id": {
                "type": "string",
                "description": "Account identifier"
            }
        }
    )
    def get_account_info(self, args, raw_data):
        account_id = args.get("account_id")
        
        # Get the current context
        current_context = self.get_active_context()
        
        if current_context.name == "customer":
            # Customer-friendly response
            return SwaigFunctionResult(
                f"Your account {account_id} is in good standing. "
                "Your next billing date is March 15th."
            )
        elif current_context.name == "internal":
            # Detailed internal response
            return SwaigFunctionResult(
                f"Account {account_id}: Status=ACTIVE, Balance=$125.50, "
                "Last_Payment=2024-02-15, Next_Bill=2024-03-15, "
                "Plan=Premium, Usage=85% of limit"
            )
        else:
            return SwaigFunctionResult("Account information retrieved.")

def main():
    agent = ContextAwareAgent()
    agent.run()

if __name__ == "__main__":
    main() 