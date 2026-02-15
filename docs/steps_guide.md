# Steps Guide

## Table of Contents

- [Overview](#overview)
- [Step Basics](#step-basics)
  - [Creating Steps](#creating-steps)
  - [Step Text](#step-text)
  - [Step Criteria](#step-criteria)
  - [Function Restrictions](#function-restrictions)
  - [Navigation with valid_steps](#navigation-with-valid_steps)
- [Gather Info Mode](#gather-info-mode)
  - [What It Solves](#what-it-solves)
  - [How It Works Internally](#how-it-works-internally)
  - [Basic Gather Example](#basic-gather-example)
  - [The Gather Prompt (Preamble)](#the-gather-prompt-preamble)
  - [Question Types](#question-types)
  - [Confirmation Flow](#confirmation-flow)
  - [Per-Question Instructions](#per-question-instructions)
  - [Per-Question Functions](#per-question-functions)
  - [Output Storage](#output-storage)
  - [Auto-Advancing After Gather](#auto-advancing-after-gather)
  - [Combining Gather with Normal Step Mode](#combining-gather-with-normal-step-mode)
- [Normal Step Mode](#normal-step-mode)
  - [Prompt Injection](#prompt-injection)
  - [Step Criteria and Completion](#step-criteria-and-completion)
  - [Multi-Step Workflows](#multi-step-workflows)
- [How Gather and Normal Mode Work Together](#how-gather-and-normal-mode-work-together)
- [Complete Examples](#complete-examples)
  - [Travel Profile Agent](#travel-profile-agent)
  - [Customer Onboarding Agent](#customer-onboarding-agent)
  - [Support Ticket Agent](#support-ticket-agent)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)

---

## Overview

Steps are the building blocks of structured AI conversations. Each step defines what the AI should do, which tools it can use, and where it can go next. Steps live inside contexts and execute sequentially or via explicit navigation.

There are two modes a step can operate in:

- **Normal Mode**: The step's text is injected as instructions. The AI follows those instructions, and the step completes based on criteria you define or by navigating to the next step.
- **Gather Info Mode**: The step collects structured information from the caller one question at a time, with zero tool artifacts in the LLM conversation history. Once all questions are answered, the step either auto-advances or returns to normal mode.

## Step Basics

### Creating Steps

Steps are created within a context using `add_step()`. The method returns a `Step` object that supports method chaining:

```python
from signalwire_agents import AgentBase

class MyAgent(AgentBase):
    def __init__(self):
        super().__init__(name="My Agent", route="/my-agent")

        self.prompt_add_section("Role", "You are a helpful assistant.")

        contexts = self.define_contexts()
        ctx = contexts.add_context("default")

        ctx.add_step("greeting") \
            .set_text("Greet the caller warmly and ask how you can help.") \
            .set_valid_steps(["collect_info", "general_help"])

        ctx.add_step("collect_info") \
            .set_text("Help the caller with their specific request.")

        ctx.add_step("general_help") \
            .set_text("Provide general assistance to the caller.")
```

### Step Text

Every step in normal mode needs instructions. You can set these as plain text or as structured POM sections:

```python
# Plain text
step.set_text("Ask the caller for their account number and verify their identity.")

# POM sections (mutually exclusive with set_text)
step.add_section("Task", "Verify the caller's identity.") \
    .add_bullets("Requirements", [
        "Ask for their account number",
        "Verify their date of birth",
        "Confirm their mailing address"
    ])
```

Steps that only use gather info mode (with `completion_action="next_step"`) can set minimal text since it won't be shown during question collection:

```python
ctx.add_step("collect_profile") \
    .set_text("Collect the caller's profile.") \
    .set_gather_info(completion_action="next_step", ...) \
    ...
```

### Step Criteria

Step criteria define when a step is considered complete. The AI uses these to decide when to call `next_step`:

```python
ctx.add_step("verify_identity") \
    .set_text("Verify the caller's identity by confirming their account number and date of birth.") \
    .set_step_criteria("The caller has provided a valid account number AND confirmed their date of birth.") \
    .set_valid_steps(["handle_request"])
```

Without criteria, the step relies on navigation via `valid_steps` or function calls to advance.

### Function Restrictions

By default, all registered functions are available in every step. You can restrict which functions are visible:

```python
# Only these functions are available in this step
ctx.add_step("lookup") \
    .set_text("Look up the caller's account.") \
    .set_functions(["account_lookup", "verify_identity"])

# No functions available in this step
ctx.add_step("greeting") \
    .set_text("Greet the caller.") \
    .set_functions("none")
```

### Navigation with valid_steps

`valid_steps` controls which steps the AI can navigate to from the current step. The AI calls the `next_step` function to move between steps:

```python
ctx.add_step("greeting") \
    .set_text("Greet the caller and find out what they need.") \
    .set_valid_steps(["billing", "support", "sales"])

ctx.add_step("billing") \
    .set_text("Help the caller with billing questions.") \
    .set_valid_steps(["greeting"])  # Can go back to greeting

ctx.add_step("support") \
    .set_text("Help the caller with technical support.")
    # No valid_steps = terminal step
```

---

## Gather Info Mode

### What It Solves

When an AI agent needs to collect structured information (name, address, account number, etc.), the traditional approach uses SWAIG functions — the AI calls a function for each piece of data, which creates `tool_call` and `tool_result` entries in the conversation history. These tool artifacts confuse some models (especially reasoning models at low effort settings), waste tokens, and can cause the model to lose track of where it is in the collection flow.

Gather info mode solves this by using **dynamic step instruction re-injection**. Questions are presented one at a time by swapping out the system instruction, and answers are recorded via an internal function that routes through the system-log path — producing **zero** tool_call/tool_result entries in the LLM-visible conversation history.

### How It Works Internally

Understanding the internal flow helps you design better gather configurations:

1. **Step entry**: When the AI enters a step with `gather_info`, the system detects it and switches to gather questioning mode.

2. **Preamble injection** (first question only): If the gather has a `prompt`, it's injected as a **persistent** system message. This stays in the conversation history for the entire gather sequence, giving the AI its personality and context.

3. **Question injection**: A minimal system instruction is injected as a **clearable** message containing only: the question text, type hint (if not string), confirmation instructions (if needed), and any per-question prompt text.

4. **Tool lockdown**: During gather mode, **all normal functions are hidden** — including `next_step`. Only `gather_submit` (an internal function) and any per-question `functions` are visible. This prevents the AI from navigating away or calling irrelevant tools.

5. **Answer submission**: When the AI calls `gather_submit`, the answer is written to `global_data` and the next question's instruction is re-injected. The `gather_submit` call itself routes through the system-log path, so the LLM never sees tool_call/tool_result for it.

6. **Completion**: When all questions are answered, either:
   - The step auto-advances to the next step (`completion_action="next_step"`)
   - The step returns to normal mode with the regular step text, plus a note that gathered data is available

Here's what the LLM conversation history looks like during gather mode:

```
[system] You are a travel assistant. You need to collect some details.    ← persistent preamble
[system] Ask the user: "What is your first name?"                        ← clearable, changes per question
         When you have the answer, call the gather_submit function.
         Do not ask the user any other questions.

[assistant] Hi there! I'm your travel assistant. What's your first name?
[user] Tony.
                                                        ← gather_submit recorded via system-log (invisible)
[system] Ask the user: "What is your last name?"        ← previous question instruction replaced
         ...

[assistant] Great, Tony! And your last name?
[user] Smith.
```

No tool_call/tool_result entries anywhere. Clean conversation history.

### Basic Gather Example

```python
ctx.add_step("collect_info") \
    .set_text("Help the caller with their request.") \
    .set_gather_info(output_key="caller_info") \
    .add_gather_question("first_name", "What is your first name?") \
    .add_gather_question("last_name", "What is your last name?") \
    .add_gather_question("email", "What is your email address?")
```

This collects three pieces of information, stores them under `caller_info` in global_data, then returns to normal step mode with the step text "Help the caller with their request."

### The Gather Prompt (Preamble)

The gather `prompt` is injected once as a persistent message when the first question begins. It gives the AI personality and context for the entire question sequence:

```python
ctx.add_step("collect_profile") \
    .set_text("Use the profile to recommend products.") \
    .set_gather_info(
        output_key="profile",
        prompt="Welcome the caller and introduce yourself as a product specialist. "
               "Explain that you need to ask a few quick questions to find the "
               "best products for them. Be friendly and conversational."
    ) \
    .add_gather_question("name", "What is your name?") \
    .add_gather_question("budget", "What is your budget?", type="number")
```

Without a gather `prompt`, the AI jumps straight into asking the first question with no introduction. The preamble lets it greet the caller naturally before the structured collection begins.

The gather prompt persists in conversation history throughout all questions. The per-question instructions are the only thing that changes between questions.

### Question Types

Each question has a `type` that controls the JSON schema of the `answer` parameter in `gather_submit`. This helps the AI provide properly typed answers:

```python
# String (default) - free text
.add_gather_question("name", "What is your name?", type="string")

# Integer - whole numbers
.add_gather_question("age", "How old are you?", type="integer")

# Number - decimal values
.add_gather_question("budget", "What is your budget in dollars?", type="number")

# Boolean - yes/no questions
.add_gather_question("has_passport", "Do you have a valid passport?", type="boolean")
```

The type is communicated to the AI via a hint in the question instruction (e.g., "The answer must be a integer value.") and enforced in the function schema.

### Confirmation Flow

When `confirm=True`, the AI must read the answer back to the caller and get explicit confirmation before submitting. This is enforced at the function level:

```python
.add_gather_question(
    "last_name",
    "What is your last name?",
    type="string",
    confirm=True
)
```

How it works:

1. The question instruction includes: "You MUST confirm the answer with the user before submitting."
2. The `gather_submit` function schema includes a required `confirmed_by_user` enum parameter with values `"true"` and `"false"`.
3. If the AI calls `gather_submit` with `confirmed_by_user` set to `"false"` (or anything other than `"true"`), the function rejects the submission and tells the AI to confirm with the user first.
4. The AI must read back the answer, get the user's "yes", then call `gather_submit` again with `confirmed_by_user: "true"`.

Example conversation flow:
```
[assistant] What is your last name?
[user] Menesael.
[assistant] Just to confirm — your last name is Menesael?
[user] Yes, that's right.
[assistant] (calls gather_submit with answer="Menesael", confirmed_by_user="true")
```

The `confirmed_by_user` parameter is only added to the schema when the current question has `confirm=True`. For non-confirm questions, the schema only has the `answer` parameter.

### Per-Question Instructions

Each question can have a `prompt` field with additional instructions specific to that question:

```python
.add_gather_question(
    "home_airport",
    "What is your home airport or nearest major city for departure?",
    type="string",
    confirm=True,
    prompt="Use the resolve_airport function to validate the airport code "
           "before submitting. If the airport is ambiguous, clarify with the user."
)
```

The per-question prompt is appended to the question instruction, after the confirmation instructions (if any) and before "Do not ask the user any other questions."

### Per-Question Functions

Normally during gather mode, only `gather_submit` is visible to the AI. You can make additional functions available for specific questions:

```python
.add_gather_question(
    "home_airport",
    "What is your home airport?",
    type="string",
    confirm=True,
    prompt="Use the resolve_airport function to validate the airport code before submitting.",
    functions=["resolve_airport"]
)
```

The `resolve_airport` function must already be registered on the agent (via `@agent.tool` or `define_tool`). The `functions` array just controls visibility — it activates those functions for this question only, alongside `gather_submit`. When the next question begins, they're deactivated again.

This is useful for validation, lookup, or enrichment during data collection without exposing the full tool set.

### Output Storage

Answers are stored in `global_data`, which is available in prompt variable expansion via `${key}`:

```python
# Store under a namespace
.set_gather_info(output_key="profile")
# Results in: global_data.profile.first_name, global_data.profile.last_name, etc.
# Accessible in prompts as: ${profile}

# Store at top level (no output_key)
.set_gather_info()
# Results in: global_data.first_name, global_data.last_name, etc.
# Accessible in prompts as: ${first_name}, ${last_name}, etc.
```

After gathering, `global_data` is refreshed so subsequent step prompts can reference the collected values using `${variable}` syntax:

```python
ctx.add_step("plan_trip") \
    .set_text(
        "The caller's travel profile is: ${profile}. "
        "Use their name, budget, and preferences to suggest destinations."
    )
```

### Auto-Advancing After Gather

With `completion_action="next_step"`, the step automatically advances when the last question is answered:

```python
ctx.add_step("collect_profile") \
    .set_text("Collect the caller's profile.") \
    .set_gather_info(
        output_key="profile",
        completion_action="next_step",
        prompt="Welcome the caller. You need to collect a few details."
    ) \
    .add_gather_question("name", "What is your name?") \
    .add_gather_question("email", "What is your email?")

# This step runs immediately after the last question is answered
ctx.add_step("process") \
    .set_text("You have the caller's profile in ${profile}. Help them with their request.")
```

When auto-advancing:
- The gather state is cleared
- The next step's instructions are injected immediately
- `valid_steps` on the gather step is not needed (the advance is automatic)

### Combining Gather with Normal Step Mode

Without `completion_action="next_step"`, the step returns to normal mode after all questions are answered. The step's `text` is injected along with a note that gathered data is available:

```python
ctx.add_step("intake") \
    .set_text(
        "Review the caller's information in ${intake_data}. "
        "Confirm everything looks correct, then proceed to scheduling."
    ) \
    .set_gather_info(output_key="intake_data") \
    .add_gather_question("name", "What is your name?") \
    .add_gather_question("dob", "What is your date of birth?") \
    .add_gather_question("reason", "What is the reason for your visit?") \
    .set_valid_steps(["schedule"])
```

Flow:
1. Gather mode: Questions are asked one at a time
2. All questions answered → step switches to normal mode
3. Step text is injected with `valid_steps` and `step_criteria` restored
4. The AI follows the normal step instructions using the gathered data
5. Navigation to `schedule` becomes available

---

## Normal Step Mode

### Prompt Injection

In normal mode, the step's text is injected as a system message with this structure:

```
[context prompt if any]

## Instructions to complete the Current Step
[your step text]

Do not mention to the user that you are following steps, or the names of the steps.
Do not ask the user any questions not explicitly related to these instructions.
Do not end the conversation when this step is complete.
[step criteria if any]
```

The step text supports `${variable}` expansion from `global_data` and prompt variables.

### Step Criteria and Completion

Step criteria tell the AI when a step is done. The AI evaluates the criteria and calls `next_step` when they're met:

```python
ctx.add_step("verify") \
    .set_text("Verify the caller's identity.") \
    .set_step_criteria(
        "The caller has provided their account number "
        "AND confirmed their date of birth."
    ) \
    .set_valid_steps(["handle_request"])
```

The criteria are included in the injected prompt as:
```
The Current Step is not complete until the following criteria has been met:
The caller has provided their account number AND confirmed their date of birth.
```

### Multi-Step Workflows

Steps can form linear flows, branching flows, or loops:

```python
# Linear flow
ctx.add_step("step1") \
    .set_text("Do thing one.") \
    .set_valid_steps(["step2"])

ctx.add_step("step2") \
    .set_text("Do thing two.") \
    .set_valid_steps(["step3"])

ctx.add_step("step3") \
    .set_text("Do thing three.")

# Branching flow
ctx.add_step("triage") \
    .set_text("Find out if the caller needs billing help or tech support.") \
    .set_valid_steps(["billing", "tech_support"])

ctx.add_step("billing") \
    .set_text("Help with billing.")

ctx.add_step("tech_support") \
    .set_text("Help with technical issues.")

# Loop back
ctx.add_step("collect") \
    .set_text("Ask for the item details.") \
    .set_valid_steps(["confirm"])

ctx.add_step("confirm") \
    .set_text("Confirm the details. If wrong, go back to collect.") \
    .set_valid_steps(["collect", "submit"])

ctx.add_step("submit") \
    .set_text("Submit the order.")
```

---

## How Gather and Normal Mode Work Together

A common pattern is: collect structured data first, then use it in a normal conversation step. Here's the complete flow:

```
┌─────────────────────────────────────────┐
│  Step: collect_profile                  │
│  Mode: GATHER                           │
│                                         │
│  [persistent] Gather prompt/preamble    │
│  [clearable]  Q1: "What is your name?"  │
│  [clearable]  Q2: "What is your email?" │
│  [clearable]  Q3: "What is your phone?" │
│                                         │
│  Tools: gather_submit only              │
│  History: zero tool artifacts           │
│                                         │
│  completion_action: "next_step"         │
│  ─── auto-advance when done ───────►    │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Step: handle_request                   │
│  Mode: NORMAL                           │
│                                         │
│  Step text: "You have the caller's      │
│  profile in ${profile}. Help them..."   │
│                                         │
│  Tools: all registered functions        │
│  Navigation: valid_steps available      │
│  Criteria: step_criteria evaluated      │
└─────────────────────────────────────────┘
```

Key transitions:
- **Gather → Normal (same step)**: When no `completion_action`, gather completes and the step's `text` is injected with all normal machinery (criteria, valid_steps, functions) restored.
- **Gather → Next Step (auto-advance)**: With `completion_action="next_step"`, gather completes and immediately advances to the next step. The gather step's `text` is never shown.
- **Normal → Gather**: Not supported within a single step. Each step is either gather or normal. Use separate steps for sequential gather-then-act flows.

---

## Complete Examples

### Travel Profile Agent

Collects a travel profile, then recommends destinations:

```python
from signalwire_agents import AgentBase


class TravelAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Travel Agent", route="/travel")

        self.prompt_add_section(
            "Role",
            "You are a friendly travel booking assistant."
        )

        contexts = self.define_contexts()
        ctx = contexts.add_context("default")

        # Step 1: Collect profile (gather mode, auto-advance)
        ctx.add_step("collect_profile") \
            .set_text("Collect the caller's travel profile.") \
            .set_gather_info(
                output_key="profile",
                completion_action="next_step",
                prompt="Welcome the caller and introduce yourself as a travel "
                       "booking assistant. You need to collect a few details "
                       "to build their travel profile. Be warm and conversational."
            ) \
            .add_gather_question(
                "first_name",
                "What is your first name?") \
            .add_gather_question(
                "last_name",
                "What is your last name?",
                confirm=True) \
            .add_gather_question(
                "party_size",
                "How many people are traveling, including yourself?",
                type="integer") \
            .add_gather_question(
                "budget_per_person",
                "Roughly what is your budget per person in dollars?",
                type="number") \
            .add_gather_question(
                "has_passport",
                "Do you have a valid passport for international travel?",
                type="boolean") \
            .add_gather_question(
                "home_airport",
                "What is your home airport or nearest major city for departure?",
                confirm=True)

        # Step 2: Recommend destinations (normal mode)
        ctx.add_step("plan_trip") \
            .set_text(
                "You now have the caller's travel profile in ${profile}. "
                "Use their name, party size, budget, passport status, and "
                "home airport to suggest three vacation destinations with "
                "rough pricing. If they don't have a passport, only suggest "
                "domestic destinations. Ask which one interests them."
            )

        self.add_language(name="English", code="en-US", voice="rime.spore")
```

### Customer Onboarding Agent

Collects customer info with validation functions, then creates their account:

```python
from signalwire_agents import AgentBase


class OnboardingAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Onboarding Agent", route="/onboard")

        self.prompt_add_section(
            "Role",
            "You are a customer onboarding specialist for Acme Corp."
        )

        # Register validation functions
        @self.tool(name="validate_email", description="Validate an email address")
        def validate_email(args, raw_data):
            email = args.get("email", "")
            # ... validation logic ...
            return f"Email {email} is valid."

        @self.tool(name="check_zip", description="Validate a US zip code")
        def check_zip(args, raw_data):
            zip_code = args.get("zip", "")
            # ... validation logic ...
            return f"Zip code {zip_code} is in Springfield, IL."

        contexts = self.define_contexts()
        ctx = contexts.add_context("default")

        # Collect customer info with per-question validation
        ctx.add_step("collect_info") \
            .set_text("Create the customer's account.") \
            .set_gather_info(
                output_key="customer",
                completion_action="next_step",
                prompt="Welcome the caller to Acme Corp. Explain that you'll "
                       "need to collect some information to set up their account."
            ) \
            .add_gather_question(
                "full_name",
                "What is your full name?",
                confirm=True) \
            .add_gather_question(
                "email",
                "What is your email address?",
                confirm=True,
                prompt="Use the validate_email function to check the email "
                       "is valid before submitting.",
                functions=["validate_email"]) \
            .add_gather_question(
                "zip_code",
                "What is your zip code?",
                prompt="Use the check_zip function to validate the zip code "
                       "and confirm the city/state with the caller.",
                functions=["check_zip"])

        # Confirm and create account
        ctx.add_step("confirm_account") \
            .set_text(
                "Review the customer information in ${customer}. "
                "Read back all the details and ask them to confirm "
                "everything is correct. Then create their account."
            ) \
            .set_functions(["create_account"]) \
            .set_step_criteria("The account has been created successfully.")

        self.add_language(name="English", code="en-US", voice="rime.spore")
```

### Support Ticket Agent

Gathers issue details, then routes to the right team:

```python
from signalwire_agents import AgentBase


class SupportAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Support Agent", route="/support")

        self.prompt_add_section(
            "Role",
            "You are a technical support agent."
        )

        contexts = self.define_contexts()
        ctx = contexts.add_context("default")

        # Collect ticket info, then return to normal mode for triage
        ctx.add_step("intake") \
            .set_text(
                "You have the caller's issue details in ${ticket}. "
                "Based on the category and description, route them to "
                "the appropriate team."
            ) \
            .set_gather_info(
                output_key="ticket",
                prompt="Thank the caller for contacting support. "
                       "You need to collect some details about their issue."
            ) \
            .add_gather_question(
                "name",
                "What is your name?") \
            .add_gather_question(
                "account_id",
                "What is your account ID?",
                confirm=True) \
            .add_gather_question(
                "category",
                "Is this about billing, a technical issue, or something else?") \
            .add_gather_question(
                "description",
                "Please describe the issue in detail.") \
            .set_valid_steps(["billing_support", "tech_support", "general_support"])

        ctx.add_step("billing_support") \
            .set_text("Help the caller with their billing issue. "
                      "Their details are in ${ticket}.")

        ctx.add_step("tech_support") \
            .set_text("Help the caller with their technical issue. "
                      "Their details are in ${ticket}.") \
            .set_functions(["run_diagnostics", "check_service_status"])

        ctx.add_step("general_support") \
            .set_text("Help the caller with their general inquiry. "
                      "Their details are in ${ticket}.")

        self.add_language(name="English", code="en-US", voice="rime.spore")
```

Note: This example uses gather **without** `completion_action`. After all questions are answered, the step returns to normal mode with the step text and `valid_steps` restored. The AI uses the gathered data to decide which support step to route to.

---

## API Reference

### Step Methods

| Method | Description |
|--------|-------------|
| `set_text(text)` | Set the step's instruction text |
| `add_section(title, body)` | Add a POM section (mutually exclusive with `set_text`) |
| `add_bullets(title, bullets)` | Add a POM bullet section |
| `set_step_criteria(criteria)` | Set completion criteria |
| `set_functions(functions)` | Set available functions (`"none"` or list of names) |
| `set_valid_steps(steps)` | Set navigable steps |
| `set_valid_contexts(contexts)` | Set navigable contexts |
| `set_gather_info(...)` | Enable gather info mode |
| `add_gather_question(...)` | Add a question to gather info |

### `set_gather_info()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_key` | str | None | Key in global_data to store answers under. If None, answers are stored at the top level. |
| `completion_action` | str | None | Set to `"next_step"` to auto-advance when all questions are answered. |
| `prompt` | str | None | Preamble text injected once as a persistent message when entering the gather step. |

### `add_gather_question()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key` | str | required | Key name for storing the answer in global_data |
| `question` | str | required | The question text presented to the AI |
| `type` | str | `"string"` | JSON schema type: `"string"`, `"integer"`, `"number"`, `"boolean"` |
| `confirm` | bool | `False` | If True, AI must confirm answer with user before submitting |
| `prompt` | str | None | Additional instruction text for this question |
| `functions` | list | None | Function names to make visible for this question only |

---

## Best Practices

1. **Use gather prompts for introductions.** The gather `prompt` is the right place for personality and context. It persists through all questions, letting the AI greet naturally on the first question.

2. **Use `completion_action="next_step"` for collect-then-act flows.** When gather is just a data collection phase before the real work, auto-advance keeps it clean. The step text can be minimal.

3. **Skip `completion_action` for triage flows.** When you need the AI to make decisions based on gathered data (routing, classification), let the step return to normal mode with `valid_steps` so the AI can navigate.

4. **Use `confirm=True` for critical data.** Names, email addresses, account numbers — anything that's hard to correct later. The confirmation flow adds one turn but prevents errors.

5. **Use per-question `functions` for validation.** Airport codes, zip codes, email addresses — let the AI validate before submitting rather than collecting bad data.

6. **Keep per-question `prompt` focused.** It's for tool usage instructions ("Use X function to validate") or special handling ("If the answer is ambiguous, ask for clarification"). Don't put personality or greetings here.

7. **Use `output_key` to namespace gathered data.** `output_key="profile"` keeps answers grouped under `${profile}` instead of polluting the top-level global_data with individual keys.

8. **Type your questions.** `type="integer"` and `type="number"` help the AI provide properly formatted answers and the function schema enforces it. Don't leave everything as the default string.
