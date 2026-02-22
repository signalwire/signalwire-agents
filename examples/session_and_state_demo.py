#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Session & State Demo

This example demonstrates session lifecycle hooks and state management features
that are not covered by other examples.

## on_summary callback

After a call ends, the platform generates a summary based on the post-prompt
instructions. Override `on_summary(self, summary, raw_data)` to receive and
process it — e.g. log to a database, trigger follow-up actions, etc.

Requires set_post_prompt() to be configured; otherwise on_summary is never called.

## set_global_data

set_global_data() sets session-wide key/value pairs accessible to the AI and to
all tool functions via ${global_data.key} variable expansion. Data persists for
the entire call session. Multiple calls merge (not replace) into existing data.

## set_post_prompt

set_post_prompt() tells the platform what to summarize after the conversation.
The summary is delivered to on_summary() or to a URL set via set_post_prompt_url().

## SwaigFunctionResult.update_global_data / hangup

Tools can modify global data mid-call with update_global_data() and terminate
the call with hangup(). These are actions attached to SwaigFunctionResult.
"""

import json
from signalwire_agents import AgentBase
from signalwire_agents.core.function_result import SwaigFunctionResult


class SessionStateDemoAgent(AgentBase):
    """Demonstrates on_summary, global data, and post-prompt features."""

    def __init__(self):
        super().__init__(name="session-state-demo", route="/session-state-demo")

        # --- Voice ---
        self.add_language(name="English", code="en-US", voice="inworld.Mark")

        # --- Prompt ---
        self.prompt_add_section(
            "Role",
            body="You are a customer service agent that tracks session state."
        )
        self.prompt_add_section("Instructions", bullets=[
            "Greet the caller and ask how you can help",
            "Use update_customer_info to record information the caller provides",
            "Use get_session_info to check what information has been collected",
            "Use end_session when the caller is done",
        ])

        # --- Global data ---
        # Seed the session with default values. The AI can reference these as
        # ${global_data.status} in prompts, and tools can read/update them.
        self.set_global_data({
            "status": "active",
            "customer_name": "",
            "issue_type": "",
        })

        # --- Post-prompt ---
        # This instructs the platform to generate a structured summary after
        # the call. The summary is delivered to on_summary() below.
        self.set_post_prompt(
            "Summarize the interaction as JSON: "
            '{"customer_name": "...", "issue_type": "...", "resolved": true/false, '
            '"notes": "brief summary"}'
        )

    # ----------------------------------------------------------------
    # Session lifecycle hooks
    # ----------------------------------------------------------------

    def on_summary(self, summary, raw_data=None):
        """
        Called after the call ends with the post-prompt summary.

        Args:
            summary: Parsed summary object (dict or string depending on AI output)
            raw_data: Full POST body from the platform, includes global_data, etc.
        """
        # Log the summary for analytics / CRM integration
        if isinstance(summary, dict):
            print(f"Session summary: {json.dumps(summary, indent=2)}")
            customer = summary.get("customer_name", "unknown")
            resolved = summary.get("resolved", False)
            print(f"Customer: {customer}, Resolved: {resolved}")
        else:
            print(f"Session summary (raw): {summary}")

        # Access global_data from raw_data for additional session info
        if raw_data:
            global_data = raw_data.get("global_data", {})
            print(f"Final global data: {json.dumps(global_data, indent=2)}")

    # ----------------------------------------------------------------
    # SWAIG tools
    # ----------------------------------------------------------------

    @AgentBase.tool(
        name="update_customer_info",
        description="Record a piece of customer information",
        parameters={
            "field": {
                "type": "string",
                "description": "The field to update (e.g. customer_name, issue_type)",
            },
            "value": {
                "type": "string",
                "description": "The value to set",
            }
        }
    )
    def update_customer_info(self, field: str, value: str):
        """Update a field in the session's global data."""
        return (
            SwaigFunctionResult(f"Updated {field} to '{value}'.")
            .update_global_data({field: value})
        )

    @AgentBase.tool(
        name="get_session_info",
        description="Retrieve current session information"
    )
    def get_session_info(self):
        """Return a summary of what has been collected so far."""
        # Note: In a real scenario, tools receive global_data in raw_data.
        # Here we return a prompt for the AI to read from its context.
        return SwaigFunctionResult(
            "Check the global data variables for current session information: "
            "${global_data.status}, ${global_data.customer_name}, ${global_data.issue_type}"
        )

    @AgentBase.tool(
        name="end_session",
        description="End the current session and hang up",
        parameters={
            "reason": {
                "type": "string",
                "description": "Reason for ending the session",
            }
        }
    )
    def end_session(self, reason: str):
        """Mark the session as closed and terminate the call."""
        return (
            SwaigFunctionResult(f"Session ended: {reason}. Goodbye!")
            .update_global_data({"status": "closed", "close_reason": reason})
            .hangup()
        )


# --- Entry point ---
if __name__ == "__main__":
    agent = SessionStateDemoAgent()
    print("Session & State Demo")
    print(f"  Route: {agent.route}")
    print("  Features: on_summary, set_global_data, set_post_prompt, update_global_data, hangup")
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    agent.run()
