#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Call Flow & Actions Demo

This example demonstrates SDK features for controlling the call lifecycle and
executing actions from within SWAIG tool functions.

## Call Flow Verbs

Call flow verbs let you insert SWML verbs at specific points in the call:

- add_pre_answer_verb()  — runs BEFORE the call is answered (e.g. ringback tone)
- add_answer_verb()      — configures the answer verb itself (e.g. max_duration)
- add_post_answer_verb() — runs AFTER answering but BEFORE the AI starts
- add_post_ai_verb()     — runs AFTER the AI conversation ends

## Debug Events

enable_debug_events() makes the SignalWire platform POST real-time events
(barge, errors, session lifecycle) to the agent. Use @agent.on_debug_event
to handle them.

## SwaigFunctionResult Actions

Tool functions return SwaigFunctionResult objects. Beyond a text response,
these can carry actions that the platform executes:

- connect()              — transfer the call to another destination
- send_sms()             — send a text message during the call
- record_call()          — start background call recording
- play_background_file() — play audio in the background
- hold()                 — put the caller on hold
- update_global_data()   — update session-wide key/value data
- toggle_functions()     — enable/disable specific tools mid-call
- update_settings()      — change AI settings (temperature, etc.) mid-call
- add_dynamic_hints()    — improve speech recognition for specific terms
- set_end_of_speech_timeout() — adjust silence detection timing
- hangup()               — terminate the call

## Post-Processing

SwaigFunctionResult(msg, post_process=True) tells the platform to let the AI
speak to the user one more time before executing the attached actions. This is
useful for confirmation workflows like "I'll transfer you now — anything else?"
"""

from signalwire_agents import AgentBase
from signalwire_agents.core.function_result import SwaigFunctionResult


class CallFlowDemoAgent(AgentBase):
    """Demonstrates call flow verbs, debug events, and SwaigFunctionResult actions."""

    def __init__(self):
        super().__init__(name="call-flow-demo", route="/call-flow-demo")

        # --- Voice ---
        self.add_language(name="English", code="en-US", voice="inworld.Mark")

        # --- Prompt ---
        self.prompt_add_section(
            "Role",
            body="You are a call center demo agent that showcases call-flow features."
        )
        self.prompt_add_section("Instructions", bullets=[
            "Use transfer_to_support when the caller asks to speak to a person",
            "Use send_confirmation to send the caller an SMS",
            "Use start_recording when the caller agrees to be recorded",
            "Use play_hold_music to play background music",
            "Use update_preferences to save caller preferences",
            "Use adjust_speech when the caller mentions unusual names or terms",
        ])

        # --- LLM parameters ---
        self.set_params({"temperature": 0.3})

        # ----------------------------------------------------------------
        # Call flow verbs — control what happens before/after the AI runs
        # ----------------------------------------------------------------

        # Play a US ringback tone before answering. auto_answer=False prevents
        # the play verb from implicitly answering the call.
        self.add_pre_answer_verb("play", {
            "urls": ["ring:us"],
            "auto_answer": False
        })

        # Configure the answer verb with a maximum call duration of 1 hour.
        self.add_answer_verb({"max_duration": 3600})

        # After answering, play a welcome message before the AI takes over.
        self.add_post_answer_verb("play", {
            "url": "say:Welcome to the call flow demo."
        })

        # After the AI conversation ends, cleanly hang up the call.
        self.add_post_ai_verb("hangup", {})

        # ----------------------------------------------------------------
        # Debug events — receive real-time call events from the platform
        # ----------------------------------------------------------------

        # Level 1 gives high-level events: barge, errors, session start/end.
        # Level 2+ adds high-volume events (every LLM request/response).
        self.enable_debug_events(level=1)

        # Register a debug event handler using the instance decorator.
        # on_debug_event is an instance method, so we register in __init__.
        # event_type is a label like "barge", "llm_error", "session_start".
        @self.on_debug_event
        def handle_debug(event_type, data):
            if event_type == "barge":
                self.log.info("barge_detected",
                              elapsed_ms=data.get("barge_elapsed_ms"))
            elif event_type == "llm_error":
                self.log.error("llm_error", error=data.get("error"))
            else:
                self.log.debug("debug_event", event_type=event_type)

    # ----------------------------------------------------------------
    # SWAIG tools demonstrating SwaigFunctionResult action helpers
    # ----------------------------------------------------------------

    @AgentBase.tool(
        name="transfer_to_support",
        description="Transfer the caller to a support department",
        parameters={
            "department": {
                "type": "string",
                "description": "Department name (e.g. billing, technical)",
            }
        }
    )
    def transfer_to_support(self, department: str):
        """Transfer the call to a support department."""
        destinations = {
            "billing":   "+15551000001",
            "technical": "+15551000002",
            "sales":     "+15551000003",
        }
        dest = destinations.get(department.lower(), "+15551000000")

        # post_process=True lets the AI speak one more time before the
        # transfer actually executes — useful for a goodbye message.
        return (
            SwaigFunctionResult(
                f"Transferring you to {department} support now.",
                post_process=True
            )
            .connect(dest, final=True)
        )

    @AgentBase.tool(
        name="send_confirmation",
        description="Send an SMS confirmation to the caller",
        parameters={
            "phone": {
                "type": "string",
                "description": "Phone number in E.164 format to send SMS to",
            },
            "message": {
                "type": "string",
                "description": "The confirmation message text",
            }
        }
    )
    def send_confirmation(self, phone: str, message: str):
        """Send an SMS message to the caller."""
        return (
            SwaigFunctionResult(f"Sending confirmation to {phone}.")
            .send_sms(
                to_number=phone,
                from_number="+15559999999",
                body=message
            )
        )

    @AgentBase.tool(
        name="start_recording",
        description="Start recording the call"
    )
    def start_recording(self):
        """Begin background call recording in stereo WAV format."""
        return (
            SwaigFunctionResult("Recording has started.")
            .record_call(control_id="demo-recording", stereo=True, format="wav")
        )

    @AgentBase.tool(
        name="play_hold_music",
        description="Play hold music in the background"
    )
    def play_hold_music(self):
        """Play background music and put the caller on hold."""
        return (
            SwaigFunctionResult("Playing hold music for you.")
            .play_background_file("https://cdn.example.com/hold-music.mp3")
            .hold(timeout=120)
        )

    @AgentBase.tool(
        name="update_preferences",
        description="Save a caller preference",
        parameters={
            "key": {
                "type": "string",
                "description": "Preference key (e.g. language, notifications)",
            },
            "value": {
                "type": "string",
                "description": "Preference value",
            }
        }
    )
    def update_preferences(self, key: str, value: str):
        """Update global session data and toggle functions based on preferences."""
        result = (
            SwaigFunctionResult(f"Preference '{key}' set to '{value}'.")
            .update_global_data({key: value})
        )
        # Example: disable the send_confirmation tool if notifications are off
        if key == "notifications" and value == "off":
            result.toggle_functions([
                {"function": "send_confirmation", "active": False}
            ])
        return result

    @AgentBase.tool(
        name="adjust_speech",
        description="Add speech recognition hints for unusual terms",
        parameters={
            "hints": {
                "type": "string",
                "description": "Comma-separated terms to add as speech hints",
            }
        }
    )
    def adjust_speech(self, hints: str):
        """Add dynamic hints and adjust speech timeout for better recognition."""
        hint_list = [h.strip() for h in hints.split(",")]
        return (
            SwaigFunctionResult(f"Added speech hints: {', '.join(hint_list)}")
            .add_dynamic_hints(hint_list)
            .set_end_of_speech_timeout(1200)
        )


# --- Entry point ---
if __name__ == "__main__":
    agent = CallFlowDemoAgent()
    print("Call Flow & Actions Demo")
    print(f"  Route: {agent.route}")
    print("  Features: pre/post-answer verbs, debug events, SwaigFunctionResult actions")
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    agent.run()
