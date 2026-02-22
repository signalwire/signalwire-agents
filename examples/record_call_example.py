#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Example: Using record_call and stop_record_call virtual helpers

This example shows how to use the new recording virtual helpers to:
1. Start background call recording with various configurations
2. Stop recordings by control ID
3. Chain recording operations with other actions
"""

import json
from signalwire_agents.core.function_result import SwaigFunctionResult


def basic_recording_example():
    """Simple background recording with default settings."""
    print("=== Basic Recording Example ===")
    
    result = SwaigFunctionResult("Starting basic call recording") \
        .record_call() \
        .say("This call is now being recorded")
    
    print(json.dumps(result.to_dict(), indent=2))
    print()


def advanced_recording_example():
    """Advanced recording with custom settings."""
    print("=== Advanced Recording Example ===")
    
    result = SwaigFunctionResult("Starting advanced call recording") \
        .record_call(
            control_id="support_call_2024_001",
            stereo=True,
            format="mp3",
            direction="both",  # Record both parties
            terminators="*#",  # Stop on * or # key
            beep=True,         # Play beep before recording
            max_length=600,    # 10 minutes max
            status_url="https://api.company.com/recording-webhook"
        ) \
        .say("This call is being recorded for quality and training purposes")
    
    print(json.dumps(result.to_dict(), indent=2))
    print()


def voicemail_recording_example():
    """Recording setup for voicemail system."""
    print("=== Voicemail Recording Example ===")
    
    result = SwaigFunctionResult("Please leave your message after the beep") \
        .record_call(
            control_id="voicemail_123456",
            format="wav",
            direction="speak",        # Only record caller's voice
            terminators="#",          # Stop on # key
            beep=True,               # Play beep before recording
            initial_timeout=5.0,     # Wait 5 seconds for speech
            end_silence_timeout=3.0, # Stop after 3 seconds of silence
            max_length=120           # 2 minute max message
        ) \
        .set_end_of_speech_timeout(2000)  # Shorter timeout for voicemail
    
    print(json.dumps(result.to_dict(), indent=2))
    print()


def stop_recording_example():
    """Stop a specific recording."""
    print("=== Stop Recording Example ===")
    
    # Stop specific recording
    result = SwaigFunctionResult("Ending call recording") \
        .stop_record_call("support_call_2024_001") \
        .say("Thank you for calling. Your feedback is important to us.")
    
    print(json.dumps(result.to_dict(), indent=2))
    print()


def customer_service_workflow():
    """Complete customer service recording workflow."""
    print("=== Customer Service Workflow ===")
    
    # Start recording when transferring to agent
    start_recording = SwaigFunctionResult("Transferring you to a customer service agent") \
        .record_call(
            control_id="cs_transfer_001",
            format="mp3",
            direction="both",
            beep=False,  # No beep so the caller doesn't hear a tone
            max_length=1800,  # 30 minutes max
            status_url="https://api.company.com/recording-status"
        ) \
        .update_global_data({"recording_id": "cs_transfer_001"}) \
        .say("Please hold while I connect you to an agent")
    
    print("Start recording:")
    print(json.dumps(start_recording.to_dict(), indent=2))
    print()
    
    # Later in the conversation, stop recording
    end_recording = SwaigFunctionResult("Call recording stopped") \
        .stop_record_call("cs_transfer_001") \
        .remove_global_data("recording_id") \
        .say("Thank you for calling. Have a wonderful day!")
    
    print("End recording:")
    print(json.dumps(end_recording.to_dict(), indent=2))
    print()


def compliance_recording_example():
    """Recording for compliance/legal purposes."""
    print("=== Compliance Recording Example ===")
    
    result = SwaigFunctionResult("This call is being recorded for compliance purposes") \
        .record_call(
            control_id="compliance_rec_001",
            stereo=True,           # High quality stereo
            format="wav",          # Uncompressed for legal requirements
            direction="both",      # Record all parties
            beep=True,            # Legal requirement notification
            input_sensitivity=50.0, # Higher sensitivity
            status_url="https://compliance.company.com/recording-webhook"
        ) \
        .set_metadata({
            "call_type": "compliance",
            "recording_start": "2024-01-01T12:00:00Z",
            "legal_notice_given": True
        })
    
    print(json.dumps(result.to_dict(), indent=2))
    print()


if __name__ == "__main__":
    print("Record Call Virtual Helper Examples\n")
    
    basic_recording_example()
    advanced_recording_example() 
    voicemail_recording_example()
    stop_recording_example()
    customer_service_workflow()
    compliance_recording_example()
    
    print("COMPLETE: All recording examples completed")
    print("\nKey Features Demonstrated:")
    print("- Basic and advanced recording configurations")
    print("- Background recording that doesn't block script execution")
    print("- Customizable audio format, direction, and quality settings")
    print("- Terminator keys for user-controlled recording stop")
    print("- Timeout controls for voicemail systems")
    print("- Webhook integration for recording status updates")
    print("- Method chaining with other actions")
    print("- Complete workflows for customer service and compliance") 