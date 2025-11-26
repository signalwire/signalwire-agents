# SWAIG Actions Reference

This document describes all supported SWAIG actions that can be returned from function calls, their expected JSON parameters, and proposed Python helper methods for the `SwaigFunctionResult` class.

## Core Call Control Actions

### 1. SWML Execution
**Purpose**: Execute a SWML document or transfer to SWML

**JSON Structure**:
```json
{
  "SWML": "<swml_content>",
  "transfer": "true|false"  // optional
}
```

**Parameters**:
- `SWML` (required): String (raw SWML JSON) or Object (structured SWML data)
- `transfer` (optional): String "true"/"false" or Boolean - controls if call exits agent

**Proposed Python Method**:
```python
result.execute_swml(swml_content, transfer=False)
```

**Method Parameters**:
- `swml_content` (required): Flexible input supporting:
  - **String**: Raw SWML JSON text
  - **Dict**: SWML data structure (Python dictionary)
  - **SWML Object**: SignalWire SWML SDK object with `.to_dict()` method
- `transfer` (optional): Boolean - whether call should exit agent after execution (default: False)

**Usage Examples**:
```python
from signalwire.swml import SWML

# 1. Raw SWML string (simple cases)
result.execute_swml('{"version":"1.0.0","sections":{"main":[{"say":"Hello World"}]}}')

# 2. SWML dictionary (programmatic construction)
swml_dict = {
    "version": "1.0.0",
    "sections": {
        "main": [
            {"say": "Please hold while I transfer you"},
            {"connect": {"to": "+15551234567"}}
        ]
    }
}
result.execute_swml(swml_dict, transfer=True)

# 3. SWML SDK object (best developer experience with IDE support)
swml_doc = SWML()
swml_doc.add_application("main", "say", {"text": "Connecting you now"})
swml_doc.add_application("main", "connect", {"to": "support@company.com"})
result.execute_swml(swml_doc, transfer=True)

# 4. Complex SWML with multiple sections
swml_doc = SWML()
swml_doc.add_application("main", "say", {"text": "Please choose an option"})
swml_doc.add_application("main", "gather", {
    "input": {"speech": {"timeout": 5}},
    "action_url": "/handle_choice"
})
result.execute_swml(swml_doc)  # No transfer, return to agent after execution
```

---

### 2. Call Hangup
**Purpose**: Terminate the call

**JSON Structure**:
```json
{
  "hangup": true
}
```

**Parameters**:
- `hangup` (required): Boolean - must be true to hangup

**Proposed Python Method**:
```python
result.hangup()
```

**Usage Examples**:
```python
result.hangup()
```

---

## Call Hold & Flow Control Actions

### 3. Hold Call
**Purpose**: Put the call on hold with optional timeout

**JSON Structure**:
```json
// Simple timeout
{
  "hold": 300
}

// String timeout (parsed)
{
  "hold": "5m"
}

// Object with timeout
{
  "hold": {
    "timeout": 300
  }
}
```

**Parameters**:
- `hold` (required): Number (seconds), String (time format), or Object
  - `timeout` (required if object): Number - timeout in seconds (max 900)

**Proposed Python Method**:
```python
result.hold(timeout=300)
```

**Usage Examples**:
```python
result.hold(60)      # 60 seconds
result.hold("5m")    # 5 minutes (if string parsing supported)
```

---

### 4. Wait for User Input
**Purpose**: Control how agent waits for user input

**JSON Structure**:
```json
{
  "wait_for_user": true|false|<number>|"answer_first"
}
```

**Parameters**:
- `wait_for_user` (required): Boolean, Number (timeout), or String ("answer_first")

**Proposed Python Method**:
```python
result.wait_for_user(enabled=True, timeout=None, answer_first=False)
```

**Usage Examples**:
```python
result.wait_for_user(True)                    # Wait indefinitely
result.wait_for_user(timeout=30)              # Wait 30 seconds
result.wait_for_user(answer_first=True)       # Special "answer_first" mode
```

---

### 5. Stop Agent
**Purpose**: Stop the agent execution

**JSON Structure**:
```json
{
  "stop": true
}
```

**Parameters**:
- `stop` (required): Boolean - must be true to stop

**Proposed Python Method**:
```python
result.stop()
```

**Usage Examples**:
```python
result.stop()
```

---

## Speech & Audio Control Actions

### 6. Say Text
**Purpose**: Make the agent speak specific text

**JSON Structure**:
```json
{
  "say": "<text_to_speak>"
}
```

**Parameters**:
- `say` (required): String - text for agent to speak

**Proposed Python Method**:
```python
result.say(text)
```

**Usage Examples**:
```python
result.say("Hello, how can I help you today?")
```

---

### 7. Play Background Audio
**Purpose**: Play audio file in background

**JSON Structure**:
```json
// Simple filename
{
  "playback_bg": "<filename>"
}

// With options
{
  "playback_bg": {
    "file": "<filename>",
    "wait": true|false  // optional
  }
}
```

**Parameters**:
- `playback_bg` (required): String (filename) or Object
  - `file` (required if object): String - audio filename/path
  - `wait` (optional): Boolean - whether to suppress attention-getting behavior during playback

**Proposed Python Method**:
```python
result.play_background_file(filename, wait=False)
```

**Usage Examples**:
```python
# Play background audio, AI will try to get user attention per attention timeout
result.play_background_file("hold_music.wav")

# Play background audio, AI won't try to get user attention while playing
result.play_background_file("announcement.mp3", wait=True)
```

**Behavior Notes**:
- Audio plays in background while AI can still hear and respond to user
- `wait=False` (default): AI will attempt to get user's attention according to attention timeout
- `wait=True`: AI suppresses attention-getting behavior during playback

---

### 8. Stop Background Audio
**Purpose**: Stop currently playing background audio

**JSON Structure**:
```json
{
  "stop_playback_bg": true
}
```

**Parameters**:
- `stop_playback_bg` (required): Boolean - must be true to stop playback

**Proposed Python Method**:
```python
result.stop_background_file()
```

**Usage Examples**:
```python
result.stop_background_file()
```

---

## Speech Recognition Settings

### 9. End of Speech Timeout
**Purpose**: Adjust speech detection timeout - milliseconds of silence after speaking has been detected to finalize speech recognition

**JSON Structure**:
```json
{
  "end_of_speech_timeout": 1500
}
```

**Parameters**:
- `end_of_speech_timeout` (required): Number or String - timeout in milliseconds of silence after speech detection to finalize recognition

**Proposed Python Method**:
```python
result.set_end_of_speech_timeout(milliseconds)
```

**Usage Examples**:
```python
result.set_end_of_speech_timeout(2000)  # Wait 2 seconds of silence to finalize speech
```

**Behavior Notes**:
- Mirrors the agent startup parameter of the same name
- Controls how long to wait for silence after speech is detected before finalizing recognition
- Higher values = more patient waiting for user to finish speaking
- Lower values = faster response but may cut off slow speakers

---

### 10. Speech Event Timeout
**Purpose**: Adjust speech event timeout - milliseconds since last speech detection event to finalize recognition

**JSON Structure**:
```json
{
  "speech_event_timeout": 5000
}
```

**Parameters**:
- `speech_event_timeout` (required): Number or String - timeout in milliseconds since last speech detection event

**Proposed Python Method**:
```python
result.set_speech_event_timeout(milliseconds)
```

**Usage Examples**:
```python
result.set_speech_event_timeout(3000)  # Finalize after 3 seconds since last speech event
```

**Behavior Notes**:
- Mirrors the agent startup parameter of the same name  
- Works better in noisy environments than `end_of_speech_timeout`
- Doesn't require silence - just lack of new speech detection events
- Useful when background noise prevents true silence detection
- More robust for real-world call environments

---

## Data Management Actions

### 11. Set Global Data
**Purpose**: Update global agent data variables

**JSON Structure**:
```json
{
  "set_global_data": {
    "key1": "value1",
    "key2": "value2"
  }
}
```

**Parameters**:
- `set_global_data` (required): Object - key-value pairs to set/update

**Proposed Python Method**:
```python
result.update_global_data(data_dict)  # Already implemented
```

**Usage Examples**:
```python
result.update_global_data({"user_name": "John", "step": 2})
```

---

### 12. Unset Global Data
**Purpose**: Remove global agent data variables

**JSON Structure**:
```json
// Single key
{
  "unset_global_data": "key_name"
}

// Multiple keys
{
  "unset_global_data": ["key1", "key2"]
}
```

**Parameters**:
- `unset_global_data` (required): String (single key) or Array (multiple keys)

**Proposed Python Method**:
```python
result.remove_global_data(keys)
```

**Usage Examples**:
```python
result.remove_global_data("temporary_data")
result.remove_global_data(["step", "temp_value"])
```

---

### 13. Set Metadata
**Purpose**: Set metadata scoped to current function's meta_data_token

**JSON Structure**:
```json
{
  "set_meta_data": {
    "key1": "value1",
    "key2": "value2"
  }
}
```

**Parameters**:
- `set_meta_data` (required): Object - key-value pairs for metadata

**Proposed Python Method**:
```python
result.set_metadata(data_dict)
```

**Usage Examples**:
```python
result.set_metadata({"session_id": "abc123", "user_tier": "premium"})
```

**Scoping Behavior**:
- Metadata is scoped based on the `meta_data_token` parameter in the SWAIG function prototype
- If no `meta_data_token` is provided: scope defaults to function name/URL
- If `meta_data_token` is provided: functions with the same token share metadata scope
- This allows grouping related functions to share metadata

---

### 14. Unset Metadata
**Purpose**: Remove metadata from current function's meta_data_token scope

**JSON Structure**:
```json
// Single key
{
  "unset_meta_data": "key_name"
}

// Multiple keys
{
  "unset_meta_data": ["key1", "key2"]
}
```

**Parameters**:
- `unset_meta_data` (required): String (single key) or Array (multiple keys)

**Proposed Python Method**:
```python
result.remove_metadata(keys)
```

**Usage Examples**:
```python
result.remove_metadata("temp_session_data")
result.remove_metadata(["cache_key", "temp_flag"])
```

**Scoping Behavior**:
- Removes metadata from the same scope as determined by `meta_data_token` in function prototype
- Scope follows same rules as `set_meta_data` action

---

## Function & Behavior Control Actions

### 15. Toggle Functions
**Purpose**: Enable/disable specific SWAIG functions

**JSON Structure**:
```json
{
  "toggle_functions": [
    {
      "function": "function_name",
      "active": true|false
    }
  ]
}
```

**Parameters**:
- `toggle_functions` (required): Array of objects
  - `function` (required): String - function name to toggle
  - `active` (required): Boolean - whether function should be active

**Proposed Python Method**:
```python
result.toggle_functions(function_toggles)
```

**Usage Examples**:
```python
result.toggle_functions([
    {"function": "transfer_call", "active": False},
    {"function": "lookup_info", "active": True}
])
```

---

### 16. Functions on Timeout
**Purpose**: Enable function calls on speaker timeout

**JSON Structure**:
```json
{
  "functions_on_speaker_timeout": true|false
}
```

**Parameters**:
- `functions_on_speaker_timeout` (required): Boolean

**Proposed Python Method**:
```python
result.enable_functions_on_timeout(enabled=True)
```

**Usage Examples**:
```python
result.enable_functions_on_timeout(True)
```

---

### 17. Extensive Data
**Purpose**: Send full data to LLM for this turn only, then use smaller replacement in subsequent turns

**JSON Structure**:
```json
{
  "extensive_data": true|false
}
```

**Parameters**:
- `extensive_data` (required): Boolean - whether to send extensive data this turn only

**Proposed Python Method**:
```python
result.enable_extensive_data(enabled=True)
```

**Usage Examples**:
```python
result.enable_extensive_data(True)
```

**Behavior Notes**:
- When `true`: Sends full/detailed data to LLM for current response only
- After this turn: System automatically replaces extensive data with smaller version  
- Useful for providing rich context without ongoing cost/latency impact
- Helps optimize token usage while maintaining context quality when needed

---

## Agent Settings & Configuration Actions

### 18. Update Settings
**Purpose**: Update agent runtime settings

**JSON Structure**:
```json
{
  "settings": {
    "frequency-penalty": -1.5,
    "presence-penalty": 0.5,
    "max-tokens": 1024,
    "top-p": 0.9,
    "confidence": 0.8,
    "barge-confidence": 0.7,
    "temperature": 0.7
  }
}
```

**Parameters**:
- `settings` (required): Object - settings key-value pairs with the following supported options:
  - `frequency-penalty` (optional): Float (-2.0 to 2.0) - Penalizes repeated tokens
  - `presence-penalty` (optional): Float (-2.0 to 2.0) - Penalizes tokens based on presence
  - `max-tokens` (optional): Integer (0 to 4096) - Maximum tokens in response
  - `top-p` (optional): Float (0.0 to 1.0) - Nucleus sampling parameter
  - `confidence` (optional): Float (0.0 to 1.0) - Speech recognition confidence threshold
  - `barge-confidence` (optional): Float (0.0 to 1.0) - Confidence threshold for barge-in
  - `temperature` (optional): Float (0.0 to 2.0, clamped to 1.5) - Randomness in responses

**Proposed Python Method**:
```python
result.update_settings(settings_dict)
```

**Usage Examples**:
```python
# Update AI model parameters
result.update_settings({
    "temperature": 0.7,
    "max-tokens": 2048,
    "frequency-penalty": -0.5
})

# Update speech recognition settings
result.update_settings({
    "confidence": 0.8,
    "barge-confidence": 0.7
})

# Update all supported settings
result.update_settings({
    "frequency-penalty": 1.0,
    "presence-penalty": 0.5,
    "max-tokens": 1024,
    "top-p": 0.9,
    "confidence": 0.75,
    "barge-confidence": 0.6,
    "temperature": 0.8
})
```

---

### 19. Context Switch
**Purpose**: Change agent context/prompt during conversation

**JSON Structure**:
```json
// Simple prompt change
{
  "context_switch": "<new_system_prompt>"
}

// Advanced context switching
{
  "context_switch": {
    "system_prompt": "<new_system_prompt>",
    "user_prompt": "<user_message>",
    "system_pom": "<pom_content>",
    "user_pom": "<user_pom_content>",
    "consolidate": true|false,
    "full_reset": true|false
  }
}
```

**Parameters**:
- `context_switch` (required): String (simple prompt) or Object (advanced)
  - `system_prompt` (optional): String - new system prompt
  - `user_prompt` (optional): String - user message to add
  - `system_pom` (optional): String - POM-based system prompt
  - `user_pom` (optional): String - POM-based user prompt  
  - `consolidate` (optional): Boolean - summarize existing conversation
  - `full_reset` (optional): Boolean - complete context reset

**Proposed Python Method**:
```python
result.switch_context(system_prompt=None, user_prompt=None, consolidate=False, full_reset=False)
```

**Usage Examples**:
```python
# Simple context switch
result.switch_context("You are now a technical support agent")

# Advanced context switch
result.switch_context(
    system_prompt="You are a billing specialist",
    user_prompt="The user needs help with their invoice",
    consolidate=True
)
```

---

### 20. User Input Simulation
**Purpose**: Queue simulated user input

**JSON Structure**:
```json
{
  "user_input": "<simulated_input_text>"
}
```

**Parameters**:
- `user_input` (required): String - text to simulate as user input

**Proposed Python Method**:
```python
result.simulate_user_input(text)
```

**Usage Examples**:
```python
result.simulate_user_input("Yes, I'd like to speak to billing")
```

---

## Notes

- **Permissions**: Some actions require specific permissions (e.g., `swaig_allow_swml`, `swaig_allow_settings`, `swaig_set_global_data`)
- **Method Chaining**: All proposed methods should return `self` to enable method chaining
- **Type Safety**: Consider using type hints and validation for all parameters
- **Backward Compatibility**: Keep existing `add_action()` and `add_actions()` methods for custom use cases 