# SwaigFunctionResult Methods Reference

This document provides a complete reference for all methods available in the `SwaigFunctionResult` class. These methods provide convenient abstractions for SWAIG actions, eliminating the need to manually construct action JSON objects.

## Core Methods

### Basic Construction & Control

#### `__init__(response=None, post_process=False)`
Creates a new result object with optional response text and post-processing behavior.

```python
result = SwaigFunctionResult("Hello, I'll help you with that")
result = SwaigFunctionResult("Processing request...", post_process=True)
```

#### `set_response(response)`
Sets or updates the response text that the AI will speak.

```python
result.set_response("I've updated your information")
```

#### `set_post_process(post_process)`
Controls whether AI gets one more turn before executing actions.

```python
result.set_post_process(True)  # AI speaks response before executing actions
result.set_post_process(False)  # Actions execute immediately
```

---

## Action Methods

### Call Control Actions

#### `execute_swml(swml_content, transfer=False)`
Execute SWML content with flexible input support and optional transfer behavior.

```python
# Raw SWML string
result.execute_swml('{"version":"1.0.0","sections":{"main":[{"say":"Hello"}]}}')

# SWML dictionary
swml_dict = {"version": "1.0.0", "sections": {"main": [{"say": "Hello"}]}}
result.execute_swml(swml_dict, transfer=True)

# SWML SDK object
from signalwire.swml import SWML
swml_doc = SWML()
swml_doc.add_application("main", "say", {"text": "Connecting now"})
result.execute_swml(swml_doc)
```

#### **[IMPLEMENTED]** - Transfer/connect call to another destination using SWML.

```python
result.connect("+15551234567", final=True)  # Permanent transfer
result.connect("support@company.com", final=False, from_addr="+15559876543")  # Temporary transfer
```

#### `send_sms(to_number, from_number, body=None, media=None, tags=None, region=None)`
**[HELPER METHOD]** - Send SMS message to PSTN phone number using SWML.

```python
# Simple text message
result.send_sms(
    to_number="+15551234567",
    from_number="+15559876543", 
    body="Your order has been confirmed!"
)

# Media message with images
result.send_sms(
    to_number="+15551234567",
    from_number="+15559876543",
    media=["https://example.com/receipt.jpg", "https://example.com/map.png"]
)

# Full featured message with tags and region
result.send_sms(
    to_number="+15551234567",
    from_number="+15559876543",
    body="Order update with receipt attached",
    media=["https://example.com/receipt.pdf"],
    tags=["order", "confirmation", "customer"],
    region="us"
)
```

**Parameters:**
- `to_number` (required): Phone number in E.164 format to send to
- `from_number` (required): Phone number in E.164 format to send from
- `body` (optional): Message text (required if no media)
- `media` (optional): Array of URLs to send (required if no body)
- `tags` (optional): Array of tags for UI searching
- `region` (optional): Region to originate message from

**Variables Set:**
- `send_sms_result`: "success" or "failed"

#### `pay(payment_connector_url, **options)`
**[HELPER METHOD]** - Process payments using SWML pay action with extensive customization.

```python
# Simple payment setup
result.pay(
    payment_connector_url="https://api.example.com/accept-payment",
    charge_amount="10.99",
    description="Monthly subscription"
)

# Advanced payment with custom prompts
from signalwire_agents.core.function_result import SwaigFunctionResult

# Create custom prompts
welcome_actions = [
    SwaigFunctionResult.create_payment_action("Say", "Welcome to our payment system"),
    SwaigFunctionResult.create_payment_action("Say", "Please enter your credit card number")
]
card_prompt = SwaigFunctionResult.create_payment_prompt("payment-card-number", welcome_actions)

error_actions = [
    SwaigFunctionResult.create_payment_action("Say", "Invalid card number, please try again")
]
error_prompt = SwaigFunctionResult.create_payment_prompt(
    "payment-card-number", 
    error_actions, 
    error_type="invalid-card-number timeout"
)

# Create payment parameters
params = [
    SwaigFunctionResult.create_payment_parameter("customer_id", "12345"),
    SwaigFunctionResult.create_payment_parameter("order_id", "ORD-789")
]

# Full payment configuration
result.pay(
    payment_connector_url="https://api.example.com/accept-payment",
    status_url="https://api.example.com/payment-status",
    timeout=10,
    max_attempts=3,
    security_code=True,
    postal_code=False,
    token_type="one-time",
    charge_amount="25.50",
    currency="usd",
    language="en-US",
    voice="polly.Sally",
    description="Premium service upgrade",
    valid_card_types="visa mastercard amex",
    parameters=params,
    prompts=[card_prompt, error_prompt]
)
```

**Core Parameters:**
- `payment_connector_url` (required): URL to process payment requests
- `input_method`: "dtmf" or "voice" (default: "dtmf")
- `payment_method`: "credit-card" (default: "credit-card")
- `timeout`: Seconds to wait for input (default: 5)
- `max_attempts`: Number of retry attempts (default: 1)

**Security & Validation:**
- `security_code`: Prompt for CVV (default: True)
- `postal_code`: Prompt for postal code or provide known code (default: True)
- `min_postal_code_length`: Minimum postal code digits (default: 0)
- `valid_card_types`: Space-separated card types (default: "visa mastercard amex")

**Payment Configuration:**
- `token_type`: "one-time" or "reusable" (default: "reusable")
- `charge_amount`: Amount as decimal string
- `currency`: Currency code (default: "usd")
- `description`: Payment description

**Customization:**
- `language`: Prompt language (default: "en-US")
- `voice`: TTS voice (default: "woman")
- `status_url`: URL for status notifications
- `parameters`: Additional name/value pairs for connector
- `prompts`: Custom prompt configurations

**Helper Methods for Payment Setup:**
```python
# Create payment action
action = SwaigFunctionResult.create_payment_action("Say", "Enter card number")

# Create payment prompt
prompt = SwaigFunctionResult.create_payment_prompt(
    "payment-card-number", 
    [action], 
    error_type="invalid-card-number"
)

# Create payment parameter
param = SwaigFunctionResult.create_payment_parameter("customer_id", "12345")
```

**Variables Set:**
- `pay_result`: "success", "too-many-failed-attempts", "payment-connector-error", etc.
- `pay_payment_results`: JSON with payment details including tokens and card info

#### `record_call(control_id=None, stereo=False, format="wav", direction="both", **options)`
**[HELPER METHOD]** - Start background call recording using SWML.

Unlike foreground recording, the script continues executing while recording happens in the background.

```python
# Simple background recording
result.record_call()

# Recording with custom settings
result.record_call(
    control_id="support_call_001",
    stereo=True,
    format="mp3",
    direction="both",
    max_length=300  # 5 minutes max
)

# Recording with terminator and status webhook
result.record_call(
    control_id="customer_voicemail", 
    format="wav",
    direction="speak",           # Only record customer voice
    terminators="#",             # Stop on '#' press
    beep=True,                   # Play beep before recording
    initial_timeout=4.0,         # Wait 4 seconds for speech
    end_silence_timeout=3.0,     # Stop after 3 seconds of silence
    status_url="https://api.example.com/recording-status"
)
```

**Core Parameters:**
- `control_id` (optional): Identifier for this recording (for use with stop_record_call)
- `stereo`: Record in stereo (default: False)
- `format`: "wav" or "mp3" (default: "wav")
- `direction`: "speak", "listen", or "both" (default: "both")

**Control Options:**
- `terminators`: Digits that stop recording when pressed
- `beep`: Play beep before recording (default: False)
- `max_length`: Maximum recording length in seconds

**Timing Options:**
- `input_sensitivity`: Input sensitivity (default: 44.0)
- `initial_timeout`: Time to wait for speech start (default: 0.0)
- `end_silence_timeout`: Time to wait in silence before ending (default: 0.0)

**Webhook Options:**
- `status_url`: URL to send recording status events to

**Variables Set:**
- `record_call_result`: "success" or "failed"
- `record_call_url`: URL of recorded file (when recording completes)

#### `stop_record_call(control_id=None)`
**[HELPER METHOD]** - Stop an active background call recording using SWML.

```python
# Stop the most recent recording
result.stop_record_call()

# Stop specific recording by ID
result.stop_record_call("support_call_001")

# Chain to stop recording and provide feedback
result.stop_record_call("customer_voicemail") \
      .say("Thank you, your message has been recorded")
```

**Parameters:**
- `control_id` (optional): Identifier for recording to stop. If not provided, stops the most recent recording.

**Variables Set:**
- `stop_record_call_result`: "success" or "failed"

#### `join_room(name)`
**[HELPER METHOD]** - Join a RELAY room using SWML.

RELAY rooms enable multi-party communication and collaboration features.

```python
# Join a conference room
result.join_room("support_team_room")

# Join customer meeting room
result.join_room("customer_meeting_001") \
      .say("Welcome to the customer meeting room")

# Join room and set metadata
result.join_room("sales_conference") \
      .set_metadata({"participant_role": "moderator", "join_time": "2024-01-01T12:00:00Z"})
```

**Parameters:**
- `name` (required): The name of the room to join

**Variables Set:**
- `join_room_result`: "success" or "failed"

#### `sip_refer(to_uri)`
**[HELPER METHOD]** - Send SIP REFER for call transfer using SWML.

SIP REFER is used for call transfer in SIP environments, allowing one endpoint to request another to initiate a new connection.

```python
# Basic SIP refer to transfer call
result.sip_refer("sip:support@company.com")

# Transfer to specific SIP address with domain
result.sip_refer("sip:agent123@pbx.company.com:5060")

# Chain with announcement
result.say("Transferring your call to our specialist") \
      .sip_refer("sip:specialist@company.com")
```

**Parameters:**
- `to_uri` (required): The SIP URI to send the REFER to

**Variables Set:**
- `sip_refer_result`: "success" or "failed"

#### `join_conference(name, **options)`
**[HELPER METHOD]** - Join an ad-hoc audio conference with RELAY and CXML calls using SWML.

Provides extensive configuration options for conference call management and recording.

```python
# Simple conference join
result.join_conference("my_conference")

# Basic conference with recording
result.join_conference(
    name="daily_standup",
    record="record-from-start",
    max_participants=10
)

# Advanced conference with callbacks and coaching
result.join_conference(
    name="customer_support_conf", 
    muted=False,
    beep="onEnter",
    start_on_enter=True,
    end_on_exit=False,
    max_participants=50,
    record="record-from-start",
    region="us-east",
    trim="trim-silence",
    status_callback="https://api.company.com/conference-events",
    status_callback_event="start end join leave",
    recording_status_callback="https://api.company.com/recording-events"
)

# Chain with other actions
result.say("Joining you to the team conference") \
      .join_conference("team_meeting") \
      .set_metadata({"meeting_type": "team_sync", "participant_role": "attendee"})
```

**Core Parameters:**
- `name` (required): Name of conference to join
- `muted`: Join muted (default: False)
- `beep`: Beep configuration - "true", "false", "onEnter", "onExit" (default: "true")
- `start_on_enter`: Conference starts when this participant enters (default: True)
- `end_on_exit`: Conference ends when this participant exits (default: False)

**Capacity & Region:**
- `max_participants`: Maximum participants <= 250 (default: 250)
- `region`: Conference region for optimization
- `wait_url`: SWML URL for custom hold music

**Recording Options:**
- `record`: "do-not-record" or "record-from-start" (default: "do-not-record")
- `trim`: "trim-silence" or "do-not-trim" (default: "trim-silence")
- `recording_status_callback`: URL for recording status events
- `recording_status_callback_method`: "GET" or "POST" (default: "POST")
- `recording_status_callback_event`: "in-progress completed absent" (default: "completed")

**Status & Coaching:**
- `coach`: SWML Call ID or CXML CallSid for coaching features
- `status_callback`: URL for conference status events
- `status_callback_method`: "GET" or "POST" (default: "POST")
- `status_callback_event`: Events to report - "start end join leave mute hold modify speaker announcement"

**Control Flow:**
- `result`: Switch on return_value (object {} or array [] for conditional logic)

**Variables Set:**
- `join_conference_result`: "completed", "answered", "no-answer", "failed", or "canceled"
- `return_value`: Same as `join_conference_result`

#### `tap(uri, **options)`
**[HELPER METHOD]** - Start background call tap using SWML.

Media is streamed over Websocket or RTP to customer controlled URI for real-time monitoring and analysis.

```python
# Simple WebSocket tap
result.tap("wss://example.com/tap")

# RTP tap with custom settings
result.tap(
    uri="rtp://192.168.1.100:5004",
    control_id="monitoring_tap_001",
    direction="both",
    codec="PCMA",
    rtp_ptime=30
)

# Advanced tap with status callbacks
result.tap(
    uri="wss://monitoring.company.com/audio-stream",
    control_id="compliance_tap",
    direction="speak",  # Only what the party says
    status_url="https://api.company.com/tap-status"
) \
.set_metadata({"tap_purpose": "compliance", "session_id": "sess_123"})
```

**Core Parameters:**
- `uri` (required): Destination of tap media stream
  - WebSocket: `ws://example.com` or `wss://example.com`
  - RTP: `rtp://IP:port`
- `control_id`: Identifier for this tap to use with stop_tap (optional, auto-generated if not provided)

**Audio Configuration:**
- `direction`: Audio direction to tap (default: "both")
  - `"speak"`: What party says
  - `"hear"`: What party hears
  - `"both"`: What party hears and says
- `codec`: Codec for tap stream - "PCMU" or "PCMA" (default: "PCMU")
- `rtp_ptime`: RTP packetization time in milliseconds (default: 20)

**Status & Monitoring:**
- `status_url`: URL for tap status change requests

**Variables Set:**
- `tap_uri`: Destination URI of the newly started tap
- `tap_result`: "success" or "failed"
- `tap_control_id`: Control ID of this tap
- `tap_rtp_src_addr`: If RTP, source address of the tap stream
- `tap_rtp_src_port`: If RTP, source port of the tap stream
- `tap_ptime`: Packetization time of the tap stream
- `tap_codec`: Codec in the tap stream
- `tap_rate`: Sample rate in the tap stream

#### `stop_tap(control_id=None)`
**[HELPER METHOD]** - Stop an active tap stream using SWML.

```python
# Stop the most recent tap
result.stop_tap()

# Stop specific tap by ID
result.stop_tap("monitoring_tap_001")

# Chain to stop tap and provide feedback
result.stop_tap("compliance_tap") \
      .say("Audio monitoring has been stopped") \
      .update_global_data({"tap_active": False})
```

**Parameters:**
- `control_id` (optional): ID of the tap to stop. If not set, the last tap started will be stopped.

**Variables Set:**
- `stop_tap_result`: "success" or "failed"

#### `hangup()`
Terminate the call immediately.

```python
result.hangup()
```

---

### Call Flow Control

#### `hold(timeout=300)`
Put call on hold with timeout (max 900 seconds).

```python
result.hold(60)    # Hold for 1 minute
result.hold(600)   # Hold for 10 minutes
```

#### `wait_for_user(enabled=None, timeout=None, answer_first=False)`
Control how agent waits for user input with flexible parameters.

```python
result.wait_for_user(True)                    # Wait indefinitely
result.wait_for_user(timeout=30)              # Wait 30 seconds
result.wait_for_user(answer_first=True)       # Special answer_first mode
result.wait_for_user(False)                   # Disable waiting
```

#### `stop()`
Stop agent execution completely.

```python
result.stop()
```

---

### Speech & Audio Control

#### `say(text)`
Make the agent speak specific text immediately.

```python
result.say("Please hold while I look that up for you")
```

#### `play_background_file(filename, wait=False)`
Play audio file in background with attention control.

```python
result.play_background_file("hold_music.wav")                    # AI tries to get attention
result.play_background_file("announcement.mp3", wait=True)       # AI suppresses attention
```

#### `stop_background_file()`
Stop currently playing background audio.

```python
result.stop_background_file()
```

---

### Speech Recognition Settings

#### `set_end_of_speech_timeout(milliseconds)`
Set silence timeout after speech detection for finalizing recognition.

```python
result.set_end_of_speech_timeout(2000)  # 2 seconds of silence
```

#### `set_speech_event_timeout(milliseconds)`
Set timeout since last speech event - better for noisy environments.

```python
result.set_speech_event_timeout(3000)  # 3 seconds since last speech event
```

---

### Data Management

#### `update_global_data(data)`
**[IMPLEMENTED]** - Update global agent data variables.

```python
result.update_global_data({"user_name": "John", "step": 2})
```

#### `remove_global_data(keys)`
Remove global data variables by key(s).

```python
result.remove_global_data("temporary_data")           # Single key
result.remove_global_data(["step", "temp_value"])     # Multiple keys
```

#### `set_metadata(data)`
Set metadata scoped to current function's meta_data_token.

```python
result.set_metadata({"session_id": "abc123", "user_tier": "premium"})
```

#### `remove_metadata(keys)`
Remove metadata from current function's scope.

```python
result.remove_metadata("temp_session_data")           # Single key  
result.remove_metadata(["cache_key", "temp_flag"])    # Multiple keys
```

---

### Function & Behavior Control

#### `toggle_functions(function_toggles)`
Enable/disable specific SWAIG functions dynamically.

```python
result.toggle_functions([
    {"function": "transfer_call", "active": False},
    {"function": "lookup_info", "active": True}
])
```

#### `enable_functions_on_timeout(enabled=True)`
Control whether functions can be called on speaker timeout.

```python
result.enable_functions_on_timeout(True)
result.enable_functions_on_timeout(False)
```

#### `enable_extensive_data(enabled=True)`
Send full data to LLM for this turn only, then use smaller replacement.

```python
result.enable_extensive_data(True)   # Send extensive data this turn
result.enable_extensive_data(False)  # Use normal data
```

#### `replace_in_history(text=True)`
Remove or replace the tool_call + tool_result pair from the LLM's conversation history after the first send. This is useful when a function call is an implementation detail that would confuse the model if it remained visible in context.

When called with a string, the tool_call/tool_result pair is replaced with an assistant message containing that text. When called with `True`, the pair is removed entirely — the LLM will never see that the function was called.

```python
# Remove entirely — LLM won't see this function was called
result = SwaigFunctionResult("Done.")
result.replace_in_history()

# Replace with a friendly assistant message instead of tool artifacts
result = SwaigFunctionResult("Profile saved.")
result.replace_in_history("I've saved your profile information.")

# Practical example: data collection function that shouldn't clutter history
@agent.tool(name="save_answer", description="Save the user's answer")
def save_answer(args, raw_data):
    answer = args.get("answer")
    result = SwaigFunctionResult(f"Answer recorded: {answer}")
    result.replace_in_history()  # Keep history clean
    return result
```

**When to use:**
- Functions that are implementation details (saving data, logging, internal state changes)
- Functions called frequently that would bloat conversation history
- Situations where tool artifacts confuse the model's reasoning (especially with reasoning models at low effort settings)

**Note:** For structured data collection, consider using [gather_info mode](contexts_guide.md#gather-info-mode) instead, which produces zero tool artifacts by design and doesn't require `replace_in_history`.

---

### Agent Settings & Configuration

#### `update_settings(settings)`
Update agent runtime settings with validation.

```python
# AI model settings
result.update_settings({
    "temperature": 0.7,
    "max-tokens": 2048,
    "frequency-penalty": -0.5
})

# Speech recognition settings  
result.update_settings({
    "confidence": 0.8,
    "barge-confidence": 0.7
})
```

**Supported Settings:**
- `frequency-penalty`: Float (-2.0 to 2.0)
- `presence-penalty`: Float (-2.0 to 2.0) 
- `max-tokens`: Integer (0 to 4096)
- `top-p`: Float (0.0 to 1.0)
- `confidence`: Float (0.0 to 1.0)
- `barge-confidence`: Float (0.0 to 1.0)
- `temperature`: Float (0.0 to 2.0, clamped to 1.5)

#### `switch_context(system_prompt=None, user_prompt=None, consolidate=False, full_reset=False)`
Change agent context/prompt during conversation.

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

#### `simulate_user_input(text)`
Queue simulated user input for testing or flow control.

```python
result.simulate_user_input("Yes, I'd like to speak to billing")
```

---

## Low-Level Methods

### Manual Action Construction

#### `add_action(name, data)`
Add a single action manually (for custom actions not covered by helper methods).

```python
result.add_action("custom_action", {"param": "value"})
```

#### `add_actions(actions)`
Add multiple actions at once.

```python
result.add_actions([
    {"say": "Hello"},
    {"hold": 300}
])
```

### Output Generation

#### `to_dict()`
Convert result to dictionary format for JSON serialization.

```python
result_dict = result.to_dict()
# Returns: {"response": "...", "action": [...], "post_process": true/false}
```

---

## Method Chaining

All methods return `self` to enable fluent method chaining:

```python
result = SwaigFunctionResult("Processing your request", post_process=True) \
    .update_global_data({"status": "processing"}) \
    .play_background_file("processing.wav", wait=True) \
    .set_end_of_speech_timeout(2500)

# Complex chaining example
result = SwaigFunctionResult("Let me transfer you to billing") \
    .set_metadata({"transfer_reason": "billing_inquiry"}) \
    .update_global_data({"last_action": "transfer_to_billing"}) \
    .connect("+15551234567", final=True)
```

---

## Implementation Status

- **[IMPLEMENTED]**: `connect()`, `update_global_data()`, and all methods listed above
- **[HELPER METHODS]**: `send_sms()`, `pay()`, `record_call()`, `stop_record_call()`, `join_room()`, `sip_refer()`, `join_conference()`, `tap()`, `stop_tap()` - Additional convenience methods that generate SWML
- **[UTILITY METHODS]**: `create_payment_prompt()`, `create_payment_action()`, `create_payment_parameter()`
- **[EXTENSIBLE]**: Additional convenience methods for common SWML patterns

## Best Practices

1. **Use post_process=True** when you want the AI to speak before executing actions
2. **Chain methods** for cleaner, more readable code
3. **Use specific methods** instead of manual action construction when available
4. **Handle errors gracefully** - methods may raise TypeError for invalid inputs
5. **Validate settings** - update_settings() relies on server-side validation 

### Final State
The framework now includes **10 virtual helpers total**:
1. connect() - Call transfer/connect
2. send_sms() - SMS messaging
3. pay() - Payment processing
4. record_call() - Start background recording
5. stop_record_call() - Stop background recording
6. join_room() - Join RELAY room
7. sip_refer() - SIP REFER transfer
8. join_conference() - Join audio conference with extensive options
9. tap() - Start background call tap for monitoring
10. stop_tap() - Stop background call tap

---

## Post Data Reference

The `post_data` object is the JSON payload sent to SWAIG function handlers. Its structure differs between webhook functions and DataMap functions.

### Base Keys (All Functions)

| Key | Type | Description |
|-----|------|-------------|
| `app_name` | string | Name of the AI application |
| `function` | string | Name of the SWAIG function being called |
| `call_id` | string | Unique UUID of the current call session |
| `ai_session_id` | string | Unique UUID of the AI session |
| `caller_id_name` | string | Caller ID name (if available) |
| `caller_id_num` | string | Caller ID number (if available) |
| `channel_active` | boolean | Whether the channel is currently up |
| `channel_offhook` | boolean | Whether the channel is off-hook |
| `channel_ready` | boolean | Whether the AI session is ready |
| `argument` | object | Parsed function arguments |
| `argument_desc` | object | Function argument schema/description |
| `purpose` | string | Description of what the function does |
| `content_type` | string | Always `"text/swaig"` |
| `version` | string | SWAIG protocol version |
| `global_data` | object | Application-level global data (when set) |
| `conversation_id` | string | Conversation identifier (when tracking enabled) |
| `project_id` | string | SignalWire project ID |
| `space_id` | string | SignalWire space ID |

### Webhook-Only Keys

These keys are only present for traditional webhook SWAIG functions:

| Key | Type | Description | Present When |
|-----|------|-------------|--------------|
| `meta_data_token` | string | Token for metadata access | Function has metadata token |
| `meta_data` | object | Function-level metadata | Function has metadata token |
| `SWMLVars` | object | SWML variables | `swaig_post_swml_vars` parameter set |
| `SWMLCall` | object | SWML call state | `swaig_post_swml_vars` parameter set |
| `call_log` | array | Processed conversation history | `swaig_post_conversation` is true |
| `raw_call_log` | array | Raw conversation history | `swaig_post_conversation` is true |

**Metadata scoping**: Functions sharing the same `meta_data_token` share access to the same metadata. If no token is specified, scope defaults to function name/URL.

**Conversation history**: `call_log` may shrink after conversation resets (consolidation), while `raw_call_log` preserves full history. Both include timing data (latency, utterance_latency, audio_latency).

### DataMap-Specific Keys

| Key | Type | Description |
|-----|------|-------------|
| `prompt_vars` | object | Template variables built from call context, SWML vars, and global_data |
| `args` | object | First parsed argument object for easy template access |
| `input` | object | Copy of entire post_data for variable expansion |

### prompt_vars Contents

| Key | Source | Description |
|-----|--------|-------------|
| `call_direction` | Call direction | `"inbound"` or `"outbound"` |
| `caller_id_name` | Channel variable | Caller's name |
| `caller_id_number` | Channel variable | Caller's number |
| `local_date` | System time | Current date in local timezone |
| `local_time` | System time | Current time with timezone |
| `time_of_day` | Derived from hour | `"morning"`, `"afternoon"`, or `"evening"` |
| `supported_languages` | App config | Available languages |
| `default_language` | App config | Primary language |

All keys from `global_data` are also merged into `prompt_vars`, with global_data taking precedence.

### SWML Parameters Controlling post_data

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `swaig_allow_swml` | boolean | true | Allow functions to execute SWML actions |
| `swaig_allow_settings` | boolean | true | Allow functions to modify AI settings |
| `swaig_post_conversation` | boolean | false | Include conversation history in post_data |
| `swaig_set_global_data` | boolean | true | Allow functions to modify global_data |
| `swaig_post_swml_vars` | boolean/array | false | Include SWML variables in post_data |

### Variable Expansion in DataMap

DataMap processing supports template expansion with access to:

- Nested object access via dot notation: `${user.name}`
- Array access: `${items[0].value}`
- Encoding functions: `${enc:url:variable}`
- Built-in functions: `@{strftime %Y-%m-%d}`, `@{expr 2+2}`

---

## Related Documentation

- **[API Reference](api_reference.md)** - Complete AgentBase and SwaigFunctionResult API reference
- **[Contexts Guide](contexts_guide.md)** - Using `swml_change_context()` and `swml_change_step()`
- **[DataMap Guide](datamap_guide.md)** - Using SwaigFunctionResult with DataMap outputs
- **[Agent Guide](agent_guide.md)** - General agent development guide

### Example Files

- `examples/simple_agent.py` - Basic SWAIG function usage
- `examples/swaig_features_agent.py` - Advanced SWAIG features with fillers
- `examples/record_call_example.py` - Recording and tapping calls
- `examples/room_and_sip_example.py` - Room joining and SIP transfer