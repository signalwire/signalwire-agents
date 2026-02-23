# SignalWire AI Agents SDK - Complete API Reference

This document provides a comprehensive reference for all public APIs in the SignalWire AI Agents SDK.

## Table of Contents

1. [AgentBase Class](#agentbase-class) - Core agent functionality
2. [SwaigFunctionResult Class](#swaigfunctionresult-class) - SWAIG (SignalWire AI Gateway) function response handling
3. [DataMap Class](#datamap-class) - Serverless API tools that execute on SignalWire's servers
4. [Context System](#context-system) - Structured workflows
5. [State Management](#state-management) - Persistent state
6. [Skills System](#skills-system) - Modular capabilities
7. [Utility Classes](#utility-classes) - Supporting classes

---

## AgentBase Class

The `AgentBase` class is the foundation for creating AI agents. It extends `SWMLService` (the base class for generating SWML -- SignalWire Markup Language -- documents) and provides comprehensive functionality for building conversational AI agents.

### Constructor

```python
AgentBase(
    name: str,
    route: str = "/",
    host: str = "0.0.0.0",
    port: int = 3000,
    basic_auth: Optional[Tuple[str, str]] = None,
    use_pom: bool = True,
    token_expiry_secs: int = 3600,
    auto_answer: bool = True,
    record_call: bool = False,
    record_format: str = "mp4",
    record_stereo: bool = True,
    default_webhook_url: Optional[str] = None,
    agent_id: Optional[str] = None,
    native_functions: Optional[List[str]] = None,
    schema_path: Optional[str] = None,
    suppress_logs: bool = False,
    enable_post_prompt_override: bool = False,
    check_for_input_override: bool = False,
    config_file: Optional[str] = None
)
```

**Parameters:**
- `name` (str): Human-readable name for the agent
- `route` (str): HTTP route path for the agent (default: "/")
- `host` (str): Host address to bind to (default: "0.0.0.0")
- `port` (int): Port number to listen on (default: 3000)
- `basic_auth` (Optional[Tuple[str, str]]): Username/password for HTTP basic auth
- `use_pom` (bool): Whether to use Prompt Object Model (default: True)
- `token_expiry_secs` (int): Security token expiration time (default: 3600)
- `auto_answer` (bool): Automatically answer incoming calls (default: True)
- `record_call` (bool): Record calls by default (default: False)
- `record_format` (str): Recording format: "mp4", "wav", "mp3" (default: "mp4")
- `record_stereo` (bool): Record in stereo (default: True)
- `default_webhook_url` (Optional[str]): Default webhook URL for functions
- `agent_id` (Optional[str]): Unique identifier for the agent
- `native_functions` (Optional[List[str]]): List of native function names to enable
- `schema_path` (Optional[str]): Path to custom SWML schema file
- `suppress_logs` (bool): Suppress logging output (default: False)
- `enable_post_prompt_override` (bool): Allow post-prompt URL override (default: False)
- `check_for_input_override` (bool): Allow check-for-input URL override (default: False)
- `config_file` (Optional[str]): Path to JSON configuration file with environment variable substitution support. See [Configuration Guide](configuration.md) for details.

### Core Methods

#### Deployment and Execution

##### `run(event=None, context=None, force_mode=None, host=None, port=None)`
Auto-detects deployment environment and runs the agent appropriately.

**Parameters:**
- `event`: Event object for serverless environments
- `context`: Context object for serverless environments  
- `force_mode` (str): Force specific mode: "server", "lambda", "cgi", "cloud_function"
- `host` (Optional[str]): Override host address
- `port` (Optional[int]): Override port number

**Usage:**
```python
# Auto-detect environment
agent.run()

# Force server mode
agent.run(force_mode="server", host="localhost", port=8080)

# Lambda handler
def lambda_handler(event, context):
    return agent.run(event, context)
```

##### `serve(host=None, port=None)`
Explicitly run as HTTP server using FastAPI/Uvicorn.

**Parameters:**
- `host` (Optional[str]): Host address to bind to
- `port` (Optional[int]): Port number to listen on

**Usage:**
```python
agent.serve()  # Use constructor defaults
agent.serve(host="0.0.0.0", port=3000)
```

### Prompt Configuration

#### Text-Based Prompts

##### `set_prompt_text(text: str) -> AgentBase`
Set the agent's prompt as raw text.

**Parameters:**
- `text` (str): The complete prompt text

**Usage:**
```python
agent.set_prompt_text("You are a helpful customer service agent.")
```

##### `set_post_prompt(text: str) -> AgentBase`
Set additional text to append after the main prompt.

**Parameters:**
- `text` (str): Text to append after main prompt

**Usage:**
```python
agent.set_post_prompt("Always be polite and professional.")
```

#### LLM Parameter Configuration

##### `set_prompt_llm_params`

```python
def set_prompt_llm_params(**params) -> AgentBase
```
Set Language Model parameters for the main prompt. Accepts any parameters which will be passed through to the SignalWire server. The server validates and applies parameters based on the target model's capabilities.

**Common Parameters:**
- `temperature`: Controls randomness. Lower = more focused
- `top_p`: Nucleus sampling threshold
- `barge_confidence`: ASR confidence to interrupt
- `presence_penalty`: Topic diversity control
- `frequency_penalty`: Repetition control

Note: No defaults are sent unless explicitly set. Invalid parameters for the selected model will be handled/ignored by the server.

**Usage:**
```python
# Configure for consistent, professional responses
agent.set_prompt_llm_params(
    temperature=0.3,
    top_p=0.9,
    barge_confidence=0.7,
    presence_penalty=0.1,
    frequency_penalty=0.2
)
```

##### `set_post_prompt_llm_params`

```python
def set_post_prompt_llm_params(**params) -> AgentBase
```
Set Language Model parameters for the post-prompt. Accepts any parameters which will be passed through to the SignalWire server. The server validates and applies parameters based on the target model's capabilities.

**Common Parameters:**
- `temperature`: Controls randomness. Lower = more focused
- `top_p`: Nucleus sampling threshold
- `presence_penalty`: Topic diversity control
- `frequency_penalty`: Repetition control

Note: barge_confidence is not applicable to post-prompt. No defaults are sent unless explicitly set.

**Usage:**
```python
# Configure for focused summaries
agent.set_post_prompt_llm_params(
    temperature=0.2,
    top_p=0.9
)
```

#### Structured Prompts (POM)

##### `prompt_add_section`

```python
def prompt_add_section(
    title: str, 
    body: str = "", 
    bullets: Optional[List[str]] = None, 
    numbered: bool = False, 
    numbered_bullets: bool = False, 
    subsections: Optional[List[Dict[str, Any]]] = None
) -> AgentBase
```
Add a structured section to the prompt using Prompt Object Model.

**Parameters:**
- `title` (str): Section title/heading
- `body` (str): Main section content (default: "")
- `bullets` (Optional[List[str]]): List of bullet points
- `numbered` (bool): Use numbered sections (default: False)
- `numbered_bullets` (bool): Use numbered bullet points (default: False)
- `subsections` (Optional[List[Dict]]): Nested subsections

**Usage:**
```python
# Simple section
agent.prompt_add_section("Role", "You are a customer service representative.")

# Section with bullets
agent.prompt_add_section(
    "Guidelines", 
    "Follow these principles:",
    bullets=["Be helpful", "Stay professional", "Listen carefully"]
)

# Numbered bullets
agent.prompt_add_section(
    "Process",
    "Follow these steps:",
    bullets=["Greet the customer", "Identify their need", "Provide solution"],
    numbered_bullets=True
)
```

##### `prompt_add_to_section`

```python
def prompt_add_to_section(
    title: str, 
    body: Optional[str] = None, 
    bullet: Optional[str] = None, 
    bullets: Optional[List[str]] = None
) -> AgentBase
```
Add content to an existing prompt section.

**Parameters:**
- `title` (str): Title of existing section to modify
- `body` (Optional[str]): Additional body text to append
- `bullet` (Optional[str]): Single bullet point to add
- `bullets` (Optional[List[str]]): Multiple bullet points to add

**Usage:**
```python
# Add body text to existing section
agent.prompt_add_to_section("Guidelines", "Remember to always verify customer identity.")

# Add single bullet
agent.prompt_add_to_section("Process", bullet="Document the interaction")

# Add multiple bullets
agent.prompt_add_to_section("Process", bullets=["Follow up", "Close ticket"])
```

##### `prompt_add_subsection`

```python
def prompt_add_subsection(
    parent_title: str, 
    title: str, 
    body: str = "", 
    bullets: Optional[List[str]] = None
) -> AgentBase
```
Add a subsection to an existing prompt section.

**Parameters:**
- `parent_title` (str): Title of parent section
- `title` (str): Subsection title
- `body` (str): Subsection content (default: "")
- `bullets` (Optional[List[str]]): Subsection bullet points

**Usage:**
```python
agent.prompt_add_subsection(
    "Guidelines",
    "Escalation Rules", 
    "Escalate when:",
    bullets=["Customer is angry", "Technical issue beyond scope"]
)
```

### Voice and Language Configuration

##### `add_language`

```python
def add_language(
    name: str, 
    code: str, 
    voice: str, 
    speech_fillers: Optional[List[str]] = None, 
    function_fillers: Optional[List[str]] = None, 
    engine: Optional[str] = None, 
    model: Optional[str] = None
) -> AgentBase
```
Configure voice and language settings for the agent.

**Parameters:**
- `name` (str): Human-readable language name
- `code` (str): Language code (e.g., "en-US", "es-ES")
- `voice` (str): Voice identifier (e.g., "rime.spore", "nova.luna")
- `speech_fillers` (Optional[List[str]]): Filler phrases during speech processing
- `function_fillers` (Optional[List[str]]): Filler phrases during function execution
- `engine` (Optional[str]): TTS engine to use
- `model` (Optional[str]): AI model to use

**Usage:**
```python
# Basic language setup
agent.add_language("English", "en-US", "rime.spore")

# With custom fillers
agent.add_language(
    "English", 
    "en-US", 
    "nova.luna",
    speech_fillers=["Let me think...", "One moment..."],
    function_fillers=["Processing...", "Looking that up..."]
)
```

##### `set_languages(languages: List[Dict[str, Any]]) -> AgentBase`
Set multiple language configurations at once.

**Parameters:**
- `languages` (List[Dict]): List of language configuration dictionaries

**Usage:**
```python
agent.set_languages([
    {"name": "English", "code": "en-US", "voice": "rime.spore"},
    {"name": "Spanish", "code": "es-ES", "voice": "nova.luna"}
])
```

### Speech Recognition Configuration

##### `add_hint(hint: str) -> AgentBase`
Add a single speech recognition hint.

**Parameters:**
- `hint` (str): Word or phrase to improve recognition accuracy

**Usage:**
```python
agent.add_hint("SignalWire")
```

##### `add_hints(hints: List[str]) -> AgentBase`
Add multiple speech recognition hints.

**Parameters:**
- `hints` (List[str]): List of words/phrases for better recognition

**Usage:**
```python
agent.add_hints(["SignalWire", "SWML", "API", "webhook", "SIP"])
```

##### `add_pattern_hint`

```python
def add_pattern_hint(
    hint: str, 
    pattern: str, 
    replace: str, 
    ignore_case: bool = False
) -> AgentBase
```
Add a pattern-based hint for speech recognition.

**Parameters:**
- `hint` (str): The hint phrase
- `pattern` (str): Regex pattern to match
- `replace` (str): Replacement text
- `ignore_case` (bool): Case-insensitive matching (default: False)

**Usage:**
```python
agent.add_pattern_hint(
    "phone number",
    r"(\d{3})-(\d{3})-(\d{4})",
    r"(\1) \2-\3"
)
```

##### `add_pronunciation`

```python
def add_pronunciation(
    replace: str, 
    with_text: str, 
    ignore_case: bool = False
) -> AgentBase
```
Add pronunciation rules for text-to-speech.

**Parameters:**
- `replace` (str): Text to replace
- `with_text` (str): Replacement pronunciation
- `ignore_case` (bool): Case-insensitive replacement (default: False)

**Usage:**
```python
agent.add_pronunciation("API", "A P I")
agent.add_pronunciation("SWML", "swim-el")
```

##### `set_pronunciations`

```python
def set_pronunciations(
    pronunciations: List[Dict[str, Any]]
) -> AgentBase
```
Set multiple pronunciation rules at once.

**Parameters:**
- `pronunciations` (List[Dict]): List of pronunciation rule dictionaries

**Usage:**
```python
agent.set_pronunciations([
    {"replace": "API", "with": "A P I"},
    {"replace": "SWML", "with": "swim-el", "ignore_case": True}
])
```

### AI Parameters Configuration

##### `set_param(key: str, value: Any) -> AgentBase`
Set a single AI parameter.

**Parameters:**
- `key` (str): Parameter name
- `value` (Any): Parameter value

**Usage:**
```python
agent.set_param("ai_model", "gpt-4.1-nano")
agent.set_param("end_of_speech_timeout", 500)
```

##### `set_params(params: Dict[str, Any]) -> AgentBase`
Set multiple AI parameters at once.

**Parameters:**
- `params` (Dict[str, Any]): Dictionary of parameter key-value pairs

**Common Parameters:**
- `ai_model`: AI model to use ("gpt-4.1-nano", "gpt-4.1-mini", etc.)
- `end_of_speech_timeout`: Milliseconds to wait for speech end (default: 1000)
- `attention_timeout`: Milliseconds before attention timeout (default: 30000)
- `background_file_volume`: Volume for background audio (-60 to 0 dB)
- `temperature`: AI creativity/randomness (0.0 to 2.0)
- `max_tokens`: Maximum response length
- `top_p`: Nucleus sampling parameter (0.0 to 1.0)

**Usage:**
```python
agent.set_params({
    "ai_model": "gpt-4.1-nano",
    "end_of_speech_timeout": 500,
    "attention_timeout": 15000,
    "background_file_volume": -20,
    "temperature": 0.7
})
```

### Global Data Management

##### `set_global_data(data: Dict[str, Any]) -> AgentBase`
Set global data available to the AI and functions.

**Parameters:**
- `data` (Dict[str, Any]): Global data dictionary

**Usage:**
```python
agent.set_global_data({
    "company_name": "Acme Corp",
    "support_hours": "9 AM - 5 PM EST",
    "escalation_number": "+1-555-0123"
})
```

##### `update_global_data(data: Dict[str, Any]) -> AgentBase`
Update existing global data (merge with existing).

**Parameters:**
- `data` (Dict[str, Any]): Data to merge with existing global data

**Usage:**
```python
agent.update_global_data({
    "current_promotion": "20% off all services",
    "promotion_expires": "2024-12-31"
})
```

### Function Definition

##### `define_tool`

```python
def define_tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    handler: Callable,
    secure: bool = True,
    fillers: Optional[Dict[str, List[str]]] = None,
    webhook_url: Optional[str] = None,
    is_typed_handler: bool = False,
    **swaig_fields
) -> AgentBase
```
Define a custom SWAIG function/tool.

**Parameters:**
- `name` (str): Function name
- `description` (str): Function description for AI
- `parameters` (Dict[str, Any]): JSON schema for function parameters. If omitted when using the decorator and the handler has type-hinted parameters, the schema is inferred automatically from the type hints.
- `handler` (Callable): Function to execute when called
- `secure` (bool): Require security token (default: True)
- `fillers` (Optional[Dict[str, List[str]]]): Language-specific filler phrases
- `webhook_url` (Optional[str]): Custom webhook URL
- `**swaig_fields`: Additional SWAIG function properties

**Usage:**
```python
def get_weather(args, raw_data):
    location = args.get("location", "Unknown")
    return SwaigFunctionResult(f"The weather in {location} is sunny and 75°F")

agent.define_tool(
    name="get_weather",
    description="Get current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name"
            }
        },
        "required": ["location"]
    },
    handler=get_weather,
    fillers={"en-US": ["Checking weather...", "Looking up forecast..."]}
)
```

##### `@AgentBase.tool(name=None, **kwargs)` (Class Decorator)
Decorator for defining tools as class methods.

**Parameters:**
- `name` (Optional[str]): Function name (defaults to method name)
- `**kwargs`: Same parameters as `define_tool()`

When `parameters` is omitted and the handler has type-hinted parameters (beyond `self`), the schema is inferred automatically from the type hints. The description is extracted from the docstring's first line, and per-parameter descriptions come from the `Args:` block.

**Usage (explicit schema):**
```python
class MyAgent(AgentBase):
    @AgentBase.tool(
        description="Get current time",
        parameters={"type": "object", "properties": {}}
    )
    def get_time(self, args, raw_data):
        import datetime
        return SwaigFunctionResult(f"Current time: {datetime.datetime.now()}")
```

**Usage (type-hinted, schema inferred):**
```python
class MyAgent(AgentBase):
    @AgentBase.tool(name="get_weather")
    def get_weather(self, city: str, units: str = "celsius"):
        """Get the weather forecast.

        Args:
            city: Name of the city
            units: Temperature units
        """
        return SwaigFunctionResult(f"Weather in {city}")
```

##### `register_swaig_function`

```python
def register_swaig_function(
    function_dict: Dict[str, Any]
) -> AgentBase
```
Register a pre-built SWAIG function dictionary.

**Parameters:**
- `function_dict` (Dict[str, Any]): Complete SWAIG function definition

**Usage:**
```python
# Register a DataMap tool
weather_tool = DataMap('get_weather').webhook('GET', 'https://api.weather.com/...')
agent.register_swaig_function(weather_tool.to_swaig_function())
```

### Session Lifecycle Hooks

SignalWire AI agents support special SWAIG functions that are automatically called at specific points in the conversation lifecycle:

##### `startup_hook`
Called when a new conversation/call begins.

**Implementation:**
```python
@AgentBase.tool(
    name="startup_hook",
    description="Called when a new conversation starts to initialize state",
    parameters={}
)
def startup_hook(self, args, raw_data):
    call_id = raw_data.get("call_id")
    # Initialize session resources, load user data, etc.
    return SwaigFunctionResult("Session initialized")
```

##### `hangup_hook`
Called when a conversation/call ends.

**Implementation:**
```python
@AgentBase.tool(
    name="hangup_hook",
    description="Called when conversation ends to clean up resources",
    parameters={}
)
def hangup_hook(self, args, raw_data):
    call_id = raw_data.get("call_id")
    # Clean up resources, save session data, etc.
    return SwaigFunctionResult("Session ended")
```

**Common Use Cases:**
- Loading user preferences at session start
- Initializing session-specific resources
- Logging conversation metrics
- Cleaning up temporary data
- Saving conversation summaries

### Skills System

##### `add_skill`

```python
def add_skill(
    skill_name: str, 
    params: Optional[Dict[str, Any]] = None
) -> AgentBase
```
Add a modular skill to the agent.

**Parameters:**
- `skill_name` (str): Name of the skill to add
- `params` (Optional[Dict[str, Any]]): Skill configuration parameters

**Available Skills:**
- `datetime`: Current date/time information
- `math`: Mathematical calculations
- `web_search`: Google Custom Search integration
- `datasphere`: SignalWire DataSphere search
- `native_vector_search`: Local document search

**Usage:**
```python
# Simple skill
agent.add_skill("datetime")
agent.add_skill("math")

# Skill with configuration
agent.add_skill("web_search", {
    "api_key": "your-google-api-key",
    "search_engine_id": "your-search-engine-id",
    "num_results": 3
})

# Multiple instances with different names
agent.add_skill("web_search", {
    "api_key": "your-api-key",
    "search_engine_id": "general-engine",
    "tool_name": "search_general"
})

agent.add_skill("web_search", {
    "api_key": "your-api-key", 
    "search_engine_id": "news-engine",
    "tool_name": "search_news"
})
```

##### `remove_skill(skill_name: str) -> AgentBase`
Remove a skill from the agent.

**Parameters:**
- `skill_name` (str): Name of skill to remove

**Usage:**
```python
agent.remove_skill("web_search")
```

##### `list_skills() -> List[str]`
Get list of currently added skills.

**Returns:**
- List[str]: Names of active skills

**Usage:**
```python
active_skills = agent.list_skills()
print(f"Active skills: {active_skills}")
```

##### `has_skill(skill_name: str) -> bool`
Check if a skill is currently added.

**Parameters:**
- `skill_name` (str): Name of skill to check

**Returns:**
- bool: True if skill is active

**Usage:**
```python
if agent.has_skill("web_search"):
    print("Web search is available")
```

### Native Functions

##### `set_native_functions`

```python
def set_native_functions(
    function_names: List[str]
) -> AgentBase
```
Enable specific native SWML functions.

**Parameters:**
- `function_names` (List[str]): List of native function names to enable

**Available Native Functions:**
- `transfer`: Transfer calls
- `hangup`: End calls
- `play`: Play audio files
- `record`: Record audio
- `send_sms`: Send SMS messages

**Usage:**
```python
agent.set_native_functions(["transfer", "hangup", "send_sms"])
```

##### `set_internal_fillers`

```python
def set_internal_fillers(
    internal_fillers: Dict[str, Dict[str, List[str]]]
) -> AgentBase
```
Set custom filler phrases for internal/native SWAIG functions.

**Parameters:**
- `internal_fillers` (Dict[str, Dict[str, List[str]]]): Function name → language code → filler phrases

**Available Internal Functions:**
- `next_step`: Moving between workflow steps (contexts system)
- `change_context`: Switching contexts in workflows  
- `check_time`: Getting current time
- `wait_for_user`: Waiting for user input
- `wait_seconds`: Pausing for specified duration
- `get_visual_input`: Processing visual data

**Usage:**
```python
agent.set_internal_fillers({
    "next_step": {
        "en-US": ["Moving to the next step...", "Let's continue..."],
        "es": ["Pasando al siguiente paso...", "Continuemos..."]
    },
    "check_time": {
        "en-US": ["Let me check the time...", "Getting current time..."]
    }
})
```

##### `add_internal_filler`

```python
def add_internal_filler(
    function_name: str, 
    language_code: str, 
    fillers: List[str]
) -> AgentBase
```
Add internal fillers for a specific function and language.

**Parameters:**
- `function_name` (str): Name of the internal function
- `language_code` (str): Language code (e.g., "en-US", "es", "fr")
- `fillers` (List[str]): List of filler phrases

**Usage:**
```python
agent.add_internal_filler("next_step", "en-US", [
    "Great! Let's move to the next step...",
    "Perfect! Moving forward..."
])
```

### Function Includes

##### `add_function_include`

```python
def add_function_include(
    url: str, 
    functions: List[str], 
    meta_data: Optional[Dict[str, Any]] = None
) -> AgentBase
```
Include external SWAIG functions from another service.

**Parameters:**
- `url` (str): URL of external SWAIG service
- `functions` (List[str]): List of function names to include
- `meta_data` (Optional[Dict[str, Any]]): Additional metadata

**Usage:**
```python
agent.add_function_include(
    "https://external-service.com/swaig",
    ["external_function1", "external_function2"],
    meta_data={"service": "external", "version": "1.0"}
)
```

##### `set_function_includes`

```python
def set_function_includes(
    includes: List[Dict[str, Any]]
) -> AgentBase
```
Set multiple function includes at once.

**Parameters:**
- `includes` (List[Dict[str, Any]]): List of function include configurations

**Usage:**
```python
agent.set_function_includes([
    {
        "url": "https://service1.com/swaig",
        "functions": ["func1", "func2"]
    },
    {
        "url": "https://service2.com/swaig", 
        "functions": ["func3"],
        "meta_data": {"priority": "high"}
    }
])
```

### Webhook Configuration

##### `set_web_hook_url(url: str) -> AgentBase`
Set default webhook URL for SWAIG functions.

**Parameters:**
- `url` (str): Default webhook URL

**Usage:**
```python
agent.set_web_hook_url("https://myserver.com/webhook")
```

##### `set_post_prompt_url(url: str) -> AgentBase`
Set URL for post-prompt processing.

**Parameters:**
- `url` (str): Post-prompt webhook URL

**Usage:**
```python
agent.set_post_prompt_url("https://myserver.com/post-prompt")
```

##### `add_swaig_query_params(params: dict) -> AgentBase`
Add query parameters to be included in all SWAIG webhook URLs.

This is useful for preserving dynamic configuration state across SWAIG callbacks. For example, if your dynamic config adds skills based on query parameters, you can pass those same parameters through to the SWAIG webhook so the same configuration is applied.

**Parameters:**
- `params` (dict): Dictionary of query parameter key-value pairs

**Usage:**
```python
# In dynamic config callback, preserve configuration parameters
def configure_agent(query_params, headers, body, agent):
    customer_id = query_params.get("customer_id")
    if customer_id:
        # Pass through to SWAIG callbacks
        agent.add_swaig_query_params({"customer_id": customer_id})
        agent.add_skill("customer_lookup", {"customer_id": customer_id})

agent.set_dynamic_config_callback(configure_agent)
```

##### `clear_swaig_query_params() -> AgentBase`
Clear all SWAIG query parameters.

**Usage:**
```python
agent.clear_swaig_query_params()
```

### Debug Events

##### `enable_debug_events`

```python
def enable_debug_events(level: int = 1) -> AgentBase
```
Enable the debug event webhook for this agent. When enabled, the AI module will POST real-time debug events to a `/debug_events` endpoint on this agent during calls. Events are automatically logged via the agent's structured logger and can optionally be handled with a custom callback via `on_debug_event()`.

**Parameters:**
- `level` (int): Debug event verbosity level. `1` = high-level events (barge, errors, session start/end, step changes). `2+` = adds high-volume events (every LLM request/response, conversation_add). Default: `1`

**Usage:**
```python
agent.enable_debug_events()        # level 1 (default)
agent.enable_debug_events(level=2) # include high-volume events
```

**How it works:**
- Registers a `/debug_events` POST endpoint on the agent's HTTP server
- Auto-sets `debug_webhook_url` and `debug_webhook_level` in the SWML `params` during rendering
- The URL is built automatically using the same auth/proxy logic as other webhook URLs
- No manual URL configuration needed

**Event types at level 1:**

| Event label | Description |
|-------------|-------------|
| `session_start` | AI session started (model, TTS engine, voice, language) |
| `session_end` | AI session ended (reason, duration, token counts) |
| `barge` | User interrupted AI speech (barge type, elapsed ms) |
| `step_change` | Conversation step changed |
| `context_change` | Conversation context changed |
| `llm_error` | LLM error (fatal, retry, max_retries) |
| `voice_error` | TTS voice configuration or runtime error |
| `hold` | Call placed on hold or taken off hold |
| `filler` | Filler phrase spoken (thinking or function filler) |
| `consolidation` | Token consolidation triggered |
| `process_action` | Webhook action being processed |
| `gather_start` | Gather flow started |
| `gather_complete` | Gather flow completed |

**Additional events at level 2+:**

| Event label | Description |
|-------------|-------------|
| `llm_request` | LLM API request initiated (input tokens) |
| `llm_response` | LLM API response received (duration, output tokens) |
| `conversation_add` | Entry added to conversation history |

### Call Flow Verb Insertion

These methods allow you to customize the SWML call flow by inserting verbs at different stages of the call lifecycle.

##### `add_pre_answer_verb(verb_name: str, config: dict) -> AgentBase`
Add a verb to run before the call is answered (while still ringing).

**Safe pre-answer verbs:** `transfer`, `execute`, `return`, `label`, `goto`, `request`, `switch`, `cond`, `if`, `eval`, `set`, `unset`, `hangup`, `send_sms`, `sleep`, `stop_record_call`, `stop_denoise`, `stop_tap`

**Parameters:**
- `verb_name` (str): The SWML verb name
- `config` (dict): Verb configuration dictionary

**Usage:**
```python
# Send SMS before answering
agent.add_pre_answer_verb("send_sms", {
    "to": "+15551234567",
    "from": "+15559876543",
    "body": "Incoming call from AI agent"
})

# Set variables before answer
agent.add_pre_answer_verb("set", {"call_start": "${system.timestamp}"})
```

##### `add_answer_verb(config: dict = None) -> AgentBase`
Configure the answer verb that connects the call.

**Parameters:**
- `config` (dict, optional): Answer verb configuration (e.g., `{"max_duration": 3600}`)

**Usage:**
```python
# Set maximum call duration to 1 hour
agent.add_answer_verb({"max_duration": 3600})
```

##### `add_post_answer_verb(verb_name: str, config: dict) -> AgentBase`
Add a verb to run after the call is answered but before the AI starts.

**Parameters:**
- `verb_name` (str): The SWML verb name (e.g., "play", "sleep")
- `config` (dict): Verb configuration dictionary

**Usage:**
```python
# Play welcome message before AI starts
agent.add_post_answer_verb("play", {
    "url": "say:Welcome to our AI assistant. This call may be recorded."
})

# Add a brief pause
agent.add_post_answer_verb("sleep", {"duration": 1})
```

##### `add_post_ai_verb(verb_name: str, config: dict) -> AgentBase`
Add a verb to run after the AI conversation ends.

**Parameters:**
- `verb_name` (str): The SWML verb name (e.g., "hangup", "transfer", "request")
- `config` (dict): Verb configuration dictionary

**Usage:**
```python
# Clean hangup after AI ends
agent.add_post_ai_verb("hangup", {})

# Transfer to human after AI conversation
agent.add_post_ai_verb("transfer", {"to": "+15551234567"})

# Log call completion
agent.add_post_ai_verb("request", {
    "url": "https://myserver.com/call-complete",
    "method": "POST"
})
```

##### `clear_pre_answer_verbs() -> AgentBase`
Remove all pre-answer verbs.

##### `clear_post_answer_verbs() -> AgentBase`
Remove all post-answer verbs.

##### `clear_post_ai_verbs() -> AgentBase`
Remove all post-AI verbs.

**Method Chaining Example:**
```python
agent.add_pre_answer_verb("set", {"source": "ai_agent"}) \
     .add_answer_verb({"max_duration": 1800}) \
     .add_post_answer_verb("play", {"url": "say:Hello!"}) \
     .add_post_ai_verb("hangup", {})
```

### Dynamic Configuration

##### `set_dynamic_config_callback`

```python
def set_dynamic_config_callback(
    callback: Callable[[dict, dict, dict, AgentBase], None]
) -> AgentBase
```
Set callback for per-request dynamic configuration.

**Parameters:**
- `callback` (Callable): Function that receives (query_params, headers, body, config)

**Usage:**
```python
def configure_agent(query_params, headers, body, config):
    # Configure based on request
    if query_params.get("language") == "spanish":
        config.add_language("Spanish", "es-ES", "nova.luna")
    
    # Set customer-specific data
    customer_id = headers.get("X-Customer-ID")
    if customer_id:
        config.set_global_data({"customer_id": customer_id})

agent.set_dynamic_config_callback(configure_agent)
```

### SIP Integration

##### `enable_sip_routing`

```python
def enable_sip_routing(
    auto_map: bool = True, 
    path: str = "/sip"
) -> AgentBase
```
Enable SIP-based routing for voice calls.

**Parameters:**
- `auto_map` (bool): Automatically map SIP usernames (default: True)
- `path` (str): SIP routing endpoint path (default: "/sip")

**Usage:**
```python
agent.enable_sip_routing()
```

##### `register_sip_username(sip_username: str) -> AgentBase`
Register a specific SIP username for this agent.

**Parameters:**
- `sip_username` (str): SIP username to register

**Usage:**
```python
agent.register_sip_username("support")
agent.register_sip_username("sales")
```

##### `register_routing_callback`

```python
def register_routing_callback(
    callback_fn: Callable[[Request, Dict[str, Any]], Optional[str]], 
    path: str = "/sip"
) -> None
```
Register custom routing logic for SIP calls.

**Parameters:**
- `callback_fn` (Callable): Function that returns agent route based on request
- `path` (str): Routing endpoint path (default: "/sip")

**Usage:**
```python
def route_call(request, body):
    sip_username = body.get("sip_username")
    if sip_username == "support":
        return "/support-agent"
    elif sip_username == "sales":
        return "/sales-agent"
    return None

agent.register_routing_callback(route_call)
```

### Utility Methods

##### `get_name() -> str`
Get the agent's name.

**Returns:**
- str: Agent name

##### `get_app()`
Get the FastAPI application instance.

**Returns:**
- FastAPI: The underlying FastAPI app

##### `as_router() -> APIRouter`
Get the agent as a FastAPI router for embedding in larger applications.

**Returns:**
- APIRouter: FastAPI router instance

**Usage:**
```python
# Embed agent in larger FastAPI app
main_app = FastAPI()
agent_router = agent.as_router()
main_app.include_router(agent_router, prefix="/agent")
```

### Event Handlers

##### `on_summary`

```python
def on_summary(
    summary: Optional[Dict[str, Any]],
    raw_data: Optional[Dict[str, Any]] = None
) -> None
```
Override to handle conversation summaries. This callback is triggered when the AI generates a summary based on your `post_prompt` configuration.

**Parameters:**
- `summary` (Optional[Dict[str, Any]]): Parsed summary data (from `post_prompt_data.parsed[0]`)
- `raw_data` (Optional[Dict[str, Any]]): Complete raw POST data including `post_prompt_data` with both `raw` and `parsed` fields

**Usage:**
```python
class MyAgent(AgentBase):
    def __init__(self):
        super().__init__(name="summary-agent", route="/agent")

        # Configure post-prompt to request JSON summary
        self.set_post_prompt("""
        Return a JSON summary of the conversation:
        {
            "topic": "MAIN_TOPIC",
            "satisfied": true/false,
            "follow_up_needed": true/false,
            "key_points": ["point1", "point2"]
        }
        """)

    def on_summary(self, summary, raw_data):
        """Handle conversation summaries after call ends"""
        if summary:
            # Access parsed JSON fields directly
            topic = summary.get("topic", "Unknown")
            satisfied = summary.get("satisfied", False)

            print(f"Call about: {topic}, Customer satisfied: {satisfied}")

            # Save to database, send to CRM, trigger follow-up, etc.
            if summary.get("follow_up_needed"):
                self.schedule_follow_up(summary)

        # Access raw summary text if needed
        if raw_data and 'post_prompt_data' in raw_data:
            raw_text = raw_data['post_prompt_data'].get('raw', '')
            print(f"Raw summary: {raw_text}")
```

##### `on_debug_event`

```python
def on_debug_event(handler: Callable) -> Callable
```
Register a handler for debug webhook events. Use as a decorator. Requires `enable_debug_events()` to be called first.

The handler receives:
- `event_type` (str): The event label (e.g. `"barge"`, `"llm_error"`, `"session_start"`)
- `data` (dict): The full event payload including `call_id`, `label`, and event-specific fields

The handler may be sync or async.

**Usage (decorator style):**
```python
agent = AgentBase("my_agent")
agent.enable_debug_events()

@agent.on_debug_event
def handle_debug(event_type, data):
    call_id = data.get("call_id")
    if event_type == "llm_error":
        print(f"LLM error on call {call_id}: {data.get('event')}")
    elif event_type == "barge":
        print(f"Barge after {data.get('barge_elapsed_ms')}ms")
    elif event_type == "session_end":
        print(f"Call ended: {data.get('reason')}, duration: {data.get('duration_ms')}ms")
```

**Usage (subclass style):**
```python
class MyAgent(AgentBase):
    def __init__(self):
        super().__init__(name="debug-agent", route="/agent")
        self.enable_debug_events(level=2)
        self.on_debug_event(self.handle_debug)

    def handle_debug(self, event_type, data):
        if event_type == "llm_error":
            self.alert_ops_team(data)
```

> **Note:** Even without registering a handler, all debug events are automatically logged via the agent's structured logger when `enable_debug_events()` is called.

##### `on_function_call`

```python
def on_function_call(
    name: str,
    args: Dict[str, Any],
    raw_data: Optional[Dict[str, Any]] = None
) -> Any
```
Override to handle function calls with custom logic.

**Parameters:**
- `name` (str): Function name being called
- `args` (Dict[str, Any]): Function arguments
- `raw_data` (Optional[Dict[str, Any]]): Raw request data

**Returns:**
- Any: Function result (typically SwaigFunctionResult)

**Usage:**
```python
class MyAgent(AgentBase):
    def on_function_call(self, name, args, raw_data):
        if name == "get_weather":
            location = args.get("location")
            # Custom weather logic
            return SwaigFunctionResult(f"Weather in {location}: Sunny")
        return super().on_function_call(name, args, raw_data)
```

##### `on_request`

```python
def on_request(
    request_data: Optional[dict] = None, 
    callback_path: Optional[str] = None
) -> Optional[dict]
```
Override to handle general requests.

**Parameters:**
- `request_data` (Optional[dict]): Request data
- `callback_path` (Optional[str]): Callback path

**Returns:**
- Optional[dict]: Response modifications

##### `on_swml_request`

```python
def on_swml_request(
    request_data: Optional[dict] = None, 
    callback_path: Optional[str] = None, 
    request: Optional[Request] = None
) -> Optional[dict]
```
Override to handle SWML generation requests.

**Parameters:**
- `request_data` (Optional[dict]): Request data
- `callback_path` (Optional[str]): Callback path  
- `request` (Optional[Request]): FastAPI request object

**Returns:**
- Optional[dict]: SWML modifications

### Authentication

##### `validate_basic_auth(username: str, password: str) -> bool`
Override to implement custom basic authentication logic.

**Parameters:**
- `username` (str): Username from basic auth
- `password` (str): Password from basic auth

**Returns:**
- bool: True if credentials are valid

**Usage:**
```python
class MyAgent(AgentBase):
    def validate_basic_auth(self, username, password):
        # Custom auth logic
        return username == "admin" and password == "secret"
```

##### `get_basic_auth_credentials`

```python
def get_basic_auth_credentials(
    include_source: bool = False
) -> Union[Tuple[str, str], Tuple[str, str, str]]
```
Get basic auth credentials from environment or constructor.

**Parameters:**
- `include_source` (bool): Include source information (default: False)

**Returns:**
- Tuple: (username, password) or (username, password, source)

### Context System

##### `define_contexts() -> ContextBuilder`
Define structured workflow contexts for the agent.

**Returns:**
- ContextBuilder: Builder for creating contexts and steps

**Usage:**
```python
contexts = agent.define_contexts()
contexts.add_context("greeting") \
    .add_step("welcome", "Welcome! How can I help?") \
    .on_completion_go_to("main_menu")

contexts.add_context("main_menu") \
    .add_step("menu", "Choose: 1) Support 2) Sales 3) Billing") \
    .allow_functions(["transfer_to_support", "transfer_to_sales"])
```

This concludes Part 1 of the API reference covering the AgentBase class. The document will continue with SwaigFunctionResult, DataMap, and other components in subsequent parts.

---

## SwaigFunctionResult Class

The `SwaigFunctionResult` class is used to create structured responses from SWAIG functions. It handles both natural language responses and structured actions that the agent should execute.

### Constructor

```python
SwaigFunctionResult(
    response: Optional[str] = None, 
    post_process: bool = False
)
```

**Parameters:**
- `response` (Optional[str]): Natural language response text for the AI to speak
- `post_process` (bool): Whether to let AI take another turn before executing actions (default: False)

**Post-processing Behavior:**
- `post_process=False` (default): Execute actions immediately after AI response
- `post_process=True`: Let AI respond to user one more time, then execute actions

**Usage:**
```python
# Simple response
result = SwaigFunctionResult("The weather is sunny and 75°F")

# Response with post-processing enabled
result = SwaigFunctionResult("I'll transfer you now", post_process=True)

# Empty response (actions only)
result = SwaigFunctionResult()
```

### Core Methods

#### Response Configuration

##### `set_response(response: str) -> SwaigFunctionResult`
Set or update the natural language response text.

**Parameters:**
- `response` (str): The text the AI should speak

**Usage:**
```python
result = SwaigFunctionResult()
result.set_response("I found your order information")
```

##### `set_post_process(post_process: bool) -> SwaigFunctionResult`
Enable or disable post-processing for this result.

**Parameters:**
- `post_process` (bool): True to let AI respond once more before executing actions

**Usage:**
```python
result = SwaigFunctionResult("I'll help you with that")
result.set_post_process(True)  # Let AI handle follow-up questions first
```

#### Action Management

##### `add_action(name: str, data: Any) -> SwaigFunctionResult`
Add a structured action to execute.

**Parameters:**
- `name` (str): Action name/type (e.g., "play", "transfer", "set_global_data")
- `data` (Any): Action data - can be string, boolean, object, or array

**Usage:**
```python
# Simple action with boolean
result.add_action("hangup", True)

# Action with string data
result.add_action("play", "welcome.mp3")

# Action with object data
result.add_action("set_global_data", {"customer_id": "12345", "status": "verified"})

# Action with array data
result.add_action("send_sms", ["+15551234567", "Your order is ready!"])
```

##### `add_actions(actions: List[Dict[str, Any]]) -> SwaigFunctionResult`
Add multiple actions at once.

**Parameters:**
- `actions` (List[Dict[str, Any]]): List of action dictionaries

**Usage:**
```python
result.add_actions([
    {"play": "hold_music.mp3"},
    {"set_global_data": {"status": "on_hold"}},
    {"wait": 5000}
])
```

### Call Control Actions

#### Call Transfer and Connection

##### `connect(destination: str, final: bool = True, from_addr: Optional[str] = None) -> SwaigFunctionResult`
Transfer or connect the call to another destination.

**Parameters:**
- `destination` (str): Phone number, SIP address, or other destination
- `final` (bool): Permanent transfer (True) vs temporary transfer (False) (default: True)
- `from_addr` (Optional[str]): Override caller ID

**Transfer Types:**
- `final=True`: Permanent transfer - call exits agent completely
- `final=False`: Temporary transfer - call returns to agent if far end hangs up

**Usage:**
```python
# Permanent transfer to phone number
result.connect("+15551234567", final=True)

# Temporary transfer to SIP address with custom caller ID
result.connect("support@company.com", final=False, from_addr="+15559876543")

# Transfer with response
result = SwaigFunctionResult("Transferring you to our sales team")
result.connect("sales@company.com")
```

##### `swml_transfer(dest: str, ai_response: str) -> SwaigFunctionResult`
Create a SWML-based transfer with AI response setup.

**Parameters:**
- `dest` (str): Transfer destination
- `ai_response` (str): AI response when transfer completes

**Usage:**
```python
result.swml_transfer(
    "+15551234567", 
    "You've been transferred back to me. How else can I help?"
)
```

##### `sip_refer(to_uri: str) -> SwaigFunctionResult`
Perform a SIP REFER transfer.

**Parameters:**
- `to_uri` (str): SIP URI to transfer to

**Usage:**
```python
result.sip_refer("sip:support@company.com")
```

#### Call Management

##### `hangup() -> SwaigFunctionResult`
End the call immediately.

**Usage:**
```python
result = SwaigFunctionResult("Thank you for calling. Goodbye!")
result.hangup()
```

##### `hold(timeout: int = 300) -> SwaigFunctionResult`
Put the call on hold.

**Parameters:**
- `timeout` (int): Hold timeout in seconds (default: 300)

**Usage:**
```python
result = SwaigFunctionResult("Please hold while I look that up")
result.hold(timeout=60)
```

##### `stop() -> SwaigFunctionResult`
Stop current audio playback or recording.

**Usage:**
```python
result.stop()
```

#### Audio Control

##### `say(text: str) -> SwaigFunctionResult`
Add text for the AI to speak.

**Parameters:**
- `text` (str): Text to speak

**Usage:**
```python
result.say("Please wait while I process your request")
```

##### `play_background_file(filename: str, wait: bool = False) -> SwaigFunctionResult`
Play an audio file in the background.

**Parameters:**
- `filename` (str): Audio file path or URL
- `wait` (bool): Wait for file to finish before continuing (default: False)

**Usage:**
```python
# Play hold music in background
result.play_background_file("hold_music.mp3")

# Play announcement and wait for completion
result.play_background_file("important_announcement.wav", wait=True)
```

##### `stop_background_file() -> SwaigFunctionResult`
Stop background audio playback.

**Usage:**
```python
result.stop_background_file()
```

### Data Management Actions

##### `set_global_data(data: Dict[str, Any]) -> SwaigFunctionResult`
Set global data for the conversation.

**Parameters:**
- `data` (Dict[str, Any]): Global data to set

**Usage:**
```python
result.set_global_data({
    "customer_id": "12345",
    "order_status": "shipped",
    "tracking_number": "1Z999AA1234567890"
})
```

##### `update_global_data(data: Dict[str, Any]) -> SwaigFunctionResult`
Update existing global data (merge with existing).

**Parameters:**
- `data` (Dict[str, Any]): Data to merge

**Usage:**
```python
result.update_global_data({
    "last_interaction": "2024-01-15T10:30:00Z",
    "agent_notes": "Customer satisfied with resolution"
})
```

##### `remove_global_data(keys: Union[str, List[str]]) -> SwaigFunctionResult`
Remove specific keys from global data.

**Parameters:**
- `keys` (Union[str, List[str]]): Key name or list of key names to remove

**Usage:**
```python
# Remove single key
result.remove_global_data("temporary_data")

# Remove multiple keys
result.remove_global_data(["temp1", "temp2", "cache_data"])
```

##### `set_metadata(data: Dict[str, Any]) -> SwaigFunctionResult`
Set metadata for the conversation.

**Parameters:**
- `data` (Dict[str, Any]): Metadata to set

**Usage:**
```python
result.set_metadata({
    "call_type": "support",
    "priority": "high",
    "department": "technical"
})
```

##### `remove_metadata(keys: Union[str, List[str]]) -> SwaigFunctionResult`
Remove specific metadata keys.

**Parameters:**
- `keys` (Union[str, List[str]]): Key name or list of key names to remove

**Usage:**
```python
result.remove_metadata(["temporary_flag", "debug_info"])
```

### AI Behavior Control

##### `set_end_of_speech_timeout(milliseconds: int) -> SwaigFunctionResult`
Adjust how long to wait for speech to end.

**Parameters:**
- `milliseconds` (int): Timeout in milliseconds

**Usage:**
```python
# Shorter timeout for quick responses
result.set_end_of_speech_timeout(300)

# Longer timeout for thoughtful responses
result.set_end_of_speech_timeout(2000)
```

##### `set_speech_event_timeout(milliseconds: int) -> SwaigFunctionResult`
Set timeout for speech events.

**Parameters:**
- `milliseconds` (int): Timeout in milliseconds

**Usage:**
```python
result.set_speech_event_timeout(5000)
```

##### `wait_for_user(enabled: Optional[bool] = None, timeout: Optional[int] = None, answer_first: bool = False) -> SwaigFunctionResult`
Control whether to wait for user input.

**Parameters:**
- `enabled` (Optional[bool]): Enable/disable waiting for user
- `timeout` (Optional[int]): Timeout in milliseconds
- `answer_first` (bool): Answer call before waiting (default: False)

**Usage:**
```python
# Wait for user input with 10 second timeout
result.wait_for_user(enabled=True, timeout=10000)

# Don't wait for user (immediate response)
result.wait_for_user(enabled=False)
```

##### `toggle_functions(function_toggles: List[Dict[str, Any]]) -> SwaigFunctionResult`
Enable or disable specific functions.

**Parameters:**
- `function_toggles` (List[Dict[str, Any]]): List of function toggle configurations

**Usage:**
```python
result.toggle_functions([
    {"name": "transfer_to_sales", "enabled": True},
    {"name": "end_call", "enabled": False},
    {"name": "escalate", "enabled": True, "timeout": 30000}
])
```

##### `enable_functions_on_timeout(enabled: bool = True) -> SwaigFunctionResult`
Control whether functions are enabled when timeout occurs.

**Parameters:**
- `enabled` (bool): Enable functions on timeout (default: True)

**Usage:**
```python
result.enable_functions_on_timeout(False)  # Disable functions on timeout
```

##### `enable_extensive_data(enabled: bool = True) -> SwaigFunctionResult`
Enable extensive data collection.

**Parameters:**
- `enabled` (bool): Enable extensive data (default: True)

**Usage:**
```python
result.enable_extensive_data(True)
```

##### `update_settings(settings: Dict[str, Any]) -> SwaigFunctionResult`
Update various AI settings.

**Parameters:**
- `settings` (Dict[str, Any]): Settings to update

**Usage:**
```python
result.update_settings({
    "temperature": 0.8,
    "max_tokens": 150,
    "end_of_speech_timeout": 800
})
```

### Context and Conversation Control

##### `switch_context(system_prompt: Optional[str] = None, user_prompt: Optional[str] = None, consolidate: bool = False, full_reset: bool = False) -> SwaigFunctionResult`
Switch conversation context or reset the conversation.

**Parameters:**
- `system_prompt` (Optional[str]): New system prompt
- `user_prompt` (Optional[str]): New user prompt
- `consolidate` (bool): Consolidate conversation history (default: False)
- `full_reset` (bool): Completely reset conversation (default: False)

**Usage:**
```python
# Switch to technical support context
result.switch_context(
    system_prompt="You are now a technical support specialist",
    user_prompt="The customer needs technical help"
)

# Reset conversation completely
result.switch_context(full_reset=True)

# Consolidate conversation history
result.switch_context(consolidate=True)
```

##### `simulate_user_input(text: str) -> SwaigFunctionResult`
Simulate user input for testing or automation.

**Parameters:**
- `text` (str): Text to simulate as user input

**Usage:**
```python
result.simulate_user_input("I need help with my order")
```

### Communication Actions

##### `send_sms(to_number: str, from_number: str, body: Optional[str] = None, media: Optional[List[str]] = None, tags: Optional[List[str]] = None, region: Optional[str] = None) -> SwaigFunctionResult`
Send an SMS message.

**Parameters:**
- `to_number` (str): Recipient phone number
- `from_number` (str): Sender phone number
- `body` (Optional[str]): SMS message text
- `media` (Optional[List[str]]): List of media URLs
- `tags` (Optional[List[str]]): Message tags
- `region` (Optional[str]): SignalWire region

**Usage:**
```python
# Simple text message
result.send_sms(
    to_number="+15551234567",
    from_number="+15559876543", 
    body="Your order #12345 has shipped!"
)

# Message with media and tags
result.send_sms(
    to_number="+15551234567",
    from_number="+15559876543",
    body="Here's your receipt",
    media=["https://example.com/receipt.pdf"],
    tags=["receipt", "order_12345"]
)
```

### Recording and Media

##### `record_call(control_id: Optional[str] = None, stereo: bool = False, format: str = "wav", direction: str = "both", terminators: Optional[str] = None, beep: bool = False, input_sensitivity: float = 44.0, initial_timeout: float = 0.0, end_silence_timeout: float = 0.0, max_length: Optional[float] = None, status_url: Optional[str] = None) -> SwaigFunctionResult`
Start call recording.

**Parameters:**
- `control_id` (Optional[str]): Unique identifier for this recording
- `stereo` (bool): Record in stereo (default: False)
- `format` (str): Recording format: "wav", "mp3", "mp4" (default: "wav")
- `direction` (str): Recording direction: "both", "inbound", "outbound" (default: "both")
- `terminators` (Optional[str]): DTMF keys to stop recording
- `beep` (bool): Play beep before recording (default: False)
- `input_sensitivity` (float): Input sensitivity level (default: 44.0)
- `initial_timeout` (float): Initial timeout in seconds (default: 0.0)
- `end_silence_timeout` (float): End silence timeout in seconds (default: 0.0)
- `max_length` (Optional[float]): Maximum recording length in seconds
- `status_url` (Optional[str]): Webhook URL for recording status

**Usage:**
```python
# Basic recording
result.record_call(format="mp3", direction="both")

# Recording with control ID and settings
result.record_call(
    control_id="customer_call_001",
    stereo=True,
    format="wav",
    beep=True,
    max_length=300.0,
    terminators="#*"
)
```

##### `stop_record_call(control_id: Optional[str] = None) -> SwaigFunctionResult`
Stop call recording.

**Parameters:**
- `control_id` (Optional[str]): Control ID of recording to stop

**Usage:**
```python
result.stop_record_call()
result.stop_record_call(control_id="customer_call_001")
```

### Conference and Room Management

##### `join_room(name: str) -> SwaigFunctionResult`
Join a SignalWire room.

**Parameters:**
- `name` (str): Room name to join

**Usage:**
```python
result.join_room("support_room_1")
```

##### `join_conference(name: str, muted: bool = False, beep: str = "true", start_on_enter: bool = True, end_on_exit: bool = False, wait_url: Optional[str] = None, max_participants: int = 250, record: str = "do-not-record", region: Optional[str] = None, trim: str = "trim-silence", coach: Optional[str] = None, status_callback_event: Optional[str] = None, status_callback: Optional[str] = None, status_callback_method: str = "POST", recording_status_callback: Optional[str] = None, recording_status_callback_method: str = "POST", recording_status_callback_event: str = "completed", result: Optional[Any] = None) -> SwaigFunctionResult`
Join a conference call.

**Parameters:**
- `name` (str): Conference name
- `muted` (bool): Join muted (default: False)
- `beep` (str): Beep setting: "true", "false", "onEnter", "onExit" (default: "true")
- `start_on_enter` (bool): Start conference when this participant enters (default: True)
- `end_on_exit` (bool): End conference when this participant exits (default: False)
- `wait_url` (Optional[str]): URL for hold music/content
- `max_participants` (int): Maximum participants (default: 250)
- `record` (str): Recording setting (default: "do-not-record")
- `region` (Optional[str]): SignalWire region
- `trim` (str): Trim setting for recordings (default: "trim-silence")
- `coach` (Optional[str]): Coach participant identifier
- `status_callback_event` (Optional[str]): Status callback events
- `status_callback` (Optional[str]): Status callback URL
- `status_callback_method` (str): Status callback HTTP method (default: "POST")
- `recording_status_callback` (Optional[str]): Recording status callback URL
- `recording_status_callback_method` (str): Recording status callback method (default: "POST")
- `recording_status_callback_event` (str): Recording status callback events (default: "completed")

**Usage:**
```python
# Basic conference join
result.join_conference("sales_meeting")

# Conference with recording and settings
result.join_conference(
    name="support_conference",
    muted=False,
    beep="onEnter",
    record="record-from-start",
    max_participants=10
)
```

### Payment Processing

##### `pay(payment_connector_url: str, input_method: str = "dtmf", status_url: Optional[str] = None, payment_method: str = "credit-card", timeout: int = 5, max_attempts: int = 1, security_code: bool = True, postal_code: Union[bool, str] = True, min_postal_code_length: int = 0, token_type: str = "reusable", charge_amount: Optional[str] = None, currency: str = "usd", language: str = "en-US", voice: str = "woman", description: Optional[str] = None, valid_card_types: str = "visa mastercard amex", parameters: Optional[List[Dict[str, str]]] = None, prompts: Optional[List[Dict[str, Any]]] = None) -> SwaigFunctionResult`
Process a payment through the call.

**Parameters:**
- `payment_connector_url` (str): Payment processor webhook URL
- `input_method` (str): Input method: "dtmf", "speech" (default: "dtmf")
- `status_url` (Optional[str]): Payment status webhook URL
- `payment_method` (str): Payment method: "credit-card" (default: "credit-card")
- `timeout` (int): Input timeout in seconds (default: 5)
- `max_attempts` (int): Maximum retry attempts (default: 1)
- `security_code` (bool): Require security code (default: True)
- `postal_code` (Union[bool, str]): Require postal code (default: True)
- `min_postal_code_length` (int): Minimum postal code length (default: 0)
- `token_type` (str): Token type: "reusable", "one-time" (default: "reusable")
- `charge_amount` (Optional[str]): Amount to charge
- `currency` (str): Currency code (default: "usd")
- `language` (str): Language for prompts (default: "en-US")
- `voice` (str): Voice for prompts (default: "woman")
- `description` (Optional[str]): Payment description
- `valid_card_types` (str): Accepted card types (default: "visa mastercard amex")
- `parameters` (Optional[List[Dict[str, str]]]): Additional parameters
- `prompts` (Optional[List[Dict[str, Any]]]): Custom prompts

**Usage:**
```python
# Basic payment processing
result.pay(
    payment_connector_url="https://payment-processor.com/webhook",
    charge_amount="29.99",
    description="Monthly subscription"
)

# Payment with custom settings
result.pay(
    payment_connector_url="https://payment-processor.com/webhook",
    input_method="speech",
    timeout=10,
    max_attempts=3,
    security_code=True,
    postal_code=True,
    charge_amount="149.99",
    currency="usd",
    description="Premium service upgrade"
)
```

### Call Monitoring

##### `tap(uri: str, control_id: Optional[str] = None, direction: str = "both", codec: str = "PCMU", rtp_ptime: int = 20, status_url: Optional[str] = None) -> SwaigFunctionResult`
Start call tapping/monitoring.

**Parameters:**
- `uri` (str): URI to send tapped audio to
- `control_id` (Optional[str]): Unique identifier for this tap
- `direction` (str): Tap direction: "both", "inbound", "outbound" (default: "both")
- `codec` (str): Audio codec: "PCMU", "PCMA", "G722" (default: "PCMU")
- `rtp_ptime` (int): RTP packet time in milliseconds (default: 20)
- `status_url` (Optional[str]): Status webhook URL

**Usage:**
```python
# Basic call tapping
result.tap("sip:monitor@company.com")

# Tap with specific settings
result.tap(
    uri="sip:quality@company.com",
    control_id="quality_monitor_001",
    direction="both",
    codec="G722"
)
```

##### `stop_tap(control_id: Optional[str] = None) -> SwaigFunctionResult`
Stop call tapping.

**Parameters:**
- `control_id` (Optional[str]): Control ID of tap to stop

**Usage:**
```python
result.stop_tap()
result.stop_tap(control_id="quality_monitor_001")
```

### Advanced SWML Execution

##### `execute_swml(swml_content, transfer: bool = False) -> SwaigFunctionResult`
Execute custom SWML content.

**Parameters:**
- `swml_content`: SWML document or content to execute
- `transfer` (bool): Whether this is a transfer operation (default: False)

**Usage:**
```python
# Execute custom SWML
custom_swml = {
    "version": "1.0.0",
    "sections": {
        "main": [
            {"play": {"url": "https://example.com/custom.mp3"}},
            {"say": {"text": "Custom SWML execution"}}
        ]
    }
}
result.execute_swml(custom_swml)
```

### Utility Methods

##### `to_dict() -> Dict[str, Any]`
Convert the result to a dictionary for serialization.

**Returns:**
- Dict[str, Any]: Dictionary representation of the result

**Usage:**
```python
result = SwaigFunctionResult("Hello world")
result.add_action("play", "music.mp3")
result_dict = result.to_dict()
print(result_dict)
# Output: {"response": "Hello world", "action": [{"play": "music.mp3"}]}
```

### Static Helper Methods

##### `create_payment_prompt(for_situation: str, actions: List[Dict[str, str]], card_type: Optional[str] = None, error_type: Optional[str] = None) -> Dict[str, Any]`
Create a payment prompt configuration.

**Parameters:**
- `for_situation` (str): Situation identifier
- `actions` (List[Dict[str, str]]): List of action configurations
- `card_type` (Optional[str]): Card type for prompts
- `error_type` (Optional[str]): Error type for error prompts

**Usage:**
```python
prompt = SwaigFunctionResult.create_payment_prompt(
    for_situation="card_number",
    actions=[
        SwaigFunctionResult.create_payment_action("say", "Please enter your card number")
    ]
)
```

##### `create_payment_action(action_type: str, phrase: str) -> Dict[str, str]`
Create a payment action configuration.

**Parameters:**
- `action_type` (str): Action type
- `phrase` (str): Action phrase

**Usage:**
```python
action = SwaigFunctionResult.create_payment_action("say", "Enter your card number")
```

##### `create_payment_parameter(name: str, value: str) -> Dict[str, str]`
Create a payment parameter configuration.

**Parameters:**
- `name` (str): Parameter name
- `value` (str): Parameter value

**Usage:**
```python
param = SwaigFunctionResult.create_payment_parameter("merchant_id", "12345")
```

### Method Chaining

All methods return `self`, enabling fluent method chaining:

```python
result = (SwaigFunctionResult("I'll help you with that")
    .set_post_process(True)
    .update_global_data({"status": "helping"})
    .set_end_of_speech_timeout(800)
    .add_action("play", "thinking.mp3"))

# Complex workflow
result = (SwaigFunctionResult("Processing your payment")
    .set_post_process(True)
    .update_global_data({"payment_status": "processing"})
    .pay(
        payment_connector_url="https://payments.com/webhook",
        charge_amount="99.99",
        description="Service payment"
    )
    .send_sms(
        to_number="+15551234567",
        from_number="+15559876543",
        body="Payment confirmation will be sent shortly"
    ))
```

This concludes Part 2 of the API reference covering the SwaigFunctionResult class. The document will continue with DataMap and other components in subsequent parts.

---

## DataMap Class

The `DataMap` class provides a declarative approach to creating SWAIG tools that integrate with REST APIs without requiring webhook infrastructure. DataMap tools execute on SignalWire's server infrastructure, eliminating the need to expose webhook endpoints.

### Constructor

```python
DataMap(function_name: str)
```

**Parameters:**
- `function_name` (str): Name of the SWAIG function this DataMap will create

**Usage:**
```python
# Create a new DataMap tool
weather_map = DataMap('get_weather')
search_map = DataMap('search_docs')
```

### Core Configuration Methods

#### Function Metadata

##### `purpose(description: str) -> DataMap`
Set the function description/purpose.

**Parameters:**
- `description` (str): Human-readable description of what this function does

**Usage:**
```python
data_map = DataMap('get_weather').purpose('Get current weather information for any city')
```

##### `description(description: str) -> DataMap`
Alias for `purpose()` - set the function description.

**Parameters:**
- `description` (str): Function description

**Usage:**
```python
data_map = DataMap('search_api').description('Search our knowledge base for information')
```

#### Parameter Definition

##### `parameter(name: str, param_type: str, description: str, required: bool = False, enum: Optional[List[str]] = None) -> DataMap`
Add a function parameter with JSON schema validation.

**Parameters:**
- `name` (str): Parameter name
- `param_type` (str): JSON schema type: "string", "number", "boolean", "array", "object"
- `description` (str): Parameter description for the AI
- `required` (bool): Whether parameter is required (default: False)
- `enum` (Optional[List[str]]): List of allowed values for validation

**Usage:**
```python
# Required string parameter
data_map.parameter('location', 'string', 'City name or ZIP code', required=True)

# Optional number parameter
data_map.parameter('days', 'number', 'Number of forecast days', required=False)

# Enum parameter with allowed values
data_map.parameter('units', 'string', 'Temperature units', 
                  enum=['celsius', 'fahrenheit'], required=False)

# Boolean parameter
data_map.parameter('include_alerts', 'boolean', 'Include weather alerts', required=False)

# Array parameter
data_map.parameter('categories', 'array', 'Search categories to include')
```

### API Integration Methods

#### HTTP Webhook Configuration

##### `webhook(method: str, url: str, headers: Optional[Dict[str, str]] = None, form_param: Optional[str] = None, input_args_as_params: bool = False, require_args: Optional[List[str]] = None) -> DataMap`
Configure an HTTP API call.

**Parameters:**
- `method` (str): HTTP method: "GET", "POST", "PUT", "DELETE", "PATCH"
- `url` (str): API endpoint URL (supports `${variable}` substitution)
- `headers` (Optional[Dict[str, str]]): HTTP headers to send
- `form_param` (Optional[str]): Send JSON body as single form parameter with this name
- `input_args_as_params` (bool): Merge function arguments into URL parameters (default: False)
- `require_args` (Optional[List[str]]): Only execute if these arguments are present

**Variable Substitution in URLs:**
- `${args.parameter_name}`: Function argument values
- `${global_data.key}`: Call-wide data store (user info, call state - NOT credentials)
- `${meta_data.call_id}`: Call and function metadata

**Usage:**
```python
# Simple GET request with parameter substitution
data_map.webhook('GET', 'https://api.weather.com/v1/current?key=API_KEY&q=${args.location}')

# POST request with authentication headers
data_map.webhook(
    'POST', 
    'https://api.company.com/search',
    headers={
        'Authorization': 'Bearer YOUR_TOKEN',
        'Content-Type': 'application/json'
    }
)

# Webhook that requires specific arguments
data_map.webhook(
    'GET',
    'https://api.service.com/data?id=${args.customer_id}',
    require_args=['customer_id']
)

# Use global data for call-related info (NOT credentials)
data_map.webhook(
    'GET',
    'https://api.service.com/customer/${global_data.customer_id}/orders',
    headers={'Authorization': 'Bearer YOUR_API_TOKEN'}  # Use static credentials
)
```

##### `body(data: Dict[str, Any]) -> DataMap`
Set the JSON body for POST/PUT requests.

**Parameters:**
- `data` (Dict[str, Any]): JSON body data (supports `${variable}` substitution)

**Usage:**
```python
# Static body with parameter substitution
data_map.body({
    'query': '${args.search_term}',
    'limit': 5,
    'filters': {
        'category': '${args.category}',
        'active': True
    }
})

# Body with call-related data (NOT sensitive info)
data_map.body({
    'customer_id': '${global_data.customer_id}',
    'request_id': '${meta_data.call_id}',
    'search': '${args.query}'
})
```

##### `params(data: Dict[str, Any]) -> DataMap`
Set URL query parameters.

**Parameters:**
- `data` (Dict[str, Any]): Query parameters (supports `${variable}` substitution)

**Usage:**
```python
# URL parameters with substitution
data_map.params({
    'api_key': 'YOUR_API_KEY',
    'q': '${args.location}',
    'units': '${args.units}',
    'lang': 'en'
})
```

#### Multiple Webhooks and Fallbacks

DataMap supports multiple webhook configurations for fallback scenarios:

```python
# Primary API with fallback
data_map = (DataMap('search_with_fallback')
    .purpose('Search with multiple API fallbacks')
    .parameter('query', 'string', 'Search query', required=True)
    
    # Primary API
    .webhook('GET', 'https://api.primary.com/search?q=${args.query}')
    .output(SwaigFunctionResult('Primary result: ${response.title}'))
    
    # Fallback API
    .webhook('GET', 'https://api.fallback.com/search?q=${args.query}')
    .output(SwaigFunctionResult('Fallback result: ${response.title}'))
    
    # Final fallback if all APIs fail
    .fallback_output(SwaigFunctionResult('Sorry, all search services are currently unavailable'))
)
```

### Response Processing

#### Basic Output

##### `output(result: SwaigFunctionResult) -> DataMap`
Set the response template for successful API calls.

**Parameters:**
- `result` (SwaigFunctionResult): Response template with variable substitution

**Variable Substitution in Outputs:**
- `${response.field}`: API response fields
- `${response.nested.field}`: Nested response fields
- `${response.array[0].field}`: Array element fields
- `${args.parameter}`: Original function arguments
- `${global_data.key}`: Call-wide data store (user info, call state)

**Usage:**
```python
# Simple response template
data_map.output(SwaigFunctionResult('Weather in ${args.location}: ${response.current.condition.text}, ${response.current.temp_f}°F'))

# Response with actions
data_map.output(
    SwaigFunctionResult('Found ${response.total_results} results')
    .update_global_data({'last_search': '${args.query}'})
    .add_action('play', 'search_complete.mp3')
)

# Complex response with nested data
data_map.output(
    SwaigFunctionResult('Order ${response.order.id} status: ${response.order.status}. Estimated delivery: ${response.order.delivery.estimated_date}')
)
```

##### `fallback_output(result: SwaigFunctionResult) -> DataMap`
Set the response when all webhooks fail.

**Parameters:**
- `result` (SwaigFunctionResult): Fallback response

**Usage:**
```python
data_map.fallback_output(
    SwaigFunctionResult('Sorry, the service is temporarily unavailable. Please try again later.')
    .add_action('play', 'service_unavailable.mp3')
)
```

#### Array Processing

##### `foreach(foreach_config: Union[str, Dict[str, Any]]) -> DataMap`
Process array responses by iterating over elements.

**Parameters:**
- `foreach_config` (Union[str, Dict]): Array path or configuration object

**Simple Array Processing:**
```python
# Process array of search results
data_map = (DataMap('search_docs')
    .webhook('GET', 'https://api.docs.com/search?q=${args.query}')
    .foreach('${response.results}')  # Iterate over results array
    .output(SwaigFunctionResult('Found: ${foreach.title} - ${foreach.summary}'))
)
```

**Advanced Array Processing:**
```python
# Complex foreach configuration
data_map.foreach({
    'array': '${response.items}',
    'limit': 3,  # Process only first 3 items
    'filter': {
        'field': 'status',
        'value': 'active'
    }
})
```

**Foreach Variable Access:**
- `${foreach.field}`: Current array element field
- `${foreach.nested.field}`: Nested fields in current element
- `${foreach_index}`: Current iteration index (0-based)
- `${foreach_count}`: Total number of items being processed

### Pattern-Based Processing

#### Expression Matching

##### `expression(test_value: str, pattern: Union[str, Pattern], output: SwaigFunctionResult, nomatch_output: Optional[SwaigFunctionResult] = None) -> DataMap`
Add pattern-based responses without API calls.

**Parameters:**
- `test_value` (str): Template string to test against pattern
- `pattern` (Union[str, Pattern]): Regex pattern or compiled Pattern object
- `output` (SwaigFunctionResult): Response when pattern matches
- `nomatch_output` (Optional[SwaigFunctionResult]): Response when pattern doesn't match

**Usage:**
```python
# Command-based responses
control_map = (DataMap('file_control')
    .purpose('Control file playback')
    .parameter('command', 'string', 'Playback command', required=True)
    .parameter('filename', 'string', 'File to control')
    
    # Start commands
    .expression(
        '${args.command}', 
        r'start|play|begin',
        SwaigFunctionResult('Starting playback')
        .add_action('start_playback', {'file': '${args.filename}'})
    )
    
    # Stop commands
    .expression(
        '${args.command}',
        r'stop|pause|halt',
        SwaigFunctionResult('Stopping playback')
        .add_action('stop_playback', True)
    )
    
    # Volume commands
    .expression(
        '${args.command}',
        r'volume (\d+)',
        SwaigFunctionResult('Setting volume to ${match.1}')
        .add_action('set_volume', '${match.1}')
    )
)
```

**Pattern Matching Variables:**
- `${match.0}`: Full match
- `${match.1}`, `${match.2}`, etc.: Capture groups
- `${match.group_name}`: Named capture groups

### Error Handling

##### `error_keys(keys: List[str]) -> DataMap`
Specify response fields that indicate errors.

**Parameters:**
- `keys` (List[str]): List of field names that indicate API errors

**Usage:**
```python
# Treat these response fields as errors
data_map.error_keys(['error', 'error_message', 'status_code'])

# If API returns {"error": "Not found"}, DataMap will treat this as an error
```

##### `global_error_keys(keys: List[str]) -> DataMap`
Set global error keys for all webhooks in this DataMap.

**Parameters:**
- `keys` (List[str]): Global error field names

**Usage:**
```python
data_map.global_error_keys(['error', 'message', 'code'])
```

### Advanced Configuration

##### `webhook_expressions(expressions: List[Dict[str, Any]]) -> DataMap`
Add expression-based webhook selection.

**Parameters:**
- `expressions` (List[Dict[str, Any]]): List of expression configurations

**Usage:**
```python
# Different APIs based on input
data_map.webhook_expressions([
    {
        'test': '${args.type}',
        'pattern': 'weather',
        'webhook': {
            'method': 'GET',
            'url': 'https://weather-api.com/current?q=${args.location}'
        }
    },
    {
        'test': '${args.type}',
        'pattern': 'news',
        'webhook': {
            'method': 'GET', 
            'url': 'https://news-api.com/search?q=${args.query}'
        }
    }
])
```

### Complete DataMap Examples

#### Simple Weather API

```python
weather_tool = (DataMap('get_weather')
    .purpose('Get current weather information')
    .parameter('location', 'string', 'City name or ZIP code', required=True)
    .parameter('units', 'string', 'Temperature units', enum=['celsius', 'fahrenheit'])
    .webhook('GET', 'https://api.weather.com/v1/current?key=API_KEY&q=${args.location}&units=${args.units}')
    .output(SwaigFunctionResult('Weather in ${args.location}: ${response.current.condition.text}, ${response.current.temp_f}°F'))
    .error_keys(['error'])
)

# Register with agent
agent.register_swaig_function(weather_tool.to_swaig_function())
```

#### Search with Array Processing

```python
search_tool = (DataMap('search_knowledge')
    .purpose('Search company knowledge base')
    .parameter('query', 'string', 'Search query', required=True)
    .parameter('category', 'string', 'Search category', enum=['docs', 'faq', 'policies'])
    .webhook(
        'POST', 
        'https://api.company.com/search',
        headers={'Authorization': 'Bearer TOKEN'}
    )
    .body({
        'query': '${args.query}',
        'category': '${args.category}',
        'limit': 5
    })
    .foreach('${response.results}')
    .output(SwaigFunctionResult('Found: ${foreach.title} - ${foreach.summary}'))
    .fallback_output(SwaigFunctionResult('Search service is temporarily unavailable'))
)
```

#### Command Processing (No API)

```python
control_tool = (DataMap('system_control')
    .purpose('Control system functions')
    .parameter('action', 'string', 'Action to perform', required=True)
    .parameter('target', 'string', 'Target for the action')
    
    # Restart commands
    .expression(
        '${args.action}',
        r'restart|reboot',
        SwaigFunctionResult('Restarting ${args.target}')
        .add_action('restart_service', {'service': '${args.target}'})
    )
    
    # Status commands
    .expression(
        '${args.action}',
        r'status|check',
        SwaigFunctionResult('Checking status of ${args.target}')
        .add_action('check_status', {'service': '${args.target}'})
    )
    
    # Default for unrecognized commands
    .expression(
        '${args.action}',
        r'.*',
        SwaigFunctionResult('Unknown command: ${args.action}'),
        nomatch_output=SwaigFunctionResult('Please specify a valid action')
    )
)
```

### Conversion and Registration

##### `to_swaig_function() -> Dict[str, Any]`
Convert the DataMap to a SWAIG function dictionary for registration.

**Returns:**
- Dict[str, Any]: Complete SWAIG function definition

**Usage:**
```python
# Build DataMap
weather_map = DataMap('get_weather').purpose('Get weather').parameter('location', 'string', 'City', required=True)

# Convert to SWAIG function and register
swaig_function = weather_map.to_swaig_function()
agent.register_swaig_function(swaig_function)
```

### Convenience Functions

The SDK provides helper functions for common DataMap patterns:

##### `create_simple_api_tool(name: str, url: str, response_template: str, parameters: Optional[Dict[str, Dict]] = None, method: str = "GET", headers: Optional[Dict[str, str]] = None, body: Optional[Dict[str, Any]] = None, error_keys: Optional[List[str]] = None) -> DataMap`

Create a simple API integration tool.

**Parameters:**
- `name` (str): Function name
- `url` (str): API endpoint URL
- `response_template` (str): Response template string
- `parameters` (Optional[Dict[str, Dict]]): Parameter definitions
- `method` (str): HTTP method (default: "GET")
- `headers` (Optional[Dict[str, str]]): HTTP headers
- `body` (Optional[Dict[str, Any]]): Request body
- `error_keys` (Optional[List[str]]): Error field names

**Usage:**
```python
from signalwire_agents.core.data_map import create_simple_api_tool

weather = create_simple_api_tool(
    name='get_weather',
    url='https://api.weather.com/v1/current?key=API_KEY&q=${location}',
    response_template='Weather in ${location}: ${response.current.condition.text}',
    parameters={
        'location': {
            'type': 'string', 
            'description': 'City name', 
            'required': True
        }
    }
)

agent.register_swaig_function(weather.to_swaig_function())
```

##### `create_expression_tool(name: str, patterns: Dict[str, Tuple[str, SwaigFunctionResult]], parameters: Optional[Dict[str, Dict]] = None) -> DataMap`

Create a pattern-based tool without API calls.

**Parameters:**
- `name` (str): Function name
- `patterns` (Dict[str, Tuple[str, SwaigFunctionResult]]): Pattern mappings
- `parameters` (Optional[Dict[str, Dict]]): Parameter definitions

**Usage:**
```python
from signalwire_agents.core.data_map import create_expression_tool

file_control = create_expression_tool(
    name='file_control',
    patterns={
        r'start.*': ('${args.command}', SwaigFunctionResult().add_action('start_playback', True)),
        r'stop.*': ('${args.command}', SwaigFunctionResult().add_action('stop_playback', True))
    },
    parameters={
        'command': {
            'type': 'string',
            'description': 'Playback command',
            'required': True
        }
    }
)

agent.register_swaig_function(file_control.to_swaig_function())
```

### Method Chaining

All DataMap methods return `self`, enabling fluent method chaining:

```python
complete_tool = (DataMap('comprehensive_search')
    .purpose('Comprehensive search with fallbacks')
    .parameter('query', 'string', 'Search query', required=True)
    .parameter('category', 'string', 'Search category', enum=['all', 'docs', 'faq'])
    .webhook('GET', 'https://primary-api.com/search?q=${args.query}&cat=${args.category}')
    .output(SwaigFunctionResult('Primary: ${response.title}'))
    .webhook('GET', 'https://backup-api.com/search?q=${args.query}')
    .output(SwaigFunctionResult('Backup: ${response.title}'))
    .fallback_output(SwaigFunctionResult('All search services unavailable'))
    .error_keys(['error', 'message'])
)
```

This concludes Part 3 of the API reference covering the DataMap class. The document will continue with Context System and other components in subsequent parts. 

---

## Context System

The Context System enhances traditional prompt-based agents by adding structured workflows with sequential steps on top of a base prompt. Each step contains its own guidance, completion criteria, and function restrictions while building upon the agent's foundational prompt.

### ContextBuilder Class

The `ContextBuilder` is accessed via `agent.define_contexts()` and provides the main interface for creating structured workflows.

#### Getting Started

```python
# Access the context builder
contexts = agent.define_contexts()

# Create contexts and steps
contexts.add_context("greeting") \
    .add_step("welcome") \
    .set_text("Welcome! How can I help you today?") \
    .set_step_criteria("User has stated their need") \
    .set_valid_steps(["next"])
```

##### `add_context(name: str) -> Context`
Create a new context in the workflow.

**Parameters:**
- `name` (str): Unique context name

**Returns:**
- Context: Context object for method chaining

**Usage:**
```python
# Create multiple contexts
greeting_context = contexts.add_context("greeting")
main_menu_context = contexts.add_context("main_menu")
support_context = contexts.add_context("support")
```

### Context Class

The Context class represents a conversation context containing multiple steps with enhanced features:

```python
class Context:
    def add_step(self, name: str) -> Step
        """Create a new step in this context"""
    
    def set_valid_contexts(self, contexts: List[str]) -> Context
        """Set which contexts can be accessed from this context"""
        
    # Context entry parameters (for context switching behavior)
    def set_post_prompt(self, post_prompt: str) -> Context
        """Override agent's post prompt when this context is active"""
    
    def set_system_prompt(self, system_prompt: str) -> Context
        """Trigger context switch with new system instructions (makes this a Context Switch Context)"""
        
    def set_consolidate(self, consolidate: bool) -> Context
        """Whether to consolidate conversation history when entering this context"""
        
    def set_full_reset(self, full_reset: bool) -> Context
        """Whether to do complete system prompt replacement vs injection"""
        
    def set_user_prompt(self, user_prompt: str) -> Context
        """User message to inject when entering this context for AI context"""
    
    # Context prompts (guidance for all steps in context)
    def set_prompt(self, prompt: str) -> Context
        """Set simple string prompt that applies to all steps in this context"""
        
    def add_section(self, title: str, body: str) -> Context
        """Add POM-style section to context prompt"""
        
    def add_bullets(self, title: str, bullets: List[str]) -> Context
        """Add POM-style bullet section to context prompt"""
```

**Context Types:**

1. **Workflow Container Context** (no `system_prompt`): Organizes steps without conversation state changes
2. **Context Switch Context** (has `system_prompt`): Triggers conversation state changes when entered, processing entry parameters like a `context_switch` SWAIG action

**Prompt Hierarchy:** Base Agent Prompt → Context Prompt → Step Prompt

#### Usage Examples

```python
# Workflow container context (just organizes steps)
main_context = contexts.add_context("main")
main_context.set_prompt("Follow standard customer service protocols")

# Context switch context (changes AI behavior)  
billing_context = contexts.add_context("billing")
billing_context.set_system_prompt("You are now a billing specialist") \
    .set_consolidate(True) \
    .set_user_prompt("Customer needs billing assistance") \
    .add_section("Department", "Billing Department") \
    .add_bullets("Services", ["Account inquiries", "Payments", "Refunds"])

# Full reset context (complete conversation reset)
manager_context = contexts.add_context("manager") 
manager_context.set_system_prompt("You are a senior manager") \
    .set_full_reset(True) \
    .set_consolidate(True)
```

---

## Skills System

The Skills System provides modular, reusable capabilities that can be easily added to any agent.

### Available Built-in Skills

#### `datetime` Skill
Provides current date and time information.

**Parameters:**
- `timezone` (Optional[str]): Timezone for date/time (default: system timezone)
- `format` (Optional[str]): Custom date/time format string

**Usage:**
```python
# Basic datetime skill
agent.add_skill("datetime")

# With timezone
agent.add_skill("datetime", {"timezone": "America/New_York"})

# With custom format
agent.add_skill("datetime", {
    "timezone": "UTC",
    "format": "%Y-%m-%d %H:%M:%S %Z"
})
```

#### `math` Skill
Safe mathematical expression evaluation.

**Parameters:**
- `precision` (Optional[int]): Decimal precision for results (default: 2)
- `max_expression_length` (Optional[int]): Maximum expression length (default: 100)

**Usage:**
```python
# Basic math skill
agent.add_skill("math")

# With custom precision
agent.add_skill("math", {"precision": 4})
```

#### `web_search` Skill
Google Custom Search API integration with web scraping.

**Parameters:**
- `api_key` (str): Google Custom Search API key (required)
- `search_engine_id` (str): Google Custom Search Engine ID (required)
- `num_results` (Optional[int]): Number of results to return (default: 3)
- `tool_name` (Optional[str]): Custom tool name for multiple instances
- `delay` (Optional[float]): Delay between requests in seconds
- `no_results_message` (Optional[str]): Custom message when no results found

**Usage:**
```python
# Basic web search
agent.add_skill("web_search", {
    "api_key": "your-google-api-key",
    "search_engine_id": "your-search-engine-id"
})

# Multiple search instances
agent.add_skill("web_search", {
    "api_key": "your-api-key",
    "search_engine_id": "general-engine-id",
    "tool_name": "search_general",
    "num_results": 5
})

agent.add_skill("web_search", {
    "api_key": "your-api-key",
    "search_engine_id": "news-engine-id",
    "tool_name": "search_news",
    "num_results": 3,
    "delay": 0.5
})
```

#### `datasphere` Skill
SignalWire DataSphere knowledge search integration.

**Parameters:**
- `space_name` (str): DataSphere space name (required)
- `project_id` (str): DataSphere project ID (required)
- `token` (str): DataSphere access token (required)
- `document_id` (Optional[str]): Specific document to search
- `tool_name` (Optional[str]): Custom tool name for multiple instances
- `count` (Optional[int]): Number of results to return (default: 3)
- `tags` (Optional[List[str]]): Filter by document tags

**Usage:**
```python
# Basic DataSphere search
agent.add_skill("datasphere", {
    "space_name": "my-space",
    "project_id": "my-project",
    "token": "my-token"
})

# Multiple DataSphere instances
agent.add_skill("datasphere", {
    "space_name": "my-space",
    "project_id": "my-project",
    "token": "my-token",
    "document_id": "drinks-menu",
    "tool_name": "search_drinks",
    "count": 5
})

agent.add_skill("datasphere", {
    "space_name": "my-space",
    "project_id": "my-project", 
    "token": "my-token",
    "tool_name": "search_policies",
    "tags": ["HR", "Policies"]
})
```

#### `native_vector_search` Skill
Local document search with vector similarity and keyword search.

**Parameters:**
- `index_path` (str): Path to search index file (required)
- `tool_name` (Optional[str]): Custom tool name (default: "search_documents")
- `max_results` (Optional[int]): Maximum results to return (default: 5)
- `similarity_threshold` (Optional[float]): Minimum similarity score 0.0-1.0 (default: 0.0). Higher values are stricter, lower values are more permissive. Typical range: 0.2-0.4 for all-MiniLM-L6-v2, 0.3-0.5 for all-mpnet-base-v2

**Usage:**
```python
# Basic local search
agent.add_skill("native_vector_search", {
    "index_path": "./knowledge.swsearch"
})

# With custom settings
agent.add_skill("native_vector_search", {
    "index_path": "./docs.swsearch",
    "tool_name": "search_docs",
    "max_results": 10,
    "similarity_threshold": 0.25
})
```

### Creating Custom Skills

#### Skill Structure

Create a new skill by extending `SkillBase`:

```python
from signalwire_agents.core.skill_base import SkillBase
from signalwire_agents.core.data_map import DataMap
from signalwire_agents.core.function_result import SwaigFunctionResult

class CustomSkill(SkillBase):
    SKILL_NAME = "custom_skill"
    SKILL_DESCRIPTION = "Description of what this skill does"
    SKILL_VERSION = "1.0.0"
    REQUIRED_PACKAGES = ["requests"]  # Python packages needed
    REQUIRED_ENV_VARS = ["API_KEY"]   # Environment variables needed
    
    def setup(self) -> bool:
        """Validate and store configuration"""
        if not self.params.get("api_key"):
            self.logger.error("api_key parameter is required")
            return False
        
        self.api_key = self.params["api_key"]
        return True
    
    def register_tools(self) -> None:
        """Register skill functions"""
        # DataMap-based tool
        tool = (DataMap("custom_function")
            .description("Custom API integration")
            .parameter("query", "string", "Search query", required=True)
            .webhook("GET", f"https://api.example.com/search?key={self.api_key}&q=${{args.query}}")
            .output(SwaigFunctionResult("Found: ${{response.title}}"))
        )
        
        self.agent.register_swaig_function(tool.to_swaig_function())
    
    def get_hints(self) -> List[str]:
        """Speech recognition hints"""
        return ["custom search", "find information"]
    
    def get_global_data(self) -> Dict[str, Any]:
        """Global data for DataMap"""
        return {"skill_version": self.SKILL_VERSION}
    
    def get_prompt_sections(self) -> List[Dict[str, Any]]:
        """Prompt sections to add"""
        return [{
            "title": "Custom Search Capability",
            "body": "You can search our custom database for information.",
            "bullets": ["Use the custom_function to search", "Results are real-time"]
        }]
```

#### Skill Registration

Skills are automatically discovered from the `signalwire_agents/skills/` directory. To register a custom skill:

1. Create directory: `signalwire_agents/skills/your_skill/`
2. Add `__init__.py`, `skill.py`, and `README.md`
3. Implement your skill class in `skill.py`
4. The skill will be automatically available

---

## Utility Classes

### SWAIGFunction Class

Represents a SWAIG function definition with metadata and validation.

#### Constructor

```python
SWAIGFunction(
    function: str,
    description: str,
    parameters: Dict[str, Any],
    **kwargs
)
```

**Parameters:**
- `function` (str): Function name
- `description` (str): Function description
- `parameters` (Dict[str, Any]): JSON schema for parameters
- `**kwargs`: Additional SWAIG properties

#### Usage

```python
from signalwire_agents.core.swaig_function import SWAIGFunction

# Create SWAIG function
swaig_func = SWAIGFunction(
    function="get_weather",
    description="Get current weather",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"]
    },
    secure=True,
    fillers={"en-US": ["Checking weather..."]}
)

# Register with agent
agent.register_swaig_function(swaig_func.to_dict())
```

### SWMLService Class

Base class providing SWML document generation and HTTP service capabilities. `AgentBase` extends this class.

#### Key Methods

##### `get_swml_document() -> Dict[str, Any]`
Generate the complete SWML document for the service.

##### `handle_request(request_data: Dict[str, Any]) -> Dict[str, Any]`
Handle incoming HTTP requests and generate appropriate responses.

### Dynamic Configuration

The dynamic configuration callback receives the agent instance directly, allowing you to configure it based on request data.

**Usage:**
```python
def dynamic_config(query_params, body_params, headers, agent):
    # Configure based on request
    if query_params.get("lang") == "es":
        agent.add_language("Spanish", "es-ES", "nova.luna")
    
    # Customer-specific configuration
    customer_id = headers.get("X-Customer-ID")
    if customer_id:
        agent.set_global_data({"customer_id": customer_id})
        agent.prompt_add_section("Customer Context", f"You are helping customer {customer_id}")
    
    # Add skills dynamically
    if query_params.get("enable_search") == "true":
        agent.add_skill("web_search", {"provider": "google"})

agent.set_dynamic_config_callback(dynamic_config)
```

---

## Environment Variables

The SDK supports various environment variables for configuration:

### Authentication
- `SWML_BASIC_AUTH_USER`: Basic auth username
- `SWML_BASIC_AUTH_PASSWORD`: Basic auth password

### SSL/HTTPS
- `SWML_SSL_ENABLED`: Enable SSL (true/false)
- `SWML_SSL_CERT_PATH`: Path to SSL certificate
- `SWML_SSL_KEY_PATH`: Path to SSL private key
- `SWML_DOMAIN`: Domain name for SSL

### Proxy Support
- `SWML_PROXY_URL_BASE`: Base URL for proxy server

### Skills Configuration
- `GOOGLE_SEARCH_API_KEY`: Google Custom Search API key
- `GOOGLE_SEARCH_ENGINE_ID`: Google Custom Search Engine ID
- `DATASPHERE_SPACE_NAME`: DataSphere space name
- `DATASPHERE_PROJECT_ID`: DataSphere project ID
- `DATASPHERE_TOKEN`: DataSphere access token

### Usage

```python
import os

# Set environment variables
os.environ["SWML_BASIC_AUTH_USER"] = "admin"
os.environ["SWML_BASIC_AUTH_PASSWORD"] = "secret"
os.environ["GOOGLE_SEARCH_API_KEY"] = "your-api-key"

# Agent will automatically use these
agent = AgentBase("My Agent")
agent.add_skill("web_search", {
    "search_engine_id": "your-engine-id"
    # api_key will be read from environment
})
```

---

## Complete Example

Here's a comprehensive example using multiple SDK components:

```python
from signalwire_agents import AgentBase, SwaigFunctionResult, DataMap

class ComprehensiveAgent(AgentBase):
    def __init__(self):
        super().__init__(
            name="Comprehensive Agent",
            auto_answer=True,
            record_call=True
        )
        
        # Configure voice and language
        self.add_language("English", "en-US", "rime.spore",
                         speech_fillers=["Let me check...", "One moment..."])
        
        # Add speech recognition hints
        self.add_hints(["SignalWire", "customer service", "technical support"])
        
        # Configure AI parameters
        self.set_params({
            "ai_model": "gpt-4.1-nano",
            "end_of_speech_timeout": 800,
            "temperature": 0.7
        })
        
        # Add skills
        self.add_skill("datetime")
        self.add_skill("math")
        self.add_skill("web_search", {
            "api_key": "your-google-api-key",
            "search_engine_id": "your-engine-id",
            "num_results": 3
        })
        
        # Set up structured workflow
        self._setup_contexts()
        
        # Add custom tools
        self._register_custom_tools()
        
        # Set global data
        self.set_global_data({
            "company_name": "Acme Corp",
            "support_hours": "9 AM - 5 PM EST",
            "version": "2.0"
        })
    
    def _setup_contexts(self):
        """Set up structured workflow contexts"""
        contexts = self.define_contexts()
        
        # Greeting context
        greeting = contexts.add_context("greeting")
        greeting.add_step("welcome") \
            .set_text("Hello! Welcome to Acme Corp support. How can I help you today?") \
            .set_step_criteria("Customer has explained their issue") \
            .set_valid_steps(["next"])
        
        greeting.add_step("categorize") \
            .add_section("Current Task", "Categorize the customer's request") \
            .add_bullets("Categories", [
                "Technical issue - use diagnostic tools",
                "Billing question - transfer to billing",
                "General inquiry - handle directly"
            ]) \
            .set_functions(["transfer_to_billing", "run_diagnostics"]) \
            .set_step_criteria("Request categorized and action taken")
        
        # Technical support context
        tech = contexts.add_context("technical_support")
        tech.add_step("diagnose") \
            .set_text("Let me run some diagnostics to identify the issue.") \
            .set_functions(["run_diagnostics", "check_system_status"]) \
            .set_step_criteria("Diagnostics completed") \
            .set_valid_steps(["resolve"])
        
        tech.add_step("resolve") \
            .set_text("Based on the diagnostics, here's how we'll fix this.") \
            .set_functions(["apply_fix", "schedule_technician"]) \
            .set_step_criteria("Issue resolved or escalated")
    
    def _register_custom_tools(self):
        """Register custom DataMap tools"""
        
        # Customer lookup tool
        lookup_tool = (DataMap("lookup_customer")
            .description("Look up customer information")
            .parameter("customer_id", "string", "Customer ID", required=True)
            .webhook("GET", "https://api.company.com/customers/${args.customer_id}",
                    headers={"Authorization": "Bearer YOUR_TOKEN"})
            .output(SwaigFunctionResult("Customer: ${response.name}, Status: ${response.status}"))
            .error_keys(["error"])
        )
        
        self.register_swaig_function(lookup_tool.to_swaig_function())
        
        # System control tool
        control_tool = (DataMap("system_control")
            .description("Control system functions")
            .parameter("action", "string", "Action to perform", required=True)
            .parameter("target", "string", "Target system")
            .expression("${args.action}", r"restart|reboot",
                       SwaigFunctionResult("Restarting ${args.target}")
                       .add_action("restart_system", {"target": "${args.target}"}))
            .expression("${args.action}", r"status|check",
                       SwaigFunctionResult("Checking ${args.target} status")
                       .add_action("check_status", {"target": "${args.target}"}))
        )
        
        self.register_swaig_function(control_tool.to_swaig_function())
    
    @AgentBase.tool(
        description="Transfer call to billing department",
        parameters={"type": "object", "properties": {}}
    )
    def transfer_to_billing(self, args, raw_data):
        """Transfer to billing with state tracking"""
        return (SwaigFunctionResult("Transferring you to our billing department")
                .update_global_data({"last_action": "transfer_to_billing"})
                .connect("billing@company.com", final=False))
    
    def on_summary(self, summary, raw_data):
        """Handle conversation summaries"""
        print(f"Conversation completed: {summary}")
        # Could save to database, send notifications, etc.

# Run the agent
if __name__ == "__main__":
    agent = ComprehensiveAgent()
    agent.run()
```

This concludes the complete API reference for the SignalWire AI Agents SDK. The SDK provides a comprehensive framework for building sophisticated AI agents with modular capabilities, structured workflows, persistent state, and deployment across multiple environments.