<!-- Header -->
<div align="center">
    <a href="https://signalwire.com" target="_blank">
        <img src="https://github.com/user-attachments/assets/0c8ed3b9-8c50-4dc6-9cc4-cc6cd137fd50" width="500" />
    </a>

# Agents SDK

#### _A Python SDK for creating, hosting, and securing SignalWire AI agents as microservices with minimal boilerplate._

<br/>

<p align="center">
  <a href="https://developer.signalwire.com/sdks/agents-sdk" target="_blank">📖 Documentation</a> &nbsp; &nbsp; <code>#</code> &nbsp; &nbsp;
  <a href="https://github.com/signalwire/signalwire-docs/issues/new/choose" target="_blank">🐛 Report an issue</a> &nbsp; &nbsp; <code>#</code> &nbsp; &nbsp;
  <a href="https://pypi.org/project/signalwire-agents/" target="_blank">🐍 PyPI</a>
</p>

<br/>

<!-- Badges -->
<div align="center">
    <a href="https://discord.com/invite/F2WNYTNjuF" target="_blank"><img src="https://img.shields.io/badge/Discord%20Community-5865F2" alt="Discord" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/MIT-License-blue" alt="MIT License" /></a>
    <a href="https://github.com/signalwire" target="_blank"><img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white&" alt="GitHub" /></a>
    <a href="https://github.com/signalwire/docs" target="_blank"><img src="https://img.shields.io/github/stars/signalwire/signalwire-agents" alt="GitHub Stars" /></a>
</div>

<br/>

<a href="https://signalwire.com/signup" target="_blank">
    <img src="https://github.com/user-attachments/assets/c2510c86-ae03-42a9-be06-ab9bcea948e1" alt="Sign Up" height="65"/>
</a>

</div>

## Features

|                   |                                                                  |
|-------------------------------|:-----------------------------------------------------------------------------:|
| 🤖     **Self-Contained Agents** | Each agent is both a web app and an AI persona                            |
| 📝     **Prompt Object Model**   | Structured prompt composition using POM                                   |
| ⚙️     **SWAIG Integration**     | Easily define and handle AI tools/functions                               |
| 🔧     **Dynamic Configuration** | Configure agents per-request for multi-tenant apps and personalization    |
| 🗺️     **Custom Routing**        | Dynamic request handling for different paths and content                  |
| 📞     **SIP Integration**       | Route SIP calls to agents based on SIP usernames                          |
| 🔒     **Security Built-In**     | Session management, function-specific security tokens, and basic auth     |
| 💾     **State Management**      | Persistent conversation state with automatic tracking                     |
| 🏗️     **Prefab Archetypes**     | Ready-to-use agent types for common scenarios                            |
| 🏢     **Multi-Agent Support**   | Host multiple agents on a single server                                  |
| �      **Modular Skills System** | Add capabilities to agents with simple one-liner calls                   |
| 🔍     **Local Search System**   | Offline document search with vector similarity and keyword search        |

## Installation

### Basic Installation

```bash
pip install signalwire-agents
```

### Optional Search Functionality

The SDK includes optional local search capabilities that can be installed separately to avoid adding large dependencies to the base installation:

#### Search Installation Options

```bash
# Query existing .swsearch files only (smallest footprint)
pip install signalwire-agents[search-queryonly]

# Basic search (vector search + keyword search + building indexes)
pip install signalwire-agents[search]

# Full search with document processing (PDF, DOCX, etc.)
pip install signalwire-agents[search-full]

# Advanced NLP features (includes spaCy)
pip install signalwire-agents[search-nlp]

# All search features
pip install signalwire-agents[search-all]
```

#### What Each Option Includes

| Option | Size | Features |
|--------|------|----------|
| `search-queryonly` | ~400MB | Query existing .swsearch files only (no building/processing) |
| `search` | ~500MB | Vector embeddings, keyword search, basic text processing |
| `search-full` | ~600MB | + PDF, DOCX, Excel, PowerPoint, HTML, Markdown processing |
| `search-nlp` | ~600MB | + Advanced spaCy NLP features |
| `search-all` | ~700MB | All search features combined |

**When to use `search-queryonly`:**
- Production containers with pre-built `.swsearch` files
- Lambda/serverless deployments
- Agents that only need to query knowledge bases (not build them)
- Smaller deployment footprint requirements

#### Search Features

- **Local/Offline Search**: No external API dependencies
- **Hybrid Search**: Vector similarity + keyword search
- **Smart Document Processing**: Markdown, Python, PDF, DOCX, etc.
- **Multiple Languages**: English, Spanish, with extensible framework
- **CLI Tools**: Build search indexes from document directories
- **HTTP API**: Standalone or embedded search service

#### Usage Example

```python
# Only available with search extras installed
from signalwire_agents.search import IndexBuilder, SearchEngine

# Build search index
builder = IndexBuilder()
builder.build_index(
    source_dir="./docs",
    output_file="knowledge.swsearch",
    file_types=['md', 'txt', 'pdf']
)

# Search documents
engine = SearchEngine("knowledge.swsearch")
results = engine.search(
    query_vector=embeddings,
    enhanced_text="search query",
    count=5
)
```

<details>
<summary><h2>Documentation</h2></summary>

### Skills System

The SignalWire Agents SDK includes a powerful modular skills system that allows you to add complex capabilities to your agents with simple one-liner calls:

```python
from signalwire_agents import AgentBase

# Create an agent
agent = AgentBase("My Assistant", route="/assistant")

# Add skills with one-liners
agent.add_skill("web_search", {
    "api_key": "your-google-api-key",
    "search_engine_id": "your-search-engine-id"
})   # Web search capability
agent.add_skill("datetime")     # Current date/time info  
agent.add_skill("math")         # Mathematical calculations

# Configure skills with parameters
agent.add_skill("web_search", {
    "api_key": "your-google-api-key",
    "search_engine_id": "your-search-engine-id",
    "num_results": 1,  # Get 1 search results
    "no_results_message": "Sorry, I couldn't find anything about '{query}'. Try rephrasing your question."
})

# Advanced: Customize SWAIG function properties
agent.add_skill("math", {
    "swaig_fields": {
        "secure": False,  # Override security settings
        "fillers": {"en-US": ["Calculating..."]}  # Custom filler phrases
    }
})

# Multiple web search instances with different tool names
agent.add_skill("web_search", {
    "api_key": "your-google-api-key", 
    "search_engine_id": "general-search-engine-id",
    "tool_name": "search_general",  # Creates search_general tool
    "num_results": 1
})

agent.add_skill("web_search", {
    "api_key": "your-google-api-key",
    "search_engine_id": "news-search-engine-id", 
    "tool_name": "search_news",  # Creates search_news tool
    "num_results": 3,
    "delay": 0.5
})

# Multiple DataSphere instances with different tool names
agent.add_skill("datasphere", {
    "space_name": "my-space",
    "project_id": "my-project", 
    "token": "my-token",
    "document_id": "drinks-doc",
    "tool_name": "search_drinks",  # Creates search_drinks tool
    "count": 2
})

agent.add_skill("datasphere", {
    "space_name": "my-space", 
    "project_id": "my-project",
    "token": "my-token", 
    "document_id": "food-doc",
    "tool_name": "search_recipes",  # Creates search_recipes tool
    "tags": ["Food", "Recipes"]
})

agent.serve()
```

#### Available Built-in Skills

- **web_search**: Google Custom Search API integration with web scraping (supports multiple instances)
- **datetime**: Current date and time with timezone support
- **math**: Safe mathematical expression evaluation
- **datasphere**: SignalWire DataSphere knowledge search (supports multiple instances)
- **native_vector_search**: Offline document search with vector similarity and keyword search

#### Benefits

- **One-liner integration**: `agent.add_skill("skill_name")`
- **Configurable parameters**: `agent.add_skill("skill_name", {"param": "value"})`
- **Automatic discovery**: Skills are automatically found from the skills directory
- **Dependency validation**: Clear error messages for missing requirements
- **Modular architecture**: Skills are self-contained and reusable

For detailed documentation, see [Skills System README](docs/skills_system.md).

### DataMap Tools

The SDK provides a DataMap system for creating SWAIG tools that integrate directly with REST APIs without requiring custom webhook endpoints. DataMap tools execute on the SignalWire server, making them simpler to deploy than traditional webhook-based tools.

#### Basic DataMap Usage

```python
from signalwire_agents import AgentBase
from signalwire_agents.core.data_map import DataMap
from signalwire_agents.core.function_result import SwaigFunctionResult

class APIAgent(AgentBase):
    def __init__(self):
        super().__init__(name="api-agent", route="/api")
        
        # Create a simple weather API tool
        weather_tool = (DataMap('get_weather')
            .description('Get current weather information')
            .parameter('location', 'string', 'City name', required=True)
            .webhook('GET', 'https://api.weather.com/v1/current?key=YOUR_API_KEY&q=${location}')
            .output(SwaigFunctionResult('Weather in ${location}: ${response.current.condition.text}, ${response.current.temp_f}°F'))
        )
        
        # Register the tool with the agent
        self.register_swaig_function(weather_tool.to_swaig_function())

agent = APIAgent()
agent.serve()
```

#### Advanced DataMap Examples

```python
# POST API with authentication
search_tool = (DataMap('search_knowledge')
    .description('Search company knowledge base')
    .parameter('query', 'string', 'Search query', required=True)
    .webhook('POST', 'https://api.company.com/search', 
             headers={'Authorization': 'Bearer YOUR_TOKEN'})
    .body({'query': '${query}', 'limit': 3})
    .output(SwaigFunctionResult('Found: ${response.title} - ${response.summary}'))
)

# Expression-based tools (no API calls)
control_tool = (DataMap('file_control')
    .description('Control file playback')
    .parameter('command', 'string', 'Playback command')
    .parameter('filename', 'string', 'File to control', required=False)
    .expression(r'start.*', SwaigFunctionResult().add_action('start_playback', {'file': '${args.filename}'}))
    .expression(r'stop.*', SwaigFunctionResult().add_action('stop_playback', True))
)

# Process API response arrays
docs_tool = (DataMap('get_latest_docs')
    .description('Get latest documentation')
    .webhook('GET', 'https://api.docs.com/latest')
    .foreach('${response.documents}')
    .output(SwaigFunctionResult('Document: ${foreach.title} (${foreach.updated_date})'))
)
```

#### Helper Functions

For simpler use cases, use the convenience functions:

```python
from signalwire_agents.core.data_map import create_simple_api_tool, create_expression_tool

# Simple API tool
weather = create_simple_api_tool(
    name='get_weather',
    url='https://api.weather.com/v1/current?key=API_KEY&q=${location}',
    response_template='Weather in ${location}: ${response.current.condition.text}',
    parameters={'location': {'type': 'string', 'description': 'City name', 'required': True}}
)

# Expression-based tool
file_control = create_expression_tool(
    name='file_control',
    patterns={
        r'start.*': SwaigFunctionResult().add_action('start_playback', {'file': '${args.filename}'}),
        r'stop.*': SwaigFunctionResult().add_action('stop_playback', True)
    },
    parameters={'command': {'type': 'string', 'description': 'Playback command'}}
)

# Register with agent
self.register_swaig_function(weather.to_swaig_function())
self.register_swaig_function(file_control.to_swaig_function())
```

#### Variable Expansion

DataMap tools support powerful variable expansion using `${variable}` syntax:

- **Function arguments**: `${args.parameter_name}`
- **API responses**: `${response.field.nested_field}`
- **Array processing**: `${foreach.item_field}` (when using foreach)
- **Global data**: `${global_data.key}`
- **Metadata**: `${meta_data.call_id}`

#### Benefits of DataMap Tools

- **No webhook infrastructure**: Tools run on SignalWire servers
- **Simplified deployment**: No need to expose endpoints
- **Built-in authentication**: Support for API keys, Bearer tokens, Basic auth
- **Response processing**: Built-in JSON path traversal and array iteration
- **Error handling**: Automatic error detection with `error_keys`
- **Pattern matching**: Expression-based responses without API calls

For detailed documentation, see [DataMap Guide](docs/datamap_guide.md).

### Contexts and Steps

The SignalWire Agents SDK provides a powerful enhancement to traditional prompts through the **Contexts and Steps** system. This feature allows you to add structured, workflow-driven AI interactions on top of your base prompt, with explicit navigation control and step-by-step guidance.

#### Why Use Contexts and Steps?

- **Structured Workflows**: Define clear, step-by-step processes for complex interactions
- **Navigation Control**: Explicitly control which steps or contexts users can access
- **Completion Criteria**: Set specific criteria for step completion and progression  
- **Function Restrictions**: Limit which AI tools are available in each step
- **Workflow Isolation**: Create separate contexts for different conversation flows
- **Enhanced Base Prompts**: Adds structured workflows on top of your existing prompt foundation

#### Basic Usage

```python
from signalwire_agents import AgentBase

class WorkflowAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Workflow Assistant", route="/workflow")
        
        # Set base prompt (required even when using contexts)
        self.prompt_add_section("Role", "You are a helpful workflow assistant.")
        self.prompt_add_section("Instructions", "Guide users through structured processes step by step.")
        
        # Define contexts and steps (adds structured workflow to base prompt)
        contexts = self.define_contexts()
        
        # Create a single context named "default" (required for single context)
        context = contexts.add_context("default")
        
        # Add step-by-step workflow
        context.add_step("greeting") \
            .set_text("Welcome! I'm here to help you complete your application. Let's start with your personal information.") \
            .set_step_criteria("User has provided their name and confirmed they want to continue") \
            .set_valid_steps(["personal_info"])  # Can only go to personal_info step
        
        context.add_step("personal_info") \
            .add_section("Instructions", "Collect the user's personal information") \
            .add_bullets(["Ask for full name", "Ask for email address", "Ask for phone number"]) \
            .set_step_criteria("All personal information has been collected and confirmed") \
            .set_valid_steps(["review", "personal_info"])  # Can stay or move to review
        
        context.add_step("review") \
            .set_text("Let me review the information you've provided. Please confirm if everything is correct.") \
            .set_step_criteria("User has confirmed or requested changes") \
            .set_valid_steps(["personal_info", "complete"])  # Can go back or complete
        
        context.add_step("complete") \
            .set_text("Thank you! Your application has been submitted successfully.") \
            .set_step_criteria("Application processing is complete")
            # No valid_steps = end of workflow

agent = WorkflowAgent()
agent.serve()
```

#### Advanced Features

```python
class MultiContextAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Multi-Context Agent", route="/multi-context")
        
        # Set base prompt (required)
        self.prompt_add_section("Role", "You are a versatile AI assistant.")
        self.prompt_add_section("Capabilities", "You can help with calculations and provide time information.")
        
        # Add skills
        self.add_skill("datetime")
        self.add_skill("math")
        
        # Define contexts for different service modes
        contexts = self.define_contexts()
        
        # Main conversation context
        main_context = contexts.add_context("main")
        main_context.add_step("welcome") \
            .set_text("Welcome! I can help with calculations or provide date/time info. What would you like to do?") \
            .set_step_criteria("User has chosen a service type") \
            .set_valid_contexts(["calculator", "datetime_info"])  # Can switch contexts
        
        # Calculator context with function restrictions
        calc_context = contexts.add_context("calculator")
        calc_context.add_step("math_mode") \
            .add_section("Role", "You are a mathematical assistant") \
            .add_section("Instructions", "Help users with calculations") \
            .set_functions(["math"])  # Only math function available \
            .set_step_criteria("Calculation is complete") \
            .set_valid_contexts(["main"])  # Can return to main
        
        # DateTime context
        datetime_context = contexts.add_context("datetime_info")
        datetime_context.add_step("time_mode") \
            .set_text("I can provide current date and time information. What would you like to know?") \
            .set_functions(["datetime"])  # Only datetime function available \
            .set_step_criteria("Date/time information has been provided") \
            .set_valid_contexts(["main"])  # Can return to main
```

#### Context and Step Methods

##### Context Methods
- `add_step(name)`: Create a new step in this context
- `set_valid_contexts(contexts)`: Control which contexts can be accessed from this context

##### Step Methods
- `set_text(text)`: Set direct text prompt for the step
- `add_section(title, body)`: Add POM-style section (alternative to set_text)
- `add_bullets(bullets)`: Add bullet points to the current or last section
- `set_step_criteria(criteria)`: Define completion criteria for this step
- `set_functions(functions)`: Restrict available functions ("none" or array of function names)
- `set_valid_steps(steps)`: Control navigation to other steps in same context
- `set_valid_contexts(contexts)`: Control navigation to other contexts

#### Navigation Rules

- **Valid Steps**: If omitted, only "next" step is implied. If specified, only those steps are allowed.
- **Valid Contexts**: If omitted, user is trapped in current context. If specified, can navigate to those contexts.
- **Single Context**: Must be named "default" for single-context workflows.
- **Function Restrictions**: Use `set_functions(["function_name"])` or `set_functions("none")` to control AI tool access.

#### Complete Example: Customer Support Workflow

```python
class SupportAgent(AgentBase):
    def __init__(self):
        super().__init__(name="Customer Support", route="/support")
        
        # Set base prompt (required)
        self.prompt_add_section("Role", "You are a professional customer support representative.")
        self.prompt_add_section("Goal", "Provide excellent customer service using structured workflows.")
        
        # Add skills for enhanced capabilities
        self.add_skill("datetime")
        self.add_skill("web_search", {"api_key": "your-key", "search_engine_id": "your-id"})
        
        # Define support workflow contexts
        contexts = self.define_contexts()
        
        # Triage context
        triage = contexts.add_context("triage")
        triage.add_step("initial_greeting") \
            .add_section("Current Task", "Understand the customer's issue and route them appropriately") \
            .add_bullets("Questions to Ask", ["What problem are you experiencing?", "How urgent is this issue?", "Have you tried any troubleshooting steps?"]) \
            .set_step_criteria("Issue type has been identified") \
            .set_valid_contexts(["technical_support", "billing_support", "general_inquiry"])
        
        # Technical support context
        tech = contexts.add_context("technical_support")
        tech.add_step("technical_diagnosis") \
            .add_section("Current Task", "Help diagnose and resolve technical issues") \
            .add_section("Available Tools", "Use web search to find solutions and datetime to check service windows") \
            .set_functions(["web_search", "datetime"])  # Can search for solutions and check times \
            .set_step_criteria("Technical issue is resolved or escalated") \
            .set_valid_contexts(["triage"])  # Can return to triage
        
        # Billing support context  
        billing = contexts.add_context("billing_support")
        billing.add_step("billing_assistance") \
            .set_text("I'll help you with your billing inquiry. Please provide your account details.") \
            .set_functions("none")  # No external tools for sensitive billing info \
            .set_step_criteria("Billing issue is addressed") \
            .set_valid_contexts(["triage"])
        
        # General inquiry context
        general = contexts.add_context("general_inquiry")
        general.add_step("general_help") \
            .set_text("I'm here to help with general questions. What can I assist you with?") \
            .set_functions(["web_search", "datetime"])  # Full access to search and time \
            .set_step_criteria("Inquiry has been answered") \
            .set_valid_contexts(["triage"])

agent = SupportAgent()
agent.serve()
```

#### Benefits

- **Clear Structure**: Explicit workflow definition makes agent behavior predictable
- **Enhanced Control**: Fine-grained control over function access and navigation
- **Improved UX**: Users understand where they are in the process and what's expected
- **Debugging**: Easy to trace and debug workflow issues
- **Scalability**: Complex multi-step processes are easier to maintain

For detailed documentation and advanced examples, see [Contexts and Steps Guide](docs/contexts_guide.md).

### Quick Start

```python
from signalwire_agents import AgentBase
from signalwire_agents.core.function_result import SwaigFunctionResult

class SimpleAgent(AgentBase):
    def __init__(self):
        super().__init__(name="simple", route="/simple")
        
        # Configure the agent's personality
        self.prompt_add_section("Personality", body="You are a helpful assistant.")
        self.prompt_add_section("Goal", body="Help users with basic questions.")
        self.prompt_add_section("Instructions", bullets=["Be concise and clear."])
        
        # Note: Use prompt_add_section() for all prompt configuration
    
    @AgentBase.tool(
        name="get_time", 
        description="Get the current time",
        parameters={}
    )
    def get_time(self, args, raw_data):
        from datetime import datetime
        now = datetime.now().strftime("%H:%M:%S")
        return SwaigFunctionResult(f"The current time is {now}")

# Run the agent
if __name__ == "__main__":
    agent = SimpleAgent()
    agent.serve(host="0.0.0.0", port=8000)
```

### Customizing LLM Parameters

The SDK allows you to customize LLM parameters for both the main prompt and post-prompt, giving you fine control over the AI's behavior:

```python
from signalwire_agents import AgentBase

class PreciseAgent(AgentBase):
    def __init__(self):
        super().__init__(name="precise", route="/precise")
        
        # Configure the agent's personality
        self.prompt_add_section("Role", "You are a precise technical assistant.")
        self.prompt_add_section("Instructions", "Provide accurate, detailed information.")
        
        # Set custom LLM parameters for the main prompt
        # These parameters are passed to the server which validates them based on the model
        self.set_prompt_llm_params(
            temperature=0.3,        # Low temperature for more consistent responses
            top_p=0.9,             # Slightly reduced for focused responses
            barge_confidence=0.7,  # Moderate interruption threshold
            presence_penalty=0.1,  # Slight penalty for repetition
            frequency_penalty=0.2  # Encourage varied vocabulary
        )
        
        # Set post-prompt for summaries
        self.set_post_prompt("Provide a concise summary of the key points discussed.")
        
        # Different parameters for post-prompt (summaries should be even more focused)
        self.set_post_prompt_llm_params(
            temperature=0.2,       # Very low for consistent summaries
            top_p=0.85            # More focused token selection
        )

agent = PreciseAgent()
agent.serve()
```

#### Common LLM Parameters

The SDK accepts any parameters which are passed to the server for validation based on the model. Common parameters include:

- **temperature**: Controls randomness. Lower = more focused, higher = more creative
- **top_p**: Nucleus sampling. Lower = more focused on likely tokens
- **barge_confidence**: ASR confidence to interrupt. Higher = harder to interrupt (main prompt only)
- **presence_penalty**: Topic diversity. Positive = new topics
- **frequency_penalty**: Repetition control. Positive = varied vocabulary

Note: No defaults are sent unless explicitly set. The server handles validation and applies appropriate defaults based on the model.

For more details on LLM parameter tuning, see [LLM Parameters Guide](docs/llm_parameters.md).

### Using Prefab Agents

```python
from signalwire_agents.prefabs import InfoGathererAgent

agent = InfoGathererAgent(
    fields=[
        {"name": "full_name", "prompt": "What is your full name?"},
        {"name": "reason", "prompt": "How can I help you today?"}
    ],
    confirmation_template="Thanks {full_name}, I'll help you with {reason}.",
    name="info-gatherer",
    route="/info-gatherer"
)

agent.serve(host="0.0.0.0", port=8000)
```

Available prefabs include:
- `InfoGathererAgent`: Collects structured information from users
- `FAQBotAgent`: Answers questions based on a knowledge base
- `ConciergeAgent`: Routes users to specialized agents
- `SurveyAgent`: Conducts structured surveys with questions and rating scales
- `ReceptionistAgent`: Greets callers and transfers them to appropriate departments

### Dynamic Agent Configuration

Configure agents dynamically based on request parameters for multi-tenant applications, A/B testing, and personalization.

#### Static vs Dynamic Configuration

- **Static**: Agent configured once at startup (traditional approach)
- **Dynamic**: Agent configured fresh for each request based on parameters

#### Basic Example

```python
from signalwire_agents import AgentBase

class DynamicAgent(AgentBase):
    def __init__(self):
        super().__init__(name="dynamic-agent", route="/dynamic")
        
        # Set up dynamic configuration callback
        self.set_dynamic_config_callback(self.configure_per_request)
    
    def configure_per_request(self, query_params, body_params, headers, agent):
        """Configure agent based on request parameters"""
        
        # Extract parameters from request
        tier = query_params.get('tier', 'standard')
        language = query_params.get('language', 'en')
        customer_id = query_params.get('customer_id')
        
        # Configure voice and language
        if language == 'es':
            agent.add_language("Spanish", "es-ES", "rime.spore:mistv2")
        else:
            agent.add_language("English", "en-US", "rime.spore:mistv2")
        
        # Configure based on service tier
        if tier == 'premium':
            agent.set_params({"end_of_speech_timeout": 300})  # Faster response
            agent.prompt_add_section("Service Level", "You provide premium support.")
        else:
            agent.set_params({"end_of_speech_timeout": 500})  # Standard response
            agent.prompt_add_section("Service Level", "You provide standard support.")
        
        # Personalize with customer data
        global_data = {"tier": tier, "language": language}
        if customer_id:
            global_data["customer_id"] = customer_id
        agent.set_global_data(global_data)

# Usage examples:
# curl "http://localhost:3000/dynamic?tier=premium&language=es&customer_id=123"
# curl "http://localhost:3000/dynamic?tier=standard&language=en"
```

#### Use Cases

- **Multi-tenant SaaS**: Different configurations per customer/organization
- **A/B Testing**: Test different agent behaviors with different user groups
- **Personalization**: Customize voice, prompts, and behavior per user
- **Localization**: Language and cultural adaptation based on user location
- **Dynamic Pricing**: Adjust features and capabilities based on subscription tiers

#### Preserving Dynamic State in SWAIG Callbacks

When using dynamic configuration to add skills or tools based on request parameters, there's a challenge: SWAIG webhook callbacks are separate HTTP requests that won't have the original query parameters. The SDK provides `add_swaig_query_params()` to solve this:

```python
class DynamicAgent(AgentBase):
    def __init__(self):
        super().__init__(name="dynamic-agent", route="/agent")
        self.set_dynamic_config_callback(self.configure_per_request)
    
    def configure_per_request(self, query_params, body_params, headers, agent):
        tier = query_params.get('tier', 'basic')
        region = query_params.get('region', 'us-east')
        
        if tier == 'premium':
            # Add premium skills dynamically
            agent.add_skill('advanced_search', {
                'api_key': 'your-api-key',
                'num_results': 5
            })
            
            # IMPORTANT: Preserve parameters for SWAIG callbacks
            agent.add_swaig_query_params({
                'tier': tier,
                'region': region
            })
            
            # Now when SignalWire calls the SWAIG webhook, these params
            # will be included, triggering the same dynamic configuration

# Initial request: GET /agent?tier=premium&region=eu-west
# SWAIG callback: POST /swaig/?tier=premium&region=eu-west
# Result: Premium skills are available in both requests!
```

**Key Points:**

- **Problem**: Dynamically added skills/tools won't exist during SWAIG callbacks without the original request parameters
- **Solution**: Use `add_swaig_query_params()` to include critical parameters in all SWAIG webhook URLs
- **Clear State**: Use `clear_swaig_query_params()` if needed to reset parameters between requests
- **Token Safety**: The SDK automatically renames security tokens from `token` to `__token` to avoid parameter collisions

This ensures that any dynamic configuration based on request parameters is consistently applied across the initial SWML request and all subsequent SWAIG function callbacks.

For detailed documentation and advanced examples, see the [Agent Guide](docs/agent_guide.md#dynamic-agent-configuration).

### Configuration

#### Environment Variables

The SDK supports the following environment variables:

- `SWML_BASIC_AUTH_USER`: Username for basic auth (default: auto-generated)
- `SWML_BASIC_AUTH_PASSWORD`: Password for basic auth (default: auto-generated)
- `SWML_PROXY_URL_BASE`: Base URL to use when behind a reverse proxy, used for constructing webhook URLs
- `SWML_SSL_ENABLED`: Enable HTTPS/SSL support (values: "true", "1", "yes")
- `SWML_SSL_CERT_PATH`: Path to SSL certificate file
- `SWML_SSL_KEY_PATH`: Path to SSL private key file
- `SWML_DOMAIN`: Domain name for SSL certificate and external URLs
- `SWML_SCHEMA_PATH`: Optional path to override the location of the schema.json file

When the auth environment variables are set, they will be used for all agents instead of generating random credentials. The proxy URL base is useful when your service is behind a reverse proxy or when you need external services to access your webhooks.

To enable HTTPS directly (without a reverse proxy), set `SWML_SSL_ENABLED` to "true", provide valid paths to your certificate and key files, and specify your domain name.

### Testing

The SDK includes powerful CLI tools for development and testing:

- **`swaig-test`**: Comprehensive local testing and serverless environment simulation
- **`sw-search`**: Build local search indexes from document directories and search within them

#### Local Testing with swaig-test

Test your agents locally without deployment:

```bash
# Install the SDK
pip install -e .

# Discover agents in a file
swaig-test examples/my_agent.py

# List available functions
swaig-test examples/my_agent.py --list-tools

# Test SWAIG functions with CLI syntax
swaig-test examples/my_agent.py --exec get_weather --location "New York"

# Multi-agent support
swaig-test examples/multi_agent.py --route /agent-path --list-tools
swaig-test examples/multi_agent.py --agent-class AgentName --exec function_name

# Generate and inspect SWML documents
swaig-test examples/my_agent.py --dump-swml
swaig-test examples/my_agent.py --dump-swml --raw | jq '.'
```

#### Serverless Environment Simulation

Test your agents in simulated serverless environments without deployment:

```bash
# Test in AWS Lambda environment
swaig-test examples/my_agent.py --simulate-serverless lambda --dump-swml

# Test Lambda function execution with proper response format
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --exec get_weather --location "Miami" --full-request

# Test with custom Lambda configuration
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --aws-function-name my-production-function \
  --aws-region us-west-2 \
  --exec my_function --param value

# Test CGI environment
swaig-test examples/my_agent.py --simulate-serverless cgi \
  --cgi-host my-server.com --cgi-https --dump-swml

# Test Google Cloud Functions
swaig-test examples/my_agent.py --simulate-serverless cloud_function \
  --gcp-function-url https://my-function.cloudfunctions.net \
  --exec my_function

# Test Azure Functions
swaig-test examples/my_agent.py --simulate-serverless azure_function \
  --azure-function-url https://my-function.azurewebsites.net \
  --exec my_function
```

#### Environment Management

Use environment files for consistent testing across platforms:

```bash
# Create environment file
cat > production.env << EOF
AWS_LAMBDA_FUNCTION_NAME=prod-my-agent
AWS_REGION=us-east-1
API_KEY=prod_api_key_123
DEBUG=false
EOF

# Test with environment file
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --env-file production.env --exec my_function

# Override specific variables
swaig-test examples/my_agent.py --simulate-serverless lambda \
  --env-file production.env --env DEBUG=true --dump-swml
```

#### Cross-Platform Testing

Test the same agent across multiple serverless platforms:

```bash
# Test across all platforms
for platform in lambda cgi cloud_function azure_function; do
  echo "Testing $platform..."
  swaig-test examples/my_agent.py --simulate-serverless $platform \
    --exec my_function --param value
done

# Compare webhook URLs across platforms
swaig-test examples/my_agent.py --simulate-serverless lambda --dump-swml | grep web_hook_url
swaig-test examples/my_agent.py --simulate-serverless cgi --cgi-host example.com --dump-swml | grep web_hook_url
```

#### Key Benefits

- **No Deployment Required**: Test serverless behavior locally
- **Environment Simulation**: Complete platform-specific environment variable setup
- **URL Generation**: Verify webhook URLs are generated correctly for each platform
- **Function Execution**: Test with platform-specific request/response formats
- **Environment Files**: Reusable configurations for different stages
- **Multi-Platform**: Test Lambda, CGI, Cloud Functions, and Azure Functions

For detailed testing documentation, see the [CLI Testing Guide](docs/cli_testing_guide.md).

### Documentation

The package includes comprehensive documentation in the `docs/` directory:

- [Agent Guide](docs/agent_guide.md) - Detailed guide to creating and customizing agents, including dynamic configuration
- [Architecture](docs/architecture.md) - Overview of the SDK architecture and core concepts
- [SWML Service Guide](docs/swml_service_guide.md) - Guide to the underlying SWML service
- [Local Search System](docs/search-system.md) - Complete guide to the local search system with vector similarity and keyword search
- [Skills System](docs/skills_system.md) - Detailed documentation on the modular skills system
- [CLI Tools](docs/cli.md) - Command-line interface tools for development and testing

These documents provide in-depth explanations of the features, APIs, and usage patterns.

</details

### ***[Read the official docs.](https://developer.signalwire.com/sdks/agents-sdk)***

---

## License

MIT
