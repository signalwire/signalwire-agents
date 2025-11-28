# Building Advanced AI Voice Agents with the SignalWire AI Agents SDK

[TOC]

## 1. Introduction

### What is the SignalWire AI Agents SDK

The SignalWire AI Agents SDK is a Python framework that provides developers with powerful tools to create, deploy, and manage conversational AI agents with minimal effort. Unlike generic AI tools that require significant customization for voice applications, this SDK is purpose-built for creating voice-centric AI agents that can understand spoken language, respond naturally, and execute complex workflows.

At its core, the SDK enables you to create self-contained AI agents as microservices, each with its own personality, capabilities, and endpoints. These agents can handle telephone calls, respond to user queries, perform actions through custom functions, and maintain context throughout conversations. The SDK abstracts away the complexities of prompt engineering, web service configuration, and conversation flow management, allowing developers to focus on designing the agent's behavior and business logic.

**Modern SDK Features**: The latest version of the SDK introduces three revolutionary capabilities that dramatically simplify agent development:

1. **Skills System**: Add complex capabilities to your agents with simple one-liner calls like `agent.add_skill("web_search")` or `agent.add_skill("datetime")`. No more manual function implementation for common tasks.

2. **DataMap Tools**: Create API integrations that run on SignalWire's servers without building webhook endpoints. Define REST API calls declaratively and let the platform handle execution.

3. **Local Search System**: Enable agents to search through document collections offline using advanced vector similarity and hybrid search techniques. Build searchable indexes from your documentation, knowledge bases, or any text collection:

```bash
sw-search docs/ --output knowledge.swsearch
```

These features work together to provide a complete toolkit for building sophisticated AI agents with minimal code, whether you need simple conversational agents or complex multi-capability systems.

### Why Voice AI Matters in Customer Engagement

Voice remains the most natural and efficient form of human communication. While text-based chatbots have become commonplace, they often fall short when handling complex interactions that require nuance, empathy, and real-time problem-solving. Voice AI agents bridge this gap by providing:

1. **Enhanced Accessibility**: Voice interfaces remove barriers for users who may struggle with typing or navigating visual interfaces.

2. **Increased Efficiency**: Speaking is typically 3-4 times faster than typing, allowing for quicker information exchange and problem resolution.

3. **Emotional Connection**: Voice conveys tone, emphasis, and emotion in ways that text simply cannot, creating more engaging and human-like interactions.

4. **Hands-Free Operation**: Voice agents enable interaction while users are engaged in other activities, expanding the contexts in which customers can engage with your services.

5. **Reduced Cognitive Load**: Natural conversation requires less mental effort than composing and reading text, making complex interactions feel simpler and more intuitive.

In customer engagement specifically, voice AI can transform key touchpoints throughout the customer journey. From initial inquiries and lead qualification to technical support and feedback collection, voice AI agents can provide personalized, efficient service at scale while reducing operational costs.

### Key Features and Benefits

The SignalWire AI Agents SDK offers several distinctive features that set it apart from general-purpose AI frameworks:

**Self-Contained Agents**: Each agent functions as both a web application and an AI persona, complete with its own HTTP endpoints, personality, and specialized capabilities. This modular approach allows for clear separation of concerns and simplified deployment.

**Skills System**: The revolutionary skills system allows you to add complex capabilities to agents with simple one-liner calls. Instead of implementing web search, date/time functions, mathematical calculations, or document search from scratch, you can add them instantly with configurable parameters:

```python
class MyAgent(AgentBase):
    def __init__(self):
        super().__init__(name="My Agent")
        self.add_skill("web_search", {
            "api_key": "your-google-api-key",
            "search_engine_id": "your-search-engine-id",
            "num_results": 3
        })
        self.add_skill("datetime")
        self.add_skill("math")
        self.add_skill("native_vector_search", {"index_file": "knowledge.swsearch"})
```

**DataMap Tools**: Create API integrations without webhook infrastructure. DataMap tools execute on SignalWire's servers, allowing you to integrate with REST APIs through declarative configuration rather than custom endpoints:

```python
from signalwire_agents.core.data_map import DataMap
from signalwire_agents.core.function_result import SwaigFunctionResult

weather_tool = (DataMap('get_weather')
    .parameter('location', 'string', 'City name', required=True)
    .webhook('GET', 'https://api.weather.com/v1/current?key=API_KEY&q=${location}')
    .output(SwaigFunctionResult('Weather: ${response.current.temp_f}°F'))
)
```

**Local Search System**: Enable agents to search through document collections offline using advanced vector similarity and hybrid search techniques. Build searchable indexes from your documentation, knowledge bases, or any text collection:

```bash
sw-search docs/ --output knowledge.swsearch
```

**Prompt Object Model (POM)**: The SDK introduces a structured approach to prompt construction through the Prompt Object Model. POM enables clean organization of the agent's personality, goals, and instructions into discrete sections, making prompts more maintainable and effective.

**SWAIG Integration**: SignalWire AI Gateway (SWAIG) functions allow agents to perform actions beyond conversation, such as retrieving data, executing commands, or integrating with external systems. These functions are defined with clear parameter schemas and can be invoked by the AI when needed.

**Multilingual Support**: Agents can be configured to understand and respond in multiple languages with appropriate voice models and speech patterns for each language.

**Stateless-First Design**: The SDK uses a stateless-first architecture for maximum scalability. When state persistence is needed, developers can use SignalWire's built-in global_data and meta_data features through SWAIG.

**Security Controls**: Basic authentication, function-specific security tokens, and session management are built into the framework, securing your agents against unauthorized access.

**Prefab Agents**: Ready-to-use agent implementations for common scenarios like information gathering, knowledge base questions, and customer routing accelerate development.

**Multi-Agent Orchestration**: Multiple specialized agents can be hosted on a single server, with routing mechanisms to direct users to the appropriate agent based on their needs.

The benefits of using the SignalWire AI Agents SDK include:

- **Dramatically Reduced Development Time**: Skills system eliminates 80% of common function implementations
- **Zero Infrastructure for API Tools**: DataMap tools run on SignalWire's servers without webhook setup
- **Offline Knowledge Search**: Local search system provides sophisticated document search without external dependencies
- **Improved Maintainability**: Structured organization of prompts and clear separation of business logic from AI infrastructure
- **Enhanced Security**: Built-in authentication and authorization mechanisms protect sensitive functions
- **Flexible Deployment**: Run as standalone services or integrate multiple agents into a cohesive system
- **Accelerated Innovation**: Focus on unique business logic rather than infrastructure setup

In the following sections, we'll explore the architecture of the SDK, walk through creating your first agent using modern features, and demonstrate advanced customization techniques to create powerful, production-ready AI voice agents. 

## 2. Understanding the Architecture

The SignalWire AI Agents SDK embodies a sophisticated architectural design that prioritizes both developer productivity and production scalability. Rather than forcing developers to choose between simplicity and power, the SDK provides a layered architecture where common tasks are effortless while advanced customization remains fully accessible.

### Architectural Philosophy

At its core, the SDK embraces three fundamental principles that distinguish it from traditional conversational AI frameworks:

**Stateless-First Design**: Unlike frameworks that burden developers with state management complexity, the SDK operates stateless by default. This design enables true horizontal scaling where agents can handle thousands of concurrent conversations without state conflicts, memory leaks, or complex synchronization logic. When applications do require state persistence, the platform provides elegant mechanisms through SWAIG's global_data and meta_data features.

**Capability Abstraction**: The Skills System represents a paradigm shift from implementation-heavy frameworks. Instead of requiring developers to build web search, mathematical calculations, or document retrieval from scratch, these capabilities become one-line additions. This abstraction doesn't sacrifice power—each skill accepts comprehensive configuration parameters and can be instantiated multiple times with different settings.

**Platform-Managed Infrastructure**: DataMap tools eliminate the traditional requirement for webhook infrastructure. Instead of managing HTTP servers, handling retries, and implementing security for API integrations, developers describe their requirements declaratively and the platform handles execution. This shifts the operational burden from the developer to SignalWire's managed infrastructure.

### Core Component Deep Dive

#### AgentBase: Foundation Layer

The `AgentBase` class serves as the foundational abstraction that defines the core interface every agent implementation must provide. This isn't simply a base class—it's an architectural contract that ensures consistency across all agent types while enabling specialized behavior.

```python
class AgentBase:
    def __init__(
        self,
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
        state_manager: Optional[StateManager] = None,
        default_webhook_url: Optional[str] = None,
        agent_id: Optional[str] = None,
        native_functions: Optional[List[str]] = None,
        schema_path: Optional[str] = None,
        suppress_logs: bool = False,
        enable_post_prompt_override: bool = False,
        check_for_input_override: bool = False
    ):
        # Core initialization with all parameters
        
    def set_prompt_text(self, prompt: str):
        # Define the agent's personality and capabilities
        
    def add_skill(self, skill_name: str, params: Optional[Dict[str, Any]] = None):
        # Register skills with optional configuration
        
    def register_swaig_function(self, function_dict: Dict[str, Any]):
        # Register SWAIG functions (including DataMap tools)
        
    @classmethod
    def tool(cls, name=None, **kwargs):
        # SWAIG function decorator for custom business logic
        def decorator(func):
            # Function implementation
            pass
```

The base class establishes the common patterns for agent lifecycle management, skill integration, and conversation handling while allowing concrete implementations to focus on domain-specific functionality.

#### AgentBase: Full-Featured Implementation

The `AgentBase` class provides the complete implementation of all SDK capabilities. Understanding its internal architecture helps developers make informed decisions about customization and optimization.

**HTTP Server Integration**: Built on FastAPI for production-ready performance and ecosystem compatibility. The agent automatically configures endpoints, request validation, and response formatting:

```python
# Automatic endpoint creation
router.get("/") and router.post("/")                    # Main conversation endpoint
router.get("/debug") and router.post("/debug")          # Debug endpoint
router.get("/swaig") and router.post("/swaig")          # SWAIG function call endpoint
router.get("/post_prompt") and router.post("/post_prompt")  # Post-prompt endpoint
router.get("/check_for_input") and router.post("/check_for_input")  # Input check endpoint
app.get("/health") and app.post("/health")              # Health check endpoint
app.get("/ready") and app.post("/ready")                # Readiness check endpoint
```

**Skills Registry**: AgentBase maintains a dynamic registry of available skills, enabling runtime configuration and multiple skill instances:

```python
# Skills can be added with specific configurations
agent.add_skill("web_search", {
    "api_key": "your-google-api-key",
    "search_engine_id": "your-search-engine-id",
    "num_results": 10
})

# Same skill, different configuration for news
agent.add_skill("web_search", {
    "api_key": "your-google-api-key",
    "search_engine_id": "news-engine-id", 
    "num_results": 3,
    "tool_name": "news_search"
})
```

**DataMap Execution Engine**: DataMap tools are processed through a sophisticated execution engine that handles variable expansion, error recovery, and response transformation:

```python
# Variable expansion patterns supported:
# ${field}             - Direct parameter access from function arguments
```

#### Skills System Architecture

The Skills System represents one of the SDK's most innovative architectural decisions. Rather than providing a library of functions that developers must integrate manually, skills are automatically discovered and made available to the AI through a dynamic registration system.

**Skill Discovery**: When an agent starts, it scans available skills and automatically generates OpenAI-compatible function schemas. This means the AI understands what capabilities are available and how to use them without additional configuration.

**Configuration Inheritance**: Skills support hierarchical configuration where global settings can be overridden by instance-specific parameters:

```python
# Global web search configuration
default_web_config = {
    "max_results": 5,
    "safe_search": "strict",
    "timeout": 10
}

# Specific instance overrides
agent.add_skill("web_search", {
    **default_web_config,
    "search_engine_id": "docs-engine-id",
    "max_results": 3  # Override global setting
})
```

**Execution Context**: Skills execute within a carefully managed context that provides access to agent state, request information, and platform services while maintaining security boundaries.

#### DataMap Tools: Declarative API Integration

DataMap tools transform API integration from an imperative programming exercise into a declarative configuration task. The architecture supports complex integration patterns while maintaining simplicity for common use cases.

**Method Chaining Interface**: DataMap uses a fluent interface that makes configuration readable and discoverable:

```python
user_lookup = (DataMap('lookup_user')
    .parameter('user_id', 'string', 'User identifier', required=True)
    .parameter('include_history', 'boolean', 'Include user history')
    .webhook('GET', 'https://api.example.com/users/${args.user_id}?history=${args.include_history}',
             headers={'Authorization': 'Bearer ${global_data.API_TOKEN}'})
    .output(SwaigFunctionResult("""
        User Information:
        Name: ${response.full_name}
        Email: ${response.email_address}
        Last Login: ${response.last_activity.login_time}
    """))
    .error_keys(['error', 'message'])
    .fallback_output(SwaigFunctionResult("Unable to retrieve user information."))
)
```

**Variable Expansion Engine**: The platform's variable expansion engine supports sophisticated data transformations and conditional logic:

```python
# Expression-based conditional responses
.expression('${response.status}', r'active', 
    SwaigFunctionResult('✅ Account Status: Active. Subscription expires: ${response.subscription.expires}'))
.expression('${response.status}', r'suspended',
    SwaigFunctionResult('⚠️ Account Status: Suspended. Reason: ${response.suspension_reason}. Contact support to reactivate.'))
.output(SwaigFunctionResult('⚠️ Account Status: ${response.status}'))
```

**Error Handling and Resilience**: DataMap tools include comprehensive error handling that maintains conversational flow while providing meaningful feedback to users.

#### Local Search System Architecture

The Local Search System implements a hybrid search architecture that combines vector similarity search with traditional keyword matching for optimal relevance.

**Index Structure**: Search indexes use a multi-layer approach:

```text
.swsearch file format:
├── Metadata Layer
│   ├── Document count and statistics
│   ├── Index creation parameters
│   └── Version compatibility information
├── Vector Layer
│   ├── Sentence-level embeddings
│   ├── Paragraph-level embeddings
│   └── Document-level embeddings
├── Keyword Layer
│   ├── Inverted index for exact matches
│   ├── n-gram indexes for partial matches
│   └── Stemmed word indexes for linguistic variants
└── Document Store
    ├── Original document content
    ├── Processed text segments
    └── Metadata and source information
```

**Query Processing Pipeline**: Search queries undergo multi-stage processing for optimal results:

1. **Query Analysis**: Parse query intent and extract keywords, entities, and semantic concepts
2. **Vector Search**: Generate query embeddings and perform similarity search across document embeddings
3. **Keyword Search**: Execute traditional full-text search for exact matches and phrase queries
4. **Hybrid Ranking**: Combine vector and keyword scores using configurable weighting algorithms
5. **Result Assembly**: Format results with highlighted matches, relevance scores, and source attribution

#### Request Processing Flow

Understanding the complete request processing flow helps developers optimize performance and debug issues:

**Incoming Request Processing**:
```text
HTTP Request → Authentication → Request Parsing → Context Assembly
```

**AI Decision Making**:
```text
Context + Available Functions → AI Processing → Function Selection + Parameters
```

**Function Execution**:
```text
Function Call → Parameter Validation → Execution (Skills/DataMap/SWAIG) → Result Processing
```

**Response Generation**:
```text
Function Results + AI Context → Response Generation → Output Formatting → HTTP Response
```

Each stage includes comprehensive error handling, logging, and performance monitoring to ensure production reliability.

### State Management Philosophy

The SDK's stateless-first approach represents a fundamental architectural decision that affects how developers approach agent design. This isn't simply an implementation detail—it's a design philosophy that enables scalable, reliable voice AI applications.

**Stateless Benefits in Practice**:

- **Horizontal Scaling**: Agents can be deployed across multiple servers or containers without session affinity requirements
- **Fault Tolerance**: Server failures don't result in lost conversation state since each request is self-contained
- **Development Simplicity**: No need to design, implement, or maintain state storage systems
- **Testing Reliability**: Agent behavior is deterministic based on inputs rather than accumulated state

**When State Becomes Necessary**:

Some applications require state persistence for functionality like multi-step workflows, user preferences, or session-specific data. The SDK provides elegant solutions through the SWAIG platform:

```python
@AgentBase.tool(
    name="start_order_process",
    description="Start a new order workflow",
    parameters={}
)
def start_order_process(self, args, raw_data):
    # Initialize order state
    order_data = {
        "order_id": generate_order_id(),
        "items": [],
        "customer_info": {},
        "step": "collecting_items"
    }

    result = SwaigFunctionResult("Let's start your order! What would you like to add?")

    # Store state in global data for AI access
    result.add_action("set_global_data", {
        "current_order": order_data
    })

    return result

@AgentBase.tool(
    name="add_item_to_order",
    description="Add an item to the current order",
    parameters={
        "item_sku": {"type": "string", "description": "Product SKU"},
        "item_name": {"type": "string", "description": "Product name"},
        "quantity": {"type": "integer", "description": "Quantity to add"}
    }
)
def add_item_to_order(self, args, raw_data):
    # Retrieve current state
    current_order = self.get_global_data("current_order", {})

    if not current_order:
        return SwaigFunctionResult("I don't see an active order. Let me start a new one for you.")

    # Update state with new item
    current_order["items"].append({
        "sku": args.get("item_sku"),
        "quantity": args.get("quantity", 1),
        "added_at": datetime.now().isoformat()
    })

    result = SwaigFunctionResult(f"Added {args.get('quantity', 1)} {args.get('item_name')} to your order.")

    # Update stored state
    result.add_action("set_global_data", {
        "current_order": current_order
    })

    return result
```

**Global Data vs Meta Data**:

- **Global Data**: Available to the AI during conversation processing, can influence AI decision-making and responses
- **Meta Data**: Available to functions but not directly to the AI, useful for tracking technical information and cross-function communication

### Class Hierarchy and Inheritance

The SDK uses a clear inheritance structure that promotes code reuse while allowing customization:

```text
AgentBase (Abstract base class)
├── Agent (Main implementation)
│   ├── Skills integration
│   ├── DataMap tools support
│   ├── Local search capabilities
│   └── SWAIG function handling
├── InfoGathererAgent (Prefab)
├── FAQBotAgent (Prefab)
├── ConciergeAgent (Prefab)
└── SurveyAgent (Prefab)
```

**AgentBase**: Defines the core interface and common functionality shared by all agent types. This includes basic web server setup, request handling, and the fundamental conversation loop.

**Agent**: The primary implementation that most developers will use. Includes full support for Skills, DataMap tools, local search, and custom SWAIG functions. Provides the complete feature set of the SDK.

**Prefab Agents**: Specialized implementations for common use cases. Each prefab extends the base Agent class with domain-specific prompts, pre-configured skills, and optimized workflows.

### Integration Architecture

The SDK is designed for seamless integration with existing systems and infrastructure through multiple architectural patterns:

**Microservice Integration**: Agents can operate as microservices within larger architectures, communicating with other services through DataMap tools or custom SWAIG functions.

**Event-Driven Architecture**: SWAIG functions can trigger events in external systems, enabling agents to participate in complex workflows and business processes.

**API Gateway Integration**: Agents work naturally behind API gateways, load balancers, and other infrastructure components common in enterprise environments.

**Authentication Integration**: The SDK supports multiple authentication patterns including OAuth, JWT tokens, API keys, and custom authentication mechanisms.

### Performance and Scalability Considerations

The architecture includes several design decisions optimized for production performance:

**Connection Pooling**: DataMap tools leverage SignalWire's connection pooling for efficient API calls without connection overhead.

**Caching Strategy**: Skills results and search indexes can be cached at multiple levels to reduce latency and improve response times.

**Resource Management**: The stateless design ensures that memory usage remains predictable and doesn't grow with conversation history.

**Concurrent Request Handling**: FastAPI's async architecture enables handling thousands of concurrent conversations on modest hardware resources.

This architectural foundation enables developers to build voice AI applications that are not only powerful and user-friendly but also maintainable, scalable, and production-ready. The layered design ensures that simple use cases remain simple while complex requirements can be addressed through the platform's extensive capabilities.

## 3. Installation and Setup

### Installation Options

The SignalWire AI Agents SDK offers flexible installation options depending on your requirements:

**Basic Installation**

For standard agent development with Skills System and DataMap tools:

```bash
pip install signalwire-agents
```

This includes all core functionality:
- Agent framework and web server
- Skills System with built-in skills (web_search, datetime, math)
- DataMap tools for API integration
- SWAIG functions for custom logic
- Prefab agents and prompt utilities

**Installation with Search Features**

The SDK provides multiple search installation options to match your needs. Choose the option that balances features with installation size:

### Basic Search (~500MB)
For vector embeddings and keyword search with minimal dependencies:

```bash
pip install "signalwire-agents[search]"
```

**Includes:**
- Core search functionality with vector similarity
- Document indexing and CLI tools  
- SQLite-based search indexes
- Basic document processing (text, markdown)

### Full Document Processing (~600MB)
For comprehensive document support including PDF, DOCX, Excel, PowerPoint:

```bash
pip install "signalwire-agents[search-full]"
```

**Adds:**
- PDF processing capabilities
- Microsoft Office document support (DOCX, Excel, PowerPoint)
- HTML and additional file format processing

### Advanced NLP Features (~700MB)
For enhanced search quality with advanced natural language processing:

```bash
pip install "signalwire-agents[search-nlp]"
```

**Adds:**
- spaCy for advanced text processing
- Better query understanding and synonym expansion
- Named entity recognition

**⚠️ Additional Setup Required:**
```bash
python -m spacy download en_core_web_sm
```

**Performance Note:** Advanced NLP features provide significantly better search quality but are 2-3x slower than basic search.

### All Search Features (~700MB)
For complete search functionality:

```bash
pip install "signalwire-agents[search-all]"
```

**Includes:** All search features combined

**⚠️ Additional Setup Required:**
```bash
python -m spacy download en_core_web_sm
```

### Feature Comparison

| Feature | Basic | Full | NLP | All |
|---------|-------|------|-----|-----|
| Vector embeddings | ✅ | ✅ | ✅ | ✅ |
| Keyword search | ✅ | ✅ | ✅ | ✅ |
| Text files (txt, md) | ✅ | ✅ | ✅ | ✅ |
| PDF processing | ❌ | ✅ | ❌ | ✅ |
| DOCX processing | ❌ | ✅ | ❌ | ✅ |
| Excel/PowerPoint | ❌ | ✅ | ❌ | ✅ |
| Advanced NLP | ❌ | ❌ | ✅ | ✅ |

**Development Installation**

For SDK development or running examples:

```bash
git clone https://github.com/signalwire/signalwire-agents
cd signalwire-agents
pip install -e ".[search]"
```

### Installation Verification

Check if search functionality is available:

```python
try:
    from signalwire_agents.search import IndexBuilder, SearchEngine
    print("✅ Search functionality is available")
except ImportError as e:
    print(f"❌ Search not available: {e}")
    print("Install with: pip install signalwire-agents[search]")
```

### Common Installation Issues

**"Illegal instruction" Error**: On older server hardware, you may encounter CPU instruction set incompatibility. Set these environment variables:

```bash
export PYTORCH_DISABLE_AVX2=1
export PYTORCH_DISABLE_AVX512=1
```

Then run your installation or commands:
```bash
PYTORCH_DISABLE_AVX2=1 PYTORCH_DISABLE_AVX512=1 pip install signalwire-agents[search]
```

**Development Dependencies**
- `pytest`: Testing framework
- `black`: Code formatting
- `flake8`: Linting
- `mypy`: Type checking

### Environment Setup

**Required Environment Variables**

Set your SignalWire credentials:

```bash
export SIGNALWIRE_PROJECT_ID="your-project-id"
export SIGNALWIRE_TOKEN="your-auth-token"
export SIGNALWIRE_SPACE="your-space-name.signalwire.com"
```

**Optional Configuration**

For enhanced functionality, consider setting:

```bash
# For web search skill (if using external search APIs)
export SEARCH_API_KEY="your-search-api-key"

# For custom authentication
export AGENT_AUTH_TOKEN="your-custom-token"

# For development
export FLASK_ENV="development"
export FLASK_DEBUG="1"
```

**Verification**

Test your installation:

```bash
python -c "import signalwire_agents; print('SDK installed successfully')"
```

For search features:

```bash
sw-search --help
```

You should see the search CLI help if the search dependencies are properly installed.

### Project Structure

A typical SignalWire AI agent project structure:

```
my-agent/
├── agent.py              # Main agent implementation
├── requirements.txt      # Dependencies
├── .env                 # Environment variables
├── knowledge/           # Documents for search indexing
│   ├── docs/
│   └── faqs/
├── indexes/             # Generated search indexes
│   └── knowledge.swsearch
└── README.md           # Project documentation
```

This structure supports both simple single-agent deployments and complex multi-agent systems with shared knowledge bases.

## 4. Building Your First AI Agent

Creating your first AI agent with the SignalWire SDK has never been easier thanks to the Skills System. In this section, we'll build a capable agent that can answer questions, search the web, and provide current date/time information—all with minimal code.

### Setting Up Your Environment

1. **Install the SDK**:
   ```bash
   pip install signalwire-agents
   ```

2. **Environment Variables** (Optional):
   For consistent authentication credentials across restarts, you can set these environment variables:
   ```bash
   export SIGNALWIRE_PROJECT_ID="your-project-id"
   export SIGNALWIRE_TOKEN="your-auth-token"
   export SIGNALWIRE_SPACE="your-space-name.signalwire.com"
   ```
   If not specified, the SDK will generate random credentials on startup.

### Creating a Simple Question-Answering Agent

Let's start with the most basic agent that can hold conversations:

```python
from signalwire_agents import AgentBase

class QuickStartAssistant(AgentBase):
    def __init__(self):
        super().__init__(name="QuickStart Assistant")
        self.set_prompt_text("You are a helpful assistant that can answer questions and provide information.")

agent = QuickStartAssistant()

if __name__ == "__main__":
    agent.run()
```

This minimal agent can already:
- Handle incoming phone calls and conversations
- Maintain context throughout the conversation
- Provide natural responses using AI
- Automatically manage web endpoints and request handling

### Using the Skills System

Now let's transform this basic agent into a powerful assistant by adding skills. The Skills System allows you to add complex capabilities with simple one-liner calls:

```python
from signalwire_agents import AgentBase

class EnhancedAssistant(AgentBase):
    def __init__(self):
        super().__init__(name="Enhanced Assistant")
        self.set_prompt_text(
            "You are a knowledgeable assistant that can help with questions, "
            "search the internet for current information, and provide date/time details."
        )
        
        # Add skills to enhance the agent's capabilities
        self.add_skill("web_search", {
            "api_key": "your-google-api-key",
            "search_engine_id": "your-search-engine-id",
            "num_results": 3,
            "delay": 0.5
        })
        
        self.add_skill("datetime")
        
        self.add_skill("math")

agent = EnhancedAssistant()

if __name__ == "__main__":
    agent.run()
```

With just three additional lines, your agent can now:

- **Search the web** for current information and news
- **Provide date/time information** in any timezone
- **Perform mathematical calculations** including complex operations

### Skills Configuration

Each skill accepts configuration parameters to customize its behavior. Here are some examples:

**Web Search Configuration**:
```python
agent.add_skill("web_search", {
    "api_key": "your-google-api-key",           # Google API key
    "search_engine_id": "your-search-engine-id", # Google Search Engine ID
    "num_results": 5,                    # Number of search results to retrieve
    "delay": 0.5                         # Delay between requests
})
```

**Date/Time Configuration**:
```python
agent.add_skill("datetime")  # Datetime skill uses simple configuration
```

**Math Configuration**:
```python
agent.add_skill("math")  # Math skill works with default settings - no configuration needed
```

### Creating Multiple Skill Instances

You can add the same skill multiple times with different configurations for specialized use cases:

```python
# General web search
agent.add_skill("web_search", {
    "api_key": "your-google-api-key",
    "search_engine_id": "your-search-engine-id",
    "num_results": 3
})

# Detailed research search
agent.add_skill("web_search", {
    "api_key": "your-google-api-key",
    "search_engine_id": "your-search-engine-id",
    "num_results": 10,
    "max_content_length": 2000,
    "tool_name": "research_search"
})

# Quick fact checking
agent.add_skill("web_search", {
    "api_key": "your-google-api-key", 
    "search_engine_id": "your-search-engine-id",
    "num_results": 1,
    "max_content_length": 200,
    "tool_name": "fact_check"
})
```

The AI will automatically choose the most appropriate search function based on the user's request.

### Adding Knowledge Search

For agents that need to search through specific documents or knowledge bases, you can add the local search skill:

```python
# First, create a search index from your documents
# Run this command in your terminal:
# sw-search docs/ --output knowledge.swsearch

# Then add the search skill to your agent
agent.add_skill("native_vector_search", {
    "index_file": "knowledge.swsearch",
    "count": 5,
    "distance_threshold": 0.7
})
```

This enables your agent to search through your own documents using advanced vector similarity and keyword search.

### Running and Testing Your Agent

To run your agent:

```bash
python agent.py
```

You'll see output similar to:
```
 * Agent 'Enhanced Assistant' running on http://0.0.0.0:5000
 * Available skills: web_search, datetime, math
 * Authentication: Basic (username: 8A7B9C, password: X3R8K9)
```

### Testing Skills

You can test your agent's new capabilities by calling it and asking questions like:

- **Web Search**: "What's the latest news about AI?"
- **Date/Time**: "What time is it in Tokyo right now?"
- **Math**: "What's the square root of 1764?"
- **Knowledge Search**: "Tell me about our refund policy" (if using local search)

The AI will automatically decide which skills to use based on the user's request and will call them seamlessly during the conversation.

### Understanding What Happened

By adding skills, you've:

1. **Eliminated Custom Function Development**: No need to implement web search APIs, timezone handling, or mathematical operations
2. **Gained Advanced Capabilities**: Each skill includes sophisticated error handling, parameter validation, and result formatting
3. **Maintained Simplicity**: The AI automatically decides when and how to use each skill
4. **Ensured Reliability**: Skills are tested and maintained as part of the SDK

### Next Steps

This basic skills-enabled agent demonstrates the power of the modern SDK approach. In the following sections, we'll explore:

- Advanced skills configuration and custom skills development
- DataMap tools for API integration without webhooks
- Local search systems for knowledge base functionality
- When to use custom SWAIG functions versus built-in capabilities
- Best practices for combining multiple approaches

The Skills System represents a fundamental shift in how AI agents are built—from implementation-heavy custom functions to configuration-driven capabilities that just work.

## 5. Skills System: Add Capabilities with One-Liners

The Skills System represents a paradigm shift in AI agent development. Instead of implementing custom functions for common tasks, you can add sophisticated capabilities to your agents with simple one-liner calls. This dramatically reduces development time while providing robust, tested functionality.

### What is the Skills System

The Skills System is a collection of pre-built, configurable capabilities that can be added to any agent using the `add_skill()` method. Each skill:

- **Encapsulates complex functionality** into simple interfaces
- **Includes comprehensive error handling** and parameter validation  
- **Provides configurable behavior** through parameter dictionaries
- **Integrates seamlessly** with the AI's decision-making process
- **Maintains consistent APIs** across different skill types
- **Supports multiple instances** of the same skill with different configurations

Skills are automatically discovered by the AI and appear as available functions that can be called during conversations. The AI decides when and how to use each skill based on user requests and the skill's description.

### Available Built-in Skills

The SDK includes several powerful built-in skills ready for immediate use:

#### Web Search Skill (`web_search`)

Enables agents to search the internet for current information, news, and facts.

**Basic Usage:**
```python
agent.add_skill("web_search")
```

**Advanced Configuration:**
```python
agent.add_skill("web_search", {
    "api_key": "your-google-api-key",           # Google Custom Search API key
    "search_engine_id": "your-search-engine-id", # Google Search Engine ID  
    "num_results": 5,                    # Number of search results (1-10)
    "delay": 1.0,                        # Delay between requests in seconds
    "no_results_message": "Sorry, I couldn't find information about '{query}'. Try a different search term."
})
```

**Use Cases:**
- Current news and events
- Product information and reviews
- Research and fact-checking
- Weather updates
- Stock prices and financial information

#### Date/Time Skill (`datetime`)

Provides comprehensive date, time, and timezone functionality.

**Basic Usage:**
```python
agent.add_skill("datetime")
```

**Advanced Configuration:**
```python
agent.add_skill("datetime")  # Datetime skill works with default settings
```

**Capabilities:**
- Current date and time in any timezone
- Time zone conversions
- Date calculations and differences
- Business hours checking
- Holiday and weekend detection
- Scheduling assistance

#### Math Skill (`math`)

Performs basic mathematical calculations using a simple expression evaluator.

**Usage:**
```python
agent.add_skill("math")  # No configuration parameters needed
```

**Capabilities:**
- Basic arithmetic operations: +, -, *, /, %, ** (power)
- Parentheses for complex expressions
- Decimal number support
- Expression validation for security

**Note:** This skill provides a single `calculate` function that safely evaluates mathematical expressions. It does not support advanced features like trigonometry, unit conversions, or financial calculations.

#### Native Vector Search Skill (`native_vector_search`)

Searches through local document collections using vector similarity and keyword search.

**Basic Usage:**
```python
# First create a search index
# sw-search docs/ --output knowledge.swsearch

agent.add_skill("native_vector_search", {
    "index_file": "knowledge.swsearch"
})
```

**Advanced Configuration:**
```python
agent.add_skill("native_vector_search", {
    "index_file": "knowledge.swsearch",
    "max_results": 10,                   # Maximum search results
    "similarity_threshold": 0.7,         # Minimum similarity score
    "hybrid_search": True,               # Combine vector + keyword
    "boost_keywords": True,              # Boost exact keyword matches
    "search_fields": ["content", "title"], # Fields to search
    "result_format": "detailed"          # How to format results
})
```

**Features:**
- Semantic search using vector embeddings
- Keyword-based search for exact matches
- Hybrid search combining both approaches
- Customizable similarity thresholds
- Multiple document format support
- Offline operation (no API calls required)

### Configuration and Parameters

Skills accept configuration parameters that customize their behavior. Here are common configuration patterns:

#### Global Skill Configuration

Set default parameters that apply to all instances of a skill:

```python
# Skills use configuration parameters passed to add_skill()
# Global configuration can be set via environment variables

# Basic web search with default parameters
agent.add_skill("web_search")
```

#### Instance-Specific Configuration

Override defaults for specific skill instances:

```python
# General search with defaults
agent.add_skill("web_search")

# News-focused search with custom config
agent.add_skill("web_search", {
    "api_key": "your-google-api-key",
    "search_engine_id": "news-search-engine-id",
    "num_results": 5,
    "tool_name": "news_search"
})
```

#### Multiple Configuration Examples

Skills can be configured with different parameters:

```python
# Basic datetime skill
agent.add_skill("datetime")

# Web search with specific parameters
agent.add_skill("web_search", {
    "api_key": "your-google-api-key",
    "search_engine_id": "your-search-engine-id",
    "num_results": 3
})
```

### Creating Multiple Skill Instances

You can add the same skill multiple times with different configurations for specialized use cases:

```python
# Quick search for immediate answers
agent.add_skill("web_search", {
    "api_key": "your-google-api-key",
    "search_engine_id": "quick-search-engine-id",
    "num_results": 1,
    "delay": 0,
    "tool_name": "quick_search"
})

# Research search for detailed information
agent.add_skill("web_search", {
    "api_key": "your-google-api-key",
    "search_engine_id": "research-search-engine-id",
    "num_results": 5,
    "delay": 1.0,
    "tool_name": "research_search"
})

# Another web search instance for news
agent.add_skill("web_search", {
    "api_key": "your-google-api-key",
    "search_engine_id": "news-engine-id", 
    "num_results": 5,
    "tool_name": "news_search"
})
```

The AI will automatically choose the most appropriate search function based on the user's request and context.

### Custom Skills Development

You can create custom skills to extend the SDK's capabilities:

#### Basic Custom Skill

```python
from signalwire_agents.core.skill_base import SkillBase

class DatabaseLookupSkill(SkillBase):
    """Custom skill for database queries"""
    
    SKILL_NAME = "database_lookup"
    SKILL_DESCRIPTION = "Look up information in the company database"
    SKILL_VERSION = "1.0.0"
    REQUIRED_PACKAGES = []
    REQUIRED_ENV_VARS = []
    
    def setup(self) -> bool:
        """Setup the skill"""
        return True
        
    def register_tools(self) -> None:
        """Register database lookup tool"""
        self.define_tool(
            name="lookup_database",
            description="Look up information in the company database",
            parameters={
                "table": {
                    "type": "string",
                    "description": "Database table to query"
                },
                "criteria": {
                    "type": "object",
                    "description": "Search criteria"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results",
                    "default": 10
                }
            },
            handler=self._database_handler
        )
    
    def _database_handler(self, args, raw_data):
        """Handle database lookup requests"""
        table = args.get("table")
        criteria = args.get("criteria", {})
        limit = args.get("limit", 10)
        
        # Your database query logic here
        results = self.query_database(table, criteria, limit)
        
        from signalwire_agents.core.function_result import SwaigFunctionResult
        return SwaigFunctionResult(f"Found {len(results)} records in {table}")
    
    def query_database(self, table, criteria, limit):
        # Implement your database query logic
        return []

# Register and use the custom skill
agent.register_skill(DatabaseLookupSkill)
agent.add_skill("database_lookup", {
    "default_table": "customers",
    "connection_string": "your_db_connection"
})
```

#### Advanced Custom Skill with Configuration

```python
class WeatherAPISkill(Skill):
    """Custom weather skill with API integration"""
    
    skill_name = "weather_api"
    description = "Get current weather and forecasts using external API"
    
    parameters = [
        SkillParameter("location", "string", "City or coordinates", required=True),
        SkillParameter("forecast_days", "integer", "Number of forecast days", default=1),
        SkillParameter("include_alerts", "boolean", "Include weather alerts", default=False)
    ]
    
    def __init__(self, config=None):
        super().__init__(config)
        self.api_key = config.get("api_key") if config else None
        self.units = config.get("units", "metric")
        self.language = config.get("language", "en")
    
    def execute(self, args, agent):
        if not self.api_key:
            return {"success": False, "error": "API key not configured"}
        
        location = args["location"]
        forecast_days = args.get("forecast_days", 1)
        include_alerts = args.get("include_alerts", False)
        
        try:
            weather_data = self.fetch_weather(location, forecast_days, include_alerts)
            return {
                "success": True,
                "data": weather_data,
                "message": f"Weather information for {location}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def fetch_weather(self, location, forecast_days, include_alerts):
        # Implement API call logic
        pass

# Use with configuration
agent.register_skill(WeatherAPISkill)
agent.add_skill("weather_api", {
    "api_key": "your_api_key",
    "units": "imperial",
    "language": "en"
})
```

### Skill Best Practices

#### Choosing the Right Skill

- **Use built-in skills** for common operations (web search, math, datetime)
- **Create custom skills** for business-specific logic or specialized APIs
- **Combine multiple skills** for complex workflows
- **Consider DataMap tools** for simple API integrations without custom code

#### Configuration Guidelines

```python
# Good: Clear, specific configuration
agent.add_skill("web_search", {
    "num_results": 3,           # Specific number
    "max_characters": 1000,     # Reasonable limit
    "search_type": "general"    # Clear intent
})

# Better: Environment-based configuration
import os

agent.add_skill("web_search", {
    "num_results": int(os.getenv("SEARCH_RESULTS", 3)),
    "api_key": os.getenv("SEARCH_API_KEY"),
    "rate_limit": int(os.getenv("SEARCH_RATE_LIMIT", 100))
})
```

#### Error Handling

```python
# Skills should handle errors gracefully
class RobustCustomSkill(Skill):
    def execute(self, args, agent):
        try:
            # Main logic here
            result = self.process_request(args)
            return {"success": True, "data": result}
        except ValueError as e:
            # Handle specific errors
            return {"success": False, "error": f"Invalid input: {e}"}
        except Exception as e:
            # Handle unexpected errors
            agent.logger.error(f"Skill error: {e}")
            return {"success": False, "error": "An unexpected error occurred"}
```

#### Performance Considerations

```python
# Cache expensive operations
class CachedAPISkill(Skill):
    def __init__(self, config=None):
        super().__init__(config)
        self.cache = {}
        self.cache_ttl = config.get("cache_ttl", 300)  # 5 minutes
    
    def execute(self, args, agent):
        cache_key = self.generate_cache_key(args)
        
        # Check cache first
        if cache_key in self.cache:
            cached_result, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_result
        
        # Fetch new data
        result = self.fetch_data(args)
        
        # Cache result
        self.cache[cache_key] = (result, time.time())
        
        return result
```

### Skills vs. Other Approaches

The Skills System is part of a broader toolkit. Here's when to use each approach:

**Use Skills When:**
- The functionality is commonly needed across multiple agents
- You want zero-implementation solutions
- You need configurable, reusable capabilities
- The operation doesn't require complex business logic

**Use DataMap Tools When:**
- You need to integrate with REST APIs
- You want server-side execution without webhooks
- The integration is primarily data transformation
- You prefer declarative configuration over code

**Use Custom SWAIG Functions When:**
- You need complex business logic
- You require deep integration with your systems
- You want full control over the implementation
- The functionality is highly specific to your use case

The Skills System represents the future of AI agent development—powerful, configurable capabilities that eliminate the need for repetitive custom implementations while maintaining the flexibility to extend functionality when needed.

## 6. DataMap Tools: Server-Side API Integration

DataMap Tools revolutionize how AI agents integrate with external APIs by eliminating the need for webhook infrastructure. Instead of writing custom endpoints and handling HTTP requests in your application, you can define API integrations declaratively and have them execute on SignalWire's servers.

### What are DataMap Tools

DataMap Tools are server-side API integration utilities that:

- **Execute on SignalWire's infrastructure**, not your servers
- **Use declarative configuration** instead of imperative code
- **Support variable expansion** for dynamic parameter substitution
- **Include built-in error handling** and retry logic
- **Provide response transformation** capabilities
- **Eliminate webhook development** for simple API integrations

Unlike traditional approaches that require you to build and maintain webhook endpoints, DataMap tools are defined in your agent code but run remotely when called by the AI.

### Builder Pattern and Configuration

DataMap tools use a fluent builder pattern that makes configuration intuitive and readable:

#### Basic DataMap Structure

```python
from signalwire_agents.core.data_map import DataMap
from signalwire_agents.core.function_result import SwaigFunctionResult

# Simple API call example
weather_tool = (DataMap('get_weather')
    .parameter('location', 'string', 'City name', required=True)
    .webhook('GET', 'https://api.weather.com/v1/current?key=API_KEY&q=${args.location}')
    .output(SwaigFunctionResult('Weather in ${args.location}: ${response.current.temp_f}°F'))
)

# Register with agent
agent.register_swaig_function(weather_tool.to_swaig_function())
```

#### Advanced Configuration Example

```python
# Complex API integration with correct configuration
user_lookup_tool = (DataMap('lookup_user')
    .parameter('user_id', 'string', 'User identifier', required=True)
    .parameter('include_history', 'boolean', 'Include order history', required=False)
    .parameter('format', 'string', 'Response format', enum=['summary', 'detailed'], required=False)
    
    # Main API call with headers defined in webhook
    .webhook('GET', 'https://api.company.com/users/${args.user_id}', 
             headers={
                 'Authorization': 'Bearer YOUR_API_TOKEN',
                 'Content-Type': 'application/json',
                 'X-Include-History': '${args.include_history}'
             })
    
    # Output with response data
    .output(SwaigFunctionResult("""
        User: ${response.full_name} (ID: ${response.id})
        Status: ${response.account_status}
        Last Login: ${response.last_activity.login_time}
    """))
    
    # Error handling via error_keys
    .error_keys(['error', 'message', 'status'])
)
```

### REST API Integration

DataMap tools support comprehensive REST API integration patterns:

#### HTTP Methods and Parameters

```python
# GET request with query parameters in URL
get_tool = (DataMap('search_products')
    .parameter('query', 'string', 'Search term', required=True)
    .parameter('category', 'string', 'Product category', required=False)
    .parameter('limit', 'integer', 'Results limit', required=False)
    .webhook('GET', 'https://api.shop.com/products?q=${args.query}&cat=${args.category}&limit=${args.limit}')
    .output(SwaigFunctionResult('Found products: ${response.count} results'))
)

# POST request with JSON body
create_tool = (DataMap('create_ticket')
    .parameter('title', 'string', 'Ticket title', required=True)
    .parameter('description', 'string', 'Ticket description', required=True)
    .parameter('priority', 'string', 'Priority level', enum=['low', 'medium', 'high'], required=False)
    .webhook('POST', 'https://api.support.com/tickets',
             headers={'Content-Type': 'application/json'})
    .body({
        'title': '${args.title}',
        'description': '${args.description}',
        'priority': '${args.priority}',
        'created_by': 'ai_agent'
    })
    .output(SwaigFunctionResult('Created ticket #${response.id}: ${response.title}'))
)

# PUT request for updates  
update_tool = (DataMap('update_status')
    .parameter('ticket_id', 'string', 'Ticket ID', required=True)
    .parameter('status', 'string', 'New status', enum=['open', 'in_progress', 'resolved'], required=True)
    .webhook('PUT', 'https://api.support.com/tickets/${args.ticket_id}',
             headers={'Authorization': 'Bearer YOUR_API_KEY'})
    .body({'status': '${args.status}', 'updated_by': 'ai_agent'})
    .output(SwaigFunctionResult('Updated ticket ${args.ticket_id} to ${args.status}'))
)

# DELETE request
delete_tool = (DataMap('delete_item')
    .parameter('item_id', 'string', 'Item to delete', required=True)
    .webhook('DELETE', 'https://api.inventory.com/items/${args.item_id}',
             headers={'Authorization': 'API-Key YOUR_API_KEY'})
    .output(SwaigFunctionResult('Item ${args.item_id} deleted successfully'))
)
```

#### Authentication Patterns

```python
# Bearer token authentication
auth_tool = (DataMap('secure_action')
    .webhook('GET', 'https://api.secure.com/data', 
             headers={'Authorization': 'Bearer YOUR_AUTH_TOKEN'})
)

# API key in header  
api_key_tool = (DataMap('api_call')
    .webhook('GET', 'https://api.service.com/endpoint',
             headers={
                 'X-API-Key': 'YOUR_API_KEY',
                 'X-Client-ID': 'YOUR_CLIENT_ID'
             })
)

# Basic authentication (you would need to base64 encode credentials beforehand)
basic_auth_tool = (DataMap('basic_auth_call')
    .webhook('GET', 'https://api.legacy.com/data',
             headers={'Authorization': 'Basic YOUR_ENCODED_CREDENTIALS'})
)

# OAuth2 token
oauth_tool = (DataMap('oauth_call') 
    .webhook('GET', 'https://api.oauth.com/data',
             headers={'Authorization': 'Bearer YOUR_OAUTH_TOKEN'})
)
```

### Variable Expansion and Processing

DataMap tools use a powerful variable expansion system that allows dynamic substitution of values:

#### Data Store Usage Guidelines

**global_data** - Call-wide data store:
- User information collected during the call: `${global_data.customer_name}`, `${global_data.account_type}`
- Call state and preferences: `${global_data.preferred_language}`, `${global_data.call_reason}`
- **NEVER** API keys, passwords, or credentials

**meta_data** - Function-scoped data store:
- Function-specific state: `${meta_data.session_id}`, `${meta_data.retry_count}`
- Shared between functions with same meta_data_token
- **NEVER** sensitive configuration or secrets

#### Variable Types

```python
# Function arguments
.webhook('GET', 'https://api.example.com/users/${args.user_id}')

# Authentication headers
.header('Authorization', 'Bearer YOUR_API_TOKEN')

# Response data from API calls
.output(SwaigFunctionResult('Status: ${response.status}, Message: ${response.data.message}'))

# Static values and response variables
.body({'created_by': 'agent', 'api_version': '1.0'})
```

#### Response Data Access

```python
# Access nested response data
.output(SwaigFunctionResult('Status: ${response.status}, Message: ${response.data.message}'))

# Use response arrays (with foreach processing)
.foreach('${response.results}')
.output(SwaigFunctionResult('Found: ${foreach.title} - ${foreach.summary}'))

# Static values and simple variable substitution
'2024-01-01T00:00:00Z'                     # Static timestamp values
'${response.created_at}'                   # Response fields
'${args.user_id}'                          # Function arguments
```

#### Simple DataMap Example

```python
# Basic API integration that actually works
pricing_tool = (DataMap('get_product_price')
    .parameter('product_id', 'string', 'Product ID', required=True)
    .webhook('GET', 'https://api.example.com/products/${args.product_id}', 
             headers={'Authorization': 'Bearer YOUR_API_TOKEN'})
    .output(SwaigFunctionResult("""
        Product: ${response.name}
        Price: $${response.price}
        In Stock: ${response.in_stock}
    """))
    .error_keys(['error', 'message'])
)

### Error Handling

```python
# Simple error handling with error_keys
robust_tool = (DataMap('robust_api_call')
    .parameter('id', 'string', 'Resource ID', required=True)
    
    # Main API call
    .webhook('GET', 'https://api.service.com/resource/${args.id}')
    
    # Handle standard response
    .output(SwaigFunctionResult('Successfully retrieved: ${response.name} (Status: ${response.status})'))
)
```

#### Simple DataMap Example

```python
# Simple, working DataMap tool
basic_tool = (DataMap('lookup_user')
    .parameter('user_id', 'string', 'User ID', required=True)
    .webhook('GET', 'https://api.users.com/users/${args.user_id}')
    .header('Accept', 'application/json')
    .output(SwaigFunctionResult('${response.name} (${response.email})'))
)
```

### DataMap vs. Other Approaches

Understanding when to use DataMap tools versus other integration methods:

**Use DataMap Tools When:**
- Integrating with REST APIs that require simple request/response patterns
- You want server-side execution without maintaining webhook infrastructure
- The integration is primarily about data transformation and formatting
- You prefer declarative configuration over imperative code
- You need built-in retry logic and error handling

**Use Custom SWAIG Functions When:**
- You need complex business logic or multi-step operations
- The integration requires database access or file system operations
- You need to maintain state between function calls
- The API integration requires complex authentication flows
- You want full control over error handling and response processing

**Use Skills When:**
- The functionality is commonly needed across multiple agents
- You want zero-implementation solutions for standard operations
- The operation doesn't require specific API integrations

**Combining Approaches:**
```python
# Use DataMap for API calls, Skills for common operations
agent.add_skill("datetime")  # Built-in datetime functionality
agent.add_skill("web_search")  # Built-in web search

# DataMap for business-specific API
customer_tool = (DataMap('get_customer')
    .parameter('customer_id', 'string', 'Customer ID', required=True)
    .webhook('GET', 'https://api.crm.com/customers/${customer_id}')
    .output(SwaigFunctionResult('Customer: ${response.name}, Status: ${response.status}'))
)
agent.register_swaig_function(customer_tool.to_swaig_function())

# Custom SWAIG function for complex logic
@agent.swaig_function
def process_order(args):
    # Complex multi-step business logic
    customer = get_customer_from_db(args['customer_id'])
    inventory = check_inventory(args['items'])
    pricing = calculate_complex_pricing(customer, args['items'])
    # ... more complex logic
    return create_order(customer, inventory, pricing)
```

DataMap Tools represent a significant advancement in API integration simplicity, allowing developers to focus on business logic rather than infrastructure while maintaining the flexibility to handle complex integration scenarios.

## 7. Local Search System: Offline Knowledge Base

The Local Search System enables AI agents to search through document collections using advanced vector similarity and hybrid search techniques—all running completely offline without external API dependencies. This powerful feature allows agents to provide accurate, context-aware answers from your own knowledge bases, documentation, and content collections.

### Overview and Benefits

The Local Search System provides:

**Offline Operation**: Once indexes are built, searches run completely offline with no external API calls, ensuring privacy, reliability, and cost control.

**Vector Similarity Search**: Uses advanced embedding models to understand semantic meaning, enabling "fuzzy" searches that find relevant content even when exact keywords don't match.

**Keyword Search**: Traditional full-text search for exact matches and specific terms, ensuring precise retrieval when needed.

**Hybrid Search**: Combines vector and keyword search scores using intelligent ranking algorithms to provide the best possible results.

**Multiple Document Formats**: Supports text files, Markdown, Word documents, PDFs, Excel files, and more.

**Scalable Performance**: Efficient SQLite-based storage with FAISS vector indexes for fast search across large document collections.

**Zero Dependencies**: No external search services, APIs, or cloud dependencies required.

### Installation and Setup

#### Installing Search Dependencies

The search system requires additional dependencies that can be installed with the search extra:

```bash
# Install SDK with search capabilities
pip install "signalwire-agents[search]"
```

This installs:
- `sentence-transformers`: For generating document embeddings
- `faiss-cpu`: Efficient vector similarity search
- `python-docx`: Microsoft Word document support
- `openpyxl`: Excel file support
- `PyPDF2`: PDF document processing

#### Verify Installation

Test that search tools are available:

```bash
sw-search --help
```

You should see the search CLI help if dependencies are properly installed.

### Building Search Indexes

#### Basic Index Creation

Create a search index from a directory of documents:

```bash
# Index all documents in a directory
sw-search docs/ --output knowledge.swsearch

# Index with specific file types
sw-search docs/ --include "*.md,*.txt,*.pdf" --output knowledge.swsearch

# Index with custom settings
sw-search docs/ --output knowledge.swsearch --chunk-size 500 --overlap 50
```

#### Advanced Index Configuration

```bash
# Comprehensive indexing with all options
sw-search docs/ \
    --output knowledge.swsearch \
    --include "*.md,*.txt,*.pdf,*.docx" \
    --exclude "*.tmp,*~" \
    --chunk-size 1000 \
    --overlap 100 \
    --model "all-MiniLM-L6-v2" \
    --batch-size 32 \
    --verbose
```

**Configuration Options:**

- `--chunk-size`: Size of text chunks for indexing (default: 1000 characters)
- `--overlap`: Overlap between chunks to preserve context (default: 200 characters)
- `--model`: Embedding model to use (default: "all-MiniLM-L6-v2")
- `--batch-size`: Processing batch size for performance tuning
- `--include`: File patterns to include (comma-separated)
- `--exclude`: File patterns to exclude (comma-separated)
- `--verbose`: Enable detailed progress output

#### Supported Document Formats

The indexing system supports multiple document formats:

```bash
# Text-based formats
sw-search docs/ --include "*.txt,*.md,*.rst,*.log"

# Office documents
sw-search docs/ --include "*.docx,*.xlsx,*.pptx"

# PDFs and web formats
sw-search docs/ --include "*.pdf,*.html,*.xml"

# Mixed content
sw-search docs/ --include "*.md,*.pdf,*.docx,*.txt"
```

#### Programmatic Index Building

You can also build indexes programmatically:

```python
from signalwire_agents.cli.search import IndexBuilder

# Create index builder - Note: This is currently CLI-only functionality
# For programmatic use, use the sw-search command line tool
# Example: sw-search docs/ --output knowledge.swsearch
```

### Using the Search Skill

#### Basic Search Integration

Add the search skill to your agent:

```python
from signalwire_agents import AgentBase

agent = AgentBase("Knowledge Assistant")
agent.set_prompt_text("I help answer questions using our knowledge base and can search the web for additional information.")

# Add search capability
agent.add_skill("native_vector_search", {
    "index_file": "knowledge.swsearch"
})

# Also add web search for external information
agent.add_skill("web_search", {
    "num_results": 3
})

if __name__ == "__main__":
    agent.run()
```

#### Advanced Search Configuration

```python
# Comprehensive search configuration
agent.add_skill("native_vector_search", {
    "index_file": "knowledge.swsearch",
    "count": 10,                          # Maximum search results
    "distance_threshold": 0.7,            # Minimum similarity score (0.0-1.0)
    "tool_name": "search_knowledge",      # Custom tool name
    "description": "Search the knowledge base",  # Tool description
    "no_results_message": "No information found for '{query}'",  # Custom no results message
    "response_prefix": "",                # Text to add before results
    "response_postfix": "",               # Text to add after results
    "tags": []                            # Filter by tags
})
```

#### Multiple Knowledge Bases

You can add multiple search skills for different knowledge domains:

```python
# General company knowledge
agent.add_skill("native_vector_search", {
    "index_file": "general_knowledge.swsearch",
    "count": 5,
    "tool_name": "search_general"
})

# Technical documentation
agent.add_skill("native_vector_search", {
    "index_file": "tech_docs.swsearch",
    "distance_threshold": 0.8,
    "count": 3,
    "tool_name": "search_technical"
})

# FAQ and support content
agent.add_skill("native_vector_search", {
    "index_file": "support_faq.swsearch",
    "count": 5,
    "tool_name": "search_support"
})
```

The AI will automatically choose the most appropriate search function based on the user's question and context.

### CLI Tools and Advanced Configuration

#### Index Management Commands

```bash
# Check index information
sw-search-info knowledge.swsearch

# Merge multiple indexes
sw-search-merge index1.swsearch index2.swsearch --output combined.swsearch

# Update existing index with new documents
sw-search-update knowledge.swsearch --add-dir new_docs/

# Validate index integrity
sw-search-validate knowledge.swsearch

# Export index contents
sw-search-export knowledge.swsearch --format json --output knowledge.json
```

#### Search Testing

Test your indexes from the command line:

```bash
# Test search queries
sw-search-query knowledge.swsearch "How do I reset my password?"

# Test with specific parameters
sw-search-query knowledge.swsearch "API documentation" \
    --max-results 5 \
    --similarity-threshold 0.7 \
    --hybrid-search

# Batch testing with query file
sw-search-batch knowledge.swsearch --queries queries.txt --output results.json
```

#### Performance Optimization

```bash
# Optimize index for faster searches
sw-search-optimize knowledge.swsearch

# Rebuild index with better settings
sw-search docs/ --output knowledge_v2.swsearch \
    --chunk-size 800 \
    --overlap 150 \
    --model "all-mpnet-base-v2" \
    --optimize-for speed

# Create compressed index for deployment
sw-search-compress knowledge.swsearch --output knowledge_compressed.swsearch
```

### Advanced Features and Configuration

#### Custom Embedding Models

Use different embedding models for specialized content:

```python
# High-quality model for general content
sw-search docs/ --model "all-mpnet-base-v2" --output general.swsearch

# Multilingual model for international content
sw-search docs/ --model "paraphrase-multilingual-MiniLM-L12-v2" --output multilingual.swsearch

# Domain-specific model for technical content
sw-search technical_docs/ --model "sentence-transformers/allenai-specter" --output technical.swsearch
```

#### Metadata and Filtering

Add rich metadata to enable filtered searches:

```python
from signalwire_agents.search import IndexBuilder

builder = IndexBuilder()

# Add documents with metadata
builder.add_file("user_guide.pdf", metadata={
    "category": "documentation",
    "audience": "end_user",
    "version": "2.1",
    "language": "en"
})

builder.add_file("api_reference.md", metadata={
    "category": "technical",
    "audience": "developer",
    "version": "2.1",
    "language": "en"
})

# Build index with metadata support
builder.build_index("categorized.swsearch")
```

Use metadata in search queries:

```python
agent.add_skill("native_vector_search", {
    "index_file": "categorized.swsearch",
    "filter_metadata": {
        "category": "documentation",
        "audience": "end_user"
    }
})
```

#### Real-time Index Updates

Keep indexes current with automatic updates:

```python
from signalwire_agents.search import IndexWatcher
import asyncio

async def maintain_search_index():
    """Keep search index updated automatically"""
    watcher = IndexWatcher(
        watch_directory="docs/",
        index_file="knowledge.swsearch",
        update_interval=300  # Check every 5 minutes
    )
    
    await watcher.start()

# Run in background
asyncio.create_task(maintain_search_index())
```

#### Search Analytics and Monitoring

Track search performance and user queries:

```python
from signalwire_agents.search import SearchAnalytics

# Enable search analytics
agent.add_skill("native_vector_search", {
    "index_file": "knowledge.swsearch",
    "enable_analytics": True,
    "analytics_file": "search_analytics.json"
})

# Analyze search patterns
analytics = SearchAnalytics.load("search_analytics.json")
print(f"Top queries: {analytics.top_queries(10)}")
print(f"Low-relevance queries: {analytics.low_relevance_queries()}")
print(f"Search success rate: {analytics.success_rate():.2%}")
```

### Best Practices

#### Index Organization

```bash
# Organize by content type
sw-search user_docs/ --output user_knowledge.swsearch
sw-search technical_docs/ --output tech_knowledge.swsearch
sw-search faq/ --output faq_knowledge.swsearch

# Organize by update frequency
sw-search static_content/ --output static.swsearch
sw-search dynamic_content/ --output dynamic.swsearch --watch
```

#### Content Preparation

**For Better Search Results:**

1. **Structure Documents Well**: Use clear headings, sections, and logical organization
2. **Include Relevant Keywords**: Ensure important terms appear in the content
3. **Write Descriptive Titles**: Help the search system understand document topics
4. **Add Metadata**: Include categories, tags, and other structured information
5. **Remove Noise**: Clean up formatting artifacts and irrelevant content

**Document Optimization Example:**

```markdown
# Password Reset Procedure
*Category: User Support | Audience: End Users | Last Updated: 2024-01-15*

## Overview
This guide explains how to reset your password when you've forgotten it or need to change it for security reasons.

## Step-by-Step Instructions
1. Navigate to the login page
2. Click "Forgot Password?"
3. Enter your email address
4. Check your email for reset instructions
5. Follow the link in the email
6. Create a new secure password

## Troubleshooting
If you don't receive the reset email:
- Check your spam folder
- Verify you're using the correct email address
- Contact support if issues persist

## Related Topics
- Account Security Best Practices
- Two-Factor Authentication Setup
- Password Requirements
```

#### Performance Tuning

```python
# Optimize for search speed vs. accuracy
fast_config = {
    "chunk_size": 500,           # Smaller chunks for faster processing
    "similarity_threshold": 0.8, # Higher threshold for fewer results
    "max_results": 3,            # Limit result count
    "hybrid_search": False       # Vector-only for speed
}

# Optimize for search accuracy
accurate_config = {
    "chunk_size": 1200,          # Larger chunks for more context
    "similarity_threshold": 0.6, # Lower threshold for more results
    "max_results": 10,           # More comprehensive results
    "hybrid_search": True,       # Best of both search types
    "rerank_results": True       # Re-rank for optimal ordering
}
```

#### Security and Privacy

```python
# Secure deployment configuration
secure_agent = Agent(
    assistant=Assistant(
        name="Secure Knowledge Assistant",
        purpose="I provide information from our secure knowledge base."
    )
)

# Add search with access controls
secure_agent.add_skill("native_vector_search", {
    "index_file": "secure_knowledge.swsearch",
    "access_control": True,
    "allowed_categories": ["public", "internal"],  # Filter by user access level
    "redact_sensitive": True,    # Automatically redact sensitive information
    "audit_searches": True       # Log all search queries for compliance
})
```

The Local Search System transforms how AI agents access and utilize knowledge, providing powerful offline search capabilities that rival cloud-based solutions while maintaining complete control over your data and infrastructure.

## 8. Advanced Agent Customization

Now that you've built a basic agent, let's explore advanced customization options that can make your agents more sophisticated, versatile, and effective in real-world scenarios.

### Prompt Building with POM

The Prompt Object Model (POM) is a key feature of the SignalWire AI Agents SDK that allows for structured, maintainable prompt construction. While we used some basic POM capabilities in our simple agent, the full power of POM offers much more flexibility:

#### Section Types and Organization

The POM supports various section types to organize your agent's instructions:

```python
# Main sections with body text
self.prompt_add_section("Personality", 
                       body="You are a customer service representative for SignalWire.")

# Sections with bullet points
self.prompt_add_section("Instructions", bullets=[
    "Always ask for account verification before sharing sensitive information",
    "Offer to escalate to a human if you cannot resolve the issue",
    "Maintain a professional but friendly tone"
])

# Sections with both body and bullets
self.prompt_add_section("Context", 
                       body="The user is calling about a technical issue.",
                       bullets=["They may be experiencing service disruption", 
                               "Have technical account details ready"])

# Numbered sections
self.prompt_add_section("Process", 
                       bullets=["Verify identity", "Diagnose problem", "Suggest solution"],
                       numbered=True)  # Creates a numbered list

# Add subsections for better organization
self.prompt_add_subsection("Instructions", "Technical Support", bullets=[
    "Ask for error messages",
    "Check service status before troubleshooting"
])
```

#### Dynamic Content

You can also make your prompts dynamic by adding content at runtime:

```python
# Add to an existing section 
self.prompt_add_to_section("Context", 
                          bullet="Current system status: All services operational")

# Add or update sections based on runtime conditions
if high_call_volume:
    self.prompt_add_section("Priority", 
                          body="Focus on quick resolution due to high call volume")
else:
    self.prompt_add_section("Priority", 
                          body="Take time to fully address customer needs")
```

#### Structured Prompt Building

The SDK uses `prompt_add_section()` for all prompt configuration:

```python
# Use prompt_add_section for all prompt configuration
self.prompt_add_section("Personality", body="You are a technical support specialist.")
self.prompt_add_section("Goal", body="Resolve customer technical issues efficiently.")
self.prompt_add_section("Instructions", bullets=[
    "Get specific error details",
    "Guide users through troubleshooting steps",
    "Document the resolution"
])
```

### Adding Multilingual Support

One of the powerful features of the SignalWire AI Agents SDK is built-in multilingual support. This allows your agent to communicate with users in their preferred language, providing a more inclusive and personalized experience.

#### Configuring Multiple Languages

Here's how to add support for multiple languages:

```python
# Add English as the primary language
self.add_language(
    name="English",
    code="en-US",
    voice="elevenlabs.josh",  # Using ElevenLabs voice
    speech_fillers=["Let me think about that...", "One moment please..."],
    function_fillers=["I'm looking that up for you...", "Let me check that..."]
)

# Add Spanish with appropriate fillers
self.add_language(
    name="Spanish",
    code="es",
    voice="elevenlabs.antonio:eleven_multilingual_v2",
    speech_fillers=["Un momento por favor...", "Estoy pensando..."],
    function_fillers=["Estoy buscando esa información...", "Déjame verificar..."]
)

# Add French with specific engine and model parameters
self.add_language(
    name="French",
    code="fr-FR",
    voice="rime.alois",
    speech_fillers=["Un instant s'il vous plaît...", "Laissez-moi réfléchir..."]
)

# Enable language detection and switching
self.set_params({"languages_enabled": True})
```

#### Voice Specification Formats

There are several formats for specifying voices:

```python
# Simple format
voice="en-US-Neural2-F"

# Provider-prefixed format
voice="elevenlabs.josh"

# Provider with model
voice="elevenlabs.antonio:eleven_multilingual_v2"

# Explicit parameters
self.add_language(
    name="German",
    code="de-DE",
    voice="hans",
    engine="rime",
    model="arcana"
)
```

#### Language Detection and Switching

With multiple languages configured and `languages_enabled` set to `True`, the agent will automatically detect the user's language based on their speech and respond in the same language using the appropriate voice and fillers.

### Configuring Pronunciation and Hints

For voices to sound natural, especially with technical terms, acronyms, or brand names, you can configure pronunciation rules and hints.

#### Adding Pronunciation Rules

Pronunciation rules map specific terms to the way they should be pronounced:

```python
# Spell out acronyms
self.add_pronunciation("API", "A P I", ignore_case=False)
self.add_pronunciation("SDK", "S D K", ignore_case=True)

# Correct pronunciation of brand names
self.add_pronunciation("SignalWire", "Signal Wire", ignore_case=False)
self.add_pronunciation("PostgreSQL", "Postgres Q L", ignore_case=True)
```

#### Pattern Hints

For more complex pattern matching, use pattern hints with regular expressions:

```python
# Help with AI terms using regex patterns
self.add_pattern_hint(
    hint="AI Agent",
    pattern="AI\\s+Agent",
    replace="A.I. Agent",
    ignore_case=True
)

# Format version numbers appropriately
self.add_pattern_hint(
    hint="version numbers",
    pattern="v(\\d+)\\.(\\d+)",
    replace="version $1 point $2",
    ignore_case=True
)
```

#### Simple Hints

For general terminology that should be recognized but doesn't need special pronunciation:

```python
# Add product and technical terms as hints
self.add_hints([
    "SignalWire",
    "SWML",
    "SWAIG",
    "WebRTC",
    "SIP"
])
```

These pronunciation rules and hints help the AI understand and pronounce technical terms correctly, creating a more professional and natural-sounding experience for users.

### Setting AI Behavior Parameters

The SDK provides extensive parameters to control how the AI agent behaves during conversations. These allow fine-tuning of the interaction style, timing, and other aspects of the conversation.

```python
# Configure AI behavior parameters
self.set_params({
    # Conversation flow
    "wait_for_user": False,          # Start speaking immediately vs. waiting for user
    "end_of_speech_timeout": 1000,   # Milliseconds of silence to detect end of speech
    "energy_level": 50,              # Sensitivity for detecting speech (0-100)
    
    # Voice and audio settings
    "ai_volume": 5,                  # Voice volume level
    "background_file": "https://example.com/hold-music.mp3",  # Background audio
    "background_file_volume": -10,   # Volume for background audio
    
    # Interruption handling
    "transparent_barge": True,       # Allow users to interrupt the AI
    "barge_min_words": 3,            # Min words needed to trigger interruption
    
    # Timeout handling
    "attention_timeout": 30000,      # Milliseconds before prompting unresponsive user
    "inactivity_timeout": 300000,    # Milliseconds of inactivity before ending session
    
    # Language settings
    "languages_enabled": True,       # Enable multilingual support
    "local_tz": "America/New_York",  # Default timezone for time functions
    
    # Advanced features
    "enable_vision": False,          # Enable visual processing capabilities
    "verbose_logs": True             # Enable detailed logging
})
```

You can also update parameters individually:

```python
# Update a single parameter
self.set_param("ai_volume", 7)
```

These parameters allow you to create conversational experiences tailored to specific use cases, from quick information retrieval to in-depth consultations.

### Handling State and Context

The SignalWire AI Agents SDK follows a **stateless-first design philosophy**. By default, agents do not maintain persistent state, making them highly scalable and suitable for microservice deployments. However, when state management is needed, the platform provides built-in mechanisms through SWAIG.

#### Understanding Stateless Design

The stateless-first approach means:

```python
from signalwire_agents import AgentBase

# This agent is stateless by default
agent = Agent(
    assistant=Assistant(
        name="Stateless Assistant",
        purpose="I help users without maintaining persistent state."
    )
)

# Each conversation is independent
# No state is carried between conversations
# Perfect for high-scale, concurrent usage
```

**Benefits of Stateless Design**:
- **High Scalability**: Handle thousands of concurrent conversations
- **Zero State Conflicts**: No risk of state corruption between conversations
- **Simple Deployment**: No state databases or persistent storage required
- **Fault Tolerance**: Restart agents without losing critical state

#### Using Global Data for AI Context

When you need the AI to remember information during a conversation, use `global_data`:

```python
@agent.swaig_function
def save_user_preference(args):
    preference_type = args.get("preference_type")
    preference_value = args.get("preference_value")
    
    # Get current global data (available to AI)
    current_data = agent.get_global_data()
    
    # Update with user preference
    current_data["user_preferences"] = current_data.get("user_preferences", {})
    current_data["user_preferences"][preference_type] = preference_value
    
    # Set updated global data - now available to AI in conversation
    agent.set_global_data(current_data)
    
    return SwaigFunctionResult(f"I've saved your {preference_type} preference as {preference_value}.")
```

The AI can now reference this information:
```python
# The AI will now know about user preferences in its responses
# Global data is automatically included in the AI's context
```

#### Using Meta Data for Function Context

For tracking metadata about function calls and results:

```python
@agent.swaig_function
def create_support_ticket(args):
    issue_description = args.get("issue_description")
    priority = args.get("priority", "medium")
    
    # Create ticket in your system
    ticket_id = create_ticket_in_system(issue_description, priority)
    
    # Return result with metadata for future reference
    return SwaigFunctionResult(
        f"I've created support ticket #{ticket_id} for your issue.",
        meta_data={
            "ticket_id": ticket_id,
            "priority": priority,
            "created_at": datetime.now().isoformat(),
            "issue_type": "general_support"
        }
    )
```

#### Multi-Turn Conversation Example

Here's how to handle multi-step workflows using global_data:

```python
@agent.swaig_function
def start_order_process(args):
    # Initialize order in global data
    agent.set_global_data({
        "current_order": {
        "status": "collecting_items",
        "items": [],
            "total": 0.0,
            "customer_info": None
        }
    })
    
    return SwaigFunctionResult("I've started a new order for you. What would you like to add?")

@agent.swaig_function
def add_item_to_order(args):
    item_name = args.get("item_name")
    quantity = args.get("quantity", 1)
    price = args.get("price")
    
    # Get current order from global data
    global_data = agent.get_global_data()
    current_order = global_data.get("current_order", {})
    
    # Add item to order
    current_order["items"].append({
        "name": item_name,
        "quantity": quantity,
        "price": price
    })
    current_order["total"] += price * quantity
    
    # Update global data
    global_data["current_order"] = current_order
    agent.set_global_data(global_data)
    
    return SwaigFunctionResult(
        f"Added {quantity}x {item_name} to your order. Current total: ${current_order['total']:.2f}"
    )

@agent.swaig_function
def complete_order(args):
    global_data = agent.get_global_data()
    order = global_data.get("current_order", {})
    
    if not order.get("items"):
        return SwaigFunctionResult("No items in your order. Please add items first.")
    
    # Process the order
    order_id = process_order_in_system(order)
    
    # Clear the order from global data
    global_data.pop("current_order", None)
    agent.set_global_data(global_data)
    
    return SwaigFunctionResult(
        f"Order #{order_id} has been placed successfully! Total: ${order['total']:.2f}",
        meta_data={
            "order_id": order_id,
            "total_amount": order["total"],
            "item_count": len(order["items"])
        }
    )
```

#### When to Choose State Management

**Remain Stateless When**:
- Building simple Q&A or information agents
- Handling independent, single-turn interactions
- Requiring maximum scalability and simplicity
- Deploying in serverless or auto-scaling environments

**Use State Management When**:
- Building multi-step workflows (ordering, booking, forms)
- Personalizing responses based on user history
- Tracking conversation context across multiple interactions
- Building complex business process agents

#### Microservice State Management

For advanced applications requiring external state management:

```python
@agent.swaig_function
def get_user_profile(args):
    user_id = args.get("user_id")
    
    # Query external database/service
    user_profile = external_database.get_user(user_id)
    
    # Store relevant info in global_data for AI access
    agent.set_global_data({
        "current_user": {
            "name": user_profile["name"],
            "tier": user_profile["tier"],
            "preferences": user_profile["preferences"]
        }
    })
    
    return SwaigFunctionResult(f"Hello {user_profile['name']}! I've loaded your profile.")
```

This approach combines the benefits of stateless design with the flexibility of external state management, allowing agents to scale while maintaining rich context when needed.

## 9. SWAIG Functions: When to Use Custom Functions

SWAIG (SignalWire AI Gateway) functions are one of the most powerful features of the SignalWire AI Agents SDK. They allow your AI agent to go beyond simple conversation by providing the ability to execute code, access external systems, and perform actions on behalf of the user. In this section, we'll dive deeper into how to define, implement, and leverage SWAIG functions effectively.

### Understanding SWAIG Functions

SWAIG functions are callable methods that the AI can invoke during conversations to accomplish specific tasks. They serve as the bridge between the conversational interface and your business logic or external systems. Unlike traditional webhook systems, SWAIG functions are seamlessly integrated into the conversation flow, allowing the AI to:

1. **Determine When to Call**: The AI decides when to invoke functions based on user requests
2. **Extract Parameters**: The AI identifies and extracts relevant parameters from the conversation
3. **Use Results**: The AI incorporates function results back into the conversation naturally

This creates a fluid experience where users don't need to follow strict command formats - they can simply express their needs conversationally, and the AI will handle the details of function invocation.

Each SWAIG function consists of:
- A **name** that the AI uses to reference the function
- A **description** that helps the AI understand when to use the function
- A **parameters schema** that defines what information the function needs
- An **implementation** that executes when the function is called
- Optional **security settings** that control access to the function

### Skills vs DataMap vs Custom Functions

When deciding when to use SWAIG functions, consider the following:

- **Skills**: Use skills when you want the AI to perform a specific task or action without needing external data.
- **DataMap**: Use DataMap when you need to call an external API to retrieve data.
- **Custom Functions**: Use custom functions when you need to perform a task that doesn't fit into the skills or DataMap categories.

### Parameter Definition

A key aspect of SWAIG functions is their well-defined parameter schema. This schema helps the AI understand what information to extract from the conversation and how to format it when calling the function. The schema uses a JSON Schema-inspired format:

```python
@AgentBase.tool(
    name="book_appointment",
    description="Book an appointment for a customer",
    parameters={
        "customer_name": {
            "type": "string",
            "description": "The full name of the customer"
        },
        "service_type": {
            "type": "string",
            "description": "The type of service requested",
            "enum": ["haircut", "coloring", "styling", "consultation"]
        },
        "preferred_date": {
            "type": "string",
            "description": "The preferred date for the appointment (YYYY-MM-DD)"
        },
        "preferred_time": {
            "type": "string",
            "description": "The preferred time for the appointment (HH:MM)",
            "pattern": "^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$"
        },
        "special_requests": {
            "type": "string",
            "description": "Any special requests or notes",
            "nullable": True
        }
    }
)
```

This schema format supports:

- **Basic Types**: string, number, integer, boolean, array, object
- **Validation Rules**: enum (allowed values), pattern (regex), minimum/maximum (for numbers)
- **Optional Parameters**: Using the `nullable` property
- **Nested Objects**: Using the object type with properties
- **Arrays**: Using the array type with items

These schemas serve multiple purposes:
1. **AI Guidance**: Help the AI understand what information to collect
2. **Validation**: Ensure the parameters meet expected formats
3. **Documentation**: Self-document the function's requirements

When designing parameter schemas, aim for clarity and specific descriptions to help the AI correctly extract the right information from user conversations.

### Function Implementation

The implementation of a SWAIG function is a Python method that receives the parameters extracted by the AI and performs the necessary actions. Here's a comprehensive example:

```python
@AgentBase.tool(
    name="check_order_status",
    description="Check the status of a customer's order",
    parameters={
        "order_id": {
            "type": "string",
            "description": "The order ID to check"
        }
    }
)
def check_order_status(self, args, raw_data):
    """
    Check the status of an order in our system.
    
    Args:
        args: Dictionary containing the parsed parameters (order_id)
        raw_data: Complete request data including call_id and other metadata
    
    Returns:
        SwaigFunctionResult with the order status information
    """
    # Extract the order ID from args
    order_id = args.get("order_id")
    
    # Log the function call
    self.log.info("check_order_status_called", order_id=order_id)
    
    try:
        # In a real implementation, you would query your order system
        # For this example, we'll simulate different statuses
        
        # Simulate a database query
        # order_data = order_database.get_order(order_id)
        
        # For demonstration, generate a status based on the order ID
        status_options = ["processing", "shipped", "delivered", "pending"]
        simulated_status = status_options[hash(order_id) % len(status_options)]
        
        # Create estimated delivery date (for demonstration)
        from datetime import datetime, timedelta
        estimated_delivery = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
        
        # Format response based on status
        if simulated_status == "processing":
            response = f"Order {order_id} is currently being processed. Estimated shipping in 1-2 business days."
        elif simulated_status == "shipped":
            response = f"Order {order_id} has been shipped and is on its way. Expected delivery by {estimated_delivery}."
        elif simulated_status == "delivered":
            response = f"Order {order_id} has been delivered. It was completed on {estimated_delivery}."
        else:  # pending
            response = f"Order {order_id} is pending payment confirmation. Please allow 24 hours for processing."
        
        # Create the result
        result = SwaigFunctionResult(response)
        
        # Add the order status to global data for future reference
        result.add_action("set_global_data", {
            "last_checked_order": {
                "id": order_id,
                "status": simulated_status,
                "estimated_delivery": estimated_delivery
            }
        })
        
        return result
        
    except Exception as e:
        # Handle errors gracefully
        self.log.error("order_status_error", error=str(e), order_id=order_id)
        return SwaigFunctionResult(
            f"I'm sorry, I couldn't retrieve information for order {order_id}. " +
            "Please verify the order number or try again later."
        )
```

This implementation demonstrates several best practices:

1. **Comprehensive Docstring**: Clearly documents what the function does
2. **Parameter Extraction**: Gets parameters from the `args` dictionary
3. **Logging**: Records function invocation and any errors
4. **Error Handling**: Gracefully handles exceptions with user-friendly messages
5. **Structured Response**: Returns a well-formatted response with relevant information
6. **Global Data Updates**: Stores information that might be needed later

### Returning Results with Actions

The `SwaigFunctionResult` class provides a flexible way to return information and trigger actions. Beyond simple text responses, you can include various actions that affect the conversation flow or system state:

```python
@AgentBase.tool(
    name="process_payment",
    description="Process a payment for an order",
    parameters={
        "amount": {
            "type": "number",
            "description": "Payment amount"
        },
        "payment_method": {
            "type": "string",
            "description": "Payment method (credit_card, bank_transfer)",
            "enum": ["credit_card", "bank_transfer"]
        }
    }
)
def process_payment(self, args, raw_data):
    amount = args.get("amount")
    payment_method = args.get("payment_method")
    
    # Simulate payment processing
    transaction_id = f"TXN-{hash(str(amount) + payment_method) % 10000:04d}"
    
    # Create the result with an initial response
    result = SwaigFunctionResult(
        f"I've processed your {payment_method} payment for ${amount:.2f}. " +
        f"Your transaction ID is {transaction_id}."
    )
    
    # Add a single action
    result.add_action("log", {
        "message": f"Payment processed: ${amount:.2f} via {payment_method}"
    })
    
    # Add multiple actions at once
    result.add_actions([
        # Update global data with payment info
        {"set_global_data": {
            "last_transaction": {
                "id": transaction_id,
                "amount": amount,
                "method": payment_method,
                "status": "completed"
            }
        }},
        # Play a success sound in the background
        {"playback_bg": {
            "file": "https://example.com/sounds/payment_success.mp3",
            "wait": False
        }}
    ])
    
    # Chain actions for readability (alternative to add_actions)
    result.add_action("say", "Your receipt will be emailed to you shortly.")
    
    return result
```

Common action types include:

- **set_global_data/unset_global_data**: Update information available to the AI
- **log**: Record information for debugging or auditing
- **say**: Have the AI speak additional text
- **playback_bg**: Play background sounds or music
- **context_switch**: Change the AI's contextual understanding
- **stop_playback_bg**: Stop background audio
- **user_input**: Inject text as if the user had said it

These actions allow for rich, interactive experiences that go beyond simple text responses.

### Best Practices for SWAIG Functions

When designing and implementing SWAIG functions, keep these best practices in mind:

1. **Single Responsibility**: Each function should do one thing well
2. **Clear Descriptions**: Write detailed descriptions that help the AI understand when to use the function
3. **Specific Parameters**: Define parameters with clear names and descriptions
4. **Error Handling**: Gracefully handle exceptions with user-friendly messages
5. **Meaningful Responses**: Return results that the AI can easily incorporate into the conversation
6. **Action Combinations**: Use multiple actions when appropriate to create rich experiences
7. **Logging**: Include logging for debugging and analytics
8. **Performance**: Keep functions fast when possible, or use fillers for longer operations

By following these practices, you can create SWAIG functions that seamlessly integrate with the conversation flow, allowing your agent to provide helpful, dynamic responses to user requests.

### Native Functions and External Integrations

In addition to custom functions that you implement, the SDK supports native functions (built into SignalWire's platform) and external function integrations:

#### Native Functions

Native functions are pre-built capabilities provided by the SignalWire platform:

```python
# Register native functions
self.set_native_functions([
    "check_time",       # Get the current time in various formats/timezones
    "wait_seconds"      # Pause the conversation for a specified duration
])
```

These functions are immediately available to the AI without requiring any implementation code.

#### External Function Includes

You can also include functions hosted at external URLs:

```python
# Include functions from an external API
self.add_function_include(
    url="https://api.example.com/ai-functions",
    functions=[
        "search_knowledge_base",
        "get_product_details",
        "calculate_shipping"
    ],
    meta_data={
        "api_key": "your-api-key",
        "organization_id": "org123"
    }
)

# Include another set of functions from a different API
self.add_function_include(
    url="https://analytics.example.org/functions",
    functions=["get_user_insights"]
)
```

This feature allows you to:
1. **Split Functionality**: Distribute functions across multiple services
2. **Reuse Functions**: Share functions between different agents
3. **Third-Party Integration**: Leverage external AI function marketplaces

External functions follow the same invocation pattern from the AI's perspective, but the execution happens at the remote endpoint rather than within your agent's code.

### Function Security and Controls

SWAIG functions often provide access to sensitive operations or data, so the SDK includes several security features:

#### Secure Functions

By default, all SWAIG functions are marked as secure, meaning they require proper authentication:

```python
@AgentBase.tool(
    name="get_account_details",
    description="Get customer account details",
    parameters={"account_id": {"type": "string"}},
    secure=True  # This is the default
)
```

Secure functions use token-based authentication for each call, with tokens that are:
- **Scoped to specific functions**: A token for one function can't be used for another
- **Tied to specific call IDs**: Prevents cross-session usage
- **Time-limited**: Expire after a configurable duration

#### Function Fillers

To improve the user experience during function execution, you can configure "fillers" - phrases that the AI will say while waiting for a function to complete:

```python
@AgentBase.tool(
    name="search_database",
    description="Search the customer database",
    parameters={"query": {"type": "string"}},
    fillers={
        "en-US": [
            "Let me search our records...",
            "I'm looking that up for you...",
            "Checking our database..."
        ],
        "es": [
            "Déjame buscar en nuestros registros...",
            "Estoy buscando esa información..."
        ]
    }
)
```

These fillers make the conversation feel more natural during delays, especially for operations that might take several seconds to complete.

## 10. Prefab Agents: Ready-to-Use Solutions

The SignalWire AI Agents SDK includes several pre-built "prefab" agents that provide ready-to-use implementations for common use cases. These prefabs can significantly accelerate your development process, allowing you to deploy fully functional agents with minimal code. In this section, we'll explore the available prefab agents, how to configure them, and strategies for extending them to meet your specific needs.

### InfoGathererAgent for Structured Data Collection

The InfoGathererAgent is designed to systematically collect information from users through a series of predefined questions. This makes it ideal for intake forms, registration processes, surveys, and any situation where you need to gather specific information in a structured format.

#### Configuration

Setting up an InfoGathererAgent is straightforward:

```python
from signalwire_agents.prefabs import InfoGathererAgent

# Create an agent that collects user registration information
registration_agent = InfoGathererAgent(
    name="registration-form",
    route="/register",
    questions=[
        {
            "key_name": "full_name",
            "question_text": "What is your full name?"
        },
        {
            "key_name": "email",
            "question_text": "What is your email address?",
            "confirm": True  # This will prompt the AI to confirm the answer
        },
        {
            "key_name": "birth_date",
            "question_text": "What is your date of birth? Please provide it in MM/DD/YYYY format."
        },
        {
            "key_name": "reason",
            "question_text": "What brings you to our service today?"
        }
    ]
)
```

The agent manages the question flow automatically, tracking which questions have been asked and storing the answers in its state. The `confirm` option for questions allows you to specify which questions require confirmation due to their importance or potential for misunderstanding.

#### How It Works

The InfoGathererAgent implements a state-based question flow with two main SWAIG functions:

1. **start_questions**: Begins the question sequence with the first question
2. **submit_answer**: Records an answer to the current question and advances to the next one

Here's a simplified view of the flow:

1. When the conversation begins, the agent introduces itself and asks if the user is ready to begin
2. Once the user confirms, the AI calls `start_questions` to begin the process
3. The AI asks the first question from the configured list
4. When the user responds, the AI extracts the answer and calls `submit_answer`
5. The answer is stored, and the agent provides the next question
6. This process continues until all questions have been answered
7. When complete, the agent confirms that all information has been collected

#### Extending InfoGathererAgent

While the base implementation is sufficient for many use cases, you can extend it to add custom validation, processing, or follow-up actions:

```python
class CustomInfoGatherer(InfoGathererAgent):
    def __init__(self):
        super().__init__(
            name="customer-intake",
            route="/intake",
            questions=[
                {"key_name": "name", "question_text": "What is your name?"},
                {"key_name": "issue", "question_text": "Please describe your issue briefly."}
            ]
        )
        # Add a custom prompt section for more personalized interactions
        self.prompt_add_section("Introduction", 
                               body="Greet the user warmly and explain that you'll need to collect some information to help them.")
    
    # Override to add custom processing after all questions are answered
    def on_all_questions_completed(self, call_id, answers):
        # Process the collected information
        print(f"All questions completed for call {call_id}")
        print(f"Answers: {answers}")
        
        # You could add automatic routing, ticket creation, etc. here
        
        # Return a custom message
        return "Thank you for providing that information. I've created a support ticket for you, and a specialist will contact you shortly."
    
    # Add a custom function for after information is collected
    @AgentBase.tool(
        name="schedule_follow_up",
        description="Schedule a follow-up call",
        parameters={
            "preferred_time": {"type": "string", "description": "Preferred time for follow-up"}
        }
    )
    def schedule_follow_up(self, args, raw_data):
        # Implementation details...
        return SwaigFunctionResult("Your follow-up has been scheduled.")
```

This extensibility allows you to maintain the core question flow functionality while adding your own business logic and custom features.

### FAQBotAgent for Knowledge Base Assistance

The FAQBotAgent is designed to answer questions using a provided knowledge base. It's ideal for creating support bots, information assistants, and any agent that needs to provide factual answers based on specific content.

#### Configuration

To set up a FAQBotAgent, you provide it with FAQ content in one of several formats:

```python
from signalwire_agents.prefabs import FAQBotAgent

# Create an agent with direct FAQ content
support_agent = FAQBotAgent(
    name="product-support",
    route="/support",
    faq_content=[
        {
            "question": "How do I reset my password?",
            "answer": "To reset your password, visit the login page and click on 'Forgot Password'. Follow the instructions sent to your email."
        },
        {
            "question": "What payment methods do you accept?",
            "answer": "We accept Visa, Mastercard, American Express, and PayPal."
        },
        {
            "question": "How long does shipping take?",
            "answer": "Standard shipping typically takes 3-5 business days within the continental US."
        }
    ]
)

# Alternatively, load from a file
documentation_agent = FAQBotAgent(
    name="api-docs",
    route="/docs",
    faq_file="documentation.json"  # JSON file with question/answer pairs
)

# Or load from a URL
knowledge_agent = FAQBotAgent(
    name="knowledge-base",
    route="/kb",
    faq_url="https://example.com/api/knowledge-base",
    refresh_interval=3600  # Refresh content every hour
)
```

The agent processes this content and makes it available to the AI as context for answering questions.

#### How It Works

The FAQBotAgent works by:

1. Loading and processing the provided FAQ content
2. Including this content in the AI's prompt as reference material
3. Configuring the agent with a personality and goal focused on providing helpful, accurate answers
4. Adding fallback mechanisms for questions outside the knowledge domain

The agent doesn't typically need custom SWAIG functions since its primary purpose is to answer questions directly using the AI's reasoning capabilities and the provided content. However, it can be extended with additional functions for more complex scenarios.

#### Advanced Configuration

For more control over the FAQBotAgent's behavior:

```python
# Create a more customized FAQ bot
advanced_faq_agent = FAQBotAgent(
    name="technical-support",
    route="/tech-support",
    faq_content=tech_support_faqs,
    
    # Customize how to handle out-of-scope questions
    unknown_question_response="I specialize in technical support for our product line. That question seems outside my area of expertise. Would you like me to connect you with general customer service?",
    
    # Configure escalation options
    escalation_prompt="I can connect you with a human specialist if you'd prefer. Would you like me to transfer you now?",
    
    # Add additional context
    system_information={
        "product_versions": ["v1.2.3", "v1.3.0", "v2.0.1"],
        "known_issues": ["Bluetooth connectivity on v1.2.3", "Sleep mode on v2.0.1"]
    }
)
```

These configuration options allow you to adjust how the agent responds to different scenarios while maintaining its core FAQ functionality.

### ConciergeAgent for Intelligent Routing

The ConciergeAgent acts as a front-line receptionist or dispatcher, helping to understand user needs and route them to the appropriate specialized agent or department. This is particularly useful for creating entry points to multi-agent systems.

#### Configuration

Setting up a basic ConciergeAgent involves defining the available routes:

```python
from signalwire_agents.prefabs import ConciergeAgent

# Create a concierge agent for a customer service system
concierge = ConciergeAgent(
    name="customer-service-concierge",
    route="/concierge",
    routing_options=[
        {
            "name": "Technical Support",
            "description": "Help with product functionality, error messages, and troubleshooting",
            "route": "/tech-support"
        },
        {
            "name": "Billing Department",
            "description": "Questions about invoices, payments, refunds, and account status",
            "route": "/billing"
        },
        {
            "name": "Sales Team",
            "description": "Information about products, pricing, and placing new orders",
            "route": "/sales"
        }
    ]
)
```

The agent uses these routing options to determine where to direct users based on their needs.

#### How It Works

The ConciergeAgent operates through a simple flow:

1. The agent greets the user and asks how it can help
2. Based on the user's response, the agent identifies which routing option best fits their needs
3. The agent confirms the routing choice with the user
4. Once confirmed, the agent transfers the call to the appropriate specialized agent

This is implemented through a key SWAIG function:

```python
@AgentBase.tool(
    name="route_call",
    description="Route the call to the appropriate department",
    parameters={
        "destination": {
            "type": "string",
            "description": "The department to route to",
            "enum": ["Technical Support", "Billing Department", "Sales Team"]
        },
        "reason": {
            "type": "string",
            "description": "Brief reason for the routing choice"
        }
    }
)
```

#### Advanced Routing

For more sophisticated routing needs, you can extend the ConciergeAgent with additional context and logic:

```python
class EnhancedConcierge(ConciergeAgent):
    def __init__(self):
        super().__init__(
            name="enhanced-concierge",
            route="/enhanced-concierge",
            routing_options=[...]
        )
        
        # Add business hours information
        self.set_global_data({
            "business_hours": {
                "Technical Support": "24/7",
                "Billing Department": "Monday-Friday, 9am-5pm EST",
                "Sales Team": "Monday-Saturday, 8am-8pm EST"
            },
            "current_wait_times": {
                "Technical Support": "Approximately 5 minutes",
                "Billing Department": "Approximately 3 minutes",
                "Sales Team": "No wait time"
            }
        })
        
        # Enhance the routing instructions
        self.prompt_add_section("Routing Guidelines", bullets=[
            "Consider the urgency of the customer's issue",
            "Check if the appropriate department is currently available",
            "Inform the customer of expected wait times before routing",
            "If multiple departments could help, ask the customer for their preference"
        ])
    
    # Customize the pre-routing confirmation
    def pre_route_confirmation(self, destination, reason):
        # This would be called before finalizing the routing
        department = destination
        wait_time = self.get_global_data().get("current_wait_times", {}).get(department, "Unknown")
        
        return (
            f"I'll connect you with our {department} team regarding your {reason}. "
            f"The current estimated wait time is {wait_time}. "
            f"Would you like to proceed?"
        )
```

This enhanced version provides more context to both the AI and the user, creating a more informative and helpful routing experience.

### SurveyAgent for Feedback Collection

The SurveyAgent specializes in collecting structured feedback through surveys. It's ideal for customer satisfaction measurement, post-interaction feedback, and other survey-based data collection.

#### Configuration

Setting up a SurveyAgent involves defining the survey questions and rating scales:

```python
from signalwire_agents.prefabs import SurveyAgent

# Create a customer satisfaction survey agent
csat_survey = SurveyAgent(
    name="satisfaction-survey",
    route="/survey",
    introduction="We'd like to get your feedback on your recent experience with our customer service team.",
    questions=[
        {
            "type": "rating",
            "question": "How would you rate your overall experience?",
            "scale": 1,  # 1-5 scale
            "key": "overall_satisfaction"
        },
        {
            "type": "rating",
            "question": "How likely are you to recommend our service to others?",
            "scale": 2,  # 1-10 scale (NPS style)
            "key": "recommendation_score"
        },
        {
            "type": "text",
            "question": "What could we have done better?",
            "key": "improvement_feedback"
        }
    ],
    conclusion="Thank you for your feedback! We appreciate your input and will use it to improve our services."
)
```

The agent guides users through each question, collecting and storing their responses.

#### Survey Types and Formats

The SurveyAgent supports various question types and formats:

- **Rating Questions**: Numerical ratings on different scales (1-5, 1-10, etc.)
- **Multiple Choice**: Selection from a predefined list of options
- **Yes/No Questions**: Simple binary responses
- **Text Feedback**: Open-ended questions for qualitative feedback
- **Conditional Questions**: Questions that appear based on previous answers

This flexibility allows for creating sophisticated surveys that can gather both quantitative and qualitative data.

#### Custom Processing

You can extend the SurveyAgent to add custom processing of survey results:

```python
class EnhancedSurveyAgent(SurveyAgent):
    def __init__(self):
        super().__init__(
            name="enhanced-survey",
            route="/enhanced-survey",
            questions=[...]
        )
    
    # Override to add custom processing when survey completes
    def on_survey_completed(self, call_id, results):
        # Process the survey results
        print(f"Survey completed for call {call_id}")
        print(f"Results: {results}")
        
        # Calculate satisfaction metrics
        nps_score = results.get("recommendation_score", 0)
        overall_score = results.get("overall_satisfaction", 0)
        
        # Store in database, send alerts, trigger workflows, etc.
        self.store_results_in_database(call_id, results)
        
        if overall_score <= 2:  # Low satisfaction
            self.trigger_follow_up(call_id, results)
        
        # Return a customized conclusion based on the results
        if overall_score >= 4:
            return "Thank you for your positive feedback! We're glad you had a great experience."
        else:
            return "Thank you for your candid feedback. We apologize for any shortcomings and will work to improve your experience in the future."
```

This allows the survey to not just collect data but also to take immediate action based on the responses.

### Creating Custom Prefabs

While the SDK provides several useful prefab agents, you may want to create your own reusable agent templates for specific use cases. Here's how to create your own prefab:

```python
from signalwire_agents import AgentBase

class AppointmentSchedulerAgent(AgentBase):
    """
    A prefab agent for scheduling appointments.
    """
    
    def __init__(
        self,
        name="appointment-scheduler",
        route="/appointments",
        available_services=None,
        available_times=None,
        confirmation_required=True,
        **kwargs
    ):
        """
        Initialize an appointment scheduler agent.
        
        Args:
            name: Agent name
            route: HTTP route
            available_services: List of services that can be scheduled
            available_times: Dict of available time slots by date
            confirmation_required: Whether to require confirmation before finalizing
            **kwargs: Additional arguments for AgentBase
        """
        # Initialize the base agent
        super().__init__(
            name=name,
            route=route,
            use_pom=True,
            **kwargs
        )
        
        # Store configuration
        self.available_services = available_services or ["Consultation", "Follow-up"]
        self.available_times = available_times or self._default_availability()
        self.confirmation_required = confirmation_required
        
        # Set up global data
        self.set_global_data({
            "available_services": self.available_services,
            "available_times": self.available_times
        })
        
        # Build prompts
        self._build_prompt()
        
        # Register functions
        self._register_functions()
    
    def _default_availability(self):
        """Generate default availability if none provided"""
        from datetime import datetime, timedelta
        
        availability = {}
        start_date = datetime.now().date()
        
        # Generate availability for the next 7 days
        for i in range(7):
            date = start_date + timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            
            # 9am-5pm, hourly slots
            times = []
            for hour in range(9, 17):
                times.append(f"{hour:02d}:00")
            
            availability[date_str] = times
        
        return availability
    
    def _build_prompt(self):
        """Build the agent's prompt"""
        self.prompt_add_section(
            "Personality",
            body="You are a helpful appointment scheduling assistant."
        )
        
        self.prompt_add_section(
            "Goal",
            body="Help users schedule appointments efficiently by gathering the necessary information and finding a suitable time slot."
        )
        
        self.prompt_add_section(
            "Instructions",
            bullets=[
                "Ask for the type of service they need",
                "Find out their preferred date and time",
                "Suggest alternative times if their preference is unavailable",
                "Collect contact information",
                "Confirm all details before finalizing"
            ]
        )
    
    def _register_functions(self):
        """Register SWAIG functions"""
        # Functions would be defined here
        pass
    
    # SWAIG functions would be defined here
    @AgentBase.tool(
        name="check_availability",
        description="Check availability for a specific date",
        parameters={
            "date": {"type": "string", "description": "Date to check (YYYY-MM-DD)"}
        }
    )
    def check_availability(self, args, raw_data):
        date = args.get("date")
        available_times = self.available_times.get(date, [])
        
        if not available_times:
            return SwaigFunctionResult(f"I'm sorry, there are no available times on {date}.")
        
        result = SwaigFunctionResult(
            f"The following times are available on {date}: {', '.join(available_times)}"
        )
        
        return result
    
    # Additional functions for booking, rescheduling, etc.
```

Once defined, this custom prefab can be used just like the built-in ones:

```python
# Create an instance with custom configuration
scheduler = AppointmentSchedulerAgent(
    name="medical-appointments",
    route="/medical-appointments",
    available_services=[
        "Initial Consultation",
        "Follow-up",
        "Annual Physical",
        "Vaccination"
    ],
    custom_availability=medical_office_hours  # Custom availability data
)

# Or with defaults
simple_scheduler = AppointmentSchedulerAgent()
```

Custom prefabs allow you to encapsulate complex behavior in reusable components, making it easier to create consistent agents across your organization.

By leveraging these prefab agents, you can quickly deploy sophisticated AI voice agents without having to build everything from scratch. Whether you use them as-is or extend them with custom functionality, prefabs provide a solid foundation for many common use cases.

## 11. Best Practices and Patterns

Building effective AI voice agents requires understanding not just the technical capabilities of the SignalWire SDK, but also the architectural patterns and design principles that lead to maintainable, scalable, and user-friendly applications. Through working with hundreds of production deployments, several key patterns have emerged that separate successful implementations from those that struggle in real-world usage.

### When to Use Skills vs DataMap vs Custom Functions

One of the most common questions developers face is choosing the right tool for each capability. The SignalWire SDK provides three primary approaches for extending agent functionality, and understanding when to use each one can dramatically impact both development speed and long-term maintainability.

**Skills are ideal when you need proven, battle-tested functionality** that doesn't require custom business logic. The web search skill, for example, handles not just the API call to search engines, but also result filtering, content extraction, rate limiting, error handling, and response formatting. Building this functionality from scratch would take weeks and likely introduce edge cases that the built-in skill already handles.

Consider a customer service agent that needs to look up current product pricing. You could implement this as a custom function that queries your pricing database, but if the pricing information is also available through your public API, using the web search skill might be simpler: `agent.add_skill("web_search", {"domain_filter": "yourcompany.com", "search_type": "product_pricing"})`. The skill handles caching, retries, and error cases automatically.

The datetime skill is particularly valuable because timezone handling, business hours calculations, and date arithmetic are surprisingly complex. A simple question like "When is our next business day?" involves weekend detection, holiday handling, and timezone conversions that can easily introduce bugs in custom implementations.

**DataMap tools excel when you need to integrate with existing APIs** without building webhook infrastructure. The key insight is that DataMap tools run on SignalWire's servers, which means you can integrate with internal APIs without exposing them to the internet or managing authentication tokens in your agent code.

A real-world example is order status checking. Your e-commerce platform likely has an API endpoint like `GET /api/orders/{order_id}` that returns order details. With a DataMap tool, you can integrate this directly:

```python
order_status_tool = (DataMap('check_order_status')
    .parameter('order_id', 'string', 'Customer order number', required=True)
    .webhook('GET', 'https://api.yourstore.com/orders/${args.order_id}')
    .header('Authorization', 'Bearer ${env.STORE_API_TOKEN}')
    .output(SwaigFunctionResult("""
        Order ${args.order_id} status: ${response.status}
        ${if response.status == "shipped"}
            Tracking: ${response.tracking_number}
            Expected delivery: ${response.estimated_delivery}
        ${endif}
    """))
)
```

The DataMap tool handles the HTTP request, error cases, and response formatting without requiring you to build webhook endpoints or manage server infrastructure.

**Custom SWAIG functions are necessary when you need complex business logic** that can't be expressed through Skills or DataMap configuration. This typically involves multi-step processes, database transactions, or integration with multiple systems that need to be coordinated.

A mortgage application agent, for example, might need a custom function that checks credit scores, validates income documentation, calculates debt-to-income ratios, and updates multiple internal systems. This workflow is too complex for DataMap tools and too specific for a generic skill:

```python
@AgentBase.tool(
    name="process_mortgage_application",
    description="Process a complete mortgage application with all validations",
    parameters={
        "applicant_ssn": {"type": "string", "description": "Social Security Number"},
        "requested_amount": {"type": "number", "description": "Loan amount requested"},
        "property_value": {"type": "number", "description": "Property value"}
    }
)
def process_mortgage_application(self, args, raw_data):
    # Complex multi-step business logic here
    # Credit check, income verification, ratio calculations, etc.
    pass
```

The decision framework is: **Skills first** (if available functionality matches your needs), **DataMap second** (if you need API integration without complex logic), **Custom functions last** (if you need complex business logic or multi-step workflows).

### Effective Prompt Design

Prompt design in voice AI agents differs significantly from text-based AI applications. Voice interactions happen in real-time, users can't easily refer back to previous statements, and the AI needs to maintain conversational flow while handling interruptions and clarifications.

**The Prompt Object Model (POM) provides structure, but the content strategy determines success.** A common mistake is treating voice agent prompts like chatbot instructions, focusing on what the AI should do rather than how it should communicate.

Effective voice prompts establish three layers: **personality** (how the AI sounds and feels), **capability** (what the AI can help with), and **boundaries** (what it cannot or should not do). Each layer serves a different purpose in the conversation.

The personality section should establish tone and communication style in a way that feels natural in speech. Instead of "You are a professional customer service representative," try "You speak in a warm, helpful tone and take time to really understand what each customer needs. You're patient with technical questions and always confirm important details to make sure you got everything right."

This approach gives the AI behavioral cues that translate into natural speech patterns. The AI will pause appropriately, use confirming language, and adopt a helpful tone because the prompt describes behaviors rather than just roles.

Capability descriptions should focus on conversation flow rather than technical features. Instead of listing every function the agent can call, describe the user experience: "When customers have questions about their orders, you can look up real-time status information, tracking details, and delivery estimates. You can also help them modify orders that haven't shipped yet or connect them with specialists for complex issues."

This framing helps the AI understand not just what functions are available, but when and how to use them in conversation. The AI learns that order modification has conditions (hasn't shipped yet) and that some issues require escalation.

Boundary setting prevents the AI from making promises it can't keep or handling situations outside its scope. Rather than technical limitations, frame boundaries in terms of user expectations: "While you can help with most account questions, you'll need to transfer customers to our security team for password resets or account access issues. This keeps their information safe and ensures they get the specialized help they need."

**Context management becomes critical in voice interactions** because users can't scroll back to see previous parts of the conversation. Effective prompts include instructions for maintaining context: "When customers mention something from earlier in the conversation, briefly acknowledge what they're referring to before providing new information. For example, 'About that shipping question you asked earlier...' or 'Going back to the product you were interested in...'"

This type of prompt guidance helps create conversations that feel natural and connected rather than a series of disconnected question-and-answer exchanges.

### Function Organization and Architecture

The architecture of your agent's functions determines both development velocity and long-term maintainability. Well-organized functions follow patterns that make them easy to test, debug, and extend as requirements evolve.

**The single responsibility principle becomes even more important in voice AI** because each function represents a conversational capability that the AI can invoke. Functions that try to do too much create confusion for the AI about when to use them and make error handling more complex.

Consider an e-commerce agent that needs to handle returns. A poorly designed function might be called `handle_returns` and try to process return requests, check return policies, schedule pickups, and issue refunds all in one function. This creates several problems: the AI doesn't know whether to call this function for policy questions or actual return processing, error handling becomes complex because failures can happen at multiple stages, and testing requires setting up multiple system dependencies.

A better approach breaks this into focused functions:

```python
@AgentBase.tool(
    name="check_return_eligibility",
    description="Check if an item is eligible for return based on our policies",
    parameters={
        "order_id": {"type": "string"},
        "item_id": {"type": "string"}
    }
)
def check_return_eligibility(self, args, raw_data):
    # Only handles eligibility checking
    pass

@AgentBase.tool(
    name="initiate_return_request",
    description="Start the return process for an eligible item",
    parameters={
        "order_id": {"type": "string"},
        "item_id": {"type": "string"},
        "reason": {"type": "string"}
    }
)
def initiate_return_request(self, args, raw_data):
    # Only handles return initiation
    pass
```

This separation allows the AI to handle return questions progressively: first checking eligibility, then initiating the process if appropriate. Each function can be tested independently, and error handling is specific to each stage.

**Function naming and descriptions guide AI behavior more than developers often realize.** The AI uses function names and descriptions to decide when to call each function, so clarity here directly impacts conversation quality.

Avoid technical names that don't map to user language. A function named `query_inventory_database` doesn't help the AI understand that it should use this function when customers ask "Do you have this in stock?" A better name like `check_product_availability` with the description "Check if a specific product is currently in stock and available for purchase" gives the AI clear guidance about when this function applies.

**Error handling should be conversational, not technical.** When functions fail, the error messages become part of the conversation, so they need to sound natural and provide helpful guidance to users.

Instead of returning technical error messages like "Database connection timeout" or "Invalid API response," return conversational alternatives: "I'm having trouble accessing our inventory system right now. Let me try a different way to check that for you, or I can connect you with someone who can help immediately."

This approach maintains conversation flow even when technical issues occur and provides users with clear next steps rather than leaving them confused about what went wrong.

### Security Considerations

Security in voice AI agents involves both traditional API security and unique considerations around conversation handling and function access. The real-time nature of voice interactions creates security challenges that don't exist in traditional web applications.

**Function-level security should follow the principle of least privilege.** Each SWAIG function runs with specific permissions, and those permissions should be as narrow as possible while still enabling the required functionality.

Consider a customer service agent that can look up account information. Rather than giving the agent broad database access, create specific functions for each type of lookup with appropriate restrictions:

```python
@AgentBase.tool(
    name="lookup_account_summary",
    description="Get basic account information for customer service",
    parameters={"phone_number": {"type": "string"}},
    secure=True  # Requires authentication
)
def lookup_account_summary(self, args, raw_data):
    # This function can only access summary information
    # No access to payment methods, passwords, or sensitive data
    pass

@AgentBase.tool(
    name="lookup_order_history",
    description="Get recent order history for customer service",
    parameters={"account_id": {"type": "string"}},
    secure=True
)
def lookup_order_history(self, args, raw_data):
    # Limited to order information only
    # No access to payment details or personal information
    pass
```

This approach ensures that even if one function is compromised, the exposure is limited to its specific scope.

**Conversation logging requires careful consideration of privacy and compliance.** Voice conversations often contain sensitive information, and logging practices need to balance debugging needs with privacy requirements.

Implement selective logging that captures operational information without storing sensitive data:

```python
def safe_log_conversation(self, call_id, user_input, ai_response, function_calls):
    # Log operational data
    self.log.info("conversation_turn", {
        "call_id": call_id,
        "user_intent": self.extract_intent(user_input),  # Intent, not exact words
        "functions_called": [f["name"] for f in function_calls],  # Function names only
        "response_type": self.classify_response(ai_response),  # Classification, not content
        "timestamp": datetime.utcnow().isoformat()
    })
    
    # Don't log actual conversation content unless required for debugging
    if self.debug_mode and self.has_user_consent(call_id):
        self.log.debug("conversation_content", {
            "call_id": call_id,
            "content": self.sanitize_sensitive_data(user_input, ai_response)
        })
```

This approach provides operational visibility while protecting user privacy and maintaining compliance with data protection regulations.

**Authentication and authorization become complex in voice interactions** because traditional methods like login forms don't work well in conversational contexts. Voice agents need to balance security with user experience.

For customer service agents, implement progressive authentication that starts with low-security information and escalates as needed:

```python
def progressive_authentication(self, user_request):
    if self.requires_account_access(user_request):
        # Start with phone number or email verification
        account_info = self.verify_contact_info()
        
        if self.requires_sensitive_access(user_request):
            # Escalate to additional verification
            additional_auth = self.request_additional_verification()
            
            if not additional_auth.success:
                return "For your security, I'll need to transfer you to a specialist who can verify your identity and help with this request."
    
    return self.process_request(user_request)
```

This pattern provides security without forcing every user through complex authentication flows for simple requests.

### Performance Optimization

Voice AI agents have unique performance requirements because users expect real-time responses and natural conversation flow. Optimization strategies need to consider both technical performance and conversational experience.

**Function execution time directly impacts conversation quality.** Users will wait 2-3 seconds for complex operations, but anything longer requires conversation management to maintain engagement.

For operations that might take longer, implement progressive response patterns:

```python
@AgentBase.tool(
    name="generate_detailed_report",
    description="Generate a comprehensive account analysis report",
    parameters={"account_id": {"type": "string"}},
    filler="I'm generating that detailed report for you now. This might take a moment since I'm pulling information from several systems..."
)
def generate_detailed_report(self, args, raw_data):
    # Long-running operation with progress updates
    account_id = args.get("account_id")
    
    # Update progress through meta_data
    self.update_meta_data("report_progress", "Gathering account data...")
    account_data = self.fetch_account_data(account_id)
    
    self.update_meta_data("report_progress", "Analyzing transaction patterns...")
    analysis = self.analyze_transactions(account_data)
    
    self.update_meta_data("report_progress", "Generating recommendations...")
    recommendations = self.generate_recommendations(analysis)
    
    result = SwaigFunctionResult(f"I've completed your account analysis report. {self.format_report_summary(analysis, recommendations)}")
    result.add_action("clear_meta_data", "report_progress")
    
    return result
```

The filler text keeps users engaged during processing, and progress updates can be used by the AI to provide status updates if users ask.

**Caching strategies should consider both data freshness and conversation context.** Some information can be cached aggressively (product catalogs, policy information), while other data needs to be fresh for each request (account balances, inventory levels).

Implement smart caching that considers conversation context:

```python
class SmartCache:
    def __init__(self):
        self.static_cache = {}  # Long-lived data (hours/days)
        self.session_cache = {}  # Conversation-specific data
        self.fresh_cache = {}    # Short-lived data (minutes)
    
    def get_product_info(self, product_id, conversation_id):
        # Product details can be cached for hours
        cache_key = f"product:{product_id}"
        if cache_key in self.static_cache:
            return self.static_cache[cache_key]
        
        # Fetch and cache
        product_info = self.fetch_product_info(product_id)
        self.static_cache[cache_key] = product_info
        return product_info
    
    def get_inventory_level(self, product_id, conversation_id):
        # Inventory needs to be fresh, but can be cached for a few minutes
        cache_key = f"inventory:{product_id}"
        if self.is_fresh(cache_key, max_age=300):  # 5 minutes
            return self.fresh_cache[cache_key]
        
        inventory = self.fetch_current_inventory(product_id)
        self.fresh_cache[cache_key] = inventory
        return inventory
```

This approach ensures users get accurate information while minimizing API calls and database queries.

**Memory management becomes important for long-running conversations** or agents that handle many concurrent calls. Voice conversations can accumulate significant context over time, and this needs to be managed efficiently.

Implement context trimming that preserves important information while managing memory usage:

```python
def manage_conversation_context(self, conversation_history):
    # Keep recent turns (last 10 exchanges)
    recent_context = conversation_history[-10:]
    
    # Preserve important events regardless of age
    important_events = [
        turn for turn in conversation_history 
        if turn.get("type") in ["authentication", "order_placement", "issue_resolution"]
    ]
    
    # Summarize older context to preserve key facts
    if len(conversation_history) > 20:
        older_context = conversation_history[:-10]
        context_summary = self.summarize_conversation_context(older_context)
        
        # Replace older context with summary
        return [context_summary] + important_events + recent_context
    
    return conversation_history
```

This approach maintains conversation continuity while preventing memory issues in long conversations.

### Deployment Strategies

Successful deployment of voice AI agents requires considering scalability, reliability, and operational concerns that go beyond basic functionality. Production deployments face challenges around traffic spikes, system integration, and ongoing maintenance that need to be planned from the beginning.

**Multi-agent architectures provide both scalability and maintainability benefits** when designed correctly. Rather than building monolithic agents that handle all use cases, successful deployments typically use specialized agents that can be developed, deployed, and maintained independently.

Consider a telecommunications company that needs AI agents for sales, technical support, and billing. A single agent could theoretically handle all three areas, but this creates several problems: the prompt becomes enormous and hard to maintain, function updates in one area risk breaking others, and different teams can't work independently on their areas.

A better approach uses specialized agents with a routing layer:

```python
# Main entry point agent
class TelecomConcierge(ConciergeAgent):
    def __init__(self):
        super().__init__(
            name="telecom-main",
            route="/telecom",
            routing_options=[
                {
                    "name": "Sales Agent",
                    "description": "New service plans, upgrades, device purchases",
                    "route": "/telecom/sales",
                    "business_hours": "24/7"
                },
                {
                    "name": "Technical Support",
                    "description": "Service issues, outages, device troubleshooting", 
                    "route": "/telecom/support",
                    "business_hours": "24/7"
                },
                {
                    "name": "Billing Agent",
                    "description": "Bill questions, payment issues, account management",
                    "route": "/telecom/billing", 
                    "business_hours": "Monday-Friday 8AM-8PM EST"
                }
            ]
        )

# Specialized agents
class TelecomSalesAgent(Agent):
    def __init__(self):
        super().__init__(name="telecom-sales", route="/telecom/sales")
        self.add_skill("web_search", {
            "api_key": "your-google-api-key", 
            "search_engine_id": "company-plans-engine-id",
            "num_results": 3
        })
        # Sales-specific functions and configuration
        
class TelecomSupportAgent(Agent):
    def __init__(self):
        super().__init__(name="telecom-support", route="/telecom/support")
        self.add_skill("native_vector_search", {"index_file": "support_kb.swsearch"})
        # Support-specific functions and configuration
```

This architecture allows different teams to maintain their agents independently while providing a unified customer experience through the concierge layer.

**Load balancing and scaling strategies need to consider conversation state and routing.** Unlike stateless web APIs, voice agents often maintain conversation context that affects how they can be scaled.

For stateless agents (which is the SDK default), standard load balancing works well:

```python
# Deploy multiple instances behind a load balancer
# Each instance can handle any conversation since there's no persistent state

# Docker Compose example
version: '3.8'
services:
  agent-1:
    image: your-agent:latest
    environment:
      - INSTANCE_ID=agent-1
      - PORT=5000
  
  agent-2:
    image: your-agent:latest  
    environment:
      - INSTANCE_ID=agent-2
      - PORT=5001
      
  load-balancer:
    image: nginx:alpine
    ports:
      - "80:80"
    # Load balancer configuration
```

For agents that need persistent state, you'll need session affinity or external state management:

```python
# External state management with Redis
class StatefulAgent(Agent):
    def __init__(self):
        super().__init__(name="stateful-agent")
        self.redis_client = redis.Redis(host='redis-cluster')
    
    def get_conversation_state(self, call_id):
        return self.redis_client.get(f"conversation:{call_id}")
    
    def save_conversation_state(self, call_id, state):
        self.redis_client.setex(f"conversation:{call_id}", 3600, state)
```

**Monitoring and observability require conversation-aware metrics** beyond standard application monitoring. Voice agents need metrics that help understand both technical performance and conversation quality.

Implement comprehensive monitoring that tracks both system and conversational metrics:

```python
class AgentMonitoring:
    def __init__(self):
        self.metrics = MetricsCollector()
        
    def track_conversation_start(self, call_id, agent_name):
        self.metrics.increment(f"conversations.started.{agent_name}")
        self.metrics.timing("conversation.start_time", time.time())
        
    def track_function_call(self, function_name, execution_time, success):
        self.metrics.timing(f"functions.{function_name}.duration", execution_time)
        self.metrics.increment(f"functions.{function_name}.{'success' if success else 'error'}")
        
    def track_conversation_end(self, call_id, agent_name, duration, resolution_type):
        self.metrics.timing(f"conversations.duration.{agent_name}", duration)
        self.metrics.increment(f"conversations.resolution.{resolution_type}")
        
    def track_conversation_quality(self, call_id, user_satisfaction, resolution_achieved):
        self.metrics.histogram("conversation.satisfaction", user_satisfaction)
        self.metrics.increment(f"conversation.resolution.{'achieved' if resolution_achieved else 'unresolved'}")
```

These metrics help identify both technical issues (slow function calls, high error rates) and conversation problems (low satisfaction, frequent escalations) that require different types of remediation.

**Deployment pipelines should include conversation testing,** not just unit tests. Voice agents need integration testing that validates conversation flows and AI behavior under realistic conditions.

```python
# Conversation flow testing
class ConversationTest:
    def test_order_lookup_flow(self):
        agent = self.create_test_agent()
        
        # Simulate conversation turns
        response1 = agent.process_input("I need to check on my order")
        assert "order number" in response1.lower()
        
        response2 = agent.process_input("It's order 12345") 
        assert self.function_was_called("lookup_order_status")
        assert "status" in response2.lower()
        
        # Test error handling
        response3 = agent.process_input("Actually, the order number is ABC123")
        assert "not found" in response3.lower() or "invalid" in response3.lower()
```

This type of testing catches conversation flow issues that unit tests miss and ensures that AI behavior remains consistent across deployments.

By following these patterns and practices, you can build voice AI agents that not only work correctly but scale effectively, remain maintainable over time, and provide excellent user experiences in production environments.

## 12. Real-World Examples

Understanding the SignalWire AI Agents SDK through practical examples helps bridge the gap between conceptual knowledge and real-world implementation. These examples demonstrate complete, production-ready solutions that combine multiple SDK features to solve common business challenges.

### E-Commerce Customer Service Agent

One of the most common applications for voice AI is enhancing customer service operations. This example shows a complete e-commerce customer service agent that combines skills, DataMap tools, and custom functions to handle order inquiries, product questions, and support requests.

#### Complete Implementation

```python
from signalwire_agents import AgentBase
from signalwire_agents.core.data_map import DataMap
from signalwire_agents.core.function_result import SwaigFunctionResult

class EcommerceServiceAgent(AgentBase):
    def __init__(self):
        super().__init__(name="ecommerce-service", route="/customer-service")
        
        # Set up the agent's personality and capabilities
        self.set_prompt_text("""
        You are Sarah, a friendly and knowledgeable customer service representative 
        for GreenTech Electronics. You're patient, thorough, and always make sure 
        customers feel heard and helped. You can look up orders, check product 
        availability, help with returns, and connect customers with specialists 
        when needed.
        
        Always confirm important details like order numbers and ensure customers 
        understand next steps before ending the conversation.
        """)
        
        # Add skills for enhanced capabilities
        self.add_skill("web_search", {
            "api_key": "your-google-api-key",
            "search_engine_id": "greentech-products-engine-id",
            "num_results": 3,
            "tool_name": "product_search"
        })
        
        self.add_skill("datetime")
        
        # Add knowledge base search
        self.add_skill("native_vector_search", {
            "index_file": "customer_service_kb.swsearch",
            "max_results": 5,
            "similarity_threshold": 0.7
        })
        
        # Set up DataMap tools for API integration
        self._setup_datamap_tools()
        
        # Configure language and voice settings
        self.add_language(
            name="English",
            code="en-US", 
            voice="elevenlabs.sarah:eleven_multilingual_v2",
            speech_fillers=["Let me look that up for you...", "One moment please..."],
            function_fillers=["I'm checking our system now...", "Let me pull up that information..."]
        )
    
    def _setup_datamap_tools(self):
        """Configure DataMap tools for order and inventory APIs"""
        
        # Order lookup tool
        order_lookup = (DataMap('lookup_order')
            .parameter('order_number', 'string', 'Customer order number', required=True)
            .webhook('GET', 'https://api.greentech-electronics.com/orders/${order_number}',
                     headers={
                         'Authorization': 'Bearer ${env.ECOMMERCE_API_TOKEN}',
                         'Content-Type': 'application/json'
                     })
            .output(SwaigFunctionResult("""
                I found your order ${response.id}. 
                Status: ${response.status}
                Total: $${response.total_amount}
                Tracking: ${response.tracking_number}
            """))
            .error_keys(['error', 'message'])
        )
        
        # Inventory check tool
        inventory_check = (DataMap('check_inventory')
            .parameter('product_sku', 'string', 'Product SKU or model number', required=True)
            .parameter('location', 'string', 'Store location or zip code', required=False)
            .webhook('GET', 'https://api.greentech-electronics.com/inventory/${product_sku}?location=${location}',
                     headers={'Authorization': 'Bearer ${env.ECOMMERCE_API_TOKEN}'})
            .output(SwaigFunctionResult("""
                ${response.product_name} (SKU: ${response.sku})
                Available: ${response.quantity_available} units
                Current price: $${response.current_price}
            """))
            .error_keys(['error', 'message'])
        )
        
        # Add the tools to the agent
        self.register_swaig_function(order_lookup.to_swaig_function())
        self.register_swaig_function(inventory_check.to_swaig_function())
    
    @AgentBase.tool(
        name="initiate_return_request",
        description="Start a return request for a customer order",
        parameters={
            "order_number": {"type": "string", "description": "Order number"},
            "item_sku": {"type": "string", "description": "SKU of item to return"},
            "reason": {"type": "string", "description": "Reason for return"},
            "condition": {"type": "string", "enum": ["unopened", "opened_unused", "defective"], "description": "Item condition"}
        }
    )
    def initiate_return_request(self, args, raw_data):
        """Process return requests with business logic validation"""
        
        order_number = args.get("order_number")
        item_sku = args.get("item_sku")
        reason = args.get("reason")
        condition = args.get("condition")
        
        try:
            # Check if order exists and is eligible for return
            order_info = self._validate_return_eligibility(order_number, item_sku)
            
            if not order_info['eligible']:
                return SwaigFunctionResult(
                    f"I'm sorry, but this item isn't eligible for return. {order_info['reason']} "
                    "If you have questions about our return policy, I can connect you with a manager."
                )
            
            # Create return request
            return_id = self._create_return_request(order_number, item_sku, reason, condition)
            
            # Generate return shipping label if needed
            if condition in ["unopened", "opened_unused"]:
                shipping_label = self._generate_return_label(return_id)
                
                result = SwaigFunctionResult(
                    f"I've created return request #{return_id} for your {order_info['product_name']}. "
                    f"You'll receive an email with a prepaid return shipping label within the next hour. "
                    f"Once we receive the item, your refund of ${order_info['refund_amount']} will be processed within 3-5 business days."
                )
                
                result.add_action("send_email", {
                    "template": "return_confirmation",
                    "return_id": return_id,
                    "shipping_label_url": shipping_label['url']
                })
                
            else:  # defective item
                result = SwaigFunctionResult(
                    f"I've created return request #{return_id} for your defective {order_info['product_name']}. "
                    f"Since this is a defective item, I'm also issuing an immediate store credit of ${order_info['refund_amount']} "
                    f"that you can use right away. You'll still need to return the defective item using the prepaid label we're sending."
                )
                
                # Issue immediate store credit for defective items
                result.add_action("issue_store_credit", {
                    "amount": order_info['refund_amount'],
                    "reason": "defective_item_return"
                })
            
            # Update order status
            result.add_action("set_global_data", {
                "last_return_request": {
                    "id": return_id,
                    "order_number": order_number,
                    "status": "pending_return"
                }
            })
            
            return result
            
        except Exception as e:
            self.log.error("return_request_error", error=str(e), order_number=order_number)
            return SwaigFunctionResult(
                "I encountered an issue processing your return request. Let me connect you with a specialist "
                "who can help you right away."
            )
    
    def _validate_return_eligibility(self, order_number, item_sku):
        """Check if item is eligible for return based on business rules"""
        # This would integrate with your order management system
        # For example purposes, showing the logic structure
        
        import datetime
        
        # Mock order lookup (replace with actual API call)
        order_date = datetime.datetime(2024, 1, 10)  # Mock date
        days_since_order = (datetime.datetime.now() - order_date).days
        
        if days_since_order > 30:
            return {
                'eligible': False,
                'reason': 'Returns must be initiated within 30 days of purchase.',
                'product_name': 'Unknown Item'
            }
        
        # Additional business logic checks would go here
        # - Product category restrictions
        # - Previous return history
        # - Order payment status
        
        return {
            'eligible': True,
            'product_name': 'Wireless Headphones Pro',
            'refund_amount': '149.99'
        }
    
    def _create_return_request(self, order_number, item_sku, reason, condition):
        """Create return request in the system"""
        # Integration with return management system
        import uuid
        return f"RET-{uuid.uuid4().hex[:8].upper()}"
    
    def _generate_return_label(self, return_id):
        """Generate prepaid return shipping label"""
        # Integration with shipping provider
        return {
            'url': f'https://shipping.greentech-electronics.com/labels/{return_id}',
            'tracking': f'1Z999AA1{return_id[:8]}'
        }

# Deploy the agent
if __name__ == "__main__":
    agent = EcommerceServiceAgent()
    agent.run()
```

#### Key Features Demonstrated

This example showcases several important patterns:

**Integrated Capability Stack**: The agent combines built-in skills (web search, datetime, vector search) with custom DataMap tools and SWAIG functions, creating a comprehensive customer service solution.

**Business Logic Integration**: The return processing function includes real business rules like eligibility checking, condition-based handling, and automatic store credit issuance for defective items.

**Error Handling and Escalation**: Both DataMap tools and custom functions include comprehensive error handling that maintains conversational flow while providing clear escalation paths.

**Action Orchestration**: Functions return not just text responses but also trigger actions like sending emails, issuing credits, and updating global state for context preservation.

### Healthcare Appointment Scheduling System

Healthcare organizations need sophisticated scheduling systems that handle complex requirements like provider availability, insurance verification, and appointment type matching. This example demonstrates a complete scheduling solution.

#### Implementation Overview

```python
from signalwire_agents import AgentBase
from signalwire_agents.prefabs import InfoGathererAgent
import datetime
import json

class HealthcareSchedulingAgent(AgentBase):
    def __init__(self):
        super().__init__(name="healthcare-scheduling", route="/scheduling")
        
        self.set_prompt_text("""
        You are Maria, a caring and efficient scheduling coordinator for Valley 
        Medical Center. You help patients schedule appointments, understand 
        insurance requirements, and prepare for their visits. You're particularly 
        good at explaining medical procedures in simple terms and ensuring patients 
        feel comfortable and informed.
        
        You always confirm appointment details twice and provide clear instructions 
        about preparation, what to bring, and parking information.
        """)
        
        # Add skills for comprehensive functionality
        self.add_skill("datetime", {
            "default_timezone": "US/Pacific",
            "business_hours": {
                "monday": {"start": "07:00", "end": "19:00"},
                "tuesday": {"start": "07:00", "end": "19:00"},
                "wednesday": {"start": "07:00", "end": "19:00"},
                "thursday": {"start": "07:00", "end": "19:00"},
                "friday": {"start": "07:00", "end": "17:00"}
            },
            "holidays": ["2024-12-25", "2024-01-01", "2024-07-04"]
        })
        
        # Medical knowledge base search
        self.add_skill("native_vector_search", {
            "index_file": "medical_procedures_kb.swsearch",
            "max_results": 3,
            "similarity_threshold": 0.8
        }, skill_name="medical_info_search")
        
        # Insurance verification search
        self.add_skill("native_vector_search", {
            "index_file": "insurance_policies_kb.swsearch", 
            "max_results": 5,
            "boost_keywords": True
        }, skill_name="insurance_search")
        
        # Set up provider and appointment type data
        self._setup_scheduling_data()
        
        # Configure DataMap tools for external integrations
        self._setup_external_integrations()
    
    def _setup_scheduling_data(self):
        """Initialize provider schedules and appointment types"""
        
        self.providers = {
            "dr_johnson": {
                "name": "Dr. Emily Johnson",
                "specialty": "Internal Medicine",
                "appointment_types": ["Annual Physical", "Follow-up", "Consultation"],
                "duration_minutes": {"Annual Physical": 60, "Follow-up": 30, "Consultation": 45},
                "availability": {
                    "monday": ["09:00", "10:00", "14:00", "15:00"],
                    "wednesday": ["09:00", "10:30", "14:00", "15:30"],
                    "friday": ["08:00", "09:00", "10:00", "11:00"]
                }
            },
            "dr_patel": {
                "name": "Dr. Raj Patel", 
                "specialty": "Cardiology",
                "appointment_types": ["Consultation", "Follow-up", "Stress Test"],
                "duration_minutes": {"Consultation": 45, "Follow-up": 30, "Stress Test": 90},
                "availability": {
                    "tuesday": ["08:00", "09:30", "13:00", "14:30"],
                    "thursday": ["08:00", "09:30", "13:00", "14:30"],
                    "friday": ["13:00", "14:00", "15:00"]
                }
            }
        }
        
        self.appointment_requirements = {
            "Annual Physical": {
                "preparation": "Fasting for 12 hours before appointment for blood work",
                "duration": "60 minutes",
                "bring": ["Insurance card", "List of current medications", "Previous test results"],
                "insurance_notes": "Most insurance covers annual physicals at 100%"
            },
            "Stress Test": {
                "preparation": "Wear comfortable walking shoes and athletic clothing. No caffeine 24 hours before.",
                "duration": "90 minutes", 
                "bring": ["Insurance card", "List of heart medications"],
                "insurance_notes": "Prior authorization may be required. We'll verify this for you."
            }
        }
    
    def _setup_external_integrations(self):
        """Configure DataMap tools for insurance and EMR integration"""
        
        # Insurance verification tool
        insurance_verify = (DataMap('verify_insurance')
            .parameter('insurance_id', 'string', 'Patient insurance member ID', required=True)
            .parameter('insurance_company', 'string', 'Insurance company name', required=True)
            .parameter('service_type', 'string', 'Type of service/appointment', required=True)
            .webhook('POST', 'https://api.insurancegateway.com/verify')
            .header('Authorization', 'Bearer ${env.INSURANCE_API_KEY}')
            .body({
                'member_id': '${args.insurance_id}',
                'payer': '${args.insurance_company}',
                'service_code': '${helper.map_service_to_code(args.service_type)}',
                'provider_npi': '${env.PROVIDER_NPI}'
            })
            .output(SwaigFunctionResult("""
                ${if response.eligible}
                    ✓ Your insurance is active and covers this service.
                    Copay: $${response.copay_amount}
                    ${if response.deductible_remaining > 0}
                        Remaining deductible: $${response.deductible_remaining}
                    ${endif}
                    ${if response.prior_auth_required}
                        Note: Prior authorization is required. We'll handle this for you.
                    ${endif}
                ${else}
                    ⚠️  Insurance verification issue: ${response.reason}
                    Please contact your insurance company or our billing department for assistance.
                ${endif}
            """))
        )
        
        self.add_datamap_tool(insurance_verify)
    
    @AgentBase.tool(
        name="check_provider_availability",
        description="Check appointment availability for a specific provider and date",
        parameters={
            "provider_id": {"type": "string", "description": "Provider identifier"},
            "date": {"type": "string", "description": "Requested date (YYYY-MM-DD)"},
            "appointment_type": {"type": "string", "description": "Type of appointment needed"}
        }
    )
    def check_provider_availability(self, args, raw_data):
        """Check real-time provider availability"""
        
        provider_id = args.get("provider_id") 
        requested_date = args.get("date")
        appointment_type = args.get("appointment_type")
        
        try:
            provider = self.providers.get(provider_id)
            if not provider:
                return SwaigFunctionResult("I couldn't find that provider. Let me show you our available providers.")
            
            # Parse the requested date
            date_obj = datetime.datetime.strptime(requested_date, "%Y-%m-%d")
            day_name = date_obj.strftime("%A").lower()
            
            # Check if provider works on that day
            if day_name not in provider["availability"]:
                return SwaigFunctionResult(
                    f"Dr. {provider['name']} doesn't have office hours on {date_obj.strftime('%A')}s. "
                    f"They're available on: {', '.join(provider['availability'].keys()).title()}"
                )
            
            # Get available time slots for that day
            available_times = provider["availability"][day_name]
            duration = provider["duration_minutes"].get(appointment_type, 30)
            
            # Check for existing appointments (in a real system, this would query the EMR)
            booked_times = self._get_booked_appointments(provider_id, requested_date)
            available_slots = [time for time in available_times if time not in booked_times]
            
            if not available_slots:
                # Suggest alternative dates
                alternatives = self._find_alternative_dates(provider_id, appointment_type, 7)
                return SwaigFunctionResult(
                    f"Dr. {provider['name']} is fully booked on {requested_date}. "
                    f"Here are the next available appointments: {', '.join(alternatives[:3])}"
                )
            
            result = SwaigFunctionResult(
                f"Dr. {provider['name']} has these times available on {requested_date}:\n" +
                f"Times: {', '.join(available_slots)}\n" +
                f"Appointment duration: {duration} minutes\n" +
                f"Which time works best for you?"
            )
            
            # Store availability info for booking
            result.add_action("set_global_data", {
                "pending_appointment": {
                    "provider_id": provider_id,
                    "provider_name": provider["name"],
                    "date": requested_date,
                    "appointment_type": appointment_type,
                    "available_times": available_slots,
                    "duration": duration
                }
            })
            
            return result
            
        except ValueError:
            return SwaigFunctionResult("I didn't understand that date format. Could you tell me the date like 'January 15th' or '1/15/2024'?")
        except Exception as e:
            self.log.error("availability_check_error", error=str(e))
            return SwaigFunctionResult("I'm having trouble checking availability right now. Let me connect you with our scheduling team.")
    
    @AgentBase.tool(
        name="book_appointment",
        description="Book a confirmed appointment slot",
        parameters={
            "patient_name": {"type": "string", "description": "Patient full name"},
            "phone_number": {"type": "string", "description": "Patient phone number"},
            "email": {"type": "string", "description": "Patient email address"},
            "insurance_id": {"type": "string", "description": "Insurance member ID"},
            "insurance_company": {"type": "string", "description": "Insurance company name"},
            "time_slot": {"type": "string", "description": "Selected time slot"},
            "special_requests": {"type": "string", "description": "Any special accommodations needed"}
        }
    )
    def book_appointment(self, args, raw_data):
        """Complete the appointment booking process"""
        
        try:
            # Get pending appointment details from global data
            pending = self.get_global_data("pending_appointment", {})
            
            if not pending:
                return SwaigFunctionResult("I don't have appointment details ready. Let me start over with checking availability.")
            
            # Validate the selected time slot
            selected_time = args.get("time_slot")
            if selected_time not in pending.get("available_times", []):
                return SwaigFunctionResult(f"That time slot isn't available. Please choose from: {', '.join(pending['available_times'])}")
            
            # Create appointment record
            appointment_id = self._create_appointment_record({
                **args,
                **pending,
                "selected_time": selected_time
            })
            
            # Get appointment preparation instructions
            appointment_type = pending["appointment_type"]
            requirements = self.appointment_requirements.get(appointment_type, {})
            
            confirmation_message = f"""
            Perfect! I've booked your {appointment_type} appointment.
            
            📅 Appointment Details:
            Provider: {pending['provider_name']}
            Date: {pending['date']}
            Time: {selected_time}
            Duration: {pending['duration']} minutes
            Confirmation #: {appointment_id}
            
            📋 What to bring:
            {chr(10).join('• ' + item for item in requirements.get('bring', ['Insurance card']))}
            
            🏥 Location: Valley Medical Center, Suite 200
            
            📞 Need to reschedule? Call us at (555) 123-4567 or use confirmation #{appointment_id}
            """
            
            if requirements.get('preparation'):
                confirmation_message += f"\n⚠️  Important preparation: {requirements['preparation']}"
            
            result = SwaigFunctionResult(confirmation_message)
            
            # Trigger confirmation actions
            result.add_actions([
                {"send_confirmation_email": {
                    "appointment_id": appointment_id,
                    "patient_email": args.get("email"),
                    "appointment_details": pending
                }},
                {"send_calendar_invite": {
                    "appointment_id": appointment_id,
                    "patient_email": args.get("email"),
                    "start_time": f"{pending['date']}T{selected_time}:00",
                    "duration_minutes": pending['duration']
                }},
                {"set_reminder": {
                    "appointment_id": appointment_id,
                    "phone_number": args.get("phone_number"),
                    "reminder_time": "24_hours_before"
                }}
            ])
            
            # Clear pending appointment data
            result.add_action("unset_global_data", "pending_appointment")
            
            return result
            
        except Exception as e:
            self.log.error("booking_error", error=str(e))
            return SwaigFunctionResult("I encountered an issue completing your booking. Let me connect you with our scheduling team to finalize this appointment.")
    
    def _get_booked_appointments(self, provider_id, date):
        """Get existing appointments for provider on specific date"""
        # In a real implementation, this would query your EMR/scheduling system
        # For demo purposes, simulating some booked slots
        mock_bookings = {
            "dr_johnson": {
                "2024-01-15": ["09:00", "14:00"],
                "2024-01-17": ["10:00"]
            }
        }
        return mock_bookings.get(provider_id, {}).get(date, [])
    
    def _find_alternative_dates(self, provider_id, appointment_type, days_ahead):
        """Find alternative appointment dates when requested date is unavailable"""
        alternatives = []
        provider = self.providers.get(provider_id)
        
        if not provider:
            return alternatives
        
        current_date = datetime.datetime.now().date()
        
        for i in range(1, days_ahead + 1):
            check_date = current_date + datetime.timedelta(days=i)
            day_name = check_date.strftime("%A").lower()
            
            if day_name in provider["availability"]:
                available_times = provider["availability"][day_name]
                booked_times = self._get_booked_appointments(provider_id, check_date.strftime("%Y-%m-%d"))
                
                if any(time not in booked_times for time in available_times):
                    alternatives.append(f"{check_date.strftime('%A, %B %d')} at {available_times[0]}")
                    
                if len(alternatives) >= 5:  # Limit suggestions
                    break
        
        return alternatives
    
    def _create_appointment_record(self, appointment_data):
        """Create appointment in the scheduling system"""
        # Integration with EMR/scheduling system would go here
        import uuid
        return f"APT-{uuid.uuid4().hex[:8].upper()}"

if __name__ == "__main__":
    agent = HealthcareSchedulingAgent()
    agent.run()
```

This healthcare example demonstrates several advanced patterns:

**Multi-System Integration**: The agent integrates with insurance verification APIs, EMR systems, and scheduling databases through a combination of DataMap tools and custom functions.

**Complex Business Logic**: Provider availability checking includes multiple constraints like working days, appointment durations, existing bookings, and alternative date suggestions.

**Comprehensive Patient Experience**: From initial inquiry through booking confirmation, the system handles the complete patient journey including preparation instructions, calendar invites, and appointment reminders.

**Compliance and Safety**: The system includes insurance verification, proper documentation, and secure handling of patient information.

### Financial Services Loan Application Assistant

Financial services require sophisticated AI agents that can guide customers through complex processes while maintaining regulatory compliance. This example shows a loan pre-qualification assistant that demonstrates advanced conversation flows and integration patterns.

#### Key Implementation Highlights

```python
class LoanApplicationAgent(AgentBase):
    def __init__(self):
        super().__init__(name="loan-application", route="/loans")
        
        # Configure for financial services compliance
        self.set_prompt_text("""
        You are Alex, a knowledgeable loan specialist at Community First Bank. 
        You help customers understand loan options and guide them through the 
        pre-qualification process. You're required to provide accurate information 
        about rates, terms, and requirements while ensuring customers understand 
        that pre-qualification is not a guarantee of final approval.
        
        Always remind customers that final terms depend on full application review 
        and that rates may vary based on creditworthiness. You must comply with 
        fair lending practices and provide equal service to all customers.
        """)
        
        # Add comprehensive financial knowledge
        self.add_skill("native_vector_search", {
            "index_file": "loan_products_kb.swsearch",
            "max_results": 5,
            "similarity_threshold": 0.8
        })
        
        self.add_skill("math", {
            "precision": 2,
            "financial_mode": True,  # Special handling for currency calculations
            "explain_steps": True
        })
        
        # Set up loan product data and rate tables
        self._initialize_loan_products()
        
        # Configure DataMap tools for credit checks and verification
        self._setup_verification_tools()
    
    @AgentBase.tool(
        name="calculate_loan_payments",
        description="Calculate monthly payments and loan costs for different scenarios",
        parameters={
            "loan_amount": {"type": "number", "description": "Principal loan amount"},
            "interest_rate": {"type": "number", "description": "Annual interest rate as percentage"},
            "term_months": {"type": "integer", "description": "Loan term in months"},
            "loan_type": {"type": "string", "description": "Type of loan (mortgage, auto, personal)"}
        }
    )
    def calculate_loan_payments(self, args, raw_data):
        """Provide comprehensive loan payment calculations"""
        
        principal = args.get("loan_amount")
        annual_rate = args.get("interest_rate") / 100  # Convert percentage
        term_months = args.get("term_months")
        loan_type = args.get("loan_type", "personal")
        
        try:
            # Calculate monthly payment using standard amortization formula
            monthly_rate = annual_rate / 12
            monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**term_months) / ((1 + monthly_rate)**term_months - 1)
            
            total_paid = monthly_payment * term_months
            total_interest = total_paid - principal
            
            # Add loan-type specific considerations
            additional_info = self._get_loan_type_info(loan_type, principal, monthly_payment)
            
            result = SwaigFunctionResult(f"""
            Loan Payment Calculation:
            
            💰 Principal Amount: ${principal:,.2f}
            📊 Interest Rate: {args.get('interest_rate')}% APR
            📅 Term: {term_months} months ({term_months//12} years)
            
            💳 Monthly Payment: ${monthly_payment:,.2f}
            💵 Total Amount Paid: ${total_paid:,.2f}
            📈 Total Interest: ${total_interest:,.2f}
            
            {additional_info}
            
            Remember: This is an estimate. Final rates depend on credit approval and may include additional fees.
            """)
            
            # Store calculation for potential application
            result.add_action("set_global_data", {
                "loan_calculation": {
                    "principal": principal,
                    "rate": args.get("interest_rate"),
                    "term": term_months,
                    "monthly_payment": monthly_payment,
                    "loan_type": loan_type
                }
            })
            
            return result
            
        except Exception as e:
            return SwaigFunctionResult("I had trouble with that calculation. Could you verify the loan amount and terms?")
```

These examples demonstrate how the SignalWire AI Agents SDK can be used to build sophisticated, production-ready applications that handle real business requirements while maintaining excellent user experiences.

## 13. Conclusion

The SignalWire AI Agents SDK represents a fundamental shift in how conversational AI applications are built and deployed. What traditionally required months of infrastructure development, complex integrations, and specialized expertise can now be accomplished in days using the SDK's Skills System, DataMap tools, and comprehensive agent framework.

### The Transformation of Voice AI Development

Throughout this guide, we've seen how the SDK abstracts away the most challenging aspects of voice AI development while preserving the flexibility needed for sophisticated applications. The Skills System eliminates the need to implement common capabilities like web search, mathematical calculations, and document search from scratch. DataMap tools provide API integration without webhook infrastructure. The Local Search System enables offline knowledge base functionality with just a few CLI commands.

But perhaps most importantly, the SDK allows developers to focus on what matters most: creating intelligent, helpful experiences for users. Instead of spending weeks building HTTP servers, managing conversation state, or implementing search algorithms, developers can concentrate on understanding their users' needs and designing agents that meet those needs effectively.

The real-world examples we've explored—from e-commerce customer service to healthcare scheduling to financial services—demonstrate that this isn't just about simplifying development. It's about enabling organizations to deploy AI capabilities that were previously accessible only to companies with massive technical resources.

### Key Architectural Insights

The SignalWire SDK's architecture teaches us several important lessons about building scalable voice AI systems:

**Stateless-first design** enables true scalability while still providing mechanisms for state management when needed. This approach allows agents to handle thousands of concurrent conversations while maintaining the flexibility to implement complex workflows that require persistence.

**Modular capability integration** through Skills, DataMap, and custom functions creates clear separation of concerns. Developers can choose the right tool for each requirement without forcing everything into a single pattern.

**Platform-managed infrastructure** removes operational overhead while maintaining security and reliability. DataMap tools running on SignalWire's servers mean fewer endpoints to secure, monitor, and maintain.

**Conversation-aware design** throughout the SDK ensures that technical capabilities translate into natural user experiences. From error handling that maintains conversational flow to function descriptions that guide AI behavior, every aspect considers the voice interaction context.

### Strategic Implications for Organizations

For organizations considering voice AI initiatives, the SignalWire SDK changes the fundamental economics and risk profile of these projects:

**Development velocity** increases dramatically when common capabilities are available as one-line additions rather than multi-week implementations. Teams can iterate quickly, test different approaches, and respond to user feedback without massive technical overhead.

**Technical risk** decreases significantly when using battle-tested, platform-provided capabilities. Skills like web search and datetime handling include edge cases, error handling, and optimizations that would take months to develop and debug in custom implementations.

**Operational complexity** shrinks when infrastructure management is handled by the platform. Teams can focus on conversation design and business logic rather than scaling servers, managing dependencies, and monitoring complex distributed systems.

**Innovation capacity** expands when developers aren't constrained by implementation complexity. Ideas that seemed too ambitious become feasible when the foundational capabilities are readily available.

### The Future of Conversational AI

The patterns and capabilities demonstrated in this guide point toward several important trends in conversational AI:

**Democratization of advanced capabilities** will continue as platforms abstract away technical complexity. The gap between what large tech companies and smaller organizations can accomplish in AI will narrow as sophisticated tools become more accessible.

**Hybrid human-AI workflows** will become the norm rather than the exception. The SDK's design for escalation, context preservation, and seamless handoffs recognizes that the most effective solutions combine AI efficiency with human expertise and empathy.

**Domain-specific specialization** will increase as the barrier to creating custom agents decreases. Instead of one-size-fits-all chatbots, organizations will deploy specialized agents for specific use cases, each optimized for their particular domain and user needs.

**Real-time, contextual assistance** will become expected across industries. As voice AI becomes more reliable and capable, users will expect intelligent assistance to be available whenever and wherever they need it.

### Practical Next Steps

For developers and organizations ready to begin their voice AI journey with the SignalWire SDK:

**Start with a specific use case** rather than trying to build a general-purpose agent. Choose a well-defined problem where you can measure success clearly—customer service for a specific product line, appointment scheduling for a particular department, or lead qualification for a specific market segment.

**Begin with Skills and DataMap tools** before building custom functions. Most use cases can be addressed with the built-in capabilities, and starting simple allows you to understand the platform's patterns and capabilities before adding complexity.

**Design for conversation flow, not just functionality.** The most successful voice AI implementations prioritize how interactions feel to users rather than just what capabilities are available. Spend time understanding your users' language patterns and conversation preferences.

**Plan for iteration and learning.** Voice AI applications improve significantly with real user feedback. Deploy quickly with core functionality, then enhance based on actual usage patterns rather than theoretical requirements.

**Consider the operational context** from the beginning. How will the agent be maintained? Who will update the knowledge base? How will you handle escalations? These operational considerations are as important as the technical implementation.

### The Broader Impact

Voice AI agents built with platforms like SignalWire's SDK represent more than just technological advancement—they represent an opportunity to reimagine how organizations interact with their customers, employees, and communities.

When implemented thoughtfully, these agents can provide more personalized, accessible, and efficient experiences than traditional interfaces. They can operate 24/7 without fatigue, handle multiple languages naturally, and scale to serve thousands of users simultaneously while maintaining consistent quality.

But perhaps most importantly, they free human workers to focus on the aspects of their jobs that require creativity, empathy, and complex problem-solving. Rather than replacing human workers, well-designed voice AI agents augment human capabilities and create opportunities for more meaningful work.

### Final Thoughts

The SignalWire AI Agents SDK provides the tools needed to build sophisticated voice AI applications, but tools alone don't create great user experiences. The most successful implementations will combine technical capability with deep understanding of user needs, thoughtful conversation design, and ongoing commitment to improvement.

As you begin building with the SDK, remember that the goal isn't just to create agents that can perform tasks—it's to create agents that make people's lives easier, more productive, and more enjoyable. The technology is ready. The question is: what will you build?

The future of human-computer interaction is conversational, and that future is available today. Start building, start learning, and start creating the voice AI experiences that will define how we interact with technology in the years to come.

### Resources and Community

Ready to start building? Here's how to connect with the SignalWire community and access additional resources:

**Technical Documentation**: [https://developer.signalwire.com](https://developer.signalwire.com) - Comprehensive API references, advanced configuration guides, and integration examples

**Developer Community**: [https://signalwire.community](https://signalwire.community) - Join thousands of developers building with SignalWire. Share projects, get help, and collaborate on innovative solutions

**SignalWire AI Platform**: [https://signalwire.ai](https://signalwire.ai) - Explore the full range of AI capabilities and see what's possible with voice AI

**GitHub Repository**: Access example projects, contribute to the SDK, and explore open-source tools that extend the platform's capabilities

**Professional Services**: For enterprise implementations or complex custom requirements, SignalWire's professional services team can provide architecture consulting, custom development, and deployment support

The conversation starts now. What will you build? 
