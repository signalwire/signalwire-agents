# SignalWire Agents Skills System

The SignalWire Agents SDK now includes a modular skills system that lets you add capabilities to your agents with simple one-liner calls and configurable parameters.

## What's New

Instead of manually implementing every agent capability, you can now:

```python
from signalwire_agents import AgentBase

# Create an agent
agent = AgentBase("My Assistant")

# Add skills with one-liners!
agent.add_skill("web_search")   # Web search capability with default settings
agent.add_skill("datetime")     # Current date/time info  
agent.add_skill("math")         # Mathematical calculations

# Add skills with custom parameters!
agent.add_skill("web_search", {
    "num_results": 3,  # Get 3 search results instead of default 1
    "delay": 0.5       # Add 0.5s delay between requests instead of default 0
})

# Your agent now has all these capabilities automatically
```

## Architecture

The skills system consists of:

### Core Infrastructure
- **`SkillBase`** - Abstract base class for all skills with parameter support
- **`SkillManager`** - Handles loading/unloading and lifecycle management with parameters
- **`AgentBase.add_skill()`** - Simple method to add skills to agents with optional parameters

### Discovery & Registry  
- **`SkillRegistry`** - Auto-discovers skills from the `skills/` directory
- **Auto-discovery** - Skills are found automatically on import
- **Validation** - Checks dependencies and environment variables

### Built-in Skills
- **`web_search`** - Google Custom Search API integration with web scraping
- **`datetime`** - Current date/time information with timezone support
- **`math`** - Basic mathematical calculations

## Available Skills

### Web Search (`web_search`)
Search the internet and extract content from web pages.

**Requirements:**
- Environment variables: `GOOGLE_SEARCH_API_KEY`, `GOOGLE_SEARCH_ENGINE_ID`
- Packages: `beautifulsoup4`, `requests`

**Parameters:**
- `num_results` (default: 1) - Number of search results to retrieve (1-10)
- `delay` (default: 0) - Delay in seconds between web requests

**Tools provided:**
- `web_search(query, num_results)` - Search and scrape web content

**Usage examples:**
```python
# Default: fast single result
agent.add_skill("web_search")

# Custom: multiple results with delay
agent.add_skill("web_search", {
    "num_results": 3,
    "delay": 0.5
})

# Speed optimized: single result, no delay
agent.add_skill("web_search", {
    "num_results": 1,
    "delay": 0
})
```

### Date/Time (`datetime`)  
Get current date and time information.

**Requirements:**
- Packages: `pytz`

**Parameters:** None (no configurable parameters)

**Tools provided:**
- `get_current_time(timezone)` - Current time in any timezone
- `get_current_date(timezone)` - Current date in any timezone

### Math (`math`)
Perform mathematical calculations.

**Requirements:** None

**Parameters:** None (no configurable parameters)

**Tools provided:**
- `calculate(expression)` - Evaluate mathematical expressions safely

### Native Vector Search (`native_vector_search`)
Search local document collections using vector similarity and keyword search.

**Requirements:**
- Packages: `sentence-transformers`, `scikit-learn`, `numpy`
- Install with: `pip install signalwire-agents[search]`

**Parameters:**
- `tool_name` (default: "search_knowledge") - Custom name for the search tool
- `index_file` (optional) - Path to local `.swsearch` index file
- `remote_url` (optional) - URL of remote search server
- `index_name` (default: "default") - Index name on remote server
- `build_index` (default: False) - Auto-build index if missing
- `source_dir` (optional) - Source directory for auto-building
- `count` (default: 3) - Number of search results to return
- `distance_threshold` (default: 0.0) - Minimum similarity score
- `response_prefix` (optional) - Text to prepend to responses
- `response_postfix` (optional) - Text to append to responses

**Tools provided:**
- `search_knowledge(query, count)` - Search documents with hybrid vector/keyword search

**Usage examples:**
```python
# Local mode with auto-build from concepts guide
agent.add_skill("native_vector_search", {
    "tool_name": "search_docs",
    "build_index": True,
    "source_dir": "./docs",  # Will build from directory
    "index_file": "concepts.swsearch"
})

# Or build from specific concepts guide file
agent.add_skill("native_vector_search", {
    "tool_name": "search_concepts",
    "index_file": "concepts.swsearch"  # Pre-built from concepts guide
})

# Remote mode
agent.add_skill("native_vector_search", {
    "remote_url": "http://localhost:8001",
    "index_name": "knowledge"
})

# Multiple instances for different document collections
agent.add_skill("native_vector_search", {
    "tool_name": "search_examples",
    "index_file": "examples.swsearch"
})
```

For complete documentation, see [Search Overview](search_overview.md).

### SWML Transfer (`swml_transfer`)
Transfer calls between agents using pattern matching.

**Requirements:** None (no additional packages or environment variables required)

**Parameters:**
- `tool_name` (default: "transfer_call") - Custom name for the transfer function
- `description` (default: "Transfer call based on pattern matching") - Tool description
- `parameter_name` (default: "transfer_type") - Name of the parameter for the transfer function
- `parameter_description` (default: "The type of transfer to perform") - Parameter description
- `transfers` (required) - Dictionary mapping regex patterns to transfer configurations:
  - Pattern (key): Regex pattern to match (e.g., "/sales/i")
  - Configuration (value): Dictionary with:
    - `url` (required): Transfer destination URL
    - `message` (optional): Pre-transfer message
    - `return_message` (optional): Post-transfer message
    - `post_process` (optional, default: True): Enable post-processing
- `default_message` (default: "Please specify a valid transfer type.") - Message when no pattern matches
- `default_post_process` (default: False) - Post-processing flag for default case
- `required_fields` (default: {}) - Object mapping field names to descriptions for data collection before transfer

**Tools provided:**
- `transfer_call(transfer_type, ...required_fields)` (or custom tool_name) - Transfer based on pattern matching with optional required fields

**Usage examples:**
```python
# Simple transfer between departments
agent.add_skill("swml_transfer", {
    "tool_name": "transfer_to_department",
    "transfers": {
        "/sales/i": {
            "url": "https://example.com/sales",
            "message": "Transferring to sales...",
            "return_message": "Sales transfer complete."
        },
        "/support/i": {
            "url": "https://example.com/support",
            "message": "Transferring to support...",
            "return_message": "Support transfer complete."
        }
    }
})

# Multiple instances for different transfer types
agent.add_skill("swml_transfer", {
    "tool_name": "route_call",
    "parameter_name": "department",
    "transfers": {
        "/sales|billing/i": {
            "url": "https://api.company.com/sales",
            "message": "Connecting to sales team...",
            "post_process": True
        },
        "/technical|support/i": {
            "url": "https://api.company.com/support",
            "message": "Connecting to support team...",
            "post_process": True
        }
    },
    "default_message": "Would you like sales or support?"
})
```

## Usage Examples

### Basic Usage
```python
from signalwire_agents import AgentBase

# Create agent and add skills
agent = AgentBase("Assistant", route="/assistant")
agent.add_skill("datetime")
agent.add_skill("math") 
agent.add_skill("web_search")  # Uses defaults: 1 result, no delay

# Start the agent
agent.run()
```

### Skills with Custom Parameters
```python
from signalwire_agents import AgentBase

# Create agent
agent = AgentBase("Research Assistant", route="/research")

# Add web search optimized for research (more results)
agent.add_skill("web_search", {
    "num_results": 5,   # Get more comprehensive results
    "delay": 1.0        # Be respectful to websites
})

# Add other skills without parameters
agent.add_skill("datetime")
agent.add_skill("math")

# Start the agent
agent.run()
```

### Different Parameter Configurations
```python
# Speed-optimized for quick responses
agent.add_skill("web_search", {
    "num_results": 1,
    "delay": 0
})

# Comprehensive research mode
agent.add_skill("web_search", {
    "num_results": 5,
    "delay": 1.0
})

# Balanced approach
agent.add_skill("web_search", {
    "num_results": 3,
    "delay": 0.5
})
```

### Check Available Skills
```python
from signalwire_agents.skills.registry import skill_registry

# List all discovered skills
for skill in skill_registry.list_skills():
    print(f"- {skill['name']}: {skill['description']}")
    if skill['required_env_vars']:
        print(f"  Requires: {', '.join(skill['required_env_vars'])}")
```

### Runtime Skill Management
```python
agent = AgentBase("Dynamic Agent")

# Add skills with different configurations
agent.add_skill("math")
agent.add_skill("datetime")
agent.add_skill("web_search", {"num_results": 2, "delay": 0.3})

# Check what's loaded
print("Loaded skills:", agent.list_skills())

# Remove a skill
agent.remove_skill("math")

# Check if specific skill is loaded
if agent.has_skill("datetime"):
    print("Date/time capabilities available")
```

## Creating Custom Skills

Create a new skill by extending `SkillBase` with parameter support:

```python
# signalwire_agents/skills/my_skill/skill.py
from signalwire_agents.core.skill_base import SkillBase
from signalwire_agents.core.function_result import SwaigFunctionResult

class MyCustomSkill(SkillBase):
    SKILL_NAME = "my_skill"
    SKILL_DESCRIPTION = "Does something awesome with configurable parameters"
    SKILL_VERSION = "1.0.0"
    REQUIRED_PACKAGES = ["requests"]  # Optional
    REQUIRED_ENV_VARS = ["API_KEY"]   # Optional
    
    def setup(self) -> bool:
        """Initialize the skill with parameters"""
        if not self.validate_env_vars() or not self.validate_packages():
            return False
            
        # Use parameters with defaults
        self.max_items = self.params.get('max_items', 10)
        self.timeout = self.params.get('timeout', 30)
        self.retry_count = self.params.get('retry_count', 3)
        
        return True
        
    def register_tools(self) -> None:
        """Register SWAIG tools with the agent"""
        self.define_tool(
            name="my_function",
            description=f"Does something cool (max {self.max_items} items)",
            parameters={
                "input": {
                    "type": "string",
                    "description": "Input parameter"
                }
            },
            handler=self._my_handler
        )
    
    def _my_handler(self, args, raw_data):
        """Handle the tool call using configured parameters"""
        # Use self.max_items, self.timeout, self.retry_count in your logic
        return SwaigFunctionResult(f"Processed with max_items={self.max_items}")
        
    def get_hints(self):
        """Speech recognition hints"""
        return ["custom", "skill", "awesome"]
        
    def get_prompt_sections(self):
        """Prompt sections to add to agent"""
        return [{
            "title": "Custom Capability",
            "body": f"You can do custom things with my_skill (configured for {self.max_items} items)."
        }]
```

The skill will be automatically discovered and available as:
```python
# Use defaults
agent.add_skill("my_skill")

# Use custom parameters
agent.add_skill("my_skill", {
    "max_items": 20,
    "timeout": 60,
    "retry_count": 5
})
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install pytz beautifulsoup4 requests
   ```

2. **Run the demo:**
   ```bash
   python examples/skills_demo.py
   ```

3. **For web search, set environment variables:**
   ```bash
   export GOOGLE_SEARCH_API_KEY="your_api_key"
   export GOOGLE_SEARCH_ENGINE_ID="your_engine_id"
   ```

## Testing

Test the skills system with parameters:

```bash
python3 -c "
from signalwire_agents import AgentBase
from signalwire_agents.skills.registry import skill_registry

# Show discovered skills
print('Available skills:', [s['name'] for s in skill_registry.list_skills()])

# Create agent and load skills with parameters
agent = AgentBase('Test', route='/test')
agent.add_skill('datetime')
agent.add_skill('math')
agent.add_skill('web_search', {'num_results': 2, 'delay': 0.5})

print('Loaded skills:', agent.list_skills())
print('Skills system with parameters working!')
"
```

## Benefits

- **One-liner integration** - `agent.add_skill("skill_name")`
- **Configurable parameters** - `agent.add_skill("skill_name", {"param": "value"})`
- **Automatic discovery** - Drop skills in the directory and they're available
- **Dependency validation** - Checks packages and environment variables
- **Modular architecture** - Skills are self-contained and reusable
- **Extensible** - Easy to create custom skills with parameters
- **Clean separation** - Skills don't interfere with each other
- **Performance tuning** - Configure skills for speed vs. comprehensiveness

## Migration Guide

**Before (manual implementation):**
```python
# Had to manually implement every capability
class WebSearchAgent(AgentBase):
    def __init__(self):
        super().__init__("WebSearchAgent")
        self.setup_google_search()
        self.define_tool("web_search", ...)
        # Lots of manual code...
```

**After (skills system with parameters):**
```python
# Simple one-liner with custom configuration
agent = AgentBase("WebSearchAgent")
agent.add_skill("web_search", {
    "num_results": 3,  # Get more results
    "delay": 0.5       # Be respectful to servers
})
# Done! Full web search capability with custom settings.
```

The skills system makes SignalWire agents more modular, maintainable, and configurable. 