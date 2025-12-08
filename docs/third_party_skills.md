# Third-Party Skills Integration Guide

This guide explains how to create and integrate third-party skills with the SignalWire AI Agents SDK. The SDK supports multiple methods for loading external skills, making it easy to extend agent capabilities without modifying the core SDK.

## Overview

Third-party skills can be integrated using four different methods:

1. **Direct Registration** - Register skill classes programmatically
2. **Directory Registration** - Add directories containing skill collections
3. **Python Entry Points** - Install skills as Python packages
4. **Environment Variables** - Configure skill paths via environment

All third-party skills are discovered and indexed the same way as built-in skills, appearing in `list_skills_with_params()` output with their parameter schemas.

## Creating a Third-Party Skill

Third-party skills follow the same structure as built-in skills. Here's a minimal example:

```python
# my_weather_skill/skill.py
from signalwire_agents.core.skill_base import SkillBase
from signalwire_agents.core.function_result import SwaigFunctionResult
from typing import Dict, Any, List

class WeatherSkill(SkillBase):
    """Custom weather information skill"""
    
    SKILL_NAME = "weather"
    SKILL_DESCRIPTION = "Get weather information for any location"
    SKILL_VERSION = "1.0.0"
    REQUIRED_PACKAGES = ["requests"]
    REQUIRED_ENV_VARS = []
    
    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Define configuration parameters"""
        schema = super().get_parameter_schema()
        
        schema.update({
            "api_key": {
                "type": "string",
                "description": "Weather API key",
                "required": True,
                "hidden": True,
                "env_var": "WEATHER_API_KEY"
            },
            "units": {
                "type": "string",
                "description": "Temperature units",
                "default": "celsius",
                "required": False,
                "enum": ["celsius", "fahrenheit", "kelvin"]
            },
            "cache_timeout": {
                "type": "integer",
                "description": "Cache timeout in seconds",
                "default": 300,
                "required": False,
                "min": 0,
                "max": 3600
            }
        })
        
        return schema
    
    def setup(self) -> bool:
        """Initialize the skill"""
        if not self.validate_packages():
            return False
            
        self.api_key = self.params.get('api_key')
        if not self.api_key:
            self.logger.error("Weather API key is required")
            return False
            
        self.units = self.params.get('units', 'celsius')
        self.cache_timeout = self.params.get('cache_timeout', 300)
        
        return True
    
    def register_tools(self) -> None:
        """Register weather tools with the agent"""
        self.define_tool(
            name="get_weather",
            description="Get current weather for a location",
            parameters={
                "location": {
                    "type": "string",
                    "description": "City name or coordinates"
                }
            },
            handler=self._get_weather_handler
        )
    
    def _get_weather_handler(self, args, raw_data):
        """Handle weather requests"""
        location = args.get('location', '').strip()
        
        if not location:
            return SwaigFunctionResult("Please provide a location")
        
        # Implementation would call weather API here
        # This is just an example
        return SwaigFunctionResult(
            f"The weather in {location} is sunny and 22°{self.units[0].upper()}"
        )
```

## Integration Methods

### Method 1: Direct Registration

Register individual skill classes programmatically:

```python
from signalwire_agents import AgentBase, register_skill
from my_weather_skill import WeatherSkill

# Register the skill globally
register_skill(WeatherSkill)

# Now use it in any agent
class MyAgent(AgentBase):
    def __init__(self):
        super().__init__(name="my-agent")
        
        # Add the registered skill
        self.add_skill("weather", {
            "api_key": "your-api-key",
            "units": "fahrenheit"
        })
```

### Method 2: Directory Registration

Register directories containing multiple skills:

```python
from signalwire_agents import add_skill_directory

# Add a directory of custom skills
add_skill_directory('/opt/custom_skills')

# Directory structure should be:
# /opt/custom_skills/
#   weather/
#     skill.py      # Contains WeatherSkill class
#   stock_market/
#     skill.py      # Contains StockMarketSkill class
#   translation/
#     skill.py      # Contains TranslationSkill class

# Now use any skill from the directory
agent.add_skill("weather", {"api_key": "..."})
agent.add_skill("stock_market", {"api_key": "..."})
```

### Method 3: Python Entry Points

Create installable skill packages using setuptools entry points:

```python
# setup.py for your skill package
from setuptools import setup, find_packages

setup(
    name="my-signalwire-skills",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "signalwire-agents",
        "requests",
    ],
    entry_points={
        'signalwire_agents.skills': [
            'weather = my_skills.weather:WeatherSkill',
            'stock = my_skills.stock:StockMarketSkill',
            'translate = my_skills.translate:TranslationSkill',
        ]
    }
)
```

After installation, skills are automatically available:

```bash
pip install my-signalwire-skills
```

```python
# Skills are automatically discovered
agent.add_skill("weather", {"api_key": "..."})
```

### Method 4: Environment Variable

Set the `SIGNALWIRE_SKILL_PATHS` environment variable:

```bash
# Single directory
export SIGNALWIRE_SKILL_PATHS=/opt/my_skills

# Multiple directories (colon-separated)
export SIGNALWIRE_SKILL_PATHS=/opt/my_skills:/home/user/custom_skills
```

Skills in these directories are automatically discovered:

```python
# No registration needed - skills are found automatically
agent.add_skill("weather", {"api_key": "..."})
```

## Directory Structure

Skills loaded from directories must follow this structure:

```
my_skills_directory/
├── weather/                 # Skill directory (matches SKILL_NAME)
│   ├── skill.py            # Required: Contains skill class
│   ├── __init__.py         # Optional: Makes it a package
│   └── README.md           # Optional: Documentation
├── translation/
│   ├── skill.py
│   └── resources/          # Optional: Additional files
│       └── languages.json
└── stock_market/
    └── skill.py
```

## Skill Discovery and Schema

Third-party skills are fully integrated with the SDK's discovery system:

```python
from signalwire_agents import list_skills_with_params

# Get all skills including third-party ones
all_skills = list_skills_with_params()

# Third-party skills include source information
print(all_skills['weather'])
# Output:
{
    "name": "weather",
    "description": "Get weather information for any location",
    "version": "1.0.0",
    "supports_multiple_instances": False,
    "required_packages": ["requests"],
    "required_env_vars": [],
    "parameters": {
        "api_key": {
            "type": "string",
            "description": "Weather API key",
            "required": True,
            "hidden": True,
            "env_var": "WEATHER_API_KEY"
        },
        "units": {
            "type": "string",
            "description": "Temperature units",
            "default": "celsius",
            "required": False,
            "enum": ["celsius", "fahrenheit", "kelvin"]
        }
    },
    "source": "external"  # Shows it's a third-party skill
}
```

## Best Practices

### 1. Skill Naming

- Use lowercase, underscore-separated names
- Choose unique names to avoid conflicts with built-in skills
- Match directory name to `SKILL_NAME` for directory-based loading

### 2. Parameter Design

- Always implement `get_parameter_schema()` for GUI compatibility
- Mark sensitive parameters as `hidden`
- Provide sensible defaults
- Use `env_var` for parameters that can come from environment

### 3. Error Handling

```python
def setup(self) -> bool:
    """Proper setup with error handling"""
    # Validate packages
    if not self.validate_packages():
        return False
    
    # Validate required parameters
    if not self.params.get('api_key'):
        self.logger.error("API key is required")
        return False
    
    # Test connectivity
    try:
        self._test_api_connection()
    except Exception as e:
        self.logger.error(f"Failed to connect to API: {e}")
        return False
    
    return True
```

### 4. Documentation

Include a README.md in your skill directory:

```markdown
# Weather Skill

Provides weather information for any location.

## Configuration

- `api_key` (required): Your weather API key
- `units` (optional): Temperature units (celsius, fahrenheit, kelvin)
- `cache_timeout` (optional): Cache timeout in seconds

## Usage

```python
agent.add_skill("weather", {
    "api_key": "your-api-key",
    "units": "fahrenheit"
})
```
```

## Advanced Features

### Multiple Instances

Support multiple instances of your skill:

```python
class WeatherSkill(SkillBase):
    SKILL_NAME = "weather"
    SUPPORTS_MULTIPLE_INSTANCES = True  # Enable multiple instances
    
    def get_instance_key(self) -> str:
        """Create unique key for this instance"""
        service = self.params.get('service', 'default')
        return f"{self.SKILL_NAME}_{service}"
```

Usage:

```python
# Add multiple weather services
agent.add_skill("weather", {
    "tool_name": "openweather",
    "service": "openweathermap",
    "api_key": "key1"
})

agent.add_skill("weather", {
    "tool_name": "weatherapi", 
    "service": "weatherapi",
    "api_key": "key2"
})
```

### Dynamic Tool Names

Customize tool names for better agent prompts:

```python
def register_tools(self) -> None:
    tool_name = self.params.get('tool_name', 'get_weather')
    
    self.define_tool(
        name=tool_name,
        description=f"Get weather using {self.params.get('service', 'default')}",
        parameters={...},
        handler=self._weather_handler
    )
```

### Skill Dependencies

Load skills that depend on other skills:

```python
def setup(self) -> bool:
    # Check if required skill is available
    if not self.agent.skill_manager.has_skill("translation"):
        self.logger.error("This skill requires the translation skill")
        return False
    
    return True
```

## Testing Third-Party Skills

Test your skills before distribution:

```python
# test_my_skill.py
import unittest
from signalwire_agents import AgentBase
from my_weather_skill import WeatherSkill

class TestWeatherSkill(unittest.TestCase):
    def setUp(self):
        self.agent = AgentBase(name="test-agent")
        
    def test_skill_registration(self):
        # Test direct registration
        from signalwire_agents import register_skill
        register_skill(WeatherSkill)
        
        # Test adding skill
        success, error = self.agent.add_skill("weather", {
            "api_key": "test-key"
        })
        self.assertTrue(success)
        
    def test_parameter_schema(self):
        schema = WeatherSkill.get_parameter_schema()
        self.assertIn("api_key", schema)
        self.assertTrue(schema["api_key"]["required"])
        self.assertTrue(schema["api_key"]["hidden"])
```

## Troubleshooting

### Skill Not Found

If your skill isn't being discovered:

1. Check the skill directory structure
2. Verify `SKILL_NAME` matches the directory name
3. Ensure `skill.py` exists and contains a valid skill class
4. Check logs for loading errors

### Import Errors

For skills with relative imports:

```python
# Use absolute imports in skill.py
from my_skills.weather.utils import parse_temperature

# Or handle import errors gracefully
try:
    from .utils import parse_temperature
except ImportError:
    from utils import parse_temperature
```

### Environment Variables

Debug environment variable loading:

```python
import os
print(f"Skill paths: {os.environ.get('SIGNALWIRE_SKILL_PATHS', 'Not set')}")

from signalwire_agents.skills.registry import skill_registry
sources = skill_registry.list_all_skill_sources()
print(f"External skills: {sources['external_paths']}")
```

## Example: Complete Third-Party Skill Package

Here's a complete example of a distributable skill package:

```
my-signalwire-skills/
├── setup.py
├── README.md
├── requirements.txt
├── my_signalwire_skills/
│   ├── __init__.py
│   ├── weather/
│   │   ├── __init__.py
│   │   ├── skill.py
│   │   └── utils.py
│   └── translation/
│       ├── __init__.py
│       └── skill.py
└── tests/
    ├── __init__.py
    ├── test_weather.py
    └── test_translation.py
```

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="my-signalwire-skills",
    version="1.0.0",
    author="Your Name",
    description="Custom skills for SignalWire AI Agents",
    packages=find_packages(),
    install_requires=[
        "signalwire-agents>=1.0.12",
        "requests>=2.25.0",
    ],
    entry_points={
        'signalwire_agents.skills': [
            'weather = my_signalwire_skills.weather.skill:WeatherSkill',
            'translate = my_signalwire_skills.translation.skill:TranslationSkill',
        ]
    },
    python_requires='>=3.7',
)
```

Install and use:

```bash
pip install git+https://github.com/yourname/my-signalwire-skills.git
```

```python
from signalwire_agents import AgentBase

agent = AgentBase(name="my-agent")
agent.add_skill("weather", {"api_key": "..."})
agent.add_skill("translate", {"api_key": "..."})