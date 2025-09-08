"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Weather API Skill

A configurable skill for getting weather information from WeatherAPI.com with customizable
temperature units and TTS-friendly responses.
"""

from typing import Dict, Any, List
from signalwire_agents.core import SwaigFunctionResult
from signalwire_agents.core.skill_base import SkillBase


class WeatherApiSkill(SkillBase):
    """
    Skill for getting weather information from WeatherAPI.com.
    
    Provides current weather data with configurable temperature units and
    TTS-optimized natural language responses.
    
    Configuration:
    - tool_name: Custom name for the generated SWAIG function
    - api_key: WeatherAPI.com API key
    - temperature_unit: "fahrenheit" or "celsius" for temperature display
    
    Example:
        agent.add_skill("weather_api", {
            "tool_name": "get_weather",
            "api_key": "your_weatherapi_key",
            "temperature_unit": "fahrenheit"
        })
    """
    
    SKILL_NAME = "weather_api"
    SKILL_DESCRIPTION = "Get current weather information from WeatherAPI.com"
    SUPPORTS_MULTIPLE_INSTANCES = False
    REQUIRED_ENV_VARS = []  # API key can be passed via params
    
    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Get parameter schema for weather API skill"""
        schema = super().get_parameter_schema()
        schema.update({
            "api_key": {
                "type": "string",
                "description": "WeatherAPI.com API key",
                "required": True,
                "hidden": True,
                "env_var": "WEATHER_API_KEY"
            },
            "tool_name": {
                "type": "string",
                "description": "Custom name for the weather tool",
                "default": "get_weather",
                "required": False
            },
            "temperature_unit": {
                "type": "string",
                "description": "Temperature unit to display",
                "default": "fahrenheit",
                "required": False,
                "enum": ["fahrenheit", "celsius"]
            }
        })
        return schema
    
    def __init__(self, agent, params: Dict[str, Any] = None):
        """
        Initialize the skill with configuration parameters.
        
        Args:
            agent: The agent instance this skill belongs to
            params: Configuration dictionary containing:
                - tool_name: Custom tool name (default: "get_weather")
                - api_key: WeatherAPI.com API key (required)
                - temperature_unit: "fahrenheit" or "celsius" (default: "fahrenheit")
        """
        super().__init__(agent, params)
        
        # Extract configuration
        self.tool_name = self.params.get('tool_name', 'get_weather')
        self.api_key = self.params.get('api_key')
        self.temperature_unit = self.params.get('temperature_unit', 'fahrenheit')
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate the skill configuration."""
        # Validate API key
        if not self.api_key or not isinstance(self.api_key, str):
            raise ValueError("api_key parameter is required and must be a non-empty string")
        
        # Validate temperature unit
        if self.temperature_unit not in ['fahrenheit', 'celsius']:
            raise ValueError("temperature_unit must be either 'fahrenheit' or 'celsius'")
    
    def setup(self) -> bool:
        """
        Setup the skill - validates API key is available.
        
        Returns:
            True if setup successful
        """
        # API key validation already done in _validate_config
        return True
    
    def register_tools(self) -> None:
        """Register SWAIG tools with the agent"""
        tools = self.get_tools()
        for tool in tools:
            # Merge any swaig_fields from params into the tool
            if self.swaig_fields:
                tool.update(self.swaig_fields)
            self.agent.register_swaig_function(tool)
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Generate the SWAIG tool with DataMap webhook.
        
        Returns:
            List containing the generated tool configuration
        """
        # Determine temperature fields based on unit
        if self.temperature_unit == 'fahrenheit':
            temp_field = 'temp_f'
            feels_like_field = 'feelslike_f'
            unit_name = 'Fahrenheit'
        else:
            temp_field = 'temp_c'
            feels_like_field = 'feelslike_c'
            unit_name = 'Celsius'
        
        # Create TTS-friendly response instruction
        response_instruction = (
            f"Tell the user the current weather conditions. "
            f"Express all temperatures in {unit_name} using natural language numbers "
            f"without abbreviations or symbols for clear text-to-speech pronunciation. "
            f"For example, say 'seventy two degrees {unit_name}' instead of '72F' or '72°F'. "
            f"Include the condition, current temperature, wind direction and speed, "
            f"cloud coverage percentage, and what the temperature feels like."
        )
        
        # Build the weather data template
        weather_template = (
            f"{response_instruction} "
            f"Current conditions: ${{current.condition.text}}. "
            f"Temperature: ${{current.{temp_field}}} degrees {unit_name}. "
            f"Wind: ${{current.wind_dir}} at ${{current.wind_mph}} miles per hour. "
            f"Cloud coverage: ${{current.cloud}} percent. "
            f"Feels like: ${{current.{feels_like_field}}} degrees {unit_name}."
        )
        
        # Create the tool configuration with DataMap webhook
        tool = {
            "function": self.tool_name,
            "description": f"Get current weather information for any location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city, state, country, or location to get weather for"
                    }
                },
                "required": ["location"]
            },
            "data_map": {
                "webhooks": [
                    {
                        "url": f"https://api.weatherapi.com/v1/current.json?key={self.api_key}&q=${{lc:enc:args.location}}&aqi=no",
                        "method": "GET",
                        "output": SwaigFunctionResult(weather_template).to_dict()
                    }
                ],
                "error_keys": ["error"],
                "output": SwaigFunctionResult(
                    "Sorry, I cannot get weather information right now. Please try again later or check if the location name is correct."
                ).to_dict()
            }
        }
        
        return [tool] 
