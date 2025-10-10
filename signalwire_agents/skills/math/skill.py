"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import re
from typing import List, Dict, Any

from signalwire_agents.core.skill_base import SkillBase
from signalwire_agents.core.function_result import SwaigFunctionResult

class MathSkill(SkillBase):
    """Provides basic mathematical calculation capabilities"""
    
    SKILL_NAME = "math"
    SKILL_DESCRIPTION = "Perform basic mathematical calculations"
    SKILL_VERSION = "1.0.0"
    REQUIRED_PACKAGES = []
    REQUIRED_ENV_VARS = []
    
    def setup(self) -> bool:
        """Setup the math skill"""
        return True
        
    def register_tools(self) -> None:
        """Register math tools with the agent"""
        
        self.define_tool(
            name="calculate",
            description="Perform a mathematical calculation with basic operations (+, -, *, /, %, **)",
            parameters={
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4', '(10 + 5) / 3')"
                }
            },
            handler=self._calculate_handler
        )
        
    def _calculate_handler(self, args, raw_data):
        """Handler for calculate tool"""
        expression = args.get("expression", "").strip()
        
        if not expression:
            return SwaigFunctionResult("Please provide a mathematical expression to calculate.")
        
        # Security: only allow safe mathematical operations
        safe_chars = re.compile(r'^[0-9+\-*/().\s%**]+$')
        if not safe_chars.match(expression):
            return SwaigFunctionResult(
                "Invalid expression. Only numbers and basic math operators (+, -, *, /, %, **, parentheses) are allowed."
            )
        
        try:
            # Evaluate the expression safely
            result = eval(expression, {"__builtins__": {}}, {})
            
            return SwaigFunctionResult(f"{expression} = {result}")
            
        except ZeroDivisionError:
            return SwaigFunctionResult("Error: Division by zero is not allowed.")
        except Exception as e:
            return SwaigFunctionResult(f"Error calculating '{expression}': Invalid expression")
        
    def get_hints(self) -> List[str]:
        """Return speech recognition hints"""
        # Currently no hints provided, but you could add them like:
        # return [
        #     "calculate", "math", "plus", "minus", "times", "multiply", 
        #     "divide", "equals", "percent", "power", "squared"
        # ]
        return []
        
    def get_prompt_sections(self) -> List[Dict[str, Any]]:
        """Return prompt sections to add to agent"""
        return [
            {
                "title": "Mathematical Calculations",
                "body": "You can perform mathematical calculations for users.",
                "bullets": [
                    "Use the calculate tool for any math expressions",
                    "Supports basic operations: +, -, *, /, %, ** (power)",
                    "Can handle parentheses for complex expressions"
                ]
            }
        ]
    
    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get the parameter schema for the math skill
        
        The math skill has no custom parameters - it inherits only
        the base parameters from SkillBase.
        """
        # Get base schema from parent
        schema = super().get_parameter_schema()
        
        # No additional parameters for math skill
        
        return schema 