"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import ast
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
        
    # Allowed AST node types for safe math evaluation
    _SAFE_OPERATORS = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }

    def _safe_eval(self, node):
        """Recursively evaluate an AST node, allowing only safe math operations."""
        if isinstance(node, ast.Expression):
            return self._safe_eval(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self._SAFE_OPERATORS:
                raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
            left = self._safe_eval(node.left)
            right = self._safe_eval(node.right)
            # Cap exponent to prevent resource exhaustion
            if op_type is ast.Pow and isinstance(right, (int, float)) and right > 1000:
                raise ValueError("Exponent too large (maximum is 1000)")
            return self._SAFE_OPERATORS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self._SAFE_OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = self._safe_eval(node.operand)
            return self._SAFE_OPERATORS[op_type](operand)
        else:
            raise ValueError(f"Unsupported expression node: {type(node).__name__}")

    def _calculate_handler(self, args, raw_data):
        """Handler for calculate tool"""
        expression = args.get("expression", "").strip()

        if not expression:
            return SwaigFunctionResult("Please provide a mathematical expression to calculate.")

        try:
            # Parse the expression into an AST
            tree = ast.parse(expression, mode='eval')
            # Evaluate using safe AST walker
            result = self._safe_eval(tree)

            return SwaigFunctionResult(f"{expression} = {result}")

        except ZeroDivisionError:
            return SwaigFunctionResult("Error: Division by zero is not allowed.")
        except (ValueError, SyntaxError, TypeError) as e:
            return SwaigFunctionResult(
                f"Error: Invalid expression. Only numbers and basic math operators (+, -, *, /, %, **, parentheses) are allowed."
            )
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