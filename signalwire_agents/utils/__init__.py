"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

from .schema_utils import SchemaUtils
from .url_validator import validate_url
from signalwire_agents.core.logging_config import get_execution_mode

def is_serverless_mode() -> bool:
    """
    Check if running in any serverless environment.

    Returns:
        bool: True if in serverless mode, False if in server mode
    """
    return get_execution_mode() != 'server'

__all__ = ["SchemaUtils", "get_execution_mode", "is_serverless_mode", "validate_url"]

