#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
MCP-SWAIG Gateway Package

HTTP/HTTPS server that bridges MCP servers with SignalWire SWAIG functions.
"""

from .gateway_service import MCPGateway, main
from .session_manager import Session, SessionManager
from .mcp_manager import MCPService, MCPClient, MCPManager

__all__ = [
    'MCPGateway',
    'main',
    'Session',
    'SessionManager',
    'MCPService',
    'MCPClient',
    'MCPManager',
]
