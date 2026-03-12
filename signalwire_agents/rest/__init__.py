"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
SignalWire REST API client module.
"""

from .client import SignalWireClient
from ._base import SignalWireRestError

__all__ = ["SignalWireClient", "SignalWireRestError"]
