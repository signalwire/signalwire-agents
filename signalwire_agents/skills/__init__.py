"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
SignalWire Agent Skills Package

This package contains built-in skills for SignalWire agents.
Skills are automatically discovered from subdirectories.
"""

# Import the registry to make it available
from .registry import skill_registry

# Import SkillBase for convenience
from signalwire_agents.core.skill_base import SkillBase

# Trigger skill discovery on import
# skill_registry.discover_skills()

__all__ = ["skill_registry", "SkillBase"] 