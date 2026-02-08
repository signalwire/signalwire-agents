"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for SWML handler module
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional, Tuple

from signalwire_agents.core.swml_handler import (
    SWMLVerbHandler,
    AIVerbHandler,
    VerbHandlerRegistry
)


class MockVerbHandler(SWMLVerbHandler):
    """Mock implementation of SWMLVerbHandler for testing"""
    
    def __init__(self, verb_name: str = "mock_verb"):
        self.verb_name = verb_name
    
    def get_verb_name(self) -> str:
        return self.verb_name
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        # Simple validation - require 'required_field'
        if 'required_field' not in config:
            return False, ["Missing required_field"]
        return True, []
    
    def build_config(self, **kwargs) -> Dict[str, Any]:
        return {"mock_config": True, **kwargs}


class TestSWMLVerbHandlerInterface:
    """Test SWMLVerbHandler abstract interface"""
    
    def test_abstract_methods_exist(self):
        """Test that SWMLVerbHandler defines required abstract methods"""
        # Should not be able to instantiate abstract class directly
        with pytest.raises(TypeError):
            SWMLVerbHandler()
    
    def test_mock_implementation(self):
        """Test mock implementation of SWMLVerbHandler"""
        handler = MockVerbHandler("test_verb")
        
        # Test get_verb_name
        assert handler.get_verb_name() == "test_verb"
        
        # Test validate_config
        valid_config = {"required_field": "value"}
        is_valid, errors = handler.validate_config(valid_config)
        assert is_valid is True
        assert errors == []
        
        invalid_config = {"other_field": "value"}
        is_valid, errors = handler.validate_config(invalid_config)
        assert is_valid is False
        assert "Missing required_field" in errors
        
        # Test build_config
        config = handler.build_config(param1="value1", param2="value2")
        assert config["mock_config"] is True
        assert config["param1"] == "value1"
        assert config["param2"] == "value2"


class TestAIVerbHandler:
    """Test AIVerbHandler implementation"""
    
    def test_initialization(self):
        """Test AIVerbHandler initialization"""
        handler = AIVerbHandler()
        assert handler.get_verb_name() == "ai"
    
    def test_validate_config_valid_prompt_text(self):
        """Test validation with valid prompt text configuration"""
        handler = AIVerbHandler()
        
        config = {
            "prompt": {
                "text": "You are a helpful assistant"
            }
        }
        
        is_valid, errors = handler.validate_config(config)
        assert is_valid is True
        assert errors == []
    
    def test_validate_config_valid_prompt_pom(self):
        """Test validation with valid prompt POM configuration"""
        handler = AIVerbHandler()
        
        config = {
            "prompt": {
                "pom": [
                    {"title": "Section 1", "body": "Content 1"},
                    {"title": "Section 2", "body": "Content 2"}
                ]
            }
        }
        
        is_valid, errors = handler.validate_config(config)
        assert is_valid is True
        assert errors == []
    
    def test_validate_config_valid_contexts(self):
        """Test validation with contexts requires base prompt (text or pom)"""
        handler = AIVerbHandler()

        # Contexts alone are not sufficient - need text or pom as base prompt
        config = {
            "prompt": {
                "contexts": {
                    "context1": {
                        "steps": [
                            {"step": "greeting", "content": "Hello"}
                        ]
                    }
                }
            }
        }

        is_valid, errors = handler.validate_config(config)
        assert is_valid is False
        assert "'prompt' must contain either 'text' or 'pom' as base prompt" in errors

    def test_validate_config_text_with_contexts(self):
        """Test validation with text and contexts combined"""
        handler = AIVerbHandler()

        config = {
            "prompt": {
                "text": "You are a helpful assistant",
                "contexts": {
                    "context1": {
                        "steps": [
                            {"step": "greeting", "content": "Hello"}
                        ]
                    }
                }
            }
        }

        is_valid, errors = handler.validate_config(config)
        assert is_valid is True
        assert errors == []
    
    def test_validate_config_missing_prompt(self):
        """Test validation fails when prompt is missing"""
        handler = AIVerbHandler()
        
        config = {
            "post_prompt": {"text": "Summary"}
        }
        
        is_valid, errors = handler.validate_config(config)
        assert is_valid is False
        assert "Missing required field 'prompt'" in errors
    
    def test_validate_config_both_prompt_options(self):
        """Test validation fails when multiple prompt options are specified"""
        handler = AIVerbHandler()
        
        config = {
            "prompt": {
                "text": "You are helpful",
                "pom": [{"title": "Section"}]
            }
        }
        
        is_valid, errors = handler.validate_config(config)
        assert is_valid is False
        assert "'prompt' can only contain one of: 'text' or 'pom' (mutually exclusive)" in errors
    
    def test_validate_config_invalid_prompt_structure(self):
        """Test validation fails with invalid prompt structure"""
        handler = AIVerbHandler()
        
        config = {
            "prompt": "invalid_string_prompt"  # Should be dict
        }
        
        is_valid, errors = handler.validate_config(config)
        assert is_valid is False
        assert "'prompt' must be an object" in errors
    
    def test_validate_config_prompt_missing_content(self):
        """Test validation fails when prompt dict has no valid content"""
        handler = AIVerbHandler()
        
        config = {
            "prompt": {"other_field": "value"}  # Missing text, pom, or contexts
        }
        
        is_valid, errors = handler.validate_config(config)
        assert is_valid is False
        assert "'prompt' must contain either 'text' or 'pom' as base prompt" in errors
    
    def test_validate_config_invalid_contexts_structure(self):
        """Test validation fails with invalid contexts structure"""
        handler = AIVerbHandler()
        
        config = {
            "prompt": {
                "contexts": "invalid_string_contexts"  # Should be dict
            }
        }
        
        is_valid, errors = handler.validate_config(config)
        assert is_valid is False
        assert "'prompt.contexts' must be an object" in errors
    
    def test_build_config_with_prompt_text(self):
        """Test building config with prompt text"""
        handler = AIVerbHandler()
        
        config = handler.build_config(
            prompt_text="You are a helpful assistant",
            post_prompt="Provide a summary",
            post_prompt_url="https://example.com/summary"
        )
        
        assert config["prompt"]["text"] == "You are a helpful assistant"
        assert config["post_prompt"]["text"] == "Provide a summary"
        assert config["post_prompt_url"] == "https://example.com/summary"
    
    def test_build_config_with_prompt_pom(self):
        """Test building config with prompt POM"""
        handler = AIVerbHandler()
        
        pom_data = [
            {"title": "Section 1", "body": "Content 1"},
            {"title": "Section 2", "body": "Content 2"}
        ]
        
        config = handler.build_config(
            prompt_pom=pom_data,
            post_prompt="Summary"
        )
        
        assert config["prompt"]["pom"] == pom_data
        assert config["post_prompt"]["text"] == "Summary"
    
    def test_build_config_with_contexts(self):
        """Test building config with contexts combined with text"""
        handler = AIVerbHandler()

        contexts_data = {
            "context1": {
                "steps": [
                    {"step": "greeting", "content": "Hello"}
                ]
            }
        }

        config = handler.build_config(
            prompt_text="You are a helpful assistant",
            contexts=contexts_data,
            post_prompt="Summary"
        )

        assert config["prompt"]["text"] == "You are a helpful assistant"
        assert config["prompt"]["contexts"] == contexts_data
        assert config["post_prompt"]["text"] == "Summary"
    
    def test_build_config_with_swaig(self):
        """Test building config with SWAIG object"""
        handler = AIVerbHandler()
        
        swaig_data = {
            "functions": [
                {"function": "test_func", "description": "Test function"}
            ]
        }
        
        config = handler.build_config(
            prompt_text="You are helpful",
            swaig=swaig_data
        )
        
        assert config["prompt"]["text"] == "You are helpful"
        assert config["SWAIG"] == swaig_data
    
    def test_build_config_minimal(self):
        """Test building minimal config"""
        handler = AIVerbHandler()
        
        config = handler.build_config(prompt_text="Hello")
        
        assert config["prompt"]["text"] == "Hello"
        assert "post_prompt" not in config
        assert "post_prompt_url" not in config
        assert "SWAIG" not in config
    
    def test_build_config_validation_error(self):
        """Test that build_config raises error for invalid configuration"""
        handler = AIVerbHandler()
        
        # Should raise ValueError when no prompt options provided
        with pytest.raises(ValueError, match="Either prompt_text or prompt_pom must be provided as base prompt"):
            handler.build_config()
    
    def test_build_config_conflicting_prompts(self):
        """Test that build_config raises error for conflicting prompt types"""
        handler = AIVerbHandler()

        # Should raise ValueError when both prompt_text and prompt_pom provided
        with pytest.raises(ValueError, match="prompt_text and prompt_pom are mutually exclusive"):
            handler.build_config(
                prompt_text="Text prompt",
                prompt_pom=[{"title": "POM section"}]
            )
    
    def test_build_config_prompt_and_contexts_combined(self):
        """Test that build_config allows combining text with contexts"""
        handler = AIVerbHandler()

        # Contexts can be combined with text or pom (they are optional)
        config = handler.build_config(
            prompt_text="Text prompt",
            contexts={"context1": {"steps": []}}
        )

        assert config["prompt"]["text"] == "Text prompt"
        assert config["prompt"]["contexts"] == {"context1": {"steps": []}}
    
    def test_build_config_with_additional_params(self):
        """Test building config with additional parameters"""
        handler = AIVerbHandler()
        
        config = handler.build_config(
            prompt_text="Hello",
            languages=["en", "es"],
            hints=["hint1", "hint2"],
            pronounce={"word": "pronunciation"},
            global_data={"key": "value"},
            custom_param="custom_value"
        )
        
        assert config["prompt"]["text"] == "Hello"
        assert config["languages"] == ["en", "es"]
        assert config["hints"] == ["hint1", "hint2"]
        assert config["pronounce"] == {"word": "pronunciation"}
        assert config["global_data"] == {"key": "value"}
        assert config["params"]["custom_param"] == "custom_value"


class TestVerbHandlerRegistry:
    """Test VerbHandlerRegistry functionality"""
    
    def test_initialization(self):
        """Test registry initialization"""
        registry = VerbHandlerRegistry()
        
        # Should have AI handler registered by default
        assert registry.has_handler("ai")
        ai_handler = registry.get_handler("ai")
        assert isinstance(ai_handler, AIVerbHandler)
    
    def test_register_handler(self):
        """Test registering a new handler"""
        registry = VerbHandlerRegistry()
        mock_handler = MockVerbHandler("custom_verb")
        
        registry.register_handler(mock_handler)
        
        assert registry.has_handler("custom_verb")
        retrieved_handler = registry.get_handler("custom_verb")
        assert retrieved_handler is mock_handler
    
    def test_get_handler_existing(self):
        """Test getting an existing handler"""
        registry = VerbHandlerRegistry()
        
        handler = registry.get_handler("ai")
        assert handler is not None
        assert isinstance(handler, AIVerbHandler)
    
    def test_get_handler_nonexistent(self):
        """Test getting a non-existent handler"""
        registry = VerbHandlerRegistry()
        
        handler = registry.get_handler("nonexistent_verb")
        assert handler is None
    
    def test_has_handler_existing(self):
        """Test checking for existing handler"""
        registry = VerbHandlerRegistry()
        
        assert registry.has_handler("ai") is True
    
    def test_has_handler_nonexistent(self):
        """Test checking for non-existent handler"""
        registry = VerbHandlerRegistry()
        
        assert registry.has_handler("nonexistent_verb") is False
    
    def test_override_existing_handler(self):
        """Test overriding an existing handler"""
        registry = VerbHandlerRegistry()
        
        # Register a custom AI handler
        custom_ai_handler = MockVerbHandler("ai")
        registry.register_handler(custom_ai_handler)
        
        # Should replace the default AI handler
        retrieved_handler = registry.get_handler("ai")
        assert retrieved_handler is custom_ai_handler
        assert isinstance(retrieved_handler, MockVerbHandler)
    
    def test_multiple_handlers(self):
        """Test registering multiple handlers"""
        registry = VerbHandlerRegistry()
        
        # Register multiple custom handlers
        handler1 = MockVerbHandler("verb1")
        handler2 = MockVerbHandler("verb2")
        handler3 = MockVerbHandler("verb3")
        
        registry.register_handler(handler1)
        registry.register_handler(handler2)
        registry.register_handler(handler3)
        
        # All should be accessible
        assert registry.has_handler("verb1")
        assert registry.has_handler("verb2")
        assert registry.has_handler("verb3")
        assert registry.has_handler("ai")  # Default should still exist
        
        assert registry.get_handler("verb1") is handler1
        assert registry.get_handler("verb2") is handler2
        assert registry.get_handler("verb3") is handler3


class TestSWMLHandlerIntegration:
    """Test integration scenarios for SWML handlers"""
    
    def test_ai_handler_complete_workflow(self):
        """Test complete workflow with AI handler"""
        handler = AIVerbHandler()
        
        # Build a complete configuration
        config = handler.build_config(
            prompt_text="You are a helpful assistant",
            post_prompt="Provide a brief summary",
            post_prompt_url="https://example.com/summary",
            swaig={
                "functions": [
                    {
                        "function": "get_weather",
                        "description": "Get weather information",
                        "parameters": {"type": "object", "properties": {}}
                    }
                ]
            }
        )
        
        # Validate the configuration
        is_valid, errors = handler.validate_config(config)
        assert is_valid is True
        assert errors == []
        
        # Verify structure
        assert config["prompt"]["text"] == "You are a helpful assistant"
        assert config["post_prompt"]["text"] == "Provide a brief summary"
        assert config["post_prompt_url"] == "https://example.com/summary"
        assert "SWAIG" in config
        assert len(config["SWAIG"]["functions"]) == 1
    
    def test_registry_with_custom_handlers(self):
        """Test registry with custom handlers"""
        registry = VerbHandlerRegistry()
        
        # Create custom handlers
        play_handler = MockVerbHandler("play")
        say_handler = MockVerbHandler("say")
        
        # Register them
        registry.register_handler(play_handler)
        registry.register_handler(say_handler)
        
        # Test that all handlers work
        handlers = ["ai", "play", "say"]
        for verb_name in handlers:
            assert registry.has_handler(verb_name)
            handler = registry.get_handler(verb_name)
            assert handler is not None
            
            # Test basic functionality
            if verb_name == "ai":
                config = handler.build_config(prompt_text="Test")
                is_valid, errors = handler.validate_config(config)
                assert is_valid is True
            else:
                config = handler.build_config(required_field="test")
                is_valid, errors = handler.validate_config(config)
                assert is_valid is True
    
    def test_handler_error_scenarios(self):
        """Test error handling scenarios"""
        handler = AIVerbHandler()
        
        # Test various invalid configurations
        invalid_configs = [
            {},  # Empty config - missing prompt
            {"prompt": {}},  # Empty prompt - no content
            {"prompt": {"invalid": "field"}},  # Invalid prompt field
            {"prompt": {"text": "test", "pom": []}},  # Both text and pom
        ]
        
        for config in invalid_configs:
            is_valid, errors = handler.validate_config(config)
            assert is_valid is False
            assert len(errors) > 0
    
    def test_handler_with_complex_swaig(self):
        """Test handler with complex SWAIG configuration"""
        handler = AIVerbHandler()
        
        complex_swaig = {
            "defaults": {
                "web_hook_url": "https://example.com/webhook"
            },
            "functions": [
                {
                    "function": "search",
                    "description": "Search for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    },
                    "fillers": {
                        "en": ["Searching...", "Looking that up..."]
                    }
                },
                {
                    "function": "calculate",
                    "description": "Perform calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "Math expression"}
                        }
                    }
                }
            ],
            "includes": [
                {
                    "url": "https://api.example.com/functions",
                    "functions": ["external_func1", "external_func2"]
                }
            ]
        }
        
        config = handler.build_config(
            prompt_text="You are a calculator and search assistant",
            swaig=complex_swaig
        )
        
        # Validate the complex configuration
        is_valid, errors = handler.validate_config(config)
        assert is_valid is True
        assert errors == []
        
        # Verify SWAIG structure is preserved
        assert config["SWAIG"] == complex_swaig
        assert len(config["SWAIG"]["functions"]) == 2
        assert "defaults" in config["SWAIG"]
        assert "includes" in config["SWAIG"]


class TestSWMLHandlerEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_handler_with_none_values(self):
        """Test handler behavior with None values"""
        handler = AIVerbHandler()
        
        # Should handle None values gracefully
        config = handler.build_config(
            prompt_text="Test",
            post_prompt=None,
            post_prompt_url=None,
            swaig=None
        )
        
        assert config["prompt"]["text"] == "Test"
        assert "post_prompt" not in config
        assert "post_prompt_url" not in config
        assert "SWAIG" not in config
    
    def test_handler_with_empty_strings(self):
        """Test handler behavior with empty strings"""
        handler = AIVerbHandler()
        
        # Should handle empty strings appropriately
        config = handler.build_config(
            prompt_text="",  # Empty but valid
            post_prompt="",
            post_prompt_url=""
        )
        
        assert config["prompt"]["text"] == ""
        assert config["post_prompt"]["text"] == ""
        assert config["post_prompt_url"] == ""  # Empty URL is included if provided
    
    def test_registry_with_invalid_handler(self):
        """Test registry behavior with invalid handler"""
        registry = VerbHandlerRegistry()
        
        # Try to register an invalid handler (not implementing interface)
        invalid_handler = "not_a_handler"
        
        # Should raise AttributeError when trying to get verb name
        with pytest.raises(AttributeError):
            registry.register_handler(invalid_handler)
    
    def test_mock_handler_edge_cases(self):
        """Test mock handler edge cases"""
        handler = MockVerbHandler()
        
        # Test with empty config
        is_valid, errors = handler.validate_config({})
        assert is_valid is False
        assert len(errors) == 1
        
        # Test with None config
        with pytest.raises(TypeError):
            handler.validate_config(None)
        
        # Test build_config with no arguments
        config = handler.build_config()
        assert config["mock_config"] is True
        assert len(config) == 1 