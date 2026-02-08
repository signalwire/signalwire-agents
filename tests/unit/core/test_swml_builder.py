"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for SWML builder module
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional

from signalwire_agents.core.swml_builder import SWMLBuilder
from signalwire_agents.core.swml_service import SWMLService


class TestSWMLBuilder:
    """Test SWMLBuilder functionality"""
    
    def test_basic_initialization(self):
        """Test basic SWMLBuilder initialization"""
        mock_service = Mock(spec=SWMLService)
        builder = SWMLBuilder(mock_service)
        
        assert builder.service is mock_service
    
    def test_answer_verb(self):
        """Test adding answer verb"""
        mock_service = Mock(spec=SWMLService)
        builder = SWMLBuilder(mock_service)

        result = builder.answer()

        assert result is builder  # Should return self for chaining
        mock_service.add_verb.assert_called_once_with("answer", {})

    def test_answer_verb_with_params(self):
        """Test adding answer verb with parameters"""
        mock_service = Mock(spec=SWMLService)
        builder = SWMLBuilder(mock_service)

        result = builder.answer(max_duration=30, codecs="PCMU,PCMA")

        assert result is builder
        mock_service.add_verb.assert_called_once_with("answer", {"max_duration": 30, "codecs": "PCMU,PCMA"})

    def test_hangup_verb(self):
        """Test adding hangup verb"""
        mock_service = Mock(spec=SWMLService)
        builder = SWMLBuilder(mock_service)

        result = builder.hangup()

        assert result is builder
        mock_service.add_verb.assert_called_once_with("hangup", {})

    def test_hangup_verb_with_reason(self):
        """Test adding hangup verb with reason"""
        mock_service = Mock(spec=SWMLService)
        builder = SWMLBuilder(mock_service)

        result = builder.hangup(reason="completed")

        assert result is builder
        mock_service.add_verb.assert_called_once_with("hangup", {"reason": "completed"})

    def test_ai_verb_basic(self):
        """Test adding basic AI verb"""
        mock_service = Mock(spec=SWMLService)
        builder = SWMLBuilder(mock_service)

        result = builder.ai(prompt_text="You are helpful")

        assert result is builder
        mock_service.add_verb.assert_called_once_with("ai", {"prompt": "You are helpful"})

    def test_ai_verb_with_pom(self):
        """Test adding AI verb with POM"""
        mock_service = Mock(spec=SWMLService)
        builder = SWMLBuilder(mock_service)

        pom_data = [{"title": "Test", "body": "Content"}]
        result = builder.ai(prompt_pom=pom_data)

        assert result is builder
        mock_service.add_verb.assert_called_once_with("ai", {"prompt": pom_data})

    def test_ai_verb_with_swaig(self):
        """Test adding AI verb with SWAIG configuration"""
        mock_service = Mock(spec=SWMLService)
        builder = SWMLBuilder(mock_service)

        swaig_config = {
            "functions": [{"function": "test", "description": "Test function"}]
        }
        result = builder.ai(prompt_text="You are helpful", swaig=swaig_config)

        assert result is builder
        mock_service.add_verb.assert_called_once_with("ai", {
            "prompt": "You are helpful",
            "SWAIG": swaig_config
        })

    def test_ai_verb_with_kwargs(self):
        """Test adding AI verb with additional parameters"""
        mock_service = Mock(spec=SWMLService)
        builder = SWMLBuilder(mock_service)

        result = builder.ai(
            prompt_text="You are helpful",
            temperature=0.7,
            max_tokens=150
        )

        assert result is builder
        mock_service.add_verb.assert_called_once_with("ai", {
            "prompt": "You are helpful",
            "temperature": 0.7,
            "max_tokens": 150
        })
    
    def test_play_verb_with_url(self):
        """Test adding play verb with single URL"""
        mock_service = Mock(spec=SWMLService)
        builder = SWMLBuilder(mock_service)
        
        result = builder.play(url="test.mp3")
        
        assert result is builder
        mock_service.add_verb.assert_called_once_with("play", {"url": "test.mp3"})
    
    def test_play_verb_with_urls(self):
        """Test adding play verb with multiple URLs"""
        mock_service = Mock(spec=SWMLService)
        builder = SWMLBuilder(mock_service)
        
        urls = ["test1.mp3", "test2.mp3"]
        result = builder.play(urls=urls)
        
        assert result is builder
        mock_service.add_verb.assert_called_once_with("play", {"urls": urls})
    
    def test_play_verb_with_options(self):
        """Test adding play verb with options"""
        mock_service = Mock(spec=SWMLService)
        builder = SWMLBuilder(mock_service)
        
        result = builder.play(
            url="test.mp3",
            volume=0.8,
            say_voice="alice",
            say_language="en-US"
        )
        
        assert result is builder
        expected_config = {
            "url": "test.mp3",
            "volume": 0.8,
            "say_voice": "alice",
            "say_language": "en-US"
        }
        mock_service.add_verb.assert_called_once_with("play", expected_config)
    
    def test_play_verb_no_url_error(self):
        """Test play verb raises error when no URL provided"""
        mock_service = Mock(spec=SWMLService)
        builder = SWMLBuilder(mock_service)
        
        with pytest.raises(ValueError, match="Either url or urls must be provided"):
            builder.play()
    
    def test_say_verb(self):
        """Test adding say verb"""
        mock_service = Mock(spec=SWMLService)
        builder = SWMLBuilder(mock_service)
        
        result = builder.say("Hello world")
        
        assert result is builder
        mock_service.add_verb.assert_called_once_with("play", {"url": "say:Hello world"})
    
    def test_say_verb_with_options(self):
        """Test adding say verb with options"""
        mock_service = Mock(spec=SWMLService)
        builder = SWMLBuilder(mock_service)
        
        result = builder.say(
            "Hello world",
            voice="alice",
            language="en-US",
            volume=0.7
        )
        
        assert result is builder
        expected_config = {
            "url": "say:Hello world",
            "say_voice": "alice",
            "say_language": "en-US",
            "volume": 0.7
        }
        mock_service.add_verb.assert_called_once_with("play", expected_config)
    
    def test_add_section(self):
        """Test adding section"""
        mock_service = Mock(spec=SWMLService)
        builder = SWMLBuilder(mock_service)
        
        result = builder.add_section("greeting")
        
        assert result is builder
        mock_service.add_section.assert_called_once_with("greeting")
    
    def test_build(self):
        """Test building document"""
        mock_service = Mock(spec=SWMLService)
        mock_service.get_document.return_value = {"version": "1.0.0", "sections": {"main": []}}
        builder = SWMLBuilder(mock_service)
        
        result = builder.build()
        
        assert result == {"version": "1.0.0", "sections": {"main": []}}
        mock_service.get_document.assert_called_once()
    
    def test_render(self):
        """Test rendering document"""
        mock_service = Mock(spec=SWMLService)
        mock_service.render_document.return_value = '{"version": "1.0.0"}'
        builder = SWMLBuilder(mock_service)
        
        result = builder.render()
        
        assert result == '{"version": "1.0.0"}'
        mock_service.render_document.assert_called_once()
    
    def test_reset(self):
        """Test resetting document"""
        mock_service = Mock(spec=SWMLService)
        builder = SWMLBuilder(mock_service)
        
        result = builder.reset()
        
        assert result is builder
        mock_service.reset_document.assert_called_once()
    
    def test_method_chaining(self):
        """Test method chaining functionality"""
        mock_service = Mock(spec=SWMLService)
        builder = SWMLBuilder(mock_service)

        result = (builder
                 .answer()
                 .say("Hello")
                 .ai(prompt_text="You are helpful")
                 .hangup())

        assert result is builder

        # Verify all methods delegated to add_verb
        calls = mock_service.add_verb.call_args_list
        assert len(calls) == 4
        assert calls[0][0][0] == "answer"
        assert calls[1][0][0] == "play"
        assert calls[2][0][0] == "ai"
        assert calls[3][0][0] == "hangup"


class TestSWMLBuilderErrorHandling:
    """Test error handling in SWMLBuilder"""
    
    def test_initialization_without_service(self):
        """Test initialization without service raises error"""
        with pytest.raises(TypeError):
            SWMLBuilder()
    
    def test_initialization_with_none_service(self):
        """Test initialization with None service"""
        # SWMLBuilder should accept None but it will fail when methods are called
        builder = SWMLBuilder(None)
        assert builder.service is None
    
    def test_service_method_errors_propagate(self):
        """Test that service method errors propagate"""
        mock_service = Mock(spec=SWMLService)
        mock_service.add_verb.side_effect = ValueError("Invalid prompt")
        builder = SWMLBuilder(mock_service)

        with pytest.raises(ValueError, match="Invalid prompt"):
            builder.ai(prompt_text="")


class TestSWMLBuilderIntegration:
    """Test integration scenarios"""
    
    def test_complete_agent_workflow(self):
        """Test complete agent building workflow"""
        mock_service = Mock(spec=SWMLService)
        mock_service.get_document.return_value = {
            "version": "1.0.0",
            "sections": {
                "main": [
                    {"answer": {}},
                    {"play": {"url": "say:Welcome!"}},
                    {"ai": {"prompt": "You are helpful"}},
                    {"hangup": {"reason": "completed"}}
                ]
            }
        }

        builder = SWMLBuilder(mock_service)

        # Build a complete workflow
        result = (builder
                 .answer()
                 .say("Welcome!")
                 .ai(prompt_text="You are helpful")
                 .hangup(reason="completed")
                 .build())

        # Verify the document structure
        assert result["version"] == "1.0.0"
        assert "sections" in result
        assert "main" in result["sections"]
        assert len(result["sections"]["main"]) == 4
    
    def test_multi_section_workflow(self):
        """Test multi-section workflow"""
        mock_service = Mock(spec=SWMLService)
        builder = SWMLBuilder(mock_service)
        
        # Build workflow with multiple sections
        result = (builder
                 .add_section("greeting")
                 .say("Hello!")
                 .add_section("main")
                 .ai(prompt_text="You are helpful")
                 .add_section("goodbye")
                 .say("Goodbye!")
                 .hangup())
        
        assert result is builder
        
        # Verify sections were added
        assert mock_service.add_section.call_count == 3
        mock_service.add_section.assert_any_call("greeting")
        mock_service.add_section.assert_any_call("main")
        mock_service.add_section.assert_any_call("goodbye")
    
    def test_complex_ai_configuration(self):
        """Test complex AI configuration"""
        mock_service = Mock(spec=SWMLService)
        builder = SWMLBuilder(mock_service)

        swaig_config = {
            "defaults": {"web_hook_url": "https://example.com/webhook"},
            "functions": [
                {
                    "function": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            ]
        }

        result = builder.ai(
            prompt_text="You are a weather assistant",
            post_prompt="Summarize the weather information provided",
            post_prompt_url="https://example.com/summary",
            swaig=swaig_config,
            temperature=0.7,
            max_tokens=150
        )

        assert result is builder
        mock_service.add_verb.assert_called_once_with("ai", {
            "prompt": "You are a weather assistant",
            "post_prompt": "Summarize the weather information provided",
            "post_prompt_url": "https://example.com/summary",
            "SWAIG": swaig_config,
            "temperature": 0.7,
            "max_tokens": 150
        })
    
    def test_service_delegation(self):
        """Test that builder properly delegates to service"""
        real_service = SWMLService(name="test_service", schema_validation=False)
        builder = SWMLBuilder(real_service)

        # Build a simple document
        builder.answer().say("Hello").hangup()

        # Get the actual document
        document = builder.build()

        # Verify structure
        assert document["version"] == "1.0.0"
        assert "sections" in document
        assert "main" in document["sections"]

        # Verify verbs were added
        main_section = document["sections"]["main"]
        assert len(main_section) > 0 