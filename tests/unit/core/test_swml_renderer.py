"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for SWML renderer module
"""

import pytest
import json
from unittest.mock import patch
from typing import Dict, List, Any, Optional

from signalwire_agents.core.swml_renderer import SwmlRenderer
from signalwire_agents.core.swml_service import SWMLService
from signalwire_agents.utils.schema_utils import SchemaValidationError


def _make_service():
    """Create a real SWMLService with schema validation disabled for renderer tests"""
    return SWMLService(name="test_renderer", schema_validation=False)


class TestSwmlRenderer:
    """Test SwmlRenderer functionality"""

    def test_render_swml_basic(self):
        """Test basic SWML rendering"""
        service = _make_service()
        result = SwmlRenderer.render_swml({"text": "You are a helpful assistant"}, service)

        assert isinstance(result, str)
        parsed = json.loads(result)

        assert parsed["version"] == "1.0.0"
        assert "sections" in parsed
        assert "main" in parsed["sections"]
        assert len(parsed["sections"]["main"]) == 1

        # Check AI verb structure
        ai_verb = parsed["sections"]["main"][0]
        assert "ai" in ai_verb

    def test_render_swml_with_post_prompt(self):
        """Test SWML rendering with post prompt"""
        service = _make_service()
        result = SwmlRenderer.render_swml(
            {"text": "You are helpful"},
            service,
            post_prompt="Provide a summary"
        )

        parsed = json.loads(result)
        ai_verb = parsed["sections"]["main"][0]

        assert "post_prompt" in ai_verb["ai"]

    def test_render_swml_with_swaig_functions(self):
        """Test SWML rendering with SWAIG functions"""
        functions = [
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

        service = _make_service()
        result = SwmlRenderer.render_swml(
            {"text": "You are helpful"},
            service,
            swaig_functions=functions
        )

        parsed = json.loads(result)
        ai_verb = parsed["sections"]["main"][0]

        assert "SWAIG" in ai_verb["ai"]
        assert "functions" in ai_verb["ai"]["SWAIG"]
        assert len(ai_verb["ai"]["SWAIG"]["functions"]) == 1
        assert ai_verb["ai"]["SWAIG"]["functions"][0]["function"] == "get_weather"

    def test_render_swml_with_pom(self):
        """Test SWML rendering with POM format"""
        pom_data = [
            {"title": "Section 1", "body": "Content 1"},
            {"title": "Section 2", "body": "Content 2"}
        ]

        service = _make_service()
        result = SwmlRenderer.render_swml(
            {"pom": pom_data},
            service,
            prompt_is_pom=True
        )

        parsed = json.loads(result)
        ai_verb = parsed["sections"]["main"][0]

        assert "ai" in ai_verb

    def test_render_swml_with_hooks(self):
        """Test SWML rendering with startup and hangup hooks"""
        service = _make_service()
        result = SwmlRenderer.render_swml(
            {"text": "You are helpful"},
            service,
            startup_hook_url="https://example.com/startup",
            hangup_hook_url="https://example.com/hangup"
        )

        parsed = json.loads(result)
        ai_verb = parsed["sections"]["main"][0]

        assert "SWAIG" in ai_verb["ai"]
        assert "functions" in ai_verb["ai"]["SWAIG"]

    def test_render_swml_with_default_webhook(self):
        """Test SWML rendering with default webhook URL"""
        service = _make_service()
        result = SwmlRenderer.render_swml(
            {"text": "You are helpful"},
            service,
            default_webhook_url="https://example.com/webhook"
        )

        parsed = json.loads(result)
        ai_verb = parsed["sections"]["main"][0]

        assert "SWAIG" in ai_verb["ai"]
        assert "defaults" in ai_verb["ai"]["SWAIG"]
        assert ai_verb["ai"]["SWAIG"]["defaults"]["web_hook_url"] == "https://example.com/webhook"

    @patch('yaml.dump')
    def test_render_swml_yaml_format(self, mock_yaml_dump):
        """Test SWML rendering in YAML format"""
        mock_yaml_dump.return_value = "version: 1.0.0\nsections:\n  main: []"

        service = _make_service()
        result = SwmlRenderer.render_swml(
            {"text": "You are helpful"},
            service,
            format="yaml"
        )

        assert isinstance(result, str)
        assert "version: 1.0.0" in result
        assert "sections:" in result
        assert "main:" in result
        mock_yaml_dump.assert_called_once()

    def test_render_function_response_swml_basic(self):
        """Test rendering function response SWML"""
        service = _make_service()
        result = SwmlRenderer.render_function_response_swml("Hello there!", service)

        assert isinstance(result, str)
        parsed = json.loads(result)

        assert parsed["version"] == "1.0.0"
        assert "sections" in parsed
        assert "main" in parsed["sections"]
        assert len(parsed["sections"]["main"]) == 1

    def test_render_function_response_swml_with_actions(self):
        """Test rendering function response SWML with actions"""
        actions = [
            {"play": {"url": "test.mp3"}},
            {"hangup": {"reason": "completed"}}
        ]

        service = _make_service()
        result = SwmlRenderer.render_function_response_swml(
            "Response complete",
            service,
            actions=actions
        )

        parsed = json.loads(result)
        main_section = parsed["sections"]["main"]

        # Should have play verb for response plus actions
        assert len(main_section) == 3  # response + 2 actions

    @patch('yaml.dump')
    def test_render_function_response_swml_yaml(self, mock_yaml_dump):
        """Test rendering function response SWML in YAML format"""
        mock_yaml_dump.return_value = "version: 1.0.0\nsections:\n  main: []"

        service = _make_service()
        result = SwmlRenderer.render_function_response_swml(
            "Hello",
            service,
            format="yaml"
        )

        assert isinstance(result, str)
        assert "version: 1.0.0" in result
        assert "sections:" in result
        mock_yaml_dump.assert_called_once()


class TestSwmlRendererErrorHandling:
    """Test error handling in SwmlRenderer"""

    def test_render_swml_empty_prompt(self):
        """Test rendering with empty prompt"""
        service = _make_service()
        result = SwmlRenderer.render_swml({"text": ""}, service)

        parsed = json.loads(result)
        ai_verb = parsed["sections"]["main"][0]

        assert "ai" in ai_verb

    def test_render_swml_none_prompt(self):
        """Test rendering with None prompt raises validation error"""
        service = _make_service()
        # None prompt means neither prompt_text nor prompt_pom is set in builder.ai(),
        # so config has no "prompt" key, which AIVerbHandler rejects
        with pytest.raises(SchemaValidationError):
            SwmlRenderer.render_swml(None, service)

    def test_render_swml_invalid_format(self):
        """Test rendering with invalid format"""
        service = _make_service()
        result = SwmlRenderer.render_swml({"text": "Hello"}, service, format="invalid")

        # Should still be valid JSON (falls through to default)
        parsed = json.loads(result)
        assert parsed["version"] == "1.0.0"

    def test_render_function_response_empty_text(self):
        """Test rendering function response with empty text"""
        service = _make_service()
        result = SwmlRenderer.render_function_response_swml("", service)

        parsed = json.loads(result)

        assert parsed["version"] == "1.0.0"
        assert "sections" in parsed
        assert "main" in parsed["sections"]
        # Empty text should not create a play verb
        assert len(parsed["sections"]["main"]) == 0


class TestSwmlRendererIntegration:
    """Test integration scenarios"""

    def test_complete_ai_agent_swml(self):
        """Test rendering complete AI agent SWML"""
        functions = [
            {
                "function": "get_account_balance",
                "description": "Get user account balance",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "account_id": {"type": "string"}
                    },
                    "required": ["account_id"]
                }
            },
            {
                "function": "transfer_funds",
                "description": "Transfer funds between accounts",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_account": {"type": "string"},
                        "to_account": {"type": "string"},
                        "amount": {"type": "number"}
                    },
                    "required": ["from_account", "to_account", "amount"]
                }
            }
        ]

        service = _make_service()
        result = SwmlRenderer.render_swml(
            prompt={"text": "You are a banking assistant. Help users with their account needs."},
            service=service,
            post_prompt="Summarize the conversation and any actions taken.",
            post_prompt_url="https://bank.example.com/conversation-summary",
            swaig_functions=functions,
            startup_hook_url="https://bank.example.com/call-start",
            hangup_hook_url="https://bank.example.com/call-end",
            default_webhook_url="https://bank.example.com/functions",
            params={"temperature": 0.7, "max_tokens": 150}
        )

        parsed = json.loads(result)
        ai_verb = parsed["sections"]["main"][0]

        # Verify AI verb exists
        assert "ai" in ai_verb

        # Verify SWAIG configuration
        swaig = ai_verb["ai"]["SWAIG"]
        assert "defaults" in swaig
        assert swaig["defaults"]["web_hook_url"] == "https://bank.example.com/functions"

        assert "functions" in swaig
        function_names = [f["function"] for f in swaig["functions"]]
        assert "startup_hook" in function_names
        assert "hangup_hook" in function_names
        assert "get_account_balance" in function_names
        assert "transfer_funds" in function_names

    def test_pom_based_agent_swml(self):
        """Test rendering POM-based agent SWML"""
        pom_sections = [
            {
                "title": "Role",
                "body": "You are a customer service representative for TechCorp."
            },
            {
                "title": "Guidelines",
                "bullets": [
                    "Always be polite and professional",
                    "Ask clarifying questions when needed",
                    "Escalate complex issues to human agents"
                ]
            },
            {
                "title": "Available Actions",
                "body": "You can help with account inquiries, technical support, and billing questions."
            }
        ]

        service = _make_service()
        result = SwmlRenderer.render_swml(
            prompt={"pom": pom_sections},
            service=service,
            prompt_is_pom=True,
            post_prompt="Provide a brief summary of how you helped the customer."
        )

        parsed = json.loads(result)
        ai_verb = parsed["sections"]["main"][0]

        assert "ai" in ai_verb

    def test_function_response_workflow(self):
        """Test function response workflow"""
        response_text = "I found your account balance: $1,234.56"
        actions = [
            {"play": {"url": "say:Is there anything else I can help you with?"}},
        ]

        service = _make_service()
        result = SwmlRenderer.render_function_response_swml(
            response_text,
            service,
            actions=actions
        )

        parsed = json.loads(result)
        main_section = parsed["sections"]["main"]

        # Should have initial response plus actions
        assert len(main_section) == 2

    @patch('yaml.dump')
    def test_yaml_output_format(self, mock_yaml_dump):
        """Test YAML output format"""
        mock_yaml_dump.return_value = "version: 1.0.0\nsections:\n  main: []"

        service = _make_service()
        result = SwmlRenderer.render_swml(
            {"text": "You are helpful"},
            service,
            swaig_functions=[{
                "function": "test",
                "description": "Test function"
            }],
            format="yaml"
        )

        # Should be valid YAML
        assert "version: 1.0.0" in result
        assert "sections:" in result
        assert "main:" in result
        mock_yaml_dump.assert_called_once()
