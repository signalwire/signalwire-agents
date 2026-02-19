"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for the Joke skill module
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from signalwire_agents.skills.joke.skill import JokeSkill
from signalwire_agents.core.function_result import SwaigFunctionResult


def _make_skill(params=None):
    """
    Helper to create a JokeSkill with a mocked agent.
    Provides sensible defaults for all required parameters.
    """
    default_params = {
        "api_key": "test-api-key-12345",
    }
    if params is not None:
        default_params.update(params)

    mock_agent = Mock()
    mock_agent.register_swaig_function = Mock()
    skill = JokeSkill(agent=mock_agent, params=default_params)
    return skill


# ---------------------------------------------------------------------------
# Class-level attributes
# ---------------------------------------------------------------------------

class TestJokeSkillClassAttributes:
    """Verify class-level constants and metadata."""

    def test_skill_name(self):
        assert JokeSkill.SKILL_NAME == "joke"

    def test_skill_description(self):
        assert JokeSkill.SKILL_DESCRIPTION == "Tell jokes using the API Ninjas joke API"

    def test_skill_version(self):
        assert JokeSkill.SKILL_VERSION == "1.0.0"

    def test_required_packages(self):
        assert JokeSkill.REQUIRED_PACKAGES == []

    def test_required_env_vars(self):
        assert JokeSkill.REQUIRED_ENV_VARS == []

    def test_supports_multiple_instances(self):
        assert JokeSkill.SUPPORTS_MULTIPLE_INSTANCES is False


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestJokeSkillInit:
    """Tests for __init__ (inherited from SkillBase)."""

    def test_agent_is_stored(self):
        mock_agent = Mock()
        skill = JokeSkill(agent=mock_agent, params={"api_key": "k"})
        assert skill.agent is mock_agent

    def test_params_stored(self):
        params = {"api_key": "my-key", "tool_name": "my_joke"}
        skill = JokeSkill(agent=Mock(), params=params)
        assert skill.params["api_key"] == "my-key"
        assert skill.params["tool_name"] == "my_joke"

    def test_params_default_to_empty_dict(self):
        skill = JokeSkill(agent=Mock())
        assert skill.params == {}

    def test_logger_created(self):
        skill = JokeSkill(agent=Mock())
        assert skill.logger is not None
        assert skill.logger.name == "signalwire_agents.skills.joke"

    def test_swaig_fields_extracted_from_params(self):
        params = {"swaig_fields": {"meta_data": {"x": 1}}, "api_key": "k"}
        skill = JokeSkill(agent=Mock(), params=params)
        assert skill.swaig_fields == {"meta_data": {"x": 1}}
        assert "swaig_fields" not in skill.params

    def test_swaig_fields_default_empty(self):
        skill = JokeSkill(agent=Mock(), params={"api_key": "k"})
        assert skill.swaig_fields == {}


# ---------------------------------------------------------------------------
# get_parameter_schema
# ---------------------------------------------------------------------------

class TestGetParameterSchema:
    """Tests for the class method get_parameter_schema."""

    def test_contains_api_key_param(self):
        schema = JokeSkill.get_parameter_schema()
        assert "api_key" in schema
        assert schema["api_key"]["required"] is True

    def test_api_key_is_hidden(self):
        schema = JokeSkill.get_parameter_schema()
        assert schema["api_key"].get("hidden") is True

    def test_api_key_env_var(self):
        schema = JokeSkill.get_parameter_schema()
        assert schema["api_key"].get("env_var") == "API_NINJAS_KEY"

    def test_contains_tool_name_param(self):
        schema = JokeSkill.get_parameter_schema()
        assert "tool_name" in schema
        assert schema["tool_name"]["required"] is False

    def test_tool_name_default(self):
        schema = JokeSkill.get_parameter_schema()
        assert schema["tool_name"]["default"] == "get_joke"

    def test_includes_base_class_swaig_fields(self):
        schema = JokeSkill.get_parameter_schema()
        assert "swaig_fields" in schema


# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------

class TestSetup:
    """Tests for the setup method."""

    def test_setup_success_with_api_key(self):
        skill = _make_skill()
        result = skill.setup()
        assert result is True

    def test_setup_stores_api_key(self):
        skill = _make_skill()
        skill.setup()
        assert skill.api_key == "test-api-key-12345"

    def test_setup_default_tool_name(self):
        skill = _make_skill()
        skill.setup()
        assert skill.tool_name == "get_joke"

    def test_setup_custom_tool_name(self):
        skill = _make_skill({"tool_name": "tell_joke"})
        skill.setup()
        assert skill.tool_name == "tell_joke"

    def test_setup_missing_api_key_returns_false(self):
        skill = _make_skill({"api_key": ""})
        result = skill.setup()
        assert result is False

    def test_setup_none_api_key_returns_false(self):
        skill = _make_skill({"api_key": None})
        result = skill.setup()
        assert result is False

    def test_setup_missing_api_key_no_params(self):
        mock_agent = Mock()
        skill = JokeSkill(agent=mock_agent, params={})
        result = skill.setup()
        assert result is False

    def test_setup_logs_error_on_missing_api_key(self):
        skill = _make_skill({"api_key": ""})
        with patch.object(skill.logger, "error") as mock_error:
            skill.setup()
            mock_error.assert_called_once()
            assert "api_key" in mock_error.call_args[0][0]


# ---------------------------------------------------------------------------
# register_tools()
# ---------------------------------------------------------------------------

class TestRegisterTools:
    """Tests for register_tools method."""

    def test_register_tools_calls_register_swaig_function(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        skill.agent.register_swaig_function.assert_called_once()

    def test_register_tools_passes_dict(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args
        swaig_func = call_args[0][0]
        assert isinstance(swaig_func, dict)

    def test_register_tools_function_name_default(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        swaig_func = skill.agent.register_swaig_function.call_args[0][0]
        assert swaig_func["function"] == "get_joke"

    def test_register_tools_function_name_custom(self):
        skill = _make_skill({"tool_name": "my_jokes"})
        skill.setup()
        skill.register_tools()
        swaig_func = skill.agent.register_swaig_function.call_args[0][0]
        assert swaig_func["function"] == "my_jokes"

    def test_register_tools_has_description(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        swaig_func = skill.agent.register_swaig_function.call_args[0][0]
        assert "description" in swaig_func
        assert len(swaig_func["description"]) > 0

    def test_register_tools_has_parameters(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        swaig_func = skill.agent.register_swaig_function.call_args[0][0]
        assert "parameters" in swaig_func
        params = swaig_func["parameters"]
        assert "properties" in params
        assert "type" in params["properties"]

    def test_register_tools_type_param_has_enum(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        swaig_func = skill.agent.register_swaig_function.call_args[0][0]
        type_param = swaig_func["parameters"]["properties"]["type"]
        assert "enum" in type_param
        assert "jokes" in type_param["enum"]
        assert "dadjokes" in type_param["enum"]

    def test_register_tools_type_param_is_required(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        swaig_func = skill.agent.register_swaig_function.call_args[0][0]
        assert "type" in swaig_func["parameters"].get("required", [])

    def test_register_tools_has_data_map(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        swaig_func = skill.agent.register_swaig_function.call_args[0][0]
        assert "data_map" in swaig_func

    def test_register_tools_data_map_has_webhooks(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        swaig_func = skill.agent.register_swaig_function.call_args[0][0]
        data_map = swaig_func["data_map"]
        assert "webhooks" in data_map
        assert len(data_map["webhooks"]) > 0

    def test_register_tools_webhook_url(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        swaig_func = skill.agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        assert "api-ninjas.com" in webhook["url"]
        assert "${args.type}" in webhook["url"]

    def test_register_tools_webhook_method(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        swaig_func = skill.agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        assert webhook["method"] == "GET"

    def test_register_tools_webhook_has_api_key_header(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        swaig_func = skill.agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        assert "headers" in webhook
        assert webhook["headers"]["X-Api-Key"] == "test-api-key-12345"

    def test_register_tools_webhook_has_output(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        swaig_func = skill.agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        assert "output" in webhook

    def test_register_tools_data_map_has_error_keys(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        swaig_func = skill.agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        assert "error_keys" in webhook
        assert "error" in webhook["error_keys"]

    def test_register_tools_has_fallback_output(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        swaig_func = skill.agent.register_swaig_function.call_args[0][0]
        data_map = swaig_func["data_map"]
        assert "output" in data_map


# ---------------------------------------------------------------------------
# get_hints()
# ---------------------------------------------------------------------------

class TestGetHints:
    """Tests for the get_hints method."""

    def test_returns_empty_list(self):
        skill = _make_skill()
        assert skill.get_hints() == []

    def test_returns_list_type(self):
        skill = _make_skill()
        assert isinstance(skill.get_hints(), list)


# ---------------------------------------------------------------------------
# get_global_data()
# ---------------------------------------------------------------------------

class TestGetGlobalData:
    """Tests for the get_global_data method."""

    def test_returns_dict(self):
        skill = _make_skill()
        skill.setup()
        data = skill.get_global_data()
        assert isinstance(data, dict)

    def test_joke_skill_enabled(self):
        skill = _make_skill()
        skill.setup()
        data = skill.get_global_data()
        assert data["joke_skill_enabled"] is True


# ---------------------------------------------------------------------------
# get_prompt_sections()
# ---------------------------------------------------------------------------

class TestGetPromptSections:
    """Tests for the get_prompt_sections method."""

    def test_returns_one_section(self):
        skill = _make_skill()
        skill.setup()
        sections = skill.get_prompt_sections()
        assert len(sections) == 1

    def test_section_has_title(self):
        skill = _make_skill()
        skill.setup()
        section = skill.get_prompt_sections()[0]
        assert "title" in section
        assert section["title"] == "Joke Telling"

    def test_section_has_body(self):
        skill = _make_skill()
        skill.setup()
        section = skill.get_prompt_sections()[0]
        assert "body" in section
        assert len(section["body"]) > 0

    def test_section_has_bullets(self):
        skill = _make_skill()
        skill.setup()
        section = skill.get_prompt_sections()[0]
        assert "bullets" in section
        assert len(section["bullets"]) > 0

    def test_section_references_tool_name(self):
        skill = _make_skill()
        skill.setup()
        section = skill.get_prompt_sections()[0]
        assert "get_joke" in section["bullets"][0]

    def test_section_references_custom_tool_name(self):
        skill = _make_skill({"tool_name": "my_jokes"})
        skill.setup()
        section = skill.get_prompt_sections()[0]
        assert any("my_jokes" in bullet for bullet in section["bullets"])


# ---------------------------------------------------------------------------
# get_instance_key()
# ---------------------------------------------------------------------------

class TestGetInstanceKey:
    """Tests for get_instance_key (single instance skill)."""

    def test_instance_key_is_skill_name(self):
        skill = _make_skill()
        assert skill.get_instance_key() == "joke"


# ---------------------------------------------------------------------------
# Edge cases and integration-style tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case and integration-style tests."""

    def test_setup_then_register_flow(self):
        """Full lifecycle: setup -> register_tools."""
        skill = _make_skill()
        assert skill.setup() is True
        skill.register_tools()
        skill.agent.register_swaig_function.assert_called_once()

    def test_custom_api_key_in_header(self):
        """Custom API key should appear in the webhook header."""
        skill = _make_skill({"api_key": "custom-secret-key"})
        skill.setup()
        skill.register_tools()
        swaig_func = skill.agent.register_swaig_function.call_args[0][0]
        webhook = swaig_func["data_map"]["webhooks"][0]
        assert webhook["headers"]["X-Api-Key"] == "custom-secret-key"

    def test_fallback_output_contains_sorry(self):
        """Fallback output should contain an apology or service message."""
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        swaig_func = skill.agent.register_swaig_function.call_args[0][0]
        fallback = swaig_func["data_map"]["output"]
        # The fallback output should have a response field
        assert "response" in fallback
        assert "sorry" in fallback["response"].lower() or "problem" in fallback["response"].lower()
