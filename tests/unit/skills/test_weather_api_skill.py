"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for the Weather API skill module
"""

import pytest
from unittest.mock import Mock

from signalwire_agents.skills.weather_api.skill import WeatherApiSkill
from signalwire_agents.core.function_result import SwaigFunctionResult


def _make_skill(params=None):
    """
    Helper to create a WeatherApiSkill with a mocked agent.
    Provides sensible defaults for all required parameters.
    """
    default_params = {
        "api_key": "test-api-key-123",
    }
    if params is not None:
        default_params.update(params)

    mock_agent = Mock()
    mock_agent.define_tool = Mock()
    mock_agent.register_swaig_function = Mock()
    skill = WeatherApiSkill(agent=mock_agent, params=default_params)
    return skill


# ---------------------------------------------------------------------------
# Class-level attributes
# ---------------------------------------------------------------------------

class TestWeatherApiSkillClassAttributes:
    """Verify class-level constants and metadata."""

    def test_skill_name(self):
        assert WeatherApiSkill.SKILL_NAME == "weather_api"

    def test_skill_description(self):
        assert WeatherApiSkill.SKILL_DESCRIPTION == "Get current weather information from WeatherAPI.com"

    def test_skill_version(self):
        assert WeatherApiSkill.SKILL_VERSION == "1.0.0"

    def test_required_packages(self):
        assert WeatherApiSkill.REQUIRED_PACKAGES == []

    def test_required_env_vars(self):
        assert WeatherApiSkill.REQUIRED_ENV_VARS == []

    def test_supports_multiple_instances(self):
        assert WeatherApiSkill.SUPPORTS_MULTIPLE_INSTANCES is False


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestWeatherApiSkillInit:
    """Tests for __init__."""

    def test_agent_is_stored(self):
        mock_agent = Mock()
        skill = WeatherApiSkill(agent=mock_agent, params={"api_key": "key123"})
        assert skill.agent is mock_agent

    def test_default_tool_name(self):
        skill = _make_skill()
        assert skill.tool_name == "get_weather"

    def test_custom_tool_name(self):
        skill = _make_skill({"tool_name": "check_weather"})
        assert skill.tool_name == "check_weather"

    def test_default_temperature_unit(self):
        skill = _make_skill()
        assert skill.temperature_unit == "fahrenheit"

    def test_custom_temperature_unit_celsius(self):
        skill = _make_skill({"temperature_unit": "celsius"})
        assert skill.temperature_unit == "celsius"

    def test_api_key_stored(self):
        skill = _make_skill({"api_key": "my-secret-key"})
        assert skill.api_key == "my-secret-key"

    def test_logger_created(self):
        skill = _make_skill()
        assert skill.logger is not None
        assert skill.logger.name == "signalwire_agents.skills.weather_api"

    def test_swaig_fields_extracted_from_params(self):
        params = {"swaig_fields": {"meta_data": {"x": 1}}, "api_key": "key"}
        skill = WeatherApiSkill(agent=Mock(), params=params)
        assert skill.swaig_fields == {"meta_data": {"x": 1}}
        assert "swaig_fields" not in skill.params

    def test_swaig_fields_default_empty(self):
        skill = _make_skill()
        assert skill.swaig_fields == {}


# ---------------------------------------------------------------------------
# _validate_config()
# ---------------------------------------------------------------------------

class TestValidateConfig:
    """Tests for configuration validation."""

    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="api_key"):
            WeatherApiSkill(agent=Mock(), params={})

    def test_none_api_key_raises(self):
        with pytest.raises(ValueError, match="api_key"):
            WeatherApiSkill(agent=Mock(), params={"api_key": None})

    def test_empty_string_api_key_raises(self):
        with pytest.raises(ValueError, match="api_key"):
            WeatherApiSkill(agent=Mock(), params={"api_key": ""})

    def test_non_string_api_key_raises(self):
        with pytest.raises(ValueError, match="api_key"):
            WeatherApiSkill(agent=Mock(), params={"api_key": 12345})

    def test_invalid_temperature_unit_raises(self):
        with pytest.raises(ValueError, match="temperature_unit"):
            WeatherApiSkill(agent=Mock(), params={"api_key": "key", "temperature_unit": "kelvin"})

    def test_valid_config_does_not_raise(self):
        skill = _make_skill()
        # If we get here without error, validation passed
        assert skill.api_key == "test-api-key-123"


# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------

class TestSetup:
    """Tests for the setup method."""

    def test_setup_returns_true(self):
        skill = _make_skill()
        assert skill.setup() is True

    def test_setup_returns_true_with_celsius(self):
        skill = _make_skill({"temperature_unit": "celsius"})
        assert skill.setup() is True


# ---------------------------------------------------------------------------
# register_tools()
# ---------------------------------------------------------------------------

class TestRegisterTools:
    """Tests for register_tools method."""

    def test_register_tools_calls_register_swaig_function(self):
        skill = _make_skill()
        skill.register_tools()
        skill.agent.register_swaig_function.assert_called_once()

    def test_register_tools_passes_tool_dict(self):
        skill = _make_skill()
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args
        tool = call_args[0][0]
        assert isinstance(tool, dict)
        assert "function" in tool
        assert "data_map" in tool

    def test_register_tools_merges_swaig_fields(self):
        """swaig_fields from params should be merged into the tool dict."""
        params = {
            "swaig_fields": {"meta_data": {"key": "val"}},
            "api_key": "test-key",
        }
        mock_agent = Mock()
        mock_agent.register_swaig_function = Mock()
        skill = WeatherApiSkill(agent=mock_agent, params=params)
        skill.register_tools()

        call_args = mock_agent.register_swaig_function.call_args
        tool = call_args[0][0]
        assert tool.get("meta_data") == {"key": "val"}


# ---------------------------------------------------------------------------
# get_tools()
# ---------------------------------------------------------------------------

class TestGetTools:
    """Tests for the get_tools method."""

    def test_returns_list_with_one_tool(self):
        skill = _make_skill()
        tools = skill.get_tools()
        assert isinstance(tools, list)
        assert len(tools) == 1

    def test_tool_function_name_default(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        assert tool["function"] == "get_weather"

    def test_tool_function_name_custom(self):
        skill = _make_skill({"tool_name": "weather_check"})
        tool = skill.get_tools()[0]
        assert tool["function"] == "weather_check"

    def test_tool_has_description(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        assert "description" in tool
        assert "weather" in tool["description"].lower()

    def test_tool_has_location_parameter(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        params = tool["parameters"]
        assert params["type"] == "object"
        assert "location" in params["properties"]
        assert params["properties"]["location"]["type"] == "string"
        assert "location" in params["required"]

    def test_tool_has_data_map_with_webhooks(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        assert "data_map" in tool
        assert "webhooks" in tool["data_map"]
        assert len(tool["data_map"]["webhooks"]) == 1

    def test_webhook_url_contains_api_key(self):
        skill = _make_skill({"api_key": "my-secret-key"})
        tool = skill.get_tools()[0]
        webhook = tool["data_map"]["webhooks"][0]
        assert "my-secret-key" in webhook["url"]

    def test_webhook_url_contains_location_placeholder(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        webhook = tool["data_map"]["webhooks"][0]
        assert "${lc:enc:args.location}" in webhook["url"]

    def test_webhook_method_is_get(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        webhook = tool["data_map"]["webhooks"][0]
        assert webhook["method"] == "GET"

    def test_webhook_output_is_dict(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        webhook = tool["data_map"]["webhooks"][0]
        assert isinstance(webhook["output"], dict)

    def test_data_map_has_error_keys(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        assert tool["data_map"]["error_keys"] == ["error"]

    def test_data_map_has_fallback_output(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        assert "output" in tool["data_map"]
        assert isinstance(tool["data_map"]["output"], dict)

    def test_fahrenheit_uses_temp_f(self):
        skill = _make_skill({"temperature_unit": "fahrenheit"})
        tool = skill.get_tools()[0]
        webhook = tool["data_map"]["webhooks"][0]
        output = webhook["output"]
        response_text = output.get("response", "")
        assert "temp_f" in response_text
        assert "feelslike_f" in response_text
        assert "Fahrenheit" in response_text

    def test_celsius_uses_temp_c(self):
        skill = _make_skill({"temperature_unit": "celsius"})
        tool = skill.get_tools()[0]
        webhook = tool["data_map"]["webhooks"][0]
        output = webhook["output"]
        response_text = output.get("response", "")
        assert "temp_c" in response_text
        assert "feelslike_c" in response_text
        assert "Celsius" in response_text

    def test_weather_template_contains_condition(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        webhook = tool["data_map"]["webhooks"][0]
        response_text = webhook["output"].get("response", "")
        assert "current.condition.text" in response_text

    def test_weather_template_contains_wind_info(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        webhook = tool["data_map"]["webhooks"][0]
        response_text = webhook["output"].get("response", "")
        assert "current.wind_dir" in response_text
        assert "current.wind_mph" in response_text

    def test_weather_template_contains_cloud_info(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        webhook = tool["data_map"]["webhooks"][0]
        response_text = webhook["output"].get("response", "")
        assert "current.cloud" in response_text

    def test_fallback_output_contains_error_message(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        fallback = tool["data_map"]["output"]
        response_text = fallback.get("response", "")
        assert "sorry" in response_text.lower() or "cannot" in response_text.lower()


# ---------------------------------------------------------------------------
# get_hints()
# ---------------------------------------------------------------------------

class TestGetHints:
    """Tests for the get_hints method."""

    def test_returns_empty_list(self):
        skill = _make_skill()
        assert skill.get_hints() == []


# ---------------------------------------------------------------------------
# get_prompt_sections()
# ---------------------------------------------------------------------------

class TestGetPromptSections:
    """Tests for the get_prompt_sections method."""

    def test_returns_empty_list(self):
        skill = _make_skill()
        assert skill.get_prompt_sections() == []


# ---------------------------------------------------------------------------
# get_parameter_schema()
# ---------------------------------------------------------------------------

class TestGetParameterSchema:
    """Tests for the class method get_parameter_schema."""

    def test_contains_api_key(self):
        schema = WeatherApiSkill.get_parameter_schema()
        assert "api_key" in schema
        assert schema["api_key"]["required"] is True

    def test_api_key_is_hidden(self):
        schema = WeatherApiSkill.get_parameter_schema()
        assert schema["api_key"].get("hidden") is True

    def test_api_key_env_var(self):
        schema = WeatherApiSkill.get_parameter_schema()
        assert schema["api_key"].get("env_var") == "WEATHER_API_KEY"

    def test_contains_tool_name(self):
        schema = WeatherApiSkill.get_parameter_schema()
        assert "tool_name" in schema
        assert schema["tool_name"]["required"] is False
        assert schema["tool_name"]["default"] == "get_weather"

    def test_contains_temperature_unit(self):
        schema = WeatherApiSkill.get_parameter_schema()
        assert "temperature_unit" in schema
        assert schema["temperature_unit"]["required"] is False
        assert schema["temperature_unit"]["default"] == "fahrenheit"

    def test_temperature_unit_enum(self):
        schema = WeatherApiSkill.get_parameter_schema()
        assert set(schema["temperature_unit"]["enum"]) == {"fahrenheit", "celsius"}

    def test_includes_base_class_swaig_fields(self):
        schema = WeatherApiSkill.get_parameter_schema()
        assert "swaig_fields" in schema

    def test_no_tool_name_from_base_because_single_instance(self):
        """Since SUPPORTS_MULTIPLE_INSTANCES is False, base class should NOT add tool_name.
        But the subclass adds its own tool_name, so it should still be there."""
        schema = WeatherApiSkill.get_parameter_schema()
        # tool_name is added by the subclass, not the base
        assert "tool_name" in schema
        # Verify it's the subclass version (has 'default' of 'get_weather')
        assert schema["tool_name"]["default"] == "get_weather"


# ---------------------------------------------------------------------------
# get_instance_key()
# ---------------------------------------------------------------------------

class TestGetInstanceKey:
    """Tests for get_instance_key."""

    def test_returns_skill_name_because_single_instance(self):
        skill = _make_skill()
        assert skill.get_instance_key() == "weather_api"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests."""

    def test_webhook_url_format(self):
        skill = _make_skill({"api_key": "abc123"})
        tool = skill.get_tools()[0]
        webhook = tool["data_map"]["webhooks"][0]
        expected_prefix = "https://api.weatherapi.com/v1/current.json?key=abc123&q=${lc:enc:args.location}&aqi=no"
        assert webhook["url"] == expected_prefix

    def test_tts_friendly_response_mentions_natural_language(self):
        """Response instruction should mention natural language for TTS."""
        skill = _make_skill()
        tool = skill.get_tools()[0]
        webhook = tool["data_map"]["webhooks"][0]
        response_text = webhook["output"].get("response", "")
        assert "natural language" in response_text.lower()
