"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for the API Ninjas Trivia skill module
"""

import pytest
from unittest.mock import Mock


from signalwire_agents.skills.api_ninjas_trivia.skill import ApiNinjasTriviaSkill
from signalwire_agents.core.function_result import SwaigFunctionResult


def _make_skill(params=None):
    """
    Helper to create an ApiNinjasTriviaSkill with a mocked agent.
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
    skill = ApiNinjasTriviaSkill(agent=mock_agent, params=default_params)
    return skill


# ---------------------------------------------------------------------------
# Class-level attributes
# ---------------------------------------------------------------------------

class TestApiNinjasTriviaSkillClassAttributes:
    """Verify class-level constants and metadata."""

    def test_skill_name(self):
        assert ApiNinjasTriviaSkill.SKILL_NAME == "api_ninjas_trivia"

    def test_skill_description(self):
        assert ApiNinjasTriviaSkill.SKILL_DESCRIPTION == "Get trivia questions from API Ninjas"

    def test_skill_version(self):
        assert ApiNinjasTriviaSkill.SKILL_VERSION == "1.0.0"

    def test_required_packages(self):
        assert ApiNinjasTriviaSkill.REQUIRED_PACKAGES == []

    def test_required_env_vars(self):
        assert ApiNinjasTriviaSkill.REQUIRED_ENV_VARS == []

    def test_supports_multiple_instances(self):
        assert ApiNinjasTriviaSkill.SUPPORTS_MULTIPLE_INSTANCES is True

    def test_valid_categories_count(self):
        assert len(ApiNinjasTriviaSkill.VALID_CATEGORIES) == 14

    def test_valid_categories_contains_expected_keys(self):
        expected = [
            "artliterature", "language", "sciencenature", "general",
            "fooddrink", "peopleplaces", "geography", "historyholidays",
            "entertainment", "toysgames", "music", "mathematics",
            "religionmythology", "sportsleisure"
        ]
        for key in expected:
            assert key in ApiNinjasTriviaSkill.VALID_CATEGORIES


# ---------------------------------------------------------------------------
# Initialization and Validation
# ---------------------------------------------------------------------------

class TestApiNinjasTriviaSkillInit:
    """Tests for __init__ and _validate_config."""

    def test_agent_is_stored(self):
        skill = _make_skill()
        assert skill.agent is not None

    def test_default_tool_name(self):
        skill = _make_skill()
        assert skill.tool_name == "get_trivia"

    def test_custom_tool_name(self):
        skill = _make_skill({"tool_name": "science_trivia"})
        assert skill.tool_name == "science_trivia"

    def test_api_key_stored(self):
        skill = _make_skill({"api_key": "my-key"})
        assert skill.api_key == "my-key"

    def test_default_categories_all(self):
        skill = _make_skill()
        assert skill.categories == list(ApiNinjasTriviaSkill.VALID_CATEGORIES.keys())

    def test_custom_categories(self):
        skill = _make_skill({"categories": ["music", "general"]})
        assert skill.categories == ["music", "general"]

    def test_logger_created(self):
        skill = _make_skill()
        assert skill.logger is not None
        assert skill.logger.name == "skill.api_ninjas_trivia"

    def test_swaig_fields_extracted_from_params(self):
        params = {"api_key": "key", "swaig_fields": {"meta_data": {"x": 1}}}
        skill = _make_skill(params)
        assert skill.swaig_fields == {"meta_data": {"x": 1}}

    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="api_key parameter is required"):
            mock_agent = Mock()
            ApiNinjasTriviaSkill(agent=mock_agent, params={})

    def test_none_api_key_raises(self):
        with pytest.raises(ValueError, match="api_key parameter is required"):
            mock_agent = Mock()
            ApiNinjasTriviaSkill(agent=mock_agent, params={"api_key": None})

    def test_empty_string_api_key_raises(self):
        with pytest.raises(ValueError, match="api_key parameter is required"):
            mock_agent = Mock()
            ApiNinjasTriviaSkill(agent=mock_agent, params={"api_key": ""})

    def test_non_string_api_key_raises(self):
        with pytest.raises(ValueError, match="api_key parameter is required"):
            mock_agent = Mock()
            ApiNinjasTriviaSkill(agent=mock_agent, params={"api_key": 12345})

    def test_empty_categories_list_raises(self):
        with pytest.raises(ValueError, match="categories parameter must be a non-empty list"):
            mock_agent = Mock()
            ApiNinjasTriviaSkill(agent=mock_agent, params={"api_key": "key", "categories": []})

    def test_non_list_categories_raises(self):
        with pytest.raises(ValueError, match="categories parameter must be a non-empty list"):
            mock_agent = Mock()
            ApiNinjasTriviaSkill(agent=mock_agent, params={"api_key": "key", "categories": "music"})

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="Category 'invalid_cat' is not valid"):
            mock_agent = Mock()
            ApiNinjasTriviaSkill(agent=mock_agent, params={"api_key": "key", "categories": ["invalid_cat"]})

    def test_non_string_category_raises(self):
        with pytest.raises(ValueError, match="Category 0 must be a string"):
            mock_agent = Mock()
            ApiNinjasTriviaSkill(agent=mock_agent, params={"api_key": "key", "categories": [123]})


# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------

class TestApiNinjasTriviaSkillSetup:
    """Tests for the setup method."""

    def test_setup_returns_true(self):
        skill = _make_skill()
        assert skill.setup() is True


# ---------------------------------------------------------------------------
# register_tools()
# ---------------------------------------------------------------------------

class TestApiNinjasTriviaSkillRegisterTools:
    """Tests for register_tools method."""

    def test_register_tools_calls_register_swaig_function(self):
        skill = _make_skill()
        skill.register_tools()
        skill.agent.register_swaig_function.assert_called_once()

    def test_register_tools_passes_tool_config(self):
        skill = _make_skill()
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        assert call_args["function"] == "get_trivia"

    def test_register_tools_merges_swaig_fields(self):
        skill = _make_skill({"api_key": "key", "swaig_fields": {"meta_data": {"x": 1}}})
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        assert call_args.get("meta_data") == {"x": 1}


# ---------------------------------------------------------------------------
# get_tools()
# ---------------------------------------------------------------------------

class TestApiNinjasTriviaSkillGetTools:
    """Tests for the get_tools method."""

    def test_get_tools_returns_list(self):
        skill = _make_skill()
        tools = skill.get_tools()
        assert isinstance(tools, list)

    def test_get_tools_returns_single_tool(self):
        skill = _make_skill()
        tools = skill.get_tools()
        assert len(tools) == 1

    def test_tool_function_name_matches(self):
        skill = _make_skill({"tool_name": "science_quiz"})
        tool = skill.get_tools()[0]
        assert tool["function"] == "science_quiz"

    def test_tool_description_includes_tool_name(self):
        skill = _make_skill({"tool_name": "get_science_trivia"})
        tool = skill.get_tools()[0]
        assert "get science trivia" in tool["description"]

    def test_tool_has_category_parameter(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        props = tool["parameters"]["properties"]
        assert "category" in props
        assert props["category"]["type"] == "string"

    def test_tool_category_enum_matches_configured_categories(self):
        cats = ["music", "general", "geography"]
        skill = _make_skill({"categories": cats})
        tool = skill.get_tools()[0]
        enum_values = tool["parameters"]["properties"]["category"]["enum"]
        assert enum_values == cats

    def test_tool_category_is_required(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        assert "category" in tool["parameters"]["required"]

    def test_tool_has_data_map_with_webhooks(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        assert "data_map" in tool
        assert "webhooks" in tool["data_map"]
        assert len(tool["data_map"]["webhooks"]) == 1

    def test_tool_webhook_url_correct(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        webhook = tool["data_map"]["webhooks"][0]
        assert webhook["url"] == "https://api.api-ninjas.com/v1/trivia?category=%{args.category}"

    def test_tool_webhook_method_is_get(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        webhook = tool["data_map"]["webhooks"][0]
        assert webhook["method"] == "GET"

    def test_tool_webhook_includes_api_key_header(self):
        skill = _make_skill({"api_key": "my-secret-key"})
        tool = skill.get_tools()[0]
        webhook = tool["data_map"]["webhooks"][0]
        assert webhook["headers"]["X-Api-Key"] == "my-secret-key"

    def test_tool_data_map_has_error_keys(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        assert tool["data_map"]["error_keys"] == ["error"]

    def test_tool_data_map_has_fallback_output(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        output = tool["data_map"]["output"]
        assert isinstance(output, dict)

    def test_tool_webhook_output_is_dict(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        webhook = tool["data_map"]["webhooks"][0]
        assert isinstance(webhook["output"], dict)

    def test_all_categories_default_enum(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        enum_values = tool["parameters"]["properties"]["category"]["enum"]
        assert len(enum_values) == 14

    def test_category_description_includes_human_readable(self):
        skill = _make_skill({"categories": ["sciencenature"]})
        tool = skill.get_tools()[0]
        desc = tool["parameters"]["properties"]["category"]["description"]
        assert "Science and Nature" in desc


# ---------------------------------------------------------------------------
# get_instance_key()
# ---------------------------------------------------------------------------

class TestApiNinjasTriviaSkillInstanceKey:
    """Tests for get_instance_key method."""

    def test_default_instance_key(self):
        skill = _make_skill()
        assert skill.get_instance_key() == "api_ninjas_trivia_get_trivia"

    def test_custom_instance_key(self):
        skill = _make_skill({"tool_name": "science_quiz"})
        assert skill.get_instance_key() == "api_ninjas_trivia_science_quiz"


# ---------------------------------------------------------------------------
# get_hints(), get_prompt_sections()
# ---------------------------------------------------------------------------

class TestApiNinjasTriviaSkillPromptMethods:
    """Tests for prompt-related methods inherited from SkillBase."""

    def test_get_hints_returns_empty_list(self):
        skill = _make_skill()
        assert skill.get_hints() == []

    def test_get_prompt_sections_returns_empty_list(self):
        skill = _make_skill()
        assert skill.get_prompt_sections() == []

    def test_get_global_data_returns_empty_dict(self):
        skill = _make_skill()
        assert skill.get_global_data() == {}


# ---------------------------------------------------------------------------
# get_parameter_schema()
# ---------------------------------------------------------------------------

class TestApiNinjasTriviaSkillParameterSchema:
    """Tests for get_parameter_schema class method."""

    def test_schema_has_api_key(self):
        schema = ApiNinjasTriviaSkill.get_parameter_schema()
        assert "api_key" in schema

    def test_api_key_is_required(self):
        schema = ApiNinjasTriviaSkill.get_parameter_schema()
        assert schema["api_key"]["required"] is True

    def test_api_key_is_hidden(self):
        schema = ApiNinjasTriviaSkill.get_parameter_schema()
        assert schema["api_key"]["hidden"] is True

    def test_api_key_has_env_var(self):
        schema = ApiNinjasTriviaSkill.get_parameter_schema()
        assert schema["api_key"]["env_var"] == "API_NINJAS_KEY"

    def test_schema_has_categories(self):
        schema = ApiNinjasTriviaSkill.get_parameter_schema()
        assert "categories" in schema

    def test_categories_type_is_array(self):
        schema = ApiNinjasTriviaSkill.get_parameter_schema()
        assert schema["categories"]["type"] == "array"

    def test_categories_not_required(self):
        schema = ApiNinjasTriviaSkill.get_parameter_schema()
        assert schema["categories"]["required"] is False

    def test_categories_default_is_all(self):
        schema = ApiNinjasTriviaSkill.get_parameter_schema()
        assert schema["categories"]["default"] == list(ApiNinjasTriviaSkill.VALID_CATEGORIES.keys())

    def test_categories_items_has_enum(self):
        schema = ApiNinjasTriviaSkill.get_parameter_schema()
        items = schema["categories"]["items"]
        assert "enum" in items
        assert len(items["enum"]) == 14

    def test_schema_has_swaig_fields(self):
        schema = ApiNinjasTriviaSkill.get_parameter_schema()
        assert "swaig_fields" in schema

    def test_schema_has_tool_name_for_multi_instance(self):
        schema = ApiNinjasTriviaSkill.get_parameter_schema()
        assert "tool_name" in schema
