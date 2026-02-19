"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for the PlayBackgroundFile skill module
"""

import pytest
from unittest.mock import Mock

from signalwire_agents.skills.play_background_file.skill import PlayBackgroundFileSkill
from signalwire_agents.core.function_result import SwaigFunctionResult


def _make_skill(params=None):
    """
    Helper to create a PlayBackgroundFileSkill with a mocked agent.
    Provides sensible defaults for all required parameters.
    """
    default_params = {
        "tool_name": "play_testimonial",
        "files": [
            {
                "key": "massey",
                "description": "Customer success story from Massey Energy",
                "url": "https://example.com/massey.mp4",
                "wait": True,
            },
            {
                "key": "demo-video",
                "description": "Product demo video",
                "url": "https://example.com/demo.mp4",
            },
        ],
    }
    if params is not None:
        default_params.update(params)

    mock_agent = Mock()
    mock_agent.register_swaig_function = Mock()
    skill = PlayBackgroundFileSkill(agent=mock_agent, params=default_params)
    return skill


# ---------------------------------------------------------------------------
# Class-level attributes
# ---------------------------------------------------------------------------


class TestPlayBackgroundFileSkillClassAttributes:
    """Verify class-level constants and metadata."""

    def test_skill_name(self):
        assert PlayBackgroundFileSkill.SKILL_NAME == "play_background_file"

    def test_skill_description(self):
        assert PlayBackgroundFileSkill.SKILL_DESCRIPTION == "Control background file playback"

    def test_skill_version(self):
        assert PlayBackgroundFileSkill.SKILL_VERSION == "1.0.0"

    def test_required_packages(self):
        assert PlayBackgroundFileSkill.REQUIRED_PACKAGES == []

    def test_required_env_vars(self):
        assert PlayBackgroundFileSkill.REQUIRED_ENV_VARS == []

    def test_supports_multiple_instances(self):
        assert PlayBackgroundFileSkill.SUPPORTS_MULTIPLE_INSTANCES is True


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestPlayBackgroundFileSkillInit:
    """Tests for __init__."""

    def test_agent_is_stored(self):
        skill = _make_skill()
        assert skill.agent is not None

    def test_tool_name_from_params(self):
        skill = _make_skill()
        assert skill.tool_name == "play_testimonial"

    def test_tool_name_default(self):
        params = {
            "files": [
                {"key": "a", "description": "desc", "url": "https://example.com/a.mp4"}
            ],
        }
        mock_agent = Mock()
        mock_agent.register_swaig_function = Mock()
        skill = PlayBackgroundFileSkill(agent=mock_agent, params=params)
        assert skill.tool_name == "play_background_file"

    def test_files_stored(self):
        skill = _make_skill()
        assert len(skill.files) == 2
        assert skill.files[0]["key"] == "massey"
        assert skill.files[1]["key"] == "demo-video"

    def test_logger_created(self):
        skill = _make_skill()
        assert skill.logger is not None
        assert skill.logger.name == "signalwire_agents.skills.play_background_file"

    def test_swaig_fields_extracted_from_params(self):
        skill = _make_skill({"swaig_fields": {"meta_data": {"x": 1}}})
        assert skill.swaig_fields == {"meta_data": {"x": 1}}

    def test_swaig_fields_default_empty(self):
        skill = _make_skill()
        assert skill.swaig_fields == {}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestPlayBackgroundFileSkillValidation:
    """Tests for _validate_config."""

    def test_files_must_be_list(self):
        with pytest.raises(ValueError, match="files parameter must be a non-empty list"):
            _make_skill({"files": "not-a-list"})

    def test_files_must_not_be_empty(self):
        with pytest.raises(ValueError, match="files parameter must be a non-empty list"):
            _make_skill({"files": []})

    def test_file_must_be_dict(self):
        with pytest.raises(ValueError, match="File 0 must be a dictionary"):
            _make_skill({"files": ["not-a-dict"]})

    def test_file_missing_key(self):
        with pytest.raises(ValueError, match="File 0 missing required field: key"):
            _make_skill({"files": [{"description": "d", "url": "http://x"}]})

    def test_file_missing_description(self):
        with pytest.raises(ValueError, match="File 0 missing required field: description"):
            _make_skill({"files": [{"key": "k", "url": "http://x"}]})

    def test_file_missing_url(self):
        with pytest.raises(ValueError, match="File 0 missing required field: url"):
            _make_skill({"files": [{"key": "k", "description": "d"}]})

    def test_file_key_empty_string(self):
        with pytest.raises(ValueError, match="must be a non-empty string"):
            _make_skill({"files": [{"key": "", "description": "d", "url": "http://x"}]})

    def test_file_key_whitespace_only(self):
        with pytest.raises(ValueError, match="must be a non-empty string"):
            _make_skill({"files": [{"key": "  ", "description": "d", "url": "http://x"}]})

    def test_file_wait_must_be_boolean(self):
        with pytest.raises(ValueError, match="must be a boolean"):
            _make_skill(
                {"files": [{"key": "k", "description": "d", "url": "http://x", "wait": "yes"}]}
            )

    def test_file_key_invalid_characters(self):
        with pytest.raises(ValueError, match="must contain only alphanumeric"):
            _make_skill(
                {"files": [{"key": "bad key!", "description": "d", "url": "http://x"}]}
            )

    def test_file_key_allows_underscores_and_hyphens(self):
        """Keys with underscores and hyphens should be accepted."""
        skill = _make_skill(
            {"files": [{"key": "my_key-1", "description": "d", "url": "http://x"}]}
        )
        assert skill.files[0]["key"] == "my_key-1"


# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------


class TestPlayBackgroundFileSkillSetup:
    """Tests for setup()."""

    def test_setup_returns_true(self):
        skill = _make_skill()
        assert skill.setup() is True


# ---------------------------------------------------------------------------
# get_instance_key()
# ---------------------------------------------------------------------------


class TestPlayBackgroundFileSkillInstanceKey:
    """Tests for get_instance_key()."""

    def test_instance_key_includes_tool_name(self):
        skill = _make_skill()
        assert skill.get_instance_key() == "play_background_file_play_testimonial"

    def test_instance_key_default_tool_name(self):
        params = {
            "files": [
                {"key": "a", "description": "desc", "url": "https://example.com/a.mp4"}
            ],
        }
        mock_agent = Mock()
        mock_agent.register_swaig_function = Mock()
        skill = PlayBackgroundFileSkill(agent=mock_agent, params=params)
        assert skill.get_instance_key() == "play_background_file_play_background_file"


# ---------------------------------------------------------------------------
# get_tools() and DataMap expression building
# ---------------------------------------------------------------------------


class TestPlayBackgroundFileSkillGetTools:
    """Tests for get_tools() and _build_expressions()."""

    def test_get_tools_returns_single_tool(self):
        skill = _make_skill()
        tools = skill.get_tools()
        assert isinstance(tools, list)
        assert len(tools) == 1

    def test_tool_function_name(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        assert tool["function"] == "play_testimonial"

    def test_tool_description_contains_tool_name(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        assert "play testimonial" in tool["description"]

    def test_tool_has_action_parameter(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        props = tool["parameters"]["properties"]
        assert "action" in props
        assert props["action"]["type"] == "string"

    def test_tool_action_enum_values(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        enum_vals = tool["parameters"]["properties"]["action"]["enum"]
        assert "start_massey" in enum_vals
        assert "start_demo-video" in enum_vals
        assert "stop" in enum_vals

    def test_tool_action_enum_count(self):
        """Enum should have one start per file + one stop."""
        skill = _make_skill()
        tool = skill.get_tools()[0]
        enum_vals = tool["parameters"]["properties"]["action"]["enum"]
        # 2 files + 1 stop
        assert len(enum_vals) == 3

    def test_tool_action_required(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        assert "action" in tool["parameters"]["required"]

    def test_tool_wait_for_fillers(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        assert tool["wait_for_fillers"] is True

    def test_tool_skip_fillers(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        assert tool["skip_fillers"] is True

    def test_tool_has_data_map_expressions(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        assert "data_map" in tool
        assert "expressions" in tool["data_map"]

    def test_expressions_count(self):
        """Should have one expression per file + one stop expression."""
        skill = _make_skill()
        tool = skill.get_tools()[0]
        expressions = tool["data_map"]["expressions"]
        # 2 files + 1 stop
        assert len(expressions) == 3

    def test_expression_pattern_start(self):
        skill = _make_skill()
        expressions = skill.get_tools()[0]["data_map"]["expressions"]
        assert expressions[0]["pattern"] == "/start_massey/i"
        assert expressions[1]["pattern"] == "/start_demo-video/i"

    def test_expression_string_uses_args_action(self):
        skill = _make_skill()
        expressions = skill.get_tools()[0]["data_map"]["expressions"]
        for expr in expressions:
            assert expr["string"] == "${args.action}"

    def test_expression_stop_pattern(self):
        skill = _make_skill()
        expressions = skill.get_tools()[0]["data_map"]["expressions"]
        stop_expr = expressions[-1]
        assert stop_expr["pattern"] == "/stop/i"

    def test_expression_output_has_response(self):
        skill = _make_skill()
        expressions = skill.get_tools()[0]["data_map"]["expressions"]
        for expr in expressions:
            assert "response" in expr["output"]

    def test_start_expression_with_wait_true(self):
        """File with wait=True should produce playback_bg action with wait."""
        skill = _make_skill()
        expressions = skill.get_tools()[0]["data_map"]["expressions"]
        massey_expr = expressions[0]
        output = massey_expr["output"]
        # Should have action array with playback_bg
        assert "action" in output
        actions = output["action"]
        playback_action = [a for a in actions if "playback_bg" in a]
        assert len(playback_action) == 1
        # wait=True means the value should be a dict with file and wait
        assert playback_action[0]["playback_bg"]["file"] == "https://example.com/massey.mp4"
        assert playback_action[0]["playback_bg"]["wait"] is True

    def test_start_expression_with_wait_false_default(self):
        """File without wait should produce playback_bg with just the url string."""
        skill = _make_skill()
        expressions = skill.get_tools()[0]["data_map"]["expressions"]
        demo_expr = expressions[1]
        output = demo_expr["output"]
        assert "action" in output
        actions = output["action"]
        playback_action = [a for a in actions if "playback_bg" in a]
        assert len(playback_action) == 1
        assert playback_action[0]["playback_bg"] == "https://example.com/demo.mp4"

    def test_stop_expression_output(self):
        """Stop expression should contain stop_playback_bg action."""
        skill = _make_skill()
        expressions = skill.get_tools()[0]["data_map"]["expressions"]
        stop_expr = expressions[-1]
        output = stop_expr["output"]
        assert "action" in output
        actions = output["action"]
        stop_action = [a for a in actions if "stop_playback_bg" in a]
        assert len(stop_action) == 1
        assert stop_action[0]["stop_playback_bg"] is True

    def test_start_expression_has_post_process(self):
        """Start expressions should have post_process=True."""
        skill = _make_skill()
        expressions = skill.get_tools()[0]["data_map"]["expressions"]
        massey_expr = expressions[0]
        assert massey_expr["output"].get("post_process") is True

    def test_stop_expression_no_post_process(self):
        """Stop expression should not have post_process (was not created with it)."""
        skill = _make_skill()
        expressions = skill.get_tools()[0]["data_map"]["expressions"]
        stop_expr = expressions[-1]
        # post_process not set when creating the stop result
        assert stop_expr["output"].get("post_process") is not True

    def test_action_description_lists_all_options(self):
        skill = _make_skill()
        tool = skill.get_tools()[0]
        desc = tool["parameters"]["properties"]["action"]["description"]
        assert "start_massey" in desc
        assert "start_demo-video" in desc
        assert "stop" in desc
        assert "Customer success story from Massey Energy" in desc


# ---------------------------------------------------------------------------
# register_tools()
# ---------------------------------------------------------------------------


class TestPlayBackgroundFileSkillRegisterTools:
    """Tests for register_tools()."""

    def test_register_tools_calls_register_swaig_function(self):
        skill = _make_skill()
        skill.register_tools()
        skill.agent.register_swaig_function.assert_called_once()

    def test_register_tools_passes_tool_config(self):
        skill = _make_skill()
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        assert call_args["function"] == "play_testimonial"

    def test_register_tools_merges_swaig_fields(self):
        skill = _make_skill({"swaig_fields": {"meta_data": {"tag": "v1"}}})
        skill.register_tools()
        call_args = skill.agent.register_swaig_function.call_args[0][0]
        assert call_args["meta_data"] == {"tag": "v1"}


# ---------------------------------------------------------------------------
# get_hints(), get_prompt_sections(), get_parameter_schema()
# ---------------------------------------------------------------------------


class TestPlayBackgroundFileSkillMiscMethods:
    """Tests for inherited helper methods."""

    def test_get_hints_returns_empty_list(self):
        skill = _make_skill()
        assert skill.get_hints() == []

    def test_get_prompt_sections_returns_empty_list(self):
        skill = _make_skill()
        assert skill.get_prompt_sections() == []

    def test_get_global_data_returns_empty_dict(self):
        skill = _make_skill()
        assert skill.get_global_data() == {}

    def test_get_parameter_schema_has_files(self):
        schema = PlayBackgroundFileSkill.get_parameter_schema()
        assert "files" in schema
        assert schema["files"]["type"] == "array"
        assert schema["files"]["required"] is True

    def test_get_parameter_schema_files_items(self):
        schema = PlayBackgroundFileSkill.get_parameter_schema()
        items = schema["files"]["items"]
        assert items["type"] == "object"
        assert "key" in items["properties"]
        assert "description" in items["properties"]
        assert "url" in items["properties"]
        assert "wait" in items["properties"]

    def test_get_parameter_schema_has_tool_name(self):
        """Multi-instance skills should include tool_name in schema."""
        schema = PlayBackgroundFileSkill.get_parameter_schema()
        assert "tool_name" in schema
        assert schema["tool_name"]["type"] == "string"

    def test_get_parameter_schema_has_swaig_fields(self):
        schema = PlayBackgroundFileSkill.get_parameter_schema()
        assert "swaig_fields" in schema

    def test_cleanup_does_not_raise(self):
        skill = _make_skill()
        skill.cleanup()  # Should not raise
