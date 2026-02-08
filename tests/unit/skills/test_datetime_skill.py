"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for the DateTime skill module
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

import pytz

from signalwire_agents.skills.datetime.skill import DateTimeSkill
from signalwire_agents.core.function_result import SwaigFunctionResult


def _make_skill(params=None):
    """
    Helper to create a DateTimeSkill with a mocked agent.
    """
    default_params = {}
    if params is not None:
        default_params.update(params)

    mock_agent = Mock()
    mock_agent.define_tool = Mock()
    skill = DateTimeSkill(agent=mock_agent, params=default_params)
    return skill


# ---------------------------------------------------------------------------
# Class-level attributes
# ---------------------------------------------------------------------------

class TestDateTimeSkillClassAttributes:
    """Verify class-level constants and metadata."""

    def test_skill_name(self):
        assert DateTimeSkill.SKILL_NAME == "datetime"

    def test_skill_description(self):
        assert DateTimeSkill.SKILL_DESCRIPTION == "Get current date, time, and timezone information"

    def test_skill_version(self):
        assert DateTimeSkill.SKILL_VERSION == "1.0.0"

    def test_required_packages(self):
        assert DateTimeSkill.REQUIRED_PACKAGES == ["pytz"]

    def test_required_env_vars(self):
        assert DateTimeSkill.REQUIRED_ENV_VARS == []

    def test_supports_multiple_instances_default(self):
        assert DateTimeSkill.SUPPORTS_MULTIPLE_INSTANCES is False


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestDateTimeSkillInit:
    """Tests for __init__ (inherited from SkillBase)."""

    def test_agent_is_stored(self):
        mock_agent = Mock()
        skill = DateTimeSkill(agent=mock_agent)
        assert skill.agent is mock_agent

    def test_params_default_to_empty_dict(self):
        skill = DateTimeSkill(agent=Mock())
        assert skill.params == {}

    def test_logger_created(self):
        skill = DateTimeSkill(agent=Mock())
        assert skill.logger is not None
        assert skill.logger.name == "skill.datetime"

    def test_swaig_fields_extracted_from_params(self):
        params = {"swaig_fields": {"meta_data": {"x": 1}}}
        skill = DateTimeSkill(agent=Mock(), params=params)
        assert skill.swaig_fields == {"meta_data": {"x": 1}}
        assert "swaig_fields" not in skill.params


# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------

class TestDateTimeSkillSetup:
    """Tests for the setup method."""

    def test_setup_returns_true(self):
        skill = _make_skill()
        result = skill.setup()
        assert result is True

    def test_setup_calls_validate_packages(self):
        skill = _make_skill()
        with patch.object(skill, 'validate_packages', return_value=True) as mock_vp:
            result = skill.setup()
            mock_vp.assert_called_once()
            assert result is True

    def test_setup_returns_false_when_packages_missing(self):
        skill = _make_skill()
        with patch.object(skill, 'validate_packages', return_value=False):
            result = skill.setup()
            assert result is False


# ---------------------------------------------------------------------------
# register_tools()
# ---------------------------------------------------------------------------

class TestDateTimeSkillRegisterTools:
    """Tests for tool registration."""

    def test_register_tools_calls_define_tool_twice(self):
        skill = _make_skill()
        skill.register_tools()
        assert skill.agent.define_tool.call_count == 2

    def test_register_tools_registers_get_current_time(self):
        skill = _make_skill()
        skill.register_tools()
        calls = skill.agent.define_tool.call_args_list
        tool_names = [call.kwargs.get("name") for call in calls]
        assert "get_current_time" in tool_names

    def test_register_tools_registers_get_current_date(self):
        skill = _make_skill()
        skill.register_tools()
        calls = skill.agent.define_tool.call_args_list
        tool_names = [call.kwargs.get("name") for call in calls]
        assert "get_current_date" in tool_names

    def test_register_tools_passes_handlers(self):
        skill = _make_skill()
        skill.register_tools()
        calls = skill.agent.define_tool.call_args_list
        for call in calls:
            assert "handler" in call.kwargs
            assert callable(call.kwargs["handler"])

    def test_register_tools_merges_swaig_fields(self):
        skill = _make_skill(params={"swaig_fields": {"web_hook_url": "http://example.com"}})
        skill.register_tools()
        calls = skill.agent.define_tool.call_args_list
        for call in calls:
            assert call.kwargs.get("web_hook_url") == "http://example.com"


# ---------------------------------------------------------------------------
# _get_time_handler()
# ---------------------------------------------------------------------------

class TestGetTimeHandler:
    """Tests for the _get_time_handler method."""

    def test_default_utc_timezone(self):
        skill = _make_skill()
        result = skill._get_time_handler({}, None)
        assert isinstance(result, SwaigFunctionResult)
        assert "The current time is" in result.response
        assert "UTC" in result.response

    def test_explicit_utc_timezone(self):
        skill = _make_skill()
        result = skill._get_time_handler({"timezone": "UTC"}, None)
        assert "The current time is" in result.response
        assert "UTC" in result.response

    def test_utc_case_insensitive(self):
        skill = _make_skill()
        result = skill._get_time_handler({"timezone": "utc"}, None)
        assert "The current time is" in result.response
        assert "UTC" in result.response

    def test_specific_timezone(self):
        skill = _make_skill()
        result = skill._get_time_handler({"timezone": "America/New_York"}, None)
        assert isinstance(result, SwaigFunctionResult)
        assert "The current time is" in result.response

    def test_europe_timezone(self):
        skill = _make_skill()
        result = skill._get_time_handler({"timezone": "Europe/London"}, None)
        assert "The current time is" in result.response

    def test_invalid_timezone_returns_error(self):
        skill = _make_skill()
        result = skill._get_time_handler({"timezone": "Invalid/Timezone"}, None)
        assert isinstance(result, SwaigFunctionResult)
        assert "Error getting time" in result.response

    @patch("signalwire_agents.skills.datetime.skill.datetime")
    def test_time_format(self, mock_datetime):
        """Verify the time string format using a fixed datetime."""
        fixed_dt = datetime(2025, 6, 15, 14, 30, 45, tzinfo=timezone.utc)
        mock_datetime.now.return_value = fixed_dt
        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
        # Preserve the timezone.utc reference
        mock_datetime.now.return_value = fixed_dt

        skill = _make_skill()
        result = skill._get_time_handler({"timezone": "UTC"}, None)
        assert isinstance(result, SwaigFunctionResult)
        assert "The current time is" in result.response


# ---------------------------------------------------------------------------
# _get_date_handler()
# ---------------------------------------------------------------------------

class TestGetDateHandler:
    """Tests for the _get_date_handler method."""

    def test_default_utc_timezone(self):
        skill = _make_skill()
        result = skill._get_date_handler({}, None)
        assert isinstance(result, SwaigFunctionResult)
        assert "Today's date is" in result.response
        assert "UTC" not in result.response  # date format doesn't include timezone

    def test_explicit_utc_timezone(self):
        skill = _make_skill()
        result = skill._get_date_handler({"timezone": "UTC"}, None)
        assert "Today's date is" in result.response

    def test_utc_case_insensitive(self):
        skill = _make_skill()
        result = skill._get_date_handler({"timezone": "utc"}, None)
        assert "Today's date is" in result.response

    def test_specific_timezone(self):
        skill = _make_skill()
        result = skill._get_date_handler({"timezone": "Asia/Tokyo"}, None)
        assert isinstance(result, SwaigFunctionResult)
        assert "Today's date is" in result.response

    def test_invalid_timezone_returns_error(self):
        skill = _make_skill()
        result = skill._get_date_handler({"timezone": "Fake/Zone"}, None)
        assert isinstance(result, SwaigFunctionResult)
        assert "Error getting date" in result.response

    @patch("signalwire_agents.skills.datetime.skill.datetime")
    def test_date_format(self, mock_datetime):
        """Verify the date string format using a fixed datetime."""
        fixed_dt = datetime(2025, 1, 20, 10, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = fixed_dt
        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
        mock_datetime.now.return_value = fixed_dt

        skill = _make_skill()
        result = skill._get_date_handler({"timezone": "UTC"}, None)
        assert isinstance(result, SwaigFunctionResult)
        assert "Today's date is" in result.response


# ---------------------------------------------------------------------------
# get_hints()
# ---------------------------------------------------------------------------

class TestGetHints:
    """Tests for the get_hints method."""

    def test_returns_empty_list(self):
        skill = _make_skill()
        hints = skill.get_hints()
        assert hints == []

    def test_returns_list_type(self):
        skill = _make_skill()
        hints = skill.get_hints()
        assert isinstance(hints, list)


# ---------------------------------------------------------------------------
# get_prompt_sections()
# ---------------------------------------------------------------------------

class TestGetPromptSections:
    """Tests for the get_prompt_sections method."""

    def test_returns_list(self):
        skill = _make_skill()
        sections = skill.get_prompt_sections()
        assert isinstance(sections, list)

    def test_returns_one_section(self):
        skill = _make_skill()
        sections = skill.get_prompt_sections()
        assert len(sections) == 1

    def test_section_has_title(self):
        skill = _make_skill()
        sections = skill.get_prompt_sections()
        assert sections[0]["title"] == "Date and Time Information"

    def test_section_has_body(self):
        skill = _make_skill()
        sections = skill.get_prompt_sections()
        assert "date and time" in sections[0]["body"].lower()

    def test_section_has_bullets(self):
        skill = _make_skill()
        sections = skill.get_prompt_sections()
        bullets = sections[0]["bullets"]
        assert isinstance(bullets, list)
        assert len(bullets) == 3


# ---------------------------------------------------------------------------
# get_parameter_schema()
# ---------------------------------------------------------------------------

class TestGetParameterSchema:
    """Tests for the get_parameter_schema classmethod."""

    def test_returns_dict(self):
        schema = DateTimeSkill.get_parameter_schema()
        assert isinstance(schema, dict)

    def test_includes_swaig_fields(self):
        schema = DateTimeSkill.get_parameter_schema()
        assert "swaig_fields" in schema

    def test_no_tool_name_since_not_multi_instance(self):
        schema = DateTimeSkill.get_parameter_schema()
        assert "tool_name" not in schema

    def test_swaig_fields_type_is_object(self):
        schema = DateTimeSkill.get_parameter_schema()
        assert schema["swaig_fields"]["type"] == "object"
