"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for the DataSphere skill module
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock

import requests

from signalwire_agents.skills.datasphere.skill import DataSphereSkill
from signalwire_agents.core.function_result import SwaigFunctionResult


def _make_skill(params=None):
    """
    Helper to create a DataSphereSkill with a mocked agent.
    Provides sensible defaults for all required parameters.
    """
    default_params = {
        "space_name": "testspace",
        "project_id": "test-project-id",
        "token": "test-token",
        "document_id": "test-doc-id",
    }
    if params is not None:
        default_params.update(params)

    mock_agent = Mock()
    mock_agent.define_tool = Mock()
    skill = DataSphereSkill(agent=mock_agent, params=default_params)
    return skill


# ---------------------------------------------------------------------------
# Class-level attributes
# ---------------------------------------------------------------------------

class TestDataSphereSkillClassAttributes:
    """Verify class-level constants and metadata."""

    def test_skill_name(self):
        assert DataSphereSkill.SKILL_NAME == "datasphere"

    def test_skill_description(self):
        assert DataSphereSkill.SKILL_DESCRIPTION == "Search knowledge using SignalWire DataSphere RAG stack"

    def test_skill_version(self):
        assert DataSphereSkill.SKILL_VERSION == "1.0.0"

    def test_required_packages(self):
        assert DataSphereSkill.REQUIRED_PACKAGES == ["requests"]

    def test_required_env_vars(self):
        assert DataSphereSkill.REQUIRED_ENV_VARS == []

    def test_supports_multiple_instances(self):
        assert DataSphereSkill.SUPPORTS_MULTIPLE_INSTANCES is True


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestDataSphereSkillInit:
    """Tests for __init__ (inherited from SkillBase)."""

    def test_agent_is_stored(self):
        mock_agent = Mock()
        skill = DataSphereSkill(agent=mock_agent, params={"space_name": "x"})
        assert skill.agent is mock_agent

    def test_params_stored(self):
        params = {"space_name": "myspace", "project_id": "pid"}
        skill = DataSphereSkill(agent=Mock(), params=params)
        assert skill.params["space_name"] == "myspace"
        assert skill.params["project_id"] == "pid"

    def test_params_default_to_empty_dict(self):
        skill = DataSphereSkill(agent=Mock())
        assert skill.params == {}

    def test_logger_created(self):
        skill = DataSphereSkill(agent=Mock())
        assert skill.logger is not None
        assert skill.logger.name == "signalwire_agents.skills.datasphere"

    def test_swaig_fields_extracted_from_params(self):
        params = {"swaig_fields": {"meta_data": {"x": 1}}, "space_name": "s"}
        skill = DataSphereSkill(agent=Mock(), params=params)
        assert skill.swaig_fields == {"meta_data": {"x": 1}}
        assert "swaig_fields" not in skill.params

    def test_swaig_fields_default_empty(self):
        skill = DataSphereSkill(agent=Mock(), params={"space_name": "s"})
        assert skill.swaig_fields == {}


# ---------------------------------------------------------------------------
# get_parameter_schema
# ---------------------------------------------------------------------------

class TestGetParameterSchema:
    """Tests for the class method get_parameter_schema."""

    def test_contains_required_params(self):
        schema = DataSphereSkill.get_parameter_schema()
        for key in ("space_name", "project_id", "token", "document_id"):
            assert key in schema, f"Missing required param: {key}"
            assert schema[key]["required"] is True

    def test_contains_optional_params(self):
        schema = DataSphereSkill.get_parameter_schema()
        for key in ("count", "distance", "tags", "language", "pos_to_expand",
                     "max_synonyms", "no_results_message"):
            assert key in schema, f"Missing optional param: {key}"
            assert schema[key]["required"] is False

    def test_token_is_hidden(self):
        schema = DataSphereSkill.get_parameter_schema()
        assert schema["token"].get("hidden") is True

    def test_project_id_env_var(self):
        schema = DataSphereSkill.get_parameter_schema()
        assert schema["project_id"].get("env_var") == "SIGNALWIRE_PROJECT_ID"

    def test_token_env_var(self):
        schema = DataSphereSkill.get_parameter_schema()
        assert schema["token"].get("env_var") == "SIGNALWIRE_TOKEN"

    def test_count_defaults(self):
        schema = DataSphereSkill.get_parameter_schema()
        assert schema["count"]["default"] == 1
        assert schema["count"]["minimum"] == 1
        assert schema["count"]["maximum"] == 10

    def test_distance_defaults(self):
        schema = DataSphereSkill.get_parameter_schema()
        assert schema["distance"]["default"] == 3.0
        assert schema["distance"]["minimum"] == 0.0
        assert schema["distance"]["maximum"] == 10.0

    def test_includes_base_class_swaig_fields(self):
        schema = DataSphereSkill.get_parameter_schema()
        assert "swaig_fields" in schema

    def test_includes_tool_name_because_multi_instance(self):
        schema = DataSphereSkill.get_parameter_schema()
        assert "tool_name" in schema

    def test_pos_to_expand_items_enum(self):
        schema = DataSphereSkill.get_parameter_schema()
        items = schema["pos_to_expand"]["items"]
        assert set(items["enum"]) == {"NOUN", "VERB", "ADJ", "ADV"}


# ---------------------------------------------------------------------------
# get_instance_key
# ---------------------------------------------------------------------------

class TestGetInstanceKey:
    """Tests for get_instance_key."""

    def test_default_instance_key(self):
        skill = _make_skill()
        assert skill.get_instance_key() == "datasphere_search_knowledge"

    def test_custom_tool_name_instance_key(self):
        skill = _make_skill({"tool_name": "my_search"})
        assert skill.get_instance_key() == "datasphere_my_search"


# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------

class TestSetup:
    """Tests for the setup method."""

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_setup_success_all_required(self, mock_session_cls):
        skill = _make_skill()
        result = skill.setup()

        assert result is True
        assert skill.space_name == "testspace"
        assert skill.project_id == "test-project-id"
        assert skill.token == "test-token"
        assert skill.document_id == "test-doc-id"

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_setup_creates_api_url(self, mock_session_cls):
        skill = _make_skill()
        skill.setup()
        assert skill.api_url == "https://testspace.signalwire.com/api/datasphere/documents/search"

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_setup_creates_session(self, mock_session_cls):
        skill = _make_skill()
        skill.setup()
        mock_session_cls.assert_called_once()
        assert skill.session is mock_session_cls.return_value

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_setup_optional_defaults(self, mock_session_cls):
        skill = _make_skill()
        skill.setup()

        assert skill.count == 1
        assert skill.distance == 3.0
        assert skill.tags is None
        assert skill.language is None
        assert skill.pos_to_expand is None
        assert skill.max_synonyms is None
        assert skill.tool_name == "search_knowledge"
        assert "{query}" in skill.no_results_message

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_setup_custom_optional_values(self, mock_session_cls):
        skill = _make_skill({
            "count": 5,
            "distance": 1.5,
            "tags": ["faq", "billing"],
            "language": "es",
            "pos_to_expand": ["NOUN", "VERB"],
            "max_synonyms": 3,
            "tool_name": "kb_search",
            "no_results_message": "Nothing found.",
        })
        skill.setup()

        assert skill.count == 5
        assert skill.distance == 1.5
        assert skill.tags == ["faq", "billing"]
        assert skill.language == "es"
        assert skill.pos_to_expand == ["NOUN", "VERB"]
        assert skill.max_synonyms == 3
        assert skill.tool_name == "kb_search"
        assert skill.no_results_message == "Nothing found."

    def test_setup_missing_space_name(self):
        skill = _make_skill({"space_name": ""})
        result = skill.setup()
        assert result is False

    def test_setup_missing_project_id(self):
        skill = _make_skill({"project_id": ""})
        result = skill.setup()
        assert result is False

    def test_setup_missing_token(self):
        skill = _make_skill({"token": ""})
        result = skill.setup()
        assert result is False

    def test_setup_missing_document_id(self):
        skill = _make_skill({"document_id": ""})
        result = skill.setup()
        assert result is False

    def test_setup_missing_multiple_params(self):
        mock_agent = Mock()
        skill = DataSphereSkill(agent=mock_agent, params={})
        result = skill.setup()
        assert result is False

    def test_setup_missing_param_none_value(self):
        skill = _make_skill({"space_name": None})
        result = skill.setup()
        assert result is False

    def test_setup_logs_error_on_missing_params(self):
        skill = _make_skill({"space_name": ""})
        with patch.object(skill.logger, "error") as mock_error:
            skill.setup()
            mock_error.assert_called_once()
            assert "space_name" in mock_error.call_args[0][0]


# ---------------------------------------------------------------------------
# register_tools()
# ---------------------------------------------------------------------------

class TestRegisterTools:
    """Tests for register_tools method."""

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_register_tools_calls_define_tool(self, mock_session_cls):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()

        skill.agent.define_tool.assert_called_once()

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_register_tools_default_name(self, mock_session_cls):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()

        call_kwargs = skill.agent.define_tool.call_args
        assert call_kwargs[1]["name"] == "search_knowledge" or call_kwargs.kwargs.get("name") == "search_knowledge"

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_register_tools_custom_name(self, mock_session_cls):
        skill = _make_skill({"tool_name": "kb_lookup"})
        skill.setup()
        skill.register_tools()

        call_kwargs = skill.agent.define_tool.call_args
        # define_tool is called with keyword args via **kwargs
        _, kw = call_kwargs
        assert kw["name"] == "kb_lookup"

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_register_tools_has_query_parameter(self, mock_session_cls):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()

        _, kw = skill.agent.define_tool.call_args
        assert "query" in kw["parameters"]
        assert kw["parameters"]["query"]["type"] == "string"

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_register_tools_handler_is_callable(self, mock_session_cls):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()

        _, kw = skill.agent.define_tool.call_args
        assert callable(kw["handler"])

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_register_tools_merges_swaig_fields(self, mock_session_cls):
        """swaig_fields from params should be merged into define_tool call."""
        params = {
            "swaig_fields": {"meta_data": {"key": "val"}},
            "space_name": "testspace",
            "project_id": "test-project-id",
            "token": "test-token",
            "document_id": "test-doc-id",
        }
        mock_agent = Mock()
        mock_agent.define_tool = Mock()
        skill = DataSphereSkill(agent=mock_agent, params=params)
        skill.setup()
        skill.register_tools()

        _, kw = mock_agent.define_tool.call_args
        assert kw.get("meta_data") == {"key": "val"}


# ---------------------------------------------------------------------------
# _search_knowledge_handler()
# ---------------------------------------------------------------------------

class TestSearchKnowledgeHandler:
    """Tests for the _search_knowledge_handler method."""

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def _setup_skill(self, mock_session_cls, params=None):
        """Helper that returns (skill, mock_session_instance)."""
        skill = _make_skill(params)
        skill.setup()
        return skill, skill.session

    def test_empty_query_returns_error(self):
        skill, _ = self._setup_skill()
        result = skill._search_knowledge_handler({"query": ""}, {})
        assert isinstance(result, SwaigFunctionResult)
        assert "provide a search query" in result.response.lower()

    def test_whitespace_query_returns_error(self):
        skill, _ = self._setup_skill()
        result = skill._search_knowledge_handler({"query": "   "}, {})
        assert isinstance(result, SwaigFunctionResult)
        assert "provide a search query" in result.response.lower()

    def test_missing_query_key_returns_error(self):
        skill, _ = self._setup_skill()
        result = skill._search_knowledge_handler({}, {})
        assert isinstance(result, SwaigFunctionResult)
        assert "provide a search query" in result.response.lower()

    def test_successful_search_single_chunk(self):
        skill, mock_session = self._setup_skill()
        mock_response = Mock()
        mock_response.json.return_value = {
            "chunks": [{"text": "Answer to your question"}]
        }
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response

        result = skill._search_knowledge_handler({"query": "test query"}, {})
        assert isinstance(result, SwaigFunctionResult)
        assert "1 result" in result.response
        assert "Answer to your question" in result.response

    def test_successful_search_multiple_chunks(self):
        skill, mock_session = self._setup_skill(params={"count": 3})
        mock_response = Mock()
        mock_response.json.return_value = {
            "chunks": [
                {"text": "First result"},
                {"text": "Second result"},
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response

        result = skill._search_knowledge_handler({"query": "test"}, {})
        assert "2 results" in result.response
        assert "First result" in result.response
        assert "Second result" in result.response

    def test_no_results_returns_default_message(self):
        skill, mock_session = self._setup_skill()
        mock_response = Mock()
        mock_response.json.return_value = {"chunks": []}
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response

        result = skill._search_knowledge_handler({"query": "unknown topic"}, {})
        assert isinstance(result, SwaigFunctionResult)
        assert "unknown topic" in result.response

    def test_no_results_custom_message_with_placeholder(self):
        skill, mock_session = self._setup_skill(
            params={"no_results_message": "Nothing for '{query}'."}
        )
        mock_response = Mock()
        mock_response.json.return_value = {"chunks": []}
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response

        result = skill._search_knowledge_handler({"query": "widgets"}, {})
        assert result.response == "Nothing for 'widgets'."

    def test_no_results_custom_message_without_placeholder(self):
        skill, mock_session = self._setup_skill(
            params={"no_results_message": "No data available."}
        )
        mock_response = Mock()
        mock_response.json.return_value = {"chunks": []}
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response

        result = skill._search_knowledge_handler({"query": "anything"}, {})
        assert result.response == "No data available."

    def test_invalid_response_data_none(self):
        skill, mock_session = self._setup_skill()
        mock_response = Mock()
        mock_response.json.return_value = None
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response

        result = skill._search_knowledge_handler({"query": "test"}, {})
        assert isinstance(result, SwaigFunctionResult)
        # Should return no-results message

    def test_invalid_response_data_not_dict(self):
        skill, mock_session = self._setup_skill()
        mock_response = Mock()
        mock_response.json.return_value = "invalid"
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response

        result = skill._search_knowledge_handler({"query": "test"}, {})
        assert isinstance(result, SwaigFunctionResult)

    def test_timeout_error(self):
        skill, mock_session = self._setup_skill()
        mock_session.post.side_effect = requests.exceptions.Timeout("timed out")

        result = skill._search_knowledge_handler({"query": "test"}, {})
        assert isinstance(result, SwaigFunctionResult)
        assert "timed out" in result.response.lower()

    def test_http_error(self):
        skill, mock_session = self._setup_skill()
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("403 Forbidden")
        mock_session.post.return_value = mock_response

        result = skill._search_knowledge_handler({"query": "test"}, {})
        assert isinstance(result, SwaigFunctionResult)
        assert "error" in result.response.lower()

    def test_generic_exception(self):
        skill, mock_session = self._setup_skill()
        mock_session.post.side_effect = RuntimeError("unexpected")

        result = skill._search_knowledge_handler({"query": "test"}, {})
        assert isinstance(result, SwaigFunctionResult)
        assert "error" in result.response.lower()

    def test_request_payload_required_fields(self):
        skill, mock_session = self._setup_skill()
        mock_response = Mock()
        mock_response.json.return_value = {"chunks": []}
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response

        skill._search_knowledge_handler({"query": "hello world"}, {})

        _, call_kwargs = mock_session.post.call_args
        payload = call_kwargs["json"]
        assert payload["document_id"] == "test-doc-id"
        assert payload["query_string"] == "hello world"
        assert payload["distance"] == 3.0
        assert payload["count"] == 1

    def test_request_payload_excludes_none_optionals(self):
        skill, mock_session = self._setup_skill()
        mock_response = Mock()
        mock_response.json.return_value = {"chunks": []}
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response

        skill._search_knowledge_handler({"query": "test"}, {})

        _, call_kwargs = mock_session.post.call_args
        payload = call_kwargs["json"]
        assert "tags" not in payload
        assert "language" not in payload
        assert "pos_to_expand" not in payload
        assert "max_synonyms" not in payload

    def test_request_payload_includes_optional_when_set(self):
        skill, mock_session = self._setup_skill(params={
            "tags": ["faq"],
            "language": "en",
            "pos_to_expand": ["NOUN"],
            "max_synonyms": 5,
        })
        mock_response = Mock()
        mock_response.json.return_value = {"chunks": []}
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response

        skill._search_knowledge_handler({"query": "test"}, {})

        _, call_kwargs = mock_session.post.call_args
        payload = call_kwargs["json"]
        assert payload["tags"] == ["faq"]
        assert payload["language"] == "en"
        assert payload["pos_to_expand"] == ["NOUN"]
        assert payload["max_synonyms"] == 5

    def test_request_uses_basic_auth(self):
        skill, mock_session = self._setup_skill()
        mock_response = Mock()
        mock_response.json.return_value = {"chunks": []}
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response

        skill._search_knowledge_handler({"query": "test"}, {})

        _, call_kwargs = mock_session.post.call_args
        assert call_kwargs["auth"] == ("test-project-id", "test-token")

    def test_request_url(self):
        skill, mock_session = self._setup_skill()
        mock_response = Mock()
        mock_response.json.return_value = {"chunks": []}
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response

        skill._search_knowledge_handler({"query": "test"}, {})

        call_args = mock_session.post.call_args
        assert call_args[0][0] == "https://testspace.signalwire.com/api/datasphere/documents/search"

    def test_request_headers(self):
        skill, mock_session = self._setup_skill()
        mock_response = Mock()
        mock_response.json.return_value = {"chunks": []}
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response

        skill._search_knowledge_handler({"query": "test"}, {})

        _, call_kwargs = mock_session.post.call_args
        assert call_kwargs["headers"]["Content-Type"] == "application/json"
        assert call_kwargs["headers"]["Accept"] == "application/json"

    def test_request_timeout_is_30(self):
        skill, mock_session = self._setup_skill()
        mock_response = Mock()
        mock_response.json.return_value = {"chunks": []}
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response

        skill._search_knowledge_handler({"query": "test"}, {})

        _, call_kwargs = mock_session.post.call_args
        assert call_kwargs["timeout"] == 30

    def test_logs_search_request(self):
        skill, mock_session = self._setup_skill()
        mock_response = Mock()
        mock_response.json.return_value = {"chunks": []}
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response

        with patch.object(skill.logger, "info") as mock_info:
            skill._search_knowledge_handler({"query": "my search"}, {})
            mock_info.assert_called_once()
            assert "my search" in mock_info.call_args[0][0]

    def test_logs_timeout_error(self):
        skill, mock_session = self._setup_skill()
        mock_session.post.side_effect = requests.exceptions.Timeout()

        with patch.object(skill.logger, "error") as mock_error:
            skill._search_knowledge_handler({"query": "test"}, {})
            mock_error.assert_called_once()
            assert "timed out" in mock_error.call_args[0][0].lower()

    def test_logs_http_error(self):
        skill, mock_session = self._setup_skill()
        mock_response = Mock()
        http_err = requests.exceptions.HTTPError("500 Server Error")
        mock_response.raise_for_status.side_effect = http_err
        mock_session.post.return_value = mock_response

        with patch.object(skill.logger, "error") as mock_error:
            skill._search_knowledge_handler({"query": "test"}, {})
            mock_error.assert_called_once()
            assert "HTTP error" in mock_error.call_args[0][0]

    def test_logs_generic_error(self):
        skill, mock_session = self._setup_skill()
        mock_session.post.side_effect = ValueError("bad value")

        with patch.object(skill.logger, "error") as mock_error:
            skill._search_knowledge_handler({"query": "test"}, {})
            mock_error.assert_called_once()

    def test_invalid_response_empty_dict(self):
        """An empty dict {} should be treated as no results since chunks is missing."""
        skill, mock_session = self._setup_skill()
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()
        mock_session.post.return_value = mock_response

        result = skill._search_knowledge_handler({"query": "test"}, {})
        assert isinstance(result, SwaigFunctionResult)
        # Should get no_results_message because chunks key yields empty list


# ---------------------------------------------------------------------------
# _format_search_results()
# ---------------------------------------------------------------------------

class TestFormatSearchResults:
    """Tests for the _format_search_results helper."""

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def _setup_skill(self, mock_session_cls):
        skill = _make_skill()
        skill.setup()
        return skill

    def test_single_result_heading(self):
        skill = self._setup_skill()
        output = skill._format_search_results("test", [{"text": "hello"}])
        assert "1 result" in output

    def test_multiple_results_heading(self):
        skill = self._setup_skill()
        chunks = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        output = skill._format_search_results("test", chunks)
        assert "3 results" in output

    def test_chunk_with_text_field(self):
        skill = self._setup_skill()
        output = skill._format_search_results("q", [{"text": "text_content"}])
        assert "text_content" in output

    def test_chunk_with_content_field(self):
        skill = self._setup_skill()
        output = skill._format_search_results("q", [{"content": "content_value"}])
        assert "content_value" in output

    def test_chunk_with_chunk_field(self):
        skill = self._setup_skill()
        output = skill._format_search_results("q", [{"chunk": "chunk_data"}])
        assert "chunk_data" in output

    def test_chunk_unknown_format_falls_back_to_json(self):
        skill = self._setup_skill()
        chunk = {"custom_field": "value123"}
        output = skill._format_search_results("q", [chunk])
        assert "custom_field" in output
        assert "value123" in output

    def test_text_field_takes_priority_over_content(self):
        """If both text and content exist, text should be used (checked first)."""
        skill = self._setup_skill()
        chunk = {"text": "primary", "content": "secondary"}
        output = skill._format_search_results("q", [chunk])
        assert "primary" in output

    def test_result_numbering(self):
        skill = self._setup_skill()
        chunks = [{"text": "first"}, {"text": "second"}]
        output = skill._format_search_results("q", chunks)
        assert "RESULT 1" in output
        assert "RESULT 2" in output

    def test_query_appears_in_output(self):
        skill = self._setup_skill()
        output = skill._format_search_results("my search term", [{"text": "r"}])
        assert "my search term" in output


# ---------------------------------------------------------------------------
# cleanup()
# ---------------------------------------------------------------------------

class TestCleanup:
    """Tests for the cleanup method."""

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_cleanup_closes_session(self, mock_session_cls):
        skill = _make_skill()
        skill.setup()
        mock_session = skill.session
        skill.cleanup()
        mock_session.close.assert_called_once()

    def test_cleanup_no_session_does_not_raise(self):
        """If setup was never called, cleanup should not raise."""
        skill = _make_skill()
        # No setup() called, so no session attribute
        skill.cleanup()  # Should not raise


# ---------------------------------------------------------------------------
# get_hints()
# ---------------------------------------------------------------------------

class TestGetHints:
    """Tests for the get_hints method."""

    def test_returns_empty_list(self):
        skill = _make_skill()
        assert skill.get_hints() == []


# ---------------------------------------------------------------------------
# get_global_data()
# ---------------------------------------------------------------------------

class TestGetGlobalData:
    """Tests for the get_global_data method."""

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_returns_correct_keys(self, mock_session_cls):
        skill = _make_skill()
        skill.setup()
        data = skill.get_global_data()

        assert data["datasphere_enabled"] is True
        assert data["document_id"] == "test-doc-id"
        assert data["knowledge_provider"] == "SignalWire DataSphere"

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_reflects_configured_document_id(self, mock_session_cls):
        skill = _make_skill({"document_id": "custom-doc-123"})
        skill.setup()
        assert skill.get_global_data()["document_id"] == "custom-doc-123"


# ---------------------------------------------------------------------------
# get_prompt_sections()
# ---------------------------------------------------------------------------

class TestGetPromptSections:
    """Tests for the get_prompt_sections method."""

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_returns_one_section(self, mock_session_cls):
        skill = _make_skill()
        skill.setup()
        sections = skill.get_prompt_sections()
        assert len(sections) == 1

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_section_title(self, mock_session_cls):
        skill = _make_skill()
        skill.setup()
        section = skill.get_prompt_sections()[0]
        assert section["title"] == "Knowledge Search Capability"

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_section_references_tool_name(self, mock_session_cls):
        skill = _make_skill()
        skill.setup()
        section = skill.get_prompt_sections()[0]
        assert "search_knowledge" in section["body"]

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_section_references_custom_tool_name(self, mock_session_cls):
        skill = _make_skill({"tool_name": "my_kb"})
        skill.setup()
        section = skill.get_prompt_sections()[0]
        assert "my_kb" in section["body"]
        assert any("my_kb" in bullet for bullet in section["bullets"])

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_section_has_bullets(self, mock_session_cls):
        skill = _make_skill()
        skill.setup()
        section = skill.get_prompt_sections()[0]
        assert "bullets" in section
        assert len(section["bullets"]) > 0


# ---------------------------------------------------------------------------
# Edge cases and integration-style tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case and integration-style tests."""

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_setup_then_register_then_handler_flow(self, mock_session_cls):
        """Full lifecycle: setup -> register -> handle search."""
        skill = _make_skill()
        assert skill.setup() is True
        skill.register_tools()

        # Extract the handler that was registered
        _, kw = skill.agent.define_tool.call_args
        handler = kw["handler"]

        # Set up mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "chunks": [{"text": "Lifecycle answer"}]
        }
        mock_response.raise_for_status = Mock()
        skill.session.post.return_value = mock_response

        result = handler({"query": "lifecycle test"}, {})
        assert isinstance(result, SwaigFunctionResult)
        assert "Lifecycle answer" in result.response

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_api_url_with_special_space_name(self, mock_session_cls):
        skill = _make_skill({"space_name": "my-company-space"})
        skill.setup()
        assert skill.api_url == "https://my-company-space.signalwire.com/api/datasphere/documents/search"

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_handler_strips_query_whitespace(self, mock_session_cls):
        skill = _make_skill()
        skill.setup()

        mock_response = Mock()
        mock_response.json.return_value = {"chunks": []}
        mock_response.raise_for_status = Mock()
        skill.session.post.return_value = mock_response

        skill._search_knowledge_handler({"query": "  padded query  "}, {})

        _, call_kwargs = skill.session.post.call_args
        assert call_kwargs["json"]["query_string"] == "padded query"

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_no_results_message_format_with_query_placeholder_in_invalid_data_path(self, mock_session_cls):
        """When API returns non-dict data and no_results_message has {query} placeholder."""
        skill = _make_skill({"no_results_message": "Sorry, '{query}' not found."})
        skill.setup()

        mock_response = Mock()
        mock_response.json.return_value = None
        mock_response.raise_for_status = Mock()
        skill.session.post.return_value = mock_response

        result = skill._search_knowledge_handler({"query": "test topic"}, {})
        assert result.response == "Sorry, 'test topic' not found."

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_setup_returns_true_only_with_all_required(self, mock_session_cls):
        """Verify setup returns True only when all four required params are present."""
        required = ["space_name", "project_id", "token", "document_id"]
        for missing in required:
            params = {k: f"val_{k}" for k in required}
            params[missing] = ""
            mock_agent = Mock()
            skill = DataSphereSkill(agent=mock_agent, params=params)
            assert skill.setup() is False, f"Should fail when {missing} is empty"

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_multiple_instances_different_tool_names(self, mock_session_cls):
        """Two instances with different tool_names should have different keys."""
        skill_a = _make_skill({"tool_name": "search_faq"})
        skill_b = _make_skill({"tool_name": "search_docs"})

        assert skill_a.get_instance_key() != skill_b.get_instance_key()
        assert "search_faq" in skill_a.get_instance_key()
        assert "search_docs" in skill_b.get_instance_key()

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_response_with_dict_no_chunks_key(self, mock_session_cls):
        """API returns a valid dict but without the 'chunks' key."""
        skill = _make_skill()
        skill.setup()

        mock_response = Mock()
        mock_response.json.return_value = {"status": "ok", "data": []}
        mock_response.raise_for_status = Mock()
        skill.session.post.return_value = mock_response

        result = skill._search_knowledge_handler({"query": "test"}, {})
        assert isinstance(result, SwaigFunctionResult)
        # Should hit the "not chunks" path and return no_results_message

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_cleanup_after_setup(self, mock_session_cls):
        """cleanup should close the session created during setup."""
        skill = _make_skill()
        skill.setup()
        session_mock = skill.session
        skill.cleanup()
        session_mock.close.assert_called_once()

    @patch("signalwire_agents.skills.datasphere.skill.requests.Session")
    def test_connection_error(self, mock_session_cls):
        """ConnectionError should be caught by the generic Exception handler."""
        skill = _make_skill()
        skill.setup()
        skill.session.post.side_effect = requests.exceptions.ConnectionError("refused")

        result = skill._search_knowledge_handler({"query": "test"}, {})
        assert isinstance(result, SwaigFunctionResult)
        assert "error" in result.response.lower()
