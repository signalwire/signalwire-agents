"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for WikipediaSearchSkill
"""

import pytest
import requests
from unittest.mock import Mock, patch, MagicMock, call

from signalwire_agents.skills.wikipedia_search.skill import WikipediaSearchSkill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill(params=None):
    """Create a WikipediaSearchSkill instance with a mocked agent."""
    mock_agent = Mock()
    mock_agent.define_tool = Mock()
    skill = WikipediaSearchSkill(agent=mock_agent, params=params)
    return skill


def _setup_skill(params=None):
    """Create *and* setup a WikipediaSearchSkill instance."""
    skill = _make_skill(params)
    with patch.object(skill, "validate_packages", return_value=True):
        result = skill.setup()
    assert result is True
    return skill


def _mock_search_response(titles):
    """Build a mock JSON response for Wikipedia search API."""
    return {
        "query": {
            "search": [{"title": t} for t in titles]
        }
    }


def _mock_extract_response(title, extract):
    """Build a mock JSON response for Wikipedia extract API."""
    return {
        "query": {
            "pages": {
                "12345": {
                    "title": title,
                    "extract": extract
                }
            }
        }
    }


def _mock_extract_response_empty_pages():
    """Build a mock JSON response with no pages."""
    return {
        "query": {
            "pages": {}
        }
    }


# ===========================================================================
# Class-Level Metadata
# ===========================================================================

class TestWikipediaSearchSkillMetadata:
    """Verify class-level attributes and metadata."""

    def test_skill_name(self):
        assert WikipediaSearchSkill.SKILL_NAME == "wikipedia_search"

    def test_skill_description(self):
        assert WikipediaSearchSkill.SKILL_DESCRIPTION == (
            "Search Wikipedia for information about a topic and get article summaries"
        )

    def test_skill_version(self):
        assert WikipediaSearchSkill.SKILL_VERSION == "1.0.0"

    def test_required_packages(self):
        assert WikipediaSearchSkill.REQUIRED_PACKAGES == ["requests"]

    def test_required_env_vars(self):
        assert WikipediaSearchSkill.REQUIRED_ENV_VARS == []

    def test_supports_multiple_instances(self):
        assert WikipediaSearchSkill.SUPPORTS_MULTIPLE_INSTANCES is False


# ===========================================================================
# Initialization
# ===========================================================================

class TestWikipediaSearchSkillInit:
    """Test __init__ behaviour inherited from SkillBase."""

    def test_init_stores_agent(self):
        mock_agent = Mock()
        skill = WikipediaSearchSkill(agent=mock_agent, params=None)
        assert skill.agent is mock_agent

    def test_init_default_params(self):
        skill = _make_skill()
        assert skill.params == {}

    def test_init_custom_params(self):
        params = {"num_results": 3, "no_results_message": "Nothing found."}
        skill = _make_skill(params)
        assert skill.params["num_results"] == 3
        assert skill.params["no_results_message"] == "Nothing found."

    def test_init_extracts_swaig_fields(self):
        params = {"swaig_fields": {"fillers": {"en-US": ["hmm"]}}, "num_results": 2}
        skill = _make_skill(params)
        assert skill.swaig_fields == {"fillers": {"en-US": ["hmm"]}}
        # swaig_fields should be popped from params
        assert "swaig_fields" not in skill.params

    def test_init_logger_name(self):
        skill = _make_skill()
        assert skill.logger.name == "skill.wikipedia_search"


# ===========================================================================
# get_parameter_schema
# ===========================================================================

class TestParameterSchema:
    """Test the get_parameter_schema class method."""

    def test_returns_dict(self):
        schema = WikipediaSearchSkill.get_parameter_schema()
        assert isinstance(schema, dict)

    def test_contains_num_results(self):
        schema = WikipediaSearchSkill.get_parameter_schema()
        assert "num_results" in schema
        assert schema["num_results"]["type"] == "integer"
        assert schema["num_results"]["default"] == 1
        assert schema["num_results"]["minimum"] == 1
        assert schema["num_results"]["maximum"] == 5

    def test_contains_no_results_message(self):
        schema = WikipediaSearchSkill.get_parameter_schema()
        assert "no_results_message" in schema
        assert schema["no_results_message"]["type"] == "string"
        assert "{query}" in schema["no_results_message"]["default"]

    def test_contains_base_swaig_fields(self):
        schema = WikipediaSearchSkill.get_parameter_schema()
        assert "swaig_fields" in schema

    def test_no_tool_name_for_single_instance_skill(self):
        """Since SUPPORTS_MULTIPLE_INSTANCES is False, tool_name should not be present."""
        schema = WikipediaSearchSkill.get_parameter_schema()
        assert "tool_name" not in schema


# ===========================================================================
# setup()
# ===========================================================================

class TestSetup:
    """Test the setup() method."""

    def test_setup_default_params(self):
        skill = _make_skill()
        with patch.object(skill, "validate_packages", return_value=True):
            result = skill.setup()
        assert result is True
        assert skill.num_results == 1
        assert "{query}" in skill.no_results_message

    def test_setup_custom_num_results(self):
        skill = _make_skill({"num_results": 4})
        with patch.object(skill, "validate_packages", return_value=True):
            skill.setup()
        assert skill.num_results == 4

    def test_setup_num_results_clamped_to_minimum(self):
        """num_results of 0 or negative should be clamped to 1."""
        skill = _make_skill({"num_results": 0})
        with patch.object(skill, "validate_packages", return_value=True):
            skill.setup()
        assert skill.num_results == 1

    def test_setup_num_results_negative_clamped(self):
        skill = _make_skill({"num_results": -5})
        with patch.object(skill, "validate_packages", return_value=True):
            skill.setup()
        assert skill.num_results == 1

    def test_setup_custom_no_results_message(self):
        msg = "Sorry, no articles for '{query}'."
        skill = _make_skill({"no_results_message": msg})
        with patch.object(skill, "validate_packages", return_value=True):
            skill.setup()
        assert skill.no_results_message == msg

    def test_setup_empty_no_results_message_uses_default(self):
        """An empty string for no_results_message should fall back to default."""
        skill = _make_skill({"no_results_message": ""})
        with patch.object(skill, "validate_packages", return_value=True):
            skill.setup()
        assert "{query}" in skill.no_results_message
        assert len(skill.no_results_message) > 0

    def test_setup_none_no_results_message_uses_default(self):
        skill = _make_skill({"no_results_message": None})
        with patch.object(skill, "validate_packages", return_value=True):
            skill.setup()
        assert "{query}" in skill.no_results_message

    def test_setup_returns_false_when_packages_missing(self):
        skill = _make_skill()
        with patch.object(skill, "validate_packages", return_value=False):
            result = skill.setup()
        assert result is False

    def test_setup_logs_info(self):
        skill = _make_skill({"num_results": 3})
        with patch.object(skill, "validate_packages", return_value=True):
            with patch.object(skill.logger, "info") as mock_info:
                skill.setup()
                mock_info.assert_called_once()
                assert "3" in mock_info.call_args[0][0]


# ===========================================================================
# register_tools()
# ===========================================================================

class TestRegisterTools:
    """Test the register_tools() method."""

    def test_register_tools_calls_define_tool(self):
        skill = _setup_skill()
        skill.register_tools()
        skill.agent.define_tool.assert_called_once()

    def test_register_tools_tool_name(self):
        skill = _setup_skill()
        skill.register_tools()
        kwargs = skill.agent.define_tool.call_args
        assert kwargs[1]["name"] == "search_wiki" or kwargs.kwargs["name"] == "search_wiki"

    def test_register_tools_tool_has_query_parameter(self):
        skill = _setup_skill()
        skill.register_tools()
        call_kwargs = skill.agent.define_tool.call_args
        # define_tool is called via self.define_tool which merges swaig_fields
        params = call_kwargs.kwargs.get("parameters") or call_kwargs[1].get("parameters")
        assert "query" in params
        assert params["query"]["type"] == "string"

    def test_register_tools_tool_has_handler(self):
        skill = _setup_skill()
        skill.register_tools()
        call_kwargs = skill.agent.define_tool.call_args
        handler = call_kwargs.kwargs.get("handler") or call_kwargs[1].get("handler")
        assert handler is not None
        assert callable(handler)

    def test_register_tools_with_swaig_fields(self):
        """swaig_fields should be merged into the define_tool call."""
        skill = _make_skill({"swaig_fields": {"meta_data": {"token": "abc"}}})
        with patch.object(skill, "validate_packages", return_value=True):
            skill.setup()
        skill.register_tools()
        call_kwargs = skill.agent.define_tool.call_args
        # The merged kwargs should include the swaig_fields
        assert call_kwargs.kwargs.get("meta_data") == {"token": "abc"} or \
               call_kwargs[1].get("meta_data") == {"token": "abc"}


# ===========================================================================
# _search_wiki_handler()
# ===========================================================================

class TestSearchWikiHandler:
    """Test the _search_wiki_handler method."""

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_handler_empty_query(self, mock_get):
        skill = _setup_skill()
        result = skill._search_wiki_handler({"query": ""}, {})
        # Should return a SwaigFunctionResult without calling the API
        mock_get.assert_not_called()
        assert result.response == "Please provide a search query for Wikipedia."

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_handler_whitespace_query(self, mock_get):
        skill = _setup_skill()
        result = skill._search_wiki_handler({"query": "   "}, {})
        mock_get.assert_not_called()
        assert result.response == "Please provide a search query for Wikipedia."

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_handler_missing_query_key(self, mock_get):
        skill = _setup_skill()
        result = skill._search_wiki_handler({}, {})
        mock_get.assert_not_called()
        assert result.response == "Please provide a search query for Wikipedia."

    @patch.object(WikipediaSearchSkill, "search_wiki", return_value="Python is a language.")
    def test_handler_delegates_to_search_wiki(self, mock_search):
        skill = _setup_skill()
        result = skill._search_wiki_handler({"query": "Python"}, {})
        mock_search.assert_called_once_with("Python")
        assert result.response == "Python is a language."

    @patch.object(WikipediaSearchSkill, "search_wiki", return_value="Some content")
    def test_handler_returns_swaig_function_result(self, mock_search):
        from signalwire_agents.core.function_result import SwaigFunctionResult
        skill = _setup_skill()
        result = skill._search_wiki_handler({"query": "test"}, {})
        assert isinstance(result, SwaigFunctionResult)

    @patch.object(WikipediaSearchSkill, "search_wiki", return_value="Trimmed result")
    def test_handler_strips_query(self, mock_search):
        skill = _setup_skill()
        skill._search_wiki_handler({"query": "  Python  "}, {})
        mock_search.assert_called_once_with("Python")


# ===========================================================================
# search_wiki()  --  Single Result
# ===========================================================================

class TestSearchWikiSingleResult:
    """Test search_wiki with a single result (default num_results=1)."""

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_single_result_success(self, mock_get):
        skill = _setup_skill()

        search_resp = Mock()
        search_resp.json.return_value = _mock_search_response(["Python (programming language)"])
        search_resp.raise_for_status = Mock()

        extract_resp = Mock()
        extract_resp.json.return_value = _mock_extract_response(
            "Python (programming language)",
            "Python is a high-level programming language."
        )
        extract_resp.raise_for_status = Mock()

        mock_get.side_effect = [search_resp, extract_resp]

        result = skill.search_wiki("Python programming")
        assert "**Python (programming language)**" in result
        assert "Python is a high-level programming language." in result

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_search_url_contains_encoded_query(self, mock_get):
        skill = _setup_skill()

        search_resp = Mock()
        search_resp.json.return_value = _mock_search_response([])
        search_resp.raise_for_status = Mock()
        mock_get.return_value = search_resp

        skill.search_wiki("C++ language")
        called_url = mock_get.call_args_list[0][0][0]
        # + should be URL-encoded as %2B
        assert "C%2B%2B" in called_url

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_search_url_contains_srlimit(self, mock_get):
        skill = _setup_skill({"num_results": 3})

        search_resp = Mock()
        search_resp.json.return_value = _mock_search_response([])
        search_resp.raise_for_status = Mock()
        mock_get.return_value = search_resp

        skill.search_wiki("test")
        called_url = mock_get.call_args_list[0][0][0]
        assert "srlimit=3" in called_url

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_timeout_is_10_seconds(self, mock_get):
        skill = _setup_skill()

        search_resp = Mock()
        search_resp.json.return_value = _mock_search_response([])
        search_resp.raise_for_status = Mock()
        mock_get.return_value = search_resp

        skill.search_wiki("test")
        assert mock_get.call_args[1]["timeout"] == 10


# ===========================================================================
# search_wiki()  --  Multiple Results
# ===========================================================================

class TestSearchWikiMultipleResults:
    """Test search_wiki when num_results > 1."""

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_multiple_results_joined_with_separator(self, mock_get):
        skill = _setup_skill({"num_results": 2})

        search_resp = Mock()
        search_resp.json.return_value = _mock_search_response(["Article One", "Article Two"])
        search_resp.raise_for_status = Mock()

        extract_resp_1 = Mock()
        extract_resp_1.json.return_value = _mock_extract_response("Article One", "Content one.")
        extract_resp_1.raise_for_status = Mock()

        extract_resp_2 = Mock()
        extract_resp_2.json.return_value = _mock_extract_response("Article Two", "Content two.")
        extract_resp_2.raise_for_status = Mock()

        mock_get.side_effect = [search_resp, extract_resp_1, extract_resp_2]

        result = skill.search_wiki("articles")
        assert "**Article One**" in result
        assert "**Article Two**" in result
        assert "=" * 50 in result

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_results_limited_to_num_results(self, mock_get):
        """Even if search returns more titles, only num_results extracts should be fetched."""
        skill = _setup_skill({"num_results": 1})

        search_resp = Mock()
        search_resp.json.return_value = _mock_search_response(["Title A", "Title B", "Title C"])
        search_resp.raise_for_status = Mock()

        extract_resp = Mock()
        extract_resp.json.return_value = _mock_extract_response("Title A", "Extract A.")
        extract_resp.raise_for_status = Mock()

        mock_get.side_effect = [search_resp, extract_resp]

        result = skill.search_wiki("many results")
        # Only 1 extract should be fetched, so requests.get called twice total
        assert mock_get.call_count == 2
        assert "**Title A**" in result


# ===========================================================================
# search_wiki()  --  No Results / Empty Content
# ===========================================================================

class TestSearchWikiNoResults:
    """Test search_wiki edge cases for empty or missing data."""

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_no_search_results(self, mock_get):
        skill = _setup_skill()

        search_resp = Mock()
        search_resp.json.return_value = {"query": {"search": []}}
        search_resp.raise_for_status = Mock()
        mock_get.return_value = search_resp

        result = skill.search_wiki("xyznonexistent")
        assert "xyznonexistent" in result
        assert "couldn't find" in result.lower() or "rephrasing" in result.lower()

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_no_search_key_in_response(self, mock_get):
        skill = _setup_skill()

        search_resp = Mock()
        search_resp.json.return_value = {}
        search_resp.raise_for_status = Mock()
        mock_get.return_value = search_resp

        result = skill.search_wiki("broken")
        # Should fall through to no_results_message
        assert "broken" in result

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_empty_extract_shows_no_summary(self, mock_get):
        skill = _setup_skill()

        search_resp = Mock()
        search_resp.json.return_value = _mock_search_response(["Some Title"])
        search_resp.raise_for_status = Mock()

        extract_resp = Mock()
        extract_resp.json.return_value = _mock_extract_response("Some Title", "")
        extract_resp.raise_for_status = Mock()

        mock_get.side_effect = [search_resp, extract_resp]

        result = skill.search_wiki("some title")
        assert "**Some Title**" in result
        assert "No summary available" in result

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_whitespace_only_extract_shows_no_summary(self, mock_get):
        skill = _setup_skill()

        search_resp = Mock()
        search_resp.json.return_value = _mock_search_response(["Blank Article"])
        search_resp.raise_for_status = Mock()

        extract_resp = Mock()
        extract_resp.json.return_value = _mock_extract_response("Blank Article", "   ")
        extract_resp.raise_for_status = Mock()

        mock_get.side_effect = [search_resp, extract_resp]

        result = skill.search_wiki("blank article")
        assert "No summary available" in result

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_empty_pages_dict(self, mock_get):
        """If the extract API returns empty pages, no article is appended."""
        skill = _setup_skill()

        search_resp = Mock()
        search_resp.json.return_value = _mock_search_response(["Ghost Article"])
        search_resp.raise_for_status = Mock()

        extract_resp = Mock()
        extract_resp.json.return_value = _mock_extract_response_empty_pages()
        extract_resp.raise_for_status = Mock()

        mock_get.side_effect = [search_resp, extract_resp]

        result = skill.search_wiki("ghost")
        # No articles appended, falls back to no_results_message
        assert "ghost" in result

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_custom_no_results_message_with_query(self, mock_get):
        msg = "Nothing for '{query}', sorry!"
        skill = _setup_skill({"no_results_message": msg})

        search_resp = Mock()
        search_resp.json.return_value = {"query": {"search": []}}
        search_resp.raise_for_status = Mock()
        mock_get.return_value = search_resp

        result = skill.search_wiki("foobar")
        assert result == "Nothing for 'foobar', sorry!"


# ===========================================================================
# search_wiki()  --  Error Handling
# ===========================================================================

class TestSearchWikiErrorHandling:
    """Test error handling in search_wiki."""

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_request_exception_on_search(self, mock_get):
        skill = _setup_skill()
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        result = skill.search_wiki("test")
        assert "Error accessing Wikipedia" in result
        assert "Connection refused" in result

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_timeout_exception_on_search(self, mock_get):
        skill = _setup_skill()
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

        result = skill.search_wiki("test")
        assert "Error accessing Wikipedia" in result

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_http_error_on_search(self, mock_get):
        skill = _setup_skill()
        resp = Mock()
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_get.return_value = resp

        result = skill.search_wiki("test")
        assert "Error accessing Wikipedia" in result

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_request_exception_on_extract(self, mock_get):
        skill = _setup_skill()

        search_resp = Mock()
        search_resp.json.return_value = _mock_search_response(["Title"])
        search_resp.raise_for_status = Mock()

        extract_resp = Mock()
        extract_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("404")

        mock_get.side_effect = [search_resp, extract_resp]

        result = skill.search_wiki("test")
        assert "Error accessing Wikipedia" in result

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_generic_exception(self, mock_get):
        skill = _setup_skill()
        mock_get.side_effect = ValueError("Unexpected error")

        result = skill.search_wiki("test")
        assert "Error searching Wikipedia" in result
        assert "Unexpected error" in result

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_json_decode_error_on_search(self, mock_get):
        skill = _setup_skill()
        resp = Mock()
        resp.raise_for_status = Mock()
        resp.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = resp

        result = skill.search_wiki("test")
        assert "Error searching Wikipedia" in result

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_json_decode_error_on_extract(self, mock_get):
        skill = _setup_skill()

        search_resp = Mock()
        search_resp.json.return_value = _mock_search_response(["Title"])
        search_resp.raise_for_status = Mock()

        extract_resp = Mock()
        extract_resp.raise_for_status = Mock()
        extract_resp.json.side_effect = ValueError("Bad JSON on extract")

        mock_get.side_effect = [search_resp, extract_resp]

        result = skill.search_wiki("test")
        assert "Error searching Wikipedia" in result


# ===========================================================================
# search_wiki()  --  Response structure edge cases
# ===========================================================================

class TestSearchWikiResponseStructure:
    """Test subtle response structure edge cases."""

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_single_article_no_separator(self, mock_get):
        """A single result should NOT contain the separator."""
        skill = _setup_skill()

        search_resp = Mock()
        search_resp.json.return_value = _mock_search_response(["Only One"])
        search_resp.raise_for_status = Mock()

        extract_resp = Mock()
        extract_resp.json.return_value = _mock_extract_response("Only One", "Content here.")
        extract_resp.raise_for_status = Mock()

        mock_get.side_effect = [search_resp, extract_resp]

        result = skill.search_wiki("only one")
        assert "=" * 50 not in result
        assert result == "**Only One**\n\nContent here."

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_extract_with_leading_trailing_whitespace_stripped(self, mock_get):
        skill = _setup_skill()

        search_resp = Mock()
        search_resp.json.return_value = _mock_search_response(["Whitespace Test"])
        search_resp.raise_for_status = Mock()

        extract_resp = Mock()
        extract_resp.json.return_value = _mock_extract_response(
            "Whitespace Test", "\n  Some content with whitespace  \n"
        )
        extract_resp.raise_for_status = Mock()

        mock_get.side_effect = [search_resp, extract_resp]

        result = skill.search_wiki("whitespace test")
        assert "**Whitespace Test**\n\nSome content with whitespace" in result

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_multiple_results_some_empty_extracts(self, mock_get):
        """Articles with empty extracts should still appear with 'No summary' message."""
        skill = _setup_skill({"num_results": 2})

        search_resp = Mock()
        search_resp.json.return_value = _mock_search_response(["Full Article", "Empty Article"])
        search_resp.raise_for_status = Mock()

        extract_resp_1 = Mock()
        extract_resp_1.json.return_value = _mock_extract_response("Full Article", "Has content.")
        extract_resp_1.raise_for_status = Mock()

        extract_resp_2 = Mock()
        extract_resp_2.json.return_value = _mock_extract_response("Empty Article", "")
        extract_resp_2.raise_for_status = Mock()

        mock_get.side_effect = [search_resp, extract_resp_1, extract_resp_2]

        result = skill.search_wiki("mixed")
        assert "**Full Article**" in result
        assert "Has content." in result
        assert "**Empty Article**" in result
        assert "No summary available" in result
        assert "=" * 50 in result

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_page_missing_extract_key(self, mock_get):
        """If a page has no 'extract' key at all, it should show 'No summary available'."""
        skill = _setup_skill()

        search_resp = Mock()
        search_resp.json.return_value = _mock_search_response(["No Extract Key"])
        search_resp.raise_for_status = Mock()

        extract_resp = Mock()
        extract_resp.json.return_value = {
            "query": {
                "pages": {
                    "99999": {
                        "title": "No Extract Key"
                        # No 'extract' key at all
                    }
                }
            }
        }
        extract_resp.raise_for_status = Mock()

        mock_get.side_effect = [search_resp, extract_resp]

        result = skill.search_wiki("no extract")
        assert "**No Extract Key**" in result
        assert "No summary available" in result


# ===========================================================================
# get_prompt_sections()
# ===========================================================================

class TestGetPromptSections:
    """Test get_prompt_sections method."""

    def test_returns_list(self):
        skill = _setup_skill()
        sections = skill.get_prompt_sections()
        assert isinstance(sections, list)

    def test_returns_one_section(self):
        skill = _setup_skill()
        sections = skill.get_prompt_sections()
        assert len(sections) == 1

    def test_section_has_title(self):
        skill = _setup_skill()
        section = skill.get_prompt_sections()[0]
        assert section["title"] == "Wikipedia Search"

    def test_section_body_includes_num_results(self):
        skill = _setup_skill({"num_results": 3})
        section = skill.get_prompt_sections()[0]
        assert "3" in section["body"]
        assert "search_wiki" in section["body"]

    def test_section_has_bullets(self):
        skill = _setup_skill()
        section = skill.get_prompt_sections()[0]
        assert isinstance(section["bullets"], list)
        assert len(section["bullets"]) == 3

    def test_section_bullets_mention_search_wiki(self):
        skill = _setup_skill()
        section = skill.get_prompt_sections()[0]
        assert any("search_wiki" in b for b in section["bullets"])


# ===========================================================================
# get_hints()
# ===========================================================================

class TestGetHints:
    """Test get_hints method."""

    def test_returns_empty_list(self):
        skill = _setup_skill()
        hints = skill.get_hints()
        assert hints == []

    def test_returns_list_type(self):
        skill = _setup_skill()
        assert isinstance(skill.get_hints(), list)


# ===========================================================================
# get_instance_key()
# ===========================================================================

class TestGetInstanceKey:
    """Test instance key behaviour for single-instance skill."""

    def test_returns_skill_name(self):
        skill = _make_skill()
        assert skill.get_instance_key() == "wikipedia_search"

    def test_ignores_tool_name_param(self):
        """Since SUPPORTS_MULTIPLE_INSTANCES is False, tool_name should be ignored."""
        skill = _make_skill({"tool_name": "custom_name"})
        assert skill.get_instance_key() == "wikipedia_search"


# ===========================================================================
# Integration-style tests (handler -> search_wiki flow)
# ===========================================================================

class TestHandlerToSearchIntegration:
    """Test the full handler -> search_wiki pipeline with mocked HTTP."""

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_full_flow_success(self, mock_get):
        skill = _setup_skill()

        search_resp = Mock()
        search_resp.json.return_value = _mock_search_response(["Albert Einstein"])
        search_resp.raise_for_status = Mock()

        extract_resp = Mock()
        extract_resp.json.return_value = _mock_extract_response(
            "Albert Einstein",
            "Albert Einstein was a German-born theoretical physicist."
        )
        extract_resp.raise_for_status = Mock()

        mock_get.side_effect = [search_resp, extract_resp]

        result = skill._search_wiki_handler({"query": "Einstein"}, {"call_id": "test"})
        assert "Albert Einstein" in result.response
        assert "theoretical physicist" in result.response

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_full_flow_no_results(self, mock_get):
        skill = _setup_skill()

        search_resp = Mock()
        search_resp.json.return_value = {"query": {"search": []}}
        search_resp.raise_for_status = Mock()
        mock_get.return_value = search_resp

        result = skill._search_wiki_handler({"query": "zzznonexistenttopic"}, {})
        assert "zzznonexistenttopic" in result.response

    @patch("signalwire_agents.skills.wikipedia_search.skill.requests.get")
    def test_full_flow_api_error(self, mock_get):
        skill = _setup_skill()
        mock_get.side_effect = requests.exceptions.ConnectionError("Network down")

        result = skill._search_wiki_handler({"query": "anything"}, {})
        assert "Error accessing Wikipedia" in result.response
