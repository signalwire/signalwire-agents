"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for the WebSearch skill module
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

import requests

from signalwire_agents.skills.web_search.skill import WebSearchSkill, GoogleSearchScraper
from signalwire_agents.core.function_result import SwaigFunctionResult


def _make_skill(params=None):
    """
    Helper to create a WebSearchSkill with a mocked agent.
    Provides sensible defaults for all required parameters.
    """
    default_params = {
        "api_key": "test-api-key",
        "search_engine_id": "test-engine-id",
    }
    if params is not None:
        default_params.update(params)

    mock_agent = Mock()
    mock_agent.define_tool = Mock()
    skill = WebSearchSkill(agent=mock_agent, params=default_params)
    return skill


# ---------------------------------------------------------------------------
# Class-level attributes
# ---------------------------------------------------------------------------

class TestWebSearchSkillClassAttributes:
    """Verify class-level constants and metadata."""

    def test_skill_name(self):
        assert WebSearchSkill.SKILL_NAME == "web_search"

    def test_skill_description(self):
        assert WebSearchSkill.SKILL_DESCRIPTION == "Search the web for information using Google Custom Search API"

    def test_skill_version(self):
        assert WebSearchSkill.SKILL_VERSION == "2.0.0"

    def test_required_packages(self):
        assert WebSearchSkill.REQUIRED_PACKAGES == ["bs4", "requests"]

    def test_required_env_vars(self):
        assert WebSearchSkill.REQUIRED_ENV_VARS == []

    def test_supports_multiple_instances(self):
        assert WebSearchSkill.SUPPORTS_MULTIPLE_INSTANCES is True


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestWebSearchSkillInit:
    """Tests for __init__ (inherited from SkillBase)."""

    def test_agent_is_stored(self):
        mock_agent = Mock()
        skill = WebSearchSkill(agent=mock_agent, params={"api_key": "k"})
        assert skill.agent is mock_agent

    def test_params_stored(self):
        params = {"api_key": "mykey", "search_engine_id": "myid"}
        skill = WebSearchSkill(agent=Mock(), params=params)
        assert skill.params["api_key"] == "mykey"
        assert skill.params["search_engine_id"] == "myid"

    def test_params_default_to_empty_dict(self):
        skill = WebSearchSkill(agent=Mock())
        assert skill.params == {}

    def test_logger_created(self):
        skill = WebSearchSkill(agent=Mock())
        assert skill.logger is not None
        assert skill.logger.name == "signalwire_agents.skills.web_search"

    def test_swaig_fields_extracted_from_params(self):
        params = {"swaig_fields": {"meta_data": {"x": 1}}, "api_key": "k"}
        skill = WebSearchSkill(agent=Mock(), params=params)
        assert skill.swaig_fields == {"meta_data": {"x": 1}}
        assert "swaig_fields" not in skill.params

    def test_swaig_fields_default_empty(self):
        skill = WebSearchSkill(agent=Mock(), params={"api_key": "k"})
        assert skill.swaig_fields == {}


# ---------------------------------------------------------------------------
# get_parameter_schema
# ---------------------------------------------------------------------------

class TestGetParameterSchema:
    """Tests for the class method get_parameter_schema."""

    def test_contains_required_params(self):
        schema = WebSearchSkill.get_parameter_schema()
        for key in ("api_key", "search_engine_id"):
            assert key in schema, f"Missing required param: {key}"
            assert schema[key]["required"] is True

    def test_contains_optional_params(self):
        schema = WebSearchSkill.get_parameter_schema()
        for key in ("num_results", "delay", "max_content_length",
                     "oversample_factor", "min_quality_score", "no_results_message"):
            assert key in schema, f"Missing optional param: {key}"
            assert schema[key]["required"] is False

    def test_api_key_is_hidden(self):
        schema = WebSearchSkill.get_parameter_schema()
        assert schema["api_key"].get("hidden") is True

    def test_search_engine_id_is_hidden(self):
        schema = WebSearchSkill.get_parameter_schema()
        assert schema["search_engine_id"].get("hidden") is True

    def test_api_key_env_var(self):
        schema = WebSearchSkill.get_parameter_schema()
        assert schema["api_key"].get("env_var") == "GOOGLE_SEARCH_API_KEY"

    def test_search_engine_id_env_var(self):
        schema = WebSearchSkill.get_parameter_schema()
        assert schema["search_engine_id"].get("env_var") == "GOOGLE_SEARCH_ENGINE_ID"

    def test_num_results_defaults(self):
        schema = WebSearchSkill.get_parameter_schema()
        assert schema["num_results"]["default"] == 3
        assert schema["num_results"]["min"] == 1
        assert schema["num_results"]["max"] == 10

    def test_delay_defaults(self):
        schema = WebSearchSkill.get_parameter_schema()
        assert schema["delay"]["default"] == 0.5
        assert schema["delay"]["min"] == 0

    def test_max_content_length_defaults(self):
        schema = WebSearchSkill.get_parameter_schema()
        assert schema["max_content_length"]["default"] == 32768
        assert schema["max_content_length"]["min"] == 1000

    def test_oversample_factor_defaults(self):
        schema = WebSearchSkill.get_parameter_schema()
        assert schema["oversample_factor"]["default"] == 2.5
        assert schema["oversample_factor"]["min"] == 1.0
        assert schema["oversample_factor"]["max"] == 3.5

    def test_min_quality_score_defaults(self):
        schema = WebSearchSkill.get_parameter_schema()
        assert schema["min_quality_score"]["default"] == 0.3
        assert schema["min_quality_score"]["min"] == 0.0
        assert schema["min_quality_score"]["max"] == 1.0

    def test_includes_base_class_swaig_fields(self):
        schema = WebSearchSkill.get_parameter_schema()
        assert "swaig_fields" in schema

    def test_includes_tool_name_because_multi_instance(self):
        schema = WebSearchSkill.get_parameter_schema()
        assert "tool_name" in schema


# ---------------------------------------------------------------------------
# get_instance_key
# ---------------------------------------------------------------------------

class TestGetInstanceKey:
    """Tests for get_instance_key."""

    def test_default_instance_key(self):
        skill = _make_skill()
        # Default: search_engine_id="test-engine-id", tool_name="web_search"
        key = skill.get_instance_key()
        assert "web_search" in key
        assert "test-engine-id" in key

    def test_custom_tool_name_instance_key(self):
        skill = _make_skill({"tool_name": "my_search"})
        key = skill.get_instance_key()
        assert "my_search" in key

    def test_custom_search_engine_id_in_instance_key(self):
        skill = _make_skill({"search_engine_id": "custom-engine"})
        key = skill.get_instance_key()
        assert "custom-engine" in key

    def test_different_instances_have_different_keys(self):
        skill_a = _make_skill({"tool_name": "search_news"})
        skill_b = _make_skill({"tool_name": "search_docs"})
        assert skill_a.get_instance_key() != skill_b.get_instance_key()


# ---------------------------------------------------------------------------
# setup()
# ---------------------------------------------------------------------------

class TestSetup:
    """Tests for the setup method."""

    def test_setup_success_all_required(self):
        skill = _make_skill()
        result = skill.setup()
        assert result is True
        assert skill.api_key == "test-api-key"
        assert skill.search_engine_id == "test-engine-id"

    def test_setup_creates_search_scraper(self):
        skill = _make_skill()
        skill.setup()
        assert isinstance(skill.search_scraper, GoogleSearchScraper)
        assert skill.search_scraper.api_key == "test-api-key"
        assert skill.search_scraper.search_engine_id == "test-engine-id"

    def test_setup_optional_defaults(self):
        skill = _make_skill()
        skill.setup()
        assert skill.default_num_results == 3
        assert skill.default_delay == 0.5
        assert skill.max_content_length == 32768
        assert skill.oversample_factor == 2.5
        assert skill.min_quality_score == 0.3
        assert skill.tool_name == "web_search"
        assert "{query}" in skill.no_results_message

    def test_setup_custom_optional_values(self):
        skill = _make_skill({
            "num_results": 5,
            "delay": 1.0,
            "max_content_length": 16384,
            "oversample_factor": 3.0,
            "min_quality_score": 0.5,
            "tool_name": "my_search",
            "no_results_message": "Nothing found for '{query}'.",
        })
        skill.setup()
        assert skill.default_num_results == 5
        assert skill.default_delay == 1.0
        assert skill.max_content_length == 16384
        assert skill.oversample_factor == 3.0
        assert skill.min_quality_score == 0.5
        assert skill.tool_name == "my_search"
        assert skill.no_results_message == "Nothing found for '{query}'."

    def test_setup_missing_api_key(self):
        skill = _make_skill({"api_key": ""})
        result = skill.setup()
        assert result is False

    def test_setup_missing_search_engine_id(self):
        skill = _make_skill({"search_engine_id": ""})
        result = skill.setup()
        assert result is False

    def test_setup_missing_multiple_params(self):
        mock_agent = Mock()
        mock_agent.define_tool = Mock()
        skill = WebSearchSkill(agent=mock_agent, params={})
        result = skill.setup()
        assert result is False

    def test_setup_missing_param_none_value(self):
        skill = _make_skill({"api_key": None})
        result = skill.setup()
        assert result is False

    def test_setup_logs_error_on_missing_params(self):
        skill = _make_skill({"api_key": ""})
        with patch.object(skill.logger, "error") as mock_error:
            skill.setup()
            mock_error.assert_called_once()
            assert "api_key" in mock_error.call_args[0][0]

    def test_setup_scraper_max_content_length_passed(self):
        skill = _make_skill({"max_content_length": 10000})
        skill.setup()
        assert skill.search_scraper.max_content_length == 10000


# ---------------------------------------------------------------------------
# register_tools()
# ---------------------------------------------------------------------------

class TestRegisterTools:
    """Tests for register_tools method."""

    def test_register_tools_calls_define_tool(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        skill.agent.define_tool.assert_called_once()

    def test_register_tools_default_name(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        _, kw = skill.agent.define_tool.call_args
        assert kw["name"] == "web_search"

    def test_register_tools_custom_name(self):
        skill = _make_skill({"tool_name": "news_search"})
        skill.setup()
        skill.register_tools()
        _, kw = skill.agent.define_tool.call_args
        assert kw["name"] == "news_search"

    def test_register_tools_has_query_parameter(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        _, kw = skill.agent.define_tool.call_args
        assert "query" in kw["parameters"]
        assert kw["parameters"]["query"]["type"] == "string"

    def test_register_tools_handler_is_callable(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        _, kw = skill.agent.define_tool.call_args
        assert callable(kw["handler"])

    def test_register_tools_merges_swaig_fields(self):
        """swaig_fields from params should be merged into define_tool call."""
        params = {
            "swaig_fields": {"meta_data": {"key": "val"}},
            "api_key": "test-api-key",
            "search_engine_id": "test-engine-id",
        }
        mock_agent = Mock()
        mock_agent.define_tool = Mock()
        skill = WebSearchSkill(agent=mock_agent, params=params)
        skill.setup()
        skill.register_tools()
        _, kw = mock_agent.define_tool.call_args
        assert kw.get("meta_data") == {"key": "val"}

    def test_register_tools_description_present(self):
        skill = _make_skill()
        skill.setup()
        skill.register_tools()
        _, kw = skill.agent.define_tool.call_args
        assert "description" in kw
        assert len(kw["description"]) > 0


# ---------------------------------------------------------------------------
# _web_search_handler()
# ---------------------------------------------------------------------------

class TestWebSearchHandler:
    """Tests for the _web_search_handler method."""

    def _setup_skill(self, params=None):
        """Helper that returns a skill ready for handler testing."""
        skill = _make_skill(params)
        skill.setup()
        return skill

    def test_empty_query_returns_error(self):
        skill = self._setup_skill()
        result = skill._web_search_handler({"query": ""}, {})
        assert isinstance(result, SwaigFunctionResult)
        assert "provide a search query" in result.response.lower()

    def test_whitespace_query_returns_error(self):
        skill = self._setup_skill()
        result = skill._web_search_handler({"query": "   "}, {})
        assert isinstance(result, SwaigFunctionResult)
        assert "provide a search query" in result.response.lower()

    def test_missing_query_key_returns_error(self):
        skill = self._setup_skill()
        result = skill._web_search_handler({}, {})
        assert isinstance(result, SwaigFunctionResult)
        assert "provide a search query" in result.response.lower()

    def test_successful_search(self):
        skill = self._setup_skill()
        mock_results = "Found 2 results meeting quality threshold from 8 searched.\nShowing top 2:\n\n=== RESULT 1 ===\nTitle: Test\nContent: Good content"
        with patch.object(skill.search_scraper, 'search_and_scrape_best', return_value=mock_results):
            result = skill._web_search_handler({"query": "test query"}, {})
            assert isinstance(result, SwaigFunctionResult)
            assert "test query" in result.response
            assert "RESULT 1" in result.response

    def test_no_search_results(self):
        skill = self._setup_skill()
        with patch.object(skill.search_scraper, 'search_and_scrape_best',
                          return_value="No search results found for query: test"):
            result = skill._web_search_handler({"query": "test"}, {})
            assert isinstance(result, SwaigFunctionResult)
            # Should trigger no_results_message
            assert "couldn't find" in result.response.lower() or "quality" in result.response.lower()

    def test_no_quality_results(self):
        skill = self._setup_skill()
        with patch.object(skill.search_scraper, 'search_and_scrape_best',
                          return_value="No quality results found for query: test. All results were below quality threshold."):
            result = skill._web_search_handler({"query": "test"}, {})
            assert isinstance(result, SwaigFunctionResult)

    def test_empty_search_results(self):
        skill = self._setup_skill()
        with patch.object(skill.search_scraper, 'search_and_scrape_best', return_value=""):
            result = skill._web_search_handler({"query": "test"}, {})
            assert isinstance(result, SwaigFunctionResult)

    def test_exception_during_search(self):
        skill = self._setup_skill()
        with patch.object(skill.search_scraper, 'search_and_scrape_best',
                          side_effect=RuntimeError("connection failed")):
            result = skill._web_search_handler({"query": "test"}, {})
            assert isinstance(result, SwaigFunctionResult)
            assert "error" in result.response.lower()

    def test_no_results_custom_message_with_placeholder(self):
        skill = self._setup_skill(
            params={"no_results_message": "Sorry, '{query}' not found."}
        )
        with patch.object(skill.search_scraper, 'search_and_scrape_best',
                          return_value="No search results found for query: widgets"):
            result = skill._web_search_handler({"query": "widgets"}, {})
            assert result.response == "Sorry, 'widgets' not found."

    def test_no_results_custom_message_without_placeholder(self):
        skill = self._setup_skill(
            params={"no_results_message": "No data available."}
        )
        with patch.object(skill.search_scraper, 'search_and_scrape_best',
                          return_value="No search results found for query: anything"):
            result = skill._web_search_handler({"query": "anything"}, {})
            assert result.response == "No data available."

    def test_handler_passes_correct_params_to_scraper(self):
        skill = self._setup_skill(params={
            "num_results": 5,
            "oversample_factor": 3.0,
            "delay": 1.0,
            "min_quality_score": 0.5,
        })
        with patch.object(skill.search_scraper, 'search_and_scrape_best',
                          return_value="some results") as mock_search:
            skill._web_search_handler({"query": "test"}, {})
            mock_search.assert_called_once_with(
                query="test",
                num_results=5,
                oversample_factor=3.0,
                delay=1.0,
                min_quality_score=0.5,
            )

    def test_handler_strips_query_whitespace(self):
        skill = self._setup_skill()
        with patch.object(skill.search_scraper, 'search_and_scrape_best',
                          return_value="some results") as mock_search:
            skill._web_search_handler({"query": "  padded query  "}, {})
            mock_search.assert_called_once()
            assert mock_search.call_args[1]["query"] == "padded query" or mock_search.call_args.kwargs["query"] == "padded query"

    def test_handler_logs_search_request(self):
        skill = self._setup_skill()
        with patch.object(skill.search_scraper, 'search_and_scrape_best', return_value="results"):
            with patch.object(skill.logger, "info") as mock_info:
                skill._web_search_handler({"query": "my search"}, {})
                mock_info.assert_called_once()
                assert "my search" in mock_info.call_args[0][0]

    def test_handler_logs_error_on_exception(self):
        skill = self._setup_skill()
        with patch.object(skill.search_scraper, 'search_and_scrape_best',
                          side_effect=ValueError("bad")):
            with patch.object(skill.logger, "error") as mock_error:
                skill._web_search_handler({"query": "test"}, {})
                mock_error.assert_called_once()


# ---------------------------------------------------------------------------
# GoogleSearchScraper
# ---------------------------------------------------------------------------

class TestGoogleSearchScraper:
    """Tests for the GoogleSearchScraper class."""

    def test_init_stores_params(self):
        scraper = GoogleSearchScraper("key", "engine_id", max_content_length=5000)
        assert scraper.api_key == "key"
        assert scraper.search_engine_id == "engine_id"
        assert scraper.max_content_length == 5000

    def test_init_default_max_content_length(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        assert scraper.max_content_length == 32768

    def test_init_session_created(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        assert scraper.session is not None

    def test_is_reddit_url_true(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        assert scraper.is_reddit_url("https://www.reddit.com/r/test") is True
        assert scraper.is_reddit_url("https://old.reddit.com/r/test") is True
        assert scraper.is_reddit_url("https://redd.it/abc123") is True

    def test_is_reddit_url_false(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        assert scraper.is_reddit_url("https://www.google.com") is False
        assert scraper.is_reddit_url("https://stackoverflow.com") is False
        assert scraper.is_reddit_url("https://example.com/reddit.com") is False


class TestSearchGoogle:
    """Tests for the search_google method."""

    def test_successful_search(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        mock_response = Mock()
        mock_response.json.return_value = {
            "items": [
                {"title": "Result 1", "link": "https://example.com/1", "snippet": "Snippet 1"},
                {"title": "Result 2", "link": "https://example.com/2", "snippet": "Snippet 2"},
            ]
        }
        mock_response.raise_for_status = Mock()
        with patch.object(scraper.session, 'get', return_value=mock_response):
            results = scraper.search_google("test query", num_results=5)
            assert len(results) == 2
            assert results[0]["title"] == "Result 1"
            assert results[0]["url"] == "https://example.com/1"
            assert results[1]["snippet"] == "Snippet 2"

    def test_search_no_items_key(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        mock_response = Mock()
        mock_response.json.return_value = {"searchInformation": {"totalResults": "0"}}
        mock_response.raise_for_status = Mock()
        with patch.object(scraper.session, 'get', return_value=mock_response):
            results = scraper.search_google("test query")
            assert results == []

    def test_search_api_error(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        with patch.object(scraper.session, 'get', side_effect=requests.exceptions.HTTPError("403")):
            results = scraper.search_google("test query")
            assert results == []

    def test_search_network_error(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        with patch.object(scraper.session, 'get', side_effect=requests.exceptions.ConnectionError("failed")):
            results = scraper.search_google("test query")
            assert results == []

    def test_search_limits_num_results_to_10(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        mock_response = Mock()
        mock_response.json.return_value = {"items": []}
        mock_response.raise_for_status = Mock()
        with patch.object(scraper.session, 'get', return_value=mock_response) as mock_get:
            scraper.search_google("test", num_results=20)
            call_kwargs = mock_get.call_args
            assert call_kwargs[1]["params"]["num"] == 10

    def test_search_missing_fields_defaults_empty(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        mock_response = Mock()
        mock_response.json.return_value = {
            "items": [{"title": "Only Title"}]  # missing link and snippet
        }
        mock_response.raise_for_status = Mock()
        with patch.object(scraper.session, 'get', return_value=mock_response):
            results = scraper.search_google("test")
            assert results[0]["title"] == "Only Title"
            assert results[0]["url"] == ""
            assert results[0]["snippet"] == ""


class TestExtractTextFromUrl:
    """Tests for extract_text_from_url routing."""

    def test_routes_reddit_to_extract_reddit_content(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        with patch.object(scraper, 'extract_reddit_content', return_value=("reddit content", {})) as mock_reddit:
            text, _ = scraper.extract_text_from_url("https://www.reddit.com/r/test/comments/123")
            mock_reddit.assert_called_once()
            assert text == "reddit content"

    def test_routes_non_reddit_to_extract_html_content(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        with patch.object(scraper, 'extract_html_content', return_value=("html content", {})) as mock_html:
            text, _ = scraper.extract_text_from_url("https://example.com/article")
            mock_html.assert_called_once()
            assert text == "html content"


class TestExtractHtmlContent:
    """Tests for extract_html_content."""

    def test_successful_extraction(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        mock_response = Mock()
        mock_response.content = b"<html><body><article>This is quality content with many sentences. " \
                                b"It has a lot of text. Multiple lines of content here.</article></body></html>"
        mock_response.raise_for_status = Mock()
        with patch.object(scraper.session, 'get', return_value=mock_response):
            text, metrics = scraper.extract_html_content("https://example.com/article")
            assert "quality content" in text
            assert "quality_score" in metrics

    def test_extraction_error_returns_empty(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        with patch.object(scraper.session, 'get', side_effect=requests.exceptions.ConnectionError("failed")):
            text, metrics = scraper.extract_html_content("https://example.com")
            assert text == ""
            assert metrics["quality_score"] == 0

    def test_content_truncation(self):
        scraper = GoogleSearchScraper("key", "engine_id", max_content_length=50)
        mock_response = Mock()
        long_text = "A" * 200
        mock_response.content = f"<html><body><article>{long_text}</article></body></html>".encode()
        mock_response.raise_for_status = Mock()
        with patch.object(scraper.session, 'get', return_value=mock_response):
            text, metrics = scraper.extract_html_content("https://example.com")
            assert len(text) <= 50

    def test_custom_content_limit(self):
        scraper = GoogleSearchScraper("key", "engine_id", max_content_length=100000)
        mock_response = Mock()
        long_text = "B" * 200
        mock_response.content = f"<html><body>{long_text}</body></html>".encode()
        mock_response.raise_for_status = Mock()
        with patch.object(scraper.session, 'get', return_value=mock_response):
            text, metrics = scraper.extract_html_content("https://example.com", content_limit=30)
            assert len(text) <= 30


class TestExtractRedditContent:
    """Tests for extract_reddit_content."""

    def _make_reddit_json(self, title="Test Post", author="testuser", score=100,
                          num_comments=50, selftext="Post body text", subreddit="test",
                          comments=None):
        """Helper to build Reddit JSON structure."""
        post_data = {
            "data": {
                "children": [{
                    "data": {
                        "title": title,
                        "author": author,
                        "score": score,
                        "num_comments": num_comments,
                        "selftext": selftext,
                        "subreddit": subreddit,
                    }
                }]
            }
        }
        if comments is None:
            comments_data = {"data": {"children": []}}
        else:
            comments_data = {"data": {"children": comments}}
        return [post_data, comments_data]

    @patch("signalwire_agents.skills.web_search.skill.requests.get")
    def test_successful_reddit_extraction(self, mock_get):
        scraper = GoogleSearchScraper("key", "engine_id")
        mock_response = Mock()
        mock_response.json.return_value = self._make_reddit_json()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        text, metrics = scraper.extract_reddit_content("https://reddit.com/r/test/comments/123")
        assert "Test Post" in text
        assert "testuser" in text
        assert metrics["is_reddit"] is True

    @patch("signalwire_agents.skills.web_search.skill.requests.get")
    def test_reddit_with_comments(self, mock_get):
        scraper = GoogleSearchScraper("key", "engine_id")
        comments = [
            {
                "kind": "t1",
                "data": {
                    "body": "This is a very helpful and detailed comment that exceeds the minimum length threshold.",
                    "author": "commenter1",
                    "score": 50,
                }
            }
        ]
        mock_response = Mock()
        mock_response.json.return_value = self._make_reddit_json(comments=comments)
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        text, metrics = scraper.extract_reddit_content("https://reddit.com/r/test/comments/123")
        assert "commenter1" in text
        assert "helpful" in text

    @patch("signalwire_agents.skills.web_search.skill.requests.get")
    def test_reddit_filters_short_comments(self, mock_get):
        scraper = GoogleSearchScraper("key", "engine_id")
        comments = [
            {
                "kind": "t1",
                "data": {"body": "Short", "author": "short_commenter", "score": 5}
            }
        ]
        mock_response = Mock()
        mock_response.json.return_value = self._make_reddit_json(comments=comments)
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        text, _ = scraper.extract_reddit_content("https://reddit.com/r/test/comments/123")
        # Short comments (< 50 chars) should be filtered
        assert "short_commenter" not in text

    @patch("signalwire_agents.skills.web_search.skill.requests.get")
    def test_reddit_filters_deleted_comments(self, mock_get):
        scraper = GoogleSearchScraper("key", "engine_id")
        comments = [
            {
                "kind": "t1",
                "data": {
                    "body": "[deleted]",
                    "author": "deleted_user",
                    "score": 100,
                }
            }
        ]
        mock_response = Mock()
        mock_response.json.return_value = self._make_reddit_json(comments=comments)
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        text, _ = scraper.extract_reddit_content("https://reddit.com/r/test/comments/123")
        assert "deleted_user" not in text

    @patch("signalwire_agents.skills.web_search.skill.requests.get")
    def test_reddit_filters_removed_selftext(self, mock_get):
        scraper = GoogleSearchScraper("key", "engine_id")
        mock_response = Mock()
        mock_response.json.return_value = self._make_reddit_json(selftext="[removed]")
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        text, _ = scraper.extract_reddit_content("https://reddit.com/r/test/comments/123")
        assert "[removed]" not in text

    @patch("signalwire_agents.skills.web_search.skill.requests.get")
    def test_reddit_appends_json_suffix(self, mock_get):
        scraper = GoogleSearchScraper("key", "engine_id")
        mock_response = Mock()
        mock_response.json.return_value = self._make_reddit_json()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        scraper.extract_reddit_content("https://reddit.com/r/test/comments/123")
        called_url = mock_get.call_args[0][0]
        assert called_url.endswith(".json")

    @patch("signalwire_agents.skills.web_search.skill.requests.get")
    def test_reddit_already_json_url(self, mock_get):
        scraper = GoogleSearchScraper("key", "engine_id")
        mock_response = Mock()
        mock_response.json.return_value = self._make_reddit_json()
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        scraper.extract_reddit_content("https://reddit.com/r/test/comments/123.json")
        called_url = mock_get.call_args[0][0]
        assert called_url == "https://reddit.com/r/test/comments/123.json"

    @patch("signalwire_agents.skills.web_search.skill.requests.get")
    def test_reddit_invalid_json_falls_back_to_html(self, mock_get):
        scraper = GoogleSearchScraper("key", "engine_id")
        mock_get.side_effect = ValueError("Invalid JSON")

        with patch.object(scraper, 'extract_html_content', return_value=("fallback", {})) as mock_html:
            text, _ = scraper.extract_reddit_content("https://reddit.com/r/test")
            mock_html.assert_called_once()
            assert text == "fallback"

    @patch("signalwire_agents.skills.web_search.skill.requests.get")
    def test_reddit_content_limit(self, mock_get):
        scraper = GoogleSearchScraper("key", "engine_id", max_content_length=100000)
        mock_response = Mock()
        mock_response.json.return_value = self._make_reddit_json(selftext="X" * 2000)
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        text, _ = scraper.extract_reddit_content("https://reddit.com/r/test", content_limit=50)
        assert len(text) <= 50

    @patch("signalwire_agents.skills.web_search.skill.requests.get")
    def test_reddit_quality_metrics(self, mock_get):
        scraper = GoogleSearchScraper("key", "engine_id")
        mock_response = Mock()
        mock_response.json.return_value = self._make_reddit_json(score=200, num_comments=100)
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        _, metrics = scraper.extract_reddit_content("https://reddit.com/r/test/comments/123")
        assert "quality_score" in metrics
        assert metrics["is_reddit"] is True
        assert metrics["score"] == 200
        assert metrics["num_comments"] == 100


class TestCalculateContentQuality:
    """Tests for _calculate_content_quality."""

    def test_empty_text_zero_quality(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        metrics = scraper._calculate_content_quality("", "https://example.com")
        assert metrics["quality_score"] == 0

    def test_short_text_low_quality(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        metrics = scraper._calculate_content_quality("Short text", "https://example.com")
        assert metrics["quality_score"] < 0.5

    def test_quality_domain_bonus(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        text = "A " * 2000  # Enough text
        metrics_quality = scraper._calculate_content_quality(text, "https://wikipedia.org/wiki/test")
        metrics_generic = scraper._calculate_content_quality(text, "https://randomsite.com/page")
        assert metrics_quality["domain_score"] > metrics_generic["domain_score"]

    def test_low_quality_domain_penalty(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        text = "A " * 2000
        metrics = scraper._calculate_content_quality(text, "https://reddit.com/r/test")
        assert metrics["domain_score"] < 1.0

    def test_boilerplate_penalty(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        text_clean = "This is good content about programming. " * 100
        text_boilerplate = "cookie privacy policy terms of service subscribe sign up " * 50
        metrics_clean = scraper._calculate_content_quality(text_clean, "https://example.com")
        metrics_boilerplate = scraper._calculate_content_quality(text_boilerplate, "https://example.com")
        assert metrics_clean["boilerplate_penalty"] > metrics_boilerplate["boilerplate_penalty"]

    def test_query_relevance_scoring(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        text = "Python programming language is great for data science and machine learning tasks."
        metrics = scraper._calculate_content_quality(text, "https://example.com", query="Python programming")
        assert metrics["query_relevance"] > 0

    def test_no_query_neutral_relevance(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        text = "Some content here."
        metrics = scraper._calculate_content_quality(text, "https://example.com", query="")
        assert metrics["query_relevance"] == 0.5


class TestSearchAndScrapeBest:
    """Tests for search_and_scrape_best."""

    def test_no_search_results(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        with patch.object(scraper, 'search_google', return_value=[]):
            result = scraper.search_and_scrape_best("test query")
            assert "No search results found" in result

    def test_all_results_below_threshold(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        search_results = [
            {"title": "Bad", "url": "https://bad.com", "snippet": "bad"}
        ]
        with patch.object(scraper, 'search_google', return_value=search_results):
            with patch.object(scraper, 'extract_text_from_url', return_value=("", {"quality_score": 0})):
                result = scraper.search_and_scrape_best("test query", delay=0)
                assert "No quality results found" in result

    def test_successful_search_and_scrape(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        search_results = [
            {"title": "Good Result", "url": "https://example.com/good", "snippet": "A good result"}
        ]
        good_metrics = {
            "quality_score": 0.8,
            "domain": "example.com",
            "text_length": 5000,
            "sentence_count": 20,
            "query_relevance": 0.9,
            "query_words_found": "2/2",
        }
        with patch.object(scraper, 'search_google', return_value=search_results):
            with patch.object(scraper, 'extract_text_from_url',
                              return_value=("Great content here", good_metrics)):
                with patch.object(scraper, '_calculate_content_quality', return_value=good_metrics):
                    result = scraper.search_and_scrape_best("test query", delay=0)
                    assert "RESULT 1" in result
                    assert "Good Result" in result

    def test_domain_diversity(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        search_results = [
            {"title": "Result A1", "url": "https://a.com/1", "snippet": "A1"},
            {"title": "Result A2", "url": "https://a.com/2", "snippet": "A2"},
            {"title": "Result B1", "url": "https://b.com/1", "snippet": "B1"},
        ]
        metrics_a = {"quality_score": 0.9, "domain": "a.com", "text_length": 5000,
                      "sentence_count": 10, "query_relevance": 0.8, "query_words_found": "1/1"}
        metrics_b = {"quality_score": 0.7, "domain": "b.com", "text_length": 5000,
                      "sentence_count": 10, "query_relevance": 0.8, "query_words_found": "1/1"}

        def mock_extract(url, **kwargs):
            if "a.com" in url:
                return ("Content A", metrics_a)
            return ("Content B", metrics_b)

        def mock_quality(text, url, query=""):
            if "a.com" in url:
                return metrics_a
            return metrics_b

        with patch.object(scraper, 'search_google', return_value=search_results):
            with patch.object(scraper, 'extract_text_from_url', side_effect=mock_extract):
                with patch.object(scraper, '_calculate_content_quality', side_effect=mock_quality):
                    result = scraper.search_and_scrape_best("test", num_results=2, delay=0)
                    # Should show results from both domains
                    assert "a.com" in result
                    assert "b.com" in result

    def test_backward_compatible_search_and_scrape(self):
        scraper = GoogleSearchScraper("key", "engine_id")
        with patch.object(scraper, 'search_and_scrape_best', return_value="results") as mock_best:
            result = scraper.search_and_scrape("test query", num_results=2, delay=0.1)
            mock_best.assert_called_once_with(
                query="test query",
                num_results=2,
                oversample_factor=4.0,
                delay=0.1,
                min_quality_score=0.2,
            )


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

    def test_returns_correct_keys(self):
        skill = _make_skill()
        skill.setup()
        data = skill.get_global_data()
        assert data["web_search_enabled"] is True
        assert data["search_provider"] == "Google Custom Search"
        assert data["quality_filtering"] is True


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

    def test_section_title(self):
        skill = _make_skill()
        skill.setup()
        section = skill.get_prompt_sections()[0]
        assert section["title"] == "Web Search Capability (Quality Enhanced)"

    def test_section_references_tool_name(self):
        skill = _make_skill()
        skill.setup()
        section = skill.get_prompt_sections()[0]
        assert "web_search" in section["body"]

    def test_section_references_custom_tool_name(self):
        skill = _make_skill({"tool_name": "news_search"})
        skill.setup()
        section = skill.get_prompt_sections()[0]
        assert "news_search" in section["body"]
        assert any("news_search" in bullet for bullet in section["bullets"])

    def test_section_has_bullets(self):
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

    def test_setup_then_register_then_handler_flow(self):
        """Full lifecycle: setup -> register -> handle search."""
        skill = _make_skill()
        assert skill.setup() is True
        skill.register_tools()

        # Extract the handler that was registered
        _, kw = skill.agent.define_tool.call_args
        handler = kw["handler"]

        with patch.object(skill.search_scraper, 'search_and_scrape_best',
                          return_value="=== RESULT 1 ===\nTitle: Lifecycle\nContent: Answer"):
            result = handler({"query": "lifecycle test"}, {})
            assert isinstance(result, SwaigFunctionResult)
            assert "Lifecycle" in result.response

    def test_setup_returns_false_for_each_missing_param(self):
        """Verify setup returns False when each required param is empty."""
        required = ["api_key", "search_engine_id"]
        for missing in required:
            params = {k: f"val_{k}" for k in required}
            params[missing] = ""
            mock_agent = Mock()
            mock_agent.define_tool = Mock()
            skill = WebSearchSkill(agent=mock_agent, params=params)
            assert skill.setup() is False, f"Should fail when {missing} is empty"

    def test_multiple_instances_different_tool_names(self):
        """Two instances with different tool_names should have different keys."""
        skill_a = _make_skill({"tool_name": "search_news"})
        skill_b = _make_skill({"tool_name": "search_docs"})
        assert skill_a.get_instance_key() != skill_b.get_instance_key()
        assert "search_news" in skill_a.get_instance_key()
        assert "search_docs" in skill_b.get_instance_key()

    def test_handler_with_none_return_from_scraper(self):
        """If scraper returns None, handler should handle gracefully."""
        skill = _make_skill()
        skill.setup()
        with patch.object(skill.search_scraper, 'search_and_scrape_best', return_value=None):
            result = skill._web_search_handler({"query": "test"}, {})
            assert isinstance(result, SwaigFunctionResult)

    def test_content_quality_with_ideal_length_text(self):
        """Text in the 2000-10000 char range should get max length score."""
        scraper = GoogleSearchScraper("key", "engine_id")
        text = "This is a good sentence. " * 200  # ~5000 chars
        metrics = scraper._calculate_content_quality(text, "https://example.com")
        assert metrics["length_score"] == 1.0

    def test_content_quality_very_long_text(self):
        """Very long text should still get decent but reduced length score."""
        scraper = GoogleSearchScraper("key", "engine_id")
        text = "Word " * 20000  # ~100000 chars
        metrics = scraper._calculate_content_quality(text, "https://example.com")
        assert metrics["length_score"] < 1.0
        assert metrics["length_score"] >= 0.8
