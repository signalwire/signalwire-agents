"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for the Spider skill module (web scraping and crawling).
"""

import pytest
import re
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from typing import Dict, Any

from signalwire_agents.core.function_result import SwaigFunctionResult


# ---------------------------------------------------------------------------
# Helpers for building a mock agent and mock HTTP responses
# ---------------------------------------------------------------------------

def _make_mock_agent():
    """Create a mock agent with a define_tool method."""
    agent = Mock()
    agent.define_tool = Mock()
    return agent


def _make_mock_response(content=b"<html><body><p>Hello world</p></body></html>",
                        url="https://example.com",
                        status_code=200,
                        text=None):
    """Create a mock requests.Response."""
    resp = Mock()
    resp.content = content
    resp.url = url
    resp.status_code = status_code
    resp.text = text or content.decode("utf-8", errors="replace")
    resp.raise_for_status = Mock()
    return resp


# ---------------------------------------------------------------------------
# Fixture: create SpiderSkill instances with mocked dependencies
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_agent():
    return _make_mock_agent()


@pytest.fixture
def default_skill(mock_agent):
    """SpiderSkill with default parameters."""
    with patch("signalwire_agents.skills.spider.skill.requests.Session") as MockSession:
        mock_session = Mock()
        mock_session.headers = {}
        MockSession.return_value = mock_session

        from signalwire_agents.skills.spider.skill import SpiderSkill
        skill = SpiderSkill(mock_agent, {})
    return skill


@pytest.fixture
def custom_skill(mock_agent):
    """SpiderSkill with explicit custom parameters."""
    params = {
        "delay": 0.5,
        "concurrent_requests": 10,
        "timeout": 15,
        "max_pages": 5,
        "max_depth": 2,
        "extract_type": "markdown",
        "max_text_length": 5000,
        "clean_text": False,
        "cache_enabled": False,
        "follow_robots_txt": True,
        "user_agent": "TestBot/1.0",
        "headers": {"X-Custom": "value"},
        "tool_name": "my_spider",
        "selectors": {"title": "//title/text()"},
        "follow_patterns": [r"/blog/.*"],
    }
    with patch("signalwire_agents.skills.spider.skill.requests.Session") as MockSession:
        mock_session = Mock()
        mock_session.headers = {}
        MockSession.return_value = mock_session

        from signalwire_agents.skills.spider.skill import SpiderSkill
        skill = SpiderSkill(mock_agent, params)
    return skill


# ===================================================================
# Class attributes
# ===================================================================

class TestSpiderSkillClassAttributes:
    """Verify class-level constants."""

    def test_skill_name(self):
        from signalwire_agents.skills.spider.skill import SpiderSkill
        assert SpiderSkill.SKILL_NAME == "spider"

    def test_skill_description(self):
        from signalwire_agents.skills.spider.skill import SpiderSkill
        assert SpiderSkill.SKILL_DESCRIPTION == "Fast web scraping and crawling capabilities"

    def test_skill_version(self):
        from signalwire_agents.skills.spider.skill import SpiderSkill
        assert SpiderSkill.SKILL_VERSION == "1.0.0"

    def test_required_packages(self):
        from signalwire_agents.skills.spider.skill import SpiderSkill
        assert "lxml" in SpiderSkill.REQUIRED_PACKAGES

    def test_required_env_vars_empty(self):
        from signalwire_agents.skills.spider.skill import SpiderSkill
        assert SpiderSkill.REQUIRED_ENV_VARS == []

    def test_supports_multiple_instances(self):
        from signalwire_agents.skills.spider.skill import SpiderSkill
        assert SpiderSkill.SUPPORTS_MULTIPLE_INSTANCES is True

    def test_whitespace_regex_compiled(self):
        from signalwire_agents.skills.spider.skill import SpiderSkill
        assert isinstance(SpiderSkill.WHITESPACE_REGEX, re.Pattern)
        assert SpiderSkill.WHITESPACE_REGEX.sub(" ", "  a   b  ") == " a b "


# ===================================================================
# get_parameter_schema
# ===================================================================

class TestGetParameterSchema:
    """Verify the parameter schema returned by the class method."""

    def test_returns_dict(self):
        from signalwire_agents.skills.spider.skill import SpiderSkill
        schema = SpiderSkill.get_parameter_schema()
        assert isinstance(schema, dict)

    def test_contains_expected_keys(self):
        from signalwire_agents.skills.spider.skill import SpiderSkill
        schema = SpiderSkill.get_parameter_schema()
        expected_keys = [
            "delay", "concurrent_requests", "timeout", "max_pages",
            "max_depth", "extract_type", "max_text_length", "clean_text",
            "selectors", "follow_patterns", "user_agent", "headers",
            "follow_robots_txt", "cache_enabled",
        ]
        for key in expected_keys:
            assert key in schema, f"Missing key: {key}"

    def test_includes_base_schema_keys(self):
        from signalwire_agents.skills.spider.skill import SpiderSkill
        schema = SpiderSkill.get_parameter_schema()
        # SkillBase adds swaig_fields and tool_name for multi-instance
        assert "swaig_fields" in schema
        assert "tool_name" in schema

    def test_delay_has_correct_defaults(self):
        from signalwire_agents.skills.spider.skill import SpiderSkill
        schema = SpiderSkill.get_parameter_schema()
        assert schema["delay"]["default"] == 0.1
        assert schema["delay"]["type"] == "number"

    def test_extract_type_enum(self):
        from signalwire_agents.skills.spider.skill import SpiderSkill
        schema = SpiderSkill.get_parameter_schema()
        assert set(schema["extract_type"]["enum"]) == {
            "fast_text", "clean_text", "full_text", "html", "custom"
        }


# ===================================================================
# __init__
# ===================================================================

class TestSpiderSkillInit:
    """Verify that __init__ correctly stores parameters and sets up state."""

    def test_default_parameter_values(self, default_skill):
        assert default_skill.delay == 0.1
        assert default_skill.concurrent_requests == 5
        assert default_skill.timeout == 5
        assert default_skill.max_pages == 1
        assert default_skill.max_depth == 0
        assert default_skill.extract_type == "fast_text"
        assert default_skill.max_text_length == 3000
        assert default_skill.clean_text is True
        assert default_skill.cache_enabled is True
        assert default_skill.follow_robots_txt is False
        assert default_skill.user_agent == "Spider/1.0 (SignalWire AI Agent)"

    def test_custom_parameter_values(self, custom_skill):
        assert custom_skill.delay == 0.5
        assert custom_skill.concurrent_requests == 10
        assert custom_skill.timeout == 15
        assert custom_skill.max_pages == 5
        assert custom_skill.max_depth == 2
        assert custom_skill.extract_type == "markdown"
        assert custom_skill.max_text_length == 5000
        assert custom_skill.clean_text is False
        assert custom_skill.cache_enabled is False
        assert custom_skill.follow_robots_txt is True
        assert custom_skill.user_agent == "TestBot/1.0"

    def test_user_agent_merged_into_headers(self, default_skill):
        assert "User-Agent" in default_skill.headers
        assert default_skill.headers["User-Agent"] == default_skill.user_agent

    def test_custom_headers_preserved(self, custom_skill):
        assert custom_skill.headers.get("X-Custom") == "value"
        assert custom_skill.headers["User-Agent"] == "TestBot/1.0"

    def test_cache_is_dict_when_enabled(self, default_skill):
        assert default_skill.cache == {}

    def test_cache_is_none_when_disabled(self, custom_skill):
        assert custom_skill.cache is None

    def test_session_created(self, default_skill):
        assert default_skill.session is not None

    def test_remove_xpaths_populated(self, default_skill):
        assert len(default_skill.remove_xpaths) > 0
        assert "//script" in default_skill.remove_xpaths
        assert "//style" in default_skill.remove_xpaths

    def test_agent_stored(self, default_skill, mock_agent):
        assert default_skill.agent is mock_agent


# ===================================================================
# get_instance_key
# ===================================================================

class TestGetInstanceKey:

    def test_default_instance_key(self, default_skill):
        # No tool_name in params; falls back to SKILL_NAME
        key = default_skill.get_instance_key()
        assert key == "spider_spider"

    def test_custom_instance_key(self, custom_skill):
        key = custom_skill.get_instance_key()
        assert key == "spider_my_spider"


# ===================================================================
# setup
# ===================================================================

class TestSetup:

    def test_valid_configuration_returns_true(self, default_skill):
        assert default_skill.setup() is True

    def test_negative_delay_returns_false(self, default_skill):
        default_skill.delay = -1
        assert default_skill.setup() is False

    def test_concurrent_requests_too_low_returns_false(self, default_skill):
        default_skill.concurrent_requests = 0
        assert default_skill.setup() is False

    def test_concurrent_requests_too_high_returns_false(self, default_skill):
        default_skill.concurrent_requests = 21
        assert default_skill.setup() is False

    def test_max_pages_too_low_returns_false(self, default_skill):
        default_skill.max_pages = 0
        assert default_skill.setup() is False

    def test_negative_max_depth_returns_false(self, default_skill):
        default_skill.max_depth = -1
        assert default_skill.setup() is False

    def test_boundary_concurrent_requests_low(self, default_skill):
        default_skill.concurrent_requests = 1
        assert default_skill.setup() is True

    def test_boundary_concurrent_requests_high(self, default_skill):
        default_skill.concurrent_requests = 20
        assert default_skill.setup() is True

    def test_zero_delay_is_valid(self, default_skill):
        default_skill.delay = 0
        assert default_skill.setup() is True

    def test_zero_max_depth_is_valid(self, default_skill):
        default_skill.max_depth = 0
        assert default_skill.setup() is True

    def test_setup_logs_info_on_success(self, default_skill):
        with patch.object(default_skill.logger, "info") as mock_info:
            default_skill.setup()
            mock_info.assert_called_once()
            assert "Spider skill configured" in mock_info.call_args[0][0]


# ===================================================================
# register_tools
# ===================================================================

class TestRegisterTools:

    def test_registers_three_tools(self, default_skill):
        default_skill.register_tools()
        assert default_skill.agent.define_tool.call_count == 3

    def test_tool_names_without_prefix(self, default_skill):
        default_skill.register_tools()
        names = [call.kwargs.get("name") or call[1].get("name")
                 for call in default_skill.agent.define_tool.call_args_list]
        assert "scrape_url" in names
        assert "crawl_site" in names
        assert "extract_structured_data" in names

    def test_tool_names_with_prefix(self, custom_skill):
        custom_skill.register_tools()
        names = [call.kwargs.get("name") or call[1].get("name")
                 for call in custom_skill.agent.define_tool.call_args_list]
        assert "my_spider_scrape_url" in names
        assert "my_spider_crawl_site" in names
        assert "my_spider_extract_structured_data" in names

    def test_handlers_are_callable(self, default_skill):
        default_skill.register_tools()
        for call in default_skill.agent.define_tool.call_args_list:
            handler = call.kwargs.get("handler") or call[1].get("handler")
            assert callable(handler)


# ===================================================================
# _fetch_url
# ===================================================================

class TestFetchUrl:

    def test_returns_cached_response(self, default_skill):
        cached = _make_mock_response()
        default_skill.cache["https://cached.com"] = cached
        result = default_skill._fetch_url("https://cached.com")
        assert result is cached

    def test_successful_fetch_stores_in_cache(self, default_skill):
        resp = _make_mock_response()
        default_skill.session.get = Mock(return_value=resp)
        result = default_skill._fetch_url("https://example.com")
        assert result is resp
        assert "https://example.com" in default_skill.cache

    def test_successful_fetch_no_cache_when_disabled(self, custom_skill):
        resp = _make_mock_response()
        custom_skill.session.get = Mock(return_value=resp)
        result = custom_skill._fetch_url("https://example.com")
        assert result is resp
        assert custom_skill.cache is None

    def test_timeout_returns_none(self, default_skill):
        import requests as req_mod
        default_skill.session.get = Mock(side_effect=req_mod.exceptions.Timeout("timeout"))
        result = default_skill._fetch_url("https://slow.com")
        assert result is None

    def test_request_exception_returns_none(self, default_skill):
        import requests as req_mod
        default_skill.session.get = Mock(
            side_effect=req_mod.exceptions.ConnectionError("refused"))
        result = default_skill._fetch_url("https://down.com")
        assert result is None

    def test_http_error_returns_none(self, default_skill):
        import requests as req_mod
        resp = _make_mock_response()
        resp.raise_for_status.side_effect = req_mod.exceptions.HTTPError("404")
        default_skill.session.get = Mock(return_value=resp)
        result = default_skill._fetch_url("https://missing.com")
        assert result is None

    def test_timeout_kwarg_passed_to_session(self, default_skill):
        resp = _make_mock_response()
        default_skill.session.get = Mock(return_value=resp)
        default_skill._fetch_url("https://example.com")
        default_skill.session.get.assert_called_once_with(
            "https://example.com", timeout=default_skill.timeout)


# ===================================================================
# _fast_text_extract
# ===================================================================

class TestFastTextExtract:

    def test_extracts_text_from_html(self, default_skill):
        resp = _make_mock_response(
            content=b"<html><body><p>Hello world</p></body></html>")
        text = default_skill._fast_text_extract(resp)
        assert "Hello world" in text

    def test_removes_script_elements(self, default_skill):
        resp = _make_mock_response(
            content=b"<html><body><script>var x=1;</script><p>Visible</p></body></html>")
        text = default_skill._fast_text_extract(resp)
        assert "var x=1" not in text
        assert "Visible" in text

    def test_removes_style_elements(self, default_skill):
        resp = _make_mock_response(
            content=b"<html><body><style>.foo{color:red}</style><p>Visible</p></body></html>")
        text = default_skill._fast_text_extract(resp)
        assert "color:red" not in text
        assert "Visible" in text

    def test_removes_nav_header_footer_aside(self, default_skill):
        html_content = (
            b"<html><body>"
            b"<nav>NavContent</nav>"
            b"<header>HeaderContent</header>"
            b"<footer>FooterContent</footer>"
            b"<aside>AsideContent</aside>"
            b"<p>MainContent</p>"
            b"</body></html>"
        )
        resp = _make_mock_response(content=html_content)
        text = default_skill._fast_text_extract(resp)
        assert "NavContent" not in text
        assert "HeaderContent" not in text
        assert "FooterContent" not in text
        assert "AsideContent" not in text
        assert "MainContent" in text

    def test_clean_text_collapses_whitespace(self, default_skill):
        resp = _make_mock_response(
            content=b"<html><body><p>Hello    \n\n   world</p></body></html>")
        text = default_skill._fast_text_extract(resp)
        # clean_text is True by default, so multiple whitespace should collapse
        assert "Hello world" in text

    def test_no_clean_text_preserves_whitespace(self, custom_skill):
        # custom_skill has clean_text=False
        resp = _make_mock_response(
            content=b"<html><body><p>Hello    world</p></body></html>")
        text = custom_skill._fast_text_extract(resp)
        # Whitespace may not be fully collapsed
        assert "Hello" in text
        assert "world" in text

    def test_truncation_when_text_exceeds_max_length(self, default_skill):
        # default max_text_length is 3000
        long_text = "A" * 5000
        resp = _make_mock_response(
            content=f"<html><body><p>{long_text}</p></body></html>".encode())
        text = default_skill._fast_text_extract(resp)
        assert "[...CONTENT TRUNCATED...]" in text
        # Text should be around max_text_length plus the truncation marker
        assert len(text) < 5000 + 100

    def test_no_truncation_when_within_limit(self, default_skill):
        short_text = "A" * 100
        resp = _make_mock_response(
            content=f"<html><body><p>{short_text}</p></body></html>".encode())
        text = default_skill._fast_text_extract(resp)
        assert "[...CONTENT TRUNCATED...]" not in text

    def test_returns_empty_string_on_parse_error(self, default_skill):
        with patch("signalwire_agents.skills.spider.skill.html.fromstring",
                    side_effect=Exception("parse error")):
            resp = _make_mock_response(content=b"not valid html at all")
            text = default_skill._fast_text_extract(resp)
            assert text == ""


# ===================================================================
# _markdown_extract
# ===================================================================

class TestMarkdownExtract:

    def test_extracts_title(self, default_skill):
        resp = _make_mock_response(
            content=b"<html><head><title>My Page</title></head><body><p>Content</p></body></html>")
        text = default_skill._markdown_extract(resp)
        assert "# My Page" in text

    def test_extracts_paragraphs(self, default_skill):
        resp = _make_mock_response(
            content=b"<html><body><p>Paragraph one</p><p>Paragraph two</p></body></html>")
        text = default_skill._markdown_extract(resp)
        assert "Paragraph one" in text
        assert "Paragraph two" in text

    def test_extracts_headings_with_correct_level(self, default_skill):
        resp = _make_mock_response(
            content=b"<html><body><h1>Heading 1</h1><h2>Heading 2</h2><h3>Heading 3</h3></body></html>")
        text = default_skill._markdown_extract(resp)
        assert "# Heading 1" in text
        assert "## Heading 2" in text
        assert "### Heading 3" in text

    def test_extracts_list_items(self, default_skill):
        resp = _make_mock_response(
            content=b"<html><body><ul><li>Item A</li><li>Item B</li></ul></body></html>")
        text = default_skill._markdown_extract(resp)
        assert "- Item A" in text
        assert "- Item B" in text

    def test_extracts_code_blocks(self, default_skill):
        resp = _make_mock_response(
            content=b"<html><body><pre>some code</pre></body></html>")
        text = default_skill._markdown_extract(resp)
        assert "```" in text
        assert "some code" in text

    def test_removes_unwanted_elements(self, default_skill):
        html_content = (
            b"<html><body>"
            b"<script>evil()</script>"
            b"<nav>NavStuff</nav>"
            b"<p>Good content</p>"
            b"</body></html>"
        )
        resp = _make_mock_response(content=html_content)
        text = default_skill._markdown_extract(resp)
        assert "evil()" not in text
        assert "NavStuff" not in text
        assert "Good content" in text

    def test_truncation_with_marker(self, default_skill):
        long_text = "X" * 5000
        resp = _make_mock_response(
            content=f"<html><body><p>{long_text}</p></body></html>".encode())
        text = default_skill._markdown_extract(resp)
        assert "[...TRUNCATED...]" in text

    def test_falls_back_to_fast_text_on_import_error(self, default_skill):
        resp = _make_mock_response(
            content=b"<html><body><p>Fallback content</p></body></html>")
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "bs4":
                raise ImportError("No bs4")
            return real_import(name, *args, **kwargs)

        # Remove bs4 from sys.modules cache so the import inside the method triggers
        import sys
        saved_bs4 = sys.modules.pop("bs4", None)
        try:
            with patch("builtins.__import__", side_effect=fake_import):
                text = default_skill._markdown_extract(resp)
                # Should fall back to fast_text extraction
                assert "Fallback content" in text
        finally:
            if saved_bs4 is not None:
                sys.modules["bs4"] = saved_bs4

    def test_falls_back_to_fast_text_on_general_error(self, default_skill):
        resp = _make_mock_response(
            content=b"<html><body><p>Some content</p></body></html>")
        with patch("bs4.BeautifulSoup", side_effect=Exception("soup error")):
            text = default_skill._markdown_extract(resp)
            # Should fall back to fast_text
            assert "Some content" in text


# ===================================================================
# _structured_extract
# ===================================================================

class TestStructuredExtract:

    def test_extracts_title(self, default_skill):
        resp = _make_mock_response(
            content=b"<html><head><title>Test Title</title></head><body></body></html>")
        result = default_skill._structured_extract(resp)
        assert result["title"] == "Test Title"

    def test_result_contains_url_and_status(self, default_skill):
        resp = _make_mock_response(url="https://example.com/page", status_code=200)
        result = default_skill._structured_extract(resp)
        assert result["url"] == "https://example.com/page"
        assert result["status_code"] == 200

    def test_no_selectors_returns_empty_data(self, default_skill):
        resp = _make_mock_response()
        result = default_skill._structured_extract(resp)
        assert result["data"] == {}

    def test_xpath_selector(self, default_skill):
        resp = _make_mock_response(
            content=b"<html><body><div class='main'><p>Hello</p></div></body></html>")
        result = default_skill._structured_extract(resp, selectors={"paragraph": "//p"})
        assert "paragraph" in result["data"]
        assert "Hello" in result["data"]["paragraph"]

    def test_xpath_selector_multiple_results(self, default_skill):
        resp = _make_mock_response(
            content=b"<html><body><p>One</p><p>Two</p><p>Three</p></body></html>")
        result = default_skill._structured_extract(resp, selectors={"items": "//p"})
        assert isinstance(result["data"]["items"], list)
        assert len(result["data"]["items"]) == 3

    def test_xpath_selector_single_result(self, default_skill):
        resp = _make_mock_response(
            content=b"<html><body><h1>Only One</h1></body></html>")
        result = default_skill._structured_extract(resp, selectors={"heading": "//h1"})
        # Single result should be a string, not a list
        assert isinstance(result["data"]["heading"], str)
        assert result["data"]["heading"] == "Only One"

    def test_invalid_xpath_returns_none_for_field(self, default_skill):
        resp = _make_mock_response(content=b"<html><body></body></html>")
        result = default_skill._structured_extract(
            resp, selectors={"bad": "///invalid[["})
        assert result["data"]["bad"] is None

    def test_general_parse_error_returns_error_dict(self, default_skill):
        with patch("signalwire_agents.skills.spider.skill.html.fromstring",
                    side_effect=Exception("parse failed")):
            resp = _make_mock_response()
            result = default_skill._structured_extract(resp)
            assert "error" in result

    def test_no_title_returns_empty_string(self, default_skill):
        resp = _make_mock_response(content=b"<html><body><p>No title</p></body></html>")
        result = default_skill._structured_extract(resp)
        assert result["title"] == ""


# ===================================================================
# _scrape_url_handler
# ===================================================================

class TestScrapeUrlHandler:

    def test_empty_url_returns_error_message(self, default_skill):
        result = default_skill._scrape_url_handler({"url": ""}, {})
        assert isinstance(result, SwaigFunctionResult)
        assert "provide a URL" in result.response

    def test_missing_url_returns_error_message(self, default_skill):
        result = default_skill._scrape_url_handler({}, {})
        assert "provide a URL" in result.response

    def test_invalid_url_no_scheme(self, default_skill):
        result = default_skill._scrape_url_handler({"url": "example.com"}, {})
        assert "Invalid URL" in result.response

    def test_invalid_url_no_netloc(self, default_skill):
        result = default_skill._scrape_url_handler({"url": "https://"}, {})
        assert "Invalid URL" in result.response

    def test_fetch_failure_returns_error(self, default_skill):
        with patch.object(default_skill, "_fetch_url", return_value=None):
            result = default_skill._scrape_url_handler(
                {"url": "https://example.com"}, {})
            assert "Failed to fetch" in result.response

    def test_successful_fast_text_extraction(self, default_skill):
        resp = _make_mock_response()
        with patch.object(default_skill, "_fetch_url", return_value=resp):
            with patch.object(default_skill, "_fast_text_extract",
                              return_value="Extracted content here"):
                result = default_skill._scrape_url_handler(
                    {"url": "https://example.com"}, {})
                assert "Extracted content here" in result.response
                assert "Content from" in result.response

    def test_successful_markdown_extraction(self, default_skill):
        default_skill.extract_type = "markdown"
        resp = _make_mock_response()
        with patch.object(default_skill, "_fetch_url", return_value=resp):
            with patch.object(default_skill, "_markdown_extract",
                              return_value="# Markdown content"):
                result = default_skill._scrape_url_handler(
                    {"url": "https://example.com"}, {})
                assert "# Markdown content" in result.response

    def test_structured_extraction(self, default_skill):
        default_skill.extract_type = "structured"
        resp = _make_mock_response()
        structured_data = {"url": "https://example.com", "title": "Test",
                           "status_code": 200, "data": {"field": "value"}}
        with patch.object(default_skill, "_fetch_url", return_value=resp):
            with patch.object(default_skill, "_structured_extract",
                              return_value=structured_data):
                result = default_skill._scrape_url_handler(
                    {"url": "https://example.com"}, {})
                assert "Extracted structured data" in result.response

    def test_empty_content_returns_no_content_message(self, default_skill):
        resp = _make_mock_response()
        with patch.object(default_skill, "_fetch_url", return_value=resp):
            with patch.object(default_skill, "_fast_text_extract",
                              return_value=""):
                result = default_skill._scrape_url_handler(
                    {"url": "https://example.com"}, {})
                assert "No content extracted" in result.response

    def test_exception_during_extraction_returns_error(self, default_skill):
        resp = _make_mock_response()
        with patch.object(default_skill, "_fetch_url", return_value=resp):
            with patch.object(default_skill, "_fast_text_extract",
                              side_effect=RuntimeError("boom")):
                result = default_skill._scrape_url_handler(
                    {"url": "https://example.com"}, {})
                assert "Error processing" in result.response

    def test_uses_configured_extract_type_not_from_args(self, default_skill):
        """Verify that extract_type comes from self.extract_type, not args."""
        resp = _make_mock_response()
        with patch.object(default_skill, "_fetch_url", return_value=resp):
            with patch.object(default_skill, "_fast_text_extract",
                              return_value="content") as mock_fast:
                # Even if args had extract_type, it should be ignored
                default_skill._scrape_url_handler(
                    {"url": "https://example.com", "extract_type": "markdown"}, {})
                mock_fast.assert_called_once()

    def test_response_includes_character_count(self, default_skill):
        resp = _make_mock_response()
        with patch.object(default_skill, "_fetch_url", return_value=resp):
            with patch.object(default_skill, "_fast_text_extract",
                              return_value="12345"):
                result = default_skill._scrape_url_handler(
                    {"url": "https://example.com"}, {})
                assert "5 characters" in result.response

    def test_whitespace_url_treated_as_empty(self, default_skill):
        result = default_skill._scrape_url_handler({"url": "   "}, {})
        assert "provide a URL" in result.response


# ===================================================================
# _crawl_site_handler
# ===================================================================

class TestCrawlSiteHandler:

    def test_empty_start_url_returns_error(self, default_skill):
        result = default_skill._crawl_site_handler({"start_url": ""}, {})
        assert "provide a starting URL" in result.response

    def test_missing_start_url_returns_error(self, default_skill):
        result = default_skill._crawl_site_handler({}, {})
        assert "provide a starting URL" in result.response

    def test_single_page_crawl(self, default_skill):
        """With max_depth=0 and max_pages=1, should crawl exactly one page."""
        resp = _make_mock_response(
            content=b"<html><body><p>Page content</p></body></html>")
        with patch.object(default_skill, "_fetch_url", return_value=resp):
            with patch.object(default_skill, "_fast_text_extract",
                              return_value="Page content"):
                result = default_skill._crawl_site_handler(
                    {"start_url": "https://example.com"}, {})
                assert "Crawled 1 pages" in result.response

    def test_no_pages_crawled_returns_error(self, default_skill):
        with patch.object(default_skill, "_fetch_url", return_value=None):
            result = default_skill._crawl_site_handler(
                {"start_url": "https://example.com"}, {})
            assert "No pages could be crawled" in result.response

    def test_multi_page_crawl_respects_max_pages(self, default_skill):
        default_skill.max_pages = 2
        default_skill.max_depth = 1
        default_skill.delay = 0  # Avoid sleep in tests

        page1_content = (
            b"<html><body>"
            b"<a href='/page2'>Link</a>"
            b"<p>Page 1</p>"
            b"</body></html>"
        )
        page2_content = (
            b"<html><body>"
            b"<a href='/page3'>Link</a>"
            b"<p>Page 2</p>"
            b"</body></html>"
        )

        resp1 = _make_mock_response(content=page1_content, url="https://example.com")
        resp2 = _make_mock_response(content=page2_content, url="https://example.com/page2")

        call_count = [0]

        def mock_fetch(url):
            call_count[0] += 1
            if call_count[0] == 1:
                return resp1
            elif call_count[0] == 2:
                return resp2
            return None

        with patch.object(default_skill, "_fetch_url", side_effect=mock_fetch):
            result = default_skill._crawl_site_handler(
                {"start_url": "https://example.com"}, {})
            assert "Crawled 2 pages" in result.response

    def test_crawl_skips_already_visited_urls(self, default_skill):
        default_skill.max_pages = 10
        default_skill.max_depth = 1
        default_skill.delay = 0

        page_content = (
            b"<html><body>"
            b"<a href='https://example.com'>Self link</a>"
            b"<a href='https://example.com'>Duplicate link</a>"
            b"<p>Content</p>"
            b"</body></html>"
        )
        resp = _make_mock_response(content=page_content, url="https://example.com")

        fetch_calls = []

        def mock_fetch(url):
            fetch_calls.append(url)
            return resp

        with patch.object(default_skill, "_fetch_url", side_effect=mock_fetch):
            result = default_skill._crawl_site_handler(
                {"start_url": "https://example.com"}, {})
            # Should only fetch the page once; the self-links should be recognized as visited
            assert len(fetch_calls) == 1
            assert "Crawled 1 pages" in result.response

    def test_crawl_respects_max_depth(self, default_skill):
        default_skill.max_pages = 10
        default_skill.max_depth = 0  # Only the start page
        default_skill.delay = 0

        page_content = (
            b"<html><body>"
            b"<a href='/page2'>Link</a>"
            b"<p>Content</p>"
            b"</body></html>"
        )
        resp = _make_mock_response(content=page_content, url="https://example.com")

        fetch_calls = []

        def mock_fetch(url):
            fetch_calls.append(url)
            return resp

        with patch.object(default_skill, "_fetch_url", side_effect=mock_fetch):
            result = default_skill._crawl_site_handler(
                {"start_url": "https://example.com"}, {})
            # With max_depth=0, should not follow links
            assert len(fetch_calls) == 1

    def test_crawl_follows_same_domain_only(self, default_skill):
        default_skill.max_pages = 10
        default_skill.max_depth = 1
        default_skill.delay = 0

        page_content = (
            b"<html><body>"
            b"<a href='https://other.com/page'>External</a>"
            b"<a href='/internal'>Internal</a>"
            b"<p>Content</p>"
            b"</body></html>"
        )
        resp = _make_mock_response(content=page_content, url="https://example.com")
        resp2 = _make_mock_response(
            content=b"<html><body><p>Internal</p></body></html>",
            url="https://example.com/internal")

        fetch_calls = []

        def mock_fetch(url):
            fetch_calls.append(url)
            if "internal" in url:
                return resp2
            return resp

        with patch.object(default_skill, "_fetch_url", side_effect=mock_fetch):
            result = default_skill._crawl_site_handler(
                {"start_url": "https://example.com"}, {})
            # Should not have fetched external domain
            assert not any("other.com" in u for u in fetch_calls)

    def test_crawl_with_follow_patterns(self, default_skill):
        default_skill.max_pages = 10
        default_skill.max_depth = 1
        default_skill.delay = 0
        default_skill.params["follow_patterns"] = [r"/blog/"]

        page_content = (
            b"<html><body>"
            b"<a href='/blog/post1'>Blog post</a>"
            b"<a href='/about'>About</a>"
            b"<p>Content</p>"
            b"</body></html>"
        )
        resp = _make_mock_response(content=page_content, url="https://example.com")
        blog_resp = _make_mock_response(
            content=b"<html><body><p>Blog</p></body></html>",
            url="https://example.com/blog/post1")

        fetch_calls = []

        def mock_fetch(url):
            fetch_calls.append(url)
            if "blog" in url:
                return blog_resp
            return resp

        with patch.object(default_skill, "_fetch_url", side_effect=mock_fetch):
            result = default_skill._crawl_site_handler(
                {"start_url": "https://example.com"}, {})
            # Should follow the blog link but not the about link
            assert any("blog" in u for u in fetch_calls)
            assert not any("about" in u for u in fetch_calls)

    def test_crawl_summary_contains_total_characters(self, default_skill):
        resp = _make_mock_response()
        with patch.object(default_skill, "_fetch_url", return_value=resp):
            with patch.object(default_skill, "_fast_text_extract",
                              return_value="Hello World"):
                result = default_skill._crawl_site_handler(
                    {"start_url": "https://example.com"}, {})
                assert "Total content:" in result.response
                assert "characters" in result.response

    def test_crawl_handles_fetch_failure_for_individual_pages(self, default_skill):
        default_skill.max_pages = 5
        default_skill.max_depth = 1
        default_skill.delay = 0

        page_content = (
            b"<html><body>"
            b"<a href='/page2'>Link</a>"
            b"<p>Content</p>"
            b"</body></html>"
        )
        resp = _make_mock_response(content=page_content, url="https://example.com")

        call_count = [0]

        def mock_fetch(url):
            call_count[0] += 1
            if call_count[0] == 1:
                return resp
            return None  # All other fetches fail

        with patch.object(default_skill, "_fetch_url", side_effect=mock_fetch):
            result = default_skill._crawl_site_handler(
                {"start_url": "https://example.com"}, {})
            assert "Crawled 1 pages" in result.response

    def test_crawl_delays_between_requests(self, default_skill):
        default_skill.max_pages = 2
        default_skill.max_depth = 1
        default_skill.delay = 0.5

        page1_content = (
            b"<html><body>"
            b"<a href='/page2'>Link</a>"
            b"<p>Page 1</p>"
            b"</body></html>"
        )
        resp1 = _make_mock_response(content=page1_content, url="https://example.com")
        resp2 = _make_mock_response(
            content=b"<html><body><p>Page 2</p></body></html>",
            url="https://example.com/page2")

        call_count = [0]

        def mock_fetch(url):
            call_count[0] += 1
            if call_count[0] == 1:
                return resp1
            return resp2

        with patch.object(default_skill, "_fetch_url", side_effect=mock_fetch):
            with patch("time.sleep") as mock_sleep:
                result = default_skill._crawl_site_handler(
                    {"start_url": "https://example.com"}, {})
                mock_sleep.assert_called_with(0.5)

    def test_content_summary_truncated_at_500(self, default_skill):
        long_content = "A" * 1000
        resp = _make_mock_response()
        with patch.object(default_skill, "_fetch_url", return_value=resp):
            with patch.object(default_skill, "_fast_text_extract",
                              return_value=long_content):
                result = default_skill._crawl_site_handler(
                    {"start_url": "https://example.com"}, {})
                # The summary field should be truncated at 500 chars
                assert "..." in result.response


# ===================================================================
# _extract_structured_handler
# ===================================================================

class TestExtractStructuredHandler:

    def test_empty_url_returns_error(self, default_skill):
        result = default_skill._extract_structured_handler({"url": ""}, {})
        assert "provide a URL" in result.response

    def test_missing_url_returns_error(self, default_skill):
        result = default_skill._extract_structured_handler({}, {})
        assert "provide a URL" in result.response

    def test_no_selectors_configured_returns_error(self, default_skill):
        result = default_skill._extract_structured_handler(
            {"url": "https://example.com"}, {})
        assert "No selectors configured" in result.response

    def test_fetch_failure_returns_error(self, custom_skill):
        # custom_skill has selectors configured
        with patch.object(custom_skill, "_fetch_url", return_value=None):
            result = custom_skill._extract_structured_handler(
                {"url": "https://example.com"}, {})
            assert "Failed to fetch" in result.response

    def test_successful_extraction(self, custom_skill):
        structured_result = {
            "url": "https://example.com",
            "title": "Test Page",
            "status_code": 200,
            "data": {"title": "Extracted Title"}
        }
        resp = _make_mock_response()
        with patch.object(custom_skill, "_fetch_url", return_value=resp):
            with patch.object(custom_skill, "_structured_extract",
                              return_value=structured_result):
                result = custom_skill._extract_structured_handler(
                    {"url": "https://example.com"}, {})
                assert "Extracted data from" in result.response
                assert "Test Page" in result.response
                assert "title: Extracted Title" in result.response

    def test_extraction_error_in_result(self, custom_skill):
        resp = _make_mock_response()
        with patch.object(custom_skill, "_fetch_url", return_value=resp):
            with patch.object(custom_skill, "_structured_extract",
                              return_value={"error": "Something went wrong"}):
                result = custom_skill._extract_structured_handler(
                    {"url": "https://example.com"}, {})
                assert "Error extracting data" in result.response

    def test_empty_data_says_no_data_extracted(self, custom_skill):
        structured_result = {
            "url": "https://example.com",
            "title": "Test Page",
            "status_code": 200,
            "data": {}
        }
        resp = _make_mock_response()
        with patch.object(custom_skill, "_fetch_url", return_value=resp):
            with patch.object(custom_skill, "_structured_extract",
                              return_value=structured_result):
                result = custom_skill._extract_structured_handler(
                    {"url": "https://example.com"}, {})
                assert "No data extracted" in result.response

    def test_uses_selectors_from_params(self, custom_skill):
        resp = _make_mock_response()
        with patch.object(custom_skill, "_fetch_url", return_value=resp):
            with patch.object(custom_skill, "_structured_extract",
                              return_value={"url": "", "title": "", "status_code": 200,
                                            "data": {}}) as mock_extract:
                custom_skill._extract_structured_handler(
                    {"url": "https://example.com"}, {})
                # Verify selectors from params are passed
                call_args = mock_extract.call_args
                assert call_args[1].get("selectors") == {"title": "//title/text()"} or \
                       call_args[0][1] == {"title": "//title/text()"}


# ===================================================================
# get_hints
# ===================================================================

class TestGetHints:

    def test_returns_list(self, default_skill):
        hints = default_skill.get_hints()
        assert isinstance(hints, list)

    def test_contains_expected_hints(self, default_skill):
        hints = default_skill.get_hints()
        assert "scrape" in hints
        assert "crawl" in hints
        assert "spider" in hints
        assert "website" in hints

    def test_hints_are_strings(self, default_skill):
        hints = default_skill.get_hints()
        for hint in hints:
            assert isinstance(hint, str)


# ===================================================================
# cleanup
# ===================================================================

class TestCleanup:

    def test_closes_session(self, default_skill):
        default_skill.cleanup()
        default_skill.session.close.assert_called_once()

    def test_clears_cache(self, default_skill):
        default_skill.cache["https://example.com"] = "cached"
        default_skill.cleanup()
        assert len(default_skill.cache) == 0

    def test_cleanup_with_none_cache(self, custom_skill):
        # custom_skill has cache_enabled=False, so cache is None
        # Should not raise an error
        custom_skill.cleanup()

    def test_cleanup_without_session_attribute(self, default_skill):
        # Remove session to simulate edge case
        del default_skill.session
        # Should not raise
        default_skill.cleanup()

    def test_cleanup_without_cache_attribute(self, default_skill):
        del default_skill.cache
        # Should not raise
        default_skill.cleanup()

    def test_cleanup_logs_info(self, default_skill):
        with patch.object(default_skill.logger, "info") as mock_info:
            default_skill.cleanup()
            mock_info.assert_called_once()
            assert "cleaned up" in mock_info.call_args[0][0]


# ===================================================================
# Edge cases and integration-style scenarios
# ===================================================================

class TestEdgeCases:

    def test_url_with_whitespace_stripped(self, default_skill):
        """URLs with leading/trailing whitespace should be stripped."""
        resp = _make_mock_response()
        with patch.object(default_skill, "_fetch_url", return_value=resp):
            with patch.object(default_skill, "_fast_text_extract",
                              return_value="content"):
                result = default_skill._scrape_url_handler(
                    {"url": "  https://example.com  "}, {})
                assert "content" in result.response

    def test_cache_prevents_duplicate_fetches(self, default_skill):
        resp = _make_mock_response()
        default_skill.session.get = Mock(return_value=resp)

        # Fetch twice
        result1 = default_skill._fetch_url("https://example.com")
        result2 = default_skill._fetch_url("https://example.com")

        # session.get should only be called once; second call hits cache
        default_skill.session.get.assert_called_once()
        assert result1 is result2

    def test_scrape_handler_url_with_only_scheme(self, default_skill):
        result = default_skill._scrape_url_handler({"url": "ftp://"}, {})
        assert "Invalid URL" in result.response

    def test_init_with_empty_params(self, mock_agent):
        """Skill should initialize fine with no params at all."""
        with patch("signalwire_agents.skills.spider.skill.requests.Session") as MockSession:
            mock_session = Mock()
            mock_session.headers = {}
            MockSession.return_value = mock_session

            from signalwire_agents.skills.spider.skill import SpiderSkill
            skill = SpiderSkill(mock_agent, {})
            assert skill.delay == 0.1
            assert skill.cache == {}

    def test_init_with_none_params(self, mock_agent):
        """Skill should handle None params gracefully (via SkillBase default)."""
        with patch("signalwire_agents.skills.spider.skill.requests.Session") as MockSession:
            mock_session = Mock()
            mock_session.headers = {}
            MockSession.return_value = mock_session

            from signalwire_agents.skills.spider.skill import SpiderSkill
            skill = SpiderSkill(mock_agent, None)
            assert skill.delay == 0.1

    def test_register_tools_no_prefix_when_tool_name_empty(self, mock_agent):
        with patch("signalwire_agents.skills.spider.skill.requests.Session") as MockSession:
            mock_session = Mock()
            mock_session.headers = {}
            MockSession.return_value = mock_session

            from signalwire_agents.skills.spider.skill import SpiderSkill
            skill = SpiderSkill(mock_agent, {"tool_name": ""})
            skill.register_tools()
            names = [call.kwargs.get("name") or call[1].get("name")
                     for call in mock_agent.define_tool.call_args_list]
            # Empty tool_name should not add a prefix
            assert "scrape_url" in names

    def test_fast_text_truncation_preserves_start_and_end(self, default_skill):
        """Verify the smart truncation keeps 2/3 from start and 1/3 from end."""
        default_skill.max_text_length = 300
        body = "S" * 200 + "M" * 100 + "E" * 200
        resp = _make_mock_response(
            content=f"<html><body><p>{body}</p></body></html>".encode())
        text = default_skill._fast_text_extract(resp)
        assert text.startswith("S")
        assert text.endswith("E")
        assert "[...CONTENT TRUNCATED...]" in text

    def test_structured_extract_css_selector(self, default_skill):
        """CSS selectors (not starting with /) should be handled via CSSSelector."""
        resp = _make_mock_response(
            content=b"<html><body><div class='content'><p>CSS content</p></div></body></html>")

        mock_element = Mock()
        mock_element.text_content.return_value = "CSS content"

        mock_css_cls = Mock()
        mock_css_instance = Mock(return_value=[mock_element])
        mock_css_cls.return_value = mock_css_instance

        # The import is `from lxml.cssselect import CSSSelector` inside the method.
        # Create a fake module and inject it into sys.modules.
        import sys
        fake_cssselect = MagicMock()
        fake_cssselect.CSSSelector = mock_css_cls

        saved = sys.modules.get("lxml.cssselect")
        sys.modules["lxml.cssselect"] = fake_cssselect
        try:
            result = default_skill._structured_extract(
                resp, selectors={"para": "div.content p"})
            assert "para" in result["data"]
            assert result["data"]["para"] == "CSS content"
        finally:
            if saved is not None:
                sys.modules["lxml.cssselect"] = saved
            else:
                sys.modules.pop("lxml.cssselect", None)

    def test_crawl_link_extraction_error_handled(self, default_skill):
        """Error during link extraction should not crash the crawl."""
        default_skill.max_pages = 5
        default_skill.max_depth = 1
        default_skill.delay = 0

        resp = _make_mock_response()
        with patch.object(default_skill, "_fetch_url", return_value=resp):
            with patch.object(default_skill, "_fast_text_extract",
                              return_value="content"):
                with patch("signalwire_agents.skills.spider.skill.html.fromstring",
                            side_effect=Exception("parse error")):
                    # The crawl handler internally calls html.fromstring for link extraction
                    # but _fast_text_extract is mocked to succeed
                    result = default_skill._crawl_site_handler(
                        {"start_url": "https://example.com"}, {})
                    # Should still return results for the page that was crawled
                    assert "Crawled 1 pages" in result.response
