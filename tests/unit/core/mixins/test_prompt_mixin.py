"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for PromptMixin
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from typing import Dict, List, Any, Optional

from signalwire_agents.core.mixins.prompt_mixin import PromptMixin


class MockPromptHost(PromptMixin):
    """
    A minimal host class that inherits from PromptMixin and provides
    all the attributes the mixin expects to find on self.
    """

    def __init__(
        self,
        use_pom=True,
        pom=None,
        name="TestAgent",
        prompt_manager=None,
        contexts_builder=None,
        contexts_defined=False,
    ):
        self._use_pom = use_pom
        self.pom = pom
        self.name = name
        self._prompt_manager = prompt_manager or Mock()
        self._contexts_builder = contexts_builder
        self._contexts_defined = contexts_defined
        self.log = Mock()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_prompt_manager():
    """Return a fresh Mock standing in for PromptManager."""
    pm = Mock()
    pm.get_prompt.return_value = None
    pm.get_post_prompt.return_value = None
    return pm


@pytest.fixture
def host(mock_prompt_manager):
    """Return a MockPromptHost wired with a mock prompt manager."""
    return MockPromptHost(prompt_manager=mock_prompt_manager)


# ===========================================================================
# Tests for set_prompt_text
# ===========================================================================

class TestSetPromptText:
    """Tests for PromptMixin.set_prompt_text"""

    def test_delegates_to_prompt_manager(self, host):
        result = host.set_prompt_text("Hello world")
        host._prompt_manager.set_prompt_text.assert_called_once_with("Hello world")

    def test_returns_self_for_chaining(self, host):
        result = host.set_prompt_text("prompt")
        assert result is host

    def test_empty_string_is_accepted(self, host):
        result = host.set_prompt_text("")
        host._prompt_manager.set_prompt_text.assert_called_once_with("")
        assert result is host

    def test_long_prompt_text(self, host):
        long_text = "x" * 10000
        result = host.set_prompt_text(long_text)
        host._prompt_manager.set_prompt_text.assert_called_once_with(long_text)
        assert result is host


# ===========================================================================
# Tests for set_post_prompt
# ===========================================================================

class TestSetPostPrompt:
    """Tests for PromptMixin.set_post_prompt"""

    def test_delegates_to_prompt_manager(self, host):
        result = host.set_post_prompt("Summarize the conversation")
        host._prompt_manager.set_post_prompt.assert_called_once_with("Summarize the conversation")

    def test_returns_self_for_chaining(self, host):
        result = host.set_post_prompt("summary")
        assert result is host

    def test_empty_string(self, host):
        result = host.set_post_prompt("")
        host._prompt_manager.set_post_prompt.assert_called_once_with("")
        assert result is host


# ===========================================================================
# Tests for set_prompt_pom
# ===========================================================================

class TestSetPromptPom:
    """Tests for PromptMixin.set_prompt_pom"""

    def test_delegates_to_prompt_manager(self, host):
        pom_data = [{"title": "Section A", "body": "Body A"}]
        result = host.set_prompt_pom(pom_data)
        host._prompt_manager.set_prompt_pom.assert_called_once_with(pom_data)

    def test_returns_self_for_chaining(self, host):
        result = host.set_prompt_pom([])
        assert result is host

    def test_empty_list(self, host):
        result = host.set_prompt_pom([])
        host._prompt_manager.set_prompt_pom.assert_called_once_with([])
        assert result is host

    def test_complex_pom_structure(self, host):
        pom_data = [
            {"title": "Section A", "body": "Body A", "bullets": ["b1", "b2"]},
            {"title": "Section B", "body": "Body B", "subsections": [
                {"title": "Sub B1", "body": "Sub body"}
            ]},
        ]
        result = host.set_prompt_pom(pom_data)
        host._prompt_manager.set_prompt_pom.assert_called_once_with(pom_data)
        assert result is host


# ===========================================================================
# Tests for prompt_add_section
# ===========================================================================

class TestPromptAddSection:
    """Tests for PromptMixin.prompt_add_section"""

    def test_delegates_basic_section(self, host):
        result = host.prompt_add_section("Intro", body="Welcome")
        host._prompt_manager.prompt_add_section.assert_called_once_with(
            title="Intro",
            body="Welcome",
            bullets=None,
            numbered=False,
            numbered_bullets=False,
            subsections=None,
        )

    def test_returns_self_for_chaining(self, host):
        result = host.prompt_add_section("Title")
        assert result is host

    def test_with_bullets(self, host):
        bullets = ["Point 1", "Point 2"]
        host.prompt_add_section("Rules", bullets=bullets)
        host._prompt_manager.prompt_add_section.assert_called_once_with(
            title="Rules",
            body="",
            bullets=bullets,
            numbered=False,
            numbered_bullets=False,
            subsections=None,
        )

    def test_with_numbered_flags(self, host):
        host.prompt_add_section("Steps", numbered=True, numbered_bullets=True)
        host._prompt_manager.prompt_add_section.assert_called_once_with(
            title="Steps",
            body="",
            bullets=None,
            numbered=True,
            numbered_bullets=True,
            subsections=None,
        )

    def test_with_subsections(self, host):
        subs = [{"title": "Sub1", "body": "sub body"}]
        host.prompt_add_section("Main", subsections=subs)
        host._prompt_manager.prompt_add_section.assert_called_once_with(
            title="Main",
            body="",
            bullets=None,
            numbered=False,
            numbered_bullets=False,
            subsections=subs,
        )

    def test_all_parameters(self, host):
        bullets = ["a", "b"]
        subs = [{"title": "Sub", "body": "sb"}]
        host.prompt_add_section(
            "Full",
            body="body text",
            bullets=bullets,
            numbered=True,
            numbered_bullets=True,
            subsections=subs,
        )
        host._prompt_manager.prompt_add_section.assert_called_once_with(
            title="Full",
            body="body text",
            bullets=bullets,
            numbered=True,
            numbered_bullets=True,
            subsections=subs,
        )


# ===========================================================================
# Tests for prompt_add_to_section
# ===========================================================================

class TestPromptAddToSection:
    """Tests for PromptMixin.prompt_add_to_section"""

    def test_add_body(self, host):
        result = host.prompt_add_to_section("Intro", body="More text")
        host._prompt_manager.prompt_add_to_section.assert_called_once_with(
            title="Intro",
            body="More text",
            bullet=None,
            bullets=None,
        )

    def test_add_single_bullet(self, host):
        host.prompt_add_to_section("Rules", bullet="New rule")
        host._prompt_manager.prompt_add_to_section.assert_called_once_with(
            title="Rules",
            body=None,
            bullet="New rule",
            bullets=None,
        )

    def test_add_multiple_bullets(self, host):
        bullets = ["rule1", "rule2"]
        host.prompt_add_to_section("Rules", bullets=bullets)
        host._prompt_manager.prompt_add_to_section.assert_called_once_with(
            title="Rules",
            body=None,
            bullet=None,
            bullets=bullets,
        )

    def test_add_body_and_bullets(self, host):
        host.prompt_add_to_section("Mixed", body="intro", bullet="b1", bullets=["b2"])
        host._prompt_manager.prompt_add_to_section.assert_called_once_with(
            title="Mixed",
            body="intro",
            bullet="b1",
            bullets=["b2"],
        )

    def test_returns_self_for_chaining(self, host):
        result = host.prompt_add_to_section("S")
        assert result is host


# ===========================================================================
# Tests for prompt_add_subsection
# ===========================================================================

class TestPromptAddSubsection:
    """Tests for PromptMixin.prompt_add_subsection"""

    def test_basic_subsection(self, host):
        result = host.prompt_add_subsection("Parent", "Child", body="child body")
        host._prompt_manager.prompt_add_subsection.assert_called_once_with(
            parent_title="Parent",
            title="Child",
            body="child body",
            bullets=None,
        )

    def test_subsection_with_bullets(self, host):
        bullets = ["x", "y"]
        host.prompt_add_subsection("P", "C", bullets=bullets)
        host._prompt_manager.prompt_add_subsection.assert_called_once_with(
            parent_title="P",
            title="C",
            body="",
            bullets=bullets,
        )

    def test_returns_self_for_chaining(self, host):
        result = host.prompt_add_subsection("P", "C")
        assert result is host


# ===========================================================================
# Tests for prompt_has_section
# ===========================================================================

class TestPromptHasSection:
    """Tests for PromptMixin.prompt_has_section"""

    def test_section_exists(self, host):
        host._prompt_manager.prompt_has_section.return_value = True
        assert host.prompt_has_section("Intro") is True
        host._prompt_manager.prompt_has_section.assert_called_once_with("Intro")

    def test_section_does_not_exist(self, host):
        host._prompt_manager.prompt_has_section.return_value = False
        assert host.prompt_has_section("NonExistent") is False

    def test_empty_title(self, host):
        host._prompt_manager.prompt_has_section.return_value = False
        assert host.prompt_has_section("") is False


# ===========================================================================
# Tests for get_prompt
# ===========================================================================

class TestGetPrompt:
    """Tests for PromptMixin.get_prompt"""

    def test_returns_prompt_manager_result_when_available(self, host):
        host._prompt_manager.get_prompt.return_value = "Manager prompt"
        assert host.get_prompt() == "Manager prompt"

    def test_returns_pom_render_dict_when_available(self, host):
        host._prompt_manager.get_prompt.return_value = None
        mock_pom = Mock()
        mock_pom.render_dict.return_value = [{"title": "S", "body": "B"}]
        host.pom = mock_pom
        host._use_pom = True

        result = host.get_prompt()
        assert result == [{"title": "S", "body": "B"}]
        mock_pom.render_dict.assert_called_once()

    def test_falls_back_to_to_dict(self, host):
        host._prompt_manager.get_prompt.return_value = None
        mock_pom = Mock(spec=[])  # no render_dict
        mock_pom.to_dict = Mock(return_value=[{"title": "T"}])
        host.pom = mock_pom
        host._use_pom = True

        result = host.get_prompt()
        assert result == [{"title": "T"}]

    def test_falls_back_to_to_list(self, host):
        host._prompt_manager.get_prompt.return_value = None
        mock_pom = Mock(spec=[])
        mock_pom.to_list = Mock(return_value=[{"title": "L"}])
        host.pom = mock_pom
        host._use_pom = True

        result = host.get_prompt()
        assert result == [{"title": "L"}]

    def test_falls_back_to_render_returning_json_string(self, host):
        host._prompt_manager.get_prompt.return_value = None
        mock_pom = Mock(spec=[])
        mock_pom.render = Mock(return_value='[{"title": "R"}]')
        host.pom = mock_pom
        host._use_pom = True

        result = host.get_prompt()
        assert result == [{"title": "R"}]

    def test_render_returning_non_json_string_returns_raw(self, host):
        """When render() returns a non-JSON string, the raw string is still returned.

        The inner try/except catches the JSON decode error and passes, but
        execution then hits ``return render_result`` which returns the raw
        string rather than falling through to the default prompt.
        """
        host._prompt_manager.get_prompt.return_value = None
        mock_pom = Mock(spec=[])
        mock_pom.render = Mock(return_value="not json at all {{{")
        host.pom = mock_pom
        host._use_pom = True

        result = host.get_prompt()
        assert result == "not json at all {{{"

    def test_render_returning_list_directly(self, host):
        host._prompt_manager.get_prompt.return_value = None
        mock_pom = Mock(spec=[])
        mock_pom.render = Mock(return_value=[{"title": "Direct"}])
        host.pom = mock_pom
        host._use_pom = True

        result = host.get_prompt()
        assert result == [{"title": "Direct"}]

    def test_falls_back_to_pom_sections_attribute(self, host):
        """When no standard method exists, the code inspects pom.__dict__['_sections']."""
        host._prompt_manager.get_prompt.return_value = None

        class BarePom:
            def __init__(self):
                self._sections = [{"title": "bare"}]

        host.pom = BarePom()
        host._use_pom = True

        result = host.get_prompt()
        assert result == [{"title": "bare"}]

    def test_default_prompt_when_pom_not_in_use(self, host):
        host._prompt_manager.get_prompt.return_value = None
        host._use_pom = False
        host.name = "Acme Bot"
        assert host.get_prompt() == "You are Acme Bot, a helpful AI assistant."

    def test_default_prompt_when_pom_is_none(self, host):
        host._prompt_manager.get_prompt.return_value = None
        host._use_pom = True
        host.pom = None
        host.name = "Helper"
        assert host.get_prompt() == "You are Helper, a helpful AI assistant."

    def test_pom_exception_falls_back_to_default(self, host):
        """When the POM raises an exception, the mixin logs and returns default."""
        host._prompt_manager.get_prompt.return_value = None
        mock_pom = Mock()
        mock_pom.render_dict.side_effect = RuntimeError("boom")
        host.pom = mock_pom
        host._use_pom = True
        host.name = "CrashBot"

        result = host.get_prompt()
        assert result == "You are CrashBot, a helpful AI assistant."
        host.log.error.assert_called_once()

    def test_pom_with_empty_sections_dict_falls_to_default(self, host):
        """When __dict__['_sections'] is not a list, fall to default."""
        host._prompt_manager.get_prompt.return_value = None

        class WeirdPom:
            def __init__(self):
                self._sections = "not a list"

        host.pom = WeirdPom()
        host._use_pom = True
        host.name = "Bot"

        result = host.get_prompt()
        assert result == "You are Bot, a helpful AI assistant."

    def test_prompt_manager_returns_list(self, host):
        """When prompt_manager.get_prompt returns a list, it is returned directly."""
        host._prompt_manager.get_prompt.return_value = [{"title": "FromManager"}]
        assert host.get_prompt() == [{"title": "FromManager"}]


# ===========================================================================
# Tests for get_post_prompt
# ===========================================================================

class TestGetPostPrompt:
    """Tests for PromptMixin.get_post_prompt"""

    def test_returns_prompt_manager_result(self, host):
        host._prompt_manager.get_post_prompt.return_value = "Post text"
        assert host.get_post_prompt() == "Post text"

    def test_returns_none_when_not_set(self, host):
        host._prompt_manager.get_post_prompt.return_value = None
        assert host.get_post_prompt() is None


# ===========================================================================
# Tests for _validate_prompt_mode_exclusivity
# ===========================================================================

class TestValidatePromptModeExclusivity:
    """Tests for PromptMixin._validate_prompt_mode_exclusivity"""

    def test_delegates_to_prompt_manager(self, host):
        host._validate_prompt_mode_exclusivity()
        host._prompt_manager._validate_prompt_mode_exclusivity.assert_called_once()

    def test_propagates_value_error(self, host):
        host._prompt_manager._validate_prompt_mode_exclusivity.side_effect = ValueError("conflict")
        with pytest.raises(ValueError, match="conflict"):
            host._validate_prompt_mode_exclusivity()


# ===========================================================================
# Tests for define_contexts (with argument)
# ===========================================================================

class TestDefineContextsWithArg:
    """Tests for PromptMixin.define_contexts when called with contexts arg"""

    def test_sets_contexts_and_returns_self(self, host):
        ctx = {"main": {"steps": []}}
        result = host.define_contexts(contexts=ctx)
        host._prompt_manager.define_contexts.assert_called_once_with(ctx)
        assert result is host

    def test_with_dict_contexts(self, host):
        ctx = {"ctx1": {"steps": [{"name": "s1"}]}}
        result = host.define_contexts(contexts=ctx)
        assert result is host
        host._prompt_manager.define_contexts.assert_called_once_with(ctx)


# ===========================================================================
# Tests for define_contexts (without argument) -- returns ContextBuilder
# ===========================================================================

class TestDefineContextsWithoutArg:
    """Tests for PromptMixin.define_contexts when called without contexts arg"""

    @patch("signalwire_agents.core.mixins.prompt_mixin.ContextBuilder")
    def test_creates_context_builder_on_first_call(self, MockCB, host):
        host._contexts_builder = None
        mock_cb_instance = MockCB.return_value

        result = host.define_contexts()

        MockCB.assert_called_once_with(host)
        assert result is mock_cb_instance
        assert host._contexts_defined is True

    @patch("signalwire_agents.core.mixins.prompt_mixin.ContextBuilder")
    def test_returns_existing_builder_on_subsequent_calls(self, MockCB, host):
        existing_builder = Mock()
        host._contexts_builder = existing_builder

        result = host.define_contexts()
        # Should not create a new ContextBuilder
        MockCB.assert_not_called()
        assert result is existing_builder


# ===========================================================================
# Tests for contexts property
# ===========================================================================

class TestContextsProperty:
    """Tests for PromptMixin.contexts property"""

    @patch("signalwire_agents.core.mixins.prompt_mixin.ContextBuilder")
    def test_returns_context_builder(self, MockCB, host):
        host._contexts_builder = None
        mock_cb = MockCB.return_value

        result = host.contexts
        assert result is mock_cb

    def test_returns_existing_builder(self, host):
        existing = Mock()
        host._contexts_builder = existing

        assert host.contexts is existing


# ===========================================================================
# Tests for _process_prompt_sections
# ===========================================================================

class TestProcessPromptSections:
    """Tests for PromptMixin._process_prompt_sections"""

    # ----- Skipping conditions -----

    def test_skipped_when_no_prompt_sections_attr(self, host):
        """No PROMPT_SECTIONS attribute => nothing happens."""
        assert not hasattr(host.__class__, "PROMPT_SECTIONS")
        host._process_prompt_sections()
        host._prompt_manager.prompt_add_section.assert_not_called()

    def test_skipped_when_prompt_sections_is_none(self, host):
        host.__class__.PROMPT_SECTIONS = None
        try:
            host._process_prompt_sections()
            host._prompt_manager.prompt_add_section.assert_not_called()
        finally:
            del host.__class__.PROMPT_SECTIONS

    def test_skipped_when_use_pom_is_false(self, host):
        host.__class__.PROMPT_SECTIONS = {"Section": "content"}
        host._use_pom = False
        try:
            host._process_prompt_sections()
            host._prompt_manager.prompt_add_section.assert_not_called()
        finally:
            del host.__class__.PROMPT_SECTIONS

    # ----- Dict-based PROMPT_SECTIONS -----

    def test_dict_with_string_content(self):
        """Dict mapping title -> plain string adds a body section."""

        class StrHost(MockPromptHost):
            PROMPT_SECTIONS = {"Greeting": "Hello there"}

        h = StrHost()
        h._process_prompt_sections()
        h.prompt_add_section = Mock()
        # Re-run after patching to verify the call was made by the real method
        # Better approach: spy on the instance
        pm = Mock()
        h._prompt_manager = pm
        # Need to call again fresh
        h2 = StrHost(prompt_manager=pm)
        # Spy on the instance method
        h2.prompt_add_section = Mock()
        h2._process_prompt_sections()
        h2.prompt_add_section.assert_called_once_with("Greeting", body="Hello there")

    def test_dict_with_list_content(self):
        """Dict mapping title -> list of strings adds bullets."""

        class ListHost(MockPromptHost):
            PROMPT_SECTIONS = {"Rules": ["Rule 1", "Rule 2"]}

        h = ListHost()
        h.prompt_add_section = Mock()
        h._process_prompt_sections()
        h.prompt_add_section.assert_called_once_with("Rules", bullets=["Rule 1", "Rule 2"])

    def test_dict_with_empty_list_skipped(self):
        """Dict mapping title -> empty list does NOT create a section."""

        class EmptyListHost(MockPromptHost):
            PROMPT_SECTIONS = {"Empty": []}

        h = EmptyListHost()
        h.prompt_add_section = Mock()
        h._process_prompt_sections()
        h.prompt_add_section.assert_not_called()

    def test_dict_with_dict_content_body_only(self):
        """Dict mapping title -> dict with body key."""

        class DictBodyHost(MockPromptHost):
            PROMPT_SECTIONS = {
                "Info": {"body": "Some info"}
            }

        h = DictBodyHost()
        h.prompt_add_section = Mock()
        h.prompt_add_subsection = Mock()
        h._process_prompt_sections()
        h.prompt_add_section.assert_called_once_with(
            "Info",
            body="Some info",
            bullets=None,
            numbered=False,
            numbered_bullets=False,
        )

    def test_dict_with_dict_content_bullets(self):
        """Dict mapping title -> dict with bullets key."""

        class DictBulletsHost(MockPromptHost):
            PROMPT_SECTIONS = {
                "Tips": {"bullets": ["Tip 1", "Tip 2"]}
            }

        h = DictBulletsHost()
        h.prompt_add_section = Mock()
        h.prompt_add_subsection = Mock()
        h._process_prompt_sections()
        h.prompt_add_section.assert_called_once_with(
            "Tips",
            body="",
            bullets=["Tip 1", "Tip 2"],
            numbered=False,
            numbered_bullets=False,
        )

    def test_dict_with_dict_content_numbered(self):
        """Dict mapping title -> dict with numbered flags."""

        class NumHost(MockPromptHost):
            PROMPT_SECTIONS = {
                "Steps": {
                    "body": "Follow these steps",
                    "bullets": ["Step 1", "Step 2"],
                    "numbered": True,
                    "numberedBullets": True,
                }
            }

        h = NumHost()
        h.prompt_add_section = Mock()
        h.prompt_add_subsection = Mock()
        h._process_prompt_sections()
        h.prompt_add_section.assert_called_once_with(
            "Steps",
            body="Follow these steps",
            bullets=["Step 1", "Step 2"],
            numbered=True,
            numbered_bullets=True,
        )

    def test_dict_with_dict_content_empty_skipped(self):
        """Dict -> dict with no body, no bullets, no subsections => section is skipped."""

        class EmptyDictHost(MockPromptHost):
            PROMPT_SECTIONS = {
                "Nothing": {}
            }

        h = EmptyDictHost()
        h.prompt_add_section = Mock()
        h._process_prompt_sections()
        h.prompt_add_section.assert_not_called()

    def test_dict_with_subsections(self):
        """Dict -> dict containing subsections list."""

        class SubHost(MockPromptHost):
            PROMPT_SECTIONS = {
                "Parent": {
                    "body": "parent body",
                    "subsections": [
                        {"title": "Child1", "body": "child1 body"},
                        {"title": "Child2", "bullets": ["b1"]},
                    ],
                }
            }

        h = SubHost()
        h.prompt_add_section = Mock()
        h.prompt_add_subsection = Mock()
        h._process_prompt_sections()

        h.prompt_add_section.assert_called_once_with(
            "Parent",
            body="parent body",
            bullets=None,
            numbered=False,
            numbered_bullets=False,
        )
        assert h.prompt_add_subsection.call_count == 2
        h.prompt_add_subsection.assert_any_call(
            "Parent", "Child1", body="child1 body", bullets=None,
        )
        h.prompt_add_subsection.assert_any_call(
            "Parent", "Child2", body="", bullets=["b1"],
        )

    def test_dict_subsection_without_title_skipped(self):
        """Subsections without a 'title' key are skipped."""

        class NoTitleSubHost(MockPromptHost):
            PROMPT_SECTIONS = {
                "Parent": {
                    "body": "body",
                    "subsections": [
                        {"body": "no title here"},
                    ],
                }
            }

        h = NoTitleSubHost()
        h.prompt_add_section = Mock()
        h.prompt_add_subsection = Mock()
        h._process_prompt_sections()
        h.prompt_add_subsection.assert_not_called()

    def test_dict_subsection_empty_body_and_bullets_skipped(self):
        """Subsections with empty body and empty bullets are skipped."""

        class EmptySubHost(MockPromptHost):
            PROMPT_SECTIONS = {
                "Parent": {
                    "body": "body",
                    "subsections": [
                        {"title": "EmptySub", "body": "", "bullets": []},
                    ],
                }
            }

        h = EmptySubHost()
        h.prompt_add_section = Mock()
        h.prompt_add_subsection = Mock()
        h._process_prompt_sections()
        h.prompt_add_subsection.assert_not_called()

    # ----- List-based PROMPT_SECTIONS -----

    def test_list_sections_with_pom(self):
        """List-based PROMPT_SECTIONS processed when POM is available."""
        mock_pom = Mock()

        class ListSectionHost(MockPromptHost):
            PROMPT_SECTIONS = [
                {"title": "Section A", "body": "Body A"},
                {"title": "Section B", "bullets": ["b1", "b2"]},
            ]

        h = ListSectionHost(pom=mock_pom)
        h.prompt_add_section = Mock()
        h.prompt_add_subsection = Mock()
        h._process_prompt_sections()

        assert h.prompt_add_section.call_count == 2
        h.prompt_add_section.assert_any_call(
            "Section A", body="Body A", bullets=None, numbered=False, numbered_bullets=False,
        )
        h.prompt_add_section.assert_any_call(
            "Section B", body="", bullets=["b1", "b2"], numbered=False, numbered_bullets=False,
        )

    def test_list_sections_without_pom_does_nothing(self):
        """List-based PROMPT_SECTIONS skipped when pom is None."""

        class ListNoPomHost(MockPromptHost):
            PROMPT_SECTIONS = [
                {"title": "A", "body": "B"},
            ]

        h = ListNoPomHost(pom=None)
        h.prompt_add_section = Mock()
        h._process_prompt_sections()
        h.prompt_add_section.assert_not_called()

    def test_list_section_without_title_skipped(self):
        """List entries without 'title' are silently skipped."""
        mock_pom = Mock()

        class NoTitleListHost(MockPromptHost):
            PROMPT_SECTIONS = [
                {"body": "No title"},
            ]

        h = NoTitleListHost(pom=mock_pom)
        h.prompt_add_section = Mock()
        h._process_prompt_sections()
        h.prompt_add_section.assert_not_called()

    def test_list_section_empty_body_and_no_bullets_skipped(self):
        """List section with empty body and no bullets (and no subsections) is skipped."""
        mock_pom = Mock()

        class EmptyListSectionHost(MockPromptHost):
            PROMPT_SECTIONS = [
                {"title": "Empty", "body": "", "bullets": []},
            ]

        h = EmptyListSectionHost(pom=mock_pom)
        h.prompt_add_section = Mock()
        h._process_prompt_sections()
        h.prompt_add_section.assert_not_called()

    def test_list_section_with_subsections(self):
        """List-based section with subsections."""
        mock_pom = Mock()

        class ListSubHost(MockPromptHost):
            PROMPT_SECTIONS = [
                {
                    "title": "Main",
                    "body": "Main body",
                    "subsections": [
                        {"title": "Sub1", "body": "sub1 body"},
                        {"title": "Sub2", "bullets": ["x"]},
                    ],
                },
            ]

        h = ListSubHost(pom=mock_pom)
        h.prompt_add_section = Mock()
        h.prompt_add_subsection = Mock()
        h._process_prompt_sections()

        h.prompt_add_section.assert_called_once()
        assert h.prompt_add_subsection.call_count == 2

    def test_list_section_subsection_without_title_skipped(self):
        """Subsections in list mode without title are skipped."""
        mock_pom = Mock()

        class ListSubNoTitleHost(MockPromptHost):
            PROMPT_SECTIONS = [
                {
                    "title": "Main",
                    "body": "body",
                    "subsections": [
                        {"body": "no title"},
                    ],
                },
            ]

        h = ListSubNoTitleHost(pom=mock_pom)
        h.prompt_add_section = Mock()
        h.prompt_add_subsection = Mock()
        h._process_prompt_sections()
        h.prompt_add_subsection.assert_not_called()

    def test_list_section_subsection_empty_content_skipped(self):
        """Subsections in list mode with empty body and bullets are skipped."""
        mock_pom = Mock()

        class ListSubEmptyHost(MockPromptHost):
            PROMPT_SECTIONS = [
                {
                    "title": "Main",
                    "body": "body",
                    "subsections": [
                        {"title": "Sub", "body": "", "bullets": []},
                    ],
                },
            ]

        h = ListSubEmptyHost(pom=mock_pom)
        h.prompt_add_section = Mock()
        h.prompt_add_subsection = Mock()
        h._process_prompt_sections()
        h.prompt_add_subsection.assert_not_called()

    def test_list_section_with_numbered_flags(self):
        """List-based sections pass numbered flags through."""
        mock_pom = Mock()

        class NumListHost(MockPromptHost):
            PROMPT_SECTIONS = [
                {
                    "title": "Steps",
                    "body": "Follow these:",
                    "numbered": True,
                    "numberedBullets": True,
                },
            ]

        h = NumListHost(pom=mock_pom)
        h.prompt_add_section = Mock()
        h.prompt_add_subsection = Mock()
        h._process_prompt_sections()
        h.prompt_add_section.assert_called_once_with(
            "Steps",
            body="Follow these:",
            bullets=None,
            numbered=True,
            numbered_bullets=True,
        )

    # ----- Multiple sections in dict -----

    def test_dict_multiple_sections(self):
        """Multiple sections in a dict are all processed."""

        class MultiHost(MockPromptHost):
            PROMPT_SECTIONS = {
                "A": "alpha",
                "B": ["b1"],
                "C": {"body": "gamma"},
            }

        h = MultiHost()
        h.prompt_add_section = Mock()
        h.prompt_add_subsection = Mock()
        h._process_prompt_sections()
        assert h.prompt_add_section.call_count == 3


# ===========================================================================
# Tests for method chaining
# ===========================================================================

class TestMethodChaining:
    """Verify that mixin methods support fluent chaining."""

    def test_chain_set_prompt_text_and_post_prompt(self, host):
        result = host.set_prompt_text("main").set_post_prompt("post")
        assert result is host

    def test_chain_add_sections(self, host):
        result = (
            host
            .prompt_add_section("A", body="a")
            .prompt_add_section("B", body="b")
            .prompt_add_to_section("A", bullet="extra")
            .prompt_add_subsection("A", "A1", body="a1")
        )
        assert result is host

    def test_chain_set_prompt_pom(self, host):
        result = host.set_prompt_pom([]).set_post_prompt("post")
        assert result is host

    def test_chain_define_contexts_with_arg(self, host):
        result = host.define_contexts(contexts={"c": {}}).set_post_prompt("post")
        assert result is host


# ===========================================================================
# Edge case / robustness tests
# ===========================================================================

class TestEdgeCases:
    """Miscellaneous edge-case tests for PromptMixin."""

    def test_get_prompt_default_with_special_chars_in_name(self, host):
        host._prompt_manager.get_prompt.return_value = None
        host._use_pom = False
        host.name = "Agent <O'Brien> & Friends"
        expected = "You are Agent <O'Brien> & Friends, a helpful AI assistant."
        assert host.get_prompt() == expected

    def test_prompt_add_section_title_only(self, host):
        """Adding a section with only a title is valid."""
        result = host.prompt_add_section("Title Only")
        host._prompt_manager.prompt_add_section.assert_called_once_with(
            title="Title Only",
            body="",
            bullets=None,
            numbered=False,
            numbered_bullets=False,
            subsections=None,
        )
        assert result is host

    def test_prompt_add_to_section_no_content(self, host):
        """Calling prompt_add_to_section with no content args still delegates."""
        result = host.prompt_add_to_section("Title")
        host._prompt_manager.prompt_add_to_section.assert_called_once_with(
            title="Title",
            body=None,
            bullet=None,
            bullets=None,
        )
        assert result is host

    def test_prompt_add_subsection_empty_body_and_bullets(self, host):
        """Subsection with default empty body and no bullets."""
        result = host.prompt_add_subsection("P", "C")
        host._prompt_manager.prompt_add_subsection.assert_called_once_with(
            parent_title="P",
            title="C",
            body="",
            bullets=None,
        )
        assert result is host

    def test_prompt_manager_raises_on_set_prompt_text(self, host):
        """If the prompt manager raises, the exception propagates."""
        host._prompt_manager.set_prompt_text.side_effect = ValueError("conflict")
        with pytest.raises(ValueError, match="conflict"):
            host.set_prompt_text("oops")

    def test_prompt_manager_raises_on_set_prompt_pom(self, host):
        host._prompt_manager.set_prompt_pom.side_effect = ValueError("use_pom must be True")
        with pytest.raises(ValueError, match="use_pom must be True"):
            host.set_prompt_pom([{"title": "T"}])

    def test_get_prompt_pom_no_usable_method_no_sections_attr(self, host):
        """POM object with no known method and no _sections in __dict__ returns default."""
        host._prompt_manager.get_prompt.return_value = None

        class MinimalPom:
            pass

        host.pom = MinimalPom()
        host._use_pom = True
        host.name = "MinBot"

        result = host.get_prompt()
        assert result == "You are MinBot, a helpful AI assistant."

    @patch("signalwire_agents.core.mixins.prompt_mixin.ContextBuilder")
    def test_define_contexts_without_arg_sets_contexts_defined(self, MockCB, host):
        host._contexts_builder = None
        host._contexts_defined = False
        host.define_contexts()
        assert host._contexts_defined is True

    @patch("signalwire_agents.core.mixins.prompt_mixin.ContextBuilder")
    def test_define_contexts_without_arg_does_not_reset_flag(self, MockCB, host):
        """Calling define_contexts() twice does not reset _contexts_defined."""
        host._contexts_builder = None
        host._contexts_defined = False
        host.define_contexts()
        assert host._contexts_defined is True
        # Second call reuses builder
        host.define_contexts()
        assert host._contexts_defined is True

    def test_process_prompt_sections_dict_subsection_with_body_and_bullets(self):
        """Subsection that has both body and bullets is added."""

        class BothSubHost(MockPromptHost):
            PROMPT_SECTIONS = {
                "Parent": {
                    "body": "parent",
                    "subsections": [
                        {"title": "Sub", "body": "sub body", "bullets": ["sb1"]},
                    ],
                }
            }

        h = BothSubHost()
        h.prompt_add_section = Mock()
        h.prompt_add_subsection = Mock()
        h._process_prompt_sections()
        h.prompt_add_subsection.assert_called_once_with(
            "Parent", "Sub", body="sub body", bullets=["sb1"],
        )

    def test_process_prompt_sections_list_subsection_with_body_and_bullets(self):
        """List-mode subsection that has both body and bullets is added."""
        mock_pom = Mock()

        class ListBothSubHost(MockPromptHost):
            PROMPT_SECTIONS = [
                {
                    "title": "P",
                    "body": "body",
                    "subsections": [
                        {"title": "S", "body": "sb", "bullets": ["x"]},
                    ],
                },
            ]

        h = ListBothSubHost(pom=mock_pom)
        h.prompt_add_section = Mock()
        h.prompt_add_subsection = Mock()
        h._process_prompt_sections()
        h.prompt_add_subsection.assert_called_once_with(
            "P", "S", body="sb", bullets=["x"],
        )

    def test_process_prompt_sections_dict_with_subsections_key_but_empty_body(self):
        """Section dict has 'subsections' key so it is created even without body."""

        class SubOnlyHost(MockPromptHost):
            PROMPT_SECTIONS = {
                "Wrapper": {
                    "subsections": [
                        {"title": "Inner", "body": "inner body"},
                    ],
                }
            }

        h = SubOnlyHost()
        h.prompt_add_section = Mock()
        h.prompt_add_subsection = Mock()
        h._process_prompt_sections()
        # Section should be created because 'subsections' key is present
        h.prompt_add_section.assert_called_once()
        h.prompt_add_subsection.assert_called_once()

    def test_process_prompt_sections_list_with_subsections_key_but_empty_body(self):
        """List mode: section has 'subsections' key so it is created even without body."""
        mock_pom = Mock()

        class ListSubOnlyHost(MockPromptHost):
            PROMPT_SECTIONS = [
                {
                    "title": "Wrapper",
                    "subsections": [
                        {"title": "Inner", "body": "inner body"},
                    ],
                },
            ]

        h = ListSubOnlyHost(pom=mock_pom)
        h.prompt_add_section = Mock()
        h.prompt_add_subsection = Mock()
        h._process_prompt_sections()
        h.prompt_add_section.assert_called_once()
        h.prompt_add_subsection.assert_called_once()
