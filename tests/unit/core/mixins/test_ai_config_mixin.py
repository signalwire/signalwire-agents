"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for AIConfigMixin
"""

import pytest
from unittest.mock import Mock

from signalwire_agents.core.mixins.ai_config_mixin import AIConfigMixin


class MockAIConfigHost(AIConfigMixin):
    """
    A minimal host class that inherits from AIConfigMixin and provides
    all the attributes the mixin expects to find on self.
    """

    def __init__(self):
        self._hints = []
        self._languages = []
        self._pronounce = []
        self._params = {}
        self._global_data = {}
        self.native_functions = []
        self._internal_fillers = {}
        self._function_includes = []
        self._prompt_llm_params = {}
        self._post_prompt_llm_params = {}
        self.log = Mock()


@pytest.fixture
def host():
    """Return a fresh MockAIConfigHost."""
    return MockAIConfigHost()


# ===========================================================================
# Tests for add_pattern_hint (lines 65-72)
# ===========================================================================

class TestAddPatternHint:
    """Tests for AIConfigMixin.add_pattern_hint"""

    def test_adds_pattern_hint_with_all_fields(self, host):
        result = host.add_pattern_hint("SignalWire", r"signal\s*wire", "SignalWire")
        assert len(host._hints) == 1
        assert host._hints[0] == {
            "hint": "SignalWire",
            "pattern": r"signal\s*wire",
            "replace": "SignalWire",
            "ignore_case": False,
        }

    def test_adds_pattern_hint_with_ignore_case(self, host):
        host.add_pattern_hint("Test", r"t.st", "Test!", ignore_case=True)
        assert host._hints[0]["ignore_case"] is True

    def test_returns_self_for_chaining(self, host):
        result = host.add_pattern_hint("h", "p", "r")
        assert result is host

    def test_empty_hint_does_not_add(self, host):
        host.add_pattern_hint("", "pattern", "replace")
        assert len(host._hints) == 0

    def test_empty_pattern_does_not_add(self, host):
        host.add_pattern_hint("hint", "", "replace")
        assert len(host._hints) == 0

    def test_empty_replace_does_not_add(self, host):
        host.add_pattern_hint("hint", "pattern", "")
        assert len(host._hints) == 0

    def test_none_hint_does_not_add(self, host):
        host.add_pattern_hint(None, "pattern", "replace")
        assert len(host._hints) == 0

    def test_multiple_pattern_hints(self, host):
        host.add_pattern_hint("A", "a", "A!")
        host.add_pattern_hint("B", "b", "B!")
        assert len(host._hints) == 2


# ===========================================================================
# Tests for add_language (lines 116-120, 123-132, 139-140, 143-144)
# ===========================================================================

class TestAddLanguage:
    """Tests for AIConfigMixin.add_language"""

    def test_simple_voice_string(self, host):
        host.add_language("English", "en-US", "en-US-Neural2-F")
        assert len(host._languages) == 1
        lang = host._languages[0]
        assert lang["name"] == "English"
        assert lang["code"] == "en-US"
        assert lang["voice"] == "en-US-Neural2-F"
        assert "engine" not in lang
        assert "model" not in lang

    def test_explicit_engine_param(self, host):
        host.add_language("English", "en-US", "josh", engine="elevenlabs")
        lang = host._languages[0]
        assert lang["voice"] == "josh"
        assert lang["engine"] == "elevenlabs"
        assert "model" not in lang

    def test_explicit_model_param(self, host):
        host.add_language("English", "en-US", "josh", model="eleven_turbo_v2_5")
        lang = host._languages[0]
        assert lang["voice"] == "josh"
        assert lang["model"] == "eleven_turbo_v2_5"
        assert "engine" not in lang

    def test_explicit_engine_and_model_params(self, host):
        host.add_language("English", "en-US", "josh", engine="elevenlabs", model="eleven_turbo_v2_5")
        lang = host._languages[0]
        assert lang["voice"] == "josh"
        assert lang["engine"] == "elevenlabs"
        assert lang["model"] == "eleven_turbo_v2_5"

    def test_combined_format_engine_voice_model(self, host):
        host.add_language("English", "en-US", "elevenlabs.josh:eleven_turbo_v2_5")
        lang = host._languages[0]
        assert lang["voice"] == "josh"
        assert lang["engine"] == "elevenlabs"
        assert lang["model"] == "eleven_turbo_v2_5"

    def test_combined_format_parse_failure_fallback(self, host):
        """Malformed combined format (dot but no colon) uses voice as-is."""
        host.add_language("English", "en-US", "some-voice-no-colon")
        lang = host._languages[0]
        assert lang["voice"] == "some-voice-no-colon"
        assert "engine" not in lang

    def test_combined_format_with_dot_but_missing_colon(self, host):
        """Voice with a dot but no colon is treated as a simple voice string."""
        host.add_language("English", "en-US", "engine.voice")
        lang = host._languages[0]
        # No colon means no combined format, simple voice
        assert lang["voice"] == "engine.voice"

    def test_combined_format_value_error_fallback(self, host):
        """When split produces wrong number of parts, fall back to voice as-is."""
        # This has colon and dot but the split on ":" gives engine_voice with no "."
        # Actually "a:b.c" -> split(":", 1) -> ("a", "b.c"), then "a".split(".", 1) -> ValueError
        # Let me think: ".:something" -> split(":", 1) -> (".", "something"), ".".split(".", 1) -> ("", "")
        # We need a case where the split actually fails. Let's try ":voice" (no dot in prefix)
        host.add_language("English", "en-US", "nodot:model")
        lang = host._languages[0]
        # "nodot:model" -> split(":", 1) -> ("nodot", "model"), then "nodot".split(".", 1) -> ValueError
        assert lang["voice"] == "nodot:model"

    def test_both_speech_fillers_and_function_fillers(self, host):
        speech = ["um", "uh"]
        func = ["let me check", "one moment"]
        host.add_language("English", "en-US", "voice1",
                          speech_fillers=speech, function_fillers=func)
        lang = host._languages[0]
        assert lang["speech_fillers"] == speech
        assert lang["function_fillers"] == func
        assert "fillers" not in lang

    def test_only_speech_fillers_uses_deprecated_field(self, host):
        speech = ["um", "uh"]
        host.add_language("English", "en-US", "voice1", speech_fillers=speech)
        lang = host._languages[0]
        assert lang["fillers"] == speech
        assert "speech_fillers" not in lang
        assert "function_fillers" not in lang

    def test_only_function_fillers_uses_deprecated_field(self, host):
        func = ["let me check"]
        host.add_language("English", "en-US", "voice1", function_fillers=func)
        lang = host._languages[0]
        assert lang["fillers"] == func
        assert "speech_fillers" not in lang
        assert "function_fillers" not in lang

    def test_returns_self_for_chaining(self, host):
        result = host.add_language("English", "en-US", "voice1")
        assert result is host

    def test_no_fillers_produces_no_filler_keys(self, host):
        host.add_language("English", "en-US", "voice1")
        lang = host._languages[0]
        assert "fillers" not in lang
        assert "speech_fillers" not in lang
        assert "function_fillers" not in lang

    def test_combined_format_colon_and_dot_both_present_but_no_dot_in_engine_part(self, host):
        """Voice like '.name:model' where engine part is empty string after split."""
        host.add_language("English", "en-US", ".voice:model")
        lang = host._languages[0]
        # Split works: engine="", voice_part="voice", model="model"
        # Empty string is falsy but the split still succeeds
        assert lang["voice"] == "voice"
        assert lang["engine"] == ""
        assert lang["model"] == "model"


# ===========================================================================
# Tests for set_languages (lines 159-161)
# ===========================================================================

class TestSetLanguages:
    """Tests for AIConfigMixin.set_languages"""

    def test_sets_languages_with_valid_list(self, host):
        langs = [{"name": "English", "code": "en-US", "voice": "voice1"}]
        result = host.set_languages(langs)
        assert host._languages is langs

    def test_returns_self_for_chaining(self, host):
        result = host.set_languages([{"name": "French", "code": "fr-FR", "voice": "v"}])
        assert result is host

    def test_empty_list_does_not_set(self, host):
        host._languages = [{"name": "existing"}]
        host.set_languages([])
        assert host._languages == [{"name": "existing"}]

    def test_none_does_not_set(self, host):
        host._languages = [{"name": "existing"}]
        host.set_languages(None)
        assert host._languages == [{"name": "existing"}]

    def test_non_list_does_not_set(self, host):
        host._languages = [{"name": "existing"}]
        host.set_languages("not a list")
        assert host._languages == [{"name": "existing"}]


# ===========================================================================
# Tests for add_pronunciation with ignore_case (line 184)
# ===========================================================================

class TestAddPronunciation:
    """Tests for AIConfigMixin.add_pronunciation"""

    def test_ignore_case_true_adds_key(self, host):
        host.add_pronunciation("SQL", "sequel", ignore_case=True)
        assert len(host._pronounce) == 1
        rule = host._pronounce[0]
        assert rule["replace"] == "SQL"
        assert rule["with"] == "sequel"
        assert rule["ignore_case"] is True

    def test_ignore_case_false_omits_key(self, host):
        host.add_pronunciation("API", "A.P.I.")
        rule = host._pronounce[0]
        assert "ignore_case" not in rule

    def test_returns_self_for_chaining(self, host):
        result = host.add_pronunciation("A", "B")
        assert result is host


# ===========================================================================
# Tests for set_pronunciations (lines 199-201)
# ===========================================================================

class TestSetPronunciations:
    """Tests for AIConfigMixin.set_pronunciations"""

    def test_sets_pronunciations_with_valid_list(self, host):
        rules = [{"replace": "SQL", "with": "sequel"}]
        result = host.set_pronunciations(rules)
        assert host._pronounce is rules

    def test_returns_self_for_chaining(self, host):
        result = host.set_pronunciations([{"replace": "A", "with": "B"}])
        assert result is host

    def test_empty_list_does_not_set(self, host):
        host._pronounce = [{"replace": "old"}]
        host.set_pronunciations([])
        assert host._pronounce == [{"replace": "old"}]

    def test_none_does_not_set(self, host):
        host._pronounce = [{"replace": "old"}]
        host.set_pronunciations(None)
        assert host._pronounce == [{"replace": "old"}]

    def test_non_list_does_not_set(self, host):
        host._pronounce = [{"replace": "old"}]
        host.set_pronunciations({"not": "a list"})
        assert host._pronounce == [{"replace": "old"}]


# ===========================================================================
# Tests for set_global_data (lines 242-244)
# ===========================================================================

class TestSetGlobalData:
    """Tests for AIConfigMixin.set_global_data"""

    def test_sets_global_data_with_valid_dict(self, host):
        data = {"key": "value", "num": 42}
        result = host.set_global_data(data)
        assert host._global_data == data

    def test_returns_self_for_chaining(self, host):
        result = host.set_global_data({"k": "v"})
        assert result is host

    def test_empty_dict_does_not_set(self, host):
        host._global_data = {"old": "data"}
        host.set_global_data({})
        assert host._global_data == {"old": "data"}

    def test_none_does_not_set(self, host):
        host._global_data = {"old": "data"}
        host.set_global_data(None)
        assert host._global_data == {"old": "data"}


# ===========================================================================
# Tests for update_global_data (lines 256-258)
# ===========================================================================

class TestUpdateGlobalData:
    """Tests for AIConfigMixin.update_global_data"""

    def test_updates_global_data(self, host):
        host._global_data = {"existing": "value"}
        result = host.update_global_data({"new": "data"})
        assert host._global_data == {"existing": "value", "new": "data"}

    def test_returns_self_for_chaining(self, host):
        result = host.update_global_data({"k": "v"})
        assert result is host

    def test_empty_dict_does_not_update(self, host):
        host._global_data = {"old": "data"}
        host.update_global_data({})
        assert host._global_data == {"old": "data"}

    def test_none_does_not_update(self, host):
        host._global_data = {"old": "data"}
        host.update_global_data(None)
        assert host._global_data == {"old": "data"}


# ===========================================================================
# Tests for set_native_functions (lines 270-272)
# ===========================================================================

class TestSetNativeFunctions:
    """Tests for AIConfigMixin.set_native_functions"""

    def test_sets_native_functions(self, host):
        result = host.set_native_functions(["check_time", "wait_for_user"])
        assert host.native_functions == ["check_time", "wait_for_user"]

    def test_returns_self_for_chaining(self, host):
        result = host.set_native_functions(["fn"])
        assert result is host

    def test_filters_non_string_entries(self, host):
        host.set_native_functions(["valid", 123, None, "also_valid"])
        assert host.native_functions == ["valid", "also_valid"]

    def test_empty_list_does_not_set(self, host):
        host.native_functions = ["old"]
        host.set_native_functions([])
        assert host.native_functions == ["old"]

    def test_none_does_not_set(self, host):
        host.native_functions = ["old"]
        host.set_native_functions(None)
        assert host.native_functions == ["old"]


# ===========================================================================
# Tests for set_internal_fillers (lines 300-304)
# ===========================================================================

class TestSetInternalFillers:
    """Tests for AIConfigMixin.set_internal_fillers"""

    def test_sets_internal_fillers_with_valid_dict(self, host):
        fillers = {
            "next_step": {"en-US": ["Moving on...", "Let's continue..."]}
        }
        result = host.set_internal_fillers(fillers)
        assert host._internal_fillers == fillers

    def test_returns_self_for_chaining(self, host):
        result = host.set_internal_fillers({"fn": {"en": ["filler"]}})
        assert result is host

    def test_creates_internal_fillers_attr_if_missing(self, host):
        del host._internal_fillers
        host.set_internal_fillers({"fn": {"en": ["filler"]}})
        assert host._internal_fillers == {"fn": {"en": ["filler"]}}

    def test_updates_existing_fillers(self, host):
        host._internal_fillers = {"fn1": {"en": ["old"]}}
        host.set_internal_fillers({"fn2": {"en": ["new"]}})
        assert "fn1" in host._internal_fillers
        assert "fn2" in host._internal_fillers

    def test_empty_dict_does_not_set(self, host):
        host._internal_fillers = {"existing": {"en": ["x"]}}
        host.set_internal_fillers({})
        assert host._internal_fillers == {"existing": {"en": ["x"]}}

    def test_none_does_not_set(self, host):
        host._internal_fillers = {"existing": {"en": ["x"]}}
        host.set_internal_fillers(None)
        assert host._internal_fillers == {"existing": {"en": ["x"]}}

    def test_non_dict_does_not_set(self, host):
        host._internal_fillers = {"existing": {"en": ["x"]}}
        host.set_internal_fillers(["not", "a", "dict"])
        assert host._internal_fillers == {"existing": {"en": ["x"]}}


# ===========================================================================
# Tests for add_internal_filler (lines 321-329)
# ===========================================================================

class TestAddInternalFiller:
    """Tests for AIConfigMixin.add_internal_filler"""

    def test_adds_filler_for_new_function(self, host):
        result = host.add_internal_filler("next_step", "en-US", ["Moving on..."])
        assert host._internal_fillers["next_step"]["en-US"] == ["Moving on..."]

    def test_returns_self_for_chaining(self, host):
        result = host.add_internal_filler("fn", "en", ["filler"])
        assert result is host

    def test_creates_internal_fillers_attr_if_missing(self, host):
        del host._internal_fillers
        host.add_internal_filler("fn", "en", ["filler"])
        assert host._internal_fillers == {"fn": {"en": ["filler"]}}

    def test_adds_language_to_existing_function(self, host):
        host._internal_fillers = {"fn": {"en": ["english filler"]}}
        host.add_internal_filler("fn", "es", ["filler espanol"])
        assert host._internal_fillers["fn"]["en"] == ["english filler"]
        assert host._internal_fillers["fn"]["es"] == ["filler espanol"]

    def test_empty_function_name_does_not_add(self, host):
        host.add_internal_filler("", "en", ["filler"])
        assert host._internal_fillers == {}

    def test_empty_language_code_does_not_add(self, host):
        host.add_internal_filler("fn", "", ["filler"])
        assert host._internal_fillers == {}

    def test_empty_fillers_list_does_not_add(self, host):
        host.add_internal_filler("fn", "en", [])
        assert host._internal_fillers == {}

    def test_none_fillers_does_not_add(self, host):
        host.add_internal_filler("fn", "en", None)
        assert host._internal_fillers == {}


# ===========================================================================
# Tests for add_function_include with meta_data (line 349)
# ===========================================================================

class TestAddFunctionInclude:
    """Tests for AIConfigMixin.add_function_include"""

    def test_adds_include_with_meta_data(self, host):
        host.add_function_include(
            "https://example.com/swaig",
            ["func1", "func2"],
            meta_data={"auth_token": "abc123"}
        )
        assert len(host._function_includes) == 1
        inc = host._function_includes[0]
        assert inc["url"] == "https://example.com/swaig"
        assert inc["functions"] == ["func1", "func2"]
        assert inc["meta_data"] == {"auth_token": "abc123"}

    def test_adds_include_without_meta_data(self, host):
        host.add_function_include("https://example.com", ["fn1"])
        inc = host._function_includes[0]
        assert "meta_data" not in inc

    def test_meta_data_non_dict_not_added(self, host):
        host.add_function_include("https://example.com", ["fn1"], meta_data="not_dict")
        inc = host._function_includes[0]
        assert "meta_data" not in inc

    def test_returns_self_for_chaining(self, host):
        result = host.add_function_include("https://example.com", ["fn"])
        assert result is host


# ===========================================================================
# Tests for set_function_includes (lines 364-373)
# ===========================================================================

class TestSetFunctionIncludes:
    """Tests for AIConfigMixin.set_function_includes"""

    def test_sets_valid_includes(self, host):
        includes = [
            {"url": "https://example.com", "functions": ["fn1", "fn2"]},
            {"url": "https://other.com", "functions": ["fn3"]},
        ]
        result = host.set_function_includes(includes)
        assert len(host._function_includes) == 2
        assert host._function_includes[0]["url"] == "https://example.com"

    def test_returns_self_for_chaining(self, host):
        result = host.set_function_includes([{"url": "u", "functions": ["f"]}])
        assert result is host

    def test_filters_out_invalid_includes_missing_url(self, host):
        includes = [
            {"functions": ["fn1"]},  # missing url
            {"url": "https://valid.com", "functions": ["fn2"]},
        ]
        host.set_function_includes(includes)
        assert len(host._function_includes) == 1
        assert host._function_includes[0]["url"] == "https://valid.com"

    def test_filters_out_invalid_includes_missing_functions(self, host):
        includes = [
            {"url": "https://example.com"},  # missing functions
        ]
        host.set_function_includes(includes)
        assert len(host._function_includes) == 0

    def test_filters_out_non_dict_includes(self, host):
        includes = [
            "not a dict",
            {"url": "https://valid.com", "functions": ["fn"]},
        ]
        host.set_function_includes(includes)
        assert len(host._function_includes) == 1

    def test_filters_out_includes_with_non_list_functions(self, host):
        includes = [
            {"url": "https://example.com", "functions": "not_a_list"},
        ]
        host.set_function_includes(includes)
        assert len(host._function_includes) == 0

    def test_empty_list_does_not_set(self, host):
        host._function_includes = [{"url": "old", "functions": ["f"]}]
        host.set_function_includes([])
        assert host._function_includes == [{"url": "old", "functions": ["f"]}]

    def test_none_does_not_set(self, host):
        host._function_includes = [{"url": "old", "functions": ["f"]}]
        host.set_function_includes(None)
        assert host._function_includes == [{"url": "old", "functions": ["f"]}]


# ===========================================================================
# Tests for set_prompt_llm_params (lines 405-408)
# ===========================================================================

class TestSetPromptLlmParams:
    """Tests for AIConfigMixin.set_prompt_llm_params"""

    def test_sets_params(self, host):
        result = host.set_prompt_llm_params(model="gpt-4o-mini", temperature=0.7)
        assert host._prompt_llm_params["model"] == "gpt-4o-mini"
        assert host._prompt_llm_params["temperature"] == 0.7

    def test_returns_self_for_chaining(self, host):
        result = host.set_prompt_llm_params(model="gpt-4o")
        assert result is host

    def test_no_params_does_not_modify(self, host):
        host._prompt_llm_params = {"existing": "value"}
        host.set_prompt_llm_params()
        assert host._prompt_llm_params == {"existing": "value"}

    def test_updates_existing_params(self, host):
        host.set_prompt_llm_params(model="old")
        host.set_prompt_llm_params(model="new", temperature=0.5)
        assert host._prompt_llm_params["model"] == "new"
        assert host._prompt_llm_params["temperature"] == 0.5

    def test_arbitrary_params_accepted(self, host):
        host.set_prompt_llm_params(
            barge_confidence=0.6,
            presence_penalty=0.3,
            frequency_penalty=0.1,
            top_p=0.9,
        )
        assert host._prompt_llm_params["barge_confidence"] == 0.6
        assert host._prompt_llm_params["top_p"] == 0.9


# ===========================================================================
# Tests for set_post_prompt_llm_params (lines 439-442)
# ===========================================================================

class TestSetPostPromptLlmParams:
    """Tests for AIConfigMixin.set_post_prompt_llm_params"""

    def test_sets_params(self, host):
        result = host.set_post_prompt_llm_params(model="gpt-4o-mini", temperature=0.5)
        assert host._post_prompt_llm_params["model"] == "gpt-4o-mini"
        assert host._post_prompt_llm_params["temperature"] == 0.5

    def test_returns_self_for_chaining(self, host):
        result = host.set_post_prompt_llm_params(model="gpt-4o")
        assert result is host

    def test_no_params_does_not_modify(self, host):
        host._post_prompt_llm_params = {"existing": "value"}
        host.set_post_prompt_llm_params()
        assert host._post_prompt_llm_params == {"existing": "value"}

    def test_updates_existing_params(self, host):
        host.set_post_prompt_llm_params(model="old")
        host.set_post_prompt_llm_params(model="new", top_p=0.8)
        assert host._post_prompt_llm_params["model"] == "new"
        assert host._post_prompt_llm_params["top_p"] == 0.8

    def test_arbitrary_params_accepted(self, host):
        host.set_post_prompt_llm_params(
            presence_penalty=0.3,
            frequency_penalty=0.1,
        )
        assert host._post_prompt_llm_params["presence_penalty"] == 0.3
        assert host._post_prompt_llm_params["frequency_penalty"] == 0.1


# ===========================================================================
# Tests for method chaining across methods
# ===========================================================================

class TestMethodChaining:
    """Verify that mixin methods support fluent chaining."""

    def test_chain_multiple_ai_config_methods(self, host):
        result = (
            host
            .add_hint("test")
            .add_hints(["a", "b"])
            .add_pattern_hint("SW", r"sw", "SignalWire")
            .add_language("English", "en-US", "voice1")
            .add_pronunciation("SQL", "sequel")
            .set_param("key", "value")
            .set_params({"k2": "v2"})
            .set_prompt_llm_params(model="gpt-4o")
            .set_post_prompt_llm_params(model="gpt-4o-mini")
        )
        assert result is host
