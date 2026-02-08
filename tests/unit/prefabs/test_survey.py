"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for SurveyAgent prefab
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_survey(
    survey_name="Test Survey",
    questions=None,
    introduction=None,
    conclusion=None,
    brand_name=None,
    max_retries=2,
    name="survey",
    route="/survey",
):
    """
    Instantiate a SurveyAgent while mocking AgentBase.__init__ so that the
    full server / schema / uvicorn machinery is never triggered.

    Returns (survey_instance, mock_super_init) so tests can inspect both.
    """
    if questions is None:
        questions = [
            {
                "id": "q1",
                "text": "How satisfied are you?",
                "type": "rating",
                "scale": 5,
                "required": True,
            }
        ]

    with patch(
        "signalwire_agents.prefabs.survey.AgentBase.__init__", return_value=None
    ) as mock_init:
        # Mock the methods that _setup_survey_agent calls on 'self'
        with patch.multiple(
            "signalwire_agents.prefabs.survey.AgentBase",
            prompt_add_section=MagicMock(),
            set_post_prompt=MagicMock(),
            add_hints=MagicMock(),
            set_params=MagicMock(),
            set_global_data=MagicMock(),
            set_native_functions=MagicMock(),
        ):
            from signalwire_agents.prefabs.survey import SurveyAgent

            survey = SurveyAgent(
                survey_name=survey_name,
                questions=questions,
                introduction=introduction,
                conclusion=conclusion,
                brand_name=brand_name,
                max_retries=max_retries,
                name=name,
                route=route,
            )
    return survey, mock_init


def _bare_survey(questions=None):
    """
    Create a SurveyAgent via __new__ (skipping __init__ entirely) and set the
    minimum attributes required for method-level tests such as validate_response
    and log_response.
    """
    from signalwire_agents.prefabs.survey import SurveyAgent

    survey = SurveyAgent.__new__(SurveyAgent)
    survey.questions = questions or [
        {
            "id": "q1",
            "text": "How satisfied are you?",
            "type": "rating",
            "scale": 5,
            "required": True,
        },
        {
            "id": "q2",
            "text": "Would you recommend us?",
            "type": "yes_no",
            "required": True,
        },
        {
            "id": "q3",
            "text": "Which feature do you like most?",
            "type": "multiple_choice",
            "options": ["Speed", "Reliability", "Support"],
            "required": True,
        },
        {
            "id": "q4",
            "text": "Any additional comments?",
            "type": "open_ended",
            "required": False,
        },
        {
            "id": "q5",
            "text": "Anything else?",
            "type": "open_ended",
            "required": True,
        },
    ]
    survey.survey_name = "Test Survey"
    survey.brand_name = "TestCo"
    survey.max_retries = 2
    return survey


# ===========================================================================
# Test classes
# ===========================================================================


class TestSurveyInitialization:
    """Test SurveyAgent construction and default values."""

    def test_basic_initialization(self):
        """SurveyAgent stores survey_name, questions, and defaults correctly."""
        questions = [
            {"id": "q1", "text": "Rate us?", "type": "rating", "scale": 5, "required": True}
        ]
        survey, mock_init = _make_survey(
            survey_name="My Survey",
            questions=questions,
            brand_name="Acme",
        )

        assert survey.survey_name == "My Survey"
        assert survey.brand_name == "Acme"
        assert survey.max_retries == 2
        assert len(survey.questions) == 1
        assert survey.questions[0]["id"] == "q1"

    def test_super_init_called_with_correct_args(self):
        """AgentBase.__init__ is invoked with the expected keyword arguments."""
        survey, mock_init = _make_survey(
            name="custom_name",
            route="/custom",
        )
        mock_init.assert_called_once_with(
            name="custom_name",
            route="/custom",
            use_pom=True,
        )

    def test_default_brand_name(self):
        """When brand_name is None, it defaults to 'Our Company'."""
        survey, _ = _make_survey(brand_name=None)
        assert survey.brand_name == "Our Company"

    def test_custom_brand_name(self):
        """A custom brand_name is preserved."""
        survey, _ = _make_survey(brand_name="WidgetCorp")
        assert survey.brand_name == "WidgetCorp"

    def test_default_introduction(self):
        """When introduction is None a default message is generated."""
        survey, _ = _make_survey(survey_name="Feedback Survey", introduction=None)
        assert "Feedback Survey" in survey.introduction
        assert "Welcome" in survey.introduction

    def test_custom_introduction(self):
        """A custom introduction overrides the default."""
        survey, _ = _make_survey(introduction="Please answer our questions.")
        assert survey.introduction == "Please answer our questions."

    def test_default_conclusion(self):
        """When conclusion is None a default thank-you message is generated."""
        survey, _ = _make_survey(conclusion=None)
        assert "Thank you" in survey.conclusion

    def test_custom_conclusion(self):
        """A custom conclusion overrides the default."""
        survey, _ = _make_survey(conclusion="Goodbye!")
        assert survey.conclusion == "Goodbye!"

    def test_max_retries_setting(self):
        """max_retries is stored correctly."""
        survey, _ = _make_survey(max_retries=5)
        assert survey.max_retries == 5

    def test_kwargs_forwarded_to_super(self):
        """Extra keyword arguments are forwarded to AgentBase.__init__."""
        questions = [
            {"id": "q1", "text": "Rate us?", "type": "rating", "scale": 5}
        ]
        with patch(
            "signalwire_agents.prefabs.survey.AgentBase.__init__", return_value=None
        ) as mock_init:
            with patch.multiple(
                "signalwire_agents.prefabs.survey.AgentBase",
                prompt_add_section=MagicMock(),
                set_post_prompt=MagicMock(),
                add_hints=MagicMock(),
                set_params=MagicMock(),
                set_global_data=MagicMock(),
                set_native_functions=MagicMock(),
            ):
                from signalwire_agents.prefabs.survey import SurveyAgent

                SurveyAgent(
                    survey_name="S",
                    questions=questions,
                    host="0.0.0.0",
                    port=9000,
                )
            mock_init.assert_called_once_with(
                name="survey",
                route="/survey",
                use_pom=True,
                host="0.0.0.0",
                port=9000,
            )


class TestQuestionValidation:
    """Tests for _validate_questions."""

    def test_missing_text_raises(self):
        """A question without 'text' raises ValueError."""
        questions = [{"id": "q1", "type": "rating"}]
        with pytest.raises(ValueError, match="missing the 'text' field"):
            _make_survey(questions=questions)

    def test_empty_text_raises(self):
        """A question with empty string text raises ValueError."""
        questions = [{"id": "q1", "text": "", "type": "rating"}]
        with pytest.raises(ValueError, match="missing the 'text' field"):
            _make_survey(questions=questions)

    def test_invalid_type_raises(self):
        """A question with an unrecognized type raises ValueError."""
        questions = [{"id": "q1", "text": "Q?", "type": "essay"}]
        with pytest.raises(ValueError, match="invalid type"):
            _make_survey(questions=questions)

    def test_missing_type_raises(self):
        """A question with no type raises ValueError."""
        questions = [{"id": "q1", "text": "Q?"}]
        with pytest.raises(ValueError, match="invalid type"):
            _make_survey(questions=questions)

    def test_multiple_choice_without_options_raises(self):
        """A multiple_choice question without options raises ValueError."""
        questions = [{"id": "q1", "text": "Choose?", "type": "multiple_choice"}]
        with pytest.raises(ValueError, match="must have options"):
            _make_survey(questions=questions)

    def test_multiple_choice_with_empty_options_raises(self):
        """A multiple_choice question with empty options list raises ValueError."""
        questions = [
            {"id": "q1", "text": "Choose?", "type": "multiple_choice", "options": []}
        ]
        with pytest.raises(ValueError, match="must have options"):
            _make_survey(questions=questions)

    def test_auto_id_generation(self):
        """When a question has no id, one is generated automatically."""
        questions = [{"text": "Rate us?", "type": "rating", "scale": 5}]
        survey, _ = _make_survey(questions=questions)
        assert survey.questions[0]["id"] == "question_1"

    def test_empty_id_gets_auto_generated(self):
        """An explicitly empty string id gets replaced."""
        questions = [{"id": "", "text": "Rate us?", "type": "rating"}]
        survey, _ = _make_survey(questions=questions)
        assert survey.questions[0]["id"] == "question_1"

    def test_required_defaults_to_true(self):
        """When 'required' is omitted it defaults to True."""
        questions = [{"id": "q1", "text": "Rate us?", "type": "rating"}]
        survey, _ = _make_survey(questions=questions)
        assert survey.questions[0]["required"] is True

    def test_rating_scale_defaults_to_five(self):
        """When a rating question omits 'scale', it defaults to 5."""
        questions = [{"id": "q1", "text": "Rate us?", "type": "rating"}]
        survey, _ = _make_survey(questions=questions)
        assert survey.questions[0]["scale"] == 5

    def test_rating_scale_preserved(self):
        """An explicit scale value is preserved."""
        questions = [{"id": "q1", "text": "Rate?", "type": "rating", "scale": 10}]
        survey, _ = _make_survey(questions=questions)
        assert survey.questions[0]["scale"] == 10

    def test_valid_multiple_choice(self):
        """A well-formed multiple_choice question passes validation."""
        questions = [
            {
                "id": "q1",
                "text": "Pick one?",
                "type": "multiple_choice",
                "options": ["A", "B", "C"],
            }
        ]
        survey, _ = _make_survey(questions=questions)
        assert survey.questions[0]["options"] == ["A", "B", "C"]

    def test_valid_yes_no(self):
        """A well-formed yes_no question passes validation."""
        questions = [{"id": "q1", "text": "Yes or no?", "type": "yes_no"}]
        survey, _ = _make_survey(questions=questions)
        assert survey.questions[0]["type"] == "yes_no"

    def test_valid_open_ended(self):
        """A well-formed open_ended question passes validation."""
        questions = [{"id": "q1", "text": "Tell us more.", "type": "open_ended"}]
        survey, _ = _make_survey(questions=questions)
        assert survey.questions[0]["type"] == "open_ended"

    def test_multiple_questions_validated(self):
        """Multiple questions are all validated in order."""
        questions = [
            {"id": "q1", "text": "Rate?", "type": "rating"},
            {"id": "q2", "text": "Yes?", "type": "yes_no"},
            {"id": "q3", "text": "Pick?", "type": "multiple_choice", "options": ["X"]},
        ]
        survey, _ = _make_survey(questions=questions)
        assert len(survey.questions) == 3

    def test_second_question_invalid_raises(self):
        """If the second question is invalid, a ValueError mentions question 2."""
        questions = [
            {"id": "q1", "text": "Good?", "type": "rating"},
            {"id": "q2", "text": "", "type": "rating"},
        ]
        with pytest.raises(ValueError, match="Question 2"):
            _make_survey(questions=questions)


class TestSetupSurveyAgent:
    """Tests for _setup_survey_agent configuration calls."""

    def test_prompt_sections_added(self):
        """_setup_survey_agent adds expected prompt sections."""
        questions = [
            {"id": "q1", "text": "Rate us?", "type": "rating", "scale": 5}
        ]

        with patch(
            "signalwire_agents.prefabs.survey.AgentBase.__init__", return_value=None
        ):
            with patch.multiple(
                "signalwire_agents.prefabs.survey.AgentBase",
                prompt_add_section=MagicMock(),
                set_post_prompt=MagicMock(),
                add_hints=MagicMock(),
                set_params=MagicMock(),
                set_global_data=MagicMock(),
                set_native_functions=MagicMock(),
            ):
                from signalwire_agents.prefabs.survey import SurveyAgent

                survey = SurveyAgent(survey_name="Test", questions=questions)

                # Collect section titles from prompt_add_section calls
                section_titles = [
                    c.args[0] if c.args else c.kwargs.get("title")
                    for c in survey.prompt_add_section.call_args_list
                ]
                assert "Personality" in section_titles
                assert "Goal" in section_titles
                assert "Instructions" in section_titles
                assert "Introduction" in section_titles
                assert "Survey Questions" in section_titles
                assert "Conclusion" in section_titles

    def test_post_prompt_set(self):
        """_setup_survey_agent calls set_post_prompt with a JSON template."""
        questions = [
            {"id": "q1", "text": "Rate us?", "type": "rating", "scale": 5}
        ]

        with patch(
            "signalwire_agents.prefabs.survey.AgentBase.__init__", return_value=None
        ):
            with patch.multiple(
                "signalwire_agents.prefabs.survey.AgentBase",
                prompt_add_section=MagicMock(),
                set_post_prompt=MagicMock(),
                add_hints=MagicMock(),
                set_params=MagicMock(),
                set_global_data=MagicMock(),
                set_native_functions=MagicMock(),
            ):
                from signalwire_agents.prefabs.survey import SurveyAgent

                survey = SurveyAgent(survey_name="Test", questions=questions)
                survey.set_post_prompt.assert_called_once()
                post_prompt_arg = survey.set_post_prompt.call_args[0][0]
                assert "survey_name" in post_prompt_arg
                assert "responses" in post_prompt_arg
                assert "completion_status" in post_prompt_arg

    def test_hints_include_survey_and_brand(self):
        """add_hints includes the survey name, brand, and type-specific terms."""
        questions = [
            {"id": "q1", "text": "Rate?", "type": "rating", "scale": 3},
            {"id": "q2", "text": "Yes?", "type": "yes_no"},
            {
                "id": "q3",
                "text": "Pick?",
                "type": "multiple_choice",
                "options": ["Alpha", "Beta"],
            },
        ]

        with patch(
            "signalwire_agents.prefabs.survey.AgentBase.__init__", return_value=None
        ):
            with patch.multiple(
                "signalwire_agents.prefabs.survey.AgentBase",
                prompt_add_section=MagicMock(),
                set_post_prompt=MagicMock(),
                add_hints=MagicMock(),
                set_params=MagicMock(),
                set_global_data=MagicMock(),
                set_native_functions=MagicMock(),
            ):
                from signalwire_agents.prefabs.survey import SurveyAgent

                survey = SurveyAgent(
                    survey_name="CX Survey",
                    questions=questions,
                    brand_name="Acme",
                )

                survey.add_hints.assert_called_once()
                hints = survey.add_hints.call_args[0][0]
                assert "CX Survey" in hints
                assert "Acme" in hints
                # Rating scale 1..3
                assert "1" in hints
                assert "2" in hints
                assert "3" in hints
                # yes_no
                assert "yes" in hints
                assert "no" in hints
                # multiple_choice options
                assert "Alpha" in hints
                assert "Beta" in hints

    def test_params_set(self):
        """set_params is called with expected AI parameters."""
        questions = [
            {"id": "q1", "text": "Rate us?", "type": "rating", "scale": 5}
        ]

        with patch(
            "signalwire_agents.prefabs.survey.AgentBase.__init__", return_value=None
        ):
            with patch.multiple(
                "signalwire_agents.prefabs.survey.AgentBase",
                prompt_add_section=MagicMock(),
                set_post_prompt=MagicMock(),
                add_hints=MagicMock(),
                set_params=MagicMock(),
                set_global_data=MagicMock(),
                set_native_functions=MagicMock(),
            ):
                from signalwire_agents.prefabs.survey import SurveyAgent

                survey = SurveyAgent(survey_name="Test", questions=questions)

                survey.set_params.assert_called_once()
                params = survey.set_params.call_args[0][0]
                assert params["wait_for_user"] is False
                assert params["end_of_speech_timeout"] == 1500
                assert params["ai_volume"] == 5
                assert params["static_greeting_no_barge"] is True
                # static_greeting equals the introduction message
                assert params["static_greeting"] == survey.introduction

    def test_global_data_set(self):
        """set_global_data is called with survey metadata."""
        questions = [
            {"id": "q1", "text": "Rate us?", "type": "rating", "scale": 5}
        ]

        with patch(
            "signalwire_agents.prefabs.survey.AgentBase.__init__", return_value=None
        ):
            with patch.multiple(
                "signalwire_agents.prefabs.survey.AgentBase",
                prompt_add_section=MagicMock(),
                set_post_prompt=MagicMock(),
                add_hints=MagicMock(),
                set_params=MagicMock(),
                set_global_data=MagicMock(),
                set_native_functions=MagicMock(),
            ):
                from signalwire_agents.prefabs.survey import SurveyAgent

                survey = SurveyAgent(
                    survey_name="GD Survey",
                    questions=questions,
                    brand_name="GD Co",
                    max_retries=3,
                )

                survey.set_global_data.assert_called_once()
                gd = survey.set_global_data.call_args[0][0]
                assert gd["survey_name"] == "GD Survey"
                assert gd["brand_name"] == "GD Co"
                assert gd["max_retries"] == 3
                assert gd["questions"] is survey.questions

    def test_native_functions_set(self):
        """set_native_functions is called with check_time."""
        questions = [
            {"id": "q1", "text": "Rate us?", "type": "rating", "scale": 5}
        ]

        with patch(
            "signalwire_agents.prefabs.survey.AgentBase.__init__", return_value=None
        ):
            with patch.multiple(
                "signalwire_agents.prefabs.survey.AgentBase",
                prompt_add_section=MagicMock(),
                set_post_prompt=MagicMock(),
                add_hints=MagicMock(),
                set_params=MagicMock(),
                set_global_data=MagicMock(),
                set_native_functions=MagicMock(),
            ):
                from signalwire_agents.prefabs.survey import SurveyAgent

                survey = SurveyAgent(survey_name="Test", questions=questions)
                survey.set_native_functions.assert_called_once_with(["check_time"])


class TestValidateResponse:
    """Tests for the validate_response tool method."""

    def test_valid_rating_response(self):
        """A numeric rating within range is valid."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q1", "response": "3"}, {}
        )
        assert "valid" in result.response.lower()

    def test_valid_rating_boundary_low(self):
        """Rating of 1 (lower boundary) is valid."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q1", "response": "1"}, {}
        )
        assert "valid" in result.response.lower()

    def test_valid_rating_boundary_high(self):
        """Rating at the upper boundary is valid."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q1", "response": "5"}, {}
        )
        assert "valid" in result.response.lower()

    def test_invalid_rating_too_high(self):
        """Rating above scale is invalid."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q1", "response": "6"}, {}
        )
        assert "invalid" in result.response.lower()

    def test_invalid_rating_too_low(self):
        """Rating of 0 is invalid."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q1", "response": "0"}, {}
        )
        assert "invalid" in result.response.lower()

    def test_invalid_rating_negative(self):
        """Negative rating is invalid."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q1", "response": "-1"}, {}
        )
        assert "invalid" in result.response.lower()

    def test_invalid_rating_non_numeric(self):
        """Non-numeric rating response is invalid."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q1", "response": "great"}, {}
        )
        assert "invalid" in result.response.lower()

    def test_valid_yes_no_yes(self):
        """'yes' is valid for a yes_no question."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q2", "response": "yes"}, {}
        )
        assert "valid" in result.response.lower()

    def test_valid_yes_no_no(self):
        """'no' is valid for a yes_no question."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q2", "response": "no"}, {}
        )
        assert "valid" in result.response.lower()

    def test_valid_yes_no_y(self):
        """'y' is valid for a yes_no question."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q2", "response": "y"}, {}
        )
        assert "valid" in result.response.lower()

    def test_valid_yes_no_n(self):
        """'n' is valid for a yes_no question."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q2", "response": "n"}, {}
        )
        assert "valid" in result.response.lower()

    def test_invalid_yes_no(self):
        """An unrecognized yes_no answer is invalid."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q2", "response": "maybe"}, {}
        )
        assert "yes" in result.response.lower() or "no" in result.response.lower()

    def test_valid_multiple_choice(self):
        """An exact option match is valid (case-insensitive)."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q3", "response": "speed"}, {}
        )
        assert "valid" in result.response.lower()

    def test_valid_multiple_choice_case_insensitive(self):
        """Multiple-choice match is case-insensitive."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q3", "response": "RELIABILITY"}, {}
        )
        assert "valid" in result.response.lower()

    def test_invalid_multiple_choice(self):
        """An option not in the list is invalid."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q3", "response": "Price"}, {}
        )
        assert "invalid" in result.response.lower()

    def test_valid_open_ended_non_required(self):
        """An empty answer to a non-required open-ended question is valid."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q4", "response": ""}, {}
        )
        # q4 is not required, so empty is fine
        assert "valid" in result.response.lower() or "recorded" in result.response.lower() or result.response != ""

    def test_invalid_open_ended_required_empty(self):
        """An empty answer to a required open-ended question is invalid."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q5", "response": ""}, {}
        )
        assert "required" in result.response.lower()

    def test_valid_open_ended_required(self):
        """A non-empty answer to a required open-ended question is valid."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q5", "response": "Looks good"}, {}
        )
        assert "valid" in result.response.lower()

    def test_unknown_question_id(self):
        """An unknown question_id returns an error message."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "nonexistent", "response": "anything"}, {}
        )
        assert "not found" in result.response.lower() or "error" in result.response.lower()

    def test_missing_question_id(self):
        """Missing question_id in args returns an error."""
        survey = _bare_survey()
        result = survey.validate_response({"response": "3"}, {})
        assert "not found" in result.response.lower() or "error" in result.response.lower()

    def test_missing_response_field(self):
        """Missing response in args defaults to empty string and validates accordingly."""
        survey = _bare_survey()
        # For a rating question, empty string should be invalid
        result = survey.validate_response({"question_id": "q1"}, {})
        assert "invalid" in result.response.lower()

    def test_returns_swaig_function_result(self):
        """validate_response returns a SwaigFunctionResult instance."""
        from signalwire_agents.core.function_result import SwaigFunctionResult

        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q1", "response": "3"}, {}
        )
        assert isinstance(result, SwaigFunctionResult)

    def test_rating_whitespace_trimmed(self):
        """Leading/trailing whitespace in a rating response is trimmed."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q1", "response": "  3  "}, {}
        )
        assert "valid" in result.response.lower()

    def test_multiple_choice_whitespace_trimmed(self):
        """Leading/trailing whitespace in a multiple-choice response is trimmed."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q3", "response": "  Speed  "}, {}
        )
        assert "valid" in result.response.lower()


class TestLogResponse:
    """Tests for the log_response tool method."""

    def test_log_known_question(self):
        """log_response acknowledges a response for a known question."""
        survey = _bare_survey()
        result = survey.log_response(
            {"question_id": "q1", "response": "5"}, {}
        )
        assert "recorded" in result.response.lower()
        assert "How satisfied" in result.response

    def test_log_unknown_question(self):
        """log_response for an unknown id still returns a result (empty text)."""
        survey = _bare_survey()
        result = survey.log_response(
            {"question_id": "unknown_q", "response": "something"}, {}
        )
        # The question_text will be empty but the response is still returned
        assert "recorded" in result.response.lower()

    def test_returns_swaig_function_result(self):
        """log_response returns a SwaigFunctionResult."""
        from signalwire_agents.core.function_result import SwaigFunctionResult

        survey = _bare_survey()
        result = survey.log_response(
            {"question_id": "q1", "response": "3"}, {}
        )
        assert isinstance(result, SwaigFunctionResult)

    def test_log_response_includes_question_text(self):
        """The acknowledgement mentions the question text."""
        survey = _bare_survey()
        result = survey.log_response(
            {"question_id": "q2", "response": "yes"}, {}
        )
        assert "Would you recommend us?" in result.response

    def test_missing_question_id(self):
        """Missing question_id still returns a result."""
        survey = _bare_survey()
        result = survey.log_response({"response": "5"}, {})
        assert "recorded" in result.response.lower()

    def test_missing_response_field(self):
        """Missing response arg defaults to empty string."""
        survey = _bare_survey()
        result = survey.log_response({"question_id": "q1"}, {})
        assert "recorded" in result.response.lower()


class TestOnSummary:
    """Tests for the on_summary callback."""

    def test_on_summary_with_dict(self, capsys):
        """on_summary prints JSON when summary is a dict."""
        survey = _bare_survey()
        summary_data = {
            "survey_name": "Test Survey",
            "responses": {"q1": "5"},
            "completion_status": "complete",
        }
        survey.on_summary(summary_data)
        captured = capsys.readouterr()
        assert "Survey completed" in captured.out
        assert "Test Survey" in captured.out

    def test_on_summary_with_string(self, capsys):
        """on_summary prints unstructured text when summary is a string."""
        survey = _bare_survey()
        survey.on_summary("Some raw summary text")
        captured = capsys.readouterr()
        assert "unstructured" in captured.out.lower()
        assert "Some raw summary text" in captured.out

    def test_on_summary_with_none(self, capsys):
        """on_summary does nothing when summary is None/falsy."""
        survey = _bare_survey()
        survey.on_summary(None)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_on_summary_with_empty_dict(self, capsys):
        """on_summary with an empty dict produces no output (empty dict is falsy)."""
        survey = _bare_survey()
        # In Python, {} is falsy, so the `if summary:` guard skips processing.
        survey.on_summary({})
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_on_summary_error_handling(self, capsys):
        """on_summary catches exceptions and prints error."""
        survey = _bare_survey()
        # Passing a dict-like that raises on json.dumps via a bad key
        bad_obj = {"key": object()}  # object() is not JSON serializable

        # Because isinstance(bad_obj, dict) is True, it will try json.dumps
        survey.on_summary(bad_obj)
        captured = capsys.readouterr()
        assert "error" in captured.out.lower()

    def test_on_summary_with_raw_data(self, capsys):
        """on_summary accepts optional raw_data argument."""
        survey = _bare_survey()
        survey.on_summary({"status": "done"}, raw_data={"call_id": "abc"})
        captured = capsys.readouterr()
        assert "Survey completed" in captured.out


class TestSurveyQuestionTypes:
    """Integration-style tests ensuring different question type combinations work."""

    def test_all_question_types_together(self):
        """A survey with all four question types initializes without error."""
        questions = [
            {"id": "r1", "text": "Rate 1-5?", "type": "rating", "scale": 5},
            {"id": "mc1", "text": "Pick one?", "type": "multiple_choice", "options": ["A", "B"]},
            {"id": "yn1", "text": "Yes or no?", "type": "yes_no"},
            {"id": "oe1", "text": "Comments?", "type": "open_ended"},
        ]
        survey, _ = _make_survey(questions=questions)
        assert len(survey.questions) == 4

    def test_many_questions(self):
        """A survey with many questions initializes correctly."""
        questions = [
            {"id": f"q{i}", "text": f"Question {i}?", "type": "open_ended"}
            for i in range(50)
        ]
        survey, _ = _make_survey(questions=questions)
        assert len(survey.questions) == 50

    def test_rating_custom_scale(self):
        """A rating question with a custom scale stores it properly."""
        questions = [
            {"id": "q1", "text": "Rate 1-10?", "type": "rating", "scale": 10}
        ]
        survey, _ = _make_survey(questions=questions)
        assert survey.questions[0]["scale"] == 10


class TestValidateResponseEdgeCases:
    """Edge cases for validate_response."""

    def test_rating_with_float_string(self):
        """A float string like '3.5' is invalid for a rating question."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q1", "response": "3.5"}, {}
        )
        assert "invalid" in result.response.lower()

    def test_empty_args(self):
        """Completely empty args dict falls through to 'not found'."""
        survey = _bare_survey()
        result = survey.validate_response({}, {})
        assert "not found" in result.response.lower() or "error" in result.response.lower()

    def test_open_ended_whitespace_only_required(self):
        """Whitespace-only answer to a required open-ended question is invalid."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q5", "response": "   "}, {}
        )
        assert "required" in result.response.lower()

    def test_yes_no_uppercase(self):
        """'YES' in uppercase is valid for a yes_no question."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q2", "response": "YES"}, {}
        )
        assert "valid" in result.response.lower()

    def test_yes_no_with_whitespace(self):
        """'  yes  ' with whitespace is valid for a yes_no question."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q2", "response": "  yes  "}, {}
        )
        assert "valid" in result.response.lower()

    def test_multiple_choice_partial_match_invalid(self):
        """A partial match like 'Spee' (not exact) is invalid for multiple_choice."""
        survey = _bare_survey()
        result = survey.validate_response(
            {"question_id": "q3", "response": "Spee"}, {}
        )
        assert "invalid" in result.response.lower()


class TestToolDecorators:
    """Test that the tool decorators are correctly applied to the class methods."""

    def test_validate_response_has_tool_metadata(self):
        """validate_response should be marked as a tool by the @tool decorator."""
        from signalwire_agents.prefabs.survey import SurveyAgent

        # The class-level @AgentBase.tool decorator sets _is_tool, _tool_name, _tool_params
        assert hasattr(SurveyAgent.validate_response, "_is_tool")
        assert SurveyAgent.validate_response._is_tool is True
        assert SurveyAgent.validate_response._tool_name == "validate_response"
        # The decorator kwargs are stored in _tool_params
        tool_params = SurveyAgent.validate_response._tool_params
        assert "parameters" in tool_params
        assert "question_id" in tool_params["parameters"]
        assert "response" in tool_params["parameters"]

    def test_log_response_has_tool_metadata(self):
        """log_response should be marked as a tool by the @tool decorator."""
        from signalwire_agents.prefabs.survey import SurveyAgent

        assert hasattr(SurveyAgent.log_response, "_is_tool")
        assert SurveyAgent.log_response._is_tool is True
        assert SurveyAgent.log_response._tool_name == "log_response"
        tool_params = SurveyAgent.log_response._tool_params
        assert "parameters" in tool_params
        assert "question_id" in tool_params["parameters"]
        assert "response" in tool_params["parameters"]


class TestSurveySetupDetails:
    """Detailed tests for how _setup_survey_agent constructs prompt sections."""

    def test_personality_section_includes_brand(self):
        """The Personality prompt section includes the brand name."""
        with patch(
            "signalwire_agents.prefabs.survey.AgentBase.__init__", return_value=None
        ):
            with patch.multiple(
                "signalwire_agents.prefabs.survey.AgentBase",
                prompt_add_section=MagicMock(),
                set_post_prompt=MagicMock(),
                add_hints=MagicMock(),
                set_params=MagicMock(),
                set_global_data=MagicMock(),
                set_native_functions=MagicMock(),
            ):
                from signalwire_agents.prefabs.survey import SurveyAgent

                survey = SurveyAgent(
                    survey_name="Test",
                    questions=[{"id": "q1", "text": "Q?", "type": "open_ended"}],
                    brand_name="MyCorp",
                )

                # Find the Personality call
                for c in survey.prompt_add_section.call_args_list:
                    if c.args and c.args[0] == "Personality":
                        body = c.kwargs.get("body", "")
                        assert "MyCorp" in body
                        break
                else:
                    pytest.fail("Personality section not found")

    def test_goal_section_includes_survey_name(self):
        """The Goal prompt section includes the survey name."""
        with patch(
            "signalwire_agents.prefabs.survey.AgentBase.__init__", return_value=None
        ):
            with patch.multiple(
                "signalwire_agents.prefabs.survey.AgentBase",
                prompt_add_section=MagicMock(),
                set_post_prompt=MagicMock(),
                add_hints=MagicMock(),
                set_params=MagicMock(),
                set_global_data=MagicMock(),
                set_native_functions=MagicMock(),
            ):
                from signalwire_agents.prefabs.survey import SurveyAgent

                survey = SurveyAgent(
                    survey_name="Satisfaction Survey",
                    questions=[{"id": "q1", "text": "Q?", "type": "open_ended"}],
                )

                for c in survey.prompt_add_section.call_args_list:
                    if c.args and c.args[0] == "Goal":
                        body = c.kwargs.get("body", "")
                        assert "Satisfaction Survey" in body
                        break
                else:
                    pytest.fail("Goal section not found")

    def test_instructions_section_has_bullets(self):
        """The Instructions prompt section contains bullet points."""
        with patch(
            "signalwire_agents.prefabs.survey.AgentBase.__init__", return_value=None
        ):
            with patch.multiple(
                "signalwire_agents.prefabs.survey.AgentBase",
                prompt_add_section=MagicMock(),
                set_post_prompt=MagicMock(),
                add_hints=MagicMock(),
                set_params=MagicMock(),
                set_global_data=MagicMock(),
                set_native_functions=MagicMock(),
            ):
                from signalwire_agents.prefabs.survey import SurveyAgent

                survey = SurveyAgent(
                    survey_name="T",
                    questions=[{"id": "q1", "text": "Q?", "type": "open_ended"}],
                    max_retries=3,
                )

                for c in survey.prompt_add_section.call_args_list:
                    if c.args and c.args[0] == "Instructions":
                        bullets = c.kwargs.get("bullets", [])
                        assert isinstance(bullets, list)
                        assert len(bullets) > 0
                        # Check that max_retries appears somewhere in the bullets
                        assert any("3" in b for b in bullets)
                        break
                else:
                    pytest.fail("Instructions section not found")

    def test_survey_questions_section_has_subsections(self):
        """The Survey Questions prompt section has subsections for each question."""
        questions = [
            {"id": "q1", "text": "Rate?", "type": "rating", "scale": 5},
            {"id": "q2", "text": "Pick?", "type": "multiple_choice", "options": ["A", "B"]},
        ]

        with patch(
            "signalwire_agents.prefabs.survey.AgentBase.__init__", return_value=None
        ):
            with patch.multiple(
                "signalwire_agents.prefabs.survey.AgentBase",
                prompt_add_section=MagicMock(),
                set_post_prompt=MagicMock(),
                add_hints=MagicMock(),
                set_params=MagicMock(),
                set_global_data=MagicMock(),
                set_native_functions=MagicMock(),
            ):
                from signalwire_agents.prefabs.survey import SurveyAgent

                survey = SurveyAgent(survey_name="T", questions=questions)

                for c in survey.prompt_add_section.call_args_list:
                    if c.args and c.args[0] == "Survey Questions":
                        subsections = c.kwargs.get("subsections", [])
                        assert len(subsections) == 2
                        # First subsection title is the question text
                        assert subsections[0]["title"] == "Rate?"
                        assert "Scale: 1-5" in subsections[0]["body"]
                        # Second subsection
                        assert subsections[1]["title"] == "Pick?"
                        assert "A, B" in subsections[1]["body"]
                        break
                else:
                    pytest.fail("Survey Questions section not found")

    def test_introduction_section_body(self):
        """The Introduction section body includes the introduction message."""
        with patch(
            "signalwire_agents.prefabs.survey.AgentBase.__init__", return_value=None
        ):
            with patch.multiple(
                "signalwire_agents.prefabs.survey.AgentBase",
                prompt_add_section=MagicMock(),
                set_post_prompt=MagicMock(),
                add_hints=MagicMock(),
                set_params=MagicMock(),
                set_global_data=MagicMock(),
                set_native_functions=MagicMock(),
            ):
                from signalwire_agents.prefabs.survey import SurveyAgent

                survey = SurveyAgent(
                    survey_name="T",
                    questions=[{"id": "q1", "text": "Q?", "type": "open_ended"}],
                    introduction="Hello and welcome!",
                )

                for c in survey.prompt_add_section.call_args_list:
                    if c.args and c.args[0] == "Introduction":
                        body = c.kwargs.get("body", "")
                        assert "Hello and welcome!" in body
                        break
                else:
                    pytest.fail("Introduction section not found")

    def test_conclusion_section_body(self):
        """The Conclusion section body includes the conclusion message."""
        with patch(
            "signalwire_agents.prefabs.survey.AgentBase.__init__", return_value=None
        ):
            with patch.multiple(
                "signalwire_agents.prefabs.survey.AgentBase",
                prompt_add_section=MagicMock(),
                set_post_prompt=MagicMock(),
                add_hints=MagicMock(),
                set_params=MagicMock(),
                set_global_data=MagicMock(),
                set_native_functions=MagicMock(),
            ):
                from signalwire_agents.prefabs.survey import SurveyAgent

                survey = SurveyAgent(
                    survey_name="T",
                    questions=[{"id": "q1", "text": "Q?", "type": "open_ended"}],
                    conclusion="Thanks for your time!",
                )

                for c in survey.prompt_add_section.call_args_list:
                    if c.args and c.args[0] == "Conclusion":
                        body = c.kwargs.get("body", "")
                        assert "Thanks for your time!" in body
                        break
                else:
                    pytest.fail("Conclusion section not found")


class TestSurveyHintsEdgeCases:
    """Edge cases for hints generation."""

    def test_hints_no_type_specific_terms(self):
        """An open_ended-only survey has no type-specific hint terms (just name/brand)."""
        questions = [{"id": "q1", "text": "Comments?", "type": "open_ended"}]
        with patch(
            "signalwire_agents.prefabs.survey.AgentBase.__init__", return_value=None
        ):
            with patch.multiple(
                "signalwire_agents.prefabs.survey.AgentBase",
                prompt_add_section=MagicMock(),
                set_post_prompt=MagicMock(),
                add_hints=MagicMock(),
                set_params=MagicMock(),
                set_global_data=MagicMock(),
                set_native_functions=MagicMock(),
            ):
                from signalwire_agents.prefabs.survey import SurveyAgent

                survey = SurveyAgent(
                    survey_name="Open Survey",
                    questions=questions,
                    brand_name="Brand",
                )
                hints = survey.add_hints.call_args[0][0]
                # Only survey_name and brand_name
                assert hints == ["Open Survey", "Brand"]

    def test_hints_large_rating_scale(self):
        """A rating question with scale=10 generates hints 1-10."""
        questions = [{"id": "q1", "text": "Rate?", "type": "rating", "scale": 10}]
        with patch(
            "signalwire_agents.prefabs.survey.AgentBase.__init__", return_value=None
        ):
            with patch.multiple(
                "signalwire_agents.prefabs.survey.AgentBase",
                prompt_add_section=MagicMock(),
                set_post_prompt=MagicMock(),
                add_hints=MagicMock(),
                set_params=MagicMock(),
                set_global_data=MagicMock(),
                set_native_functions=MagicMock(),
            ):
                from signalwire_agents.prefabs.survey import SurveyAgent

                survey = SurveyAgent(
                    survey_name="S",
                    questions=questions,
                    brand_name="B",
                )
                hints = survey.add_hints.call_args[0][0]
                # Should have S, B, then "1".."10"
                for i in range(1, 11):
                    assert str(i) in hints
