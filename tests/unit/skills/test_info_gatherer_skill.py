"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for InfoGathererSkill and SkillBase namespace helpers
"""

import pytest
from unittest.mock import Mock

from signalwire_agents.core.skill_base import SkillBase
from signalwire_agents.core.function_result import SwaigFunctionResult
from signalwire_agents.skills.info_gatherer.skill import InfoGathererSkill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_QUESTIONS = [
    {"key_name": "full_name", "question_text": "What is your full name?"},
    {"key_name": "email", "question_text": "What is your email?", "confirm": True},
    {"key_name": "reason", "question_text": "How can we help you today?"},
]


def _make_skill(params=None):
    """Create an InfoGathererSkill with a mocked agent."""
    mock_agent = Mock()
    mock_agent.define_tool = Mock()
    return InfoGathererSkill(agent=mock_agent, params=params)


def _setup_skill(params=None):
    """Create, setup and register an InfoGathererSkill."""
    skill = _make_skill(params)
    assert skill.setup() is True
    skill.register_tools()
    return skill


# ===========================================================================
# Class-Level Metadata
# ===========================================================================

class TestInfoGathererSkillMetadata:
    def test_skill_name(self):
        assert InfoGathererSkill.SKILL_NAME == "info_gatherer"

    def test_skill_description(self):
        assert InfoGathererSkill.SKILL_DESCRIPTION is not None

    def test_supports_multiple_instances(self):
        assert InfoGathererSkill.SUPPORTS_MULTIPLE_INSTANCES is True

    def test_no_required_packages(self):
        assert InfoGathererSkill.REQUIRED_PACKAGES == []

    def test_no_required_env_vars(self):
        assert InfoGathererSkill.REQUIRED_ENV_VARS == []


# ===========================================================================
# Parameter Schema
# ===========================================================================

class TestParameterSchema:
    def test_schema_has_questions(self):
        schema = InfoGathererSkill.get_parameter_schema()
        assert "questions" in schema
        assert schema["questions"]["required"] is True

    def test_schema_has_prefix(self):
        schema = InfoGathererSkill.get_parameter_schema()
        assert "prefix" in schema
        assert schema["prefix"]["required"] is False

    def test_schema_has_completion_message(self):
        schema = InfoGathererSkill.get_parameter_schema()
        assert "completion_message" in schema

    def test_schema_has_tool_name(self):
        schema = InfoGathererSkill.get_parameter_schema()
        assert "tool_name" in schema  # from SUPPORTS_MULTIPLE_INSTANCES


# ===========================================================================
# Setup & Validation
# ===========================================================================

class TestSetup:
    def test_setup_success(self):
        skill = _make_skill({"questions": SAMPLE_QUESTIONS})
        assert skill.setup() is True

    def test_setup_missing_questions(self):
        skill = _make_skill({})
        assert skill.setup() is False

    def test_setup_empty_questions(self):
        skill = _make_skill({"questions": []})
        assert skill.setup() is False

    def test_setup_invalid_question_missing_key_name(self):
        skill = _make_skill({"questions": [{"question_text": "hi"}]})
        assert skill.setup() is False

    def test_setup_invalid_question_missing_question_text(self):
        skill = _make_skill({"questions": [{"key_name": "x"}]})
        assert skill.setup() is False

    def test_setup_questions_not_list(self):
        skill = _make_skill({"questions": "not a list"})
        assert skill.setup() is False

    def test_setup_question_not_dict(self):
        skill = _make_skill({"questions": ["not a dict"]})
        assert skill.setup() is False


# ===========================================================================
# Instance Key
# ===========================================================================

class TestInstanceKey:
    def test_instance_key_without_prefix(self):
        skill = _make_skill({"questions": SAMPLE_QUESTIONS})
        skill.setup()
        assert skill.get_instance_key() == "info_gatherer"

    def test_instance_key_with_prefix(self):
        skill = _make_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        skill.setup()
        assert skill.get_instance_key() == "info_gatherer_intake"


# ===========================================================================
# Tool Name Derivation
# ===========================================================================

class TestToolNames:
    def test_tool_names_without_prefix(self):
        skill = _setup_skill({"questions": SAMPLE_QUESTIONS})
        assert skill.start_tool_name == "start_questions"
        assert skill.submit_tool_name == "submit_answer"

    def test_tool_names_with_prefix(self):
        skill = _setup_skill({"questions": SAMPLE_QUESTIONS, "prefix": "medical"})
        assert skill.start_tool_name == "medical_start_questions"
        assert skill.submit_tool_name == "medical_submit_answer"

    def test_define_tool_called_with_correct_names(self):
        skill = _setup_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        calls = skill.agent.define_tool.call_args_list
        tool_names = [c.kwargs["name"] for c in calls]
        assert "intake_start_questions" in tool_names
        assert "intake_submit_answer" in tool_names


# ===========================================================================
# Namespace Helpers (SkillBase)
# ===========================================================================

class TestNamespaceHelpers:
    def test_namespace_with_prefix(self):
        skill = _make_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        skill.setup()
        assert skill._get_skill_namespace() == "skill:intake"

    def test_namespace_without_prefix(self):
        skill = _make_skill({"questions": SAMPLE_QUESTIONS})
        skill.setup()
        assert skill._get_skill_namespace() == "skill:info_gatherer"

    def test_get_skill_data_present(self):
        skill = _make_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        skill.setup()
        raw_data = {
            "global_data": {
                "skill:intake": {"question_index": 2, "answers": [{"key_name": "x", "answer": "y"}]}
            }
        }
        data = skill.get_skill_data(raw_data)
        assert data["question_index"] == 2

    def test_get_skill_data_missing(self):
        skill = _make_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        skill.setup()
        raw_data = {"global_data": {}}
        data = skill.get_skill_data(raw_data)
        assert data == {}

    def test_get_skill_data_no_global_data(self):
        skill = _make_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        skill.setup()
        data = skill.get_skill_data({})
        assert data == {}

    def test_update_skill_data(self):
        skill = _make_skill({"questions": SAMPLE_QUESTIONS, "prefix": "billing"})
        skill.setup()
        result = SwaigFunctionResult("test")
        returned = skill.update_skill_data(result, {"question_index": 1, "answers": []})
        assert returned is result
        result_dict = result.to_dict()
        assert "action" in result_dict
        action_data = result_dict["action"]
        # Should have a set_global_data action with the namespaced key
        found = False
        for action in action_data:
            if "set_global_data" in action:
                assert "skill:billing" in action["set_global_data"]
                found = True
        assert found


# ===========================================================================
# Global Data (initial state)
# ===========================================================================

class TestGlobalData:
    def test_global_data_structure(self):
        skill = _make_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        skill.setup()
        gd = skill.get_global_data()
        assert "skill:intake" in gd
        state = gd["skill:intake"]
        assert state["questions"] == SAMPLE_QUESTIONS
        assert state["question_index"] == 0
        assert state["answers"] == []

    def test_global_data_without_prefix(self):
        skill = _make_skill({"questions": SAMPLE_QUESTIONS})
        skill.setup()
        gd = skill.get_global_data()
        assert "skill:info_gatherer" in gd


# ===========================================================================
# Prompt Sections
# ===========================================================================

class TestPromptSections:
    def test_prompt_sections_returned(self):
        skill = _setup_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        sections = skill.get_prompt_sections()
        assert len(sections) == 1
        assert "intake_start_questions" in sections[0]["body"]
        assert "intake_submit_answer" in sections[0]["body"]

    def test_prompt_sections_no_prefix(self):
        skill = _setup_skill({"questions": SAMPLE_QUESTIONS})
        sections = skill.get_prompt_sections()
        assert "start_questions" in sections[0]["body"]


# ===========================================================================
# start_questions handler
# ===========================================================================

class TestStartQuestions:
    def _raw_data(self, prefix, questions, index=0):
        ns = f"skill:{prefix}"
        return {
            "global_data": {
                ns: {
                    "questions": questions,
                    "question_index": index,
                    "answers": [],
                }
            }
        }

    def test_returns_first_question(self):
        skill = _setup_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        raw = self._raw_data("intake", SAMPLE_QUESTIONS)
        result = skill._handle_start_questions({}, raw)
        text = result.to_dict()["response"]
        assert "What is your full name?" in text

    def test_first_question_is_marked_first(self):
        skill = _setup_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        raw = self._raw_data("intake", SAMPLE_QUESTIONS)
        result = skill._handle_start_questions({}, raw)
        text = result.to_dict()["response"]
        assert "Ask each question one at a time" in text
        assert "Previous answer saved" not in text

    def test_no_questions(self):
        skill = _setup_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        raw = self._raw_data("intake", [])
        result = skill._handle_start_questions({}, raw)
        assert "don't have any questions" in result.to_dict()["response"]

    def test_index_out_of_range(self):
        skill = _setup_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        raw = self._raw_data("intake", SAMPLE_QUESTIONS, index=99)
        result = skill._handle_start_questions({}, raw)
        assert "don't have any questions" in result.to_dict()["response"]

    def test_missing_skill_data(self):
        skill = _setup_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        raw = {"global_data": {}}
        result = skill._handle_start_questions({}, raw)
        assert "don't have any questions" in result.to_dict()["response"]


# ===========================================================================
# submit_answer handler
# ===========================================================================

class TestSubmitAnswer:
    def _raw_data(self, prefix, questions, index=0, answers=None):
        ns = f"skill:{prefix}"
        return {
            "global_data": {
                ns: {
                    "questions": questions,
                    "question_index": index,
                    "answers": answers or [],
                }
            }
        }

    def test_submit_first_answer_returns_next_question(self):
        skill = _setup_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        raw = self._raw_data("intake", SAMPLE_QUESTIONS, index=0)
        result = skill._handle_submit_answer({"answer": "John Doe"}, raw)
        d = result.to_dict()
        assert "What is your email?" in d["response"]
        assert "Previous answer saved" in d["response"]

    def test_submit_stores_answer_in_global_data(self):
        skill = _setup_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        raw = self._raw_data("intake", SAMPLE_QUESTIONS, index=0)
        result = skill._handle_submit_answer({"answer": "John Doe"}, raw)
        d = result.to_dict()
        # Find the set_global_data action
        gd_action = None
        for action in d.get("action", []):
            if "set_global_data" in action:
                gd_action = action["set_global_data"]
        assert gd_action is not None
        state = gd_action["skill:intake"]
        assert state["question_index"] == 1
        assert len(state["answers"]) == 1
        assert state["answers"][0] == {"key_name": "full_name", "answer": "John Doe"}

    def test_submit_last_answer_returns_completion(self):
        skill = _setup_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        raw = self._raw_data("intake", SAMPLE_QUESTIONS, index=2, answers=[
            {"key_name": "full_name", "answer": "John"},
            {"key_name": "email", "answer": "john@test.com"},
        ])
        result = skill._handle_submit_answer({"answer": "Need help"}, raw)
        d = result.to_dict()
        assert "All questions have been answered" in d["response"]

    def test_submit_custom_completion_message(self):
        skill = _setup_skill({
            "questions": [{"key_name": "name", "question_text": "Name?"}],
            "prefix": "quick",
            "completion_message": "Done! Thanks for answering.",
        })
        raw = {
            "global_data": {
                "skill:quick": {
                    "questions": [{"key_name": "name", "question_text": "Name?"}],
                    "question_index": 0,
                    "answers": [],
                }
            }
        }
        result = skill._handle_submit_answer({"answer": "Alice"}, raw)
        assert "Done! Thanks for answering." in result.to_dict()["response"]

    def test_submit_beyond_last_question(self):
        skill = _setup_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        raw = self._raw_data("intake", SAMPLE_QUESTIONS, index=99)
        result = skill._handle_submit_answer({"answer": "extra"}, raw)
        assert "already been answered" in result.to_dict()["response"]

    def test_confirm_flag_propagated(self):
        skill = _setup_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        # Submit answer to question 0, should get question 1 (email, confirm=True)
        raw = self._raw_data("intake", SAMPLE_QUESTIONS, index=0)
        result = skill._handle_submit_answer({"answer": "John"}, raw)
        text = result.to_dict()["response"]
        assert "Read the answer back" in text


# ===========================================================================
# Multiple Instance Isolation
# ===========================================================================

class TestMultipleInstances:
    def test_two_instances_have_different_namespaces(self):
        skill_a = _setup_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        skill_b = _setup_skill({
            "questions": [{"key_name": "allergy", "question_text": "Allergies?"}],
            "prefix": "medical",
        })
        assert skill_a._get_skill_namespace() != skill_b._get_skill_namespace()
        assert skill_a._get_skill_namespace() == "skill:intake"
        assert skill_b._get_skill_namespace() == "skill:medical"

    def test_two_instances_have_different_tool_names(self):
        skill_a = _setup_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        skill_b = _setup_skill({
            "questions": [{"key_name": "allergy", "question_text": "Allergies?"}],
            "prefix": "medical",
        })
        assert skill_a.start_tool_name == "intake_start_questions"
        assert skill_b.start_tool_name == "medical_start_questions"
        assert skill_a.submit_tool_name == "intake_submit_answer"
        assert skill_b.submit_tool_name == "medical_submit_answer"

    def test_two_instances_have_different_instance_keys(self):
        skill_a = _setup_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        skill_b = _setup_skill({
            "questions": [{"key_name": "allergy", "question_text": "Allergies?"}],
            "prefix": "medical",
        })
        assert skill_a.get_instance_key() != skill_b.get_instance_key()

    def test_two_instances_read_isolated_state(self):
        skill_a = _setup_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        skill_b = _setup_skill({
            "questions": [{"key_name": "allergy", "question_text": "Allergies?"}],
            "prefix": "medical",
        })
        raw = {
            "global_data": {
                "skill:intake": {"question_index": 1, "answers": [{"key_name": "name", "answer": "John"}]},
                "skill:medical": {"question_index": 0, "answers": []},
            }
        }
        data_a = skill_a.get_skill_data(raw)
        data_b = skill_b.get_skill_data(raw)
        assert data_a["question_index"] == 1
        assert data_b["question_index"] == 0

    def test_two_instances_global_data_isolated(self):
        skill_a = _setup_skill({"questions": SAMPLE_QUESTIONS, "prefix": "intake"})
        skill_b = _setup_skill({
            "questions": [{"key_name": "allergy", "question_text": "Allergies?"}],
            "prefix": "medical",
        })
        gd_a = skill_a.get_global_data()
        gd_b = skill_b.get_global_data()
        # They should have different namespace keys
        assert set(gd_a.keys()) != set(gd_b.keys())
        assert "skill:intake" in gd_a
        assert "skill:medical" in gd_b


# ===========================================================================
# Question instruction generation
# ===========================================================================

class TestQuestionInstruction:
    def test_first_question_format(self):
        text = InfoGathererSkill._generate_question_instruction(
            "What is your name?", needs_confirmation=False, is_first_question=True,
        )
        assert "Ask each question one at a time" in text
        assert "What is your name?" in text
        assert "Previous answer saved" not in text

    def test_subsequent_question_format(self):
        text = InfoGathererSkill._generate_question_instruction(
            "What is your email?", needs_confirmation=False, is_first_question=False,
        )
        assert "Previous answer saved" in text
        assert "What is your email?" in text

    def test_confirmation_required(self):
        text = InfoGathererSkill._generate_question_instruction(
            "SSN?", needs_confirmation=True, is_first_question=True,
        )
        assert "Read the answer back" in text

    def test_no_confirmation(self):
        text = InfoGathererSkill._generate_question_instruction(
            "Color?", needs_confirmation=False, is_first_question=True,
        )
        assert "Read the answer back" not in text

    def test_prompt_add_included(self):
        text = InfoGathererSkill._generate_question_instruction(
            "DOB?", needs_confirmation=False, is_first_question=True,
            prompt_add="Format in YYYY-MM-DD",
        )
        assert "Note: Format in YYYY-MM-DD" in text

    def test_prompt_add_empty(self):
        text = InfoGathererSkill._generate_question_instruction(
            "DOB?", needs_confirmation=False, is_first_question=True,
            prompt_add="",
        )
        assert "Note:" not in text
