"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

from typing import Dict, Any, List, Optional

from signalwire_agents.core.skill_base import SkillBase
from signalwire_agents.core.function_result import SwaigFunctionResult


class InfoGathererSkill(SkillBase):
    """
    Skill that guides an AI agent through a series of questions, collecting
    and storing answers in namespaced global_data.

    Supports multiple instances with different prefixes so several
    question sets can coexist on a single agent (e.g. "intake" and
    "medical" questionnaires running side by side).
    """

    SKILL_NAME = "info_gatherer"
    SKILL_DESCRIPTION = "Gather answers to a configurable list of questions"
    SKILL_VERSION = "1.0.0"
    REQUIRED_PACKAGES = []
    REQUIRED_ENV_VARS = []
    SUPPORTS_MULTIPLE_INSTANCES = True

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        schema = super().get_parameter_schema()
        schema.update({
            "questions": {
                "type": "array",
                "description": (
                    "List of question objects. Each must have 'key_name' (str) and "
                    "'question_text' (str). Optional 'confirm' (bool) asks the agent "
                    "to confirm the answer before proceeding."
                ),
                "required": True,
                "items": {
                    "type": "object",
                    "properties": {
                        "key_name": {"type": "string"},
                        "question_text": {"type": "string"},
                        "confirm": {"type": "boolean"},
                        "prompt_add": {"type": "string"},
                    },
                },
            },
            "prefix": {
                "type": "string",
                "description": (
                    "Optional prefix for tool names and namespace. When set, tools "
                    "are named <prefix>_start_questions / <prefix>_submit_answer and "
                    "state is stored under 'skill:<prefix>' in global_data."
                ),
                "required": False,
            },
            "completion_message": {
                "type": "string",
                "description": "Message returned after all questions are answered",
                "default": (
                    "Thank you! All questions have been answered. You can now "
                    "summarize the information collected or ask if there's anything "
                    "else the user would like to discuss."
                ),
                "required": False,
            },
        })
        return schema

    # ------------------------------------------------------------------ #
    # Instance key
    # ------------------------------------------------------------------ #

    def get_instance_key(self) -> str:
        prefix = self.params.get("prefix")
        if prefix:
            return f"info_gatherer_{prefix}"
        return "info_gatherer"

    # ------------------------------------------------------------------ #
    # Setup & validation
    # ------------------------------------------------------------------ #

    def setup(self) -> bool:
        questions = self.params.get("questions")
        if questions is None:
            self.logger.error("'questions' parameter is required")
            return False

        try:
            self._validate_questions(questions)
        except ValueError as exc:
            self.logger.error(str(exc))
            return False

        self.questions = questions

        # Derive tool names
        prefix = self.params.get("prefix")
        if prefix:
            self.start_tool_name = f"{prefix}_start_questions"
            self.submit_tool_name = f"{prefix}_submit_answer"
        else:
            self.start_tool_name = "start_questions"
            self.submit_tool_name = "submit_answer"

        self.completion_message = self.params.get(
            "completion_message",
            "Thank you! All questions have been answered. You can now "
            "summarize the information collected or ask if there's anything "
            "else the user would like to discuss.",
        )

        return True

    # ------------------------------------------------------------------ #
    # Global data (initial state)
    # ------------------------------------------------------------------ #

    def get_global_data(self) -> Dict[str, Any]:
        namespace = self._get_skill_namespace()
        return {
            namespace: {
                "questions": self.questions,
                "question_index": 0,
                "answers": [],
            }
        }

    # ------------------------------------------------------------------ #
    # Prompt sections
    # ------------------------------------------------------------------ #

    def _get_prompt_sections(self) -> List[Dict[str, Any]]:
        return [
            {
                "title": f"Info Gatherer ({self.get_instance_key()})",
                "body": (
                    f"You need to gather answers to a series of questions from the user. "
                    f"Start by introducing yourself and asking the user if they are ready "
                    f"to answer some questions. Once the user confirms they are ready, "
                    f"call the {self.start_tool_name} function to get the first question. "
                    f"Ask the user that question, wait for their response, then call "
                    f"{self.submit_tool_name} with the answer they gave you. "
                    f"Each call to {self.submit_tool_name} will return the next question "
                    f"to ask. Repeat this process until all questions are complete."
                ),
            }
        ]

    # ------------------------------------------------------------------ #
    # Tools
    # ------------------------------------------------------------------ #

    def register_tools(self) -> None:
        self.define_tool(
            name=self.start_tool_name,
            description="Start the question sequence with the first question",
            parameters={},
            handler=self._handle_start_questions,
        )

        self.define_tool(
            name=self.submit_tool_name,
            description="Submit an answer to the current question and move to the next one",
            parameters={
                "answer": {
                    "type": "string",
                    "description": "The user's answer to the current question",
                },
                "confirmed": {
                    "type": "boolean",
                    "description": "Set to true only after you have read the answer back to the user and they have explicitly confirmed it is correct. Required for questions that need confirmation.",
                },
            },
            handler=self._handle_submit_answer,
        )

    # ------------------------------------------------------------------ #
    # Handlers
    # ------------------------------------------------------------------ #

    def _handle_start_questions(self, args, raw_data):
        state = self.get_skill_data(raw_data)
        questions = state.get("questions", [])
        question_index = state.get("question_index", 0)

        if not questions or question_index >= len(questions):
            return SwaigFunctionResult("I don't have any questions to ask.")

        current = questions[question_index]
        instruction = self._generate_question_instruction(
            question_text=current.get("question_text", ""),
            needs_confirmation=current.get("confirm", False),
            is_first_question=True,
            prompt_add=current.get("prompt_add", ""),
            submit_tool_name=self.submit_tool_name,
        )
        return SwaigFunctionResult(instruction)

    def _handle_submit_answer(self, args, raw_data):
        answer = args.get("answer", "")
        confirmed = args.get("confirmed", False)
        state = self.get_skill_data(raw_data)

        questions = state.get("questions", [])
        question_index = state.get("question_index", 0)
        answers = state.get("answers", [])

        if question_index >= len(questions):
            return SwaigFunctionResult("All questions have already been answered.")

        current = questions[question_index]
        key_name = current.get("key_name", "")

        # Enforce confirmation: reject the submission if the question requires
        # confirmation but the confirmed flag was not set to true.
        if current.get("confirm", False) and not confirmed:
            return SwaigFunctionResult(
                f"Before submitting, you must read the answer \"{answer}\" back to the user "
                f"and ask them to confirm it is correct. Then call this function again with "
                f"confirmed set to true. If the user says it is wrong, ask the question again."
            )

        new_answers = answers + [{"key_name": key_name, "answer": answer}]
        new_index = question_index + 1

        if new_index < len(questions):
            next_q = questions[new_index]
            instruction = self._generate_question_instruction(
                question_text=next_q.get("question_text", ""),
                needs_confirmation=next_q.get("confirm", False),
                is_first_question=False,
                prompt_add=next_q.get("prompt_add", ""),
                submit_tool_name=self.submit_tool_name,
            )
            result = SwaigFunctionResult(instruction)
        else:
            result = SwaigFunctionResult(self.completion_message)
            result.toggle_functions([
                {"function": self.start_tool_name, "active": False},
                {"function": self.submit_tool_name, "active": False},
            ])

        new_state = {
            "questions": questions,
            "question_index": new_index,
            "answers": new_answers,
        }
        self.update_skill_data(result, new_state)
        return result

    # ------------------------------------------------------------------ #
    # Helpers (ported from InfoGathererAgent prefab)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _generate_question_instruction(
        question_text: str,
        needs_confirmation: bool,
        is_first_question: bool = False,
        prompt_add: str = "",
        submit_tool_name: str = "submit_answer",
    ) -> str:
        if is_first_question:
            instruction = f"Say this question to the user: \"{question_text}\"\n\n"
        else:
            instruction = f"Say the next question to the user: \"{question_text}\"\n\n"

        if prompt_add:
            instruction += f"Additional instructions: {prompt_add}\n\n"

        instruction += (
            f"After the user answers, make sure the answer fits the scope and context "
            f"of the question. If the answer is incomplete, ask for more detail. "
            f"Then call {submit_tool_name} with the user's answer."
        )

        if needs_confirmation:
            instruction += (
                f" IMPORTANT: Before calling {submit_tool_name}, you MUST read the answer "
                f"back to the user and ask them to confirm it is correct. "
                f"Do NOT call {submit_tool_name} until the user has confirmed. "
                f"If they say it is wrong, ask the question again."
            )

        return instruction

    @staticmethod
    def _validate_questions(questions):
        if not questions:
            raise ValueError("At least one question is required")
        if not isinstance(questions, list):
            raise ValueError("Questions must be a list")
        for i, question in enumerate(questions):
            if not isinstance(question, dict):
                raise ValueError(f"Question {i+1} must be a dictionary")
            if "key_name" not in question:
                raise ValueError(f"Question {i+1} is missing 'key_name' field")
            if "question_text" not in question:
                raise ValueError(f"Question {i+1} is missing 'question_text' field")
