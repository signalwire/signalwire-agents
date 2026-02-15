"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
InfoGathererAgent - Prefab agent for collecting answers to a series of questions

Supports both static (questions provided at init) and dynamic (questions determined 
by a callback function) configuration modes.
"""

from typing import List, Dict, Any, Optional, Union, Callable
import json

from signalwire_agents.core.agent_base import AgentBase
from signalwire_agents.core.function_result import SwaigFunctionResult


class InfoGathererAgent(AgentBase):
    """
    A prefab agent designed to collect answers to a series of questions.
    
    This agent will:
    1. Ask if the user is ready to begin
    2. Ask each question in sequence
    3. Store the answers for later use
    
    Example:
        agent = InfoGathererAgent(
            questions=[
                {"key_name": "full_name", "question_text": "What is your full name?"},
                {"key_name": "email", "question_text": "What is your email address?", "confirm": True},
                {"key_name": "reason", "question_text": "How can I help you today?"}
            ]
        )
    """
    
    def __init__(
        self,
        questions: Optional[List[Dict[str, str]]] = None,
        name: str = "info_gatherer", 
        route: str = "/info_gatherer",
        **kwargs
    ):
        """
        Initialize an information gathering agent
        
        Args:
            questions: Optional list of questions to ask. If None, questions will be determined
                      dynamically via a callback function. Each question dict should have:
                - key_name: Identifier for storing the answer
                - question_text: The actual question to ask the user
                - confirm: (Optional) If set to True, the agent will confirm the answer before submitting
            name: Agent name for the route
            route: HTTP route for this agent
            **kwargs: Additional arguments for AgentBase
        """
        # Initialize the base agent
        super().__init__(
            name=name,
            route=route,
            use_pom=True,
            **kwargs
        )
        
        # Store whether we're in static or dynamic mode
        self._static_questions = questions
        self._question_callback = None
        
        if questions is not None:
            # Static mode: validate questions and set up immediately
            self._validate_questions(questions)
            self.set_global_data({
                "questions": questions,
                "question_index": 0,
                "answers": []
            })
            # Build prompt for static configuration
            self._build_prompt()
        else:
            # Dynamic mode: questions will be set up via callback in on_swml_request
            # Build a generic prompt
            self._build_prompt("dynamic")
        
        # Configure additional agent settings
        self._configure_agent_settings()
    
    def set_question_callback(self, callback: Callable[[dict, dict, dict], List[Dict[str, str]]]):
        """
        Set a callback function for dynamic question configuration
        
        Args:
            callback: Function that takes (query_params, body_params, headers) and returns
                     a list of question dictionaries. Each question dict should have:
                     - key_name: Identifier for storing the answer
                     - question_text: The actual question to ask the user
                     - confirm: (Optional) If True, agent will confirm answer before submitting
                     
        Example:
            def my_question_callback(query_params, body_params, headers):
                question_set = query_params.get('set', 'default')
                if question_set == 'support':
                    return [
                        {"key_name": "name", "question_text": "What is your name?"},
                        {"key_name": "issue", "question_text": "What's the issue?"}
                    ]
                else:
                    return [{"key_name": "name", "question_text": "What is your name?"}]
            
            agent.set_question_callback(my_question_callback)
        """
        self._question_callback = callback
    
    def _validate_questions(self, questions):
        """Validate that questions are in the correct format"""
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
    
    def _build_prompt(self, mode="static"):
        """Build a minimal prompt with just the objective"""
        if mode == "dynamic":
            # Generic prompt for dynamic mode - will be customized later
            self.prompt_add_section(
                "Objective", 
                body="Your role is to gather information by asking questions. Begin by asking the user if they are ready to answer some questions. If they confirm they are ready, call the start_questions function to begin the process."
            )
        else:
            # Original static prompt
            self.prompt_add_section(
                "Objective", 
                body="Your role is to get answers to a series of questions. Begin by asking the user if they are ready to answer some questions. If they confirm they are ready, call the start_questions function to begin the process."
            )
    
    def _configure_agent_settings(self):
        """Configure additional agent settings"""
        # Set AI behavior parameters
        self.set_params({
            "end_of_speech_timeout": 800,
            "speech_event_timeout": 1000  # Slightly longer for thoughtful responses
        })
    
    def on_swml_request(self, request_data=None, callback_path=None, request=None):
        """
        Handle dynamic configuration using the callback function
        
        This method is called when SWML is requested and allows us to configure
        the agent just-in-time using the provided callback.
        """
        # Only process if we're in dynamic mode (no static questions)
        if self._static_questions is not None:
            return None
            
        # If no callback is set, provide a basic fallback
        if self._question_callback is None:
            fallback_questions = [
                {"key_name": "name", "question_text": "What is your name?"},
                {"key_name": "message", "question_text": "How can I help you today?"}
            ]
            return {
                "global_data": {
                    "questions": fallback_questions,
                    "question_index": 0,
                    "answers": []
                }
            }
        
        # Extract request information for callback
        query_params = {}
        body_params = request_data or {}
        headers = {}
        
        if request and hasattr(request, 'query_params'):
            query_params = dict(request.query_params)
        
        if request and hasattr(request, 'headers'):
            headers = dict(request.headers)
        
        try:
            # Call the user-provided callback to get questions
            print(f"Calling question callback with query_params: {query_params}")
            questions = self._question_callback(query_params, body_params, headers)
            print(f"Callback returned {len(questions)} questions")
            
            # Validate the returned questions
            self._validate_questions(questions)
            
            # Return global data modifications
            return {
                "global_data": {
                    "questions": questions,
                    "question_index": 0,
                    "answers": []
                }
            }
            
        except Exception as e:
            # Log error and fall back to basic questions
            print(f"Error in question callback: {e}")
            fallback_questions = [
                {"key_name": "name", "question_text": "What is your name?"},
                {"key_name": "message", "question_text": "How can I help you today?"}
            ]
            return {
                "global_data": {
                    "questions": fallback_questions,
                    "question_index": 0,
                    "answers": []
                }
            }
    

    
    def _generate_question_instruction(self, question_text: str, needs_confirmation: bool, is_first_question: bool = False) -> str:
        """
        Generate the instruction text for asking a question
        
        Args:
            question_text: The question to ask
            needs_confirmation: Whether confirmation is required
            is_first_question: Whether this is the first question or a subsequent one
            
        Returns:
            Formatted instruction text
        """
        # Start with the appropriate prefix based on whether this is the first question
        if is_first_question:
            instruction = f"Ask the user to answer the following question: {question_text}\n\n"
        else:
            instruction = f"Previous Answer recorded. Now ask the user to answer the following question: {question_text}\n\n"
        
        # Add the common part
        instruction += "Make sure the answer fits the scope and context of the question before submitting it. "
        
        # Add confirmation guidance if needed
        if needs_confirmation:
            instruction += "Insist that the user confirms the answer as many times as needed until they say it is correct."
        else:
            instruction += "You don't need the user to confirm the answer to this question."
            
        return instruction
    
    @AgentBase.tool(
        name="start_questions",
        description="Start the question sequence with the first question",
        parameters={}
    )
    def start_questions(self, args, raw_data):
        """
        Start the question sequence by retrieving the first question
        
        This function gets the current question index from global_data
        and returns the corresponding question.
        """
        # Get global data
        global_data = raw_data.get("global_data", {})
        questions = global_data.get("questions", [])
        question_index = global_data.get("question_index", 0)
        
        # Check if we have questions
        if not questions or question_index >= len(questions):
            return SwaigFunctionResult("I don't have any questions to ask.")
        
        # Get the current question
        current_question = questions[question_index]
        question_text = current_question.get("question_text", "")
        needs_confirmation = current_question.get("confirm", False)
        
        # Generate instruction using the helper method
        instruction = self._generate_question_instruction(
            question_text=question_text,
            needs_confirmation=needs_confirmation,
            is_first_question=True
        )
        
        # Return a prompt to ask the question
        result = SwaigFunctionResult(instruction)
        result.replace_in_history("Welcome! Let me ask you a few questions.")
        return result
    
    @AgentBase.tool(
        name="submit_answer",
        description="Submit an answer to the current question and move to the next one",
        parameters={
            "answer": {
                "type": "string",
                "description": "The user's answer to the current question"
            }
        }
    )
    def submit_answer(self, args, raw_data):
        """
        Submit an answer to the current question and move to the next one
        
        This function:
        1. Stores the answer in global_data
        2. Increments the question index
        3. Returns the next question or completion message
        """
        # Get the answer
        answer = args.get("answer", "")
        
        # Get global data
        global_data = raw_data.get("global_data", {})
        questions = global_data.get("questions", [])
        question_index = global_data.get("question_index", 0)
        answers = global_data.get("answers", [])
        
        # Check if we're within bounds
        if question_index >= len(questions):
            return SwaigFunctionResult("All questions have already been answered.")
        
        # Get the current question
        current_question = questions[question_index]
        key_name = current_question.get("key_name", "")
        
        # Store the answer
        new_answer = {"key_name": key_name, "answer": answer}
        new_answers = answers + [new_answer]
        
        # Increment question index
        new_question_index = question_index + 1
        
        print(f"new_question_index: {new_question_index} len(questions): {len(questions)}")
        
        # Check if we have more questions
        if new_question_index < len(questions):

            print(f"Asking next question: {new_question_index} question: {questions[new_question_index]}")
            
            # Get the next question
            next_question = questions[new_question_index]
            next_question_text = next_question.get("question_text", "")
            needs_confirmation = next_question.get("confirm", False)
            
            # Generate instruction using the helper method
            instruction = self._generate_question_instruction(
                question_text=next_question_text,
                needs_confirmation=needs_confirmation,
                is_first_question=False
            )
            
            # Create response with the global data update and next question
            result = SwaigFunctionResult(instruction)
            result.replace_in_history()

            # Use the helper method to update global data
            result.update_global_data({
                "answers": new_answers,
                "question_index": new_question_index
            })

            return result
        else:
            # No more questions - create response with global data update and completion message
            result = SwaigFunctionResult(
                "Thank you! All questions have been answered. You can now summarize the information collected or ask if there's anything else the user would like to discuss."
            )
            result.replace_in_history()

            # Use the helper method to update global data
            result.update_global_data({
                "answers": new_answers,
                "question_index": new_question_index
            })

            return result

