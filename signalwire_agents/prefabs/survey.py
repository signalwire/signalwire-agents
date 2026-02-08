"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
SurveyAgent - Prefab agent for conducting automated surveys
"""

from typing import List, Dict, Any, Optional, Union
import json
import os
from datetime import datetime

from signalwire_agents.core.agent_base import AgentBase
from signalwire_agents.core.function_result import SwaigFunctionResult


class SurveyAgent(AgentBase):
    """
    A prefab agent designed to conduct automated surveys with users.
    
    This agent will:
    1. Introduce the survey purpose and structure
    2. Ask predefined questions in sequence
    3. Collect and validate responses
    4. Provide a summary of collected responses
    
    Example:
        agent = SurveyAgent(
            survey_name="Customer Satisfaction Survey",
            introduction="We'd like to get your feedback on your recent experience.",
            questions=[
                {
                    "id": "satisfaction",
                    "text": "How satisfied were you with our service?",
                    "type": "rating",
                    "scale": 5,
                    "required": True
                },
                {
                    "id": "comments",
                    "text": "Do you have any additional comments?",
                    "type": "open_ended",
                    "required": False
                }
            ]
        )
    """
    
    def __init__(
        self,
        survey_name: str,
        questions: List[Dict[str, Any]],
        introduction: Optional[str] = None,
        conclusion: Optional[str] = None,
        brand_name: Optional[str] = None,
        max_retries: int = 2,
        name: str = "survey",
        route: str = "/survey",
        **kwargs
    ):
        """
        Initialize a survey agent
        
        Args:
            survey_name: Name of the survey
            questions: List of question objects with the following keys:
                - id: Unique identifier for the question
                - text: The question text to ask
                - type: Type of question (rating, multiple_choice, yes_no, open_ended)
                - options: List of options for multiple_choice questions
                - scale: For rating questions, the scale (e.g., 1-5)
                - required: Whether the question requires an answer
            introduction: Optional custom introduction message
            conclusion: Optional custom conclusion message
            brand_name: Optional brand or company name
            max_retries: Maximum number of times to retry invalid answers
            name: Name for the agent (default: "survey")
            route: HTTP route for the agent (default: "/survey")
            **kwargs: Additional arguments for AgentBase
        """
        # Initialize the base agent
        super().__init__(
            name=name,
            route=route,
            use_pom=True,
            **kwargs
        )
        
        # Store configuration
        self.survey_name = survey_name
        self.questions = questions
        self.brand_name = brand_name or "Our Company"
        self.max_retries = max_retries
        
        # Default messages if not provided
        self.introduction = introduction or f"Welcome to our {survey_name}. We appreciate your participation."
        self.conclusion = conclusion or "Thank you for completing our survey. Your feedback is valuable to us."
        
        # Validate questions
        self._validate_questions()
        
        # Set up the agent's configuration
        self._setup_survey_agent()
    
    def _validate_questions(self):
        """Validate the question format and structure"""
        valid_types = ["rating", "multiple_choice", "yes_no", "open_ended"]
        
        for i, question in enumerate(self.questions):
            # Ensure required fields are present
            if "id" not in question or not question["id"]:
                question["id"] = f"question_{i+1}"
                
            if "text" not in question or not question["text"]:
                raise ValueError(f"Question {i+1} is missing the 'text' field")
                
            if "type" not in question or question["type"] not in valid_types:
                raise ValueError(f"Question {i+1} has an invalid type. Must be one of: {', '.join(valid_types)}")
                
            # Set defaults for optional fields
            if "required" not in question:
                question["required"] = True
                
            # Type-specific validation
            if question["type"] == "multiple_choice" and ("options" not in question or not question["options"]):
                raise ValueError(f"Multiple choice question '{question['id']}' must have options")
                
            if question["type"] == "rating" and "scale" not in question:
                question["scale"] = 5  # Default to 5-point scale
    
    def _setup_survey_agent(self):
        """Configure the survey agent with appropriate settings"""
        # Basic personality and instructions
        self.prompt_add_section("Personality", 
            body=f"You are a friendly and professional survey agent representing {self.brand_name}."
        )
        
        self.prompt_add_section("Goal", 
            body=f"Conduct the '{self.survey_name}' survey by asking questions and collecting responses."
        )
        
        # Build detailed instructions
        instructions = [
            "Guide the user through each survey question in sequence.",
            "Ask only one question at a time and wait for a response.",
            "For rating questions, explain the scale (e.g., 1-5, where 5 is best).",
            "For multiple choice questions, list all the options.",
            f"If a response is invalid, explain and retry up to {self.max_retries} times.",
            "Be conversational but stay focused on collecting the survey data.",
            "After all questions are answered, thank the user for their participation."
        ]
        
        self.prompt_add_section("Instructions", bullets=instructions)
        
        # Add introduction section
        self.prompt_add_section("Introduction", 
            body=f"Begin with this introduction: {self.introduction}"
        )
        
        # Questions section with all the survey questions
        questions_subsections = []
        for q in self.questions:
            # Build a description based on question type
            description = f"ID: {q['id']}\nType: {q['type']}\nRequired: {q['required']}"
            
            if q["type"] == "rating":
                description += f"\nScale: 1-{q['scale']}"
                
            if q["type"] == "multiple_choice" and "options" in q:
                options_list = ", ".join(q["options"])
                description += f"\nOptions: {options_list}"
                
            questions_subsections.append({
                "title": q["text"],
                "body": description
            })
            
        self.prompt_add_section("Survey Questions", 
            body="Ask these questions in order:",
            subsections=questions_subsections
        )
        
        # Add conclusion section
        self.prompt_add_section("Conclusion", 
            body=f"End with this conclusion: {self.conclusion}"
        )
        
        # Set up the post-prompt for summary
        post_prompt = """
        Return a JSON summary of the survey responses:
        {
            "survey_name": "SURVEY_NAME",
            "responses": {
                "QUESTION_ID_1": "RESPONSE_1",
                "QUESTION_ID_2": "RESPONSE_2",
                ...
            },
            "completion_status": "complete/incomplete",
            "timestamp": "CURRENT_TIMESTAMP"
        }
        """
        self.set_post_prompt(post_prompt)
        
        # Configure hints to help the AI understand survey terminology
        type_terms = []
        for q in self.questions:
            if q["type"] == "rating":
                type_terms.extend([str(i) for i in range(1, q["scale"]+1)])
            elif q["type"] == "multiple_choice" and "options" in q:
                type_terms.extend(q["options"])
            elif q["type"] == "yes_no":
                type_terms.extend(["yes", "no"])
                
        self.add_hints([
            self.survey_name,
            self.brand_name,
            *type_terms
        ])
        
        # Set AI behavior parameters
        self.set_params({
            "wait_for_user": False,
            "end_of_speech_timeout": 1500,  # Longer timeout for thoughtful responses
            "ai_volume": 5,
            "static_greeting": self.introduction,
            "static_greeting_no_barge": True
        })
        
        # Add global data available to the AI
        self.set_global_data({
            "survey_name": self.survey_name,
            "brand_name": self.brand_name,
            "questions": self.questions,
            "max_retries": self.max_retries
        })
        
        # Configure native functions
        self.set_native_functions(["check_time"])
    
    @AgentBase.tool(
        name="validate_response",
        description="Validate if a response meets the requirements for a specific question",
        parameters={
            "question_id": {
                "type": "string",
                "description": "The ID of the question"
            },
            "response": {
                "type": "string",
                "description": "The user's response to validate"
            }
        }
    )
    def validate_response(self, args, raw_data):
        """
        Validate if a response meets the requirements for a specific question
        
        This function checks if a user's response is valid for the specified question
        based on the question type and constraints.
        """
        question_id = args.get("question_id", "")
        response = args.get("response", "")
        
        # Find the question by ID
        question = None
        for q in self.questions:
            if q["id"] == question_id:
                question = q
                break
                
        if not question:
            return SwaigFunctionResult(f"Error: Question with ID '{question_id}' not found.")
        
        # Validate based on question type
        valid = True
        message = f"Response to '{question_id}' is valid."
        
        if question["type"] == "rating":
            try:
                rating = int(response.strip())
                if rating < 1 or rating > question.get("scale", 5):
                    valid = False
                    message = f"Invalid rating. Please provide a number between 1 and {question.get('scale', 5)}."
            except ValueError:
                valid = False
                message = f"Invalid rating. Please provide a number between 1 and {question.get('scale', 5)}."
                
        elif question["type"] == "multiple_choice":
            options = question.get("options", [])
            if not any(response.lower().strip() == option.lower() for option in options):
                valid = False
                message = f"Invalid choice. Please select one of: {', '.join(options)}."
                
        elif question["type"] == "yes_no":
            response_lower = response.lower().strip()
            if response_lower not in ["yes", "no", "y", "n"]:
                valid = False
                message = "Please answer with 'yes' or 'no'."
        
        # For open-ended, any non-empty response is valid
        elif question["type"] == "open_ended":
            if not response.strip() and question.get("required", True):
                valid = False
                message = "A response is required for this question."
                
        return SwaigFunctionResult(message)
    
    @AgentBase.tool(
        name="log_response",
        description="Log a validated response to a survey question",
        parameters={
            "question_id": {
                "type": "string",
                "description": "The ID of the question"
            },
            "response": {
                "type": "string",
                "description": "The user's validated response"
            }
        }
    )
    def log_response(self, args, raw_data):
        """
        Log a validated response to a survey question
        
        This function would typically connect to a database or API to store the response.
        In this example, it just acknowledges that the response was received.
        """
        question_id = args.get("question_id", "")
        response = args.get("response", "")
        
        # Find the question by ID for a more informative message
        question_text = ""
        for q in self.questions:
            if q["id"] == question_id:
                question_text = q["text"]
                break
        
        # In a real implementation, you would store this response in a database
        # For this example, we just acknowledge it
        message = f"Response to '{question_text}' has been recorded."
        
        return SwaigFunctionResult(message)
    
    def on_summary(self, summary, raw_data=None):
        """
        Process the survey results summary
        
        Args:
            summary: Summary data containing survey responses
            raw_data: The complete raw POST data from the request
        """
        if summary:
            try:
                # Log survey completion
                if isinstance(summary, dict):
                    print(f"Survey completed: {json.dumps(summary, indent=2)}")
                    
                    # Here you would typically:
                    # 1. Store the responses in a database
                    # 2. Trigger any follow-up actions
                    # 3. Send notifications if needed
                    
                else:
                    print(f"Survey summary (unstructured): {summary}")
                    
            except Exception as e:
                print(f"Error processing survey summary: {str(e)}")
