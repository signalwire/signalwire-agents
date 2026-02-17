"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Contexts and Steps System for SignalWire Agents

This module provides an alternative to traditional POM-based prompts by allowing
agents to be defined as structured contexts with sequential steps. Each step
contains its own prompt, completion criteria, and function restrictions.
"""

from typing import Dict, List, Optional, Union, Any


class GatherQuestion:
    """Represents a single question in a gather_info configuration"""

    def __init__(self, key: str, question: str, type: str = "string",
                 confirm: bool = False, prompt: Optional[str] = None,
                 functions: Optional[List[str]] = None):
        self.key = key
        self.question = question
        self.type = type
        self.confirm = confirm
        self.prompt = prompt
        self.functions = functions

    def to_dict(self) -> Dict[str, Any]:
        """Convert question to dictionary for SWML generation"""
        d: Dict[str, Any] = {"key": self.key, "question": self.question}
        if self.type != "string":
            d["type"] = self.type
        if self.confirm:
            d["confirm"] = True
        if self.prompt:
            d["prompt"] = self.prompt
        if self.functions:
            d["functions"] = self.functions
        return d


class GatherInfo:
    """Configuration for gathering information in a step via the C-side gather_info system.

    This produces zero tool_call/tool_result entries in LLM-visible history,
    instead using dynamic step instruction re-injection to present one question
    at a time.
    """

    def __init__(self, output_key: Optional[str] = None,
                 completion_action: Optional[str] = None,
                 prompt: Optional[str] = None):
        self._questions: List[GatherQuestion] = []
        self._output_key = output_key
        self._completion_action = completion_action
        self._prompt = prompt

    def add_question(self, key: str, question: str, **kwargs) -> 'GatherInfo':
        """
        Add a question to gather.

        Args:
            key: Key name for storing the answer in global_data
            question: The question text to ask
            **kwargs: Optional fields - type, confirm, prompt, functions

        Returns:
            Self for method chaining
        """
        q = GatherQuestion(
            key=key,
            question=question,
            type=kwargs.get('type', 'string'),
            confirm=kwargs.get('confirm', False),
            prompt=kwargs.get('prompt'),
            functions=kwargs.get('functions')
        )
        self._questions.append(q)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for SWML generation"""
        if not self._questions:
            raise ValueError("gather_info must have at least one question")
        d: Dict[str, Any] = {"questions": [q.to_dict() for q in self._questions]}
        if self._prompt:
            d["prompt"] = self._prompt
        if self._output_key:
            d["output_key"] = self._output_key
        if self._completion_action:
            d["completion_action"] = self._completion_action
        return d


class Step:
    """Represents a single step within a context"""
    
    def __init__(self, name: str):
        self.name = name
        self._text: Optional[str] = None
        self._step_criteria: Optional[str] = None
        self._functions: Optional[Union[str, List[str]]] = None
        self._valid_steps: Optional[List[str]] = None
        self._valid_contexts: Optional[List[str]] = None
        
        # POM-style sections for rich prompts
        self._sections: List[Dict[str, Any]] = []
        
        # Gather info configuration
        self._gather_info: Optional[GatherInfo] = None

        # Reset object for context switching from steps
        self._reset_system_prompt: Optional[str] = None
        self._reset_user_prompt: Optional[str] = None
        self._reset_consolidate: bool = False
        self._reset_full_reset: bool = False
    
    def set_text(self, text: str) -> 'Step':
        """
        Set the step's prompt text directly
        
        Args:
            text: The prompt text for this step
            
        Returns:
            Self for method chaining
        """
        if self._sections:
            raise ValueError("Cannot use set_text() when POM sections have been added. Use one approach or the other.")
        self._text = text
        return self
    
    def add_section(self, title: str, body: str) -> 'Step':
        """
        Add a POM section to the step
        
        Args:
            title: Section title
            body: Section body text
            
        Returns:
            Self for method chaining
        """
        if self._text is not None:
            raise ValueError("Cannot add POM sections when set_text() has been used. Use one approach or the other.")
        self._sections.append({"title": title, "body": body})
        return self
    
    def add_bullets(self, title: str, bullets: List[str]) -> 'Step':
        """
        Add a POM section with bullet points
        
        Args:
            title: Section title
            bullets: List of bullet points
            
        Returns:
            Self for method chaining
        """
        if self._text is not None:
            raise ValueError("Cannot add POM sections when set_text() has been used. Use one approach or the other.")
        self._sections.append({"title": title, "bullets": bullets})
        return self
    
    def set_step_criteria(self, criteria: str) -> 'Step':
        """
        Set the criteria for determining when this step is complete
        
        Args:
            criteria: Description of step completion criteria
            
        Returns:
            Self for method chaining
        """
        self._step_criteria = criteria
        return self
    
    def set_functions(self, functions: Union[str, List[str]]) -> 'Step':
        """
        Set which functions are available in this step
        
        Args:
            functions: "none" to disable all functions, or list of function names
            
        Returns:
            Self for method chaining
        """
        self._functions = functions
        return self
    
    def set_valid_steps(self, steps: List[str]) -> 'Step':
        """
        Set which steps can be navigated to from this step
        
        Args:
            steps: List of valid step names (include "next" for sequential flow)
            
        Returns:
            Self for method chaining
        """
        self._valid_steps = steps
        return self
    
    def set_valid_contexts(self, contexts: List[str]) -> 'Step':
        """
        Set which contexts can be navigated to from this step
        
        Args:
            contexts: List of valid context names
            
        Returns:
            Self for method chaining
        """
        self._valid_contexts = contexts
        return self
    
    def set_gather_info(self, output_key: Optional[str] = None,
                        completion_action: Optional[str] = None,
                        prompt: Optional[str] = None) -> 'Step':
        """
        Enable info gathering for this step. Questions are presented one at a time
        via dynamic step instruction re-injection, producing zero tool_call/tool_result
        entries in LLM-visible history.

        After calling this, use add_gather_question() to define questions.

        Args:
            output_key: Key in global_data to store answers under (default: top-level)
            completion_action: "next_step" to auto-advance when all questions are answered
            prompt: Preamble text injected once when entering the gather step, giving
                the model personality/context for why it is asking these questions

        Returns:
            Self for method chaining
        """
        self._gather_info = GatherInfo(output_key=output_key,
                                       completion_action=completion_action,
                                       prompt=prompt)
        return self

    def add_gather_question(self, key: str, question: str, type: str = "string",
                            confirm: bool = False, prompt: Optional[str] = None,
                            functions: Optional[List[str]] = None) -> 'Step':
        """
        Add a question to this step's gather_info configuration.
        set_gather_info() must be called before this method.

        Args:
            key: Key name for storing the answer in global_data
            question: The question text to ask the user
            type: JSON schema type for the answer param (default: "string")
            confirm: If True, model must confirm answer with user before submitting
            prompt: Extra instruction text appended for this question
            functions: Additional function names to make visible for this question

        Returns:
            Self for method chaining
        """
        if self._gather_info is None:
            raise ValueError("Must call set_gather_info() before add_gather_question()")
        self._gather_info.add_question(key=key, question=question, type=type,
                                       confirm=confirm, prompt=prompt,
                                       functions=functions)
        return self

    def clear_sections(self) -> 'Step':
        """
        Remove all POM sections and direct text from this step, allowing it
        to be repopulated with new content.

        Returns:
            Self for method chaining
        """
        self._sections = []
        self._text = None
        return self

    def set_reset_system_prompt(self, system_prompt: str) -> 'Step':
        """
        Set system prompt for context switching when this step navigates to a context

        Args:
            system_prompt: New system prompt for context switching

        Returns:
            Self for method chaining
        """
        self._reset_system_prompt = system_prompt
        return self
    
    def set_reset_user_prompt(self, user_prompt: str) -> 'Step':
        """
        Set user prompt for context switching when this step navigates to a context
        
        Args:
            user_prompt: User message to inject for context switching
            
        Returns:
            Self for method chaining
        """
        self._reset_user_prompt = user_prompt
        return self
    
    def set_reset_consolidate(self, consolidate: bool) -> 'Step':
        """
        Set whether to consolidate conversation when this step switches contexts
        
        Args:
            consolidate: Whether to consolidate previous conversation
            
        Returns:
            Self for method chaining
        """
        self._reset_consolidate = consolidate
        return self
    
    def set_reset_full_reset(self, full_reset: bool) -> 'Step':
        """
        Set whether to do full reset when this step switches contexts
        
        Args:
            full_reset: Whether to completely rewrite system prompt vs inject
            
        Returns:
            Self for method chaining
        """
        self._reset_full_reset = full_reset
        return self
    
    def _render_text(self) -> str:
        """Render the step's prompt text"""
        if self._text is not None:
            return self._text
        
        if not self._sections:
            raise ValueError(f"Step '{self.name}' has no text or POM sections defined")
        
        # Convert POM sections to markdown
        markdown_parts = []
        for section in self._sections:
            if "bullets" in section:
                markdown_parts.append(f"## {section['title']}")
                for bullet in section["bullets"]:
                    markdown_parts.append(f"- {bullet}")
            else:
                markdown_parts.append(f"## {section['title']}")
                markdown_parts.append(section["body"])
            markdown_parts.append("")  # Add spacing
        
        return "\n".join(markdown_parts).strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for SWML generation"""
        step_dict = {
            "name": self.name,
            "text": self._render_text()
        }
        
        if self._step_criteria:
            step_dict["step_criteria"] = self._step_criteria
            
        if self._functions is not None:
            step_dict["functions"] = self._functions
            
        if self._valid_steps is not None:
            step_dict["valid_steps"] = self._valid_steps
            
        if self._valid_contexts is not None:
            step_dict["valid_contexts"] = self._valid_contexts
        
        # Add reset object if any reset parameters are set
        reset_obj = {}
        if self._reset_system_prompt is not None:
            reset_obj["system_prompt"] = self._reset_system_prompt
        if self._reset_user_prompt is not None:
            reset_obj["user_prompt"] = self._reset_user_prompt
        if self._reset_consolidate:
            reset_obj["consolidate"] = self._reset_consolidate
        if self._reset_full_reset:
            reset_obj["full_reset"] = self._reset_full_reset
            
        if reset_obj:
            step_dict["reset"] = reset_obj

        if self._gather_info is not None:
            step_dict["gather_info"] = self._gather_info.to_dict()

        return step_dict


class Context:
    """Represents a single context containing multiple steps"""
    
    def __init__(self, name: str):
        self.name = name
        self._steps: Dict[str, Step] = {}
        self._step_order: List[str] = []
        self._valid_contexts: Optional[List[str]] = None
        
        # Context entry parameters
        self._post_prompt: Optional[str] = None
        self._system_prompt: Optional[str] = None
        self._system_prompt_sections: List[Dict[str, Any]] = []  # For POM-style system prompts
        self._consolidate: bool = False
        self._full_reset: bool = False
        self._user_prompt: Optional[str] = None
        self._isolated: bool = False
        
        # Context prompt (separate from system_prompt)
        self._prompt_text: Optional[str] = None
        self._prompt_sections: List[Dict[str, Any]] = []
        
        # Context fillers
        self._enter_fillers: Optional[Dict[str, List[str]]] = None
        self._exit_fillers: Optional[Dict[str, List[str]]] = None
    
    def add_step(self, name: str) -> Step:
        """
        Add a new step to this context
        
        Args:
            name: Step name
            
        Returns:
            Step object for method chaining
        """
        if name in self._steps:
            raise ValueError(f"Step '{name}' already exists in context '{self.name}'")
        
        step = Step(name)
        self._steps[name] = step
        self._step_order.append(name)
        return step
    
    def get_step(self, name: str) -> Optional['Step']:
        """
        Get an existing step by name for inspection or modification.

        Args:
            name: Step name

        Returns:
            Step object if found, None otherwise
        """
        return self._steps.get(name)

    def remove_step(self, name: str) -> 'Context':
        """
        Remove a step from this context entirely.

        Args:
            name: Step name to remove

        Returns:
            Self for method chaining
        """
        if name in self._steps:
            del self._steps[name]
            self._step_order = [s for s in self._step_order if s != name]
        return self

    def move_step(self, name: str, position: int) -> 'Context':
        """
        Move an existing step to a specific position in the step order.

        Args:
            name: Step name to move
            position: Target index in the step order (0 = first)

        Returns:
            Self for method chaining
        """
        if name not in self._steps:
            raise ValueError(f"Step '{name}' not found in context '{self.name}'")
        self._step_order.remove(name)
        self._step_order.insert(position, name)
        return self

    def set_valid_contexts(self, contexts: List[str]) -> 'Context':
        """
        Set which contexts can be navigated to from this context

        Args:
            contexts: List of valid context names

        Returns:
            Self for method chaining
        """
        self._valid_contexts = contexts
        return self
    
    def set_post_prompt(self, post_prompt: str) -> 'Context':
        """
        Set post prompt override for this context
        
        Args:
            post_prompt: Post prompt text to use when this context is active
            
        Returns:
            Self for method chaining
        """
        self._post_prompt = post_prompt
        return self
    
    def set_system_prompt(self, system_prompt: str) -> 'Context':
        """
        Set system prompt for context switching (triggers context reset)
        
        Args:
            system_prompt: New system prompt for when this context is entered
            
        Returns:
            Self for method chaining
        """
        if self._system_prompt_sections:
            raise ValueError("Cannot use set_system_prompt() when POM sections have been added for system prompt. Use one approach or the other.")
        self._system_prompt = system_prompt
        return self
    
    def set_consolidate(self, consolidate: bool) -> 'Context':
        """
        Set whether to consolidate conversation history when entering this context
        
        Args:
            consolidate: Whether to consolidate previous conversation
            
        Returns:
            Self for method chaining
        """
        self._consolidate = consolidate
        return self
    
    def set_full_reset(self, full_reset: bool) -> 'Context':
        """
        Set whether to do full reset when entering this context
        
        Args:
            full_reset: Whether to completely rewrite system prompt vs inject
            
        Returns:
            Self for method chaining
        """
        self._full_reset = full_reset
        return self
    
    def set_user_prompt(self, user_prompt: str) -> 'Context':
        """
        Set user prompt to inject when entering this context
        
        Args:
            user_prompt: User message to inject for context
            
        Returns:
            Self for method chaining
        """
        self._user_prompt = user_prompt
        return self
    
    def set_isolated(self, isolated: bool) -> 'Context':
        """
        Set whether to truncate conversation history when entering this context
        
        Args:
            isolated: Whether to truncate conversation on context switch
            
        Returns:
            Self for method chaining
        """
        self._isolated = isolated
        return self
    
    def add_system_section(self, title: str, body: str) -> 'Context':
        """
        Add a POM section to the system prompt
        
        Args:
            title: Section title
            body: Section body text
            
        Returns:
            Self for method chaining
        """
        if self._system_prompt is not None:
            raise ValueError("Cannot add POM sections for system prompt when set_system_prompt() has been used. Use one approach or the other.")
        self._system_prompt_sections.append({"title": title, "body": body})
        return self
    
    def add_system_bullets(self, title: str, bullets: List[str]) -> 'Context':
        """
        Add a POM section with bullet points to the system prompt
        
        Args:
            title: Section title
            bullets: List of bullet points
            
        Returns:
            Self for method chaining
        """
        if self._system_prompt is not None:
            raise ValueError("Cannot add POM sections for system prompt when set_system_prompt() has been used. Use one approach or the other.")
        self._system_prompt_sections.append({"title": title, "bullets": bullets})
        return self
    
    def set_prompt(self, prompt: str) -> 'Context':
        """
        Set the context's prompt text directly
        
        Args:
            prompt: The prompt text for this context
            
        Returns:
            Self for method chaining
        """
        if self._prompt_sections:
            raise ValueError("Cannot use set_prompt() when POM sections have been added. Use one approach or the other.")
        self._prompt_text = prompt
        return self
    
    def add_section(self, title: str, body: str) -> 'Context':
        """
        Add a POM section to the context prompt
        
        Args:
            title: Section title
            body: Section body text
            
        Returns:
            Self for method chaining
        """
        if self._prompt_text is not None:
            raise ValueError("Cannot add POM sections when set_prompt() has been used. Use one approach or the other.")
        self._prompt_sections.append({"title": title, "body": body})
        return self
    
    def add_bullets(self, title: str, bullets: List[str]) -> 'Context':
        """
        Add a POM section with bullet points to the context prompt
        
        Args:
            title: Section title
            bullets: List of bullet points
            
        Returns:
            Self for method chaining
        """
        if self._prompt_text is not None:
            raise ValueError("Cannot add POM sections when set_prompt() has been used. Use one approach or the other.")
        self._prompt_sections.append({"title": title, "bullets": bullets})
        return self
    
    def set_enter_fillers(self, enter_fillers: Dict[str, List[str]]) -> 'Context':
        """
        Set fillers that the AI says when entering this context
        
        Args:
            enter_fillers: Dictionary mapping language codes (or "default") to lists of filler phrases
                          Example: {"en-US": ["Welcome...", "Hello..."], "default": ["Entering..."]}
            
        Returns:
            Self for method chaining
        """
        if enter_fillers and isinstance(enter_fillers, dict):
            self._enter_fillers = enter_fillers
        return self
    
    def set_exit_fillers(self, exit_fillers: Dict[str, List[str]]) -> 'Context':
        """
        Set fillers that the AI says when exiting this context
        
        Args:
            exit_fillers: Dictionary mapping language codes (or "default") to lists of filler phrases
                         Example: {"en-US": ["Goodbye...", "Thank you..."], "default": ["Exiting..."]}
            
        Returns:
            Self for method chaining
        """
        if exit_fillers and isinstance(exit_fillers, dict):
            self._exit_fillers = exit_fillers
        return self
    
    def add_enter_filler(self, language_code: str, fillers: List[str]) -> 'Context':
        """
        Add enter fillers for a specific language
        
        Args:
            language_code: Language code (e.g., "en-US", "es") or "default" for catch-all
            fillers: List of filler phrases for entering this context
            
        Returns:
            Self for method chaining
        """
        if language_code and fillers and isinstance(fillers, list):
            if self._enter_fillers is None:
                self._enter_fillers = {}
            self._enter_fillers[language_code] = fillers
        return self
    
    def add_exit_filler(self, language_code: str, fillers: List[str]) -> 'Context':
        """
        Add exit fillers for a specific language
        
        Args:
            language_code: Language code (e.g., "en-US", "es") or "default" for catch-all
            fillers: List of filler phrases for exiting this context
            
        Returns:
            Self for method chaining
        """
        if language_code and fillers and isinstance(fillers, list):
            if self._exit_fillers is None:
                self._exit_fillers = {}
            self._exit_fillers[language_code] = fillers
        return self
    
    def _render_prompt(self) -> Optional[str]:
        """Render the context's prompt text"""
        if self._prompt_text is not None:
            return self._prompt_text
        
        if not self._prompt_sections:
            return None
        
        # Convert POM sections to markdown
        markdown_parts = []
        for section in self._prompt_sections:
            if "bullets" in section:
                markdown_parts.append(f"## {section['title']}")
                for bullet in section["bullets"]:
                    markdown_parts.append(f"- {bullet}")
            else:
                markdown_parts.append(f"## {section['title']}")
                markdown_parts.append(section["body"])
            markdown_parts.append("")  # Add spacing
        
        return "\n".join(markdown_parts).strip()
    
    def _render_system_prompt(self) -> Optional[str]:
        """Render the system prompt text"""
        if self._system_prompt is not None:
            return self._system_prompt
        
        if not self._system_prompt_sections:
            return None
        
        # Convert POM sections to markdown
        markdown_parts = []
        for section in self._system_prompt_sections:
            if "bullets" in section:
                markdown_parts.append(f"## {section['title']}")
                for bullet in section["bullets"]:
                    markdown_parts.append(f"- {bullet}")
            else:
                markdown_parts.append(f"## {section['title']}")
                markdown_parts.append(section["body"])
            markdown_parts.append("")  # Add spacing
        
        return "\n".join(markdown_parts).strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for SWML generation"""
        if not self._steps:
            raise ValueError(f"Context '{self.name}' has no steps defined")
        
        context_dict = {
            "steps": [self._steps[name].to_dict() for name in self._step_order]
        }
        
        if self._valid_contexts is not None:
            context_dict["valid_contexts"] = self._valid_contexts
        
        # Add context entry parameters
        if self._post_prompt is not None:
            context_dict["post_prompt"] = self._post_prompt
        
        rendered_system_prompt = self._render_system_prompt()
        if rendered_system_prompt is not None:
            context_dict["system_prompt"] = rendered_system_prompt
            
        if self._consolidate:
            context_dict["consolidate"] = self._consolidate
            
        if self._full_reset:
            context_dict["full_reset"] = self._full_reset
            
        if self._user_prompt is not None:
            context_dict["user_prompt"] = self._user_prompt
            
        if self._isolated:
            context_dict["isolated"] = self._isolated
        
        # Add context prompt - use POM structure if sections exist, otherwise use string
        if self._prompt_sections:
            # Use structured POM format
            context_dict["pom"] = self._prompt_sections
        elif self._prompt_text is not None:
            # Use string format
            context_dict["prompt"] = self._prompt_text
        
        # Add enter and exit fillers if defined
        if self._enter_fillers is not None:
            context_dict["enter_fillers"] = self._enter_fillers
        
        if self._exit_fillers is not None:
            context_dict["exit_fillers"] = self._exit_fillers
            
        return context_dict


class ContextBuilder:
    """Main builder class for creating contexts and steps"""
    
    def __init__(self, agent):
        self._agent = agent
        self._contexts: Dict[str, Context] = {}
        self._context_order: List[str] = []
    
    def add_context(self, name: str) -> Context:
        """
        Add a new context
        
        Args:
            name: Context name
            
        Returns:
            Context object for method chaining
        """
        if name in self._contexts:
            raise ValueError(f"Context '{name}' already exists")
        
        context = Context(name)
        self._contexts[name] = context
        self._context_order.append(name)
        return context
    
    def get_context(self, name: str) -> Optional[Context]:
        """
        Get an existing context by name for inspection or modification.

        Args:
            name: Context name

        Returns:
            Context object if found, None otherwise
        """
        return self._contexts.get(name)

    def validate(self) -> None:
        """Validate the contexts configuration"""
        if not self._contexts:
            raise ValueError("At least one context must be defined")
        
        # If only one context, it must be named "default"
        if len(self._contexts) == 1:
            context_name = list(self._contexts.keys())[0]
            if context_name != "default":
                raise ValueError("When using a single context, it must be named 'default'")
        
        # Validate each context has at least one step
        for context_name, context in self._contexts.items():
            if not context._steps:
                raise ValueError(f"Context '{context_name}' must have at least one step")
        
        # Validate step references in valid_steps
        for context_name, context in self._contexts.items():
            for step_name, step in context._steps.items():
                if step._valid_steps:
                    for valid_step in step._valid_steps:
                        if valid_step != "next" and valid_step not in context._steps:
                            raise ValueError(
                                f"Step '{step_name}' in context '{context_name}' "
                                f"references unknown step '{valid_step}'"
                            )
        
        # Validate context references in valid_contexts (context-level)
        for context_name, context in self._contexts.items():
            if context._valid_contexts:
                for valid_context in context._valid_contexts:
                    if valid_context not in self._contexts:
                        raise ValueError(
                            f"Context '{context_name}' references unknown context '{valid_context}'"
                        )

        # Validate context references in valid_contexts (step-level)
        for context_name, context in self._contexts.items():
            for step_name, step in context._steps.items():
                if hasattr(step, '_valid_contexts') and step._valid_contexts:
                    for valid_context in step._valid_contexts:
                        if valid_context not in self._contexts:
                            raise ValueError(
                                f"Step '{step_name}' in context '{context_name}' "
                                f"references unknown context '{valid_context}'"
                            )

        # Validate gather_info configurations
        for context_name, context in self._contexts.items():
            for step_name, step in context._steps.items():
                if step._gather_info is not None:
                    if not step._gather_info._questions:
                        raise ValueError(
                            f"Step '{step_name}' in context '{context_name}' "
                            f"has gather_info with no questions"
                        )
                    keys_seen = set()
                    for q in step._gather_info._questions:
                        if q.key in keys_seen:
                            raise ValueError(
                                f"Step '{step_name}' in context '{context_name}' "
                                f"has duplicate gather_info question key '{q.key}'"
                            )
                        keys_seen.add(q.key)

    def to_dict(self) -> Dict[str, Any]:
        """Convert all contexts to dictionary for SWML generation"""
        self.validate()
        
        return {
            context_name: context.to_dict() 
            for context_name in self._context_order 
            for context_name, context in [(context_name, self._contexts[context_name])]
        }


def create_simple_context(name: str = "default") -> Context:
    """
    Helper function to create a simple single context
    
    Args:
        name: Context name (defaults to "default")
        
    Returns:
        Context object for method chaining
    """
    return Context(name) 