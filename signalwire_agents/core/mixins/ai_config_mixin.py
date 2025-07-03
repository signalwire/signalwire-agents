"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

from typing import List, Dict, Any, Optional


class AIConfigMixin:
    """
    Mixin class containing all AI configuration methods for AgentBase
    """
    
    def add_hint(self, hint: str) -> 'AgentBase':
        """
        Add a simple string hint to help the AI agent understand certain words better
        
        Args:
            hint: The hint string to add
            
        Returns:
            Self for method chaining
        """
        if isinstance(hint, str) and hint:
            self._hints.append(hint)
        return self

    def add_hints(self, hints: List[str]) -> 'AgentBase':
        """
        Add multiple string hints
        
        Args:
            hints: List of hint strings
            
        Returns:
            Self for method chaining
        """
        if hints and isinstance(hints, list):
            for hint in hints:
                if isinstance(hint, str) and hint:
                    self._hints.append(hint)
        return self

    def add_pattern_hint(self, 
                         hint: str, 
                         pattern: str, 
                         replace: str, 
                         ignore_case: bool = False) -> 'AgentBase':
        """
        Add a complex hint with pattern matching
        
        Args:
            hint: The hint to match
            pattern: Regular expression pattern
            replace: Text to replace the hint with
            ignore_case: Whether to ignore case when matching
            
        Returns:
            Self for method chaining
        """
        if hint and pattern and replace:
            self._hints.append({
                "hint": hint,
                "pattern": pattern,
                "replace": replace,
                "ignore_case": ignore_case
            })
        return self

    def add_language(self, 
                     name: str, 
                     code: str, 
                     voice: str,
                     speech_fillers: Optional[List[str]] = None,
                     function_fillers: Optional[List[str]] = None,
                     engine: Optional[str] = None,
                     model: Optional[str] = None) -> 'AgentBase':
        """
        Add a language configuration to support multilingual conversations
        
        Args:
            name: Name of the language (e.g., "English", "French")
            code: Language code (e.g., "en-US", "fr-FR")
            voice: TTS voice to use. Can be a simple name (e.g., "en-US-Neural2-F") 
                  or a combined format "engine.voice:model" (e.g., "elevenlabs.josh:eleven_turbo_v2_5")
            speech_fillers: Optional list of filler phrases for natural speech
            function_fillers: Optional list of filler phrases during function calls
            engine: Optional explicit engine name (e.g., "elevenlabs", "rime")
            model: Optional explicit model name (e.g., "eleven_turbo_v2_5", "arcana")
            
        Returns:
            Self for method chaining
            
        Examples:
            # Simple voice name
            agent.add_language("English", "en-US", "en-US-Neural2-F")
            
            # Explicit parameters
            agent.add_language("English", "en-US", "josh", engine="elevenlabs", model="eleven_turbo_v2_5")
            
            # Combined format
            agent.add_language("English", "en-US", "elevenlabs.josh:eleven_turbo_v2_5")
        """
        language = {
            "name": name,
            "code": code
        }
        
        # Handle voice formatting (either explicit params or combined string)
        if engine or model:
            # Use explicit parameters if provided
            language["voice"] = voice
            if engine:
                language["engine"] = engine
            if model:
                language["model"] = model
        elif "." in voice and ":" in voice:
            # Parse combined string format: "engine.voice:model"
            try:
                engine_voice, model_part = voice.split(":", 1)
                engine_part, voice_part = engine_voice.split(".", 1)
                
                language["voice"] = voice_part
                language["engine"] = engine_part
                language["model"] = model_part
            except ValueError:
                # If parsing fails, use the voice string as-is
                language["voice"] = voice
        else:
            # Simple voice string
            language["voice"] = voice
        
        # Add fillers if provided
        if speech_fillers and function_fillers:
            language["speech_fillers"] = speech_fillers
            language["function_fillers"] = function_fillers
        elif speech_fillers or function_fillers:
            # If only one type of fillers is provided, use the deprecated "fillers" field
            fillers = speech_fillers or function_fillers
            language["fillers"] = fillers
        
        self._languages.append(language)
        return self

    def set_languages(self, languages: List[Dict[str, Any]]) -> 'AgentBase':
        """
        Set all language configurations at once
        
        Args:
            languages: List of language configuration dictionaries
            
        Returns:
            Self for method chaining
        """
        if languages and isinstance(languages, list):
            self._languages = languages
        return self

    def add_pronunciation(self, 
                         replace: str, 
                         with_text: str, 
                         ignore_case: bool = False) -> 'AgentBase':
        """
        Add a pronunciation rule to help the AI speak certain words correctly
        
        Args:
            replace: The expression to replace
            with_text: The phonetic spelling to use instead
            ignore_case: Whether to ignore case when matching
            
        Returns:
            Self for method chaining
        """
        if replace and with_text:
            rule = {
                "replace": replace,
                "with": with_text
            }
            if ignore_case:
                rule["ignore_case"] = True
            
            self._pronounce.append(rule)
        return self

    def set_pronunciations(self, pronunciations: List[Dict[str, Any]]) -> 'AgentBase':
        """
        Set all pronunciation rules at once
        
        Args:
            pronunciations: List of pronunciation rule dictionaries
            
        Returns:
            Self for method chaining
        """
        if pronunciations and isinstance(pronunciations, list):
            self._pronounce = pronunciations
        return self

    def set_param(self, key: str, value: Any) -> 'AgentBase':
        """
        Set a single AI parameter
        
        Args:
            key: Parameter name
            value: Parameter value
            
        Returns:
            Self for method chaining
        """
        if key:
            self._params[key] = value
        return self

    def set_params(self, params: Dict[str, Any]) -> 'AgentBase':
        """
        Set multiple AI parameters at once
        
        Args:
            params: Dictionary of parameter name/value pairs
            
        Returns:
            Self for method chaining
        """
        if params and isinstance(params, dict):
            self._params.update(params)
        return self

    def set_global_data(self, data: Dict[str, Any]) -> 'AgentBase':
        """
        Set the global data available to the AI throughout the conversation
        
        Args:
            data: Dictionary of global data
            
        Returns:
            Self for method chaining
        """
        if data and isinstance(data, dict):
            self._global_data = data
        return self

    def update_global_data(self, data: Dict[str, Any]) -> 'AgentBase':
        """
        Update the global data with new values
        
        Args:
            data: Dictionary of global data to update
            
        Returns:
            Self for method chaining
        """
        if data and isinstance(data, dict):
            self._global_data.update(data)
        return self

    def set_native_functions(self, function_names: List[str]) -> 'AgentBase':
        """
        Set the list of native functions to enable
        
        Args:
            function_names: List of native function names
            
        Returns:
            Self for method chaining
        """
        if function_names and isinstance(function_names, list):
            self.native_functions = [name for name in function_names if isinstance(name, str)]
        return self

    def set_internal_fillers(self, internal_fillers: Dict[str, Dict[str, List[str]]]) -> 'AgentBase':
        """
        Set internal fillers for native SWAIG functions
        
        Internal fillers provide custom phrases the AI says while executing
        internal/native functions like check_time, wait_for_user, next_step, etc.
        
        Args:
            internal_fillers: Dictionary mapping function names to language-specific filler phrases
                            Format: {"function_name": {"language_code": ["phrase1", "phrase2"]}}
                            Example: {"next_step": {"en-US": ["Moving to the next step...", "Great, let's continue..."]}}
            
        Returns:
            Self for method chaining
            
        Example:
            agent.set_internal_fillers({
                "next_step": {
                    "en-US": ["Moving to the next step...", "Great, let's continue..."],
                    "es": ["Pasando al siguiente paso...", "Excelente, continuemos..."]
                },
                "check_time": {
                    "en-US": ["Let me check the time...", "Getting the current time..."]
                }
            })
        """
        if internal_fillers and isinstance(internal_fillers, dict):
            if not hasattr(self, '_internal_fillers'):
                self._internal_fillers = {}
            self._internal_fillers.update(internal_fillers)
        return self

    def add_internal_filler(self, function_name: str, language_code: str, fillers: List[str]) -> 'AgentBase':
        """
        Add internal fillers for a specific function and language
        
        Args:
            function_name: Name of the internal function (e.g., 'next_step', 'check_time')
            language_code: Language code (e.g., 'en-US', 'es', 'fr')
            fillers: List of filler phrases for this function and language
            
        Returns:
            Self for method chaining
            
        Example:
            agent.add_internal_filler("next_step", "en-US", ["Moving to the next step...", "Great, let's continue..."])
        """
        if function_name and language_code and fillers and isinstance(fillers, list):
            if not hasattr(self, '_internal_fillers'):
                self._internal_fillers = {}
            
            if function_name not in self._internal_fillers:
                self._internal_fillers[function_name] = {}
                
            self._internal_fillers[function_name][language_code] = fillers
        return self

    def add_function_include(self, url: str, functions: List[str], meta_data: Optional[Dict[str, Any]] = None) -> 'AgentBase':
        """
        Add a remote function include to the SWAIG configuration
        
        Args:
            url: URL to fetch remote functions from
            functions: List of function names to include
            meta_data: Optional metadata to include with the function include
            
        Returns:
            Self for method chaining
        """
        if url and functions and isinstance(functions, list):
            include = {
                "url": url,
                "functions": functions
            }
            if meta_data and isinstance(meta_data, dict):
                include["meta_data"] = meta_data
            
            self._function_includes.append(include)
        return self

    def set_function_includes(self, includes: List[Dict[str, Any]]) -> 'AgentBase':
        """
        Set the complete list of function includes
        
        Args:
            includes: List of include objects, each with url and functions properties
            
        Returns:
            Self for method chaining
        """
        if includes and isinstance(includes, list):
            # Validate each include has required properties
            valid_includes = []
            for include in includes:
                if isinstance(include, dict) and "url" in include and "functions" in include:
                    if isinstance(include["functions"], list):
                        valid_includes.append(include)
            
            self._function_includes = valid_includes
        return self
    
    def set_prompt_llm_params(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        confidence: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None
    ) -> 'AgentBase':
        """
        Set LLM parameters for the main prompt.
        
        Args:
            temperature: Randomness setting (0.0-1.5). Lower values make output more deterministic.
                        Default: 0.3
            top_p: Alternative to temperature (0.0-1.0). Controls nucleus sampling.
                   Default: 1.0
            confidence: Speech detection threshold (0.0-1.0). Lower values reduce pause after user speaks.
                       Default: 0.75
            presence_penalty: Topic diversity (-2.0 to 2.0). Positive values encourage new topics.
                             Default: 0.1
            frequency_penalty: Repetition control (-2.0 to 2.0). Positive values reduce repetition.
                              Default: 0.1
        
        Returns:
            Self for method chaining
            
        Example:
            agent.set_prompt_llm_params(
                temperature=0.7,
                top_p=0.9,
                confidence=0.6
            )
        """
        # Initialize prompt LLM params if not exists
        if not hasattr(self, '_prompt_llm_params'):
            self._prompt_llm_params = {}
        
        # Validate and set temperature
        if temperature is not None:
            if not 0.0 <= temperature <= 1.5:
                raise ValueError("temperature must be between 0.0 and 1.5")
            self._prompt_llm_params['temperature'] = temperature
        
        # Validate and set top_p
        if top_p is not None:
            if not 0.0 <= top_p <= 1.0:
                raise ValueError("top_p must be between 0.0 and 1.0")
            self._prompt_llm_params['top_p'] = top_p
        
        # Validate and set confidence
        if confidence is not None:
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be between 0.0 and 1.0")
            self._prompt_llm_params['confidence'] = confidence
        
        # Validate and set presence_penalty
        if presence_penalty is not None:
            if not -2.0 <= presence_penalty <= 2.0:
                raise ValueError("presence_penalty must be between -2.0 and 2.0")
            self._prompt_llm_params['presence_penalty'] = presence_penalty
        
        # Validate and set frequency_penalty
        if frequency_penalty is not None:
            if not -2.0 <= frequency_penalty <= 2.0:
                raise ValueError("frequency_penalty must be between -2.0 and 2.0")
            self._prompt_llm_params['frequency_penalty'] = frequency_penalty
        
        return self
    
    def set_post_prompt_llm_params(
        self,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        confidence: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None
    ) -> 'AgentBase':
        """
        Set LLM parameters for the post-prompt.
        
        Args:
            temperature: Randomness setting (0.0-1.5). Lower values make output more deterministic.
                        Default: 0.0
            top_p: Alternative to temperature (0.0-1.0). Controls nucleus sampling.
                   Default: 1.0
            confidence: Speech detection threshold (0.0-1.0). Lower values reduce pause after user speaks.
                       Default: 0.75
            presence_penalty: Topic diversity (-2.0 to 2.0). Positive values encourage new topics.
                             Default: 0.0
            frequency_penalty: Repetition control (-2.0 to 2.0). Positive values reduce repetition.
                              Default: 0.0
        
        Returns:
            Self for method chaining
            
        Example:
            agent.set_post_prompt_llm_params(
                temperature=0.5,  # More deterministic for post-prompt
                confidence=0.7
            )
        """
        # Initialize post prompt LLM params if not exists
        if not hasattr(self, '_post_prompt_llm_params'):
            self._post_prompt_llm_params = {}
        
        # Validate and set temperature
        if temperature is not None:
            if not 0.0 <= temperature <= 1.5:
                raise ValueError("temperature must be between 0.0 and 1.5")
            self._post_prompt_llm_params['temperature'] = temperature
        
        # Validate and set top_p
        if top_p is not None:
            if not 0.0 <= top_p <= 1.0:
                raise ValueError("top_p must be between 0.0 and 1.0")
            self._post_prompt_llm_params['top_p'] = top_p
        
        # Validate and set confidence
        if confidence is not None:
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be between 0.0 and 1.0")
            self._post_prompt_llm_params['confidence'] = confidence
        
        # Validate and set presence_penalty
        if presence_penalty is not None:
            if not -2.0 <= presence_penalty <= 2.0:
                raise ValueError("presence_penalty must be between -2.0 and 2.0")
            self._post_prompt_llm_params['presence_penalty'] = presence_penalty
        
        # Validate and set frequency_penalty
        if frequency_penalty is not None:
            if not -2.0 <= frequency_penalty <= 2.0:
                raise ValueError("frequency_penalty must be between -2.0 and 2.0")
            self._post_prompt_llm_params['frequency_penalty'] = frequency_penalty
        
        return self