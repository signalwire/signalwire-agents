"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

from typing import Optional, Union, List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from signalwire_agents.core.agent_base import AgentBase
    from signalwire_agents.core.contexts import ContextBuilder
else:
    from signalwire_agents.core.contexts import ContextBuilder


class PromptMixin:
    """
    Mixin class containing all prompt-related methods for AgentBase
    """
    
    def _process_prompt_sections(self):
        """
        Process declarative PROMPT_SECTIONS attribute from a subclass
        
        This auto-vivifies section methods and bootstraps the prompt
        from class declaration, allowing for declarative agents.
        """
        # Skip if no PROMPT_SECTIONS defined or not using POM
        cls = self.__class__
        if not hasattr(cls, 'PROMPT_SECTIONS') or cls.PROMPT_SECTIONS is None or not self._use_pom:
            return
            
        sections = cls.PROMPT_SECTIONS
        
        # If sections is a dictionary mapping section names to content
        if isinstance(sections, dict):
            for title, content in sections.items():
                # Handle different content types
                if isinstance(content, str):
                    # Plain text - add as body
                    self.prompt_add_section(title, body=content)
                elif isinstance(content, list) and content:  # Only add if non-empty
                    # List of strings - add as bullets
                    self.prompt_add_section(title, bullets=content)
                elif isinstance(content, dict):
                    # Dictionary with body/bullets/subsections
                    body = content.get('body', '')
                    bullets = content.get('bullets', [])
                    numbered = content.get('numbered', False)
                    numbered_bullets = content.get('numberedBullets', False)
                    
                    # Only create section if it has content
                    if body or bullets or 'subsections' in content:
                        # Create the section
                        self.prompt_add_section(
                            title, 
                            body=body, 
                            bullets=bullets if bullets else None,
                            numbered=numbered,
                            numbered_bullets=numbered_bullets
                        )
                        
                        # Process subsections if any
                        subsections = content.get('subsections', [])
                        for subsection in subsections:
                            if 'title' in subsection:
                                sub_title = subsection['title']
                                sub_body = subsection.get('body', '')
                                sub_bullets = subsection.get('bullets', [])
                                
                                # Only add subsection if it has content
                                if sub_body or sub_bullets:
                                    self.prompt_add_subsection(
                                        title, 
                                        sub_title,
                                        body=sub_body,
                                        bullets=sub_bullets if sub_bullets else None
                                    )
        # If sections is a list of section objects, use the POM format directly
        elif isinstance(sections, list):
            if self.pom:
                # Process each section using auto-vivifying methods
                for section in sections:
                    if 'title' in section:
                        title = section['title']
                        body = section.get('body', '')
                        bullets = section.get('bullets', [])
                        numbered = section.get('numbered', False)
                        numbered_bullets = section.get('numberedBullets', False)
                        
                        # Only create section if it has content
                        if body or bullets or 'subsections' in section:
                            self.prompt_add_section(
                                title,
                                body=body,
                                bullets=bullets if bullets else None,
                                numbered=numbered,
                                numbered_bullets=numbered_bullets
                            )
                            
                            # Process subsections if any
                            subsections = section.get('subsections', [])
                            for subsection in subsections:
                                if 'title' in subsection:
                                    sub_title = subsection['title']
                                    sub_body = subsection.get('body', '')
                                    sub_bullets = subsection.get('bullets', [])
                                    
                                    # Only add subsection if it has content
                                    if sub_body or sub_bullets:
                                        self.prompt_add_subsection(
                                            title,
                                            sub_title,
                                            body=sub_body,
                                            bullets=sub_bullets if sub_bullets else None
                                        )
    
    def define_contexts(self, contexts=None) -> Union['AgentBase', 'ContextBuilder']:
        """
        Define contexts and steps for this agent (alternative to POM/prompt)
        
        Args:
            contexts: Optional context configuration (dict or ContextBuilder)
            
        Returns:
            ContextBuilder for method chaining if no contexts provided
            
        Note:
            Contexts can coexist with traditional prompts. The restriction is only
            that you can't mix POM sections with raw text in the main prompt.
        """
        if contexts is not None:
            # New behavior - set contexts
            self._prompt_manager.define_contexts(contexts)
            return self
        else:
            # Legacy behavior - return ContextBuilder
            if self._contexts_builder is None:
                self._contexts_builder = ContextBuilder(self)
                self._contexts_defined = True
            
            return self._contexts_builder
    
    @property
    def contexts(self) -> 'ContextBuilder':
        """
        Get the ContextBuilder for this agent
        
        Returns:
            ContextBuilder instance for defining contexts
        """
        return self.define_contexts()
    
    def _validate_prompt_mode_exclusivity(self):
        """
        Validate that POM sections and raw text are not mixed in the main prompt
        
        Note: This does NOT prevent contexts from being used alongside traditional prompts
        """
        # Delegate to prompt manager
        self._prompt_manager._validate_prompt_mode_exclusivity()
    
    def set_prompt_text(self, text: str) -> 'AgentBase':
        """
        Set the prompt as raw text instead of using POM
        
        Args:
            text: The raw prompt text
            
        Returns:
            Self for method chaining
        """
        self._prompt_manager.set_prompt_text(text)
        return self
    
    def set_post_prompt(self, text: str) -> 'AgentBase':
        """
        Set the post-prompt text for summary generation
        
        Args:
            text: The post-prompt text
            
        Returns:
            Self for method chaining
        """
        self._prompt_manager.set_post_prompt(text)
        return self
    
    def set_prompt_pom(self, pom: List[Dict[str, Any]]) -> 'AgentBase':
        """
        Set the prompt as a POM dictionary
        
        Args:
            pom: POM dictionary structure
            
        Returns:
            Self for method chaining
        """
        self._prompt_manager.set_prompt_pom(pom)
        return self
    
    def prompt_add_section(
        self, 
        title: str, 
        body: str = "", 
        bullets: Optional[List[str]] = None,
        numbered: bool = False,
        numbered_bullets: bool = False,
        subsections: Optional[List[Dict[str, Any]]] = None
    ) -> 'AgentBase':
        """
        Add a section to the prompt
        
        Args:
            title: Section title
            body: Optional section body text
            bullets: Optional list of bullet points
            numbered: Whether this section should be numbered
            numbered_bullets: Whether bullets should be numbered
            subsections: Optional list of subsection objects
            
        Returns:
            Self for method chaining
        """
        self._prompt_manager.prompt_add_section(
            title=title,
            body=body,
            bullets=bullets,
            numbered=numbered,
            numbered_bullets=numbered_bullets,
            subsections=subsections
        )
        return self
    
    def prompt_add_to_section(
        self,
        title: str,
        body: Optional[str] = None,
        bullet: Optional[str] = None,
        bullets: Optional[List[str]] = None
    ) -> 'AgentBase':
        """
        Add content to an existing section (creating it if needed)
        
        Args:
            title: Section title
            body: Optional text to append to section body
            bullet: Optional single bullet point to add
            bullets: Optional list of bullet points to add
            
        Returns:
            Self for method chaining
        """
        self._prompt_manager.prompt_add_to_section(
            title=title,
            body=body,
            bullet=bullet,
            bullets=bullets
        )
        return self
    
    def prompt_add_subsection(
        self,
        parent_title: str,
        title: str,
        body: str = "",
        bullets: Optional[List[str]] = None
    ) -> 'AgentBase':
        """
        Add a subsection to an existing section (creating parent if needed)
        
        Args:
            parent_title: Parent section title
            title: Subsection title
            body: Optional subsection body text
            bullets: Optional list of bullet points
            
        Returns:
            Self for method chaining
        """
        self._prompt_manager.prompt_add_subsection(
            parent_title=parent_title,
            title=title,
            body=body,
            bullets=bullets
        )
        return self
    
    def prompt_has_section(self, title: str) -> bool:
        """
        Check if a section exists in the prompt
        
        Args:
            title: Section title to check
            
        Returns:
            True if section exists, False otherwise
        """
        return self._prompt_manager.prompt_has_section(title)
    
    def get_prompt(self) -> Union[str, List[Dict[str, Any]]]:
        """
        Get the prompt for the agent
        
        Returns:
            Either a string prompt or a POM object as list of dicts
        """
        # First check if prompt manager has a prompt
        prompt_result = self._prompt_manager.get_prompt()
        if prompt_result is not None:
            return prompt_result
            
        # If using POM, return the POM structure
        if self._use_pom and self.pom:
            try:
                # Try different methods that might be available on the POM implementation
                if hasattr(self.pom, 'render_dict'):
                    return self.pom.render_dict()
                elif hasattr(self.pom, 'to_dict'):
                    return self.pom.to_dict()
                elif hasattr(self.pom, 'to_list'):
                    return self.pom.to_list()
                elif hasattr(self.pom, 'render'):
                    render_result = self.pom.render()
                    # If render returns a string, we need to convert it to JSON
                    if isinstance(render_result, str):
                        try:
                            import json
                            return json.loads(render_result)
                        except:
                            # If we can't parse as JSON, fall back to raw text
                            pass
                    return render_result
                else:
                    # Last resort: attempt to convert the POM object directly to a list/dict
                    # This assumes the POM object has a reasonable __str__ or __repr__ method
                    pom_data = self.pom.__dict__
                    if '_sections' in pom_data and isinstance(pom_data['_sections'], list):
                        return pom_data['_sections']
                    # Fall through to default if nothing worked
            except Exception as e:
                self.log.error("pom_rendering_failed", error=str(e))
                # Fall back to raw text if POM fails
                
        # Return default text
        return f"You are {self.name}, a helpful AI assistant."
    
    def get_post_prompt(self) -> Optional[str]:
        """
        Get the post-prompt for the agent
        
        Returns:
            Post-prompt text or None if not set
        """
        return self._prompt_manager.get_post_prompt()