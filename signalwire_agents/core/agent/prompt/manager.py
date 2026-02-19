"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""Prompt management functionality for AgentBase."""

from typing import Dict, Any, Optional, List, Union
import inspect

from signalwire_agents.core.logging_config import get_logger

logger = get_logger(__name__)


class PromptManager:
    """Manages prompt building and configuration."""
    
    def __init__(self, agent):
        """
        Initialize PromptManager with reference to parent agent.
        
        Args:
            agent: Parent AgentBase instance
        """
        self.agent = agent
        self._prompt_text = None
        self._post_prompt_text = None
        self._contexts = None
    
    def _validate_prompt_mode_exclusivity(self):
        """
        Check that only one prompt mode is in use.
        
        Raises:
            ValueError: If both prompt modes are in use
        """
        if self._prompt_text and self.agent._use_pom and self.agent.pom:
            pom_sections = self.agent.pom.to_dict() if hasattr(self.agent.pom, 'to_dict') else []
            if pom_sections:
                raise ValueError(
                    "Cannot use both prompt_text and POM sections. "
                    "Please use either set_prompt_text() OR the prompt_add_* methods, not both."
                )
    
    def _process_prompt_sections(self) -> Optional[Union[str, List[Dict[str, Any]]]]:
        """
        Process prompt sections from POM or raw prompt text.
        
        Returns:
            String, List of section dictionaries, or None
        """
        # First check if we have contexts - they take precedence
        if self._contexts:
            return None  # Contexts handle their own prompt sections
            
        # Check if we have a raw prompt text - return it directly
        if self._prompt_text:
            return self._prompt_text
        
        # Otherwise use POM sections if available
        if self.agent._use_pom and self.agent.pom:
            sections = self.agent.pom.to_dict()
            if sections:
                return sections
        
        return None
    
    def define_contexts(self, contexts: Union[Dict[str, Any], Any]) -> None:
        """
        Define contexts for the agent.
        
        Args:
            contexts: Context configuration (dict or ContextBuilder)
        """
        if hasattr(contexts, 'to_dict'):
            # It's a ContextBuilder
            self._contexts = contexts.to_dict()
        elif isinstance(contexts, dict):
            # It's already a dictionary
            self._contexts = contexts
        else:
            raise ValueError("contexts must be a dictionary or a ContextBuilder object")
        
        logger.debug(f"Defined contexts: {self._contexts}")
    
    def set_prompt_text(self, text: str) -> None:
        """
        Set the agent's prompt as raw text.
        
        Args:
            text: Prompt text
        """
        self._validate_prompt_mode_exclusivity()
        self._prompt_text = text
        logger.debug(f"Set prompt text: {text[:100]}...")
    
    def set_post_prompt(self, text: str) -> None:
        """
        Set the post-prompt text.
        
        Args:
            text: Post-prompt text
        """
        self._post_prompt_text = text
        logger.debug(f"Set post-prompt text: {text[:100]}...")
    
    def set_prompt_pom(self, pom: List[Dict[str, Any]]) -> None:
        """
        Set the prompt as a POM dictionary.
        
        Args:
            pom: POM dictionary structure
            
        Raises:
            ValueError: If use_pom is False
        """
        if self.agent._use_pom:
            self.agent.pom = pom
        else:
            raise ValueError("use_pom must be True to use set_prompt_pom")
    
    def prompt_add_section(
        self, 
        title: str, 
        body: str = "", 
        bullets: Optional[List[str]] = None,
        numbered: bool = False,
        numbered_bullets: bool = False,
        subsections: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add a section to the prompt.
        
        Args:
            title: Section title
            body: Optional section body text
            bullets: Optional list of bullet points
            numbered: Whether this section should be numbered
            numbered_bullets: Whether bullets should be numbered
            subsections: Optional list of subsection objects
        """
        self._validate_prompt_mode_exclusivity()
        if self.agent._use_pom and self.agent.pom:
            # Create parameters for add_section based on what's supported
            kwargs = {}
            
            # Start with basic parameters
            kwargs['title'] = title
            kwargs['body'] = body
            if bullets:
                kwargs['bullets'] = bullets
            
            # Add optional parameters if they look supported
            if hasattr(self.agent.pom, 'add_section'):
                sig = inspect.signature(self.agent.pom.add_section)
                if 'numbered' in sig.parameters:
                    kwargs['numbered'] = numbered
                if 'numberedBullets' in sig.parameters:
                    kwargs['numberedBullets'] = numbered_bullets
            
            # Create the section
            section = self.agent.pom.add_section(**kwargs)
            
            # Now add subsections if provided, by calling add_subsection on the section
            if subsections:
                for subsection in subsections:
                    if 'title' in subsection:
                        section.add_subsection(
                            title=subsection.get('title'),
                            body=subsection.get('body', ''),
                            bullets=subsection.get('bullets', [])
                        )
    
    def prompt_add_to_section(
        self,
        title: str,
        body: Optional[str] = None,
        bullet: Optional[str] = None,
        bullets: Optional[List[str]] = None
    ) -> None:
        """
        Add content to an existing section (creating it if needed).
        
        Args:
            title: Section title
            body: Optional text to append to section body
            bullet: Optional single bullet point to add
            bullets: Optional list of bullet points to add
        """
        if self.agent._use_pom and self.agent.pom:
            # Find the section first
            section = self.agent.pom.find_section(title)
            
            if section is None:
                # Section doesn't exist, create it
                section = self.agent.pom.add_section(title=title)
            
            # Add content to the section
            if body:
                if section.body:
                    section.body = f"{section.body}\n\n{body}"
                else:
                    section.body = body
            
            # Process bullets
            bullets_to_add = []
            if bullet:
                bullets_to_add.append(bullet)
            if bullets:
                bullets_to_add.extend(bullets)
            
            if bullets_to_add:
                section.add_bullets(bullets_to_add)
    
    def prompt_add_subsection(
        self,
        parent_title: str,
        title: str,
        body: str = "",
        bullets: Optional[List[str]] = None
    ) -> None:
        """
        Add a subsection to an existing section (creating parent if needed).
        
        Args:
            parent_title: Parent section title
            title: Subsection title
            body: Optional subsection body text
            bullets: Optional list of bullet points
        """
        if self.agent._use_pom and self.agent.pom:
            # First find or create the parent section
            parent_section = None
            
            # Try to find the parent section by title
            if hasattr(self.agent.pom, 'sections'):
                for section in self.agent.pom.sections:
                    if hasattr(section, 'title') and section.title == parent_title:
                        parent_section = section
                        break
            
            # If parent section not found, create it
            if not parent_section:
                parent_section = self.agent.pom.add_section(title=parent_title)
            
            # Now call add_subsection on the parent section object, not on POM
            parent_section.add_subsection(
                title=title,
                body=body,
                bullets=bullets or []
            )
    
    def prompt_has_section(self, title: str) -> bool:
        """
        Check if a section exists in the prompt.
        
        Args:
            title: Section title to check
            
        Returns:
            True if section exists, False otherwise
        """
        if self.agent._use_pom and self.agent.pom:
            # Use find_section method from POM
            return self.agent.pom.find_section(title) is not None
        return False
    
    def get_prompt(self) -> Optional[Union[str, List[Dict[str, Any]]]]:
        """
        Get the prompt configuration.
        
        Returns:
            Prompt text or sections or None
        """
        return self._process_prompt_sections()
    
    def get_raw_prompt(self) -> Optional[str]:
        """
        Get the raw prompt text if set.
        
        Returns:
            Raw prompt text or None
        """
        return self._prompt_text
    
    def get_post_prompt(self) -> Optional[str]:
        """
        Get the post-prompt text.
        
        Returns:
            Post-prompt text or None
        """
        return self._post_prompt_text
    
    def get_contexts(self) -> Optional[Dict[str, Any]]:
        """
        Get the contexts configuration.
        
        Returns:
            Contexts dict or None
        """
        return self._contexts