"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Wikipedia Search Skill

Provides Wikipedia search capabilities using the Wikipedia API.
"""

import requests
from urllib.parse import quote
from typing import Dict, Any, Optional
from signalwire_agents.core.skill_base import SkillBase


class WikipediaSearchSkill(SkillBase):
    """
    Skill for searching Wikipedia articles and retrieving content.
    
    This skill uses the Wikipedia API to search for articles and retrieve
    their introductory content, similar to getting a summary of a topic.
    """
    
    # Skill metadata
    SKILL_NAME = "wikipedia_search"
    SKILL_DESCRIPTION = "Search Wikipedia for information about a topic and get article summaries"
    SKILL_VERSION = "1.0.0"
    REQUIRED_PACKAGES = ["requests"]
    REQUIRED_ENV_VARS = []  # No environment variables required
    
    # Does not support multiple instances
    SUPPORTS_MULTIPLE_INSTANCES = False
    
    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Get parameter schema for Wikipedia search skill"""
        schema = super().get_parameter_schema()
        schema.update({
            "num_results": {
                "type": "integer",
                "description": "Maximum number of Wikipedia articles to return",
                "default": 1,
                "required": False,
                "minimum": 1,
                "maximum": 5
            },
            "no_results_message": {
                "type": "string",
                "description": "Custom message when no Wikipedia articles are found",
                "default": "I couldn't find any Wikipedia articles for '{query}'. Try rephrasing your search or using different keywords.",
                "required": False
            }
        })
        return schema
    
    def setup(self) -> bool:
        """
        Setup the Wikipedia search skill.
        
        Returns:
            True if setup successful, False otherwise
        """
        # Extract configuration from params
        self.num_results = max(1, self.params.get('num_results', 1))  # Ensure at least 1 result
        self.no_results_message = self.params.get('no_results_message') or (
            "I couldn't find any Wikipedia articles for '{query}'. "
            "Try rephrasing your search or using different keywords."
        )
        
        # Validate that requests package is available
        if not self.validate_packages():
            return False
            
        self.logger.info(f"Wikipedia search skill initialized with {self.num_results} max results")
        return True
    
    def register_tools(self) -> None:
        """
        Register the SWAIG tool for Wikipedia search.
        """
        self.define_tool(
            name="search_wiki",
            description="Search Wikipedia for information about a topic and get article summaries",
            parameters={
                "query": {
                    "type": "string",
                    "description": "The search term or topic to look up on Wikipedia"
                }
            },
            handler=self._search_wiki_handler
        )
    
    def _search_wiki_handler(self, args, raw_data):
        """Handler for search_wiki tool"""
        from signalwire_agents.core.function_result import SwaigFunctionResult
        
        query = args.get("query", "").strip()
        if not query:
            return SwaigFunctionResult("Please provide a search query for Wikipedia.")
        
        result = self.search_wiki(query)
        return SwaigFunctionResult(result)
    
    def search_wiki(self, query: str) -> str:
        """
        Search Wikipedia for articles matching the query.
        
        Args:
            query: The search term to look up
            
        Returns:
            String containing the Wikipedia article content or error message
        """
        try:
            # Step 1: Search for articles matching the query
            search_url = (
                "https://en.wikipedia.org/w/api.php"
                "?action=query&list=search&format=json"
                f"&srsearch={quote(query)}"
                f"&srlimit={self.num_results}"
            )
            
            response = requests.get(search_url, timeout=10)
            response.raise_for_status()
            search_data = response.json()
            
            # Check if we got any search results
            search_results = search_data.get('query', {}).get('search', [])
            if not search_results:
                return self.no_results_message.format(query=query)
            
            # Step 2: Get article extracts for each result
            articles = []
            for i, result in enumerate(search_results[:self.num_results]):
                title = result['title']
                
                # Get the page extract
                extract_url = (
                    "https://en.wikipedia.org/w/api.php"
                    "?action=query&prop=extracts&exintro&explaintext&format=json"
                    f"&titles={quote(title)}"
                )
                
                extract_response = requests.get(extract_url, timeout=10)
                extract_response.raise_for_status()
                extract_data = extract_response.json()
                
                # Get the first (and only) page from the response
                pages = extract_data.get('query', {}).get('pages', {})
                if pages:
                    page = next(iter(pages.values()))
                    extract = page.get('extract', '').strip()
                    
                    if extract:
                        # Format the article with title and content
                        articles.append(f"**{title}**\n\n{extract}")
                    else:
                        # Handle case where extract is empty
                        articles.append(f"**{title}**\n\nNo summary available for this article.")
            
            if not articles:
                return self.no_results_message.format(query=query)
            
            # Join multiple articles with separators
            if len(articles) == 1:
                return articles[0]
            else:
                return "\n\n" + "="*50 + "\n\n".join(articles)
                
        except requests.exceptions.RequestException as e:
            return f"Error accessing Wikipedia: {str(e)}"
        except Exception as e:
            return f"Error searching Wikipedia: {str(e)}"
    
    def get_prompt_sections(self) -> list:
        """
        Return additional context for the agent prompt.
        
        Returns:
            List of prompt sections to add to the agent
        """
        return [{
            "title": "Wikipedia Search",
            "body": f"You can search Wikipedia for factual information using search_wiki. This will return up to {self.num_results} Wikipedia article summaries.",
            "bullets": [
                "Use search_wiki for factual, encyclopedic information",
                "Great for answering questions about people, places, concepts, and history",
                "Returns reliable, well-sourced information from Wikipedia articles"
            ]
        }]
    
    def get_hints(self) -> list:
        """
        Return speech recognition hints for better accuracy.
        
        Returns:
            List of words/phrases to help with speech recognition
        """
        # Currently no hints provided, but you could add them like:
        # return [
        #     "Wikipedia", "wiki", "search Wikipedia", "look up", "tell me about",
        #     "what is", "who is", "information about", "facts about"
        # ]
        return [] 