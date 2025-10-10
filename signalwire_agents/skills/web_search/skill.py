"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import os
import requests
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import json
from typing import Optional, List, Dict, Any

from signalwire_agents.core.skill_base import SkillBase
from signalwire_agents.core.function_result import SwaigFunctionResult

class GoogleSearchScraper:
    """Google Search and Web Scraping functionality"""
    
    def __init__(self, api_key: str, search_engine_id: str, max_content_length: int = 32768):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.max_content_length = max_content_length
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def search_google(self, query: str, num_results: int = 5) -> list:
        """Search Google using Custom Search JSON API"""
        url = "https://www.googleapis.com/customsearch/v1"
        
        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': query,
            'num': min(num_results, 10)
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'items' not in data:
                return []
            
            results = []
            for item in data['items'][:num_results]:
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', '')
                })
            
            return results
            
        except Exception as e:
            return []

    def extract_text_from_url(self, url: str, content_limit: int = None, timeout: int = 10) -> str:
        """Scrape a URL and extract readable text content

        Args:
            url: URL to scrape
            content_limit: Maximum characters to return (uses self.max_content_length if not provided)
            timeout: Request timeout in seconds
        """
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove unwanted elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()

            text = soup.get_text()

            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            # Limit text length
            limit = content_limit if content_limit is not None else self.max_content_length
            if len(text) > limit:
                text = text[:limit]

            return text

        except Exception as e:
            return ""

    def search_and_scrape(self, query: str, num_results: int = 3, delay: float = 0.5) -> str:
        """Main function: search Google and scrape the resulting pages

        Dynamically calculates per-result content limit based on total max_content_length
        and number of results to ensure total response stays within bounds.
        """
        search_results = self.search_google(query, num_results)

        if not search_results:
            return f"No search results found for query: {query}"

        # Calculate per-result content budget
        # Reserve ~300 chars per result for overhead (titles, URLs, snippets, formatting)
        estimated_overhead_per_result = 300
        total_overhead = num_results * estimated_overhead_per_result
        available_for_content = self.max_content_length - total_overhead

        # Ensure we have at least 1000 chars per result
        per_result_limit = max(1000, available_for_content // num_results)

        all_text = []

        for i, result in enumerate(search_results, 1):
            text_content = f"=== RESULT {i} ===\n"
            text_content += f"Title: {result['title']}\n"
            text_content += f"URL: {result['url']}\n"
            text_content += f"Snippet: {result['snippet']}\n"
            text_content += f"Content:\n"

            # Pass the calculated per-result limit
            page_text = self.extract_text_from_url(result['url'], content_limit=per_result_limit)

            if page_text:
                text_content += page_text
            else:
                text_content += "Failed to extract content from this page."

            text_content += f"\n{'='*50}\n\n"
            all_text.append(text_content)

            if i < len(search_results):
                time.sleep(delay)

        return '\n'.join(all_text)


class WebSearchSkill(SkillBase):
    """Web search capability using Google Custom Search API"""
    
    SKILL_NAME = "web_search"
    SKILL_DESCRIPTION = "Search the web for information using Google Custom Search API"
    SKILL_VERSION = "1.0.0"
    REQUIRED_PACKAGES = ["bs4", "requests"]
    REQUIRED_ENV_VARS = []  # No required env vars since all config comes from params
    
    # Enable multiple instances support
    SUPPORTS_MULTIPLE_INSTANCES = True
    
    def get_instance_key(self) -> str:
        """
        Get the key used to track this skill instance
        
        For web search, we use the search_engine_id to differentiate instances
        """
        search_engine_id = self.params.get('search_engine_id', 'default')
        tool_name = self.params.get('tool_name', 'web_search')
        return f"{self.SKILL_NAME}_{search_engine_id}_{tool_name}"
    
    def setup(self) -> bool:
        """Setup the web search skill"""
        # Validate required parameters
        required_params = ['api_key', 'search_engine_id']
        missing_params = [param for param in required_params if not self.params.get(param)]
        if missing_params:
            self.logger.error(f"Missing required parameters: {missing_params}")
            return False
        
        if not self.validate_packages():
            return False
            
        # Set parameters from config
        self.api_key = self.params['api_key']
        self.search_engine_id = self.params['search_engine_id']
        
        # Set default parameters
        self.default_num_results = self.params.get('num_results', 1)
        self.default_delay = self.params.get('delay', 0)
        self.max_content_length = self.params.get('max_content_length', 32768)
        self.no_results_message = self.params.get('no_results_message', 
            "I couldn't find any results for '{query}'. "
            "This might be due to a very specific query or temporary issues. "
            "Try rephrasing your search or asking about a different topic."
        )
        
        # Tool name (for multiple instances)
        self.tool_name = self.params.get('tool_name', 'web_search')
        
        # Initialize the search scraper
        self.search_scraper = GoogleSearchScraper(
            api_key=self.api_key,
            search_engine_id=self.search_engine_id,
            max_content_length=self.max_content_length
        )
        
        return True
        
    def register_tools(self) -> None:
        """Register web search tool with the agent"""
        self.define_tool(
            name=self.tool_name,
            description="Search the web for information on any topic and return detailed results with content from multiple sources",
            parameters={
                "query": {
                    "type": "string",
                    "description": "The search query - what you want to find information about"
                }
            },
            handler=self._web_search_handler
        )
        
    def _web_search_handler(self, args, raw_data):
        """Handler for web search tool"""
        query = args.get("query", "").strip()
        
        if not query:
            return SwaigFunctionResult(
                "Please provide a search query. What would you like me to search for?"
            )
        
        # Use the configured number of results (no longer a parameter)
        num_results = self.default_num_results
        
        self.logger.info(f"Web search requested: '{query}' ({num_results} results)")
        
        # Perform the search
        try:
            search_results = self.search_scraper.search_and_scrape(
                query=query,
                num_results=num_results,
                delay=self.default_delay
            )
            
            if not search_results or "No search results found" in search_results:
                # Format the no results message with the query if it contains a placeholder
                formatted_message = self.no_results_message.format(query=query) if '{query}' in self.no_results_message else self.no_results_message
                return SwaigFunctionResult(formatted_message)
            
            response = f"Here are {num_results} results for '{query}':\n\nReiterate them to the user in a concise summary format\n\n{search_results}"
            return SwaigFunctionResult(response)
            
        except Exception as e:
            self.logger.error(f"Error performing web search: {e}")
            return SwaigFunctionResult(
                "Sorry, I encountered an error while searching. Please try again later."
            )
        
    def get_hints(self) -> List[str]:
        """Return speech recognition hints"""
        # Currently no hints provided, but you could add them like:
        # return [
        #     "Google", "search", "internet", "web", "information",
        #     "find", "look up", "research", "query", "results"
        # ]
        return []
        
    def get_global_data(self) -> Dict[str, Any]:
        """Return global data for agent context"""
        return {
            "web_search_enabled": True,
            "search_provider": "Google Custom Search"
        }
        
    def get_prompt_sections(self) -> List[Dict[str, Any]]:
        """Return prompt sections to add to agent"""
        return [
            {
                "title": "Web Search Capability",
                "body": f"You can search the internet for current, accurate information on any topic using the {self.tool_name} tool.",
                "bullets": [
                    f"Use the {self.tool_name} tool when users ask for information you need to look up",
                    "Search for news, current events, product information, or any current data",
                    "Summarize search results in a clear, helpful way",
                    "Include relevant URLs so users can read more if interested"
                ]
            }
        ]
    
    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get the parameter schema for the web search skill
        
        Returns all configurable parameters for web search including
        API credentials, search settings, and response customization.
        """
        # Get base schema from parent
        schema = super().get_parameter_schema()
        
        # Add web search specific parameters
        schema.update({
            "api_key": {
                "type": "string",
                "description": "Google Custom Search API key",
                "required": True,
                "hidden": True,  # Mark as hidden since it's a secret
                "env_var": "GOOGLE_SEARCH_API_KEY"
            },
            "search_engine_id": {
                "type": "string",
                "description": "Google Custom Search Engine ID",
                "required": True,
                "hidden": True,  # Also a secret
                "env_var": "GOOGLE_SEARCH_ENGINE_ID"
            },
            "num_results": {
                "type": "integer",
                "description": "Default number of search results to return",
                "default": 1,
                "required": False,
                "min": 1,
                "max": 10
            },
            "delay": {
                "type": "number",
                "description": "Delay between scraping pages in seconds",
                "default": 0,
                "required": False,
                "min": 0
            },
            "max_content_length": {
                "type": "integer",
                "description": "Maximum total response size in characters (distributed across all results)",
                "default": 32768,
                "required": False,
                "min": 1000
            },
            "no_results_message": {
                "type": "string",
                "description": "Message to show when no results are found. Use {query} as placeholder.",
                "default": "I couldn't find any results for '{query}'. This might be due to a very specific query or temporary issues. Try rephrasing your search or asking about a different topic.",
                "required": False
            }
        })
        
        return schema 
