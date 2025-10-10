"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import requests
import json
from typing import Optional, List, Dict, Any

from signalwire_agents.core.skill_base import SkillBase
from signalwire_agents.core.function_result import SwaigFunctionResult

class DataSphereSkill(SkillBase):
    """SignalWire DataSphere knowledge search capability"""
    
    SKILL_NAME = "datasphere"
    SKILL_DESCRIPTION = "Search knowledge using SignalWire DataSphere RAG stack"
    SKILL_VERSION = "1.0.0"
    REQUIRED_PACKAGES = ["requests"]
    REQUIRED_ENV_VARS = []  # No required env vars since all config comes from params
    
    # Enable multiple instances support
    SUPPORTS_MULTIPLE_INSTANCES = True
    
    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Get parameter schema for DataSphere skill"""
        schema = super().get_parameter_schema()
        schema.update({
            "space_name": {
                "type": "string",
                "description": "SignalWire space name (e.g., 'mycompany' from mycompany.signalwire.com)",
                "required": True
            },
            "project_id": {
                "type": "string",
                "description": "SignalWire project ID",
                "required": True,
                "env_var": "SIGNALWIRE_PROJECT_ID"
            },
            "token": {
                "type": "string",
                "description": "SignalWire API token",
                "required": True,
                "hidden": True,
                "env_var": "SIGNALWIRE_TOKEN"
            },
            "document_id": {
                "type": "string",
                "description": "DataSphere document ID to search within",
                "required": True
            },
            "count": {
                "type": "integer",
                "description": "Number of search results to return",
                "default": 1,
                "required": False,
                "minimum": 1,
                "maximum": 10
            },
            "distance": {
                "type": "number",
                "description": "Maximum distance threshold for results (lower is more relevant)",
                "default": 3.0,
                "required": False,
                "minimum": 0.0,
                "maximum": 10.0
            },
            "tags": {
                "type": "array",
                "description": "Tags to filter search results",
                "required": False,
                "items": {
                    "type": "string"
                }
            },
            "language": {
                "type": "string",
                "description": "Language code for query expansion (e.g., 'en', 'es')",
                "required": False
            },
            "pos_to_expand": {
                "type": "array",
                "description": "Parts of speech to expand with synonyms",
                "required": False,
                "items": {
                    "type": "string",
                    "enum": ["NOUN", "VERB", "ADJ", "ADV"]
                }
            },
            "max_synonyms": {
                "type": "integer",
                "description": "Maximum number of synonyms to use for query expansion",
                "required": False,
                "minimum": 1,
                "maximum": 10
            },
            "no_results_message": {
                "type": "string",
                "description": "Message to return when no results are found",
                "default": "I couldn't find any relevant information for '{query}' in the knowledge base. Try rephrasing your question or asking about a different topic.",
                "required": False
            }
        })
        return schema
    
    def get_instance_key(self) -> str:
        """
        Get the key used to track this skill instance
        
        For DataSphere, we use 'search_knowledge' as the default tool name instead of 'datasphere'
        """
        tool_name = self.params.get('tool_name', 'search_knowledge')
        return f"{self.SKILL_NAME}_{tool_name}"
    
    def setup(self) -> bool:
        """Setup the datasphere skill"""
        # Validate required parameters
        required_params = ['space_name', 'project_id', 'token', 'document_id']
        missing_params = [param for param in required_params if not self.params.get(param)]
        if missing_params:
            self.logger.error(f"Missing required parameters: {missing_params}")
            return False
        
        # Set required parameters
        self.space_name = self.params['space_name']
        self.project_id = self.params['project_id'] 
        self.token = self.params['token']
        self.document_id = self.params['document_id']
        
        # Set optional parameters with defaults
        self.count = self.params.get('count', 1)
        self.distance = self.params.get('distance', 3.0)
        self.tags = self.params.get('tags', None)  # None means don't include in request
        self.language = self.params.get('language', None)  # None means don't include in request
        self.pos_to_expand = self.params.get('pos_to_expand', None)  # None means don't include in request
        self.max_synonyms = self.params.get('max_synonyms', None)  # None means don't include in request
        
        # Tool name (for multiple instances)
        self.tool_name = self.params.get('tool_name', 'search_knowledge')
        
        # No results message
        self.no_results_message = self.params.get('no_results_message',
            "I couldn't find any relevant information for '{query}' in the knowledge base. "
            "Try rephrasing your question or asking about a different topic."
        )
        
        # Build API URL
        self.api_url = f"https://{self.space_name}.signalwire.com/api/datasphere/documents/search"
        
        # Setup session for requests
        self.session = requests.Session()
        
        return True
        
    def register_tools(self) -> None:
        """Register knowledge search tool with the agent"""
        self.define_tool(
            name=self.tool_name,
            description="Search the knowledge base for information on any topic and return relevant results",
            parameters={
                "query": {
                    "type": "string",
                    "description": "The search query - what information you're looking for in the knowledge base"
                }
            },
            handler=self._search_knowledge_handler
        )
        
    def _search_knowledge_handler(self, args, raw_data):
        """Handler for knowledge search tool"""
        query = args.get("query", "").strip()
        
        if not query:
            return SwaigFunctionResult(
                "Please provide a search query. What would you like me to search for in the knowledge base?"
            )
        
        self.logger.info(f"DataSphere search requested: '{query}' (document: {self.document_id})")
        
        # Build request payload
        payload = {
            "document_id": self.document_id,
            "query_string": query,
            "distance": self.distance,
            "count": self.count
        }
        
        # Add optional parameters only if they were provided
        if self.tags is not None:
            payload["tags"] = self.tags
        if self.language is not None:
            payload["language"] = self.language
        if self.pos_to_expand is not None:
            payload["pos_to_expand"] = self.pos_to_expand
        if self.max_synonyms is not None:
            payload["max_synonyms"] = self.max_synonyms
        
        try:
            # Make API request
            response = self.session.post(
                self.api_url,
                auth=(self.project_id, self.token),
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Check if we have valid response data
            if not data or not isinstance(data, dict):
                self.logger.warning(f"DataSphere API returned invalid data: {data}")
                formatted_message = self.no_results_message.format(query=query) if '{query}' in self.no_results_message else self.no_results_message
                return SwaigFunctionResult(formatted_message)
            
            # Extract results - DataSphere API returns 'chunks', not 'results'
            chunks = data.get('chunks', [])
            
            if not chunks:
                formatted_message = self.no_results_message.format(query=query) if '{query}' in self.no_results_message else self.no_results_message
                return SwaigFunctionResult(formatted_message)
            
            # Format the results
            formatted_results = self._format_search_results(query, chunks)
            return SwaigFunctionResult(formatted_results)
            
        except requests.exceptions.Timeout:
            self.logger.error("DataSphere API request timed out")
            return SwaigFunctionResult(
                "Sorry, the knowledge search timed out. Please try again."
            )
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"DataSphere API HTTP error: {e}")
            return SwaigFunctionResult(
                "Sorry, there was an error accessing the knowledge base. Please try again later."
            )
        except Exception as e:
            self.logger.error(f"Error performing DataSphere search: {e}")
            return SwaigFunctionResult(
                "Sorry, I encountered an error while searching the knowledge base. Please try again later."
            )
    
    def _format_search_results(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Format search results for display"""
        if len(chunks) == 1:
            result_text = f"I found 1 result for '{query}':\n\n"
        else:
            result_text = f"I found {len(chunks)} results for '{query}':\n\n"
        
        formatted_results = []
        
        for i, chunk in enumerate(chunks, 1):
            result_content = f"=== RESULT {i} ===\n"
            
            # DataSphere API returns chunks with 'text' field
            if 'text' in chunk:
                result_content += chunk['text']
            elif 'content' in chunk:
                result_content += chunk['content']
            elif 'chunk' in chunk:
                result_content += chunk['chunk']
            else:
                # Fallback to the entire result as JSON if we don't recognize the format
                result_content += json.dumps(chunk, indent=2)
            
            result_content += f"\n{'='*50}\n\n"
            formatted_results.append(result_content)
        
        return result_text + '\n'.join(formatted_results)
        
    def get_hints(self) -> List[str]:
        """Return speech recognition hints"""
        # Currently no hints provided, but you could add them like:
        # return [
        #     "knowledge", "search", "information", "database", "find",
        #     "look up", "research", "query", "datasphere", "document"
        # ]
        return []
        
    def get_global_data(self) -> Dict[str, Any]:
        """Return global data for agent context"""
        return {
            "datasphere_enabled": True,
            "document_id": self.document_id,
            "knowledge_provider": "SignalWire DataSphere"
        }
        
    def get_prompt_sections(self) -> List[Dict[str, Any]]:
        """Return prompt sections to add to agent"""
        return [
            {
                "title": "Knowledge Search Capability",
                "body": f"You can search a knowledge base for information using the {self.tool_name} tool.",
                "bullets": [
                    f"Use the {self.tool_name} tool when users ask for information that might be in the knowledge base",
                    "Search for relevant information using clear, specific queries",
                    "Summarize search results in a clear, helpful way",
                    "If no results are found, suggest the user try rephrasing their question"
                ]
            }
        ] 