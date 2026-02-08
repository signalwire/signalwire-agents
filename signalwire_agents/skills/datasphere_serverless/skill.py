"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import base64
from typing import Optional, List, Dict, Any

from signalwire_agents.core.skill_base import SkillBase
from signalwire_agents.core.data_map import DataMap
from signalwire_agents.core.function_result import SwaigFunctionResult

class DataSphereServerlessSkill(SkillBase):
    """SignalWire DataSphere knowledge search using DataMap (serverless execution)"""
    
    SKILL_NAME = "datasphere_serverless"
    SKILL_DESCRIPTION = "Search knowledge using SignalWire DataSphere with serverless DataMap execution"
    SKILL_VERSION = "1.0.0"
    REQUIRED_PACKAGES = []  # DataMap handles API calls serverlessly
    REQUIRED_ENV_VARS = []  # No required env vars since all config comes from params
    
    # Enable multiple instances support
    SUPPORTS_MULTIPLE_INSTANCES = True
    
    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Get parameter schema for DataSphere Serverless skill"""
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
        
        For DataSphere Serverless, we use 'search_knowledge' as the default tool name
        """
        tool_name = self.params.get('tool_name', 'search_knowledge')
        return f"{self.SKILL_NAME}_{tool_name}"
    
    def setup(self) -> bool:
        """Setup the datasphere serverless skill"""
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
        self.tags = self.params.get('tags', None)  
        self.language = self.params.get('language', None)  
        self.pos_to_expand = self.params.get('pos_to_expand', None)  
        self.max_synonyms = self.params.get('max_synonyms', None)  
        
        # Tool name (for multiple instances)
        self.tool_name = self.params.get('tool_name', 'search_knowledge')
        
        # No results message
        self.no_results_message = self.params.get('no_results_message',
            "I couldn't find any relevant information for '{query}' in the knowledge base. "
            "Try rephrasing your question or asking about a different topic."
        )
        
        # Build API URL
        self.api_url = f"https://{self.space_name}.signalwire.com/api/datasphere/documents/search"
        
        # Build auth header for DataMap
        auth_string = f"{self.project_id}:{self.token}"
        self.auth_header = base64.b64encode(auth_string.encode()).decode()
        
        return True
        
    def register_tools(self) -> None:
        """Register knowledge search tool using DataMap"""
        
        # Build webhook params with configuration values
        webhook_params = {
            "document_id": self.document_id,
            "query_string": "${args.query}",  # Only this is dynamic from user input
            "count": self.count,
            "distance": self.distance
        }
        
        # Add optional parameters only if they were provided
        if self.tags is not None:
            webhook_params["tags"] = self.tags
        if self.language is not None:
            webhook_params["language"] = self.language
        if self.pos_to_expand is not None:
            webhook_params["pos_to_expand"] = self.pos_to_expand
        if self.max_synonyms is not None:
            webhook_params["max_synonyms"] = self.max_synonyms
        
        # Create DataMap tool for DataSphere search
        datasphere_tool = (DataMap(self.tool_name)
            .description("Search the knowledge base for information on any topic and return relevant results")
            .parameter('query', 'string', 'The search query - what information you\'re looking for in the knowledge base', required=True)
            .webhook('POST', self.api_url,
                     headers={
                         'Content-Type': 'application/json',
                         'Authorization': f'Basic {self.auth_header}'
                     })
            .params(webhook_params)
            .foreach({
                "input_key": "chunks",
                "output_key": "formatted_results",
                "max": self.count,
                "append": "=== RESULT ===\n${this.text}\n" + "="*50 + "\n\n"
            })
            .output(SwaigFunctionResult('I found results for "${args.query}":\n\n${formatted_results}'))
            .error_keys(['error'])
            .fallback_output(SwaigFunctionResult(self.no_results_message.replace('{query}', '${args.query}')))
        )
        
        # Convert DataMap to SWAIG function and apply swaig_fields
        swaig_function = datasphere_tool.to_swaig_function()
        
        # Merge swaig_fields from skill params into the function definition
        swaig_function.update(self.swaig_fields)
        
        # Register the enhanced DataMap tool with the agent
        self.agent.register_swaig_function(swaig_function)
        
    def get_hints(self) -> List[str]:
        """Return speech recognition hints"""
        return []
        
    def get_global_data(self) -> Dict[str, Any]:
        """Return global data for agent context"""
        return {
            "datasphere_serverless_enabled": True,
            "document_id": self.document_id,
            "knowledge_provider": "SignalWire DataSphere (Serverless)"
        }
        
    def get_prompt_sections(self) -> List[Dict[str, Any]]:
        """Return prompt sections to add to agent"""
        return [
            {
                "title": "Knowledge Search Capability (Serverless)",
                "body": f"You can search a knowledge base for information using the {self.tool_name} tool.",
                "bullets": [
                    f"Use the {self.tool_name} tool when users ask for information that might be in the knowledge base",
                    "Search for relevant information using clear, specific queries",
                    "Summarize search results in a clear, helpful way",
                    "If no results are found, suggest the user try rephrasing their question",
                    "This tool executes on SignalWire servers for optimal performance"
                ]
            }
        ] 