"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import os
import tempfile
import shutil
from typing import Dict, Any, Optional, List
from pathlib import Path

from signalwire_agents.core.skill_base import SkillBase
from signalwire_agents.core.function_result import SwaigFunctionResult

class NativeVectorSearchSkill(SkillBase):
    """Native vector search capability using local document indexes or remote search servers"""
    
    SKILL_NAME = "native_vector_search"
    SKILL_DESCRIPTION = "Search document indexes using vector similarity and keyword search (local or remote)"
    SKILL_VERSION = "1.0.0"
    REQUIRED_PACKAGES = []  # Optional packages checked at runtime
    REQUIRED_ENV_VARS = []  # No required env vars since all config comes from params
    
    # Enable multiple instances support
    SUPPORTS_MULTIPLE_INSTANCES = True
    
    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Get parameter schema for Native Vector Search skill
        
        This skill supports three modes of operation:
        1. Network Mode: Set 'remote_url' to connect to a remote search server
        2. Local pgvector: Set backend='pgvector' with connection_string and collection_name
        3. Local SQLite: Set 'index_file' to use a local .swsearch file (default)
        """
        schema = super().get_parameter_schema()
        schema.update({
            "index_file": {
                "type": "string",
                "description": "Path to .swsearch index file (SQLite backend only). Use this for local file-based search",
                "required": False
            },
            "build_index": {
                "type": "boolean",
                "description": "Whether to build index from source files",
                "default": False,
                "required": False
            },
            "source_dir": {
                "type": "string",
                "description": "Directory containing documents to index (required if build_index=True)",
                "required": False
            },
            "remote_url": {
                "type": "string",
                "description": "URL of remote search server for network mode (e.g., http://localhost:8001). Use this instead of index_file or pgvector for centralized search",
                "required": False
            },
            "index_name": {
                "type": "string",
                "description": "Name of index on remote server (network mode only, used with remote_url)",
                "default": "default",
                "required": False
            },
            "count": {
                "type": "integer",
                "description": "Number of search results to return",
                "default": 5,
                "required": False,
                "minimum": 1,
                "maximum": 20
            },
            "similarity_threshold": {
                "type": "number",
                "description": "Minimum similarity score for results (0.0 = no limit, 1.0 = exact match)",
                "default": 0.0,
                "required": False,
                "minimum": 0.0,
                "maximum": 1.0
            },
            "tags": {
                "type": "array",
                "description": "Tags to filter search results",
                "default": [],
                "required": False,
                "items": {
                    "type": "string"
                }
            },
            "global_tags": {
                "type": "array",
                "description": "Tags to apply to all indexed documents",
                "default": [],
                "required": False,
                "items": {
                    "type": "string"
                }
            },
            "file_types": {
                "type": "array",
                "description": "File extensions to include when building index",
                "default": ["md", "txt", "pdf", "docx", "html"],
                "required": False,
                "items": {
                    "type": "string"
                }
            },
            "exclude_patterns": {
                "type": "array",
                "description": "Patterns to exclude when building index",
                "default": ["**/node_modules/**", "**/.git/**", "**/dist/**", "**/build/**"],
                "required": False,
                "items": {
                    "type": "string"
                }
            },
            "no_results_message": {
                "type": "string",
                "description": "Message when no results are found",
                "default": "No information found for '{query}'",
                "required": False
            },
            "response_prefix": {
                "type": "string",
                "description": "Prefix to add to search results",
                "default": "",
                "required": False
            },
            "response_postfix": {
                "type": "string",
                "description": "Postfix to add to search results",
                "default": "",
                "required": False
            },
            "max_content_length": {
                "type": "integer",
                "description": "Maximum total response size in characters (distributed across all results)",
                "default": 32768,
                "required": False,
                "minimum": 1000
            },
            "response_format_callback": {
                "type": "callable",
                "description": "Optional callback function to format/transform the response. Called with (response, agent, query, results, args). Must return a string.",
                "required": False
            },
            "description": {
                "type": "string",
                "description": "Tool description",
                "default": "Search the knowledge base for information",
                "required": False
            },
            "hints": {
                "type": "array",
                "description": "Speech recognition hints",
                "default": [],
                "required": False,
                "items": {
                    "type": "string"
                }
            },
            "nlp_backend": {
                "type": "string",
                "description": "NLP backend for query processing",
                "default": "basic",
                "required": False,
                "enum": ["basic", "spacy", "nltk"]
            },
            "query_nlp_backend": {
                "type": "string",
                "description": "NLP backend for query expansion",
                "required": False,
                "enum": ["basic", "spacy", "nltk"]
            },
            "index_nlp_backend": {
                "type": "string",
                "description": "NLP backend for indexing",
                "required": False,
                "enum": ["basic", "spacy", "nltk"]
            },
            "backend": {
                "type": "string",
                "description": "Storage backend for local database mode: 'sqlite' for file-based or 'pgvector' for PostgreSQL. Ignored if remote_url is set",
                "default": "sqlite",
                "required": False,
                "enum": ["sqlite", "pgvector"]
            },
            "connection_string": {
                "type": "string",
                "description": "PostgreSQL connection string (pgvector backend only, e.g., 'postgresql://user:pass@localhost:5432/dbname'). Required when backend='pgvector'",
                "required": False
            },
            "collection_name": {
                "type": "string",
                "description": "Collection/table name in PostgreSQL (pgvector backend only). Required when backend='pgvector'",
                "required": False
            },
            "verbose": {
                "type": "boolean",
                "description": "Enable verbose logging",
                "default": False,
                "required": False
            },
            "keyword_weight": {
                "type": "number",
                "description": "Manual keyword weight (0.0-1.0). Overrides automatic weight detection",
                "default": None,
                "required": False,
                "minimum": 0.0,
                "maximum": 1.0
            },
            "model_name": {
                "type": "string",
                "description": "Embedding model to use. Options: 'mini' (fastest, 384 dims), 'base' (balanced, 768 dims), 'large' (same as base). Or specify full model name like 'sentence-transformers/all-MiniLM-L6-v2'",
                "default": "mini",
                "required": False
            },
            "overwrite": {
                "type": "boolean",
                "description": "Overwrite existing pgvector collection when building index (pgvector backend only)",
                "default": False,
                "required": False
            }
        })
        return schema
    
    def get_instance_key(self) -> str:
        """
        Get the key used to track this skill instance
        
        For native vector search, we use the tool name to differentiate instances
        """
        tool_name = self.params.get('tool_name', 'search_knowledge')
        index_file = self.params.get('index_file', 'default')
        return f"{self.SKILL_NAME}_{tool_name}_{index_file}"
    
    def setup(self) -> bool:
        """Setup the native vector search skill"""
        
        # Get configuration first
        self.tool_name = self.params.get('tool_name', 'search_knowledge')
        self.backend = self.params.get('backend', 'sqlite')
        self.connection_string = self.params.get('connection_string')
        self.collection_name = self.params.get('collection_name')
        self.index_file = self.params.get('index_file')
        self.build_index = self.params.get('build_index', False)
        self.source_dir = self.params.get('source_dir')
        self.count = self.params.get('count', 5)
        self.similarity_threshold = self.params.get('similarity_threshold', 0.0)
        self.tags = self.params.get('tags', [])
        self.no_results_message = self.params.get(
            'no_results_message', 
            "No information found for '{query}'"
        )
        self.response_prefix = self.params.get('response_prefix', '')
        self.response_postfix = self.params.get('response_postfix', '')
        self.max_content_length = self.params.get('max_content_length', 32768)
        self.response_format_callback = self.params.get('response_format_callback')
        self.keyword_weight = self.params.get('keyword_weight')
        self.model_name = self.params.get('model_name', 'mini')
        
        # Remote search server configuration
        self.remote_url = self.params.get('remote_url')  # e.g., "http://user:pass@localhost:8001"
        self.index_name = self.params.get('index_name', 'default')  # For remote searches
        
        # Parse auth from URL if present
        self.remote_auth = None
        self.remote_base_url = self.remote_url
        if self.remote_url:
            from urllib.parse import urlparse
            parsed = urlparse(self.remote_url)
            if parsed.username and parsed.password:
                self.remote_auth = (parsed.username, parsed.password)
                # Reconstruct URL without auth for display
                self.remote_base_url = f"{parsed.scheme}://{parsed.hostname}"
                if parsed.port:
                    self.remote_base_url += f":{parsed.port}"
                if parsed.path:
                    self.remote_base_url += parsed.path
        
        # SWAIG fields are already extracted by SkillBase.__init__()
        # No need to re-fetch from params - use self.swaig_fields inherited from parent
        
        # **EARLY REMOTE CHECK - Option 1**
        # If remote URL is configured, skip all heavy local imports and just validate remote connectivity
        if self.remote_url:
            self.use_remote = True
            self.search_engine = None  # No local search engine needed
            self.logger.info(f"Using remote search server: {self.remote_url}")
            
            # Test remote connection (lightweight check)
            try:
                import requests
                # Use parsed base URL and auth
                response = requests.get(
                    f"{self.remote_base_url}/health", 
                    auth=self.remote_auth,
                    timeout=5
                )
                if response.status_code == 200:
                    self.logger.info(f"Remote search server is available at {self.remote_base_url}")
                    self.search_available = True
                    return True  # Success - skip all local setup
                elif response.status_code == 401:
                    self.logger.error("Authentication failed for remote search server. Check credentials.")
                    self.search_available = False
                    return False
                else:
                    self.logger.error(f"Remote search server returned status {response.status_code}")
                    self.search_available = False
                    return False
            except Exception as e:
                self.logger.error(f"Failed to connect to remote search server: {e}")
                self.search_available = False
                return False
        
        # **LOCAL MODE SETUP - Only when no remote URL**
        self.use_remote = False
        
        # NLP backend configuration (only needed for local mode)
        self.nlp_backend = self.params.get('nlp_backend')  # Backward compatibility
        self.index_nlp_backend = self.params.get('index_nlp_backend', 'nltk')  # Default to fast NLTK for indexing
        self.query_nlp_backend = self.params.get('query_nlp_backend', 'nltk')  # Default to fast NLTK for search
        
        # Handle backward compatibility
        if self.nlp_backend is not None:
            self.logger.warning("Parameter 'nlp_backend' is deprecated. Use 'index_nlp_backend' and 'query_nlp_backend' instead.")
            # If old parameter is used, apply it to both
            self.index_nlp_backend = self.nlp_backend
            self.query_nlp_backend = self.nlp_backend
        
        # Validate parameters
        if self.index_nlp_backend not in ['nltk', 'spacy']:
            self.logger.warning(f"Invalid index_nlp_backend '{self.index_nlp_backend}', using 'nltk'")
            self.index_nlp_backend = 'nltk'
            
        if self.query_nlp_backend not in ['nltk', 'spacy']:
            self.logger.warning(f"Invalid query_nlp_backend '{self.query_nlp_backend}', using 'nltk'")
            self.query_nlp_backend = 'nltk'
        
        # Check if local search functionality is available (heavy imports only for local mode)
        try:
            from signalwire_agents.search import IndexBuilder, SearchEngine
            from signalwire_agents.search.query_processor import preprocess_query
            self.search_available = True
        except ImportError as e:
            self.search_available = False
            self.import_error = str(e)
            self.logger.warning(f"Search dependencies not available: {e}")
            # Don't fail setup - we'll provide helpful error messages at runtime
        
        # Auto-build index if requested and search is available
        if self.build_index and self.source_dir and self.search_available:
            # Handle auto-build for different backends
            if self.backend == 'sqlite':
                if not self.index_file:
                    # Generate index filename from source directory
                    source_name = Path(self.source_dir).name
                    self.index_file = f"{source_name}.swsearch"
                
                # Build index if it doesn't exist
                if not os.path.exists(self.index_file):
                    try:
                        self.logger.info(f"Building search index from {self.source_dir}...")
                        from signalwire_agents.search import IndexBuilder
                        
                        # Resolve model alias if needed
                        from signalwire_agents.search.models import resolve_model_alias
                        model_to_use = resolve_model_alias(self.model_name)
                        
                        builder = IndexBuilder(
                            model_name=model_to_use,
                            verbose=self.params.get('verbose', False),
                            index_nlp_backend=self.index_nlp_backend
                        )
                        builder.build_index(
                            source_dir=self.source_dir,
                            output_file=self.index_file,
                            file_types=self.params.get('file_types', ['md', 'txt']),
                            exclude_patterns=self.params.get('exclude_patterns'),
                            tags=self.params.get('global_tags')
                        )
                        self.logger.info(f"Search index created: {self.index_file}")
                    except Exception as e:
                        self.logger.error(f"Failed to build search index: {e}")
                        self.search_available = False
                        
            elif self.backend == 'pgvector':
                # Auto-build for pgvector
                if self.connection_string and self.collection_name:
                    try:
                        self.logger.info(f"Building pgvector index from {self.source_dir}...")
                        from signalwire_agents.search import IndexBuilder
                        from signalwire_agents.search.models import resolve_model_alias
                        
                        model_to_use = resolve_model_alias(self.model_name)
                        
                        builder = IndexBuilder(
                            backend='pgvector',
                            connection_string=self.connection_string,
                            model_name=model_to_use,
                            verbose=self.params.get('verbose', False),
                            index_nlp_backend=self.index_nlp_backend
                        )
                        
                        builder.build_index(
                            source_dir=self.source_dir,
                            output_file=self.collection_name,  # pgvector uses this as collection name
                            file_types=self.params.get('file_types', ['md', 'txt']),
                            exclude_patterns=self.params.get('exclude_patterns'),
                            tags=self.params.get('global_tags'),
                            overwrite=self.params.get('overwrite', False)
                        )
                        self.logger.info(f"pgvector collection created: {self.collection_name}")
                    except Exception as e:
                        self.logger.error(f"Failed to build pgvector index: {e}")
                        # Don't set search_available to False - we might be connecting to existing collection
                else:
                    self.logger.warning("pgvector auto-build requires connection_string and collection_name")
        
        # Initialize local search engine
        self.search_engine = None
        if self.search_available:
            if self.backend == 'pgvector':
                # Initialize pgvector backend
                if self.connection_string and self.collection_name:
                    try:
                        from signalwire_agents.search import SearchEngine
                        self.search_engine = SearchEngine(
                            backend='pgvector',
                            connection_string=self.connection_string,
                            collection_name=self.collection_name
                        )
                        self.logger.info(f"Connected to pgvector collection: {self.collection_name}")
                    except Exception as e:
                        self.logger.error(f"Failed to connect to pgvector: {e}")
                        self.search_available = False
                else:
                    self.logger.error("pgvector backend requires connection_string and collection_name")
                    self.search_available = False
            elif self.index_file and os.path.exists(self.index_file):
                # Initialize SQLite backend
                try:
                    from signalwire_agents.search import SearchEngine
                    self.search_engine = SearchEngine(backend='sqlite', index_path=self.index_file)
                    # The SearchEngine will auto-detect the model from the index
                    # Get the model name from config for query processing
                    if hasattr(self.search_engine, 'config'):
                        index_model = self.search_engine.config.get('embedding_model')
                        if index_model:
                            self.logger.info(f"Using model from index: {index_model}")
                except Exception as e:
                    self.logger.error(f"Failed to load search index {self.index_file}: {e}")
                    self.search_available = False
        
        return True
        
    def register_tools(self) -> None:
        """Register native vector search tool with the agent"""
        
        # Get description from params or use default
        description = self.params.get(
            'description', 
            'Search the local knowledge base for information'
        )
        
        self.define_tool(
            name=self.tool_name,
            description=description,
            parameters={
                "query": {
                    "type": "string",
                    "description": "Search query or question"
                },
                "count": {
                    "type": "integer",
                    "description": f"Number of results to return (default: {self.count})",
                    "default": self.count
                }
            },
            handler=self._search_handler
        )
        
        # Add our tool to the Knowledge Search section
        search_mode = "remote search server" if self.use_remote else "local document indexes"
        section_title = "Knowledge Search"
        
        # Try to check if section exists, but handle if method doesn't exist
        section_exists = False
        try:
            if hasattr(self.agent, 'prompt_has_section'):
                section_exists = self.agent.prompt_has_section(section_title)
        except Exception:
            # Method might not work, assume section doesn't exist
            pass
        
        if section_exists:
            # Add bullet to existing section
            self.agent.prompt_add_to_section(
                title=section_title,
                bullet=f"Use {self.tool_name} to search {search_mode}: {description}"
            )
        else:
            # Create the section with this tool
            self.agent.prompt_add_section(
                title=section_title,
                body="You can search various knowledge sources using the following tools:",
                bullets=[
                    f"Use {self.tool_name} to search {search_mode}: {description}",
                    "Search for relevant information using clear, specific queries",
                    "If no results are found, suggest the user try rephrasing their question or try another knowledge source"
                ]
            )
        
    def _search_handler(self, args, raw_data):
        """Handle search requests"""
        
        # Debug logging to see what arguments are being passed
        self.logger.info(f"Search handler called with args: {args}")
        self.logger.info(f"Args type: {type(args)}")
        self.logger.info(f"Raw data: {raw_data}")
        
        if not self.search_available:
            return SwaigFunctionResult(
                f"Search functionality is not available. {getattr(self, 'import_error', '')}\n"
                f"Install with: pip install signalwire-agents[search]"
            )
        
        if not self.use_remote and not self.search_engine:
            return SwaigFunctionResult(
                f"Search index not available. "
                f"{'Index file not found: ' + (self.index_file or 'not specified') if self.index_file else 'No index file configured'}"
            )
        
        # Get arguments - the framework handles parsing correctly
        query = args.get('query', '').strip()
        self.logger.error(f"DEBUG: Extracted query: '{query}' (length: {len(query)})")
        self.logger.info(f"Query bool value: {bool(query)}")
        
        if not query:
            self.logger.error(f"Query validation failed - returning error message")
            return SwaigFunctionResult("Please provide a search query.")
        
        self.logger.info(f"Query validation passed - proceeding with search")
        count = args.get('count', self.count)
        
        try:
            # Perform search (local or remote)
            self.logger.info(f"DEBUG: use_remote={self.use_remote}, remote_base_url={self.remote_base_url}")
            if self.use_remote:
                # For remote searches, let the server handle query preprocessing
                self.logger.info(f"DEBUG: Calling _search_remote with query='{query}', count={count}")
                results = self._search_remote(query, None, count)
                self.logger.info(f"DEBUG: _search_remote returned {len(results)} results")
            else:
                # For local searches, preprocess the query locally
                from signalwire_agents.search.query_processor import preprocess_query
                
                # Get model name from index config if available
                model_for_query = None
                if hasattr(self.search_engine, 'config'):
                    model_for_query = self.search_engine.config.get('embedding_model')
                
                enhanced = preprocess_query(
                    query, 
                    language='en', 
                    vector=True, 
                    query_nlp_backend=self.query_nlp_backend,
                    model_name=model_for_query,  # Use model from index
                    preserve_original=True,  # Keep original query terms
                    max_synonyms=2  # Reduce synonym expansion
                )
                results = self.search_engine.search(
                    query_vector=enhanced.get('vector', []),
                    enhanced_text=enhanced['enhanced_text'],
                    count=count,
                    similarity_threshold=self.similarity_threshold,
                    tags=self.tags,
                    keyword_weight=self.keyword_weight,
                    original_query=query  # Pass original for exact match boosting
                )
            
            if not results:
                no_results_msg = self.no_results_message.format(query=query)
                if self.response_prefix:
                    no_results_msg = f"{self.response_prefix} {no_results_msg}"
                if self.response_postfix:
                    no_results_msg = f"{no_results_msg} {self.response_postfix}"
                
                # Apply custom formatting callback for no results case
                if self.response_format_callback and callable(self.response_format_callback):
                    try:
                        callback_context = {
                            'response': no_results_msg,
                            'agent': self.agent,
                            'query': query,
                            'results': [],  # Empty results
                            'args': args,
                            'count': count,
                            'skill': self
                        }
                        formatted_response = self.response_format_callback(**callback_context)
                        if isinstance(formatted_response, str):
                            no_results_msg = formatted_response
                    except Exception as e:
                        self.logger.error(f"Error in response_format_callback (no results): {e}", exc_info=True)
                
                return SwaigFunctionResult(no_results_msg)
            
            # Format results with dynamic per-result truncation
            response_parts = []

            # Add response prefix if configured
            if self.response_prefix:
                response_parts.append(self.response_prefix)

            response_parts.append(f"Found {len(results)} relevant results for '{query}':\n")

            # Calculate per-result content budget
            # Estimate overhead per result: metadata (~200 chars) + formatting (~100 chars)
            estimated_overhead_per_result = 300
            # Account for prefix/postfix/header in total overhead
            prefix_postfix_overhead = len(self.response_prefix) + len(self.response_postfix) + 100
            total_overhead = (len(results) * estimated_overhead_per_result) + prefix_postfix_overhead
            available_for_content = self.max_content_length - total_overhead

            # Ensure minimum of 500 chars per result
            per_result_limit = max(500, available_for_content // len(results)) if len(results) > 0 else 1000

            for i, result in enumerate(results, 1):
                filename = result['metadata']['filename']
                section = result['metadata'].get('section', '')
                score = result['score']
                content = result['content']

                # Truncate content to per-result limit
                if len(content) > per_result_limit:
                    content = content[:per_result_limit] + "..."

                # Get tags from either top level or metadata
                tags = result.get('tags', [])
                if not tags and 'metadata' in result['metadata'] and 'tags' in result['metadata']['metadata']:
                    # Handle double-nested metadata from older indexes
                    tags = result['metadata']['metadata']['tags']
                elif not tags and 'tags' in result['metadata']:
                    # Check in metadata directly
                    tags = result['metadata']['tags']

                result_text = f"**Result {i}** (from {filename}"
                if section:
                    result_text += f", section: {section}"
                if tags:
                    result_text += f", tags: {', '.join(tags)}"
                result_text += f", relevance: {score:.2f})\n{content}\n"

                response_parts.append(result_text)

            # Add response postfix if configured
            if self.response_postfix:
                response_parts.append(self.response_postfix)
            
            # Build the initial response
            response = "\n".join(response_parts)
            
            # Apply custom formatting callback if provided
            if self.response_format_callback and callable(self.response_format_callback):
                try:
                    # Prepare callback context
                    callback_context = {
                        'response': response,
                        'agent': self.agent,
                        'query': query,
                        'results': results,
                        'args': args,
                        'count': count,
                        'skill': self
                    }
                    
                    # Call the callback
                    formatted_response = self.response_format_callback(**callback_context)
                    
                    # Validate callback returned a string
                    if isinstance(formatted_response, str):
                        response = formatted_response
                    else:
                        self.logger.warning(f"response_format_callback returned non-string type: {type(formatted_response)}")
                        
                except Exception as e:
                    self.logger.error(f"Error in response_format_callback: {e}", exc_info=True)
                    # Continue with original response if callback fails
            
            return SwaigFunctionResult(response)
            
        except Exception as e:
            # Log the full error details for debugging
            self.logger.error(f"Search error for query '{query}': {str(e)}", exc_info=True)
            
            # Return user-friendly error message
            user_msg = "I'm sorry, I encountered an issue while searching. "
            
            # Check for specific error types and provide helpful guidance
            error_str = str(e).lower()
            if 'punkt' in error_str or 'nltk' in error_str:
                user_msg += "It looks like some language processing resources are missing. Please try again in a moment."
            elif 'vector' in error_str or 'embedding' in error_str:
                user_msg += "There was an issue with the search indexing. Please try rephrasing your question."
            elif 'timeout' in error_str or 'connection' in error_str:
                user_msg += "The search service is temporarily unavailable. Please try again later."
            else:
                user_msg += "Please try rephrasing your question or contact support if the issue persists."
                
            return SwaigFunctionResult(user_msg)
    
    def _search_remote(self, query: str, enhanced: dict, count: int) -> list:
        """Perform search using remote search server"""
        try:
            import requests
            
            search_request = {
                "query": query,
                "index_name": self.index_name,
                "count": count,
                "similarity_threshold": self.similarity_threshold,
                "tags": self.tags
            }

            url = f"{self.remote_base_url}/search"
            self.logger.info(f"DEBUG: Sending POST to {url} with request: {search_request}")

            response = requests.post(
                url,
                json=search_request,
                auth=self.remote_auth,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                self.logger.info(f"DEBUG: Got response with {len(data.get('results', []))} results")
                # Convert remote response format to local format
                results = []
                for result in data.get('results', []):
                    results.append({
                        'content': result['content'],
                        'score': result['score'],
                        'metadata': result['metadata']
                    })
                return results
            else:
                self.logger.error(f"Remote search failed with status {response.status_code}: {response.text}")
                return []
                
        except Exception as e:
            self.logger.error(f"Remote search error: {e}")
            return []
    
    def get_hints(self) -> List[str]:
        """Return speech recognition hints for this skill"""
        hints = [
            "search",
            "find",
            "look up",
            "documentation",
            "knowledge base"
        ]
        
        # Add custom hints from params
        custom_hints = self.params.get('hints', [])
        hints.extend(custom_hints)
        
        return hints
        
    def get_global_data(self) -> Dict[str, Any]:
        """Return data to add to agent's global context"""
        global_data = {}
        
        if self.search_engine:
            try:
                stats = self.search_engine.get_stats()
                global_data['search_stats'] = stats
            except:
                pass
        
        return global_data
        
    def get_prompt_sections(self) -> List[Dict[str, Any]]:
        """Return prompt sections to add to agent"""
        # We'll handle this in register_tools after the agent is set
        return []
    
    def _add_prompt_section(self, agent):
        """Add prompt section to agent (called during skill loading)"""
        try:
            agent.prompt_add_section(
                title="Local Document Search",
                body=f"You can search local document indexes using the {self.tool_name} tool.",
                bullets=[
                    f"Use the {self.tool_name} tool when users ask questions about topics that might be in the indexed documents",
                    "Search for relevant information using clear, specific queries", 
                    "Provide helpful summaries of the search results",
                    "If no results are found, suggest the user try rephrasing their question or ask about different topics"
                ]
            )
        except Exception as e:
            self.logger.error(f"Failed to add prompt section: {e}")
            # Continue without the prompt section
    
    def cleanup(self) -> None:
        """Cleanup when skill is removed or agent shuts down"""
        # Clean up any temporary files if we created them
        if hasattr(self, '_temp_dirs'):
            for temp_dir in self._temp_dirs:
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass 