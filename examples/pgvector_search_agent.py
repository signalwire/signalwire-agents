#!/usr/bin/env python3
"""
PGVector Search Agent Example

This example demonstrates how to use pgvector backend for document search
within a SignalWire AI Agent using the native_vector_search skill.

Prerequisites:
1. PostgreSQL with pgvector extension running
2. A collection already created in the database
3. Required packages: pip install signalwire-agents[search-all]

To create a collection:
    sw-search ./docs --backend pgvector \
      --connection-string "postgresql://signalwire:signalwire123@localhost:5432/knowledge" \
      --output docs_collection
"""

import os
from signalwire_agents import AgentBase

class PGVectorSearchAgent(AgentBase):
    """Agent that uses pgvector for document search"""
    
    def __init__(self):
        super().__init__()
        
        # Set agent name and description
        self.agent_name = "PGVector Search Assistant"
        
        # Configure the main prompt
        self.prompt_add_section(
            "Role",
            "You are a helpful search assistant with access to a document knowledge base. "
            "When users ask questions, search the knowledge base to find relevant information. "
            "Provide detailed answers based on the search results."
        )
        
        # Add native vector search skill with pgvector backend
        self.add_skill("native_vector_search", {
            "tool_name": "search_docs",
            "description": "Search the document knowledge base for information",
            "backend": "pgvector",
            "connection_string": os.getenv("PGVECTOR_CONNECTION_STRING", 
                "postgresql://signalwire:signalwire123@localhost:5432/knowledge"),
            "collection_name": os.getenv("PGVECTOR_COLLECTION", "docs_collection"),
            "count": 5,
            "similarity_threshold": 0.7,
            "no_results_message": "I couldn't find any relevant information about '{query}' in the knowledge base.",
            "response_prefix": "Based on my search of the knowledge base:\n\n",
            "response_postfix": "\n\nIs there anything specific about this topic you'd like to know more about?"
        })
        
        # Optional: Add a second collection for different content
        if os.getenv("ENABLE_API_SEARCH", "false").lower() == "true":
            self.add_skill("native_vector_search", {
                "tool_name": "search_api",
                "description": "Search API documentation",
                "backend": "pgvector",
                "connection_string": os.getenv("PGVECTOR_CONNECTION_STRING", 
                    "postgresql://signalwire:signalwire123@localhost:5432/knowledge"),
                "collection_name": "api_docs",
                "count": 3,
                "tags": ["api", "reference"],  # Filter by tags
                "response_prefix": "From the API documentation:\n\n"
            })
    
    @AgentBase.tool("list_collections")
    def list_available_collections(self):
        """List available search collections"""
        collections = ["docs_collection"]
        if os.getenv("ENABLE_API_SEARCH", "false").lower() == "true":
            collections.append("api_docs")
        
        return {
            "result": "success",
            "collections": collections,
            "message": f"Available collections: {', '.join(collections)}"
        }
    
    @AgentBase.tool("search_with_metadata")
    def search_with_metadata_filter(self, query: str, category: str = None, tags: list = None):
        """Advanced search with metadata filtering (requires custom implementation)"""
        # This is a placeholder showing how you might implement custom search
        # In practice, you'd use the SearchEngine directly for advanced queries
        return {
            "result": "info",
            "message": "This would perform an advanced search with metadata filters",
            "query": query,
            "filters": {
                "category": category,
                "tags": tags
            }
        }

if __name__ == "__main__":
    # Example of running the agent
    agent = PGVectorSearchAgent()
    
    # Print configuration info
    print("PGVector Search Agent")
    print("=" * 50)
    print(f"Connection: {os.getenv('PGVECTOR_CONNECTION_STRING', 'postgresql://signalwire:signalwire123@localhost:5432/knowledge')}")
    print(f"Collection: {os.getenv('PGVECTOR_COLLECTION', 'docs_collection')}")
    print(f"API Search: {'Enabled' if os.getenv('ENABLE_API_SEARCH', 'false').lower() == 'true' else 'Disabled'}")
    print("=" * 50)
    print("\nStarting agent server on http://localhost:8000")
    print("\nEnvironment variables you can set:")
    print("- PGVECTOR_CONNECTION_STRING: PostgreSQL connection string")
    print("- PGVECTOR_COLLECTION: Main collection name")
    print("- ENABLE_API_SEARCH: Enable API documentation search")
    print("\nPress Ctrl+C to stop")
    
    # Serve the agent
    agent.serve()