"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

try:
    from fastapi import FastAPI, HTTPException, Request, Response, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBasic, HTTPBasicCredentials
    from pydantic import BaseModel
except ImportError:
    FastAPI = None
    HTTPException = None
    BaseModel = None
    Request = None
    Response = None
    Depends = None
    CORSMiddleware = None
    HTTPBasic = None
    HTTPBasicCredentials = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .query_processor import preprocess_query, set_global_model
from .search_engine import SearchEngine
from signalwire_agents.core.security_config import SecurityConfig
from signalwire_agents.core.config_loader import ConfigLoader
from signalwire_agents.core.logging_config import get_logger

logger = get_logger("search_service")

# Simple LRU cache for query results
from functools import lru_cache
import hashlib
import json

# Pydantic models for API
if BaseModel:
    class SearchRequest(BaseModel):
        query: str
        index_name: str = "default"
        count: int = 3
        similarity_threshold: float = 0.0
        tags: Optional[List[str]] = None
        language: Optional[str] = None

    class SearchResult(BaseModel):
        content: str
        score: float
        metadata: Dict[str, Any]

    class SearchResponse(BaseModel):
        results: List[SearchResult]
        query_analysis: Optional[Dict[str, Any]] = None
else:
    # Fallback classes when FastAPI is not available
    class SearchRequest:
        def __init__(self, query: str, index_name: str = "default", count: int = 3,
                     similarity_threshold: float = 0.0, tags: Optional[List[str]] = None,
                     language: Optional[str] = None):
            self.query = query
            self.index_name = index_name
            self.count = count
            self.similarity_threshold = similarity_threshold
            self.tags = tags
            self.language = language

    class SearchResult:
        def __init__(self, content: str, score: float, metadata: Dict[str, Any]):
            self.content = content
            self.score = score
            self.metadata = metadata

    class SearchResponse:
        def __init__(self, results: List[SearchResult], query_analysis: Optional[Dict[str, Any]] = None):
            self.results = results
            self.query_analysis = query_analysis

def _cache_key(query: str, index_name: str, count: int, tags: Optional[List[str]] = None) -> str:
    """Generate cache key for query results"""
    key_data = {
        'query': query.lower().strip(),
        'index': index_name,
        'count': count,
        'tags': sorted(tags) if tags else []
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()

class SearchService:
    """Local search service with HTTP API supporting both SQLite and pgvector backends"""
    
    def __init__(self, port: int = 8001, indexes: Dict[str, str] = None, 
                 basic_auth: Optional[Tuple[str, str]] = None,
                 config_file: Optional[str] = None,
                 backend: str = 'sqlite',
                 connection_string: Optional[str] = None):
        # Load configuration first
        self._load_config(config_file)
        
        # Override with constructor params if provided
        self.port = port
        self.backend = backend
        self.connection_string = connection_string
        
        if indexes is not None:
            self.indexes = indexes
        
        self.search_engines = {}
        self.model = None
        self._query_cache = {}  # Simple query result cache
        self._cache_size = 100  # Max number of cached queries
        
        # Load security configuration with optional config file
        self.security = SecurityConfig(config_file=config_file, service_name="search")
        self.security.log_config("SearchService")
        
        # Set up authentication
        self._basic_auth = basic_auth or self.security.get_basic_auth()
        
        if FastAPI:
            self.app = FastAPI(title="SignalWire Local Search Service")
            self._setup_security()
            self._setup_routes()
        else:
            self.app = None
            logger.warning("FastAPI not available. HTTP service will not be available.")
        
        self._load_resources()
    
    def _load_config(self, config_file: Optional[str]):
        """Load configuration from file if available"""
        # Initialize defaults
        self.indexes = {}
        self.backend = 'sqlite'
        self.connection_string = None
        
        # Find config file
        if not config_file:
            config_file = ConfigLoader.find_config_file("search")
        
        if not config_file:
            return
            
        # Load config
        config_loader = ConfigLoader([config_file])
        if not config_loader.has_config():
            return
            
        logger.info("loading_config_from_file", file=config_file)
        
        # Get service section
        service_config = config_loader.get_section('service')
        if service_config:
            if 'port' in service_config:
                self.port = int(service_config['port'])
            
            if 'backend' in service_config:
                self.backend = service_config['backend']
                
            if 'connection_string' in service_config:
                self.connection_string = service_config['connection_string']
            
            if 'indexes' in service_config and isinstance(service_config['indexes'], dict):
                self.indexes = service_config['indexes']
    
    def _setup_security(self):
        """Setup security middleware and authentication"""
        if not self.app:
            return
            
        # Add CORS middleware if FastAPI has it
        if CORSMiddleware:
            self.app.add_middleware(
                CORSMiddleware,
                **self.security.get_cors_config()
            )
        
        # Add security headers middleware
        @self.app.middleware("http")
        async def add_security_headers(request: Request, call_next):
            response = await call_next(request)
            
            # Add security headers
            is_https = request.url.scheme == "https"
            headers = self.security.get_security_headers(is_https)
            for header, value in headers.items():
                response.headers[header] = value
            
            return response
        
        # Add host validation middleware
        @self.app.middleware("http")
        async def validate_host(request: Request, call_next):
            host = request.headers.get("host", "").split(":")[0]
            if host and not self.security.should_allow_host(host):
                return Response(content="Invalid host", status_code=400)
            
            return await call_next(request)
    
    def _get_current_username(self, credentials: HTTPBasicCredentials = None) -> str:
        """Validate basic auth credentials"""
        if not credentials:
            return None
            
        correct_username, correct_password = self._basic_auth
        
        # Compare credentials
        import secrets
        username_correct = secrets.compare_digest(credentials.username, correct_username)
        password_correct = secrets.compare_digest(credentials.password, correct_password)
        
        if not (username_correct and password_correct):
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Basic"},
            )
        
        return credentials.username
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        if not self.app:
            return
        
        # Create security dependency if HTTPBasic is available
        security = HTTPBasic() if HTTPBasic else None
        
        # Create dependency for authenticated routes
        def get_authenticated():
            if security:
                return security
            return None
            
        @self.app.post("/search", response_model=SearchResponse)
        async def search(
            request: SearchRequest,
            credentials: HTTPBasicCredentials = None if not security else Depends(security)
        ):
            if security:
                self._get_current_username(credentials)
            return await self._handle_search(request)
        
        @self.app.get("/health")
        async def health():
            return {
                "status": "healthy", 
                "backend": self.backend,
                "indexes": list(self.indexes.keys()),
                "ssl_enabled": self.security.ssl_enabled,
                "auth_required": bool(security),
                "connection_string": self.connection_string if self.backend == 'pgvector' else None
            }
        
        @self.app.post("/reload_index")
        async def reload_index(
            index_name: str, 
            index_path: str,
            credentials: HTTPBasicCredentials = None if not security else Depends(security)
        ):
            """Reload or add new index/collection"""
            if security:
                self._get_current_username(credentials)
            
            if self.backend == 'pgvector':
                # For pgvector, index_path is actually the collection name
                self.indexes[index_name] = index_path
                try:
                    self.search_engines[index_name] = SearchEngine(
                        backend='pgvector',
                        connection_string=self.connection_string,
                        collection_name=index_path
                    )
                    return {"status": "reloaded", "index": index_name, "backend": "pgvector"}
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to load pgvector collection: {e}")
            else:
                # SQLite backend
                self.indexes[index_name] = index_path
                self.search_engines[index_name] = SearchEngine(
                    backend='sqlite',
                    index_path=index_path,
                    model=self.model
                )
                return {"status": "reloaded", "index": index_name, "backend": "sqlite"}
    
    def _load_resources(self):
        """Load embedding model and search indexes"""
        if self.backend == 'pgvector':
            # For pgvector, we need to load models for query embeddings
            # Different collections might use different models
            self.models = {}  # model_name -> SentenceTransformer instance
            self.collection_models = {}  # collection_name -> model_name

            # Load search engines for each collection and their models
            for collection_name in self.indexes.keys():
                try:
                    search_engine = SearchEngine(
                        backend='pgvector',
                        connection_string=self.connection_string,
                        collection_name=collection_name
                    )
                    self.search_engines[collection_name] = search_engine

                    # Get the model name from the collection config
                    model_name = search_engine.config.get('model_name')
                    if model_name:
                        self.collection_models[collection_name] = model_name

                        # Load the model if we haven't already
                        if model_name not in self.models:
                            logger.info(f"Loading model {model_name} for collection {collection_name}")
                            try:
                                model = SentenceTransformer(model_name)
                                model.model_name = model_name  # Store for cache comparison
                                self.models[model_name] = model
                            except Exception as e:
                                logger.error(f"Failed to load model {model_name}: {e}")
                                raise
                        else:
                            logger.info(f"Using cached model {model_name} for collection {collection_name}")
                    else:
                        logger.warning(f"No model_name in config for collection {collection_name}")

                    logger.info(f"Loaded pgvector collection: {collection_name}")
                except Exception as e:
                    logger.error(f"Error loading pgvector collection {collection_name}: {e}")
        else:
            # SQLite backend - original behavior
            # Load model (shared across all indexes)
            if self.indexes and SentenceTransformer:
                # Get model name from first index
                sample_index = next(iter(self.indexes.values()))
                model_name = self._get_model_name(sample_index)
                try:
                    self.model = SentenceTransformer(model_name)
                    # Set the global model for query processor to avoid reloading
                    set_global_model(self.model)
                except Exception as e:
                    logger.warning(f"Could not load sentence transformer model: {e}")
                    self.model = None
            
            # Load search engines for each index
            for index_name, index_path in self.indexes.items():
                try:
                    self.search_engines[index_name] = SearchEngine(
                        backend='sqlite',
                        index_path=index_path,
                        model=self.model
                    )
                except Exception as e:
                    logger.error(f"Error loading search engine for {index_name}: {e}")
    
    def _get_model_name(self, index_path: str) -> str:
        """Get embedding model name from index config"""
        if self.backend == 'pgvector':
            # For pgvector, we might want to store model info in the database
            # For now, return default model
            return 'sentence-transformers/all-mpnet-base-v2'
        else:
            # SQLite backend
            try:
                import sqlite3
                conn = sqlite3.connect(index_path)
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM config WHERE key = 'embedding_model'")
                result = cursor.fetchone()
                conn.close()
                return result[0] if result else 'sentence-transformers/all-mpnet-base-v2'
            except Exception as e:
                logger.warning(f"Could not get model name from index: {e}")
                return 'sentence-transformers/all-mpnet-base-v2'
    
    async def _handle_search(self, request: SearchRequest) -> SearchResponse:
        """Handle search request with caching"""
        if request.index_name not in self.search_engines:
            if HTTPException:
                raise HTTPException(status_code=404, detail=f"Index '{request.index_name}' not found")
            else:
                raise ValueError(f"Index '{request.index_name}' not found")
        
        # Check cache first
        cache_key = _cache_key(request.query, request.index_name, request.count, request.tags)
        if cache_key in self._query_cache:
            logger.info(f"Cache hit for query: {request.query[:50]}...")
            return self._query_cache[cache_key]
        
        search_engine = self.search_engines[request.index_name]

        # For pgvector, set the correct model globally before query processing
        if self.backend == 'pgvector' and hasattr(self, 'collection_models'):
            collection_model_name = self.collection_models.get(request.index_name)
            if collection_model_name and collection_model_name in self.models:
                # Set this model globally so query processor uses it
                set_global_model(self.models[collection_model_name])
                logger.debug(f"Set global model to {collection_model_name} for collection {request.index_name}")

        # Get model name from the search engine config
        model_name = None
        if hasattr(search_engine, 'config') and search_engine.config:
            # pgvector uses 'model_name', sqlite uses 'embedding_model'
            model_name = search_engine.config.get('model_name') or search_engine.config.get('embedding_model')

        # Enhance query
        try:
            enhanced = preprocess_query(
                request.query,
                language=request.language or 'auto',
                vector=True,
                model_name=model_name  # Pass the correct model!
            )
        except Exception as e:
            logger.error(f"Error preprocessing query: {e}")
            enhanced = {
                'enhanced_text': request.query,
                'vector': [],
                'language': 'en'
            }
        
        # Perform search
        try:
            results = search_engine.search(
                query_vector=enhanced.get('vector', []),
                enhanced_text=enhanced['enhanced_text'],
                count=request.count,
                similarity_threshold=request.similarity_threshold,
                tags=request.tags
            )
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            results = []
        
        # Format response
        search_results = [
            SearchResult(
                content=result['content'],
                score=result['score'],
                metadata=result['metadata']
            )
            for result in results
        ]
        
        response = SearchResponse(
            results=search_results,
            query_analysis={
                'original_query': request.query,
                'enhanced_query': enhanced['enhanced_text'],
                'detected_language': enhanced.get('language'),
                'pos_analysis': enhanced.get('POS')
            }
        )
        
        # Cache the result
        if len(self._query_cache) >= self._cache_size:
            # Simple FIFO eviction
            first_key = next(iter(self._query_cache))
            del self._query_cache[first_key]
        self._query_cache[cache_key] = response
        
        return response
    
    def search_direct(self, query: str, index_name: str = "default", count: int = 3, 
                     distance: float = 0.0, tags: Optional[List[str]] = None, 
                     language: Optional[str] = None) -> Dict[str, Any]:
        """Direct search method (non-async) for programmatic use"""
        request = SearchRequest(
            query=query,
            index_name=index_name,
            count=count,
            distance=distance,
            tags=tags,
            language=language
        )
        
        # Use asyncio to run the async method
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        response = loop.run_until_complete(self._handle_search(request))
        
        return {
            'results': [
                {
                    'content': r.content,
                    'score': r.score,
                    'metadata': r.metadata
                }
                for r in response.results
            ],
            'query_analysis': response.query_analysis
        }
    
    def start(self, host: str = "0.0.0.0", port: Optional[int] = None,
              ssl_cert: Optional[str] = None, ssl_key: Optional[str] = None):
        """
        Start the service with optional HTTPS support.
        
        Args:
            host: Host to bind to (default: "0.0.0.0")
            port: Port to bind to (default: self.port)
            ssl_cert: Path to SSL certificate file (overrides environment)
            ssl_key: Path to SSL key file (overrides environment)
        """
        if not self.app:
            raise RuntimeError("FastAPI not available. Cannot start HTTP service.")
        
        port = port or self.port
        
        # Get SSL configuration
        ssl_kwargs = {}
        if ssl_cert and ssl_key:
            # Use provided SSL files
            ssl_kwargs = {
                'ssl_certfile': ssl_cert,
                'ssl_keyfile': ssl_key
            }
        else:
            # Use security config SSL settings
            ssl_kwargs = self.security.get_ssl_context_kwargs()
        
        # Build startup URL
        scheme = "https" if ssl_kwargs else "http"
        startup_url = f"{scheme}://{host}:{port}"
        
        # Get auth credentials
        username, password = self._basic_auth
        
        # Log startup information
        logger.info(
            "starting_search_service",
            url=startup_url,
            ssl_enabled=bool(ssl_kwargs),
            indexes=list(self.indexes.keys()),
            username=username
        )
        
        # Print user-friendly startup message
        print(f"\nSignalWire Search Service starting...")
        print(f"URL: {startup_url}")
        print(f"Indexes: {', '.join(self.indexes.keys()) if self.indexes else 'None'}")
        print(f"Basic Auth: {username}:{password}")
        if ssl_kwargs:
            print(f"SSL: Enabled")
        print("")
        
        try:
            import uvicorn
            uvicorn.run(
                self.app, 
                host=host, 
                port=port,
                **ssl_kwargs
            )
        except ImportError:
            raise RuntimeError("uvicorn not available. Cannot start HTTP service.")
    
    def stop(self):
        """Stop the service (placeholder for cleanup)"""
        pass 