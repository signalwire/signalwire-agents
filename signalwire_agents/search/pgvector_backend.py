"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    import psycopg2
    from psycopg2.extras import execute_values
    from pgvector.psycopg2 import register_vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    psycopg2 = None
    register_vector = None

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


class PgVectorBackend:
    """PostgreSQL pgvector backend for search indexing and retrieval"""
    
    def __init__(self, connection_string: str):
        """
        Initialize pgvector backend
        
        Args:
            connection_string: PostgreSQL connection string
        """
        if not PGVECTOR_AVAILABLE:
            raise ImportError(
                "pgvector dependencies not available. Install with: "
                "pip install psycopg2-binary pgvector"
            )
        
        self.connection_string = connection_string
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(self.connection_string)
            register_vector(self.conn)
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            error_msg = str(e)
            if "vector type not found" in error_msg:
                logger.error(
                    "pgvector extension not installed in database. "
                    "Run: CREATE EXTENSION IF NOT EXISTS vector;"
                )
            else:
                logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _ensure_connection(self):
        """Ensure database connection is active"""
        if self.conn is None or self.conn.closed:
            self._connect()
    
    def create_schema(self, collection_name: str, embedding_dim: int = 768):
        """
        Create database schema for a collection
        
        Args:
            collection_name: Name of the collection
            embedding_dim: Dimension of embeddings
        """
        self._ensure_connection()
        
        with self.conn.cursor() as cursor:
            # Create extensions
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            
            # Create table
            table_name = f"chunks_{collection_name}"
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    processed_content TEXT,
                    embedding vector({embedding_dim}),
                    filename TEXT,
                    section TEXT,
                    tags JSONB DEFAULT '[]'::jsonb,
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    metadata_text TEXT,  -- Searchable text representation of all metadata
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Create indexes
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding 
                ON {table_name} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_content 
                ON {table_name} USING gin (content gin_trgm_ops)
            """)
            
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_tags 
                ON {table_name} USING gin (tags)
            """)
            
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_metadata 
                ON {table_name} USING gin (metadata)
            """)
            
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table_name}_metadata_text 
                ON {table_name} USING gin (metadata_text gin_trgm_ops)
            """)
            
            # Create config table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS collection_config (
                    collection_name TEXT PRIMARY KEY,
                    model_name TEXT,
                    embedding_dimensions INTEGER,
                    chunking_strategy TEXT,
                    languages JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    metadata JSONB DEFAULT '{}'::jsonb
                )
            """)
            
            self.conn.commit()
            logger.info(f"Created schema for collection '{collection_name}'")
    
    def _extract_metadata_from_json_content(self, content: str) -> Dict[str, Any]:
        """
        Extract metadata from JSON content if present
        
        Returns:
            metadata_dict
        """
        metadata_dict = {}
        
        # Try to extract metadata from JSON structure in content
        if '"metadata":' in content:
            try:
                import re
                # Find all metadata objects
                pattern = r'"metadata"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
                matches = re.finditer(pattern, content)
                
                for match in matches:
                    try:
                        json_metadata = json.loads(match.group(1))
                        # Merge all found metadata
                        if isinstance(json_metadata, dict):
                            metadata_dict.update(json_metadata)
                    except:
                        pass
            except Exception as e:
                logger.debug(f"Error extracting JSON metadata: {e}")
        
        return metadata_dict

    def store_chunks(self, chunks: List[Dict[str, Any]], collection_name: str, 
                    config: Dict[str, Any]):
        """
        Store document chunks in the database
        
        Args:
            chunks: List of processed chunks with embeddings
            collection_name: Name of the collection
            config: Configuration metadata
        """
        self._ensure_connection()
        
        table_name = f"chunks_{collection_name}"
        
        # Prepare data for batch insert
        data = []
        for chunk in chunks:
            embedding = chunk.get('embedding')
            if embedding is not None:
                # Convert to list if it's a numpy array
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
            
            metadata = chunk.get('metadata', {})
            
            # Extract fields - they might be at top level or in metadata
            filename = chunk.get('filename') or metadata.get('filename', '')
            section = chunk.get('section') or metadata.get('section', '')
            tags = chunk.get('tags', []) or metadata.get('tags', [])
            
            # Extract metadata from JSON content and merge with chunk metadata
            json_metadata = self._extract_metadata_from_json_content(chunk['content'])
            
            # Build metadata from all fields except the ones we store separately
            chunk_metadata = {}
            for key, value in chunk.items():
                if key not in ['content', 'processed_content', 'embedding', 'filename', 'section', 'tags']:
                    chunk_metadata[key] = value
            # Also include any extra metadata
            for key, value in metadata.items():
                if key not in ['filename', 'section', 'tags']:
                    chunk_metadata[key] = value
            
            # Merge metadata: chunk metadata takes precedence over JSON metadata
            merged_metadata = {**json_metadata, **chunk_metadata}
            
            # Create searchable metadata text
            metadata_text_parts = []
            
            # Add all metadata keys and values
            for key, value in merged_metadata.items():
                metadata_text_parts.append(str(key).lower())
                if isinstance(value, list):
                    metadata_text_parts.extend(str(v).lower() for v in value)
                else:
                    metadata_text_parts.append(str(value).lower())
            
            # Add tags
            if tags:
                metadata_text_parts.extend(str(tag).lower() for tag in tags)
            
            # Add section if present
            if section:
                metadata_text_parts.append(section.lower())
            
            metadata_text = ' '.join(metadata_text_parts)
            
            data.append((
                chunk['content'],
                chunk.get('processed_content', chunk['content']),
                embedding,
                filename,
                section,
                json.dumps(tags),
                json.dumps(merged_metadata),
                metadata_text
            ))
        
        # Batch insert chunks
        with self.conn.cursor() as cursor:
            execute_values(
                cursor,
                f"""
                INSERT INTO {table_name} 
                (content, processed_content, embedding, filename, section, tags, metadata, metadata_text)
                VALUES %s
                """,
                data,
                template="(%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)"
            )
            
            # Update or insert config
            cursor.execute("""
                INSERT INTO collection_config 
                (collection_name, model_name, embedding_dimensions, chunking_strategy, 
                 languages, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (collection_name) 
                DO UPDATE SET 
                    model_name = EXCLUDED.model_name,
                    embedding_dimensions = EXCLUDED.embedding_dimensions,
                    chunking_strategy = EXCLUDED.chunking_strategy,
                    languages = EXCLUDED.languages,
                    metadata = EXCLUDED.metadata
            """, (
                collection_name,
                config.get('model_name'),
                config.get('embedding_dimensions'),
                config.get('chunking_strategy'),
                json.dumps(config.get('languages', [])),
                json.dumps(config.get('metadata', {}))
            ))
            
            self.conn.commit()
            logger.info(f"Stored {len(chunks)} chunks in collection '{collection_name}'")
    
    def get_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a collection"""
        self._ensure_connection()
        
        table_name = f"chunks_{collection_name}"
        
        with self.conn.cursor() as cursor:
            # Get chunk count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_chunks = cursor.fetchone()[0]
            
            # Get unique files
            cursor.execute(f"SELECT COUNT(DISTINCT filename) FROM {table_name}")
            total_files = cursor.fetchone()[0]
            
            # Get config
            cursor.execute(
                "SELECT * FROM collection_config WHERE collection_name = %s",
                (collection_name,)
            )
            config_row = cursor.fetchone()
            
            if config_row:
                config = {
                    'model_name': config_row[1],
                    'embedding_dimensions': config_row[2],
                    'chunking_strategy': config_row[3],
                    'languages': config_row[4],
                    'created_at': config_row[5].isoformat() if config_row[5] else None,
                    'metadata': config_row[6]
                }
            else:
                config = {}
            
            return {
                'total_chunks': total_chunks,
                'total_files': total_files,
                'config': config
            }
    
    def list_collections(self) -> List[str]:
        """List all collections in the database"""
        self._ensure_connection()
        
        with self.conn.cursor() as cursor:
            cursor.execute("SELECT collection_name FROM collection_config ORDER BY collection_name")
            return [row[0] for row in cursor.fetchall()]
    
    def delete_collection(self, collection_name: str):
        """Delete a collection and its data"""
        self._ensure_connection()
        
        table_name = f"chunks_{collection_name}"
        
        with self.conn.cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            cursor.execute(
                "DELETE FROM collection_config WHERE collection_name = %s",
                (collection_name,)
            )
            self.conn.commit()
            logger.info(f"Deleted collection '{collection_name}'")
    
    def close(self):
        """Close database connection"""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logger.info("Closed database connection")


class PgVectorSearchBackend:
    """PostgreSQL pgvector backend for search operations"""
    
    def __init__(self, connection_string: str, collection_name: str):
        """
        Initialize search backend
        
        Args:
            connection_string: PostgreSQL connection string
            collection_name: Name of the collection to search
        """
        if not PGVECTOR_AVAILABLE:
            raise ImportError(
                "pgvector dependencies not available. Install with: "
                "pip install psycopg2-binary pgvector"
            )
        
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.table_name = f"chunks_{collection_name}"
        self.conn = None
        self._connect()
        self.config = self._load_config()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(self.connection_string)
            register_vector(self.conn)
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _ensure_connection(self):
        """Ensure database connection is active"""
        if self.conn is None or self.conn.closed:
            self._connect()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load collection configuration"""
        self._ensure_connection()
        
        with self.conn.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM collection_config WHERE collection_name = %s",
                (self.collection_name,)
            )
            row = cursor.fetchone()
            
            if row:
                return {
                    'model_name': row[1],
                    'embedding_dimensions': row[2],
                    'chunking_strategy': row[3],
                    'languages': row[4],
                    'metadata': row[6]
                }
            return {}
    
    def search(self, query_vector: List[float], enhanced_text: str,
              count: int = 5, similarity_threshold: float = 0.0,
              tags: Optional[List[str]] = None,
              keyword_weight: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (vector + keyword + metadata)
        
        Args:
            query_vector: Embedding vector for the query
            enhanced_text: Processed query text for keyword search
            count: Number of results to return
            similarity_threshold: Minimum similarity score
            tags: Filter by tags
            keyword_weight: Manual keyword weight (0.0-1.0). If None, uses default weighting
            
        Returns:
            List of search results with scores and metadata
        """
        self._ensure_connection()

        # Extract query terms for metadata search
        query_terms = enhanced_text.lower().split()

        # Vector search
        vector_results = self._vector_search(query_vector, count * 2, tags)

        # Apply similarity threshold to raw vector scores BEFORE weighting
        # This ensures threshold behaves intuitively (filters on actual similarity, not weighted score)
        if similarity_threshold > 0:
            vector_results = [r for r in vector_results if r['score'] >= similarity_threshold]

        # Keyword search
        keyword_results = self._keyword_search(enhanced_text, count * 2, tags)

        # Metadata search
        metadata_results = self._metadata_search(query_terms, count * 2, tags)

        # Merge all results (threshold already applied to vector results)
        merged_results = self._merge_all_results(vector_results, keyword_results, metadata_results, keyword_weight)

        # Ensure 'score' field exists for CLI compatibility
        for r in merged_results:
            if 'score' not in r:
                r['score'] = r.get('final_score', 0.0)

        return merged_results[:count]
    
    def _vector_search(self, query_vector: List[float], count: int,
                      tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        with self.conn.cursor() as cursor:
            # Set ef_search for HNSW index to ensure we get enough results
            # ef_search must be at least as large as the LIMIT
            cursor.execute(f"SET LOCAL hnsw.ef_search = {max(count, 40)}")
            # Build query
            query = f"""
                SELECT id, content, filename, section, tags, metadata,
                       1 - (embedding <=> %s::vector) as similarity
                FROM {self.table_name}
                WHERE embedding IS NOT NULL
            """
            
            params = [query_vector]
            
            # Add tag filter if specified
            if tags:
                query += " AND tags ?| %s"
                params.append(tags)
            
            query += " ORDER BY embedding <=> %s::vector LIMIT %s"
            params.extend([query_vector, count])

            cursor.execute(query, params)

            results = []
            for row in cursor.fetchall():
                chunk_id, content, filename, section, tags_json, metadata_json, similarity = row
                
                results.append({
                    'id': chunk_id,
                    'content': content,
                    'score': float(similarity),
                    'metadata': {
                        'filename': filename,
                        'section': section,
                        'tags': tags_json if isinstance(tags_json, list) else [],
                        **metadata_json
                    },
                    'search_type': 'vector'
                })
            
            return results
    
    def _keyword_search(self, enhanced_text: str, count: int,
                       tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Perform full-text search"""
        with self.conn.cursor() as cursor:
            # Use PostgreSQL text search
            query = f"""
                SELECT id, content, filename, section, tags, metadata,
                       ts_rank(to_tsvector('english', content), 
                              plainto_tsquery('english', %s)) as rank
                FROM {self.table_name}
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
            """
            
            params = [enhanced_text, enhanced_text]
            
            # Add tag filter if specified
            if tags:
                query += " AND tags ?| %s"
                params.append(tags)
            
            query += " ORDER BY rank DESC LIMIT %s"
            params.append(count)
            
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                chunk_id, content, filename, section, tags_json, metadata_json, rank = row
                
                # Normalize rank to 0-1 score
                score = min(1.0, rank / 10.0)
                
                results.append({
                    'id': chunk_id,
                    'content': content,
                    'score': float(score),
                    'metadata': {
                        'filename': filename,
                        'section': section,
                        'tags': tags_json if isinstance(tags_json, list) else [],
                        **metadata_json
                    },
                    'search_type': 'keyword'
                })
            
            return results
    
    def _metadata_search(self, query_terms: List[str], count: int,
                        tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Perform metadata search using JSONB operators and metadata_text
        """
        with self.conn.cursor() as cursor:
            # Build WHERE conditions
            where_conditions = []
            params = []
            
            # Use metadata_text for trigram search
            if query_terms:
                # Create AND conditions for all terms
                for term in query_terms:
                    where_conditions.append(f"metadata_text ILIKE %s")
                    params.append(f'%{term}%')
            
            # Add tag filter if specified
            if tags:
                where_conditions.append("tags ?| %s")
                params.append(tags)
            
            # Build query
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            query = f"""
                SELECT id, content, filename, section, tags, metadata,
                       metadata_text
                FROM {self.table_name}
                WHERE {where_clause}
                LIMIT %s
            """
            
            params.append(count)
            
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                chunk_id, content, filename, section, tags_json, metadata_json, metadata_text = row
                
                # Calculate score based on term matches
                score = 0.0
                if metadata_text:
                    metadata_lower = metadata_text.lower()
                    for term in query_terms:
                        if term.lower() in metadata_lower:
                            score += 0.3  # Base score for each match
                
                # Bonus for exact matches in JSONB keys/values
                if metadata_json:
                    json_str = json.dumps(metadata_json).lower()
                    for term in query_terms:
                        if term.lower() in json_str:
                            score += 0.2
                
                # Normalize score
                score = min(1.0, score)
                
                results.append({
                    'id': chunk_id,
                    'content': content,
                    'score': float(score),
                    'metadata': {
                        'filename': filename,
                        'section': section,
                        'tags': tags_json if isinstance(tags_json, list) else [],
                        **metadata_json
                    },
                    'search_type': 'metadata'
                })
            
            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:count]
    
    def _merge_results(self, vector_results: List[Dict[str, Any]], 
                      keyword_results: List[Dict[str, Any]],
                      keyword_weight: Optional[float] = None) -> List[Dict[str, Any]]:
        """Merge and rank results from vector and keyword search"""
        # Use provided weights or defaults
        if keyword_weight is None:
            keyword_weight = 0.3
        vector_weight = 1.0 - keyword_weight
        
        # Create a map to track unique results
        results_map = {}
        
        # Add vector results
        for result in vector_results:
            chunk_id = result['id']
            if chunk_id not in results_map:
                results_map[chunk_id] = result
                results_map[chunk_id]['score'] *= vector_weight
            else:
                # Combine scores if result appears in both
                results_map[chunk_id]['score'] += result['score'] * vector_weight
        
        # Add keyword results
        for result in keyword_results:
            chunk_id = result['id']
            if chunk_id not in results_map:
                results_map[chunk_id] = result
                results_map[chunk_id]['score'] *= keyword_weight
            else:
                # Combine scores if result appears in both
                results_map[chunk_id]['score'] += result['score'] * keyword_weight
        
        # Sort by combined score
        merged = list(results_map.values())
        merged.sort(key=lambda x: x['score'], reverse=True)
        
        return merged
    
    def _merge_all_results(self, vector_results: List[Dict[str, Any]], 
                          keyword_results: List[Dict[str, Any]],
                          metadata_results: List[Dict[str, Any]],
                          keyword_weight: Optional[float] = None) -> List[Dict[str, Any]]:
        """Merge and rank results from vector, keyword, and metadata search"""
        # Use provided weights or defaults
        if keyword_weight is None:
            keyword_weight = 0.3
        vector_weight = 0.5
        metadata_weight = 0.2
        
        # Create a map to track unique results
        results_map = {}
        all_sources = {}
        
        # Add vector results
        for result in vector_results:
            chunk_id = result['id']
            if chunk_id not in results_map:
                results_map[chunk_id] = result.copy()
                results_map[chunk_id]['score'] = result['score'] * vector_weight
                all_sources[chunk_id] = {'vector': result['score']}
            else:
                results_map[chunk_id]['score'] += result['score'] * vector_weight
                all_sources[chunk_id]['vector'] = result['score']
        
        # Add keyword results
        for result in keyword_results:
            chunk_id = result['id']
            if chunk_id not in results_map:
                results_map[chunk_id] = result.copy()
                results_map[chunk_id]['score'] = result['score'] * keyword_weight
                all_sources.setdefault(chunk_id, {})['keyword'] = result['score']
            else:
                results_map[chunk_id]['score'] += result['score'] * keyword_weight
                all_sources[chunk_id]['keyword'] = result['score']
        
        # Add metadata results
        for result in metadata_results:
            chunk_id = result['id']
            if chunk_id not in results_map:
                results_map[chunk_id] = result.copy()
                results_map[chunk_id]['score'] = result['score'] * metadata_weight
                all_sources.setdefault(chunk_id, {})['metadata'] = result['score']
            else:
                results_map[chunk_id]['score'] += result['score'] * metadata_weight
                all_sources[chunk_id]['metadata'] = result['score']
        
        # Add sources to results for transparency
        for chunk_id, result in results_map.items():
            result['sources'] = all_sources.get(chunk_id, {})
            result['final_score'] = result['score']
        
        # Sort by combined score
        merged = list(results_map.values())
        merged.sort(key=lambda x: x['score'], reverse=True)
        
        return merged
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for the collection"""
        backend = PgVectorBackend(self.connection_string)
        stats = backend.get_stats(self.collection_name)
        backend.close()
        return stats
    
    def close(self):
        """Close database connection"""
        if self.conn and not self.conn.closed:
            self.conn.close()