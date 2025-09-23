"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional, Union

try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    NDArray = np.ndarray
except ImportError:
    np = None
    cosine_similarity = None
    NDArray = Any  # Fallback type for when numpy is not available

logger = logging.getLogger(__name__)

class SearchEngine:
    """Hybrid search engine for vector and keyword search"""
    
    def __init__(self, backend: str = 'sqlite', index_path: Optional[str] = None, 
                 connection_string: Optional[str] = None, collection_name: Optional[str] = None,
                 model=None):
        """
        Initialize search engine
        
        Args:
            backend: Storage backend ('sqlite' or 'pgvector')
            index_path: Path to .swsearch file (for sqlite backend)
            connection_string: PostgreSQL connection string (for pgvector backend)
            collection_name: Collection name (for pgvector backend)
            model: Optional sentence transformer model
        """
        self.backend = backend
        self.model = model
        
        if backend == 'sqlite':
            if not index_path:
                raise ValueError("index_path is required for sqlite backend")
            self.index_path = index_path
            self.config = self._load_config()
            self.embedding_dim = int(self.config.get('embedding_dimensions', 768))
            self._backend = None  # SQLite uses direct connection
        elif backend == 'pgvector':
            if not connection_string or not collection_name:
                raise ValueError("connection_string and collection_name are required for pgvector backend")
            from .pgvector_backend import PgVectorSearchBackend
            self._backend = PgVectorSearchBackend(connection_string, collection_name)
            self.config = self._backend.config
            self.embedding_dim = int(self.config.get('embedding_dimensions', 768))
        else:
            raise ValueError(f"Invalid backend '{backend}'. Must be 'sqlite' or 'pgvector'")
    
    def _load_config(self) -> Dict[str, str]:
        """Load index configuration"""
        try:
            conn = sqlite3.connect(self.index_path)
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM config")
            config = dict(cursor.fetchall())
            conn.close()
            return config
        except Exception as e:
            logger.error(f"Error loading config from {self.index_path}: {e}")
            return {}
    
    def search(self, query_vector: List[float], enhanced_text: str, 
              count: int = 3, distance_threshold: float = 0.0,
              tags: Optional[List[str]] = None, 
              keyword_weight: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (vector + keyword)
        
        Args:
            query_vector: Embedding vector for the query
            enhanced_text: Processed query text for keyword search
            count: Number of results to return
            distance_threshold: Minimum similarity score
            tags: Filter by tags
            
        Returns:
            List of search results with scores and metadata
        """
        
        # Use pgvector backend if available
        if self.backend == 'pgvector':
            return self._backend.search(query_vector, enhanced_text, count, distance_threshold, tags, keyword_weight)
        
        # Original SQLite implementation
        if not np or not cosine_similarity:
            logger.warning("NumPy or scikit-learn not available. Using keyword search only.")
            return self._keyword_search_only(enhanced_text, count, tags)
        
        # Convert query vector to numpy array
        try:
            query_array = np.array(query_vector).reshape(1, -1)
        except Exception as e:
            logger.error(f"Error converting query vector: {e}")
            return self._keyword_search_only(enhanced_text, count, tags)
        
        # If manual weight specified, use it
        if keyword_weight is not None:
            vector_weight = 1.0 - keyword_weight
            vector_results = self._vector_search(query_array, count * 2)
            keyword_results = self._keyword_search(enhanced_text, count * 2)
            merged_results = self._merge_results(vector_results, keyword_results,
                                               vector_weight=vector_weight,
                                               keyword_weight=keyword_weight)
            # Apply filters and return
            if tags:
                merged_results = self._filter_by_tags(merged_results, tags)
            filtered_results = [r for r in merged_results if r['score'] >= distance_threshold]
            return filtered_results[:count]
        
        # Progressive search strategy
        # 1. Try keyword-only first (fast)
        keyword_results = self._keyword_search(enhanced_text, count)
        
        if len(keyword_results) >= count/2:
            # Good keyword matches - blend with some vector
            vector_results = self._vector_search(query_array, count)
            merged_results = self._merge_results(vector_results, keyword_results, 
                                               vector_weight=0.3, 
                                               keyword_weight=0.7)
        else:
            # 2. Few keyword matches - try vector search
            vector_results = self._vector_search(query_array, count * 2)
            
            if not vector_results or (vector_results and vector_results[0]['score'] < 0.3):
                # Poor vector matches too - return keyword results anyway
                merged_results = keyword_results
            else:
                # 3. Normal blending when both have results
                merged_results = self._merge_results(vector_results, keyword_results,
                                                   vector_weight=0.7,
                                                   keyword_weight=0.3)
        
        # Filter by tags if specified
        if tags:
            merged_results = self._filter_by_tags(merged_results, tags)
        
        # Filter by distance threshold (only apply to vector results)
        filtered_results = [
            r for r in merged_results 
            if r.get('search_type') in ['keyword', 'fallback'] or r['score'] >= distance_threshold
        ]
        
        return filtered_results[:count]
    
    def _keyword_search_only(self, enhanced_text: str, count: int, 
                           tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Fallback to keyword search only when vector search is unavailable"""
        keyword_results = self._keyword_search(enhanced_text, count)
        
        if tags:
            keyword_results = self._filter_by_tags(keyword_results, tags)
        
        return keyword_results[:count]
    
    def _vector_search(self, query_vector: Union[NDArray, Any], count: int) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        if not np or not cosine_similarity:
            return []
            
        try:
            conn = sqlite3.connect(self.index_path)
            cursor = conn.cursor()
            
            # Get all embeddings (for small datasets, this is fine)
            # For large datasets, we'd use FAISS or similar
            cursor.execute('''
                SELECT id, content, embedding, filename, section, tags, metadata
                FROM chunks
                WHERE embedding IS NOT NULL AND embedding != ''
            ''')
            
            results = []
            for row in cursor.fetchall():
                chunk_id, content, embedding_blob, filename, section, tags_json, metadata_json = row
                
                if not embedding_blob:
                    continue
                
                try:
                    # Convert embedding back to numpy array
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32).reshape(1, -1)
                    
                    # Calculate similarity
                    similarity = cosine_similarity(query_vector, embedding)[0][0]
                    
                    results.append({
                        'id': chunk_id,
                        'content': content,
                        'score': float(similarity),
                        'metadata': {
                            'filename': filename,
                            'section': section,
                            'tags': json.loads(tags_json) if tags_json else [],
                            'metadata': json.loads(metadata_json) if metadata_json else {}
                        },
                        'search_type': 'vector'
                    })
                except Exception as e:
                    logger.warning(f"Error processing embedding for chunk {chunk_id}: {e}")
                    continue
            
            conn.close()
            
            # Sort by similarity score
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:count]
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []
    
    def _keyword_search(self, enhanced_text: str, count: int) -> List[Dict[str, Any]]:
        """Perform full-text search"""
        try:
            conn = sqlite3.connect(self.index_path)
            cursor = conn.cursor()
            
            # Escape FTS5 special characters
            escaped_text = self._escape_fts_query(enhanced_text)
            
            # FTS5 search
            cursor.execute('''
                SELECT c.id, c.content, c.filename, c.section, c.tags, c.metadata,
                       chunks_fts.rank
                FROM chunks_fts
                JOIN chunks c ON chunks_fts.rowid = c.id
                WHERE chunks_fts MATCH ?
                ORDER BY chunks_fts.rank
                LIMIT ?
            ''', (escaped_text, count))
            
            results = []
            for row in cursor.fetchall():
                chunk_id, content, filename, section, tags_json, metadata_json, rank = row
                
                # Convert FTS rank to similarity score (higher rank = lower score)
                # FTS5 rank is negative, so we convert it to a positive similarity score
                score = 1.0 / (1.0 + abs(rank))
                
                results.append({
                    'id': chunk_id,
                    'content': content,
                    'score': float(score),
                    'metadata': {
                        'filename': filename,
                        'section': section,
                        'tags': json.loads(tags_json) if tags_json else [],
                        'metadata': json.loads(metadata_json) if metadata_json else {}
                    },
                    'search_type': 'keyword'
                })
            
            conn.close()
            
            # If FTS returns no results, try fallback LIKE search
            if not results:
                logger.debug(f"FTS returned no results for '{enhanced_text}', trying fallback search")
                return self._fallback_search(enhanced_text, count)
                
            return results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            # Fallback to simple LIKE search
            return self._fallback_search(enhanced_text, count)
    
    def _escape_fts_query(self, query: str) -> str:
        """Escape special characters for FTS5 queries"""
        # FTS5 special characters that need escaping
        special_chars = ['"', "'", '(', ')', '*', '-', '+', ':', '^']
        
        escaped = query
        for char in special_chars:
            escaped = escaped.replace(char, f'\\{char}')
        
        return escaped
    
    def _fallback_search(self, enhanced_text: str, count: int) -> List[Dict[str, Any]]:
        """Fallback search using LIKE when FTS fails"""
        try:
            conn = sqlite3.connect(self.index_path)
            cursor = conn.cursor()
            
            # Simple LIKE search with word boundaries
            search_terms = enhanced_text.lower().split()
            like_conditions = []
            params = []
            
            for term in search_terms[:5]:  # Limit to 5 terms to avoid too complex queries
                # Search for term with word boundaries (space or punctuation)
                like_conditions.append("""
                    (LOWER(processed_content) LIKE ? 
                     OR LOWER(processed_content) LIKE ? 
                     OR LOWER(processed_content) LIKE ?
                     OR LOWER(processed_content) LIKE ?)
                """)
                params.extend([
                    f"% {term} %",  # space on both sides
                    f"{term} %",    # at beginning
                    f"% {term}",    # at end
                    f"{term}"       # exact match
                ])
            
            if not like_conditions:
                return []
            
            # Also search in original content
            content_conditions = []
            for term in search_terms[:5]:
                content_conditions.append("""
                    (LOWER(content) LIKE ? 
                     OR LOWER(content) LIKE ? 
                     OR LOWER(content) LIKE ?
                     OR LOWER(content) LIKE ?)
                """)
                params.extend([
                    f"% {term} %",  # with spaces
                    f"{term} %",    # at beginning
                    f"% {term}",    # at end
                    f"{term}"       # exact match
                ])
            
            query = f'''
                SELECT id, content, filename, section, tags, metadata
                FROM chunks
                WHERE ({" OR ".join(like_conditions)}) 
                   OR ({" OR ".join(content_conditions)})
                LIMIT ?
            '''
            params.append(count)
            
            
            cursor.execute(query, params)
            
            results = []
            for row in cursor.fetchall():
                chunk_id, content, filename, section, tags_json, metadata_json = row
                
                # Simple scoring based on term matches with word boundaries
                content_lower = content.lower()
                # Check for whole word matches
                word_matches = 0
                for term in search_terms:
                    term_lower = term.lower()
                    # Check word boundaries
                    if (f" {term_lower} " in f" {content_lower} " or 
                        content_lower.startswith(f"{term_lower} ") or 
                        content_lower.endswith(f" {term_lower}") or
                        content_lower == term_lower):
                        word_matches += 1
                score = word_matches / len(search_terms) if search_terms else 0.0
                
                results.append({
                    'id': chunk_id,
                    'content': content,
                    'score': float(score),
                    'metadata': {
                        'filename': filename,
                        'section': section,
                        'tags': json.loads(tags_json) if tags_json else [],
                        'metadata': json.loads(metadata_json) if metadata_json else {}
                    },
                    'search_type': 'fallback'
                })
            
            conn.close()
            
            # Sort by score
            results.sort(key=lambda x: x['score'], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in fallback search: {e}")
            return []
    
    def _merge_results(self, vector_results: List[Dict], keyword_results: List[Dict],
                      vector_weight: Optional[float] = None, 
                      keyword_weight: Optional[float] = None) -> List[Dict[str, Any]]:
        """Merge and rank vector and keyword search results"""
        # Use provided weights or defaults
        if vector_weight is None:
            vector_weight = 0.7
        if keyword_weight is None:
            keyword_weight = 0.3
        
        # Create a combined list with weighted scores
        combined = {}
        
        # Add vector results with weight
        for result in vector_results:
            chunk_id = result['id']
            combined[chunk_id] = result.copy()
            combined[chunk_id]['vector_score'] = result['score']
            combined[chunk_id]['keyword_score'] = 0.0
        
        # Add keyword results with weight
        for result in keyword_results:
            chunk_id = result['id']
            if chunk_id in combined:
                combined[chunk_id]['keyword_score'] = result['score']
            else:
                combined[chunk_id] = result.copy()
                combined[chunk_id]['vector_score'] = 0.0
                combined[chunk_id]['keyword_score'] = result['score']
        
        # Calculate combined score (weighted average)
        
        for chunk_id, result in combined.items():
            vector_score = result.get('vector_score', 0.0)
            keyword_score = result.get('keyword_score', 0.0)
            result['score'] = (vector_score * vector_weight + keyword_score * keyword_weight)
            
            # Add debug info
            result['metadata']['search_scores'] = {
                'vector': vector_score,
                'keyword': keyword_score,
                'combined': result['score']
            }
        
        # Sort by combined score
        sorted_results = sorted(combined.values(), key=lambda x: x['score'], reverse=True)
        return sorted_results
    
    def _filter_by_tags(self, results: List[Dict], required_tags: List[str]) -> List[Dict[str, Any]]:
        """Filter results by required tags"""
        filtered = []
        for result in results:
            result_tags = result['metadata'].get('tags', [])
            if any(tag in result_tags for tag in required_tags):
                filtered.append(result)
        return filtered
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the search index"""
        # Use pgvector backend if available
        if self.backend == 'pgvector':
            return self._backend.get_stats()
        
        # Original SQLite implementation
        conn = sqlite3.connect(self.index_path)
        cursor = conn.cursor()
        
        try:
            # Get total chunks
            cursor.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]
            
            # Get total files
            cursor.execute("SELECT COUNT(DISTINCT filename) FROM chunks")
            total_files = cursor.fetchone()[0]
            
            # Get average chunk size
            cursor.execute("SELECT AVG(LENGTH(content)) FROM chunks")
            avg_chunk_size = cursor.fetchone()[0] or 0
            
            # Get file types
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN filename LIKE '%.md' THEN 'markdown'
                        WHEN filename LIKE '%.py' THEN 'python'
                        WHEN filename LIKE '%.txt' THEN 'text'
                        WHEN filename LIKE '%.pdf' THEN 'pdf'
                        WHEN filename LIKE '%.docx' THEN 'docx'
                        ELSE 'other'
                    END as file_type,
                    COUNT(DISTINCT filename) as count
                FROM chunks 
                GROUP BY file_type
            """)
            file_types = dict(cursor.fetchall())
            
            # Get languages
            cursor.execute("SELECT language, COUNT(*) FROM chunks GROUP BY language")
            languages = dict(cursor.fetchall())
            
            return {
                'total_chunks': total_chunks,
                'total_files': total_files,
                'avg_chunk_size': int(avg_chunk_size),
                'file_types': file_types,
                'languages': languages,
                'config': self.config
            }
            
        finally:
            conn.close() 