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
              count: int = 3, similarity_threshold: float = 0.0,
              tags: Optional[List[str]] = None, 
              keyword_weight: Optional[float] = None,
              original_query: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform improved search with fast filtering and vector re-ranking
        
        Strategy:
        1. Fast candidate collection (filename, metadata, keywords)
        2. Vector re-ranking on candidates only
        3. Fallback to full vector search if few candidates
        
        Args:
            query_vector: Embedding vector for the query
            enhanced_text: Processed query text for keyword search
            count: Number of results to return
            similarity_threshold: Minimum similarity score
            tags: Filter by tags
            keyword_weight: Optional manual weight for keyword vs vector
            original_query: Original query for exact matching
            
        Returns:
            List of search results with scores and metadata
        """
        
        # Use pgvector backend if available
        if self.backend == 'pgvector':
            return self._backend.search(query_vector, enhanced_text, count, similarity_threshold, tags, keyword_weight)
        
        # Check for numpy/sklearn availability
        if not np or not cosine_similarity:
            logger.warning("NumPy or scikit-learn not available. Using keyword search only.")
            return self._keyword_search_only(enhanced_text, count, tags, original_query)
        
        # Convert query vector to numpy array
        try:
            query_array = np.array(query_vector).reshape(1, -1)
        except Exception as e:
            logger.error(f"Error converting query vector: {e}")
            return self._keyword_search_only(enhanced_text, count, tags, original_query)
        
        # HYBRID APPROACH: Search vector AND metadata in parallel
        # Stage 1: Run both search types simultaneously
        search_multiplier = 3

        # Vector search (semantic similarity - primary ranking signal)
        vector_results = self._vector_search(query_array, count * search_multiplier)

        # Metadata/keyword searches (confirmation signals and backfill)
        filename_results = self._filename_search(original_query or enhanced_text, count * search_multiplier)
        metadata_results = self._metadata_search(original_query or enhanced_text, count * search_multiplier)
        keyword_results = self._keyword_search(enhanced_text, count * search_multiplier, original_query)

        logger.debug(f"Parallel search: vector={len(vector_results)}, filename={len(filename_results)}, "
                    f"metadata={len(metadata_results)}, keyword={len(keyword_results)}")

        # Stage 2: Merge all results into candidate pool
        candidates = {}

        # Add vector results first (primary signal)
        for result in vector_results:
            chunk_id = result['id']
            candidates[chunk_id] = result
            candidates[chunk_id]['vector_score'] = result['score']
            candidates[chunk_id]['vector_distance'] = 1 - result['score']
            candidates[chunk_id]['sources'] = {'vector': True}
            candidates[chunk_id]['source_scores'] = {'vector': result['score']}

        # Add metadata/keyword results (secondary signals that boost or backfill)
        for result_set, source_type, source_weight in [(filename_results, 'filename', 2.0),
                                                        (metadata_results, 'metadata', 1.5),
                                                        (keyword_results, 'keyword', 1.0)]:
            for result in result_set:
                chunk_id = result['id']
                if chunk_id not in candidates:
                    # New candidate from metadata/keyword (no vector match)
                    candidates[chunk_id] = result
                    candidates[chunk_id]['sources'] = {source_type: True}
                    candidates[chunk_id]['source_scores'] = {source_type: result['score'] * source_weight}
                else:
                    # Exists in vector results - add metadata/keyword as confirmation signal
                    candidates[chunk_id]['sources'][source_type] = True
                    candidates[chunk_id]['source_scores'][source_type] = result['score'] * source_weight
        
        # Stage 3: Score and rank all candidates
        final_results = []
        for chunk_id, candidate in candidates.items():
            # Calculate final score combining all signals
            score = self._calculate_combined_score(candidate, similarity_threshold)
            candidate['final_score'] = score
            final_results.append(candidate)
        
        # Sort by final score
        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Filter by tags if specified
        if tags:
            final_results = [r for r in final_results 
                           if any(tag in r['metadata'].get('tags', []) for tag in tags)]
        
        # Apply distance threshold as final filter (soft threshold already applied in scoring)
        if similarity_threshold > 0:
            final_results = [r for r in final_results 
                           if r.get('vector_distance', 0) <= similarity_threshold * 1.5 
                           or 'vector' not in r.get('sources', {})]
        
        # Boost exact matches if we have the original query
        if original_query:
            final_results = self._boost_exact_matches(final_results, original_query)
            # Re-sort after boosting
            final_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Apply diversity penalties to prevent single-file dominance
        final_results = self._apply_diversity_penalties(final_results, count)

        # Ensure 'score' field exists for CLI compatibility
        for r in final_results:
            if 'score' not in r:
                r['score'] = r.get('final_score', 0.0)

        return final_results[:count]
    
    def _keyword_search_only(self, enhanced_text: str, count: int, 
                           tags: Optional[List[str]] = None, original_query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fallback to keyword search only when vector search is unavailable"""
        keyword_results = self._keyword_search(enhanced_text, count, original_query)
        
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
    
    def _keyword_search(self, enhanced_text: str, count: int, original_query: Optional[str] = None) -> List[Dict[str, Any]]:
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
    
    def _boost_exact_matches(self, results: List[Dict[str, Any]], original_query: str) -> List[Dict[str, Any]]:
        """Boost scores for results that contain exact matches of the original query"""
        if not original_query:
            return results
            
        # Extract key phrases to look for
        query_lower = original_query.lower()
        
        for result in results:
            content_lower = result['content'].lower()
            filename_lower = result['metadata'].get('filename', '').lower()
            
            # Boost for exact phrase match in content
            if query_lower in content_lower:
                result['score'] *= 2.0  # Double score for exact match
                
            # Boost for matches in filenames that suggest relevance
            if any(term in filename_lower for term in ['example', 'sample', 'demo', 'tutorial', 'guide']):
                if 'example' in query_lower or 'sample' in query_lower or 'code' in query_lower:
                    result['score'] *= 1.5
                    
            # Boost for "getting started" type queries
            if 'getting started' in query_lower and 'start' in content_lower:
                result['score'] *= 1.5
                
        return results
    
    def _filename_search(self, query: str, count: int) -> List[Dict[str, Any]]:
        """Search for query in filenames with term coverage scoring"""
        try:
            conn = sqlite3.connect(self.index_path)
            cursor = conn.cursor()
            
            query_lower = query.lower()
            terms = query_lower.split()
            
            # First try exact phrase match
            cursor.execute('''
                SELECT DISTINCT id, content, filename, section, tags, metadata
                FROM chunks
                WHERE LOWER(filename) LIKE ?
                LIMIT ?
            ''', (f'%{query_lower}%', count))
            
            results = []
            seen_ids = set()
            
            # Process exact matches
            for row in cursor.fetchall():
                chunk_id, content, filename, section, tags_json, metadata_json = row
                seen_ids.add(chunk_id)
                
                # High score for exact phrase match
                filename_lower = filename.lower()
                basename = filename_lower.split('/')[-1] if '/' in filename_lower else filename_lower
                if query_lower in basename:
                    score = 3.0  # Exact match in basename (increased weight)
                else:
                    score = 2.0  # Exact match in path
                
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
                    'search_type': 'filename',
                    'match_coverage': 1.0  # Exact match = 100% coverage
                })
            
            # Then search for files containing ANY of the terms
            if terms and len(results) < count * 3:  # Get more candidates
                # Build OR query for any term match
                conditions = []
                params = []
                for term in terms:
                    conditions.append("LOWER(filename) LIKE ?")
                    params.append(f'%{term}%')
                
                sql = f'''
                    SELECT DISTINCT id, content, filename, section, tags, metadata
                    FROM chunks
                    WHERE ({' OR '.join(conditions)})
                    AND id NOT IN ({','.join(['?' for _ in seen_ids]) if seen_ids else '0'})
                    LIMIT ?
                '''
                if seen_ids:
                    params.extend(seen_ids)
                params.append(count * 3)
                
                cursor.execute(sql, params)
                
                for row in cursor.fetchall():
                    chunk_id, content, filename, section, tags_json, metadata_json = row
                    
                    # Enhanced scoring based on term coverage
                    filename_lower = filename.lower()
                    basename = filename_lower.split('/')[-1] if '/' in filename_lower else filename_lower
                    
                    # Count matches in basename vs full path
                    basename_matches = sum(1 for term in terms if term in basename)
                    path_matches = sum(1 for term in terms if term in filename_lower)
                    
                    # Calculate term coverage (what % of query terms are matched)
                    term_coverage = path_matches / len(terms) if terms else 0
                    basename_coverage = basename_matches / len(terms) if terms else 0
                    
                    # Check for substring bonus (e.g., "code_examples" contains both terms together)
                    substring_bonus = 0
                    if len(terms) > 1:
                        # Check if terms appear consecutively
                        for i in range(len(terms) - 1):
                            if f"{terms[i]}_{terms[i+1]}" in filename_lower or f"{terms[i]}{terms[i+1]}" in filename_lower:
                                substring_bonus = 0.3
                                break
                    
                    # Score based on coverage with exponential boost for more matches
                    if basename_coverage > 0:
                        # Exponential scoring for basename matches
                        score = basename_coverage ** 1.5 + substring_bonus
                    else:
                        # Lower score for path-only matches
                        score = (term_coverage * 0.5) ** 1.5 + substring_bonus
                    
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
                        'search_type': 'filename',
                        'match_coverage': term_coverage
                    })
            
            conn.close()
            
            # Sort by score and return top results
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:count]
            
        except Exception as e:
            logger.error(f"Error in filename search: {e}")
            return []
    
    def _metadata_search(self, query: str, count: int) -> List[Dict[str, Any]]:
        """Search in all metadata fields (tags, sections, category, product, source)"""
        try:
            conn = sqlite3.connect(self.index_path)
            cursor = conn.cursor()
            
            query_lower = query.lower()
            terms = query_lower.split()
            results = []
            seen_ids = set()
            
            # First, try to use the metadata_text column if it exists
            try:
                # Check if metadata_text column exists
                cursor.execute("PRAGMA table_info(chunks)")
                columns = [col[1] for col in cursor.fetchall()]
                has_metadata_text = 'metadata_text' in columns
            except:
                has_metadata_text = False
            
            if has_metadata_text:
                # Use the new metadata_text column for efficient searching
                # Build conditions for each term
                conditions = []
                for term in terms:
                    conditions.append(f"metadata_text LIKE '%{term}%'")
                
                if conditions:
                    query_sql = f'''
                        SELECT id, content, filename, section, tags, metadata
                        FROM chunks
                        WHERE {' AND '.join(conditions)}
                        LIMIT ?
                    '''
                    cursor.execute(query_sql, (count * 10,))
                    
                    for row in cursor.fetchall():
                        chunk_id, content, filename, section, tags_json, metadata_json = row
                        
                        if chunk_id in seen_ids:
                            continue
                        
                        # Parse metadata
                        metadata = json.loads(metadata_json) if metadata_json else {}
                        tags = json.loads(tags_json) if tags_json else []
                        
                        # Calculate score based on how many terms match
                        score = 0
                        for term in terms:
                            # Check metadata values
                            metadata_str = json.dumps(metadata).lower()
                            if term in metadata_str:
                                score += 1.5
                            # Check tags
                            if any(term in str(tag).lower() for tag in tags):
                                score += 1.0
                            # Check section
                            if section and term in section.lower():
                                score += 0.8
                        
                        if score > 0:
                            seen_ids.add(chunk_id)
                            results.append({
                                'id': chunk_id,
                                'content': content,
                                'score': score,
                                'metadata': {
                                    'filename': filename,
                                    'section': section,
                                    'tags': tags,
                                    'metadata': metadata
                                },
                                'search_type': 'metadata'
                            })
            
            # Fallback: search for JSON metadata embedded in content
            # This ensures backwards compatibility
            if len(results) < count:
                # Build specific conditions for known patterns
                specific_conditions = []
                
                # Look for specific high-value patterns first
                if 'code' in terms and 'examples' in terms:
                    specific_conditions.append('content LIKE \'%"category": "Code Examples"%\'')
                if 'sdk' in terms:
                    specific_conditions.append('content LIKE \'%"product": "%\' || \'SDK\' || \'%"%\'')
                
                # General term search in JSON content
                for term in terms:
                    specific_conditions.append(f"content LIKE '%\"{term}%'")
                
                if specific_conditions:
                    # Limit conditions to avoid too broad search
                    conditions_to_use = specific_conditions[:10]
                    query_sql = f'''
                        SELECT id, content, filename, section, tags, metadata
                        FROM chunks
                        WHERE ({' OR '.join(conditions_to_use)})
                        AND id NOT IN ({','.join(str(id) for id in seen_ids) if seen_ids else '0'})
                        LIMIT ?
                    '''
                    cursor.execute(query_sql, (count * 5,))
                
                rows = cursor.fetchall()
                
                for row in rows:
                    chunk_id, content, filename, section, tags_json, metadata_json = row
                    
                    if chunk_id in seen_ids:
                        continue
                        
                    # Try to extract metadata from JSON content
                    json_metadata = {}
                    try:
                        # Look for metadata in JSON structure
                        if '"metadata":' in content:
                            import re
                            # More robust regex to extract nested JSON object
                            # This handles nested braces properly
                            start = content.find('"metadata":')
                            if start != -1:
                                # Find the opening brace
                                brace_start = content.find('{', start)
                                if brace_start != -1:
                                    # Count braces to find matching closing brace
                                    brace_count = 0
                                    i = brace_start
                                    while i < len(content):
                                        if content[i] == '{':
                                            brace_count += 1
                                        elif content[i] == '}':
                                            brace_count -= 1
                                            if brace_count == 0:
                                                # Found matching closing brace
                                                metadata_str = content[brace_start:i+1]
                                                json_metadata = json.loads(metadata_str)
                                                break
                                        i += 1
                    except:
                        pass
                    
                    # Calculate score based on matches
                    score = 0
                    fields_matched = 0
                    
                    # Check JSON metadata extracted from content
                    if json_metadata:
                        # Check category - count how many terms match
                        category = json_metadata.get('category', '').lower()
                        if category:
                            category_matches = sum(1 for term in terms if term in category)
                            if category_matches > 0:
                                score += 1.8 * (category_matches / len(terms) if terms else 1)
                                fields_matched += 1
                        
                        # Check product - count how many terms match
                        product = json_metadata.get('product', '').lower() 
                        if product:
                            product_matches = sum(1 for term in terms if term in product)
                            if product_matches > 0:
                                score += 1.5 * (product_matches / len(terms) if terms else 1)
                                fields_matched += 1
                            
                        # Check source
                        source = json_metadata.get('source', '').lower()
                        if source:
                            source_matches = sum(1 for term in terms if term in source)
                            if source_matches > 0:
                                score += 1.2 * (source_matches / len(terms) if terms else 1)
                                fields_matched += 1
                    
                    # Also check tags from JSON metadata
                    json_tags = json_metadata.get('tags', [])
                    if json_tags:
                        tags_str = str(json_tags).lower()
                        tag_matches = sum(1 for term in terms if term in tags_str)
                        if tag_matches > 0:
                            score += 1.3 * (tag_matches / len(terms) if terms else 1)
                            fields_matched += 1
                    
                    if score > 0:
                        seen_ids.add(chunk_id)
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
                            'search_type': 'metadata',
                            'fields_matched': fields_matched
                        })
                        logger.debug(f"Metadata match: {filename} - score={score:.2f}, fields_matched={fields_matched}, json_metadata={json_metadata}")
            
            # Also get chunks with regular metadata
            cursor.execute('''
                SELECT id, content, filename, section, tags, metadata
                FROM chunks
                WHERE (tags IS NOT NULL AND tags != '') 
                   OR (metadata IS NOT NULL AND metadata != '{}')
                   OR (section IS NOT NULL AND section != '')
                LIMIT ?
            ''', (count * 10,))  # Get more to search through
            
            for row in cursor.fetchall():
                chunk_id, content, filename, section, tags_json, metadata_json = row
                
                if chunk_id in seen_ids:
                    continue
                
                # Parse metadata
                tags = json.loads(tags_json) if tags_json else []
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                # Flatten nested metadata if present
                if 'metadata' in metadata:
                    # Handle double-nested metadata from some indexes
                    nested_meta = metadata['metadata']
                    metadata.update(nested_meta)
                
                # Initialize scoring components
                score_components = {
                    'tags': 0,
                    'section': 0,
                    'category': 0,
                    'product': 0,
                    'source': 0,
                    'description': 0
                }
                
                # Check tags
                if tags:
                    tag_matches = 0
                    for tag in tags:
                        tag_lower = tag.lower()
                        # Full query match in tag
                        if query_lower in tag_lower:
                            tag_matches += 2.0
                        else:
                            # Individual term matches
                            term_matches = sum(1 for term in terms if term in tag_lower)
                            tag_matches += term_matches * 0.5
                    
                    if tag_matches > 0:
                        score_components['tags'] = min(1.0, tag_matches / len(tags))
                
                # Check section
                if section and section.lower() != 'none':
                    section_lower = section.lower()
                    if query_lower in section_lower:
                        score_components['section'] = 1.0
                    else:
                        term_matches = sum(1 for term in terms if term in section_lower)
                        score_components['section'] = (term_matches / len(terms)) * 0.8 if terms else 0
                
                # Check category field
                category = metadata.get('category', '')
                if category:
                    category_lower = category.lower()
                    if query_lower in category_lower:
                        score_components['category'] = 1.0
                    else:
                        term_matches = sum(1 for term in terms if term in category_lower)
                        score_components['category'] = (term_matches / len(terms)) * 0.9 if terms else 0
                
                # Check product field
                product = metadata.get('product', '')
                if product:
                    product_lower = product.lower()
                    if query_lower in product_lower:
                        score_components['product'] = 1.0
                    else:
                        term_matches = sum(1 for term in terms if term in product_lower)
                        score_components['product'] = (term_matches / len(terms)) * 0.8 if terms else 0
                
                # Check source field (original filename)
                source = metadata.get('source', '')
                if source:
                    source_lower = source.lower()
                    if query_lower in source_lower:
                        score_components['source'] = 1.0
                    else:
                        term_matches = sum(1 for term in terms if term in source_lower)
                        score_components['source'] = (term_matches / len(terms)) * 0.7 if terms else 0
                
                # Check description or title fields
                description = metadata.get('description', metadata.get('title', ''))
                if description:
                    desc_lower = description.lower()
                    if query_lower in desc_lower:
                        score_components['description'] = 0.8
                    else:
                        term_matches = sum(1 for term in terms if term in desc_lower)
                        score_components['description'] = (term_matches / len(terms)) * 0.6 if terms else 0
                
                # Calculate total score with weights
                weights = {
                    'category': 1.8,    # Strong signal
                    'product': 1.5,     # Strong signal
                    'tags': 1.3,        # Good signal
                    'source': 1.2,      # Good signal
                    'section': 1.0,     # Moderate signal
                    'description': 0.8  # Weaker signal
                }
                
                total_score = sum(score_components[field] * weights.get(field, 1.0) 
                                for field in score_components)
                
                # Track match coverage
                fields_matched = sum(1 for score in score_components.values() if score > 0)
                match_coverage = sum(1 for term in terms if any(
                    term in str(field_value).lower() 
                    for field_value in [tags, section, category, product, source, description]
                    if field_value
                )) / len(terms) if terms else 0
                
                if total_score > 0:
                    results.append({
                        'id': chunk_id,
                        'content': content,
                        'score': float(total_score),
                        'metadata': {
                            'filename': filename,
                            'section': section,
                            'tags': tags,
                            'metadata': metadata,
                            'category': category,
                            'product': product,
                            'source': source
                        },
                        'search_type': 'metadata',
                        'metadata_matches': score_components,
                        'fields_matched': fields_matched,
                        'match_coverage': match_coverage
                    })
                    seen_ids.add(chunk_id)
            
            conn.close()
            
            # Sort by score and return top results
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:count]
            
        except Exception as e:
            logger.error(f"Error in metadata search: {e}")
            return []
    
    def _add_vector_scores_to_candidates(self, candidates: Dict[str, Dict], query_vector: NDArray, 
                                       similarity_threshold: float):
        """Add vector similarity scores to existing candidates"""
        if not candidates or not np:
            return
            
        try:
            conn = sqlite3.connect(self.index_path)
            cursor = conn.cursor()
            
            # Get embeddings for candidate chunks only
            chunk_ids = list(candidates.keys())
            placeholders = ','.join(['?' for _ in chunk_ids])
            
            cursor.execute(f'''
                SELECT id, embedding
                FROM chunks
                WHERE id IN ({placeholders}) AND embedding IS NOT NULL AND embedding != ''
            ''', chunk_ids)
            
            for row in cursor.fetchall():
                chunk_id, embedding_blob = row
                
                if not embedding_blob:
                    continue
                
                try:
                    # Convert embedding back to numpy array
                    embedding = np.frombuffer(embedding_blob, dtype=np.float32).reshape(1, -1)
                    
                    # Calculate similarity
                    similarity = cosine_similarity(query_vector, embedding)[0][0]
                    distance = 1 - similarity
                    
                    # Add vector scores to candidate
                    candidates[chunk_id]['vector_score'] = float(similarity)
                    candidates[chunk_id]['vector_distance'] = float(distance)
                    candidates[chunk_id]['sources']['vector_rerank'] = True
                    
                except Exception as e:
                    logger.debug(f"Error processing embedding for chunk {chunk_id}: {e}")
                    continue
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error in vector re-ranking: {e}")
    
    def _calculate_combined_score(self, candidate: Dict, similarity_threshold: float) -> float:
        """Calculate final score with hybrid vector + metadata weighting

        Hybrid approach:
        - Vector score is the primary ranking signal (semantic similarity)
        - Metadata/keyword matches provide confirmation boost
        - Multiple signal types indicate high relevance (confirmation bonus)
        - Special boost for 'code' tag matches when query contains code-related terms
        """
        sources = candidate.get('sources', {})
        source_scores = candidate.get('source_scores', {})

        # Vector score is PRIMARY
        if 'vector_score' in candidate:
            vector_score = candidate['vector_score']
            base_score = vector_score

            # Metadata/keyword matches provide confirmation boost
            if len(sources) > 1:
                # Has both vector AND metadata/keyword matches - strong confirmation signal
                keyword_signals = sum(source_scores.get(k, 0) for k in ['keyword', 'filename', 'metadata'])
                if keyword_signals > 0:
                    # Normalize and apply boost (up to 30% for strong confirmation)
                    keyword_boost = min(0.3, keyword_signals * 0.15)
                    base_score = vector_score * (1.0 + keyword_boost)

                    # Additional boost if multiple signal types confirm (2+ sources)
                    num_metadata_sources = sum(1 for s in ['keyword', 'filename', 'metadata'] if s in sources)
                    if num_metadata_sources >= 2:
                        # Multiple confirmation signals - very high confidence
                        base_score *= 1.1

            # Check for code-related tags to boost code examples
            tags = candidate.get('metadata', {}).get('tags', [])
            if 'code' in tags:
                # This chunk contains code - boost if query is code-related
                # (metadata search would have found it if query mentioned code/example/python/etc)
                if 'metadata' in sources or 'keyword' in sources:
                    # Query matched code-related metadata - apply code boost
                    base_score *= 1.2
        else:
            # No vector score - this is a keyword-only result (backfill)
            # Use keyword scores but penalize for lack of semantic match
            base_score = sum(source_scores.values()) * 0.6  # 40% penalty for no vector

            # Still boost code chunks if metadata matched
            tags = candidate.get('metadata', {}).get('tags', [])
            if 'code' in tags and 'metadata' in sources:
                base_score *= 1.15

        return base_score
    
    def _apply_diversity_penalties(self, results: List[Dict], target_count: int) -> List[Dict]:
        """Apply penalties to prevent single-file dominance while maintaining quality"""
        if not results:
            return results
        
        # Track file occurrences
        file_counts = {}
        penalized_results = []
        
        # Define penalty multipliers
        occurrence_penalties = {
            1: 1.0,    # First chunk: no penalty
            2: 0.85,   # Second chunk: 15% penalty
            3: 0.7,    # Third chunk: 30% penalty
            4: 0.5,    # Fourth chunk: 50% penalty
        }
        
        for result in results:
            filename = result['metadata']['filename']
            
            # Get current count for this file
            current_count = file_counts.get(filename, 0) + 1
            file_counts[filename] = current_count
            
            # Apply penalty based on occurrence
            penalty = occurrence_penalties.get(current_count, 0.4)  # 60% penalty for 5+ chunks
            
            # Create a copy to avoid modifying original
            penalized_result = result.copy()
            penalized_result['diversity_penalty'] = penalty
            penalized_result['final_score'] = result.get('final_score', result.get('score', 0)) * penalty
            
            penalized_results.append(penalized_result)
        
        # Re-sort by penalized scores
        penalized_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Ensure minimum diversity if we have enough results
        if len(penalized_results) > target_count:
            unique_files = len(set(r['metadata']['filename'] for r in penalized_results[:target_count]))
            
            # If top results are too homogeneous (e.g., all from 1-2 files)
            if unique_files < min(3, target_count):
                # Try to inject some diversity
                selected = penalized_results[:target_count]
                seen_files = set(r['metadata']['filename'] for r in selected)
                
                # Look for high-quality results from other files
                for result in penalized_results[target_count:]:
                    if result['metadata']['filename'] not in seen_files:
                        # If it's reasonably good (within 50% of top score), include it
                        if result['final_score'] > 0.5 * selected[0]['final_score']:
                            # Replace the lowest scoring result from an over-represented file
                            for i in range(len(selected) - 1, -1, -1):
                                if file_counts[selected[i]['metadata']['filename']] > 2:
                                    selected[i] = result
                                    seen_files.add(result['metadata']['filename'])
                                    break
                
                penalized_results[:target_count] = selected
        
        return penalized_results

    def _apply_match_type_diversity(self, results: List[Dict], target_count: int) -> List[Dict]:
        """Ensure diversity of match types in final results

        Ensures we have a mix of:
        - Vector-only matches (semantic similarity, good for code examples)
        - Keyword-only matches (exact term matches)
        - Hybrid matches (both vector + keyword/metadata)
        """
        if not results or len(results) <= target_count:
            return results

        # Categorize results by match type
        vector_only = []
        keyword_only = []
        hybrid = []

        for result in results:
            sources = result.get('sources', {})
            has_vector = 'vector' in sources
            has_keyword = any(k in sources for k in ['keyword', 'filename', 'metadata'])

            if has_vector and not has_keyword:
                vector_only.append(result)
            elif has_keyword and not has_vector:
                keyword_only.append(result)
            else:
                hybrid.append(result)

        # Build diverse result set
        # Target distribution: 40% hybrid, 40% vector-only, 20% keyword-only
        # This ensures we include semantic matches (code examples) even if keywords don't match
        diversified = []

        # Take top hybrid matches first (best overall)
        hybrid_target = max(1, int(target_count * 0.4))
        diversified.extend(hybrid[:hybrid_target])

        # Ensure we have vector-only matches (critical for code examples)
        vector_target = max(1, int(target_count * 0.4))
        diversified.extend(vector_only[:vector_target])

        # Add keyword-only matches
        keyword_target = max(1, int(target_count * 0.2))
        diversified.extend(keyword_only[:keyword_target])

        # Fill remaining slots with best remaining results regardless of type
        remaining_slots = target_count - len(diversified)
        if remaining_slots > 0:
            # Get all unused results
            used_ids = set(r['id'] for r in diversified)
            unused = [r for r in results if r['id'] not in used_ids]
            diversified.extend(unused[:remaining_slots])

        # Sort by final score to maintain quality ordering
        diversified.sort(key=lambda x: x['final_score'], reverse=True)

        return diversified

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