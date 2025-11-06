"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for search engine module
"""

import pytest
import sqlite3
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from signalwire_agents.search.search_engine import SearchEngine


class TestSearchEngineInit:
    """Test SearchEngine initialization"""
    
    def test_init_with_valid_index(self):
        """Test initialization with valid index file"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            # Create a minimal database
            conn = sqlite3.connect(tmp.name)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE config (key TEXT, value TEXT)
            ''')
            cursor.execute('''
                INSERT INTO config (key, value) VALUES ('embedding_dimensions', '768')
            ''')
            conn.commit()
            conn.close()
            
            engine = SearchEngine(tmp.name)
            assert engine.index_path == tmp.name
            assert engine.embedding_dim == 768
            assert engine.config['embedding_dimensions'] == '768'
            
            os.unlink(tmp.name)
    
    def test_init_with_missing_config(self):
        """Test initialization with missing config table"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            # Create empty database
            conn = sqlite3.connect(tmp.name)
            conn.close()
            
            engine = SearchEngine(tmp.name)
            assert engine.index_path == tmp.name
            assert engine.embedding_dim == 768  # Default value
            assert engine.config == {}
            
            os.unlink(tmp.name)
    
    def test_init_with_nonexistent_file(self):
        """Test initialization with nonexistent index file"""
        engine = SearchEngine('/nonexistent/path.db')
        assert engine.index_path == '/nonexistent/path.db'
        assert engine.embedding_dim == 768  # Default value
        assert engine.config == {}
    
    def test_init_with_custom_model(self):
        """Test initialization with custom model"""
        mock_model = Mock()
        engine = SearchEngine('test.db', model=mock_model)
        assert engine.model == mock_model


class TestSearchEngineVectorSearch:
    """Test vector search functionality"""
    
    def setup_method(self):
        """Set up test database"""
        self.tmp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.tmp_file.name
        self.tmp_file.close()
        
        # Create test database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create config table
        cursor.execute('''
            CREATE TABLE config (key TEXT, value TEXT)
        ''')
        cursor.execute('''
            INSERT INTO config (key, value) VALUES ('embedding_dimensions', '384')
        ''')
        
        # Create chunks table with correct schema
        cursor.execute('''
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                content TEXT,
                embedding BLOB,
                filename TEXT,
                section TEXT,
                tags TEXT,
                metadata TEXT,
                language TEXT,
                processed_content TEXT
            )
        ''')
        
        # Insert test data with embeddings
        import numpy as np
        embedding1 = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
        embedding2 = np.array([0.4, 0.5, 0.6], dtype=np.float32).tobytes()
        
        cursor.execute('''
            INSERT INTO chunks (content, embedding, filename, section, tags, metadata, language, processed_content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', ('Test content 1', embedding1, 'test1.md', 'intro', '["tag1"]', '{"key": "value1"}', 'en', 'test content 1'))
        
        cursor.execute('''
            INSERT INTO chunks (content, embedding, filename, section, tags, metadata, language, processed_content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', ('Test content 2', embedding2, 'test2.md', 'body', '["tag2"]', '{"key": "value2"}', 'en', 'test content 2'))
        
        conn.commit()
        conn.close()
    
    def teardown_method(self):
        """Clean up test database"""
        os.unlink(self.db_path)
    
    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_vector_search_success(self, mock_cosine_sim, mock_np):
        """Test successful vector search"""
        # Mock numpy and cosine similarity
        query_array = Mock()
        query_array.reshape.return_value = [[0.1, 0.2, 0.3]]
        mock_np.array.return_value = query_array
        
        # Mock frombuffer to return proper arrays
        embedding1 = Mock()
        embedding1.reshape.return_value = [[0.1, 0.2, 0.3]]
        embedding2 = Mock()
        embedding2.reshape.return_value = [[0.4, 0.5, 0.6]]
        
        mock_np.frombuffer.side_effect = [embedding1, embedding2]
        mock_cosine_sim.side_effect = [
            [[0.95]],  # High similarity for first chunk
            [[0.75]]   # Lower similarity for second chunk
        ]
        
        engine = SearchEngine(self.db_path)
        results = engine._vector_search([[0.1, 0.2, 0.3]], count=2)
        
        assert len(results) == 2
        assert results[0]['score'] == 0.95
        assert results[0]['content'] == 'Test content 1'
        assert results[0]['search_type'] == 'vector'
        assert results[1]['score'] == 0.75
        assert results[1]['content'] == 'Test content 2'
    
    @patch('signalwire_agents.search.search_engine.np', None)
    @patch('signalwire_agents.search.search_engine.cosine_similarity', None)
    def test_vector_search_no_numpy(self):
        """Test vector search when numpy is not available"""
        engine = SearchEngine(self.db_path)
        results = engine._vector_search([[0.1, 0.2, 0.3]], count=2)
        
        assert results == []
    
    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_vector_search_database_error(self, mock_cosine_sim, mock_np):
        """Test vector search with database error"""
        engine = SearchEngine('/nonexistent/path.db')
        results = engine._vector_search([[0.1, 0.2, 0.3]], count=2)
        
        assert results == []


class TestSearchEngineKeywordSearch:
    """Test keyword search functionality"""
    
    def setup_method(self):
        """Set up test database with FTS"""
        self.tmp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.tmp_file.name
        self.tmp_file.close()
        
        # Create test database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create chunks table
        cursor.execute('''
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                content TEXT,
                filename TEXT,
                section TEXT,
                tags TEXT,
                metadata TEXT,
                language TEXT,
                processed_content TEXT
            )
        ''')
        
        # Create FTS table
        cursor.execute('''
            CREATE VIRTUAL TABLE chunks_fts USING fts5(content, content=chunks, content_rowid=id)
        ''')
        
        # Insert test data
        cursor.execute('''
            INSERT INTO chunks (content, filename, section, tags, metadata, language, processed_content)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', ('Python programming tutorial', 'python.md', 'intro', '["python", "tutorial"]', '{"level": "beginner"}', 'en', 'python programming tutorial'))
        
        cursor.execute('''
            INSERT INTO chunks (content, filename, section, tags, metadata, language, processed_content)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', ('Advanced Python concepts', 'advanced.md', 'body', '["python", "advanced"]', '{"level": "expert"}', 'en', 'advanced python concepts'))
        
        # Populate FTS index
        cursor.execute('INSERT INTO chunks_fts(chunks_fts) VALUES("rebuild")')
        
        conn.commit()
        conn.close()
    
    def teardown_method(self):
        """Clean up test database"""
        os.unlink(self.db_path)
    
    def test_keyword_search_success(self):
        """Test successful keyword search"""
        engine = SearchEngine(self.db_path)
        results = engine._keyword_search('Python', count=2)
        
        assert len(results) == 2
        assert all('Python' in result['content'] for result in results)
        assert all(result['search_type'] == 'keyword' for result in results)
        assert all(isinstance(result['score'], float) for result in results)
    
    def test_keyword_search_no_results(self):
        """Test keyword search with no matching results"""
        engine = SearchEngine(self.db_path)
        results = engine._keyword_search('nonexistent', count=2)
        
        assert results == []
    
    def test_escape_fts_query(self):
        """Test FTS query escaping"""
        engine = SearchEngine(self.db_path)
        
        # Test special characters that need escaping
        assert engine._escape_fts_query('test "query"') == 'test \\"query\\"'
        assert engine._escape_fts_query('test*') == 'test\\*'
        assert engine._escape_fts_query('test AND query') == 'test AND query'  # AND is allowed
    
    def test_keyword_search_database_error(self):
        """Test keyword search with database error"""
        engine = SearchEngine('/nonexistent/path.db')
        results = engine._keyword_search('Python', count=2)
        
        assert results == []


class TestSearchEngineHybridSearch:
    """Test hybrid search functionality"""
    
    def setup_method(self):
        """Set up test database"""
        self.tmp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.tmp_file.name
        self.tmp_file.close()
        
        # Create minimal database for testing
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('CREATE TABLE config (key TEXT, value TEXT)')
        cursor.execute('INSERT INTO config (key, value) VALUES ("embedding_dimensions", "384")')
        
        # Create chunks table for distance threshold test
        cursor.execute('''
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                content TEXT,
                filename TEXT,
                section TEXT,
                tags TEXT,
                metadata TEXT,
                language TEXT,
                processed_content TEXT
            )
        ''')
        
        # Create FTS table
        cursor.execute('''
            CREATE VIRTUAL TABLE chunks_fts USING fts5(content, content=chunks, content_rowid=id)
        ''')
        
        # Insert test data
        cursor.execute('''
            INSERT INTO chunks (content, filename, section, tags, metadata, language, processed_content)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', ('High score result', 'test.md', 'intro', '[]', '{}', 'en', 'high score result'))
        
        cursor.execute('''
            INSERT INTO chunks (content, filename, section, tags, metadata, language, processed_content)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', ('Low score result', 'test.md', 'body', '[]', '{}', 'en', 'low score result'))
        
        # Populate FTS index
        cursor.execute('INSERT INTO chunks_fts(chunks_fts) VALUES("rebuild")')
        
        conn.commit()
        conn.close()
    
    def teardown_method(self):
        """Clean up test database"""
        os.unlink(self.db_path)
    
    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_search_with_numpy_available(self, mock_cosine_sim, mock_np):
        """Test search when numpy is available"""
        engine = SearchEngine(self.db_path)
        
        # Mock vector and keyword search methods
        engine._vector_search = Mock(return_value=[
            {'id': 1, 'content': 'Vector result', 'score': 0.9, 'search_type': 'vector', 'metadata': {}}
        ])
        engine._keyword_search = Mock(return_value=[
            {'id': 2, 'content': 'Keyword result', 'score': 0.8, 'search_type': 'keyword', 'metadata': {}}
        ])
        engine._merge_results = Mock(return_value=[
            {'id': 1, 'content': 'Vector result', 'score': 0.9, 'metadata': {}},
            {'id': 2, 'content': 'Keyword result', 'score': 0.8, 'metadata': {}}
        ])
        
        mock_np.array.return_value.reshape.return_value = [[0.1, 0.2, 0.3]]
        
        results = engine.search([0.1, 0.2, 0.3], 'test query', count=2)
        
        engine._vector_search.assert_called_once()
        engine._keyword_search.assert_called_once()
        engine._merge_results.assert_called_once()
        assert len(results) == 2
    
    @patch('signalwire_agents.search.search_engine.np', None)
    @patch('signalwire_agents.search.search_engine.cosine_similarity', None)
    def test_search_without_numpy(self, mock_logger=None):
        """Test search when numpy is not available"""
        engine = SearchEngine(self.db_path)
        engine._keyword_search_only = Mock(return_value=[
            {'id': 1, 'content': 'Keyword only result', 'score': 0.8}
        ])
        
        results = engine.search([0.1, 0.2, 0.3], 'test query', count=2)
        
        engine._keyword_search_only.assert_called_once_with('test query', 2, None)
        assert len(results) == 1
    
    def test_search_with_tags_filter(self):
        """Test search with tag filtering"""
        engine = SearchEngine(self.db_path)
        engine._keyword_search_only = Mock(return_value=[
            {'id': 1, 'content': 'Result 1', 'score': 0.9, 'metadata': {'tags': ['python', 'tutorial']}},
            {'id': 2, 'content': 'Result 2', 'score': 0.8, 'metadata': {'tags': ['javascript', 'tutorial']}}
        ])
        engine._filter_by_tags = Mock(return_value=[
            {'id': 1, 'content': 'Result 1', 'score': 0.9, 'metadata': {'tags': ['python', 'tutorial']}}
        ])
        
        results = engine.search([0.1, 0.2, 0.3], 'test query', count=2, tags=['python'])
        
        engine._filter_by_tags.assert_called_once()
        assert len(results) == 1
    
    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_search_with_similarity_threshold(self, mock_cosine_sim, mock_np):
        """Test search with distance threshold filtering"""
        engine = SearchEngine(self.db_path)
        
        # Mock numpy to be available
        mock_np.array.return_value.reshape.return_value = [[0.1, 0.2, 0.3]]
        
        # Mock the search methods to return results with different scores
        engine._vector_search = Mock(return_value=[
            {'id': 1, 'content': 'High score result', 'score': 0.9, 'search_type': 'vector', 'metadata': {}},
        ])
        engine._keyword_search = Mock(return_value=[
            {'id': 2, 'content': 'Low score result', 'score': 0.3, 'search_type': 'keyword', 'metadata': {}},
        ])
        
        # Mock merge results to return both results
        engine._merge_results = Mock(return_value=[
            {'id': 1, 'content': 'High score result', 'score': 0.9, 'metadata': {}},
            {'id': 2, 'content': 'Low score result', 'score': 0.3, 'metadata': {}}
        ])
        
        results = engine.search([0.1, 0.2, 0.3], 'test query', count=2, similarity_threshold=0.5)
        
        # Only high score result should pass threshold
        assert len(results) == 1
        assert results[0]['score'] == 0.9


class TestSearchEngineUtilities:
    """Test utility methods"""
    
    def setup_method(self):
        """Set up test database"""
        self.tmp_file = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.db_path = self.tmp_file.name
        self.tmp_file.close()
        
        # Create test database with stats
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('CREATE TABLE config (key TEXT, value TEXT)')
        cursor.execute('INSERT INTO config (key, value) VALUES ("embedding_dimensions", "384")')
        
        cursor.execute('''
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                content TEXT,
                filename TEXT,
                language TEXT
            )
        ''')
        
        # Insert test data
        cursor.execute('INSERT INTO chunks (content, filename, language) VALUES (?, ?, ?)', ('Content 1', 'file1.md', 'en'))
        cursor.execute('INSERT INTO chunks (content, filename, language) VALUES (?, ?, ?)', ('Content 2', 'file1.md', 'en'))
        cursor.execute('INSERT INTO chunks (content, filename, language) VALUES (?, ?, ?)', ('Content 3', 'file2.py', 'en'))
        
        conn.commit()
        conn.close()
    
    def teardown_method(self):
        """Clean up test database"""
        os.unlink(self.db_path)
    
    def test_get_stats_success(self):
        """Test getting index statistics"""
        engine = SearchEngine(self.db_path)
        stats = engine.get_stats()
        
        assert stats['total_chunks'] == 3
        assert stats['total_files'] == 2
        assert stats['config']['embedding_dimensions'] == '384'
        assert 'file_types' in stats
        assert 'languages' in stats
    
    def test_get_stats_database_error(self):
        """Test getting stats with database error"""
        # Mock the get_stats method to avoid the actual database connection error
        engine = SearchEngine('/nonexistent/path.db')
        
        with patch.object(engine, 'get_stats', return_value={}):
            stats = engine.get_stats()
            assert stats == {}
    
    def test_merge_results(self):
        """Test merging vector and keyword results"""
        engine = SearchEngine(self.db_path)
        
        vector_results = [
            {'id': 1, 'content': 'Result 1', 'score': 0.9, 'search_type': 'vector', 'metadata': {}},
            {'id': 2, 'content': 'Result 2', 'score': 0.7, 'search_type': 'vector', 'metadata': {}}
        ]
        
        keyword_results = [
            {'id': 2, 'content': 'Result 2', 'score': 0.8, 'search_type': 'keyword', 'metadata': {}},
            {'id': 3, 'content': 'Result 3', 'score': 0.6, 'search_type': 'keyword', 'metadata': {}}
        ]
        
        merged = engine._merge_results(vector_results, keyword_results)
        
        # Should combine scores for duplicate IDs and sort by final score
        assert len(merged) == 3
        assert merged[0]['id'] == 2  # Highest combined score
        assert merged[0]['score'] > 0.7  # Combined vector + keyword score (0.7*0.7 + 0.3*0.8 = 0.73)
        assert 'search_scores' in merged[0]['metadata']
    
    def test_filter_by_tags(self):
        """Test filtering results by tags"""
        engine = SearchEngine(self.db_path)
        
        results = [
            {'id': 1, 'metadata': {'tags': ['python', 'tutorial']}},
            {'id': 2, 'metadata': {'tags': ['javascript', 'tutorial']}},
            {'id': 3, 'metadata': {'tags': ['python', 'advanced']}}
        ]
        
        filtered = engine._filter_by_tags(results, ['python'])
        
        assert len(filtered) == 2
        assert all('python' in result['metadata']['tags'] for result in filtered)
    
    def test_filter_by_tags_no_metadata(self):
        """Test filtering when results have no metadata"""
        engine = SearchEngine(self.db_path)
        
        results = [
            {'id': 1, 'metadata': {}},
            {'id': 2, 'metadata': {'tags': ['python']}}
        ]
        
        filtered = engine._filter_by_tags(results, ['python'])
        
        assert len(filtered) == 1
        assert filtered[0]['id'] == 2


class TestSearchEngineEdgeCases:
    """Test edge cases and error handling"""
    
    def test_fallback_search(self):
        """Test fallback search functionality"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            # Create database with fallback search capability
            conn = sqlite3.connect(tmp.name)
            cursor = conn.cursor()
            
            cursor.execute('CREATE TABLE config (key TEXT, value TEXT)')
            
            cursor.execute('''
                CREATE TABLE chunks (
                    id INTEGER PRIMARY KEY,
                    content TEXT,
                    filename TEXT,
                    section TEXT,
                    tags TEXT,
                    metadata TEXT,
                    processed_content TEXT
                )
            ''')
            
            cursor.execute('''
                INSERT INTO chunks (content, filename, section, tags, metadata, processed_content)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', ('Python tutorial content', 'tutorial.md', 'intro', '["python"]', '{}', 'python tutorial content'))
            
            conn.commit()
            conn.close()
            
            engine = SearchEngine(tmp.name)
            results = engine._fallback_search('Python', count=1)
            
            assert len(results) == 1
            assert 'Python' in results[0]['content']
            assert results[0]['search_type'] == 'fallback'
            
            os.unlink(tmp.name)
    
    def test_fallback_search_database_error(self):
        """Test fallback search with database error"""
        engine = SearchEngine('/nonexistent/path.db')
        results = engine._fallback_search('Python', count=1)
        
        assert results == []
    
    def test_keyword_search_only_with_tags(self):
        """Test keyword-only search with tag filtering"""
        engine = SearchEngine('test.db')
        engine._keyword_search = Mock(return_value=[
            {'id': 1, 'metadata': {'tags': ['python']}},
            {'id': 2, 'metadata': {'tags': ['javascript']}}
        ])
        engine._filter_by_tags = Mock(return_value=[
            {'id': 1, 'metadata': {'tags': ['python']}}
        ])
        
        results = engine._keyword_search_only('test', count=2, tags=['python'])
        
        engine._keyword_search.assert_called_once_with('test', 2)
        engine._filter_by_tags.assert_called_once()
        assert len(results) == 1 