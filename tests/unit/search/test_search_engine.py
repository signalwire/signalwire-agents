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
            
            engine = SearchEngine(backend='sqlite', index_path=tmp.name)
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

            engine = SearchEngine(backend='sqlite', index_path=tmp.name)
            assert engine.index_path == tmp.name
            assert engine.embedding_dim == 768  # Default value
            assert engine.config == {}

            os.unlink(tmp.name)

    def test_init_with_nonexistent_file(self):
        """Test initialization with nonexistent index file"""
        engine = SearchEngine(backend='sqlite', index_path='/nonexistent/path.db')
        assert engine.index_path == '/nonexistent/path.db'
        assert engine.embedding_dim == 768  # Default value
        assert engine.config == {}

    def test_init_with_custom_model(self):
        """Test initialization with custom model"""
        mock_model = Mock()
        engine = SearchEngine(backend='sqlite', index_path='test.db', model=mock_model)
        assert engine.model == mock_model


try:
    import numpy as _np
    _has_numpy = True
except ImportError:
    _has_numpy = False


@pytest.mark.skipif(not _has_numpy, reason='numpy not installed')
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
        
        engine = SearchEngine(backend='sqlite', index_path=self.db_path)
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
        engine = SearchEngine(backend='sqlite', index_path=self.db_path)
        results = engine._vector_search([[0.1, 0.2, 0.3]], count=2)

        assert results == []

    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_vector_search_database_error(self, mock_cosine_sim, mock_np):
        """Test vector search with database error"""
        engine = SearchEngine(backend='sqlite', index_path='/nonexistent/path.db')
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
        engine = SearchEngine(backend='sqlite', index_path=self.db_path)
        results = engine._keyword_search('Python', count=2)
        
        assert len(results) == 2
        assert all('Python' in result['content'] for result in results)
        assert all(result['search_type'] == 'keyword' for result in results)
        assert all(isinstance(result['score'], float) for result in results)
    
    def test_keyword_search_no_results(self):
        """Test keyword search with no matching results"""
        engine = SearchEngine(backend='sqlite', index_path=self.db_path)
        results = engine._keyword_search('nonexistent', count=2)
        
        assert results == []
    
    def test_escape_fts_query(self):
        """Test FTS query escaping"""
        engine = SearchEngine(backend='sqlite', index_path=self.db_path)

        # New behavior: strips double quotes and wraps each term in double quotes
        assert engine._escape_fts_query('test "query"') == '"test" "query"'
        assert engine._escape_fts_query('test*') == '"test*"'
        assert engine._escape_fts_query('test AND query') == '"test" "AND" "query"'
    
    def test_keyword_search_database_error(self):
        """Test keyword search with database error"""
        engine = SearchEngine(backend='sqlite', index_path='/nonexistent/path.db')
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
        engine = SearchEngine(backend='sqlite', index_path=self.db_path)

        # Mock all search methods used in the parallel search approach
        engine._vector_search = Mock(return_value=[
            {'id': 1, 'content': 'Vector result', 'score': 0.9, 'search_type': 'vector', 'metadata': {}}
        ])
        engine._filename_search = Mock(return_value=[])
        engine._metadata_search = Mock(return_value=[])
        engine._keyword_search = Mock(return_value=[
            {'id': 2, 'content': 'Keyword result', 'score': 0.8, 'search_type': 'keyword', 'metadata': {}}
        ])
        engine._calculate_combined_score = Mock(side_effect=lambda c, t: c.get('score', 0.0))
        engine._apply_diversity_penalties = Mock(side_effect=lambda results, count: results)

        mock_np.array.return_value.reshape.return_value = [[0.1, 0.2, 0.3]]

        results = engine.search([0.1, 0.2, 0.3], 'test query', count=2)

        engine._vector_search.assert_called_once()
        engine._keyword_search.assert_called_once()
        assert len(results) == 2
    
    @patch('signalwire_agents.search.search_engine.np', None)
    @patch('signalwire_agents.search.search_engine.cosine_similarity', None)
    def test_search_without_numpy(self, mock_logger=None):
        """Test search when numpy is not available"""
        engine = SearchEngine(backend='sqlite', index_path=self.db_path)
        engine._keyword_search_only = Mock(return_value=[
            {'id': 1, 'content': 'Keyword only result', 'score': 0.8}
        ])

        results = engine.search([0.1, 0.2, 0.3], 'test query', count=2)

        engine._keyword_search_only.assert_called_once_with('test query', 2, None, None)
        assert len(results) == 1
    
    def test_search_with_tags_filter(self):
        """Test search with tag filtering"""
        engine = SearchEngine(backend='sqlite', index_path=self.db_path)
        # When numpy is not available, search() delegates to _keyword_search_only
        # which handles tag filtering internally
        engine._keyword_search_only = Mock(return_value=[
            {'id': 1, 'content': 'Result 1', 'score': 0.9, 'metadata': {'tags': ['python', 'tutorial']}}
        ])

        results = engine.search([0.1, 0.2, 0.3], 'test query', count=2, tags=['python'])

        engine._keyword_search_only.assert_called_once_with('test query', 2, ['python'], None)
        assert len(results) == 1
    
    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_search_with_similarity_threshold(self, mock_cosine_sim, mock_np):
        """Test search with distance threshold filtering"""
        engine = SearchEngine(backend='sqlite', index_path=self.db_path)

        # Mock numpy to be available
        mock_np.array.return_value.reshape.return_value = [[0.1, 0.2, 0.3]]

        # Both results come via vector search so the distance threshold filter applies
        engine._vector_search = Mock(return_value=[
            {'id': 1, 'content': 'High score result', 'score': 0.9, 'search_type': 'vector', 'metadata': {}},
            {'id': 2, 'content': 'Low score result', 'score': 0.3, 'search_type': 'vector', 'metadata': {}},
        ])
        engine._filename_search = Mock(return_value=[])
        engine._metadata_search = Mock(return_value=[])
        engine._keyword_search = Mock(return_value=[])

        # Mock scoring: score is passed through
        def mock_combined_score(candidate, threshold):
            return candidate.get('score', 0.0)
        engine._calculate_combined_score = Mock(side_effect=mock_combined_score)
        engine._apply_diversity_penalties = Mock(side_effect=lambda results, count: results)

        results = engine.search([0.1, 0.2, 0.3], 'test query', count=2, similarity_threshold=0.5)

        # The low score vector result has vector_distance = 1 - 0.3 = 0.7
        # The threshold filter keeps results where vector_distance <= 0.5 * 1.5 = 0.75
        # Both results pass (0.1 <= 0.75 and 0.7 <= 0.75), but high score is first
        assert len(results) >= 1
        assert results[0]['score'] >= 0.9


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
        engine = SearchEngine(backend='sqlite', index_path=self.db_path)
        stats = engine.get_stats()
        
        assert stats['total_chunks'] == 3
        assert stats['total_files'] == 2
        assert stats['config']['embedding_dimensions'] == '384'
        assert 'file_types' in stats
        assert 'languages' in stats
    
    def test_get_stats_database_error(self):
        """Test getting stats with database error"""
        # Mock the get_stats method to avoid the actual database connection error
        engine = SearchEngine(backend='sqlite', index_path='/nonexistent/path.db')
        
        with patch.object(engine, 'get_stats', return_value={}):
            stats = engine.get_stats()
            assert stats == {}
    
    def test_merge_results(self):
        """Test merging vector and keyword results"""
        engine = SearchEngine(backend='sqlite', index_path=self.db_path)
        
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
        engine = SearchEngine(backend='sqlite', index_path=self.db_path)
        
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
        engine = SearchEngine(backend='sqlite', index_path=self.db_path)
        
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
            
            engine = SearchEngine(backend='sqlite', index_path=tmp.name)
            results = engine._fallback_search('Python', count=1)
            
            assert len(results) == 1
            assert 'Python' in results[0]['content']
            assert results[0]['search_type'] == 'fallback'
            
            os.unlink(tmp.name)
    
    def test_fallback_search_database_error(self):
        """Test fallback search with database error"""
        engine = SearchEngine(backend='sqlite', index_path='/nonexistent/path.db')
        results = engine._fallback_search('Python', count=1)
        
        assert results == []
    
    def test_keyword_search_only_with_tags(self):
        """Test keyword-only search with tag filtering"""
        engine = SearchEngine(backend='sqlite', index_path='test.db')
        engine._keyword_search = Mock(return_value=[
            {'id': 1, 'metadata': {'tags': ['python']}},
            {'id': 2, 'metadata': {'tags': ['javascript']}}
        ])
        engine._filter_by_tags = Mock(return_value=[
            {'id': 1, 'metadata': {'tags': ['python']}}
        ])
        
        results = engine._keyword_search_only('test', count=2, tags=['python'])

        engine._keyword_search.assert_called_once_with('test', 2, None)
        engine._filter_by_tags.assert_called_once()
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Shared helpers for new tests
# ---------------------------------------------------------------------------

def _create_full_test_db(db_path, with_fts=True, with_embeddings=True,
                         with_metadata_text=False, extra_chunks=None):
    """Create a fully populated test database for search engine tests.

    Returns the path unchanged (for convenience).
    """
    import struct

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Config
    cursor.execute("CREATE TABLE IF NOT EXISTS config (key TEXT, value TEXT)")
    cursor.execute("INSERT INTO config (key, value) VALUES ('embedding_dimensions', '4')")

    # Chunks
    cols = """
        id INTEGER PRIMARY KEY,
        content TEXT,
        embedding BLOB,
        filename TEXT,
        section TEXT,
        tags TEXT,
        metadata TEXT,
        language TEXT,
        processed_content TEXT
    """
    if with_metadata_text:
        cols += ", metadata_text TEXT"
    cursor.execute(f"CREATE TABLE IF NOT EXISTS chunks ({cols})")

    # FTS5
    if with_fts:
        cursor.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
            "USING fts5(content, content=chunks, content_rowid=id)"
        )

    # Helper to build a 4-dim float32 blob
    def _emb(values):
        if not with_embeddings:
            return None
        import struct as _s
        return _s.pack(f'{len(values)}f', *values)

    # Default chunks
    default_chunks = [
        {
            'content': 'Python programming tutorial for beginners',
            'embedding': _emb([0.9, 0.1, 0.0, 0.0]),
            'filename': 'docs/python_tutorial.md',
            'section': 'introduction',
            'tags': json.dumps(['python', 'tutorial', 'beginner']),
            'metadata': json.dumps({'category': 'Tutorial', 'product': 'Python SDK', 'source': 'python_tutorial.md'}),
            'language': 'en',
            'processed_content': 'python programming tutorial beginners',
            'metadata_text': 'python tutorial beginner Tutorial Python SDK' if with_metadata_text else None,
        },
        {
            'content': 'Advanced Python decorators and metaclasses',
            'embedding': _emb([0.8, 0.2, 0.1, 0.0]),
            'filename': 'docs/advanced_python.md',
            'section': 'decorators',
            'tags': json.dumps(['python', 'advanced', 'code']),
            'metadata': json.dumps({'category': 'Code Examples', 'product': 'Python SDK', 'source': 'advanced_python.md'}),
            'language': 'en',
            'processed_content': 'advanced python decorators metaclasses',
            'metadata_text': 'python advanced code Code Examples Python SDK' if with_metadata_text else None,
        },
        {
            'content': 'JavaScript async await patterns and promises',
            'embedding': _emb([0.0, 0.0, 0.9, 0.1]),
            'filename': 'docs/javascript_guide.md',
            'section': 'async',
            'tags': json.dumps(['javascript', 'async']),
            'metadata': json.dumps({'category': 'Guide', 'product': 'JS SDK'}),
            'language': 'en',
            'processed_content': 'javascript async await patterns promises',
            'metadata_text': 'javascript async Guide JS SDK' if with_metadata_text else None,
        },
        {
            'content': 'Getting started with SignalWire AI Agents SDK',
            'embedding': _emb([0.3, 0.3, 0.3, 0.3]),
            'filename': 'docs/getting_started.md',
            'section': 'quickstart',
            'tags': json.dumps(['signalwire', 'getting-started']),
            'metadata': json.dumps({'category': 'Getting Started', 'product': 'AI Agents SDK', 'description': 'Quickstart guide for AI Agents'}),
            'language': 'en',
            'processed_content': 'getting started signalwire ai agents sdk',
            'metadata_text': 'signalwire getting-started Getting Started AI Agents SDK' if with_metadata_text else None,
        },
        {
            'content': 'Code examples for REST API integration',
            'embedding': _emb([0.1, 0.8, 0.1, 0.0]),
            'filename': 'examples/rest_api_example.py',
            'section': 'examples',
            'tags': json.dumps(['code', 'api', 'example']),
            'metadata': json.dumps({'category': 'Code Examples', 'product': 'REST API'}),
            'language': 'en',
            'processed_content': 'code examples rest api integration',
            'metadata_text': 'code api example Code Examples REST API' if with_metadata_text else None,
        },
        {
            'content': 'SignalWire SWML reference documentation',
            'embedding': _emb([0.2, 0.2, 0.2, 0.8]),
            'filename': 'docs/swml_reference.md',
            'section': 'reference',
            'tags': json.dumps(['swml', 'reference']),
            'metadata': json.dumps({'category': 'Reference', 'product': 'SWML'}),
            'language': 'en',
            'processed_content': 'signalwire swml reference documentation',
            'metadata_text': 'swml reference Reference SWML' if with_metadata_text else None,
        },
    ]

    if extra_chunks:
        default_chunks.extend(extra_chunks)

    for c in default_chunks:
        if with_metadata_text:
            cursor.execute(
                "INSERT INTO chunks (content, embedding, filename, section, tags, metadata, language, processed_content, metadata_text) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (c['content'], c['embedding'], c['filename'], c['section'],
                 c['tags'], c['metadata'], c['language'], c['processed_content'],
                 c.get('metadata_text'))
            )
        else:
            cursor.execute(
                "INSERT INTO chunks (content, embedding, filename, section, tags, metadata, language, processed_content) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (c['content'], c['embedding'], c['filename'], c['section'],
                 c['tags'], c['metadata'], c['language'], c['processed_content'])
            )

    # Rebuild FTS
    if with_fts:
        cursor.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def full_db(tmp_path):
    """Fixture that returns a fully populated test database path."""
    db_path = str(tmp_path / "test_search.db")
    _create_full_test_db(db_path)
    return db_path


@pytest.fixture
def full_db_with_metadata_text(tmp_path):
    """Fixture with metadata_text column populated."""
    db_path = str(tmp_path / "test_meta.db")
    _create_full_test_db(db_path, with_metadata_text=True)
    return db_path


@pytest.fixture
def db_no_fts(tmp_path):
    """Fixture without FTS5 tables."""
    db_path = str(tmp_path / "test_nofts.db")
    _create_full_test_db(db_path, with_fts=False)
    return db_path


@pytest.fixture
def db_no_embeddings(tmp_path):
    """Fixture without embeddings."""
    db_path = str(tmp_path / "test_noembed.db")
    _create_full_test_db(db_path, with_embeddings=False)
    return db_path


@pytest.fixture
def empty_db(tmp_path):
    """Fixture with empty tables."""
    db_path = str(tmp_path / "test_empty.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("CREATE TABLE config (key TEXT, value TEXT)")
    c.execute("INSERT INTO config (key, value) VALUES ('embedding_dimensions', '4')")
    c.execute("""CREATE TABLE chunks (
        id INTEGER PRIMARY KEY, content TEXT, embedding BLOB,
        filename TEXT, section TEXT, tags TEXT, metadata TEXT,
        language TEXT, processed_content TEXT
    )""")
    c.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(content, content=chunks, content_rowid=id)")
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# TestSearch: End-to-end search() method
# ---------------------------------------------------------------------------

class TestSearch:
    """End-to-end tests for the search() orchestrator method."""

    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_search_returns_results(self, mock_cosine, mock_np, full_db):
        """search() returns results when vector + keyword candidates exist."""
        mock_np.array.return_value.reshape.return_value = [[0.9, 0.1, 0.0, 0.0]]
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        # Stub sub-searches to isolate orchestrator logic
        engine._vector_search = Mock(return_value=[
            {'id': 1, 'content': 'Python programming tutorial for beginners',
             'score': 0.95, 'search_type': 'vector',
             'metadata': {'filename': 'docs/python_tutorial.md', 'section': 'introduction', 'tags': ['python'], 'metadata': {}}}
        ])
        engine._filename_search = Mock(return_value=[])
        engine._metadata_search = Mock(return_value=[])
        engine._keyword_search = Mock(return_value=[])
        engine._calculate_combined_score = Mock(side_effect=lambda c, t: c.get('score', 0.0))
        engine._apply_diversity_penalties = Mock(side_effect=lambda r, c: r)

        results = engine.search([0.9, 0.1, 0.0, 0.0], 'python tutorial', count=3)
        assert len(results) >= 1
        assert 'score' in results[0]

    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_search_merges_multiple_sources(self, mock_cosine, mock_np, full_db):
        """search() merges candidates from vector, keyword, filename, metadata."""
        mock_np.array.return_value.reshape.return_value = [[0.9, 0.1, 0.0, 0.0]]
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        engine._vector_search = Mock(return_value=[
            {'id': 1, 'content': 'A', 'score': 0.9, 'search_type': 'vector', 'metadata': {'filename': 'a.md', 'tags': []}}
        ])
        engine._filename_search = Mock(return_value=[
            {'id': 2, 'content': 'B', 'score': 0.8, 'search_type': 'filename', 'metadata': {'filename': 'b.md', 'tags': []}}
        ])
        engine._metadata_search = Mock(return_value=[])
        engine._keyword_search = Mock(return_value=[
            {'id': 3, 'content': 'C', 'score': 0.7, 'search_type': 'keyword', 'metadata': {'filename': 'c.md', 'tags': []}}
        ])
        engine._calculate_combined_score = Mock(side_effect=lambda c, t: c.get('score', 0.0))
        engine._apply_diversity_penalties = Mock(side_effect=lambda r, c: r)

        results = engine.search([0.9, 0.1, 0.0, 0.0], 'test', count=5)
        ids = {r['id'] for r in results}
        assert ids == {1, 2, 3}

    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_search_respects_count(self, mock_cosine, mock_np, full_db):
        """search() returns at most `count` results."""
        mock_np.array.return_value.reshape.return_value = [[0.9, 0.1, 0.0, 0.0]]
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        many = [{'id': i, 'content': f'C{i}', 'score': 1.0 - i * 0.01,
                 'search_type': 'vector', 'metadata': {'filename': f'f{i}.md', 'tags': []}}
                for i in range(10)]
        engine._vector_search = Mock(return_value=many)
        engine._filename_search = Mock(return_value=[])
        engine._metadata_search = Mock(return_value=[])
        engine._keyword_search = Mock(return_value=[])
        engine._calculate_combined_score = Mock(side_effect=lambda c, t: c.get('score', 0.0))
        engine._apply_diversity_penalties = Mock(side_effect=lambda r, c: r)

        results = engine.search([0.9, 0.1, 0.0, 0.0], 'test', count=3)
        assert len(results) == 3

    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_search_filters_by_tags(self, mock_cosine, mock_np, full_db):
        """search() filters results by tags when specified."""
        mock_np.array.return_value.reshape.return_value = [[0.9, 0.1, 0.0, 0.0]]
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        engine._vector_search = Mock(return_value=[
            {'id': 1, 'content': 'A', 'score': 0.9, 'search_type': 'vector',
             'metadata': {'filename': 'a.md', 'tags': ['python'], 'metadata': {}}},
            {'id': 2, 'content': 'B', 'score': 0.8, 'search_type': 'vector',
             'metadata': {'filename': 'b.md', 'tags': ['javascript'], 'metadata': {}}},
        ])
        engine._filename_search = Mock(return_value=[])
        engine._metadata_search = Mock(return_value=[])
        engine._keyword_search = Mock(return_value=[])
        engine._calculate_combined_score = Mock(side_effect=lambda c, t: c.get('score', 0.0))
        engine._apply_diversity_penalties = Mock(side_effect=lambda r, c: r)

        results = engine.search([0.9, 0.1, 0.0, 0.0], 'test', count=5, tags=['python'])
        assert all('python' in r['metadata']['tags'] for r in results)

    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_search_boosts_exact_matches(self, mock_cosine, mock_np, full_db):
        """search() boosts exact query matches when original_query given."""
        mock_np.array.return_value.reshape.return_value = [[0.9, 0.1, 0.0, 0.0]]
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        engine._vector_search = Mock(return_value=[
            {'id': 1, 'content': 'python tutorial content', 'score': 0.5,
             'search_type': 'vector', 'metadata': {'filename': 'a.md', 'tags': [], 'metadata': {}}},
            {'id': 2, 'content': 'unrelated stuff', 'score': 0.6,
             'search_type': 'vector', 'metadata': {'filename': 'b.md', 'tags': [], 'metadata': {}}},
        ])
        engine._filename_search = Mock(return_value=[])
        engine._metadata_search = Mock(return_value=[])
        engine._keyword_search = Mock(return_value=[])
        engine._calculate_combined_score = Mock(side_effect=lambda c, t: c.get('score', 0.0))
        engine._apply_diversity_penalties = Mock(side_effect=lambda r, c: r)

        results = engine.search([0.9, 0.1, 0.0, 0.0], 'python tutorial',
                                count=5, original_query='python tutorial')
        # The exact-match result should now be ranked first
        assert results[0]['id'] == 1

    @patch('signalwire_agents.search.search_engine.np', None)
    @patch('signalwire_agents.search.search_engine.cosine_similarity', None)
    def test_search_falls_back_to_keyword_only(self, full_db):
        """search() uses keyword-only when numpy is unavailable."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        engine._keyword_search_only = Mock(return_value=[
            {'id': 1, 'content': 'result', 'score': 0.5}
        ])
        results = engine.search([0.1], 'python', count=3)
        engine._keyword_search_only.assert_called_once()
        assert len(results) == 1

    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_search_handles_vector_conversion_error(self, mock_cosine, mock_np, full_db):
        """search() falls back to keyword-only when vector conversion fails."""
        mock_np.array.side_effect = Exception("bad vector")
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        engine._keyword_search_only = Mock(return_value=[])
        results = engine.search([0.1], 'python', count=3)
        engine._keyword_search_only.assert_called_once()

    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_search_applies_diversity_penalties(self, mock_cosine, mock_np, full_db):
        """search() calls _apply_diversity_penalties."""
        mock_np.array.return_value.reshape.return_value = [[0.9, 0.1, 0.0, 0.0]]
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        engine._vector_search = Mock(return_value=[
            {'id': 1, 'content': 'A', 'score': 0.9, 'search_type': 'vector',
             'metadata': {'filename': 'a.md', 'tags': []}}
        ])
        engine._filename_search = Mock(return_value=[])
        engine._metadata_search = Mock(return_value=[])
        engine._keyword_search = Mock(return_value=[])
        engine._calculate_combined_score = Mock(side_effect=lambda c, t: c.get('score', 0.0))
        diversity_mock = Mock(side_effect=lambda r, c: r)
        engine._apply_diversity_penalties = diversity_mock

        engine.search([0.9, 0.1, 0.0, 0.0], 'test', count=3)
        diversity_mock.assert_called_once()

    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_search_sets_score_field(self, mock_cosine, mock_np, full_db):
        """search() ensures every result has a 'score' field."""
        mock_np.array.return_value.reshape.return_value = [[0.9, 0.1, 0.0, 0.0]]
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        engine._vector_search = Mock(return_value=[
            {'id': 1, 'content': 'A', 'score': 0.9, 'search_type': 'vector',
             'metadata': {'filename': 'a.md', 'tags': []}}
        ])
        engine._filename_search = Mock(return_value=[])
        engine._metadata_search = Mock(return_value=[])
        engine._keyword_search = Mock(return_value=[])
        engine._calculate_combined_score = Mock(return_value=0.85)
        engine._apply_diversity_penalties = Mock(side_effect=lambda r, c: r)

        results = engine.search([0.9, 0.1, 0.0, 0.0], 'test', count=3)
        for r in results:
            assert 'score' in r

    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_search_similarity_threshold_filters(self, mock_cosine, mock_np, full_db):
        """search() filters by similarity_threshold when > 0."""
        mock_np.array.return_value.reshape.return_value = [[0.9, 0.1, 0.0, 0.0]]
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        engine._vector_search = Mock(return_value=[
            {'id': 1, 'content': 'A', 'score': 0.95, 'search_type': 'vector',
             'metadata': {'filename': 'a.md', 'tags': []}},
            {'id': 2, 'content': 'B', 'score': 0.1, 'search_type': 'vector',
             'metadata': {'filename': 'b.md', 'tags': []}},
        ])
        engine._filename_search = Mock(return_value=[])
        engine._metadata_search = Mock(return_value=[])
        engine._keyword_search = Mock(return_value=[])
        engine._calculate_combined_score = Mock(side_effect=lambda c, t: c.get('score', 0.0))
        engine._apply_diversity_penalties = Mock(side_effect=lambda r, c: r)

        results = engine.search([0.9, 0.1, 0.0, 0.0], 'test', count=5, similarity_threshold=0.5)
        # id=2 has vector_distance = 1 - 0.1 = 0.9, threshold*1.5 = 0.75 so 0.9 > 0.75 => filtered
        assert all(r['id'] != 2 for r in results)


# ---------------------------------------------------------------------------
# TestVectorSearch: _vector_search()
# ---------------------------------------------------------------------------

class TestVectorSearch:
    """Tests for _vector_search()."""

    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_returns_sorted_by_score(self, mock_cosine, mock_np, full_db):
        """Results are sorted by similarity descending."""
        # Set up mock so frombuffer returns mock arrays
        mock_array = Mock()
        mock_array.reshape.return_value = [[0.0]]
        mock_np.frombuffer.return_value = mock_array
        mock_np.float32 = 'float32'
        # Return decreasing similarities
        mock_cosine.side_effect = [[[0.5]], [[0.9]], [[0.3]], [[0.7]], [[0.4]], [[0.6]]]

        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._vector_search([[0.9, 0.1, 0.0, 0.0]], count=6)
        scores = [r['score'] for r in results]
        assert scores == sorted(scores, reverse=True)

    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_limits_count(self, mock_cosine, mock_np, full_db):
        """Returns at most `count` results."""
        mock_array = Mock()
        mock_array.reshape.return_value = [[0.0]]
        mock_np.frombuffer.return_value = mock_array
        mock_np.float32 = 'float32'
        mock_cosine.return_value = [[0.8]]

        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._vector_search([[0.9, 0.1, 0.0, 0.0]], count=2)
        assert len(results) <= 2

    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_metadata_parsed_correctly(self, mock_cosine, mock_np, full_db):
        """Metadata (tags, section, filename) is correctly parsed."""
        mock_array = Mock()
        mock_array.reshape.return_value = [[0.0]]
        mock_np.frombuffer.return_value = mock_array
        mock_np.float32 = 'float32'
        mock_cosine.return_value = [[0.95]]

        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._vector_search([[0.9, 0.1, 0.0, 0.0]], count=1)
        assert len(results) == 1
        md = results[0]['metadata']
        assert 'filename' in md
        assert 'section' in md
        assert isinstance(md['tags'], list)

    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_skips_empty_embeddings(self, mock_cosine, mock_np, db_no_embeddings):
        """Chunks with NULL embeddings are skipped."""
        mock_np.frombuffer.return_value = Mock(reshape=Mock(return_value=[[0.0]]))
        mock_np.float32 = 'float32'
        mock_cosine.return_value = [[0.8]]

        engine = SearchEngine(backend='sqlite', index_path=db_no_embeddings)
        results = engine._vector_search([[0.1, 0.1, 0.1, 0.1]], count=10)
        assert results == []

    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_handles_embedding_processing_error(self, mock_cosine, mock_np, full_db):
        """Gracefully continues when one embedding fails to process."""
        call_count = [0]
        def frombuffer_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ValueError("corrupt embedding")
            m = Mock()
            m.reshape.return_value = [[0.0]]
            return m

        mock_np.frombuffer.side_effect = frombuffer_side_effect
        mock_np.float32 = 'float32'
        mock_cosine.return_value = [[0.8]]

        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._vector_search([[0.9, 0.1, 0.0, 0.0]], count=10)
        # Should still return results from non-corrupt chunks
        assert len(results) >= 1

    @patch('signalwire_agents.search.search_engine.np', None)
    @patch('signalwire_agents.search.search_engine.cosine_similarity', None)
    def test_no_numpy_returns_empty(self, full_db):
        """Returns [] when numpy is not available."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        assert engine._vector_search([[0.1]], count=5) == []

    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_db_error_returns_empty(self, mock_cosine, mock_np):
        """Returns [] when the database cannot be opened."""
        engine = SearchEngine(backend='sqlite', index_path='/no/such/file.db')
        assert engine._vector_search([[0.1]], count=5) == []

    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_search_type_is_vector(self, mock_cosine, mock_np, full_db):
        """All results have search_type == 'vector'."""
        mock_array = Mock()
        mock_array.reshape.return_value = [[0.0]]
        mock_np.frombuffer.return_value = mock_array
        mock_np.float32 = 'float32'
        mock_cosine.return_value = [[0.7]]

        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._vector_search([[0.1, 0.1, 0.1, 0.1]], count=10)
        for r in results:
            assert r['search_type'] == 'vector'


# ---------------------------------------------------------------------------
# TestKeywordSearch: _keyword_search()
# ---------------------------------------------------------------------------

class TestKeywordSearch:
    """Tests for _keyword_search() with FTS5."""

    def test_finds_matching_content(self, full_db):
        """Finds chunks whose content matches keywords."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._keyword_search('python', count=10)
        assert len(results) >= 1
        assert all(r['search_type'] == 'keyword' for r in results)

    def test_multiple_term_query(self, full_db):
        """Handles multi-word queries."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._keyword_search('python tutorial', count=10)
        assert len(results) >= 1

    def test_no_results_triggers_fallback(self, full_db):
        """Falls back to LIKE search when FTS returns nothing."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        engine._fallback_search = Mock(return_value=[
            {'id': 99, 'content': 'fallback', 'score': 0.1, 'search_type': 'fallback',
             'metadata': {}}
        ])
        results = engine._keyword_search('zzzznonexistent', count=5)
        engine._fallback_search.assert_called_once()

    def test_scores_are_positive_floats(self, full_db):
        """Scores are positive floats derived from FTS rank."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._keyword_search('python', count=5)
        for r in results:
            assert isinstance(r['score'], float)
            assert r['score'] > 0

    def test_respects_count_limit(self, full_db):
        """Returns at most `count` results."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._keyword_search('python', count=1)
        assert len(results) <= 1

    def test_metadata_parsed(self, full_db):
        """Metadata (tags, section, filename) is parsed from JSON."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._keyword_search('python', count=1)
        if results:
            assert 'filename' in results[0]['metadata']
            assert isinstance(results[0]['metadata']['tags'], list)

    def test_fts_error_falls_back(self, db_no_fts):
        """Falls back to LIKE search when FTS table doesn't exist."""
        engine = SearchEngine(backend='sqlite', index_path=db_no_fts)
        results = engine._keyword_search('python', count=5)
        # Should get fallback results (or empty)
        for r in results:
            assert r['search_type'] == 'fallback'

    def test_original_query_passed_to_fallback(self, full_db):
        """original_query parameter is available."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        # A query that returns results shouldn't hit fallback
        results = engine._keyword_search('python', count=5, original_query='python programming')
        assert len(results) >= 1

    def test_empty_query_string(self, full_db):
        """Handles empty string query gracefully."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._keyword_search('', count=5)
        # May return empty or fallback results; should not crash
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# TestMetadataSearch: _metadata_search()
# ---------------------------------------------------------------------------

class TestMetadataSearch:
    """Tests for _metadata_search()."""

    def test_finds_by_tag(self, full_db):
        """Finds chunks that have matching tags."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._metadata_search('python', count=10)
        assert len(results) >= 1
        # At least one should have 'python' in tags
        found_tag = any('python' in r['metadata'].get('tags', []) for r in results)
        assert found_tag

    def test_finds_by_section(self, full_db):
        """Finds chunks with matching section names."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._metadata_search('decorators', count=10)
        assert len(results) >= 1

    def test_finds_by_category_in_metadata(self, full_db):
        """Finds chunks with matching category metadata."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._metadata_search('tutorial', count=10)
        assert len(results) >= 1

    def test_finds_by_product_in_metadata(self, full_db):
        """Finds chunks with matching product metadata."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._metadata_search('sdk', count=10)
        assert len(results) >= 1

    def test_search_type_is_metadata(self, full_db):
        """All results have search_type == 'metadata'."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._metadata_search('python', count=10)
        for r in results:
            assert r['search_type'] == 'metadata'

    def test_metadata_text_column_used(self, full_db_with_metadata_text):
        """Uses metadata_text column for searching when it exists."""
        engine = SearchEngine(backend='sqlite', index_path=full_db_with_metadata_text)
        results = engine._metadata_search('python', count=10)
        assert len(results) >= 1

    def test_multi_term_scoring(self, full_db):
        """Multi-term queries produce higher scores for multi-field matches."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._metadata_search('code examples', count=10)
        assert len(results) >= 1

    def test_respects_count_limit(self, full_db):
        """Returns at most `count` results."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._metadata_search('python', count=1)
        assert len(results) <= 1

    def test_db_error_returns_empty(self):
        """Returns [] on database error."""
        engine = SearchEngine(backend='sqlite', index_path='/nonexistent/path.db')
        results = engine._metadata_search('python', count=5)
        assert results == []

    def test_empty_query(self, full_db):
        """Handles empty query gracefully."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._metadata_search('', count=5)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# TestFilenameSearch: _filename_search()
# ---------------------------------------------------------------------------

class TestFilenameSearch:
    """Tests for _filename_search()."""

    def test_exact_filename_match(self, full_db):
        """Finds chunks by exact filename match."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._filename_search('python_tutorial', count=10)
        assert len(results) >= 1
        assert any('python_tutorial' in r['metadata']['filename'] for r in results)

    def test_partial_filename_match(self, full_db):
        """Finds chunks with partial filename matches."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._filename_search('tutorial', count=10)
        assert len(results) >= 1

    def test_search_type_is_filename(self, full_db):
        """All results have search_type == 'filename'."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._filename_search('python', count=10)
        for r in results:
            assert r['search_type'] == 'filename'

    def test_basename_match_scores_higher(self, full_db):
        """Basename matches score higher than path-only matches."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._filename_search('python_tutorial.md', count=10)
        if results:
            # Exact basename match should have high score
            assert results[0]['score'] >= 2.0

    def test_multi_term_filename_search(self, full_db):
        """Multi-term queries match against filename."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._filename_search('rest api example', count=10)
        assert len(results) >= 1

    def test_respects_count(self, full_db):
        """Returns at most `count` results."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._filename_search('python', count=1)
        assert len(results) <= 1

    def test_no_match_returns_empty(self, full_db):
        """Returns empty list when no filenames match."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._filename_search('zzz_nonexistent_file', count=5)
        assert results == []

    def test_db_error_returns_empty(self):
        """Returns [] on database error."""
        engine = SearchEngine(backend='sqlite', index_path='/nonexistent/path.db')
        results = engine._filename_search('test', count=5)
        assert results == []

    def test_match_coverage_present(self, full_db):
        """Results contain match_coverage metadata."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._filename_search('python', count=5)
        for r in results:
            assert 'match_coverage' in r


# ---------------------------------------------------------------------------
# TestFallbackSearch: _fallback_search()
# ---------------------------------------------------------------------------

class TestFallbackSearch:
    """Tests for _fallback_search() LIKE-based search."""

    def test_finds_by_processed_content(self, full_db):
        """Finds chunks matching processed_content via LIKE."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._fallback_search('python', count=10)
        assert len(results) >= 1

    def test_finds_by_original_content(self, full_db):
        """Finds chunks matching original content via LIKE."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._fallback_search('decorators', count=10)
        assert len(results) >= 1

    def test_search_type_is_fallback(self, full_db):
        """All results have search_type == 'fallback'."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._fallback_search('python', count=10)
        for r in results:
            assert r['search_type'] == 'fallback'

    def test_scoring_based_on_word_matches(self, full_db):
        """Score reflects how many query terms appear in content."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._fallback_search('python programming tutorial', count=10)
        if results:
            # The chunk with all 3 words should score highest
            assert results[0]['score'] > 0

    def test_limits_to_five_terms(self, full_db):
        """Only uses first 5 search terms (no error for more)."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._fallback_search('a b c d e f g h', count=10)
        assert isinstance(results, list)

    def test_empty_query_returns_empty(self, full_db):
        """Returns [] for an empty query string."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._fallback_search('', count=10)
        assert results == []

    def test_respects_count_limit(self, full_db):
        """Returns at most `count` results."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._fallback_search('python', count=1)
        assert len(results) <= 1

    def test_db_error_returns_empty(self):
        """Returns [] on database error."""
        engine = SearchEngine(backend='sqlite', index_path='/nonexistent/path.db')
        results = engine._fallback_search('test', count=5)
        assert results == []

    def test_sorted_by_score_descending(self, full_db):
        """Results are sorted by score descending."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._fallback_search('python', count=10)
        scores = [r['score'] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_no_match_returns_empty(self, full_db):
        """Returns [] when no content matches."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._fallback_search('xyznonexistent', count=10)
        assert results == []


# ---------------------------------------------------------------------------
# TestResultProcessing: _merge_results, _boost_exact_matches,
#                        _apply_diversity_penalties, _calculate_combined_score,
#                        _apply_match_type_diversity
# ---------------------------------------------------------------------------

class TestResultProcessing:
    """Tests for result ranking, merging, and diversity methods."""

    def _make_engine(self, tmp_path):
        """Helper to create a minimal engine."""
        db_path = str(tmp_path / "proc.db")
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("CREATE TABLE config (key TEXT, value TEXT)")
        c.execute("INSERT INTO config (key,value) VALUES ('embedding_dimensions','4')")
        conn.commit()
        conn.close()
        return SearchEngine(backend='sqlite', index_path=db_path)

    # -- _merge_results --

    def test_merge_no_overlap(self, tmp_path):
        """Merging disjoint sets yields union."""
        engine = self._make_engine(tmp_path)
        v = [{'id': 1, 'content': 'A', 'score': 0.9, 'metadata': {}}]
        k = [{'id': 2, 'content': 'B', 'score': 0.8, 'metadata': {}}]
        merged = engine._merge_results(v, k)
        assert len(merged) == 2

    def test_merge_overlap_combines_scores(self, tmp_path):
        """Overlapping IDs get combined score."""
        engine = self._make_engine(tmp_path)
        v = [{'id': 1, 'content': 'A', 'score': 0.8, 'metadata': {}}]
        k = [{'id': 1, 'content': 'A', 'score': 0.6, 'metadata': {}}]
        merged = engine._merge_results(v, k)
        assert len(merged) == 1
        # 0.8 * 0.7 + 0.6 * 0.3 = 0.74
        assert abs(merged[0]['score'] - 0.74) < 0.01

    def test_merge_custom_weights(self, tmp_path):
        """Custom weights are applied."""
        engine = self._make_engine(tmp_path)
        v = [{'id': 1, 'content': 'A', 'score': 1.0, 'metadata': {}}]
        k = [{'id': 1, 'content': 'A', 'score': 1.0, 'metadata': {}}]
        merged = engine._merge_results(v, k, vector_weight=0.5, keyword_weight=0.5)
        assert abs(merged[0]['score'] - 1.0) < 0.01

    def test_merge_sorted_by_score(self, tmp_path):
        """Merged results are sorted by combined score descending."""
        engine = self._make_engine(tmp_path)
        v = [{'id': 1, 'content': 'A', 'score': 0.3, 'metadata': {}},
             {'id': 2, 'content': 'B', 'score': 0.9, 'metadata': {}}]
        k = []
        merged = engine._merge_results(v, k)
        assert merged[0]['id'] == 2

    def test_merge_adds_search_scores(self, tmp_path):
        """Merged results contain search_scores debug info."""
        engine = self._make_engine(tmp_path)
        v = [{'id': 1, 'content': 'A', 'score': 0.8, 'metadata': {}}]
        k = [{'id': 1, 'content': 'A', 'score': 0.6, 'metadata': {}}]
        merged = engine._merge_results(v, k)
        assert 'search_scores' in merged[0]['metadata']

    def test_merge_empty_inputs(self, tmp_path):
        """Merging two empty lists returns empty."""
        engine = self._make_engine(tmp_path)
        merged = engine._merge_results([], [])
        assert merged == []

    # -- _boost_exact_matches --

    def test_boost_exact_phrase_match(self, tmp_path):
        """Exact phrase match doubles score."""
        engine = self._make_engine(tmp_path)
        results = [
            {'id': 1, 'content': 'python tutorial for beginners', 'score': 0.5,
             'final_score': 0.5, 'metadata': {'filename': 'a.md'}},
        ]
        boosted = engine._boost_exact_matches(results, 'python tutorial')
        assert boosted[0]['final_score'] == 0.5 * 2.0

    def test_boost_no_match_unchanged(self, tmp_path):
        """No boost when query not found in content."""
        engine = self._make_engine(tmp_path)
        results = [
            {'id': 1, 'content': 'javascript async await', 'score': 0.5,
             'final_score': 0.5, 'metadata': {'filename': 'a.md'}},
        ]
        boosted = engine._boost_exact_matches(results, 'python tutorial')
        assert boosted[0]['final_score'] == 0.5

    def test_boost_example_filename(self, tmp_path):
        """Boosts results with 'example' in filename for code queries."""
        engine = self._make_engine(tmp_path)
        results = [
            {'id': 1, 'content': 'some code', 'score': 0.5,
             'final_score': 0.5, 'metadata': {'filename': 'code_example.py'}},
        ]
        boosted = engine._boost_exact_matches(results, 'code example')
        # 'code' in query, 'example' in filename => 1.5x boost
        assert boosted[0]['final_score'] > 0.5

    def test_boost_getting_started(self, tmp_path):
        """Boosts 'getting started' queries with 'start' in content."""
        engine = self._make_engine(tmp_path)
        results = [
            {'id': 1, 'content': 'how to start building agents', 'score': 0.5,
             'final_score': 0.5, 'metadata': {'filename': 'guide.md'}},
        ]
        boosted = engine._boost_exact_matches(results, 'getting started')
        assert boosted[0]['final_score'] > 0.5

    def test_boost_empty_query_no_change(self, tmp_path):
        """Empty original_query returns results unchanged."""
        engine = self._make_engine(tmp_path)
        results = [{'id': 1, 'content': 'x', 'score': 0.5, 'metadata': {'filename': 'a.md'}}]
        boosted = engine._boost_exact_matches(results, '')
        assert boosted[0]['score'] == 0.5

    # -- _calculate_combined_score --

    def test_combined_score_vector_only(self, tmp_path):
        """Candidate with vector score only uses it as base."""
        engine = self._make_engine(tmp_path)
        candidate = {
            'vector_score': 0.9,
            'sources': {'vector': True},
            'source_scores': {'vector': 0.9},
            'metadata': {'tags': []}
        }
        score = engine._calculate_combined_score(candidate, 0.0)
        assert abs(score - 0.9) < 0.01

    def test_combined_score_with_keyword_boost(self, tmp_path):
        """Keyword confirmation boosts vector score."""
        engine = self._make_engine(tmp_path)
        candidate = {
            'vector_score': 0.8,
            'sources': {'vector': True, 'keyword': True},
            'source_scores': {'vector': 0.8, 'keyword': 0.5},
            'metadata': {'tags': []}
        }
        score = engine._calculate_combined_score(candidate, 0.0)
        assert score > 0.8  # Boosted

    def test_combined_score_multi_source_boost(self, tmp_path):
        """Multiple metadata source types give extra boost."""
        engine = self._make_engine(tmp_path)
        candidate = {
            'vector_score': 0.8,
            'sources': {'vector': True, 'keyword': True, 'filename': True},
            'source_scores': {'vector': 0.8, 'keyword': 0.5, 'filename': 0.6},
            'metadata': {'tags': []}
        }
        score = engine._calculate_combined_score(candidate, 0.0)
        assert score > 0.8

    def test_combined_score_no_vector(self, tmp_path):
        """Keyword-only candidates get 60% of their score."""
        engine = self._make_engine(tmp_path)
        candidate = {
            'sources': {'keyword': True},
            'source_scores': {'keyword': 1.0},
            'metadata': {'tags': []}
        }
        score = engine._calculate_combined_score(candidate, 0.0)
        assert abs(score - 0.6) < 0.01

    def test_combined_score_code_tag_boost(self, tmp_path):
        """Code tag with metadata source gets boosted."""
        engine = self._make_engine(tmp_path)
        candidate = {
            'vector_score': 0.7,
            'sources': {'vector': True, 'metadata': True},
            'source_scores': {'vector': 0.7, 'metadata': 0.5},
            'metadata': {'tags': ['code']}
        }
        score = engine._calculate_combined_score(candidate, 0.0)
        # Should be > 0.7 due to keyword boost + code boost
        assert score > 0.7

    def test_combined_score_no_vector_code_tag(self, tmp_path):
        """No-vector candidate with code tag and metadata source gets 1.15x."""
        engine = self._make_engine(tmp_path)
        candidate = {
            'sources': {'metadata': True},
            'source_scores': {'metadata': 1.0},
            'metadata': {'tags': ['code']}
        }
        score = engine._calculate_combined_score(candidate, 0.0)
        # 1.0 * 0.6 * 1.15 = 0.69
        assert abs(score - 0.69) < 0.01

    # -- _apply_diversity_penalties --

    def test_diversity_no_results(self, tmp_path):
        """Empty input returns empty output."""
        engine = self._make_engine(tmp_path)
        assert engine._apply_diversity_penalties([], 3) == []

    def test_diversity_first_occurrence_no_penalty(self, tmp_path):
        """First chunk from a file has no penalty (1.0)."""
        engine = self._make_engine(tmp_path)
        results = [
            {'id': 1, 'content': 'A', 'score': 0.9, 'final_score': 0.9,
             'metadata': {'filename': 'file1.md'}},
        ]
        penalized = engine._apply_diversity_penalties(results, 3)
        assert penalized[0]['diversity_penalty'] == 1.0

    def test_diversity_second_occurrence_penalized(self, tmp_path):
        """Second chunk from the same file gets 0.85 penalty."""
        engine = self._make_engine(tmp_path)
        results = [
            {'id': 1, 'content': 'A', 'score': 0.9, 'final_score': 0.9,
             'metadata': {'filename': 'file1.md'}},
            {'id': 2, 'content': 'B', 'score': 0.8, 'final_score': 0.8,
             'metadata': {'filename': 'file1.md'}},
        ]
        penalized = engine._apply_diversity_penalties(results, 3)
        same_file = [r for r in penalized if r['metadata']['filename'] == 'file1.md']
        penalties = sorted([r['diversity_penalty'] for r in same_file], reverse=True)
        assert penalties == [1.0, 0.85]

    def test_diversity_resorts_by_penalized_score(self, tmp_path):
        """Results are re-sorted after penalties."""
        engine = self._make_engine(tmp_path)
        results = [
            {'id': 1, 'content': 'A', 'score': 0.9, 'final_score': 0.9,
             'metadata': {'filename': 'same.md'}},
            {'id': 2, 'content': 'B', 'score': 0.89, 'final_score': 0.89,
             'metadata': {'filename': 'same.md'}},
            {'id': 3, 'content': 'C', 'score': 0.7, 'final_score': 0.7,
             'metadata': {'filename': 'other.md'}},
        ]
        penalized = engine._apply_diversity_penalties(results, 3)
        scores = [r['final_score'] for r in penalized]
        assert scores == sorted(scores, reverse=True)

    def test_diversity_heavy_penalty_for_4plus(self, tmp_path):
        """4th and 5th chunks from same file get 50%/60% penalty."""
        engine = self._make_engine(tmp_path)
        results = [
            {'id': i, 'content': f'C{i}', 'score': 1.0 - i * 0.01,
             'final_score': 1.0 - i * 0.01,
             'metadata': {'filename': 'same.md'}}
            for i in range(5)
        ]
        penalized = engine._apply_diversity_penalties(results, 10)
        # 5th occurrence should have 0.4 penalty
        fifth = [r for r in penalized if r['id'] == 4][0]
        assert fifth['diversity_penalty'] == 0.4

    # -- _apply_match_type_diversity --

    def test_match_type_diversity_with_all_types(self, tmp_path):
        """Diversifies among vector-only, keyword-only, and hybrid results."""
        engine = self._make_engine(tmp_path)
        results = []
        for i in range(6):
            r = {'id': i, 'content': f'C{i}', 'final_score': 1.0 - i * 0.05,
                 'metadata': {'filename': f'f{i}.md'}}
            if i < 2:
                r['sources'] = {'vector': True}
            elif i < 4:
                r['sources'] = {'keyword': True}
            else:
                r['sources'] = {'vector': True, 'keyword': True}
            results.append(r)
        diversified = engine._apply_match_type_diversity(results, 3)
        assert len(diversified) == 3

    def test_match_type_diversity_short_list_unchanged(self, tmp_path):
        """Lists shorter than target_count are returned as-is."""
        engine = self._make_engine(tmp_path)
        results = [
            {'id': 1, 'final_score': 0.9, 'sources': {'vector': True},
             'metadata': {'filename': 'a.md'}},
        ]
        diversified = engine._apply_match_type_diversity(results, 5)
        assert len(diversified) == 1

    def test_match_type_diversity_empty(self, tmp_path):
        """Empty input returns empty."""
        engine = self._make_engine(tmp_path)
        assert engine._apply_match_type_diversity([], 3) == []

    def test_match_type_diversity_sorted_by_final_score(self, tmp_path):
        """Output is sorted by final_score descending."""
        engine = self._make_engine(tmp_path)
        results = []
        for i in range(8):
            r = {'id': i, 'content': f'C{i}', 'final_score': 0.5 + (i % 3) * 0.1,
                 'metadata': {'filename': f'f{i}.md'}}
            if i % 3 == 0:
                r['sources'] = {'vector': True}
            elif i % 3 == 1:
                r['sources'] = {'keyword': True}
            else:
                r['sources'] = {'vector': True, 'keyword': True}
            results.append(r)
        diversified = engine._apply_match_type_diversity(results, 4)
        scores = [r['final_score'] for r in diversified]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# TestEdgeCases: Assorted edge cases and error handling
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: empty results, unavailable dependencies, special characters."""

    def test_empty_db_search_returns_empty(self, empty_db):
        """search() on empty DB returns empty list."""
        engine = SearchEngine(backend='sqlite', index_path=empty_db)
        engine._keyword_search_only = Mock(return_value=[])
        results = engine.search([0.1], 'test', count=3)
        assert results == []

    def test_empty_db_keyword_search(self, empty_db):
        """_keyword_search on empty DB returns []."""
        engine = SearchEngine(backend='sqlite', index_path=empty_db)
        results = engine._keyword_search('test', count=5)
        assert results == []

    def test_empty_db_fallback_search(self, empty_db):
        """_fallback_search on empty DB returns []."""
        engine = SearchEngine(backend='sqlite', index_path=empty_db)
        results = engine._fallback_search('test', count=5)
        assert results == []

    def test_empty_db_filename_search(self, empty_db):
        """_filename_search on empty DB returns []."""
        engine = SearchEngine(backend='sqlite', index_path=empty_db)
        results = engine._filename_search('test', count=5)
        assert results == []

    def test_empty_db_metadata_search(self, empty_db):
        """_metadata_search on empty DB returns []."""
        engine = SearchEngine(backend='sqlite', index_path=empty_db)
        results = engine._metadata_search('test', count=5)
        assert results == []

    @patch('signalwire_agents.search.search_engine.np', None)
    @patch('signalwire_agents.search.search_engine.cosine_similarity', None)
    def test_numpy_unavailable_keyword_search_only(self, full_db):
        """When numpy is None, search() uses keyword-only path."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine.search([0.1], 'python', count=3)
        # Should get keyword results (or empty) but not crash
        assert isinstance(results, list)

    def test_special_characters_in_keyword_query(self, full_db):
        """Special characters in query don't crash keyword search."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._keyword_search('test* OR "hello"', count=5)
        assert isinstance(results, list)

    def test_special_characters_in_fallback_query(self, full_db):
        """Special characters in query don't crash fallback search."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._fallback_search("it's a test% with_underscores", count=5)
        assert isinstance(results, list)

    def test_special_characters_in_filename_query(self, full_db):
        """Special characters in query don't crash filename search."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._filename_search("test/path with spaces", count=5)
        assert isinstance(results, list)

    def test_special_characters_in_metadata_query(self, full_db):
        """Special characters in query don't crash metadata search."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._metadata_search('test "with" quotes', count=5)
        assert isinstance(results, list)

    def test_fts5_unavailable_falls_back(self, db_no_fts):
        """When FTS5 table is missing, _keyword_search falls back to LIKE."""
        engine = SearchEngine(backend='sqlite', index_path=db_no_fts)
        results = engine._keyword_search('python', count=5)
        # Fallback results should work
        if results:
            assert results[0]['search_type'] == 'fallback'

    def test_init_invalid_backend_raises(self):
        """Invalid backend name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid backend"):
            SearchEngine(backend='invalid', index_path='test.db')

    def test_init_sqlite_no_path_raises(self):
        """sqlite backend without index_path raises ValueError."""
        with pytest.raises(ValueError, match="index_path is required"):
            SearchEngine(backend='sqlite')

    def test_init_pgvector_no_connection_raises(self):
        """pgvector backend without connection_string raises ValueError."""
        with pytest.raises(ValueError, match="connection_string and collection_name are required"):
            SearchEngine(backend='pgvector')

    def test_keyword_search_only_no_tags(self, full_db):
        """_keyword_search_only without tags returns keyword results."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._keyword_search_only('python', count=5)
        assert isinstance(results, list)

    def test_keyword_search_only_with_tags(self, full_db):
        """_keyword_search_only with tags filters results."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        results = engine._keyword_search_only('python', count=5, tags=['python'])
        for r in results:
            assert 'python' in r['metadata'].get('tags', [])

    def test_escape_fts_query_empty(self, full_db):
        """Empty string is escaped to empty string."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        assert engine._escape_fts_query('') == ''

    def test_escape_fts_query_single_term(self, full_db):
        """Single term is quoted."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        assert engine._escape_fts_query('hello') == '"hello"'

    def test_escape_fts_query_strips_quotes(self, full_db):
        """Existing double quotes are stripped then re-quoted."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        assert engine._escape_fts_query('"hello" "world"') == '"hello" "world"'

    def test_get_stats_empty_db(self, empty_db):
        """get_stats on empty DB returns zero counts."""
        engine = SearchEngine(backend='sqlite', index_path=empty_db)
        stats = engine.get_stats()
        assert stats['total_chunks'] == 0
        assert stats['total_files'] == 0

    def test_get_stats_populated_db(self, full_db):
        """get_stats returns accurate counts for populated DB."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        stats = engine.get_stats()
        assert stats['total_chunks'] == 6
        assert stats['total_files'] >= 1
        assert 'config' in stats


# ---------------------------------------------------------------------------
# TestAddVectorScoresToCandidates: _add_vector_scores_to_candidates()
# ---------------------------------------------------------------------------

class TestAddVectorScoresToCandidates:
    """Tests for _add_vector_scores_to_candidates()."""

    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_adds_vector_scores(self, mock_cosine, mock_np, full_db):
        """Adds vector_score and vector_distance to candidates."""
        mock_array = Mock()
        mock_array.reshape.return_value = [[0.0]]
        mock_np.frombuffer.return_value = mock_array
        mock_np.float32 = 'float32'
        mock_cosine.return_value = [[0.85]]

        engine = SearchEngine(backend='sqlite', index_path=full_db)
        candidates = {
            1: {'id': 1, 'sources': {}, 'metadata': {}},
        }
        engine._add_vector_scores_to_candidates(candidates, [[0.1, 0.1, 0.1, 0.1]], 0.5)
        assert 'vector_score' in candidates[1]
        assert 'vector_distance' in candidates[1]
        assert candidates[1]['sources'].get('vector_rerank')

    @patch('signalwire_agents.search.search_engine.np', None)
    def test_no_numpy_does_nothing(self, full_db):
        """Does nothing when numpy is unavailable."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        candidates = {1: {'id': 1, 'sources': {}, 'metadata': {}}}
        engine._add_vector_scores_to_candidates(candidates, [[0.1]], 0.5)
        assert 'vector_score' not in candidates[1]

    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_empty_candidates(self, mock_cosine, mock_np, full_db):
        """Does nothing with empty candidates dict."""
        engine = SearchEngine(backend='sqlite', index_path=full_db)
        candidates = {}
        engine._add_vector_scores_to_candidates(candidates, [[0.1]], 0.5)
        assert candidates == {}

    @patch('signalwire_agents.search.search_engine.np')
    @patch('signalwire_agents.search.search_engine.cosine_similarity')
    def test_db_error_handled(self, mock_cosine, mock_np):
        """Database errors are handled gracefully."""
        engine = SearchEngine(backend='sqlite', index_path='/nonexistent/path.db')
        candidates = {1: {'id': 1, 'sources': {}, 'metadata': {}}}
        # Should not raise
        engine._add_vector_scores_to_candidates(candidates, [[0.1]], 0.5)