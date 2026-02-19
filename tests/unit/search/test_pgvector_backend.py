"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for pgvector backend module
"""

import pytest
import json
import logging
from unittest.mock import Mock, patch, MagicMock, call, PropertyMock
from datetime import datetime


# ---------------------------------------------------------------------------
# We must mock the optional dependencies (psycopg2, pgvector, numpy) *before*
# importing the module under test so that the module-level try/except blocks
# pick up our mocks.
# ---------------------------------------------------------------------------

# Create mock modules for psycopg2 and pgvector
mock_psycopg2 = MagicMock()
mock_psycopg2_extras = MagicMock()
mock_pgvector_psycopg2 = MagicMock()
mock_register_vector = MagicMock()
mock_execute_values = MagicMock()

mock_psycopg2.extras = mock_psycopg2_extras
mock_psycopg2_extras.execute_values = mock_execute_values
mock_pgvector_psycopg2.register_vector = mock_register_vector

import sys

sys.modules.setdefault('psycopg2', mock_psycopg2)
sys.modules.setdefault('psycopg2.extras', mock_psycopg2_extras)
sys.modules.setdefault('pgvector', MagicMock())
sys.modules.setdefault('pgvector.psycopg2', mock_pgvector_psycopg2)

# Now import the module under test.  Because we injected the mocks above,
# PGVECTOR_AVAILABLE will be True.
from signalwire_agents.search.pgvector_backend import (
    PgVectorBackend,
    PgVectorSearchBackend,
    PGVECTOR_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Helper: build a mock psycopg2 connection and cursor
# ---------------------------------------------------------------------------

def _make_mock_conn():
    """Return a mock connection with a context-managed cursor."""
    mock_conn = MagicMock()
    mock_conn.closed = False

    mock_cursor = MagicMock()
    # Support ``with conn.cursor() as cur:``
    mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)

    return mock_conn, mock_cursor


# ===================================================================
# PgVectorBackend tests
# ===================================================================


class TestPgVectorBackendInit:
    """Test PgVectorBackend initialization"""

    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    def test_init_success(self, mock_reg, mock_pg):
        """Test successful initialization connects and registers vector"""
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn

        backend = PgVectorBackend("postgresql://localhost/testdb")

        mock_pg.connect.assert_called_once_with("postgresql://localhost/testdb")
        mock_reg.assert_called_once_with(mock_conn)
        assert backend.connection_string == "postgresql://localhost/testdb"
        assert backend.conn is mock_conn

    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', False)
    def test_init_pgvector_not_available(self):
        """Test initialization raises ImportError when pgvector is not installed"""
        with pytest.raises(ImportError, match="pgvector dependencies not available"):
            PgVectorBackend("postgresql://localhost/testdb")

    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    def test_init_connection_failure(self, mock_reg, mock_pg):
        """Test initialization when connection fails"""
        mock_pg.connect.side_effect = Exception("Connection refused")

        with pytest.raises(Exception, match="Connection refused"):
            PgVectorBackend("postgresql://localhost/testdb")

    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    def test_init_vector_type_not_found(self, mock_reg, mock_pg):
        """Test initialization when vector extension is missing in database"""
        mock_pg.connect.side_effect = Exception("vector type not found in database")

        with pytest.raises(Exception, match="vector type not found"):
            PgVectorBackend("postgresql://localhost/testdb")


class TestPgVectorBackendConnect:
    """Test PgVectorBackend _connect method"""

    @pytest.fixture(autouse=True)
    def _enable_propagation(self):
        """Ensure logging is configured and propagation is on so caplog works."""
        from signalwire_agents.core.logging_config import reset_logging_configuration, configure_logging
        reset_logging_configuration()
        configure_logging()
        sw = logging.getLogger("signalwire_agents")
        sw.propagate = True
        yield
        sw.propagate = False

    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    def test_connect_logs_success(self, mock_reg, mock_pg, caplog):
        """Test that successful connection logs info message"""
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn

        with caplog.at_level(logging.INFO, logger="signalwire_agents.search.pgvector_backend"):
            backend = PgVectorBackend("postgresql://localhost/testdb")

        assert "Connected to PostgreSQL database" in caplog.text

    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    def test_connect_vector_type_error_logs_specific_message(self, mock_reg, mock_pg, caplog):
        """Test vector type not found produces specific error log"""
        mock_pg.connect.side_effect = Exception("vector type not found in the catalog")

        with caplog.at_level(logging.ERROR, logger="signalwire_agents.search.pgvector_backend"):
            with pytest.raises(Exception):
                PgVectorBackend("postgresql://localhost/testdb")

        assert "pgvector extension not installed" in caplog.text

    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    def test_connect_generic_error_logs_message(self, mock_reg, mock_pg, caplog):
        """Test generic connection failure produces error log"""
        mock_pg.connect.side_effect = Exception("host not reachable")

        with caplog.at_level(logging.ERROR, logger="signalwire_agents.search.pgvector_backend"):
            with pytest.raises(Exception):
                PgVectorBackend("postgresql://localhost/testdb")

        assert "Failed to connect to database" in caplog.text


class TestPgVectorBackendEnsureConnection:
    """Test PgVectorBackend _ensure_connection"""

    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    def test_ensure_connection_when_open(self, mock_reg, mock_pg):
        """Test _ensure_connection does nothing when connection is alive"""
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_pg.connect.return_value = mock_conn

        backend = PgVectorBackend("postgresql://localhost/testdb")
        assert mock_pg.connect.call_count == 1

        backend._ensure_connection()
        # Should not reconnect
        assert mock_pg.connect.call_count == 1

    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    def test_ensure_connection_reconnects_when_closed(self, mock_reg, mock_pg):
        """Test _ensure_connection reconnects when connection is closed"""
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_pg.connect.return_value = mock_conn

        backend = PgVectorBackend("postgresql://localhost/testdb")
        assert mock_pg.connect.call_count == 1

        # Mark connection as closed
        mock_conn.closed = True
        backend._ensure_connection()
        assert mock_pg.connect.call_count == 2

    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    def test_ensure_connection_reconnects_when_none(self, mock_reg, mock_pg):
        """Test _ensure_connection reconnects when conn is None"""
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_pg.connect.return_value = mock_conn

        backend = PgVectorBackend("postgresql://localhost/testdb")
        assert mock_pg.connect.call_count == 1

        backend.conn = None
        backend._ensure_connection()
        assert mock_pg.connect.call_count == 2


class TestPgVectorBackendCreateSchema:
    """Test PgVectorBackend create_schema"""

    def _make_backend(self):
        """Create a PgVectorBackend with a mocked connection."""
        with patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True), \
             patch('signalwire_agents.search.pgvector_backend.psycopg2') as mock_pg, \
             patch('signalwire_agents.search.pgvector_backend.register_vector'):
            mock_conn, mock_cursor = _make_mock_conn()
            mock_pg.connect.return_value = mock_conn
            backend = PgVectorBackend("postgresql://localhost/testdb")
            return backend, mock_conn, mock_cursor

    def test_create_schema_creates_table_and_indexes(self):
        """Test create_schema issues all expected DDL statements"""
        backend, mock_conn, mock_cursor = self._make_backend()

        backend.create_schema("my_collection", embedding_dim=512)

        # Collect all SQL strings executed
        executed_sqls = [
            str(c[0][0]).strip() for c in mock_cursor.execute.call_args_list
        ]

        # Extensions
        assert any("CREATE EXTENSION IF NOT EXISTS vector" in sql for sql in executed_sqls)
        assert any("CREATE EXTENSION IF NOT EXISTS pg_trgm" in sql for sql in executed_sqls)

        # Main table
        assert any("CREATE TABLE IF NOT EXISTS chunks_my_collection" in sql for sql in executed_sqls)
        assert any("vector(512)" in sql for sql in executed_sqls)

        # Indexes (embedding, content, tags, metadata, metadata_text)
        assert any("idx_chunks_my_collection_embedding" in sql for sql in executed_sqls)
        assert any("idx_chunks_my_collection_content" in sql for sql in executed_sqls)
        assert any("idx_chunks_my_collection_tags" in sql for sql in executed_sqls)
        assert any("idx_chunks_my_collection_metadata" in sql for sql in executed_sqls)
        assert any("idx_chunks_my_collection_metadata_text" in sql for sql in executed_sqls)

        # Config table
        assert any("CREATE TABLE IF NOT EXISTS collection_config" in sql for sql in executed_sqls)

        mock_conn.commit.assert_called_once()

    def test_create_schema_sanitizes_collection_name(self):
        """Test that special characters in collection name are replaced"""
        backend, mock_conn, mock_cursor = self._make_backend()

        backend.create_schema("my-collection.v2!", embedding_dim=768)

        executed_sqls = [
            str(c[0][0]).strip() for c in mock_cursor.execute.call_args_list
        ]

        # Dashes, dots, and exclamation should be replaced with underscores
        assert any("chunks_my_collection_v2_" in sql for sql in executed_sqls)
        # No raw special characters in table names
        for sql in executed_sqls:
            if "chunks_" in sql and "CREATE TABLE" in sql:
                assert "-" not in sql.split("chunks_")[1].split("(")[0]

    def test_create_schema_default_embedding_dim(self):
        """Test create_schema uses default embedding_dim of 768"""
        backend, mock_conn, mock_cursor = self._make_backend()

        backend.create_schema("test")

        executed_sqls = [
            str(c[0][0]).strip() for c in mock_cursor.execute.call_args_list
        ]

        assert any("vector(768)" in sql for sql in executed_sqls)

    def test_create_schema_calls_ensure_connection(self):
        """Test that create_schema checks connection"""
        backend, mock_conn, mock_cursor = self._make_backend()

        with patch.object(backend, '_ensure_connection') as mock_ensure:
            backend.create_schema("test")
            mock_ensure.assert_called_once()


class TestPgVectorBackendExtractMetadata:
    """Test PgVectorBackend _extract_metadata_from_json_content"""

    def _make_backend(self):
        with patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True), \
             patch('signalwire_agents.search.pgvector_backend.psycopg2') as mock_pg, \
             patch('signalwire_agents.search.pgvector_backend.register_vector'):
            mock_conn = MagicMock()
            mock_conn.closed = False
            mock_pg.connect.return_value = mock_conn
            return PgVectorBackend("postgresql://localhost/testdb")

    def test_extract_no_metadata(self):
        """Test extraction from content with no metadata key"""
        backend = self._make_backend()
        result = backend._extract_metadata_from_json_content("plain text with no metadata")
        assert result == {}

    def test_extract_valid_json_metadata(self):
        """Test extraction of valid JSON metadata"""
        backend = self._make_backend()
        content = '{"title": "doc", "metadata": {"author": "Alice", "version": "1.0"}}'
        result = backend._extract_metadata_from_json_content(content)
        assert result["author"] == "Alice"
        assert result["version"] == "1.0"

    def test_extract_multiple_metadata_blocks(self):
        """Test extraction merges multiple metadata blocks"""
        backend = self._make_backend()
        content = (
            '{"metadata": {"key1": "val1"}} '
            'some text '
            '{"metadata": {"key2": "val2"}}'
        )
        result = backend._extract_metadata_from_json_content(content)
        assert result.get("key1") == "val1"
        assert result.get("key2") == "val2"

    def test_extract_invalid_json_metadata(self):
        """Test extraction handles malformed JSON gracefully"""
        backend = self._make_backend()
        content = '"metadata": {not valid json}'
        result = backend._extract_metadata_from_json_content(content)
        assert result == {}

    def test_extract_metadata_keyword_but_no_json(self):
        """Test content with metadata keyword but not as JSON"""
        backend = self._make_backend()
        content = 'This document has "metadata": information about something'
        result = backend._extract_metadata_from_json_content(content)
        # Should not crash, returns empty or partial depending on parsing
        assert isinstance(result, dict)


class TestPgVectorBackendStoreChunks:
    """Test PgVectorBackend store_chunks"""

    @patch('signalwire_agents.search.pgvector_backend.execute_values')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    def test_store_chunks_basic(self, mock_pg, mock_reg, mock_ev):
        """Test storing a single chunk"""
        mock_conn, mock_cursor = _make_mock_conn()
        mock_pg.connect.return_value = mock_conn
        backend = PgVectorBackend("postgresql://localhost/testdb")

        chunks = [
            {
                "content": "Hello world",
                "processed_content": "hello world",
                "embedding": [0.1, 0.2, 0.3],
                "filename": "test.txt",
                "section": "intro",
                "tags": ["greeting"],
                "metadata": {"source": "test"},
            }
        ]
        config = {
            "model_name": "test-model",
            "embedding_dimensions": 3,
            "chunking_strategy": "sentence",
            "languages": ["en"],
            "metadata": {"version": 1},
        }

        backend.store_chunks(chunks, "my_collection", config)

        # execute_values should have been called for chunk insert
        mock_ev.assert_called_once()
        # Config upsert
        mock_cursor.execute.assert_called()
        mock_conn.commit.assert_called()

    @patch('signalwire_agents.search.pgvector_backend.execute_values')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    def test_store_chunks_numpy_embedding(self, mock_pg, mock_reg, mock_ev):
        """Test storing chunks with numpy array embeddings"""
        mock_conn, mock_cursor = _make_mock_conn()
        mock_pg.connect.return_value = mock_conn
        backend = PgVectorBackend("postgresql://localhost/testdb")

        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]

        chunks = [
            {
                "content": "Test content",
                "embedding": mock_embedding,
            }
        ]
        config = {"model_name": "m"}

        backend.store_chunks(chunks, "col", config)

        # The embedding's tolist() should have been called
        mock_embedding.tolist.assert_called_once()
        mock_ev.assert_called_once()

    @patch('signalwire_agents.search.pgvector_backend.execute_values')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    def test_store_chunks_no_embedding(self, mock_pg, mock_reg, mock_ev):
        """Test storing chunks without embeddings"""
        mock_conn, mock_cursor = _make_mock_conn()
        mock_pg.connect.return_value = mock_conn
        backend = PgVectorBackend("postgresql://localhost/testdb")

        chunks = [
            {
                "content": "No embedding here",
            }
        ]
        config = {}

        backend.store_chunks(chunks, "col", config)
        mock_ev.assert_called_once()

        # The embedding in the data tuple should be None
        data_arg = mock_ev.call_args[0][2]
        assert data_arg[0][2] is None  # embedding position

    @patch('signalwire_agents.search.pgvector_backend.execute_values')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    def test_store_chunks_metadata_from_chunk_keys(self, mock_pg, mock_reg, mock_ev):
        """Test that extra chunk keys become metadata"""
        mock_conn, mock_cursor = _make_mock_conn()
        mock_pg.connect.return_value = mock_conn
        backend = PgVectorBackend("postgresql://localhost/testdb")

        chunks = [
            {
                "content": "Test",
                "embedding": None,
                "custom_field": "custom_value",
                "another_field": 42,
            }
        ]
        config = {}

        backend.store_chunks(chunks, "col", config)
        mock_ev.assert_called_once()

        data_arg = mock_ev.call_args[0][2]
        metadata_json = json.loads(data_arg[0][6])  # metadata position
        assert metadata_json["custom_field"] == "custom_value"
        assert metadata_json["another_field"] == 42

    @patch('signalwire_agents.search.pgvector_backend.execute_values')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    def test_store_chunks_metadata_text_generation(self, mock_pg, mock_reg, mock_ev):
        """Test that searchable metadata text is generated correctly"""
        mock_conn, mock_cursor = _make_mock_conn()
        mock_pg.connect.return_value = mock_conn
        backend = PgVectorBackend("postgresql://localhost/testdb")

        chunks = [
            {
                "content": "Test",
                "embedding": None,
                "filename": "doc.txt",
                "section": "Overview",
                "tags": ["important", "v2"],
                "metadata": {"author": "Bob"},
            }
        ]
        config = {}

        backend.store_chunks(chunks, "col", config)

        data_arg = mock_ev.call_args[0][2]
        metadata_text = data_arg[0][7]  # metadata_text position
        assert "important" in metadata_text
        assert "v2" in metadata_text
        assert "overview" in metadata_text  # section, lowered
        assert "bob" in metadata_text or "author" in metadata_text

    @patch('signalwire_agents.search.pgvector_backend.execute_values')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    def test_store_chunks_multiple(self, mock_pg, mock_reg, mock_ev):
        """Test storing multiple chunks at once"""
        mock_conn, mock_cursor = _make_mock_conn()
        mock_pg.connect.return_value = mock_conn
        backend = PgVectorBackend("postgresql://localhost/testdb")

        chunks = [
            {"content": f"Chunk {i}", "embedding": None}
            for i in range(5)
        ]
        config = {}

        backend.store_chunks(chunks, "col", config)
        mock_ev.assert_called_once()

        data_arg = mock_ev.call_args[0][2]
        assert len(data_arg) == 5

    @patch('signalwire_agents.search.pgvector_backend.execute_values')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    def test_store_chunks_json_metadata_in_content(self, mock_pg, mock_reg, mock_ev):
        """Test metadata extraction from JSON content during storage"""
        mock_conn, mock_cursor = _make_mock_conn()
        mock_pg.connect.return_value = mock_conn
        backend = PgVectorBackend("postgresql://localhost/testdb")

        chunks = [
            {
                "content": '{"title": "doc", "metadata": {"source": "web"}}',
                "embedding": None,
            }
        ]
        config = {}

        backend.store_chunks(chunks, "col", config)
        mock_ev.assert_called_once()

        data_arg = mock_ev.call_args[0][2]
        metadata_json = json.loads(data_arg[0][6])
        # JSON metadata should be merged (but chunk metadata takes precedence)
        assert "source" in metadata_json

    @patch('signalwire_agents.search.pgvector_backend.execute_values')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    def test_store_chunks_calls_ensure_connection(self, mock_pg, mock_reg, mock_ev):
        """Test that store_chunks calls _ensure_connection"""
        mock_conn, mock_cursor = _make_mock_conn()
        mock_pg.connect.return_value = mock_conn
        backend = PgVectorBackend("postgresql://localhost/testdb")

        with patch.object(backend, '_ensure_connection') as mock_ensure:
            backend.store_chunks([{"content": "test", "embedding": None}], "col", {})
            mock_ensure.assert_called_once()

    @patch('signalwire_agents.search.pgvector_backend.execute_values')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    def test_store_chunks_config_upsert(self, mock_pg, mock_reg, mock_ev):
        """Test that config is upserted with ON CONFLICT"""
        mock_conn, mock_cursor = _make_mock_conn()
        mock_pg.connect.return_value = mock_conn
        backend = PgVectorBackend("postgresql://localhost/testdb")

        config = {
            "model_name": "test-model",
            "embedding_dimensions": 768,
            "chunking_strategy": "sentence",
            "languages": ["en", "fr"],
            "metadata": {"version": "2.0"},
        }

        backend.store_chunks([{"content": "test", "embedding": None}], "my_col", config)

        # Find the config upsert call
        config_call = None
        for c in mock_cursor.execute.call_args_list:
            if c[0] and "collection_config" in str(c[0][0]):
                config_call = c
                break

        assert config_call is not None
        assert "ON CONFLICT" in str(config_call[0][0])
        params = config_call[0][1]
        assert params[0] == "my_col"
        assert params[1] == "test-model"
        assert params[2] == 768


class TestPgVectorBackendGetStats:
    """Test PgVectorBackend get_stats"""

    def _make_backend(self):
        with patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True), \
             patch('signalwire_agents.search.pgvector_backend.psycopg2') as mock_pg, \
             patch('signalwire_agents.search.pgvector_backend.register_vector'):
            mock_conn, mock_cursor = _make_mock_conn()
            mock_pg.connect.return_value = mock_conn
            backend = PgVectorBackend("postgresql://localhost/testdb")
            return backend, mock_conn, mock_cursor

    def test_get_stats_with_config(self):
        """Test get_stats returns correct statistics with config"""
        backend, mock_conn, mock_cursor = self._make_backend()

        created_at = datetime(2025, 1, 15, 12, 0, 0)
        mock_cursor.fetchone.side_effect = [
            (42,),       # total chunks
            (7,),        # unique files
            (             # config row
                "test_col", "model-v1", 768, "sentence",
                ["en"], created_at, {"version": 1}
            ),
        ]

        stats = backend.get_stats("test_col")

        assert stats["total_chunks"] == 42
        assert stats["total_files"] == 7
        assert stats["config"]["model_name"] == "model-v1"
        assert stats["config"]["embedding_dimensions"] == 768
        assert stats["config"]["chunking_strategy"] == "sentence"
        assert stats["config"]["languages"] == ["en"]
        assert stats["config"]["created_at"] == "2025-01-15T12:00:00"
        assert stats["config"]["metadata"] == {"version": 1}

    def test_get_stats_without_config(self):
        """Test get_stats when no config exists"""
        backend, mock_conn, mock_cursor = self._make_backend()

        mock_cursor.fetchone.side_effect = [
            (10,),  # total chunks
            (3,),   # unique files
            None,   # no config row
        ]

        stats = backend.get_stats("missing_col")

        assert stats["total_chunks"] == 10
        assert stats["total_files"] == 3
        assert stats["config"] == {}

    def test_get_stats_with_none_created_at(self):
        """Test get_stats when created_at is None"""
        backend, mock_conn, mock_cursor = self._make_backend()

        mock_cursor.fetchone.side_effect = [
            (0,),
            (0,),
            ("col", "model", 768, "sentence", [], None, {}),
        ]

        stats = backend.get_stats("col")
        assert stats["config"]["created_at"] is None


class TestPgVectorBackendListCollections:
    """Test PgVectorBackend list_collections"""

    def _make_backend(self):
        with patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True), \
             patch('signalwire_agents.search.pgvector_backend.psycopg2') as mock_pg, \
             patch('signalwire_agents.search.pgvector_backend.register_vector'):
            mock_conn, mock_cursor = _make_mock_conn()
            mock_pg.connect.return_value = mock_conn
            backend = PgVectorBackend("postgresql://localhost/testdb")
            return backend, mock_conn, mock_cursor

    def test_list_collections_returns_names(self):
        """Test list_collections returns collection names"""
        backend, mock_conn, mock_cursor = self._make_backend()
        mock_cursor.fetchall.return_value = [("alpha",), ("beta",), ("gamma",)]

        result = backend.list_collections()

        assert result == ["alpha", "beta", "gamma"]

    def test_list_collections_empty(self):
        """Test list_collections with no collections"""
        backend, mock_conn, mock_cursor = self._make_backend()
        mock_cursor.fetchall.return_value = []

        result = backend.list_collections()

        assert result == []

    def test_list_collections_calls_ensure_connection(self):
        """Test that list_collections checks connection"""
        backend, mock_conn, mock_cursor = self._make_backend()
        mock_cursor.fetchall.return_value = []

        with patch.object(backend, '_ensure_connection') as mock_ensure:
            backend.list_collections()
            mock_ensure.assert_called_once()


class TestPgVectorBackendDeleteCollection:
    """Test PgVectorBackend delete_collection"""

    def _make_backend(self):
        with patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True), \
             patch('signalwire_agents.search.pgvector_backend.psycopg2') as mock_pg, \
             patch('signalwire_agents.search.pgvector_backend.register_vector'):
            mock_conn, mock_cursor = _make_mock_conn()
            mock_pg.connect.return_value = mock_conn
            backend = PgVectorBackend("postgresql://localhost/testdb")
            return backend, mock_conn, mock_cursor

    def test_delete_collection_drops_table_and_config(self):
        """Test that delete_collection drops the table and removes config"""
        backend, mock_conn, mock_cursor = self._make_backend()

        backend.delete_collection("my_collection")

        executed_sqls = [str(c[0][0]).strip() for c in mock_cursor.execute.call_args_list]

        assert any("DROP TABLE IF EXISTS chunks_my_collection" in sql for sql in executed_sqls)
        assert any("DELETE FROM collection_config" in sql for sql in executed_sqls)
        mock_conn.commit.assert_called()

    def test_delete_collection_sanitizes_name(self):
        """Test that delete_collection sanitizes collection name"""
        backend, mock_conn, mock_cursor = self._make_backend()

        backend.delete_collection("bad-name.here!")

        executed_sqls = [str(c[0][0]).strip() for c in mock_cursor.execute.call_args_list]

        # The DROP should use the sanitized name
        assert any("chunks_bad_name_here_" in sql for sql in executed_sqls)

    def test_delete_collection_calls_ensure_connection(self):
        """Test that delete_collection checks connection"""
        backend, mock_conn, mock_cursor = self._make_backend()

        with patch.object(backend, '_ensure_connection') as mock_ensure:
            backend.delete_collection("test")
            mock_ensure.assert_called_once()


class TestPgVectorBackendClose:
    """Test PgVectorBackend close"""

    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    def test_close_closes_connection(self, mock_reg, mock_pg):
        """Test close closes the database connection"""
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_pg.connect.return_value = mock_conn

        backend = PgVectorBackend("postgresql://localhost/testdb")
        backend.close()

        mock_conn.close.assert_called_once()

    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    def test_close_already_closed(self, mock_reg, mock_pg):
        """Test close does nothing when connection is already closed"""
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_pg.connect.return_value = mock_conn

        backend = PgVectorBackend("postgresql://localhost/testdb")
        mock_conn.closed = True
        backend.close()

        mock_conn.close.assert_not_called()

    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    def test_close_when_conn_is_none(self, mock_reg, mock_pg):
        """Test close does nothing when conn is None"""
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_pg.connect.return_value = mock_conn

        backend = PgVectorBackend("postgresql://localhost/testdb")
        backend.conn = None
        backend.close()  # Should not raise


# ===================================================================
# PgVectorSearchBackend tests
# ===================================================================


class TestPgVectorSearchBackendInit:
    """Test PgVectorSearchBackend initialization"""

    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    def test_init_success(self, mock_reg, mock_pg):
        """Test successful initialization"""
        mock_conn, mock_cursor = _make_mock_conn()
        mock_pg.connect.return_value = mock_conn
        mock_cursor.fetchone.return_value = (
            "test_col", "model-v1", 768, "sentence", ["en"], datetime.now(), {}
        )

        sb = PgVectorSearchBackend("postgresql://localhost/testdb", "test_col")

        assert sb.connection_string == "postgresql://localhost/testdb"
        assert sb.collection_name == "test_col"
        assert sb.table_name == "chunks_test_col"
        assert sb.conn is mock_conn
        assert sb.config["model_name"] == "model-v1"

    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', False)
    def test_init_pgvector_not_available(self):
        """Test initialization raises ImportError when pgvector not installed"""
        with pytest.raises(ImportError, match="pgvector dependencies not available"):
            PgVectorSearchBackend("postgresql://localhost/testdb", "col")

    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    def test_init_connection_failure(self, mock_reg, mock_pg):
        """Test initialization when connection fails"""
        mock_pg.connect.side_effect = Exception("Connection refused")

        with pytest.raises(Exception, match="Connection refused"):
            PgVectorSearchBackend("postgresql://localhost/testdb", "col")

    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    def test_init_no_config(self, mock_reg, mock_pg):
        """Test initialization when collection has no config"""
        mock_conn, mock_cursor = _make_mock_conn()
        mock_pg.connect.return_value = mock_conn
        mock_cursor.fetchone.return_value = None

        sb = PgVectorSearchBackend("postgresql://localhost/testdb", "new_col")

        assert sb.config == {}


class TestPgVectorSearchBackendLoadConfig:
    """Test PgVectorSearchBackend _load_config"""

    def _make_search_backend(self, config_row=None):
        """Create a PgVectorSearchBackend with mocked connection."""
        with patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True), \
             patch('signalwire_agents.search.pgvector_backend.psycopg2') as mock_pg, \
             patch('signalwire_agents.search.pgvector_backend.register_vector'):
            mock_conn, mock_cursor = _make_mock_conn()
            mock_pg.connect.return_value = mock_conn
            mock_cursor.fetchone.return_value = config_row
            sb = PgVectorSearchBackend("postgresql://localhost/testdb", "col")
            return sb, mock_conn, mock_cursor

    def test_load_config_with_data(self):
        """Test _load_config returns config when row exists"""
        config_row = ("col", "model-v1", 512, "sliding", ["en", "de"], datetime.now(), {"k": "v"})
        sb, _, _ = self._make_search_backend(config_row)

        assert sb.config["model_name"] == "model-v1"
        assert sb.config["embedding_dimensions"] == 512
        assert sb.config["chunking_strategy"] == "sliding"
        assert sb.config["languages"] == ["en", "de"]
        assert sb.config["metadata"] == {"k": "v"}

    def test_load_config_no_data(self):
        """Test _load_config returns empty dict when no row"""
        sb, _, _ = self._make_search_backend(config_row=None)
        assert sb.config == {}


class TestPgVectorSearchBackendVectorSearch:
    """Test PgVectorSearchBackend _vector_search"""

    def _make_search_backend(self):
        with patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True), \
             patch('signalwire_agents.search.pgvector_backend.psycopg2') as mock_pg, \
             patch('signalwire_agents.search.pgvector_backend.register_vector'):
            mock_conn, mock_cursor = _make_mock_conn()
            mock_pg.connect.return_value = mock_conn
            # For _load_config
            mock_cursor.fetchone.return_value = None
            sb = PgVectorSearchBackend("postgresql://localhost/testdb", "col")
            # Reset cursor mocks for the test
            mock_cursor.reset_mock()
            return sb, mock_conn, mock_cursor

    def test_vector_search_returns_results(self):
        """Test _vector_search returns properly formatted results"""
        sb, _, mock_cursor = self._make_search_backend()

        mock_cursor.fetchall.return_value = [
            (1, "Content A", "file_a.txt", "intro", ["tag1"], {"key": "val"}, 0.92),
            (2, "Content B", "file_b.txt", "body", ["tag2"], {"key2": "val2"}, 0.85),
        ]

        results = sb._vector_search([0.1, 0.2, 0.3], count=5)

        assert len(results) == 2
        assert results[0]["id"] == 1
        assert results[0]["content"] == "Content A"
        assert results[0]["score"] == 0.92
        assert results[0]["search_type"] == "vector"
        assert results[0]["metadata"]["filename"] == "file_a.txt"
        assert results[0]["metadata"]["section"] == "intro"
        assert results[0]["metadata"]["tags"] == ["tag1"]
        assert results[0]["metadata"]["key"] == "val"

    def test_vector_search_empty_results(self):
        """Test _vector_search with no results"""
        sb, _, mock_cursor = self._make_search_backend()
        mock_cursor.fetchall.return_value = []

        results = sb._vector_search([0.1, 0.2], count=5)

        assert results == []

    def test_vector_search_with_tags_filter(self):
        """Test _vector_search includes tag filter in query"""
        sb, _, mock_cursor = self._make_search_backend()
        mock_cursor.fetchall.return_value = []

        sb._vector_search([0.1, 0.2], count=5, tags=["python", "tutorial"])

        # Verify tags were included in params
        call_args = mock_cursor.execute.call_args
        query_str = str(call_args[0][0])
        assert "tags ?|" in query_str

    def test_vector_search_without_tags(self):
        """Test _vector_search query without tags"""
        sb, _, mock_cursor = self._make_search_backend()
        mock_cursor.fetchall.return_value = []

        sb._vector_search([0.1, 0.2], count=3)

        call_args = mock_cursor.execute.call_args
        query_str = str(call_args[0][0])
        assert "tags ?|" not in query_str

    def test_vector_search_sets_ivfflat_probes(self):
        """Test _vector_search sets probes for IVFFlat index"""
        sb, _, mock_cursor = self._make_search_backend()
        mock_cursor.fetchall.return_value = []

        sb._vector_search([0.1], count=5)

        # First execute should set probes
        first_call = mock_cursor.execute.call_args_list[0]
        assert "ivfflat.probes" in str(first_call[0][0])

    def test_vector_search_tags_not_list(self):
        """Test _vector_search handles non-list tags in results"""
        sb, _, mock_cursor = self._make_search_backend()
        mock_cursor.fetchall.return_value = [
            (1, "Content", "file.txt", "sec", "not_a_list", {}, 0.9),
        ]

        results = sb._vector_search([0.1], count=5)

        # When tags_json is not a list, should return empty list
        assert results[0]["metadata"]["tags"] == []


class TestPgVectorSearchBackendKeywordSearch:
    """Test PgVectorSearchBackend _keyword_search"""

    def _make_search_backend(self):
        with patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True), \
             patch('signalwire_agents.search.pgvector_backend.psycopg2') as mock_pg, \
             patch('signalwire_agents.search.pgvector_backend.register_vector'):
            mock_conn, mock_cursor = _make_mock_conn()
            mock_pg.connect.return_value = mock_conn
            mock_cursor.fetchone.return_value = None
            sb = PgVectorSearchBackend("postgresql://localhost/testdb", "col")
            mock_cursor.reset_mock()
            return sb, mock_conn, mock_cursor

    def test_keyword_search_returns_results(self):
        """Test _keyword_search returns properly formatted results"""
        sb, _, mock_cursor = self._make_search_backend()
        mock_cursor.fetchall.return_value = [
            (1, "Python tutorial", "python.md", "intro", ["python"], {"level": "beginner"}, 5.0),
        ]

        results = sb._keyword_search("Python tutorial", count=5)

        assert len(results) == 1
        assert results[0]["id"] == 1
        assert results[0]["content"] == "Python tutorial"
        assert results[0]["search_type"] == "keyword"
        # score = min(1.0, 5.0/10.0) = 0.5
        assert results[0]["score"] == 0.5

    def test_keyword_search_score_normalization(self):
        """Test keyword search score is normalized to 0-1 range"""
        sb, _, mock_cursor = self._make_search_backend()
        mock_cursor.fetchall.return_value = [
            (1, "Content", "f.txt", "s", [], {}, 15.0),  # rank > 10
        ]

        results = sb._keyword_search("query", count=5)

        # min(1.0, 15.0/10.0) = 1.0
        assert results[0]["score"] == 1.0

    def test_keyword_search_with_tags(self):
        """Test _keyword_search includes tag filter"""
        sb, _, mock_cursor = self._make_search_backend()
        mock_cursor.fetchall.return_value = []

        sb._keyword_search("query", count=5, tags=["python"])

        call_args = mock_cursor.execute.call_args
        query_str = str(call_args[0][0])
        assert "tags ?|" in query_str

    def test_keyword_search_empty(self):
        """Test _keyword_search with no matches"""
        sb, _, mock_cursor = self._make_search_backend()
        mock_cursor.fetchall.return_value = []

        results = sb._keyword_search("nonexistent", count=5)
        assert results == []


class TestPgVectorSearchBackendMetadataSearch:
    """Test PgVectorSearchBackend _metadata_search"""

    def _make_search_backend(self):
        with patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True), \
             patch('signalwire_agents.search.pgvector_backend.psycopg2') as mock_pg, \
             patch('signalwire_agents.search.pgvector_backend.register_vector'):
            mock_conn, mock_cursor = _make_mock_conn()
            mock_pg.connect.return_value = mock_conn
            mock_cursor.fetchone.return_value = None
            sb = PgVectorSearchBackend("postgresql://localhost/testdb", "col")
            mock_cursor.reset_mock()
            return sb, mock_conn, mock_cursor

    def test_metadata_search_basic(self):
        """Test _metadata_search returns properly formatted results"""
        sb, _, mock_cursor = self._make_search_backend()
        mock_cursor.fetchall.return_value = [
            (1, "Content A", "file.txt", "intro", ["tag1"], {"author": "alice"}, "author alice tag1 intro"),
        ]

        results = sb._metadata_search(["alice"], count=5)

        assert len(results) == 1
        assert results[0]["id"] == 1
        assert results[0]["search_type"] == "metadata"
        assert results[0]["score"] > 0  # Should have some score from matches

    def test_metadata_search_term_scoring(self):
        """Test metadata search scoring based on term matches"""
        sb, _, mock_cursor = self._make_search_backend()
        mock_cursor.fetchall.return_value = [
            (1, "Content", "f.txt", "s", [], {"key": "val"}, "key val term1 term2"),
        ]

        results = sb._metadata_search(["term1", "term2"], count=5)

        # Each matching term adds 0.3 for metadata_text + 0.2 for json match
        # Two terms: (0.3 + 0.2) * 2 = 1.0 (capped)
        assert results[0]["score"] > 0

    def test_metadata_search_score_capped_at_one(self):
        """Test that metadata search score is capped at 1.0"""
        sb, _, mock_cursor = self._make_search_backend()
        mock_cursor.fetchall.return_value = [
            (1, "C", "f", "s", [], {"a": "b", "c": "d", "e": "f"},
             "a b c d e f g h i j k"),
        ]

        results = sb._metadata_search(["a", "b", "c", "d", "e"], count=5)

        assert results[0]["score"] <= 1.0

    def test_metadata_search_with_tags_filter(self):
        """Test _metadata_search includes tag filter"""
        sb, _, mock_cursor = self._make_search_backend()
        mock_cursor.fetchall.return_value = []

        sb._metadata_search(["term"], count=5, tags=["python"])

        call_args = mock_cursor.execute.call_args
        query_str = str(call_args[0][0])
        assert "tags ?|" in query_str

    def test_metadata_search_no_terms(self):
        """Test _metadata_search with empty query terms"""
        sb, _, mock_cursor = self._make_search_backend()
        mock_cursor.fetchall.return_value = []

        results = sb._metadata_search([], count=5)

        assert results == []
        # Query should use "1=1" fallback
        call_args = mock_cursor.execute.call_args
        query_str = str(call_args[0][0])
        assert "1=1" in query_str

    def test_metadata_search_null_metadata_text(self):
        """Test _metadata_search when metadata_text is None"""
        sb, _, mock_cursor = self._make_search_backend()
        mock_cursor.fetchall.return_value = [
            (1, "Content", "f.txt", "s", [], {}, None),
        ]

        results = sb._metadata_search(["term"], count=5)

        # Should not crash, score should be 0 from metadata_text
        assert len(results) == 1

    def test_metadata_search_null_metadata_json(self):
        """Test _metadata_search when metadata_json is None raises TypeError.

        This documents a known edge case in the source code: when the database
        returns None for the metadata JSONB column, the ``**metadata_json``
        unpacking raises a TypeError.
        """
        sb, _, mock_cursor = self._make_search_backend()
        mock_cursor.fetchall.return_value = [
            (1, "Content", "f.txt", "s", [], None, "some text"),
        ]

        with pytest.raises(TypeError):
            sb._metadata_search(["some"], count=5)

    def test_metadata_search_empty_metadata_json(self):
        """Test _metadata_search when metadata_json is an empty dict"""
        sb, _, mock_cursor = self._make_search_backend()
        mock_cursor.fetchall.return_value = [
            (1, "Content", "f.txt", "s", [], {}, "some text with term"),
        ]

        results = sb._metadata_search(["term"], count=5)

        assert len(results) == 1

    def test_metadata_search_sorted_by_score(self):
        """Test _metadata_search results are sorted by score descending"""
        sb, _, mock_cursor = self._make_search_backend()
        mock_cursor.fetchall.return_value = [
            (1, "Low", "f.txt", "s", [], {}, "unrelated"),
            (2, "High", "f.txt", "s", [], {"term": "term"}, "term term"),
        ]

        results = sb._metadata_search(["term"], count=5)

        if len(results) >= 2:
            assert results[0]["score"] >= results[1]["score"]


class TestPgVectorSearchBackendMergeResults:
    """Test PgVectorSearchBackend _merge_results"""

    def _make_search_backend(self):
        with patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True), \
             patch('signalwire_agents.search.pgvector_backend.psycopg2') as mock_pg, \
             patch('signalwire_agents.search.pgvector_backend.register_vector'):
            mock_conn, mock_cursor = _make_mock_conn()
            mock_pg.connect.return_value = mock_conn
            mock_cursor.fetchone.return_value = None
            sb = PgVectorSearchBackend("postgresql://localhost/testdb", "col")
            return sb

    def test_merge_results_default_weights(self):
        """Test _merge_results uses default keyword_weight of 0.3"""
        sb = self._make_search_backend()

        vector_results = [
            {"id": 1, "content": "V1", "score": 1.0, "search_type": "vector", "metadata": {}},
        ]
        keyword_results = [
            {"id": 1, "content": "K1", "score": 1.0, "search_type": "keyword", "metadata": {}},
        ]

        merged = sb._merge_results(vector_results, keyword_results)

        assert len(merged) == 1
        # vector_weight=0.7, keyword_weight=0.3 => 1.0*0.7 + 1.0*0.3 = 1.0
        assert abs(merged[0]["score"] - 1.0) < 1e-9

    def test_merge_results_custom_keyword_weight(self):
        """Test _merge_results with custom keyword_weight"""
        sb = self._make_search_backend()

        vector_results = [
            {"id": 1, "content": "V1", "score": 1.0, "search_type": "vector", "metadata": {}},
        ]
        keyword_results = [
            {"id": 1, "content": "K1", "score": 1.0, "search_type": "keyword", "metadata": {}},
        ]

        merged = sb._merge_results(vector_results, keyword_results, keyword_weight=0.5)

        # vector_weight=0.5, keyword_weight=0.5 => 1.0*0.5 + 1.0*0.5 = 1.0
        assert abs(merged[0]["score"] - 1.0) < 1e-9

    def test_merge_results_unique_ids(self):
        """Test _merge_results keeps unique results"""
        sb = self._make_search_backend()

        vector_results = [
            {"id": 1, "content": "Only vector", "score": 0.9, "search_type": "vector", "metadata": {}},
        ]
        keyword_results = [
            {"id": 2, "content": "Only keyword", "score": 0.8, "search_type": "keyword", "metadata": {}},
        ]

        merged = sb._merge_results(vector_results, keyword_results)

        assert len(merged) == 2

    def test_merge_results_sorted_by_score(self):
        """Test _merge_results sorts by combined score descending"""
        sb = self._make_search_backend()

        vector_results = [
            {"id": 1, "content": "V1", "score": 0.5, "search_type": "vector", "metadata": {}},
            {"id": 2, "content": "V2", "score": 0.9, "search_type": "vector", "metadata": {}},
        ]
        keyword_results = []

        merged = sb._merge_results(vector_results, keyword_results)

        assert merged[0]["id"] == 2  # Higher score first

    def test_merge_results_empty_inputs(self):
        """Test _merge_results with empty inputs"""
        sb = self._make_search_backend()

        merged = sb._merge_results([], [])
        assert merged == []


class TestPgVectorSearchBackendMergeAllResults:
    """Test PgVectorSearchBackend _merge_all_results"""

    def _make_search_backend(self):
        with patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True), \
             patch('signalwire_agents.search.pgvector_backend.psycopg2') as mock_pg, \
             patch('signalwire_agents.search.pgvector_backend.register_vector'):
            mock_conn, mock_cursor = _make_mock_conn()
            mock_pg.connect.return_value = mock_conn
            mock_cursor.fetchone.return_value = None
            sb = PgVectorSearchBackend("postgresql://localhost/testdb", "col")
            return sb

    def test_merge_all_results_three_sources(self):
        """Test _merge_all_results combines vector, keyword, and metadata"""
        sb = self._make_search_backend()

        vector = [{"id": 1, "content": "V", "score": 0.9, "search_type": "vector", "metadata": {}}]
        keyword = [{"id": 1, "content": "K", "score": 0.8, "search_type": "keyword", "metadata": {}}]
        metadata = [{"id": 1, "content": "M", "score": 0.7, "search_type": "metadata", "metadata": {}}]

        merged = sb._merge_all_results(vector, keyword, metadata)

        assert len(merged) == 1
        # default weights: vector=0.5, keyword=0.3, metadata=0.2
        expected_score = 0.9 * 0.5 + 0.8 * 0.3 + 0.7 * 0.2
        assert abs(merged[0]["score"] - expected_score) < 1e-9
        assert abs(merged[0]["final_score"] - expected_score) < 1e-9

    def test_merge_all_results_includes_sources(self):
        """Test _merge_all_results includes source breakdown"""
        sb = self._make_search_backend()

        vector = [{"id": 1, "content": "V", "score": 0.9, "search_type": "vector", "metadata": {}}]
        keyword = [{"id": 1, "content": "K", "score": 0.8, "search_type": "keyword", "metadata": {}}]
        metadata = []

        merged = sb._merge_all_results(vector, keyword, metadata)

        assert "sources" in merged[0]
        assert merged[0]["sources"]["vector"] == 0.9
        assert merged[0]["sources"]["keyword"] == 0.8

    def test_merge_all_results_unique_from_different_sources(self):
        """Test _merge_all_results handles results unique to each source"""
        sb = self._make_search_backend()

        vector = [{"id": 1, "content": "V", "score": 0.9, "search_type": "vector", "metadata": {}}]
        keyword = [{"id": 2, "content": "K", "score": 0.8, "search_type": "keyword", "metadata": {}}]
        metadata = [{"id": 3, "content": "M", "score": 0.7, "search_type": "metadata", "metadata": {}}]

        merged = sb._merge_all_results(vector, keyword, metadata)

        assert len(merged) == 3
        ids = {r["id"] for r in merged}
        assert ids == {1, 2, 3}

    def test_merge_all_results_sorted_by_score(self):
        """Test _merge_all_results are sorted descending by score"""
        sb = self._make_search_backend()

        vector = [
            {"id": 1, "content": "Low", "score": 0.2, "search_type": "vector", "metadata": {}},
            {"id": 2, "content": "High", "score": 0.95, "search_type": "vector", "metadata": {}},
        ]

        merged = sb._merge_all_results(vector, [], [])

        assert merged[0]["id"] == 2

    def test_merge_all_results_empty(self):
        """Test _merge_all_results with all empty sources"""
        sb = self._make_search_backend()

        merged = sb._merge_all_results([], [], [])
        assert merged == []

    def test_merge_all_results_custom_keyword_weight(self):
        """Test _merge_all_results with custom keyword_weight"""
        sb = self._make_search_backend()

        keyword = [{"id": 1, "content": "K", "score": 1.0, "search_type": "keyword", "metadata": {}}]

        merged_default = sb._merge_all_results([], keyword, [])
        merged_custom = sb._merge_all_results([], keyword, [], keyword_weight=0.8)

        # Custom weight 0.8 should give higher score than default 0.3
        assert merged_custom[0]["score"] > merged_default[0]["score"]


class TestPgVectorSearchBackendSearch:
    """Test PgVectorSearchBackend search (main entry point)"""

    def _make_search_backend(self):
        with patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True), \
             patch('signalwire_agents.search.pgvector_backend.psycopg2') as mock_pg, \
             patch('signalwire_agents.search.pgvector_backend.register_vector'):
            mock_conn, mock_cursor = _make_mock_conn()
            mock_pg.connect.return_value = mock_conn
            mock_cursor.fetchone.return_value = None
            sb = PgVectorSearchBackend("postgresql://localhost/testdb", "col")
            return sb

    def test_search_calls_all_sub_searches(self):
        """Test search invokes vector, keyword, and metadata searches"""
        sb = self._make_search_backend()

        sb._vector_search = Mock(return_value=[])
        sb._keyword_search = Mock(return_value=[])
        sb._metadata_search = Mock(return_value=[])
        sb._merge_all_results = Mock(return_value=[])

        sb.search([0.1, 0.2], "test query", count=5)

        sb._vector_search.assert_called_once()
        sb._keyword_search.assert_called_once()
        sb._metadata_search.assert_called_once()
        sb._merge_all_results.assert_called_once()

    def test_search_returns_limited_results(self):
        """Test search returns at most count results"""
        sb = self._make_search_backend()

        many_results = [
            {"id": i, "content": f"Result {i}", "score": 1.0 - i * 0.1, "final_score": 1.0 - i * 0.1, "metadata": {}}
            for i in range(10)
        ]

        sb._vector_search = Mock(return_value=[])
        sb._keyword_search = Mock(return_value=[])
        sb._metadata_search = Mock(return_value=[])
        sb._merge_all_results = Mock(return_value=many_results)

        results = sb.search([0.1], "query", count=3)

        assert len(results) == 3

    def test_search_applies_similarity_threshold_before_merge(self):
        """Test search filters vector results by similarity threshold before merging"""
        sb = self._make_search_backend()

        vector_results = [
            {"id": 1, "content": "High", "score": 0.9, "search_type": "vector", "metadata": {}},
            {"id": 2, "content": "Low", "score": 0.3, "search_type": "vector", "metadata": {}},
        ]

        sb._vector_search = Mock(return_value=vector_results)
        sb._keyword_search = Mock(return_value=[])
        sb._metadata_search = Mock(return_value=[])
        sb._merge_all_results = Mock(return_value=[])

        sb.search([0.1], "query", count=5, similarity_threshold=0.5)

        # _merge_all_results should receive only the high-score vector result
        merge_call_args = sb._merge_all_results.call_args
        filtered_vector = merge_call_args[0][0]
        assert len(filtered_vector) == 1
        assert filtered_vector[0]["id"] == 1

    def test_search_no_threshold_keeps_all_vector_results(self):
        """Test search with threshold=0.0 keeps all vector results"""
        sb = self._make_search_backend()

        vector_results = [
            {"id": 1, "content": "High", "score": 0.9, "search_type": "vector", "metadata": {}},
            {"id": 2, "content": "Low", "score": 0.1, "search_type": "vector", "metadata": {}},
        ]

        sb._vector_search = Mock(return_value=vector_results)
        sb._keyword_search = Mock(return_value=[])
        sb._metadata_search = Mock(return_value=[])
        sb._merge_all_results = Mock(return_value=[])

        sb.search([0.1], "query", count=5, similarity_threshold=0.0)

        merge_call_args = sb._merge_all_results.call_args
        filtered_vector = merge_call_args[0][0]
        assert len(filtered_vector) == 2

    def test_search_with_tags(self):
        """Test search passes tags to sub-searches"""
        sb = self._make_search_backend()

        sb._vector_search = Mock(return_value=[])
        sb._keyword_search = Mock(return_value=[])
        sb._metadata_search = Mock(return_value=[])
        sb._merge_all_results = Mock(return_value=[])

        sb.search([0.1], "query", count=5, tags=["python"])

        # All sub-searches should receive tags
        _, kwargs = sb._vector_search.call_args
        assert kwargs.get("tags") == ["python"] or sb._vector_search.call_args[0][2] == ["python"]

    def test_search_with_keyword_weight(self):
        """Test search passes keyword_weight to merge"""
        sb = self._make_search_backend()

        sb._vector_search = Mock(return_value=[])
        sb._keyword_search = Mock(return_value=[])
        sb._metadata_search = Mock(return_value=[])
        sb._merge_all_results = Mock(return_value=[])

        sb.search([0.1], "query", count=5, keyword_weight=0.7)

        merge_call = sb._merge_all_results.call_args
        assert merge_call[0][3] == 0.7 or merge_call[1].get("keyword_weight") == 0.7

    def test_search_ensures_score_field(self):
        """Test search adds score field if missing from merged results"""
        sb = self._make_search_backend()

        merged = [
            {"id": 1, "content": "Test", "final_score": 0.8, "metadata": {}},
        ]

        sb._vector_search = Mock(return_value=[])
        sb._keyword_search = Mock(return_value=[])
        sb._metadata_search = Mock(return_value=[])
        sb._merge_all_results = Mock(return_value=merged)

        results = sb.search([0.1], "query", count=5)

        assert results[0]["score"] == 0.8

    def test_search_keeps_existing_score_field(self):
        """Test search does not overwrite existing score field"""
        sb = self._make_search_backend()

        merged = [
            {"id": 1, "content": "Test", "score": 0.9, "final_score": 0.8, "metadata": {}},
        ]

        sb._vector_search = Mock(return_value=[])
        sb._keyword_search = Mock(return_value=[])
        sb._metadata_search = Mock(return_value=[])
        sb._merge_all_results = Mock(return_value=merged)

        results = sb.search([0.1], "query", count=5)

        assert results[0]["score"] == 0.9

    def test_search_extracts_query_terms(self):
        """Test search splits enhanced_text into query terms for metadata search"""
        sb = self._make_search_backend()

        sb._vector_search = Mock(return_value=[])
        sb._keyword_search = Mock(return_value=[])
        sb._metadata_search = Mock(return_value=[])
        sb._merge_all_results = Mock(return_value=[])

        sb.search([0.1], "hello world test", count=5)

        # _metadata_search should receive lowercased split terms
        meta_call = sb._metadata_search.call_args
        terms = meta_call[0][0]
        assert terms == ["hello", "world", "test"]

    def test_search_vector_count_multiplied(self):
        """Test search requests count*2 results from sub-searches"""
        sb = self._make_search_backend()

        sb._vector_search = Mock(return_value=[])
        sb._keyword_search = Mock(return_value=[])
        sb._metadata_search = Mock(return_value=[])
        sb._merge_all_results = Mock(return_value=[])

        sb.search([0.1], "query", count=5)

        vector_count_arg = sb._vector_search.call_args[0][1]
        assert vector_count_arg == 10  # 5 * 2


class TestPgVectorSearchBackendGetStats:
    """Test PgVectorSearchBackend get_stats"""

    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    def test_get_stats_creates_pgvector_backend(self, mock_reg, mock_pg):
        """Test get_stats creates a PgVectorBackend internally"""
        mock_conn, mock_cursor = _make_mock_conn()
        mock_pg.connect.return_value = mock_conn
        mock_cursor.fetchone.return_value = None

        sb = PgVectorSearchBackend("postgresql://localhost/testdb", "col")

        with patch('signalwire_agents.search.pgvector_backend.PgVectorBackend') as MockBackend:
            mock_inner = MagicMock()
            mock_inner.get_stats.return_value = {"total_chunks": 100}
            MockBackend.return_value = mock_inner

            stats = sb.get_stats()

            MockBackend.assert_called_once_with("postgresql://localhost/testdb")
            mock_inner.get_stats.assert_called_once_with("col")
            mock_inner.close.assert_called_once()
            assert stats == {"total_chunks": 100}


class TestPgVectorSearchBackendClose:
    """Test PgVectorSearchBackend close"""

    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    def test_close_closes_connection(self, mock_reg, mock_pg):
        """Test close closes the database connection"""
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_pg.connect.return_value = mock_conn

        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_cursor.fetchone.return_value = None

        sb = PgVectorSearchBackend("postgresql://localhost/testdb", "col")
        sb.close()

        mock_conn.close.assert_called_once()

    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    def test_close_already_closed(self, mock_reg, mock_pg):
        """Test close does nothing when already closed"""
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_pg.connect.return_value = mock_conn

        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_cursor.fetchone.return_value = None

        sb = PgVectorSearchBackend("postgresql://localhost/testdb", "col")
        mock_conn.closed = True
        mock_conn.close.reset_mock()
        sb.close()

        mock_conn.close.assert_not_called()

    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    def test_close_when_conn_is_none(self, mock_reg, mock_pg):
        """Test close does nothing when conn is None"""
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_pg.connect.return_value = mock_conn

        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_cursor.fetchone.return_value = None

        sb = PgVectorSearchBackend("postgresql://localhost/testdb", "col")
        sb.conn = None
        sb.close()  # Should not raise


class TestPgVectorSearchBackendEnsureConnection:
    """Test PgVectorSearchBackend _ensure_connection"""

    @patch('signalwire_agents.search.pgvector_backend.PGVECTOR_AVAILABLE', True)
    @patch('signalwire_agents.search.pgvector_backend.psycopg2')
    @patch('signalwire_agents.search.pgvector_backend.register_vector')
    def test_ensure_connection_reconnects_when_closed(self, mock_reg, mock_pg):
        """Test _ensure_connection reconnects when connection is closed"""
        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_pg.connect.return_value = mock_conn

        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_cursor.fetchone.return_value = None

        sb = PgVectorSearchBackend("postgresql://localhost/testdb", "col")
        assert mock_pg.connect.call_count == 1

        mock_conn.closed = True
        sb._ensure_connection()
        assert mock_pg.connect.call_count == 2


class TestPgvectorAvailableFlag:
    """Test the PGVECTOR_AVAILABLE module-level flag behavior"""

    def test_pgvector_available_is_true(self):
        """Verify PGVECTOR_AVAILABLE is True in our test environment (mocked imports)"""
        assert PGVECTOR_AVAILABLE is True
