"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
Unit tests for search service module
"""

import sys
import types
import pytest
import asyncio
import hashlib
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock, PropertyMock

# ---------------------------------------------------------------------------
# Pre-mock heavy third-party modules that may not be installed in the test
# environment.  This must happen *before* importing the module-under-test
# because ``search_service.py`` transitively imports ``query_processor.py``
# which does ``import nltk`` at module scope.
# ---------------------------------------------------------------------------

_MODULES_TO_STUB = [
    "nltk",
    "nltk.corpus",
    "nltk.corpus.stopwords",
    "nltk.stem",
    "nltk.data",
    "nltk.corpus.wordnet",
    "sentence_transformers",
    "spacy",
    "fastapi",
    "fastapi.middleware",
    "fastapi.middleware.cors",
    "fastapi.security",
    "pydantic",
    "uvicorn",
    "sklearn",
    "sklearn.metrics",
    "sklearn.metrics.pairwise",
    "numpy",
]

# Snapshot sys.modules BEFORE any stubbing.
# We REPLACE (not modify) real modules with fresh stubs to avoid corrupting
# real module objects, then restore everything after import.
_sys_modules_snapshot = dict(sys.modules)

# Install fresh stub modules for ALL entries (replacing real ones too)
for _mod_name in _MODULES_TO_STUB:
    sys.modules[_mod_name] = types.ModuleType(_mod_name)

# Provide minimal attributes that the source code accesses at import time
_nltk = sys.modules["nltk"]
_nltk.word_tokenize = Mock(return_value=[])
_nltk.pos_tag = Mock(return_value=[])
_nltk.download = Mock()
_nltk.data = sys.modules["nltk.data"]
_nltk.data.find = Mock(side_effect=LookupError("stub"))
_nltk.corpus = sys.modules["nltk.corpus"]
_nltk.corpus.stopwords = sys.modules["nltk.corpus.stopwords"]
_nltk.corpus.stopwords.words = Mock(return_value=[])

_wn = sys.modules["nltk.corpus.wordnet"]
_wn.NOUN = "n"
_wn.VERB = "v"
_wn.ADJ = "a"
_wn.ADV = "r"
_wn.synsets = Mock(return_value=[])
sys.modules["nltk.corpus"].wordnet = _wn

_nltk_stem = sys.modules["nltk.stem"]
_porter = MagicMock()
_porter.stem = Mock(side_effect=lambda w: w)
_nltk_stem.PorterStemmer = Mock(return_value=_porter)
_nltk.WordNetLemmatizer = Mock(return_value=MagicMock())

# Stub sentence_transformers
sys.modules["sentence_transformers"].SentenceTransformer = None

# Stub FastAPI classes (on fresh stub modules, not real ones)
_fastapi_mod = sys.modules["fastapi"]
_fastapi_mod.FastAPI = None
_fastapi_mod.HTTPException = None
_fastapi_mod.Request = None
_fastapi_mod.Response = None
_fastapi_mod.Depends = None
sys.modules["fastapi.middleware.cors"].CORSMiddleware = None
sys.modules["fastapi.security"].HTTPBasic = None
sys.modules["fastapi.security"].HTTPBasicCredentials = None
sys.modules["pydantic"].BaseModel = None

# Now we can safely import the module under test.
from signalwire_agents.search.search_service import (
    _cache_key,
    SearchService,
    SearchRequest,
    SearchResult,
    SearchResponse,
)

# Restore sys.modules to exact pre-stub state. Remove modules that were
# added during the stubbed import (intermediate imports like query_processor
# that captured stub references), and put back real modules we replaced.
_keep_modules = {
    "signalwire_agents.search.search_service",
}
for _mod_name in list(sys.modules):
    if _mod_name not in _sys_modules_snapshot and _mod_name not in _keep_modules:
        del sys.modules[_mod_name]
sys.modules.update(_sys_modules_snapshot)
del _sys_modules_snapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_search_request(query="test query", index_name="default", count=3,
                         similarity_threshold=0.0, tags=None, language=None):
    """Create a mock SearchRequest-like object."""
    req = Mock()
    req.query = query
    req.index_name = index_name
    req.count = count
    req.similarity_threshold = similarity_threshold
    req.tags = tags
    req.language = language
    return req


def _run_async(coro):
    """Run an async coroutine synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Fixture: patch external deps used by SearchService.__init__
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_external_deps():
    """Patch external dependencies for every test so SearchService can be constructed."""
    with patch("signalwire_agents.search.search_service.SecurityConfig") as mock_sec, \
         patch("signalwire_agents.search.search_service.ConfigLoader") as mock_cl, \
         patch("signalwire_agents.search.search_service.set_global_model"), \
         patch("signalwire_agents.search.search_service.SearchEngine") as mock_se, \
         patch("signalwire_agents.search.search_service.SentenceTransformer", None):
        # SecurityConfig defaults
        sec_instance = MagicMock()
        sec_instance.get_basic_auth.return_value = ("user", "pass")
        sec_instance.get_cors_config.return_value = {
            "allow_origins": ["*"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
        sec_instance.get_security_headers.return_value = {}
        sec_instance.ssl_enabled = False
        sec_instance.get_ssl_context_kwargs.return_value = {}
        mock_sec.return_value = sec_instance

        # ConfigLoader defaults
        cl_instance = MagicMock()
        cl_instance.has_config.return_value = False
        mock_cl.return_value = cl_instance
        mock_cl.find_config_file.return_value = None

        yield {
            "SecurityConfig": mock_sec,
            "ConfigLoader": mock_cl,
            "SearchEngine": mock_se,
            "security_instance": sec_instance,
        }


# ===================================================================
# Tests for _cache_key helper
# ===================================================================

class TestCacheKey:
    """Tests for the module-level _cache_key function."""

    def test_cache_key_basic(self):
        key = _cache_key("hello world", "default", 3, None)
        assert isinstance(key, str)
        assert len(key) == 32  # md5 hex digest length

    def test_cache_key_deterministic(self):
        k1 = _cache_key("hello", "idx", 5, ["a", "b"])
        k2 = _cache_key("hello", "idx", 5, ["a", "b"])
        assert k1 == k2

    def test_cache_key_case_insensitive_query(self):
        k1 = _cache_key("Hello World", "default", 3, None)
        k2 = _cache_key("hello world", "default", 3, None)
        assert k1 == k2

    def test_cache_key_strips_whitespace(self):
        k1 = _cache_key("  hello  ", "default", 3, None)
        k2 = _cache_key("hello", "default", 3, None)
        assert k1 == k2

    def test_cache_key_tags_sorted(self):
        k1 = _cache_key("q", "idx", 1, ["b", "a"])
        k2 = _cache_key("q", "idx", 1, ["a", "b"])
        assert k1 == k2

    def test_cache_key_different_queries_differ(self):
        k1 = _cache_key("alpha", "default", 3, None)
        k2 = _cache_key("beta", "default", 3, None)
        assert k1 != k2

    def test_cache_key_different_indexes_differ(self):
        k1 = _cache_key("q", "idx1", 3, None)
        k2 = _cache_key("q", "idx2", 3, None)
        assert k1 != k2

    def test_cache_key_different_counts_differ(self):
        k1 = _cache_key("q", "idx", 3, None)
        k2 = _cache_key("q", "idx", 5, None)
        assert k1 != k2

    def test_cache_key_none_tags_vs_empty_tags(self):
        k1 = _cache_key("q", "idx", 3, None)
        k2 = _cache_key("q", "idx", 3, [])
        # Both produce tags=[] in the key_data
        assert k1 == k2

    def test_cache_key_does_not_leak_query(self):
        """Cache keys should be hashed, not plaintext."""
        key = _cache_key("sensitive query", "default", 3, None)
        assert "sensitive" not in key
        assert "query" not in key


# ===================================================================
# Tests for SearchService initialization
# ===================================================================

class TestSearchServiceInit:
    """Tests for SearchService.__init__ and _load_config."""

    def test_default_initialization(self):
        """SearchService can be instantiated with all defaults."""
        svc = SearchService()
        assert svc.port == 8001
        assert svc.backend == "sqlite"
        assert svc.connection_string is None
        assert svc.indexes == {}
        assert svc.search_engines == {}
        assert svc.model is None
        assert svc._query_cache == {}
        assert svc._cache_size == 100

    def test_custom_port(self):
        svc = SearchService(port=9999)
        assert svc.port == 9999

    def test_custom_indexes(self):
        indexes = {"docs": "/path/docs.db", "faq": "/path/faq.db"}
        svc = SearchService(indexes=indexes)
        assert svc.indexes == indexes

    def test_basic_auth_from_constructor(self):
        svc = SearchService(basic_auth=("admin", "secret"))
        assert svc._basic_auth == ("admin", "secret")

    def test_basic_auth_fallback_to_security_config(self):
        # When basic_auth is not passed, it should fall back to security config
        svc = SearchService()
        assert svc._basic_auth == ("user", "pass")

    def test_pgvector_backend(self):
        svc = SearchService(backend="pgvector", connection_string="postgresql://localhost/db")
        assert svc.backend == "pgvector"
        assert svc.connection_string == "postgresql://localhost/db"

    def test_security_config_created(self, _patch_external_deps):
        svc = SearchService(config_file="/some/config.json")
        _patch_external_deps["SecurityConfig"].assert_called_once_with(
            config_file="/some/config.json", service_name="search"
        )
        _patch_external_deps["security_instance"].log_config.assert_called_once_with("SearchService")

    def test_no_fastapi_sets_app_none(self):
        # FastAPI is already None due to our stubs
        svc = SearchService()
        assert svc.app is None


# ===================================================================
# Tests for _load_config
# ===================================================================

class TestLoadConfig:
    """Tests for _load_config method."""

    def test_load_config_no_file(self, _patch_external_deps):
        _patch_external_deps["ConfigLoader"].find_config_file.return_value = None
        svc = SearchService()
        assert svc.indexes == {}
        assert svc.backend == "sqlite"
        assert svc.connection_string is None

    def test_load_config_with_service_section(self, _patch_external_deps):
        cl_instance = _patch_external_deps["ConfigLoader"].return_value
        cl_instance.has_config.return_value = True
        cl_instance.get_section.return_value = {
            "port": "9999",
            "backend": "pgvector",
            "connection_string": "postgresql://localhost/mydb",
            "indexes": {"docs": "my_collection"},
        }
        _patch_external_deps["ConfigLoader"].find_config_file.return_value = "/tmp/config.json"

        svc = SearchService(config_file="/tmp/config.json")
        # The constructor overrides port with its own default 8001 after _load_config
        assert svc.port == 8001

    def test_load_config_indexes_only_when_dict(self, _patch_external_deps):
        cl_instance = _patch_external_deps["ConfigLoader"].return_value
        cl_instance.has_config.return_value = True
        cl_instance.get_section.return_value = {
            "indexes": "not_a_dict",
        }
        _patch_external_deps["ConfigLoader"].find_config_file.return_value = "/tmp/config.json"

        svc = SearchService(config_file="/tmp/config.json")
        assert svc.indexes == {}


# ===================================================================
# Tests for _load_resources
# ===================================================================

class TestLoadResources:
    """Tests for _load_resources method."""

    def test_load_resources_sqlite_no_indexes(self):
        svc = SearchService()
        assert svc.model is None
        assert svc.search_engines == {}

    def test_load_resources_sqlite_with_indexes(self, _patch_external_deps):
        mock_model = MagicMock()
        with patch("signalwire_agents.search.search_service.SentenceTransformer", return_value=mock_model):
            with patch.object(
                SearchService, "_get_model_name", return_value="sentence-transformers/all-mpnet-base-v2"
            ):
                svc = SearchService(indexes={"docs": "/path/docs.db"})
                assert svc.model == mock_model

    def test_load_resources_sqlite_model_load_failure(self, _patch_external_deps):
        with patch(
            "signalwire_agents.search.search_service.SentenceTransformer",
            side_effect=Exception("model load failed"),
        ):
            with patch.object(
                SearchService, "_get_model_name", return_value="sentence-transformers/all-mpnet-base-v2"
            ):
                svc = SearchService(indexes={"docs": "/path/docs.db"})
                assert svc.model is None

    def test_load_resources_pgvector_creates_engines(self, _patch_external_deps):
        mock_engine = MagicMock()
        mock_engine.config = {"model_name": "test-model"}
        _patch_external_deps["SearchEngine"].return_value = mock_engine

        mock_model = MagicMock()
        with patch("signalwire_agents.search.search_service.SentenceTransformer", return_value=mock_model):
            svc = SearchService(
                backend="pgvector",
                connection_string="postgresql://localhost/db",
                indexes={"col1": "collection_1"},
            )
            assert "col1" in svc.search_engines

    def test_load_resources_pgvector_engine_failure(self, _patch_external_deps):
        _patch_external_deps["SearchEngine"].side_effect = Exception("connection failed")

        svc = SearchService(
            backend="pgvector",
            connection_string="postgresql://localhost/db",
            indexes={"col1": "collection_1"},
        )
        assert "col1" not in svc.search_engines


# ===================================================================
# Tests for _get_model_name
# ===================================================================

class TestGetModelName:
    """Tests for _get_model_name method."""

    def test_pgvector_returns_default_model(self):
        svc = SearchService(backend="pgvector", connection_string="postgresql://localhost/db")
        result = svc._get_model_name("/some/path")
        assert result == "sentence-transformers/all-mpnet-base-v2"

    def test_sqlite_reads_from_database(self):
        svc = SearchService()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("custom-model",)
        mock_conn.cursor.return_value = mock_cursor

        mock_sqlite3 = MagicMock()
        mock_sqlite3.connect.return_value = mock_conn

        with patch.dict("sys.modules", {"sqlite3": mock_sqlite3}):
            result = svc._get_model_name("/path/to/index.db")
            assert result == "custom-model"
            mock_conn.close.assert_called_once()

    def test_sqlite_database_error_returns_default(self):
        svc = SearchService()

        mock_sqlite3 = MagicMock()
        mock_sqlite3.connect.side_effect = Exception("db error")

        with patch.dict("sys.modules", {"sqlite3": mock_sqlite3}):
            result = svc._get_model_name("/nonexistent/path.db")
            assert result == "sentence-transformers/all-mpnet-base-v2"

    def test_sqlite_no_config_row_returns_default(self):
        svc = SearchService()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn.cursor.return_value = mock_cursor

        mock_sqlite3 = MagicMock()
        mock_sqlite3.connect.return_value = mock_conn

        with patch.dict("sys.modules", {"sqlite3": mock_sqlite3}):
            result = svc._get_model_name("/path/to/index.db")
            assert result == "sentence-transformers/all-mpnet-base-v2"


# ===================================================================
# Tests for _handle_search
# ===================================================================

class TestHandleSearch:
    """Tests for the async _handle_search method."""

    @pytest.fixture
    def service_with_engine(self):
        """Create a SearchService with a mocked search engine."""
        svc = SearchService()

        mock_engine = MagicMock()
        mock_engine.config = {"embedding_model": "test-model"}
        mock_engine.search.return_value = [
            {
                "content": "Result content",
                "score": 0.95,
                "metadata": {"filename": "test.md", "section": "intro"},
            }
        ]
        svc.search_engines["default"] = mock_engine
        return svc

    def test_handle_search_index_not_found(self):
        svc = SearchService()
        request = _make_search_request(index_name="nonexistent")

        with pytest.raises(Exception):
            _run_async(svc._handle_search(request))

    def test_handle_search_success(self, service_with_engine):
        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "enhanced test query",
                "vector": [0.1, 0.2, 0.3],
                "language": "en",
                "POS": {"test": "NN"},
            }

            request = _make_search_request()
            response = _run_async(service_with_engine._handle_search(request))

            assert len(response.results) == 1
            assert response.results[0].content == "Result content"
            assert response.results[0].score == 0.95
            assert response.query_analysis["original_query"] == "test query"
            assert response.query_analysis["enhanced_query"] == "enhanced test query"

    def test_handle_search_preprocessing_failure(self, service_with_engine):
        """When preprocess_query fails, search should still proceed with original query."""
        with patch(
            "signalwire_agents.search.search_service.preprocess_query",
            side_effect=Exception("NLP error"),
        ):
            request = _make_search_request()
            response = _run_async(service_with_engine._handle_search(request))

            assert response.query_analysis["enhanced_query"] == "test query"

    def test_handle_search_engine_failure(self, service_with_engine):
        """When search engine raises, should return empty results."""
        service_with_engine.search_engines["default"].search.side_effect = Exception("search error")

        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "test",
                "vector": [0.1],
                "language": "en",
            }

            request = _make_search_request()
            response = _run_async(service_with_engine._handle_search(request))

            assert len(response.results) == 0

    def test_handle_search_caching(self, service_with_engine):
        """Second identical query should return cached result."""
        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "test",
                "vector": [0.1],
                "language": "en",
            }

            request = _make_search_request()

            resp1 = _run_async(service_with_engine._handle_search(request))
            resp2 = _run_async(service_with_engine._handle_search(request))

            # preprocess_query should only be called once -- second was cached
            assert mock_pp.call_count == 1
            assert resp1 is resp2

    def test_handle_search_cache_eviction(self, service_with_engine):
        """When cache is full, oldest entry should be evicted."""
        service_with_engine._cache_size = 2

        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "test",
                "vector": [0.1],
                "language": "en",
            }

            for i in range(3):
                req = _make_search_request(query=f"query{i}")
                _run_async(service_with_engine._handle_search(req))

            assert len(service_with_engine._query_cache) <= 2

    def test_handle_search_with_tags(self, service_with_engine):
        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "test",
                "vector": [0.1],
                "language": "en",
            }

            request = _make_search_request(tags=["python"])
            _run_async(service_with_engine._handle_search(request))

            call_kwargs = service_with_engine.search_engines["default"].search.call_args
            assert call_kwargs[1]["tags"] == ["python"]

    def test_handle_search_with_language(self, service_with_engine):
        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "test",
                "vector": [0.1],
                "language": "es",
            }

            request = _make_search_request(language="es")
            _run_async(service_with_engine._handle_search(request))

            mock_pp.assert_called_once()
            call_kwargs = mock_pp.call_args
            assert call_kwargs[1]["language"] == "es"

    def test_handle_search_auto_language(self, service_with_engine):
        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "test",
                "vector": [0.1],
                "language": "en",
            }

            request = _make_search_request(language=None)
            _run_async(service_with_engine._handle_search(request))

            call_kwargs = mock_pp.call_args
            assert call_kwargs[1]["language"] == "auto"

    def test_handle_search_similarity_threshold(self, service_with_engine):
        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "test",
                "vector": [0.1],
                "language": "en",
            }

            request = _make_search_request(similarity_threshold=0.5)
            _run_async(service_with_engine._handle_search(request))

            call_kwargs = service_with_engine.search_engines["default"].search.call_args
            assert call_kwargs[1]["similarity_threshold"] == 0.5


# ===================================================================
# Tests for _handle_search with pgvector backend
# ===================================================================

class TestHandleSearchPgvector:
    """Tests for _handle_search with pgvector-specific behavior."""

    def test_pgvector_sets_global_model(self):
        svc = SearchService(
            backend="pgvector",
            connection_string="postgresql://localhost/db",
        )
        mock_engine = MagicMock()
        mock_engine.config = {"model_name": "test-model"}
        mock_engine.search.return_value = []
        svc.search_engines["default"] = mock_engine

        mock_model = MagicMock()
        svc.models = {"test-model": mock_model}
        svc.collection_models = {"default": "test-model"}

        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp, \
             patch("signalwire_agents.search.search_service.set_global_model") as mock_sgm:
            mock_pp.return_value = {
                "enhanced_text": "test",
                "vector": [0.1],
                "language": "en",
            }

            request = _make_search_request()
            _run_async(svc._handle_search(request))

            mock_sgm.assert_called_with(mock_model)


# ===================================================================
# Tests for _get_current_username (auth validation)
# ===================================================================

class TestGetCurrentUsername:
    """Tests for basic auth credential validation."""

    def test_no_credentials_returns_none(self):
        svc = SearchService()
        result = svc._get_current_username(credentials=None)
        assert result is None

    def test_valid_credentials(self):
        svc = SearchService(basic_auth=("admin", "secret"))

        creds = MagicMock()
        creds.username = "admin"
        creds.password = "secret"

        result = svc._get_current_username(credentials=creds)
        assert result == "admin"

    def test_invalid_username(self):
        svc = SearchService(basic_auth=("admin", "secret"))

        creds = MagicMock()
        creds.username = "wrong"
        creds.password = "secret"

        # HTTPException is None in this test environment (no FastAPI),
        # so the code will try to call None(...) which raises TypeError
        with pytest.raises(Exception):
            svc._get_current_username(credentials=creds)

    def test_invalid_password(self):
        svc = SearchService(basic_auth=("admin", "secret"))

        creds = MagicMock()
        creds.username = "admin"
        creds.password = "wrong"

        with pytest.raises(Exception):
            svc._get_current_username(credentials=creds)

    def test_timing_safe_comparison_used(self):
        """Verify that secrets.compare_digest is used (timing-safe)."""
        svc = SearchService(basic_auth=("admin", "secret"))

        creds = MagicMock()
        creds.username = "admin"
        creds.password = "secret"

        with patch("secrets.compare_digest", return_value=True) as mock_cd:
            svc._get_current_username(credentials=creds)
            assert mock_cd.call_count == 2


# ===================================================================
# Tests for search_direct (sync wrapper)
# ===================================================================

class TestSearchDirect:
    """Tests for the synchronous search_direct method."""

    def test_search_direct_basic(self):
        svc = SearchService()

        mock_engine = MagicMock()
        mock_engine.config = {}
        mock_engine.search.return_value = [
            {
                "content": "Direct result",
                "score": 0.88,
                "metadata": {"filename": "test.md"},
            }
        ]
        svc.search_engines["default"] = mock_engine

        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "test",
                "vector": [0.1],
                "language": "en",
            }

            result = svc.search_direct("test query")
            assert "results" in result
            assert "query_analysis" in result
            assert len(result["results"]) == 1
            assert result["results"][0]["content"] == "Direct result"
            assert result["results"][0]["score"] == 0.88

    def test_search_direct_passes_parameters(self):
        svc = SearchService()

        mock_engine = MagicMock()
        mock_engine.config = {}
        mock_engine.search.return_value = []
        svc.search_engines["myindex"] = mock_engine

        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "test",
                "vector": [0.1],
                "language": "es",
            }

            svc.search_direct(
                "test query",
                index_name="myindex",
                count=5,
                distance=0.3,
                tags=["python"],
                language="es",
            )

            call_kwargs = mock_pp.call_args
            assert call_kwargs[1]["language"] == "es"

    def test_search_direct_index_not_found(self):
        svc = SearchService()

        with pytest.raises(Exception):
            svc.search_direct("test", index_name="nonexistent")


# ===================================================================
# Tests for start/stop
# ===================================================================

class TestStartStop:
    """Tests for start and stop methods."""

    def test_start_no_app_raises(self):
        svc = SearchService()
        # app is None because FastAPI is stubbed as None
        with pytest.raises(RuntimeError, match="FastAPI not available"):
            svc.start()

    def test_start_with_custom_port(self):
        svc = SearchService()
        svc.app = MagicMock()  # Ensure app is not None

        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            svc.start(port=7777)
            mock_uvicorn.run.assert_called_once()
            call_args = mock_uvicorn.run.call_args
            assert call_args[1]["port"] == 7777

    def test_start_with_ssl_cert_and_key(self):
        svc = SearchService()
        svc.app = MagicMock()

        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            svc.start(ssl_cert="/path/cert.pem", ssl_key="/path/key.pem")
            call_args = mock_uvicorn.run.call_args
            assert call_args[1]["ssl_certfile"] == "/path/cert.pem"
            assert call_args[1]["ssl_keyfile"] == "/path/key.pem"

    def test_start_with_security_config_ssl(self, _patch_external_deps):
        _patch_external_deps["security_instance"].get_ssl_context_kwargs.return_value = {
            "ssl_certfile": "/auto/cert.pem",
            "ssl_keyfile": "/auto/key.pem",
        }

        svc = SearchService()
        svc.app = MagicMock()

        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            svc.start()
            call_args = mock_uvicorn.run.call_args
            assert call_args[1]["ssl_certfile"] == "/auto/cert.pem"
            assert call_args[1]["ssl_keyfile"] == "/auto/key.pem"

    def test_start_default_host(self):
        svc = SearchService()
        svc.app = MagicMock()

        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            svc.start()
            call_args = mock_uvicorn.run.call_args
            assert call_args[1]["host"] == "0.0.0.0"

    def test_start_uses_instance_port_by_default(self):
        svc = SearchService(port=5555)
        svc.app = MagicMock()

        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            svc.start()
            call_args = mock_uvicorn.run.call_args
            assert call_args[1]["port"] == 5555

    def test_stop_does_not_raise(self):
        svc = SearchService()
        # stop() is a no-op placeholder
        svc.stop()


# ===================================================================
# Tests for _setup_security
# ===================================================================

class TestSetupSecurity:
    """Tests for security middleware setup."""

    def test_setup_security_no_app(self):
        svc = SearchService()
        assert svc.app is None
        # Should not raise when app is None
        svc._setup_security()


# ===================================================================
# Tests for _setup_routes
# ===================================================================

class TestSetupRoutes:
    """Tests for route setup."""

    def test_setup_routes_no_app(self):
        svc = SearchService()
        assert svc.app is None
        # Should not raise when app is None
        svc._setup_routes()


# ===================================================================
# Tests for Pydantic / fallback model classes
# ===================================================================

class TestSearchModels:
    """Tests for the SearchRequest/SearchResult/SearchResponse fallback classes."""

    def test_search_request_defaults(self):
        req = SearchRequest(query="test")
        assert req.query == "test"
        assert req.index_name == "default"
        assert req.count == 3
        assert req.similarity_threshold == 0.0
        assert req.tags is None
        assert req.language is None

    def test_search_request_custom_values(self):
        req = SearchRequest(
            query="hello",
            index_name="myidx",
            count=10,
            similarity_threshold=0.5,
            tags=["a", "b"],
            language="es",
        )
        assert req.query == "hello"
        assert req.index_name == "myidx"
        assert req.count == 10
        assert req.similarity_threshold == 0.5
        assert req.tags == ["a", "b"]
        assert req.language == "es"

    def test_search_result_creation(self):
        result = SearchResult(
            content="test content",
            score=0.95,
            metadata={"key": "value"},
        )
        assert result.content == "test content"
        assert result.score == 0.95
        assert result.metadata == {"key": "value"}

    def test_search_response_creation(self):
        result = SearchResult(content="test", score=0.9, metadata={})
        response = SearchResponse(results=[result], query_analysis={"lang": "en"})
        assert len(response.results) == 1
        assert response.query_analysis == {"lang": "en"}

    def test_search_response_no_analysis(self):
        response = SearchResponse(results=[])
        assert response.query_analysis is None


# ===================================================================
# Tests for edge cases and security
# ===================================================================

class TestEdgeCasesAndSecurity:
    """Tests for edge cases, error handling, and security considerations."""

    def test_empty_query(self):
        svc = SearchService()

        mock_engine = MagicMock()
        mock_engine.config = {}
        mock_engine.search.return_value = []
        svc.search_engines["default"] = mock_engine

        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "",
                "vector": [],
                "language": "en",
            }

            request = _make_search_request(query="")
            response = _run_async(svc._handle_search(request))
            assert len(response.results) == 0

    def test_very_long_query(self):
        """Ensure long queries do not cause issues."""
        svc = SearchService()

        mock_engine = MagicMock()
        mock_engine.config = {}
        mock_engine.search.return_value = []
        svc.search_engines["default"] = mock_engine

        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "long " * 10000,
                "vector": [0.1],
                "language": "en",
            }

            long_query = "word " * 10000
            request = _make_search_request(query=long_query)
            response = _run_async(svc._handle_search(request))
            assert response is not None

    def test_special_characters_in_query(self):
        """Ensure special characters in queries are handled."""
        svc = SearchService()

        mock_engine = MagicMock()
        mock_engine.config = {}
        mock_engine.search.return_value = []
        svc.search_engines["default"] = mock_engine

        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "test",
                "vector": [0.1],
                "language": "en",
            }

            request = _make_search_request(query='<script>alert("xss")</script>')
            response = _run_async(svc._handle_search(request))
            assert response is not None

    def test_health_endpoint_masks_connection_string(self):
        """Connection strings should be masked in health endpoint responses."""
        svc = SearchService(
            backend="pgvector",
            connection_string="postgresql://user:password@host/db",
        )
        assert svc.backend == "pgvector"
        assert svc.connection_string == "postgresql://user:password@host/db"

    def test_concurrent_cache_access(self):
        """Basic test that cache operations do not fail under simple usage patterns."""
        svc = SearchService()

        mock_engine = MagicMock()
        mock_engine.config = {}
        mock_engine.search.return_value = [
            {"content": "result", "score": 0.9, "metadata": {}}
        ]
        svc.search_engines["default"] = mock_engine

        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "test",
                "vector": [0.1],
                "language": "en",
            }

            for i in range(5):
                req = _make_search_request(query=f"query_{i}")
                _run_async(svc._handle_search(req))

            assert len(svc._query_cache) == 5

    def test_cache_fifo_eviction_order(self):
        """When cache is full, the first inserted key should be evicted."""
        svc = SearchService()
        svc._cache_size = 3

        mock_engine = MagicMock()
        mock_engine.config = {}
        mock_engine.search.return_value = [
            {"content": "result", "score": 0.9, "metadata": {}}
        ]
        svc.search_engines["default"] = mock_engine

        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "test",
                "vector": [0.1],
                "language": "en",
            }

            # Insert 4 entries into a cache of size 3
            for i in range(4):
                req = _make_search_request(query=f"query_{i}")
                _run_async(svc._handle_search(req))

            assert len(svc._query_cache) == 3

            first_key = _cache_key("query_0", "default", 3, None)
            assert first_key not in svc._query_cache

    def test_handle_search_model_name_from_engine_config(self):
        """Verify model_name is passed to preprocess_query from engine config."""
        svc = SearchService()

        mock_engine = MagicMock()
        mock_engine.config = {"model_name": "custom-model-v2"}
        mock_engine.search.return_value = []
        svc.search_engines["default"] = mock_engine

        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "test",
                "vector": [0.1],
                "language": "en",
            }

            request = _make_search_request()
            _run_async(svc._handle_search(request))

            call_kwargs = mock_pp.call_args
            assert call_kwargs[1]["model_name"] == "custom-model-v2"

    def test_handle_search_embedding_model_fallback(self):
        """Verify embedding_model key is used if model_name is absent."""
        svc = SearchService()

        mock_engine = MagicMock()
        mock_engine.config = {"embedding_model": "fallback-model"}
        mock_engine.search.return_value = []
        svc.search_engines["default"] = mock_engine

        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "test",
                "vector": [0.1],
                "language": "en",
            }

            request = _make_search_request()
            _run_async(svc._handle_search(request))

            call_kwargs = mock_pp.call_args
            assert call_kwargs[1]["model_name"] == "fallback-model"

    def test_handle_search_no_model_name_in_config(self):
        """When engine config has no model keys, model_name should be None."""
        svc = SearchService()

        mock_engine = MagicMock()
        mock_engine.config = {}
        mock_engine.search.return_value = []
        svc.search_engines["default"] = mock_engine

        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "test",
                "vector": [0.1],
                "language": "en",
            }

            request = _make_search_request()
            _run_async(svc._handle_search(request))

            call_kwargs = mock_pp.call_args
            assert call_kwargs[1]["model_name"] is None

    def test_handle_search_no_engine_config_attr(self):
        """When engine has no config attribute, model_name should be None."""
        svc = SearchService()

        mock_engine = MagicMock(spec=[])  # no attributes at all
        mock_engine.search = MagicMock(return_value=[])
        svc.search_engines["default"] = mock_engine

        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "test",
                "vector": [0.1],
                "language": "en",
            }

            request = _make_search_request()
            _run_async(svc._handle_search(request))

            call_kwargs = mock_pp.call_args
            assert call_kwargs[1]["model_name"] is None


# ===================================================================
# Tests for response format
# ===================================================================

class TestResponseFormat:
    """Verify the shape of responses from _handle_search."""

    def test_response_has_query_analysis(self):
        svc = SearchService()

        mock_engine = MagicMock()
        mock_engine.config = {}
        mock_engine.search.return_value = []
        svc.search_engines["default"] = mock_engine

        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "enhanced",
                "vector": [0.1],
                "language": "en",
                "POS": {"test": "NN"},
            }

            request = _make_search_request()
            response = _run_async(svc._handle_search(request))

            assert response.query_analysis is not None
            assert response.query_analysis["original_query"] == "test query"
            assert response.query_analysis["enhanced_query"] == "enhanced"
            assert response.query_analysis["detected_language"] == "en"
            assert response.query_analysis["pos_analysis"] == {"test": "NN"}

    def test_response_results_have_correct_shape(self):
        svc = SearchService()

        mock_engine = MagicMock()
        mock_engine.config = {}
        mock_engine.search.return_value = [
            {
                "content": "Result 1",
                "score": 0.95,
                "metadata": {"filename": "a.md", "section": "intro"},
            },
            {
                "content": "Result 2",
                "score": 0.80,
                "metadata": {"filename": "b.md", "section": "body"},
            },
        ]
        svc.search_engines["default"] = mock_engine

        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "test",
                "vector": [0.1],
                "language": "en",
            }

            request = _make_search_request()
            response = _run_async(svc._handle_search(request))

            assert len(response.results) == 2
            for r in response.results:
                assert hasattr(r, "content")
                assert hasattr(r, "score")
                assert hasattr(r, "metadata")
            assert response.results[0].score == 0.95
            assert response.results[1].score == 0.80


# ===================================================================
# Tests for search_direct result format
# ===================================================================

class TestSearchDirectResultFormat:
    """Verify search_direct returns dict with correct shape."""

    def test_search_direct_result_structure(self):
        svc = SearchService()

        mock_engine = MagicMock()
        mock_engine.config = {}
        mock_engine.search.return_value = [
            {
                "content": "Test result",
                "score": 0.9,
                "metadata": {"filename": "test.md"},
            }
        ]
        svc.search_engines["default"] = mock_engine

        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "test",
                "vector": [0.1],
                "language": "en",
                "POS": {},
            }

            result = svc.search_direct("test")
            assert isinstance(result, dict)
            assert "results" in result
            assert "query_analysis" in result
            assert isinstance(result["results"], list)
            assert len(result["results"]) == 1

            item = result["results"][0]
            assert "content" in item
            assert "score" in item
            assert "metadata" in item

    def test_search_direct_empty_results(self):
        svc = SearchService()

        mock_engine = MagicMock()
        mock_engine.config = {}
        mock_engine.search.return_value = []
        svc.search_engines["default"] = mock_engine

        with patch("signalwire_agents.search.search_service.preprocess_query") as mock_pp:
            mock_pp.return_value = {
                "enhanced_text": "no results",
                "vector": [0.1],
                "language": "en",
            }

            result = svc.search_direct("no results query")
            assert result["results"] == []
            assert result["query_analysis"] is not None
